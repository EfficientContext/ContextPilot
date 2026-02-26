"""
HTTP Intercept Parser for ContextPilot

Pure parsing/extraction/reconstruction logic for intercepting LLM API requests.
Extracts documents from system messages, supports reordering, and reconstructs
the request body with reordered documents.

No server dependencies — independently testable.
"""

import re
import copy
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# XML wrapper tag names we recognize (outer wrapper)
_KNOWN_WRAPPER_TAGS = {"documents", "contexts", "docs", "passages", "references"}

# XML item tag names we recognize (inner items)
_KNOWN_ITEM_TAGS = {"document", "context", "doc", "passage", "reference"}

# Numbered pattern: [1] ... [2] ... etc.
_NUMBERED_RE = re.compile(r"\[(\d+)\]\s*")

# Separator patterns for auto-detection
_SEPARATOR_PATTERNS = ["---", "==="]


@dataclass
class InterceptConfig:
    """Configuration parsed from X-ContextPilot-* headers."""

    enabled: bool = True
    mode: str = "auto"  # xml_tag, separator, numbered, auto
    tag: str = "document"  # XML tag name for xml_tag mode
    separator: str = "---"  # Delimiter for separator mode
    alpha: float = 0.001
    linkage_method: str = "average"


@dataclass
class ExtractionResult:
    """Result of extracting documents from a message."""

    documents: List[str]
    prefix: str = ""  # Text before the documents block
    suffix: str = ""  # Text after the documents block
    mode: str = ""  # Which extraction mode matched
    # XML-specific
    wrapper_tag: str = ""  # e.g. "documents"
    item_tag: str = ""  # e.g. "document"
    # Separator-specific
    separator_char: str = ""
    # Original content for fallback
    original_content: str = ""


def parse_intercept_headers(headers: Dict[str, str]) -> InterceptConfig:
    """Parse X-ContextPilot-* headers into an InterceptConfig."""
    def get(name: str, default: str = "") -> str:
        # Headers are case-insensitive; try common casings
        key = f"x-contextpilot-{name}"
        for k, v in headers.items():
            if k.lower() == key:
                return v
        return default

    enabled_str = get("enabled", "true").lower()
    enabled = enabled_str not in ("false", "0", "no")

    return InterceptConfig(
        enabled=enabled,
        mode=get("mode", "auto").lower(),
        tag=get("tag", "document").lower(),
        separator=get("separator", "---"),
        alpha=float(get("alpha", "0.001")),
        linkage_method=get("linkage", "average"),
    )


# ── Document extraction ─────────────────────────────────────────────────────


def _extract_xml_tags(text: str, config: InterceptConfig) -> Optional[ExtractionResult]:
    """Extract documents from XML-tagged blocks.

    Supports patterns like:
        <documents><document>...</document><document>...</document></documents>
    Also handles custom tags and known alternatives (contexts, docs, passages, references).
    """
    # Determine which wrapper/item tags to try
    if config.mode == "xml_tag":
        # User specified xml_tag mode — try their custom tag first
        item_tags_to_try = [config.tag]
        wrapper_tags_to_try = [config.tag + "s"]  # e.g. "document" -> "documents"
        # Also add known tags as fallback
        item_tags_to_try.extend(t for t in _KNOWN_ITEM_TAGS if t != config.tag)
        wrapper_tags_to_try.extend(t for t in _KNOWN_WRAPPER_TAGS if t != config.tag + "s")
    else:
        item_tags_to_try = list(_KNOWN_ITEM_TAGS)
        wrapper_tags_to_try = list(_KNOWN_WRAPPER_TAGS)

    # Try with wrapper tags first (e.g. <documents>...<document>...</document>...</documents>)
    for wrapper_tag in wrapper_tags_to_try:
        wrapper_pattern = re.compile(
            rf"(<{wrapper_tag}(?:\s[^>]*)?>)(.*?)(</{wrapper_tag}>)",
            re.DOTALL,
        )
        wrapper_match = wrapper_pattern.search(text)
        if not wrapper_match:
            continue

        inner_text = wrapper_match.group(2)
        prefix = text[: wrapper_match.start()]
        suffix = text[wrapper_match.end() :]

        # Try each item tag inside the wrapper
        for item_tag in item_tags_to_try:
            item_pattern = re.compile(
                rf"<{item_tag}(?:\s[^>]*)?>(.+?)</{item_tag}>",
                re.DOTALL,
            )
            items = item_pattern.findall(inner_text)
            if items:
                return ExtractionResult(
                    documents=[item.strip() for item in items],
                    prefix=prefix,
                    suffix=suffix,
                    mode="xml_tag",
                    wrapper_tag=wrapper_tag,
                    item_tag=item_tag,
                    original_content=text,
                )

    # Try without wrapper (just repeated item tags)
    for item_tag in item_tags_to_try:
        item_pattern = re.compile(
            rf"<{item_tag}(?:\s[^>]*)?>(.+?)</{item_tag}>",
            re.DOTALL,
        )
        items = list(item_pattern.finditer(text))
        if len(items) >= 2:
            first_start = items[0].start()
            last_end = items[-1].end()
            return ExtractionResult(
                documents=[m.group(1).strip() for m in items],
                prefix=text[:first_start],
                suffix=text[last_end:],
                mode="xml_tag",
                wrapper_tag="",
                item_tag=item_tag,
                original_content=text,
            )

    return None


def _extract_numbered(text: str, config: InterceptConfig) -> Optional[ExtractionResult]:
    """Extract documents from numbered format: [1] doc text [2] doc text ..."""
    splits = _NUMBERED_RE.split(text)
    # splits will be like: [prefix, "1", doc1, "2", doc2, ...]
    # If we found numbered items, splits has at least 4 elements (prefix + one item)
    if len(splits) < 4:
        return None

    prefix = splits[0]
    documents = []
    i = 1
    while i + 1 < len(splits):
        # splits[i] is the number, splits[i+1] is the content
        doc_text = splits[i + 1].strip()
        if doc_text:
            documents.append(doc_text)
        i += 2

    if len(documents) < 2:
        return None

    return ExtractionResult(
        documents=documents,
        prefix=prefix,
        suffix="",
        mode="numbered",
        original_content=text,
    )


def _extract_separator(
    text: str, config: InterceptConfig
) -> Optional[ExtractionResult]:
    """Extract documents separated by delimiters (--- or ===)."""
    sep = config.separator
    # For auto mode, try common separators
    if config.mode == "auto":
        for candidate in _SEPARATOR_PATTERNS:
            # Need the separator to appear on its own line (or at boundary)
            parts = re.split(r"\n" + re.escape(candidate) + r"\n", text)
            if len(parts) >= 3:
                sep = candidate
                break
        else:
            return None
        documents = [p.strip() for p in parts if p.strip()]
    else:
        parts = re.split(r"\n" + re.escape(sep) + r"\n", text)
        documents = [p.strip() for p in parts if p.strip()]

    if len(documents) < 2:
        return None

    return ExtractionResult(
        documents=documents,
        prefix="",
        suffix="",
        mode="separator",
        separator_char=sep,
        original_content=text,
    )


def extract_documents(
    text: str, config: InterceptConfig
) -> Optional[ExtractionResult]:
    """Extract documents from text using the configured mode.

    Auto-detection priority: xml_tag > numbered > separator.
    Returns None if no documents are found (caller should bypass).
    """
    if config.mode == "xml_tag":
        return _extract_xml_tags(text, config)
    elif config.mode == "numbered":
        return _extract_numbered(text, config)
    elif config.mode == "separator":
        return _extract_separator(text, config)
    else:
        # Auto mode: try each in priority order
        result = _extract_xml_tags(text, config)
        if result:
            return result
        result = _extract_numbered(text, config)
        if result:
            return result
        result = _extract_separator(text, config)
        if result:
            return result
        return None


# ── Reconstruction ───────────────────────────────────────────────────────────


def reconstruct_content(
    extraction: ExtractionResult, reordered_docs: List[str]
) -> str:
    """Reconstruct the message content with reordered documents.

    Preserves the original format (XML tags, numbering, separators).
    """
    if extraction.mode == "xml_tag":
        return _reconstruct_xml(extraction, reordered_docs)
    elif extraction.mode == "numbered":
        return _reconstruct_numbered(extraction, reordered_docs)
    elif extraction.mode == "separator":
        return _reconstruct_separator(extraction, reordered_docs)
    else:
        # Should not happen, but fallback
        return extraction.original_content


def _reconstruct_xml(
    extraction: ExtractionResult, reordered_docs: List[str]
) -> str:
    item_tag = extraction.item_tag
    items = "\n".join(f"<{item_tag}>{doc}</{item_tag}>" for doc in reordered_docs)

    if extraction.wrapper_tag:
        wrapper = extraction.wrapper_tag
        block = f"<{wrapper}>\n{items}\n</{wrapper}>"
    else:
        block = items

    return extraction.prefix + block + extraction.suffix


def _reconstruct_numbered(
    extraction: ExtractionResult, reordered_docs: List[str]
) -> str:
    parts = [extraction.prefix] if extraction.prefix else []
    for i, doc in enumerate(reordered_docs, 1):
        parts.append(f"[{i}] {doc}")
    result = "\n".join(parts) if parts else ""
    if extraction.suffix:
        result += extraction.suffix
    return result


def _reconstruct_separator(
    extraction: ExtractionResult, reordered_docs: List[str]
) -> str:
    sep = extraction.separator_char or "---"
    return ("\n" + sep + "\n").join(reordered_docs)


# ── OpenAI Chat format ──────────────────────────────────────────────────────


def extract_from_openai_chat(
    body: Dict[str, Any], config: InterceptConfig
) -> Optional[Tuple[ExtractionResult, int]]:
    """Extract documents from an OpenAI chat completions request body.

    Looks for the system message and extracts documents from its content.

    Returns:
        Tuple of (ExtractionResult, system_message_index) or None.
    """
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return None

    for i, msg in enumerate(messages):
        if msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            result = extract_documents(content, config)
            if result:
                return result, i
        elif isinstance(content, list):
            # Content blocks (e.g. [{type: "text", text: "..."}])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    result = extract_documents(block.get("text", ""), config)
                    if result:
                        return result, i
    return None


def reconstruct_openai_chat(
    body: Dict[str, Any],
    extraction: ExtractionResult,
    reordered_docs: List[str],
    system_msg_index: int,
) -> Dict[str, Any]:
    """Reconstruct an OpenAI chat completions request body with reordered docs."""
    body = copy.deepcopy(body)
    new_content = reconstruct_content(extraction, reordered_docs)
    msg = body["messages"][system_msg_index]

    if isinstance(msg.get("content"), str):
        msg["content"] = new_content
    elif isinstance(msg.get("content"), list):
        # Find the text block that matched and replace it
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                if extract_documents(block.get("text", ""), InterceptConfig()):
                    block["text"] = new_content
                    break
    return body


# ── Anthropic Messages format ───────────────────────────────────────────────


def extract_from_anthropic_messages(
    body: Dict[str, Any], config: InterceptConfig
) -> Optional[ExtractionResult]:
    """Extract documents from an Anthropic messages request body.

    Looks at body["system"] which can be a string or list of content blocks.

    Returns:
        ExtractionResult or None.
    """
    system = body.get("system")
    if system is None:
        return None

    if isinstance(system, str):
        return extract_documents(system, config)
    elif isinstance(system, list):
        # Content blocks: [{type: "text", text: "..."}]
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                result = extract_documents(block.get("text", ""), config)
                if result:
                    return result
    return None


def reconstruct_anthropic_messages(
    body: Dict[str, Any],
    extraction: ExtractionResult,
    reordered_docs: List[str],
) -> Dict[str, Any]:
    """Reconstruct an Anthropic messages request body with reordered docs."""
    body = copy.deepcopy(body)
    new_content = reconstruct_content(extraction, reordered_docs)

    if isinstance(body.get("system"), str):
        body["system"] = new_content
    elif isinstance(body.get("system"), list):
        for block in body["system"]:
            if isinstance(block, dict) and block.get("type") == "text":
                if extract_documents(block.get("text", ""), InterceptConfig()):
                    block["text"] = new_content
                    break
    return body
