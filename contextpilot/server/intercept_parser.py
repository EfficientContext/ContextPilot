"""
HTTP Intercept Parser for ContextPilot

Pure parsing/extraction/reconstruction logic for intercepting LLM API requests.
Extracts documents from system messages, supports reordering, and reconstructs
the request body with reordered documents.

No server dependencies — independently testable.
"""

import hashlib
import json
import re
import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# XML wrapper tag names we recognize (outer wrapper)
_KNOWN_WRAPPER_TAGS = {"documents", "contexts", "docs", "passages", "references", "files"}

# XML item tag names we recognize (inner items)
_KNOWN_ITEM_TAGS = {"document", "context", "doc", "passage", "reference", "file"}

# Numbered pattern: [1] ... [2] ... etc.
_NUMBERED_RE = re.compile(r"\[(\d+)\]\s*")

# Separator patterns for auto-detection
_SEPARATOR_PATTERNS = ["---", "==="]

# Minimum content length to track a single-doc tool_result for dedup.
# Skips tiny results like "ok", error messages, short status outputs.
_SINGLE_DOC_MIN_CHARS = 200


@dataclass
class InterceptConfig:
    """Configuration parsed from X-ContextPilot-* headers."""

    enabled: bool = True
    mode: str = "auto"  # xml_tag, separator, numbered, markdown_header, auto
    tag: str = "document"  # XML tag name for xml_tag mode
    separator: str = "---"  # Delimiter for separator mode
    alpha: float = 0.001
    linkage_method: str = "average"
    scope: str = "all"  # "system", "tool_results", "all"


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
    # JSON-results-specific: full result objects (documents = content strings)
    json_items: Optional[List[Any]] = None
    json_envelope_path: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class ToolResultLocation:
    """Identifies a tool_result's position in the messages array."""
    msg_index: int
    block_index: int = -1        # -1 = content is string
    inner_block_index: int = -1  # For Anthropic nested content blocks


@dataclass
class SingleDocExtraction:
    """A tool_result containing a single document (e.g., file read, single API response).

    Not reorderable (only 1 item), but trackable for cross-turn deduplication.
    The content_hash enables efficient comparison without storing full content.
    """
    content: str
    content_hash: str  # SHA-256 of stripped content
    tool_call_id: str = ""  # tool_call_id (OpenAI) or tool_use_id (Anthropic)


@dataclass
class MultiExtractionResult:
    """Aggregates extractions from system prompt and tool_result messages."""
    system_extraction: Optional[Tuple["ExtractionResult", int]] = None
    tool_extractions: List[Tuple["ExtractionResult", ToolResultLocation]] = field(default_factory=list)
    single_doc_extractions: List[Tuple["SingleDocExtraction", ToolResultLocation]] = field(default_factory=list)

    @property
    def has_extractions(self) -> bool:
        return (self.system_extraction is not None
                or len(self.tool_extractions) > 0
                or len(self.single_doc_extractions) > 0)

    @property
    def total_documents(self) -> int:
        total = len(self.single_doc_extractions)
        if self.system_extraction:
            total += len(self.system_extraction[0].documents)
        for ext, _ in self.tool_extractions:
            total += len(ext.documents)
        return total


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

    scope = get("scope", "all").lower()
    if scope not in ("system", "tool_results", "all"):
        scope = "all"

    return InterceptConfig(
        enabled=enabled,
        mode=get("mode", "auto").lower(),
        tag=get("tag", "document").lower(),
        separator=get("separator", "---"),
        alpha=float(get("alpha", "0.001")),
        linkage_method=get("linkage", "average"),
        scope=scope,
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


def _extract_markdown_headers(
    text: str, config: InterceptConfig
) -> Optional[ExtractionResult]:
    """Extract documents by splitting on markdown headers (# or ##).

    Each header + its body becomes one document. Requires >= 2 sections.
    Text before the first header is preserved as prefix.
    """
    # Split on lines that start with # or ## (but not ### or deeper)
    parts = re.split(r"(?=^#{1,2}\s)", text, flags=re.MULTILINE)
    # parts[0] is text before first header (prefix), rest are sections
    if not parts:
        return None

    prefix = ""
    sections = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if re.match(r"^#{1,2}\s", stripped):
            sections.append(stripped)
        else:
            # Text before first header
            prefix = part

    if len(sections) < 2:
        return None

    return ExtractionResult(
        documents=sections,
        prefix=prefix,
        suffix="",
        mode="markdown_header",
        original_content=text,
    )


def _reconstruct_markdown_headers(
    extraction: ExtractionResult, reordered_docs: List[str]
) -> str:
    """Reconstruct markdown-header-split content."""
    parts = []
    if extraction.prefix.strip():
        parts.append(extraction.prefix.rstrip())
    parts.extend(reordered_docs)
    return "\n\n".join(parts)


# Keys that carry a URL / path identifier suitable for clustering.
_JSON_ID_KEYS = ("url", "path", "file", "filename", "uri", "href")

# Keys that carry substantive content text — preferred over URL/path for
# clustering so that the reordering algorithm works on actual page text
# rather than short URL strings.
_JSON_CONTENT_KEYS = ("description", "snippet", "content", "text", "body", "summary")

_JSON_RESULTS_NESTED_KEYS = (
    "stdout",
    "output",
    "result",
    "content",
    "text",
    "aggregated",
)


def _extract_json_id(item: dict) -> Optional[str]:
    """Extract a URL or path identifier from a JSON result item for clustering."""
    for key in _JSON_ID_KEYS:
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_json_content(item: dict) -> Optional[str]:
    """Extract substantive content text from a JSON result item for clustering.

    Prefers description/snippet/content/text over URL/path identifiers so
    that clustering operates on meaningful text rather than short strings.
    """
    for key in _JSON_CONTENT_KEYS:
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_json_results_from_obj(
    obj: Any, path: Tuple[str, ...] = ()
) -> Optional[Tuple[List[str], List[Any], Tuple[str, ...]]]:
    """Extract result identifiers from a parsed JSON object or nested envelope."""
    if not isinstance(obj, dict):
        return None

    results = obj.get("results")
    if isinstance(results, list) and len(results) >= 2:
        documents = []
        for item in results:
            if isinstance(item, dict):
                # Prefer substantive content text for clustering;
                # fall back to URL/path, then full JSON serialisation.
                doc_text = _extract_json_content(item)
                if doc_text is None:
                    doc_text = _extract_json_id(item)
                if doc_text is not None:
                    documents.append(doc_text)
                else:
                    documents.append(json.dumps(item, ensure_ascii=False))
            else:
                documents.append(json.dumps(item, ensure_ascii=False))
        if len(documents) >= 2:
            return documents, results, path

    for key in _JSON_RESULTS_NESTED_KEYS:
        nested = obj.get(key)
        if isinstance(nested, str):
            stripped = nested.strip()
            if not stripped.startswith("{"):
                continue
            try:
                nested_obj = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                continue
            found = _extract_json_results_from_obj(nested_obj, path + (key,))
            if found:
                return found
        elif isinstance(nested, dict):
            found = _extract_json_results_from_obj(nested, path + (key,))
            if found:
                return found
    return None


def _extract_json_results(
    text: str, config: InterceptConfig
) -> Optional[ExtractionResult]:
    """Extract documents from JSON tool results with a ``results`` array.

    OpenClaw tools (memory search, web search, web fetch) return
    ``JSON.stringify(payload, null, 2)`` where *payload* contains a
    ``results`` list.  The content field (description/snippet/content/text)
    of each item is used for clustering; the full objects are stored in
    ``json_items`` so reconstruction can reorder them intact.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    found = _extract_json_results_from_obj(obj)
    if not found:
        return None
    documents, results, envelope_path = found

    return ExtractionResult(
        documents=documents,
        prefix="",
        suffix="",
        mode="json_results",
        original_content=text,
        json_items=results,
        json_envelope_path=envelope_path,
    )


def extract_documents(
    text: str, config: InterceptConfig
) -> Optional[ExtractionResult]:
    """Extract documents from text using the configured mode.

    Auto-detection priority: xml_tag > numbered > json_results.
    ``separator`` and ``markdown_header`` are only used when explicitly
    requested — they match structural content (YAML frontmatter, prompt
    sections) too aggressively for auto mode.
    Returns None if no documents are found (caller should bypass).
    """
    if config.mode == "xml_tag":
        return _extract_xml_tags(text, config)
    elif config.mode == "numbered":
        return _extract_numbered(text, config)
    elif config.mode == "json_results":
        return _extract_json_results(text, config)
    elif config.mode == "separator":
        return _extract_separator(text, config)
    elif config.mode == "markdown_header":
        return _extract_markdown_headers(text, config)
    else:
        # Auto mode: only formats that clearly delimit independent documents.
        # separator and markdown_header excluded — too aggressive on
        # structural content (YAML frontmatter, prompt sections).
        result = _extract_xml_tags(text, config)
        if result:
            return result
        result = _extract_numbered(text, config)
        if result:
            return result
        result = _extract_json_results(text, config)
        if result:
            return result
        return None


# ── Reconstruction ───────────────────────────────────────────────────────────


def reconstruct_content(
    extraction: ExtractionResult, reordered_docs: List[str]
) -> str:
    """Reconstruct the message content with reordered documents.

    Preserves the original format (XML tags, numbering, separators, markdown headers).
    """
    if extraction.mode == "xml_tag":
        return _reconstruct_xml(extraction, reordered_docs)
    elif extraction.mode == "numbered":
        return _reconstruct_numbered(extraction, reordered_docs)
    elif extraction.mode == "json_results":
        return _reconstruct_json_results(extraction, reordered_docs)
    elif extraction.mode == "separator":
        return _reconstruct_separator(extraction, reordered_docs)
    elif extraction.mode == "markdown_header":
        return _reconstruct_markdown_headers(extraction, reordered_docs)
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


def _reconstruct_json_results(
    extraction: ExtractionResult, reordered_docs: List[str]
) -> str:
    """Reconstruct a JSON tool result with reordered ``results`` array.

    When ``json_items`` is set (content-key extraction), maps reordered
    content strings back to their full result objects via index lookup.
    Falls back to parsing reordered_docs as JSON for backward compat.
    """
    obj = json.loads(extraction.original_content)
    if extraction.json_items is not None:
        # Build content → original-index mapping
        orig_docs = extraction.documents
        doc_to_indices: dict = {}
        for i, doc in enumerate(orig_docs):
            doc_to_indices.setdefault(doc, []).append(i)
        used: set = set()
        reordered_items = []
        for doc in reordered_docs:
            for idx in doc_to_indices.get(doc, []):
                if idx not in used:
                    reordered_items.append(extraction.json_items[idx])
                    used.add(idx)
                    break
        reordered_results = reordered_items
    else:
        reordered_results = [json.loads(doc) for doc in reordered_docs]

    if extraction.json_envelope_path:
        container = obj
        for key in extraction.json_envelope_path[:-1]:
            container = container[key]
        leaf_key = extraction.json_envelope_path[-1]
        leaf = container.get(leaf_key)
        if isinstance(leaf, str):
            nested_obj = json.loads(leaf)
            nested_obj["results"] = reordered_results
            container[leaf_key] = json.dumps(nested_obj, indent=2, ensure_ascii=False)
        elif isinstance(leaf, dict):
            leaf["results"] = reordered_results
        else:
            raise TypeError(f"Unsupported JSON envelope leaf type for {leaf_key!r}: {type(leaf)!r}")
    else:
        obj["results"] = reordered_results
    return json.dumps(obj, indent=2, ensure_ascii=False)


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


# ── Tool result extraction ─────────────────────────────────────────────────


def extract_from_openai_tool_results(
    body: Dict[str, Any], config: InterceptConfig
) -> List[Tuple[ExtractionResult, ToolResultLocation]]:
    """Extract documents from OpenAI tool result messages (role=="tool").

    Returns a list of (ExtractionResult, ToolResultLocation) for each tool
    result message that contains extractable documents.
    """
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return []

    results = []
    for i, msg in enumerate(messages):
        if msg.get("role") not in ("tool", "toolResult"):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            extraction = extract_documents(content, config)
            if extraction and len(extraction.documents) >= 2:
                loc = ToolResultLocation(msg_index=i)
                results.append((extraction, loc))
        elif isinstance(content, list):
            for j, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "text":
                    extraction = extract_documents(block.get("text", ""), config)
                    if extraction and len(extraction.documents) >= 2:
                        loc = ToolResultLocation(msg_index=i, block_index=j)
                        results.append((extraction, loc))
    return results


def extract_from_anthropic_tool_results(
    body: Dict[str, Any], config: InterceptConfig
) -> List[Tuple[ExtractionResult, ToolResultLocation]]:
    """Extract documents from Anthropic tool_result content blocks.

    Anthropic tool results appear as messages with role=="user" containing
    content blocks of type=="tool_result".
    """
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return []

    results = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") not in ("tool_result", "toolResult"):
                continue
            tr_content = block.get("content", "")
            if isinstance(tr_content, str):
                extraction = extract_documents(tr_content, config)
                if extraction and len(extraction.documents) >= 2:
                    loc = ToolResultLocation(msg_index=i, block_index=j)
                    results.append((extraction, loc))
            elif isinstance(tr_content, list):
                for k, inner in enumerate(tr_content):
                    if isinstance(inner, dict) and inner.get("type") == "text":
                        extraction = extract_documents(inner.get("text", ""), config)
                        if extraction and len(extraction.documents) >= 2:
                            loc = ToolResultLocation(msg_index=i, block_index=j, inner_block_index=k)
                            results.append((extraction, loc))
    return results


# ── Tool result reconstruction ─────────────────────────────────────────────


def reconstruct_openai_tool_result(
    body: Dict[str, Any],
    extraction: ExtractionResult,
    reordered_docs: List[str],
    location: ToolResultLocation,
) -> None:
    """Reconstruct an OpenAI tool result message in-place."""
    new_content = reconstruct_content(extraction, reordered_docs)
    msg = body["messages"][location.msg_index]
    if location.block_index == -1:
        msg["content"] = new_content
    else:
        msg["content"][location.block_index]["text"] = new_content


def reconstruct_anthropic_tool_result(
    body: Dict[str, Any],
    extraction: ExtractionResult,
    reordered_docs: List[str],
    location: ToolResultLocation,
) -> None:
    """Reconstruct an Anthropic tool_result content block in-place."""
    new_content = reconstruct_content(extraction, reordered_docs)
    msg = body["messages"][location.msg_index]
    block = msg["content"][location.block_index]
    if location.inner_block_index == -1:
        block["content"] = new_content
    else:
        block["content"][location.inner_block_index]["text"] = new_content


# ── Aggregate extraction ───────────────────────────────────────────────────


def extract_all_openai(
    body: Dict[str, Any], config: InterceptConfig
) -> MultiExtractionResult:
    """Extract documents from both system message and tool results (OpenAI format)."""
    result = MultiExtractionResult()
    if config.scope in ("system", "all"):
        sys_result = extract_from_openai_chat(body, config)
        if sys_result:
            result.system_extraction = sys_result
    if config.scope in ("tool_results", "all"):
        result.tool_extractions = extract_from_openai_tool_results(body, config)
        result.single_doc_extractions = extract_single_docs_from_openai_tool_results(
            body, config)
    return result


def extract_all_anthropic(
    body: Dict[str, Any], config: InterceptConfig
) -> MultiExtractionResult:
    """Extract documents from both system prompt and tool results (Anthropic format)."""
    result = MultiExtractionResult()
    if config.scope in ("system", "all"):
        sys_extraction = extract_from_anthropic_messages(body, config)
        if sys_extraction and len(sys_extraction.documents) >= 2:
            # Use -1 as sentinel for "system field" (not in messages array)
            result.system_extraction = (sys_extraction, -1)
    if config.scope in ("tool_results", "all"):
        result.tool_extractions = extract_from_anthropic_tool_results(body, config)
        result.single_doc_extractions = extract_single_docs_from_anthropic_tool_results(
            body, config)
    return result


# ── Single-document extraction (for cross-turn dedup) ─────────────────────


def _make_single_doc(content: str, tool_call_id: str = "") -> SingleDocExtraction:
    """Create a SingleDocExtraction with content hash."""
    stripped = content.strip()
    content_hash = hashlib.sha256(stripped.encode()).hexdigest()
    return SingleDocExtraction(
        content=stripped,
        content_hash=content_hash,
        tool_call_id=tool_call_id,
    )


def extract_single_docs_from_openai_tool_results(
    body: Dict[str, Any], config: InterceptConfig
) -> List[Tuple[SingleDocExtraction, ToolResultLocation]]:
    """Extract single-document tool results for dedup tracking (OpenAI format).

    Only captures tool_results where multi-doc extraction failed — these are
    individual file reads, single API responses, etc. that contain substantial
    content worth deduplicating across turns.
    """
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return []

    results = []
    for i, msg in enumerate(messages):
        if msg.get("role") not in ("tool", "toolResult"):
            continue
        tool_call_id = msg.get("tool_call_id", "")
        content = msg.get("content", "")

        if isinstance(content, str):
            # Skip if multi-doc extraction would succeed
            extraction = extract_documents(content, config)
            if extraction and len(extraction.documents) >= 2:
                continue
            if len(content.strip()) >= _SINGLE_DOC_MIN_CHARS:
                loc = ToolResultLocation(msg_index=i)
                results.append((_make_single_doc(content, tool_call_id), loc))
        elif isinstance(content, list):
            for j, block in enumerate(content):
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = block.get("text", "")
                extraction = extract_documents(text, config)
                if extraction and len(extraction.documents) >= 2:
                    continue
                if len(text.strip()) >= _SINGLE_DOC_MIN_CHARS:
                    loc = ToolResultLocation(msg_index=i, block_index=j)
                    results.append((_make_single_doc(text, tool_call_id), loc))
    return results


def extract_single_docs_from_anthropic_tool_results(
    body: Dict[str, Any], config: InterceptConfig
) -> List[Tuple[SingleDocExtraction, ToolResultLocation]]:
    """Extract single-document tool results for dedup tracking (Anthropic format).

    Anthropic tool results appear as user messages with type=="tool_result" blocks.
    """
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return []

    results = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") not in ("tool_result", "toolResult"):
                continue
            tool_use_id = block.get("tool_use_id", "")
            tr_content = block.get("content", "")

            if isinstance(tr_content, str):
                extraction = extract_documents(tr_content, config)
                if extraction and len(extraction.documents) >= 2:
                    continue
                if len(tr_content.strip()) >= _SINGLE_DOC_MIN_CHARS:
                    loc = ToolResultLocation(msg_index=i, block_index=j)
                    results.append((_make_single_doc(tr_content, tool_use_id), loc))
            elif isinstance(tr_content, list):
                for k, inner in enumerate(tr_content):
                    if not isinstance(inner, dict) or inner.get("type") != "text":
                        continue
                    text = inner.get("text", "")
                    extraction = extract_documents(text, config)
                    if extraction and len(extraction.documents) >= 2:
                        continue
                    if len(text.strip()) >= _SINGLE_DOC_MIN_CHARS:
                        loc = ToolResultLocation(msg_index=i, block_index=j, inner_block_index=k)
                        results.append((_make_single_doc(text, tool_use_id), loc))
    return results


# ── Single-document hint replacement ──────────────────────────────────────


def replace_single_doc_openai(
    body: Dict[str, Any], location: ToolResultLocation, hint: str,
) -> None:
    """Replace a single-doc OpenAI tool result content with a dedup hint in-place."""
    msg = body["messages"][location.msg_index]
    if location.block_index == -1:
        msg["content"] = hint
    else:
        msg["content"][location.block_index]["text"] = hint


def replace_single_doc_anthropic(
    body: Dict[str, Any], location: ToolResultLocation, hint: str,
) -> None:
    """Replace a single-doc Anthropic tool_result content with a dedup hint in-place."""
    msg = body["messages"][location.msg_index]
    block = msg["content"][location.block_index]
    if location.inner_block_index == -1:
        block["content"] = hint
    else:
        block["content"][location.inner_block_index]["text"] = hint


# ── Format handler abstraction ─────────────────────────────────────────────
# Strategy pattern: encapsulates all format-specific operations so the
# intercept logic in http_server.py can be completely format-agnostic.


class FormatHandler(ABC):
    """Abstract handler for API format-specific operations.

    Implement one per LLM API format (OpenAI Chat, Anthropic Messages, etc.).
    The intercept pipeline calls these methods instead of branching on format.
    """

    @abstractmethod
    def extract_all(self, body: Dict[str, Any],
                    config: InterceptConfig) -> MultiExtractionResult: ...

    @abstractmethod
    def reconstruct_system(self, body: Dict[str, Any],
                           extraction: ExtractionResult,
                           docs: List[str], sys_idx: int) -> None: ...

    @abstractmethod
    def reconstruct_tool_result(self, body: Dict[str, Any],
                                extraction: ExtractionResult,
                                docs: List[str],
                                location: ToolResultLocation) -> None: ...

    @abstractmethod
    def replace_single_doc(self, body: Dict[str, Any],
                           location: ToolResultLocation,
                           hint: str) -> None: ...

    @abstractmethod
    def tool_call_present(self, body: Dict[str, Any],
                          tool_call_id: str) -> bool: ...

    @abstractmethod
    def target_path(self) -> str: ...

    @abstractmethod
    def cache_system(self, body: Dict[str, Any]) -> Any: ...

    @abstractmethod
    def restore_system(self, body: Dict[str, Any], cached: Any) -> None: ...


class OpenAIChatHandler(FormatHandler):
    """Handler for OpenAI Chat Completions format."""

    def extract_all(self, body, config):
        return extract_all_openai(body, config)

    def reconstruct_system(self, body, extraction, docs, sys_idx):
        new_content = reconstruct_content(extraction, docs)
        msg = body["messages"][sys_idx]
        if isinstance(msg.get("content"), str):
            msg["content"] = new_content
        elif isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    if extract_documents(block.get("text", ""), InterceptConfig()):
                        block["text"] = new_content
                        break

    def reconstruct_tool_result(self, body, extraction, docs, location):
        reconstruct_openai_tool_result(body, extraction, docs, location)

    def replace_single_doc(self, body, location, hint):
        replace_single_doc_openai(body, location, hint)

    def tool_call_present(self, body, tool_call_id):
        for msg in (body.get("messages") or []):
            if msg.get("role") in ("tool", "toolResult"):
                if msg.get("tool_call_id") == tool_call_id:
                    return True
        return False

    def target_path(self):
        return "/v1/chat/completions"

    def cache_system(self, body):
        return None  # System prompt is inside messages array

    def restore_system(self, body, cached):
        pass  # No-op: cached with messages


class AnthropicMessagesHandler(FormatHandler):
    """Handler for Anthropic Messages API format."""

    def extract_all(self, body, config):
        return extract_all_anthropic(body, config)

    def reconstruct_system(self, body, extraction, docs, sys_idx):
        new_content = reconstruct_content(extraction, docs)
        if isinstance(body.get("system"), str):
            body["system"] = new_content
        elif isinstance(body.get("system"), list):
            for block in body["system"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    if extract_documents(block.get("text", ""), InterceptConfig()):
                        block["text"] = new_content
                        break

    def reconstruct_tool_result(self, body, extraction, docs, location):
        reconstruct_anthropic_tool_result(body, extraction, docs, location)

    def replace_single_doc(self, body, location, hint):
        replace_single_doc_anthropic(body, location, hint)

    def tool_call_present(self, body, tool_call_id):
        for msg in (body.get("messages") or []):
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if (isinstance(block, dict)
                            and block.get("type") in ("tool_result", "toolResult")
                            and block.get("tool_use_id") == tool_call_id):
                        return True
        return False

    def target_path(self):
        return "/v1/messages"

    def cache_system(self, body):
        return copy.deepcopy(body.get("system"))

    def restore_system(self, body, cached):
        if cached is not None:
            body["system"] = copy.deepcopy(cached)


# Handler registry — add new formats here.
_FORMAT_HANDLERS: Dict[str, FormatHandler] = {
    "openai_chat": OpenAIChatHandler(),
    "anthropic_messages": AnthropicMessagesHandler(),
}


def get_format_handler(api_format: str) -> FormatHandler:
    """Return the handler for the given API format, defaulting to OpenAI."""
    return _FORMAT_HANDLERS.get(api_format, _FORMAT_HANDLERS["openai_chat"])
