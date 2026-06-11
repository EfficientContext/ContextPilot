import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MIN_BLOCK_CHARS = 40
MIN_CONTENT_CHARS = 200

CHUNK_MODULUS = 13
CHUNK_MIN_LINES = 3
CHUNK_MAX_LINES = 40

# Matches line-number prefixes like "     1|", "    42|", "   100|" etc.
_LINE_NUM_PREFIX_RE = re.compile(r"^\s*\d+\|")


@dataclass
class DedupResult:
    blocks_deduped: int = 0
    blocks_total: int = 0
    system_blocks_matched: int = 0
    chars_before: int = 0
    chars_after: int = 0
    chars_saved: int = 0


def _build_tool_name_map_openai(messages: list) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id", "")
            fn = tc.get("function", {})
            if isinstance(fn, dict) and fn.get("name"):
                mapping[tc_id] = fn["name"]
    return mapping


def _build_tool_name_map_responses(items: list) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            cid = item.get("call_id", "")
            name = item.get("name", "")
            if cid and name:
                mapping[cid] = name
    return mapping


def _strip_line_prefix(line: str) -> str:
    """Strip line-number prefixes (e.g. '     1|') for normalization."""
    return _LINE_NUM_PREFIX_RE.sub("", line)


def _content_defined_chunking(
    text: str, chunk_modulus: int = CHUNK_MODULUS
) -> List[str]:
    lines = text.split("\n")
    if len(lines) <= CHUNK_MIN_LINES:
        return [text]

    blocks: List[str] = []
    current: List[str] = []

    for line in lines:
        current.append(line)
        # Strip line-number prefixes before hashing for boundary detection
        # so the same source line produces the same boundary decision
        # regardless of its position in different files.
        normalized_line = _strip_line_prefix(line).strip()
        line_hash = int.from_bytes(
            hashlib.md5(normalized_line.encode("utf-8", errors="replace")).digest()[:4],
            "little",
        )
        is_boundary = (
            line_hash % chunk_modulus == 0 and len(current) >= CHUNK_MIN_LINES
        ) or len(current) >= CHUNK_MAX_LINES
        if is_boundary:
            blocks.append("\n".join(current))
            current = []

    if current:
        if blocks and len(current) < CHUNK_MIN_LINES:
            blocks[-1] += "\n" + "\n".join(current)
        else:
            blocks.append("\n".join(current))

    return blocks


def _hash_block(block: str) -> str:
    """Hash a block for dedup comparison.

    Strips line-number prefixes before hashing so identical source code
    at different line offsets produces the same hash.
    """
    lines = block.strip().split("\n")
    normalized = "\n".join(_strip_line_prefix(line) for line in lines)
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()[:20]


def _dedup_text(
    text: str,
    seen_blocks: Dict[str, Tuple[int, str, int]],
    msg_idx: int,
    fn_name: str,
    result: DedupResult,
    min_block_chars: int,
    chunk_modulus: int,
    pre_seen: Optional[Dict[str, Tuple[int, str, int]]] = None,
) -> Optional[str]:
    """Core dedup loop shared by all entry points.

    Returns the deduped text if any blocks were deduped, or None otherwise.
    """
    if pre_seen:
        for h, origin in pre_seen.items():
            if h not in seen_blocks:
                seen_blocks[h] = origin

    blocks = _content_defined_chunking(text, chunk_modulus)
    if len(blocks) < 2:
        for b in blocks:
            if len(b.strip()) >= min_block_chars:
                h = _hash_block(b)
                result.blocks_total += 1
                if h not in seen_blocks:
                    seen_blocks[h] = (msg_idx, fn_name, 0)
        return None

    new_blocks = []
    deduped_count = 0

    for block_idx, block in enumerate(blocks):
        if len(block.strip()) < min_block_chars:
            new_blocks.append(block)
            continue

        h = _hash_block(block)
        result.blocks_total += 1

        if h in seen_blocks and seen_blocks[h][0] != msg_idx:
            orig_msg_idx, orig_fn, _ = seen_blocks[h]
            first_line = block.strip().split("\n")[0][:80]
            ref = f'[... "{first_line}" — identical to earlier {orig_fn} result, see above ...]'
            if orig_msg_idx == -1:
                result.system_blocks_matched += 1
            chars_saved = len(block) - len(ref)
            if chars_saved > 0:
                new_blocks.append(ref)
                deduped_count += 1
                result.blocks_deduped += 1
            else:
                new_blocks.append(block)
        else:
            if h not in seen_blocks:
                seen_blocks[h] = (msg_idx, fn_name, block_idx)
            new_blocks.append(block)

    if deduped_count > 0:
        return "\n".join(new_blocks)

    # Nothing deduped — register all blocks for future lookups
    for block_idx, block in enumerate(blocks):
        if len(block.strip()) >= min_block_chars:
            h = _hash_block(block)
            if h not in seen_blocks:
                seen_blocks[h] = (msg_idx, fn_name, block_idx)
    return None


def _prescan_system_blocks(
    system_content: Optional[str],
    min_block_chars: int,
    chunk_modulus: int,
) -> Dict[str, Tuple[int, str, int]]:
    """Hash and register dedup-eligible blocks from system prompt content."""
    pre_seen: Dict[str, Tuple[int, str, int]] = {}
    if not isinstance(system_content, str) or not system_content.strip():
        return pre_seen

    blocks = _content_defined_chunking(system_content, chunk_modulus)
    for block_idx, block in enumerate(blocks):
        if len(block.strip()) < min_block_chars:
            continue
        h = _hash_block(block)
        if h not in pre_seen:
            pre_seen[h] = (-1, "system prompt", block_idx)
    return pre_seen


def _extract_text_for_dedup(content: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract the dedupable text from a tool result.

    Many tools return JSON like {"content": "...", "path": "..."}.
    The content field is the actual multi-line text we want to dedup.

    Returns (text_to_dedup, json_key) or (None, None) if not JSON-wrapped.
    """
    stripped = content.strip()
    if not stripped.startswith("{"):
        return None, None
    try:
        import json as _json
        obj = _json.loads(stripped)
        if not isinstance(obj, dict):
            return None, None
        # Look for the primary text field
        for key in ("content", "output", "result", "text", "stdout"):
            val = obj.get(key)
            if isinstance(val, str) and len(val) >= MIN_CONTENT_CHARS:
                return val, key
    except (ValueError, TypeError):
        pass
    return None, None


def _rebuild_json_content(original: str, key: str, new_text: str) -> str:
    """Replace the text field in a JSON tool result with deduped version."""
    import json as _json
    try:
        obj = _json.loads(original)
        obj[key] = new_text
        return _json.dumps(obj, ensure_ascii=False)
    except (ValueError, TypeError):
        return original


def _account(result: DedupResult, original_len: int, new_len: int) -> None:
    """Roll a single field's before/after lengths into the aggregate result."""
    result.chars_before += original_len
    result.chars_after += new_len
    result.chars_saved += original_len - new_len


def _register_blocks_only(
    text: Optional[str],
    seen_blocks: Dict[str, Tuple[int, str, int]],
    msg_idx: int,
    label: str,
    result: DedupResult,
    min_block_chars: int,
    chunk_modulus: int,
) -> None:
    """Register a message's blocks as dedup *sources* without modifying it.

    Used for the canonical first copy — e.g. the system / skill prompt — which
    may seed references for later duplicates but must itself stay verbatim.
    """
    if not isinstance(text, str) or not text.strip():
        return
    for block_idx, block in enumerate(_content_defined_chunking(text, chunk_modulus)):
        if len(block.strip()) < min_block_chars:
            continue
        result.blocks_total += 1
        h = _hash_block(block)
        if h not in seen_blocks:
            seen_blocks[h] = (msg_idx, label, block_idx)


def _dedup_string_field(
    text: str,
    seen_blocks: Dict[str, Tuple[int, str, int]],
    msg_idx: int,
    label: str,
    result: DedupResult,
    min_block_chars: int,
    chunk_modulus: int,
) -> Optional[str]:
    """Dedup one plain-text field against earlier blocks in the same payload."""
    new_text = _dedup_text(
        text, seen_blocks, msg_idx, label, result, min_block_chars, chunk_modulus
    )
    if new_text is not None:
        _account(result, len(text), len(new_text))
        logger.info(
            "Block dedup: msg[%d] %s — saved %d chars",
            msg_idx,
            label,
            len(text) - len(new_text),
        )
    return new_text


def _dedup_assistant_message(
    msg: dict,
    idx: int,
    seen_blocks: Dict[str, Tuple[int, str, int]],
    result: DedupResult,
    min_block_chars: int,
    min_content_chars: int,
    chunk_modulus: int,
) -> None:
    """Dedup assistant content (string or list-of-text-blocks) against earlier blocks."""
    raw = msg.get("content", "")
    if isinstance(raw, str):
        if len(raw) < min_content_chars:
            return
        new_content = _dedup_string_field(
            raw, seen_blocks, idx, "assistant message", result,
            min_block_chars, chunk_modulus,
        )
        if new_content is not None:
            msg["content"] = new_content
    elif isinstance(raw, list):
        # OpenClaw sends [{type: "text", text: "..."}, ...]
        for block in raw:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            t = block.get("text", "")
            if not isinstance(t, str) or len(t) < min_content_chars:
                continue
            new_text = _dedup_string_field(
                t, seen_blocks, idx, "assistant message", result,
                min_block_chars, chunk_modulus,
            )
            if new_text is not None:
                block["text"] = new_text


def _dedup_tool_message(
    msg: dict,
    idx: int,
    tool_names: Dict[str, str],
    seen_blocks: Dict[str, Tuple[int, str, int]],
    result: DedupResult,
    min_block_chars: int,
    min_content_chars: int,
    chunk_modulus: int,
) -> None:
    """Dedup a tool result (JSON-aware) against earlier blocks in the payload."""
    content = msg.get("content", "")
    if not isinstance(content, str) or len(content) < min_content_chars:
        return

    tc_id = msg.get("tool_call_id", "")
    fn_name = tool_names.get(tc_id, msg.get("name", "")) or "tool"

    # Extract text from JSON-wrapped tool results for proper chunking.
    extracted_text, json_key = _extract_text_for_dedup(content)
    dedup_target = extracted_text if extracted_text else content

    new_content = _dedup_text(
        dedup_target, seen_blocks, idx, fn_name, result, min_block_chars, chunk_modulus
    )
    if new_content is None:
        return

    original_len = len(content)
    if json_key and extracted_text:
        msg["content"] = _rebuild_json_content(content, json_key, new_content)
    else:
        msg["content"] = new_content
    new_len = len(msg["content"])
    _account(result, original_len, new_len)
    logger.info(
        "Block dedup: msg[%d] %s — saved %d chars", idx, fn_name, original_len - new_len
    )


def dedup_chat_completions(
    body: dict,
    min_block_chars: int = MIN_BLOCK_CHARS,
    min_content_chars: int = MIN_CONTENT_CHARS,
    chunk_modulus: int = CHUNK_MODULUS,
    system_content: Optional[str] = None,
) -> DedupResult:
    """Exact-block dedup across ALL roles within a single chat payload.

    Walks messages in document order with a shared block table. The first
    (earliest) occurrence of any block — across system/skill prompt, user,
    assistant, and tool messages — keeps its full text; later EXACT occurrences
    anywhere in the *same* payload are replaced by a short reference pointing to
    the earlier copy ("see above"). Only exact hash matches are ever replaced;
    references only ever point backward, to a block in this same payload.

    The system / skill prompt is treated as the canonical source: its blocks are
    registered but it is never itself shortened.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return DedupResult()

    tool_names = _build_tool_name_map_openai(messages)
    seen_blocks: Dict[str, Tuple[int, str, int]] = {}
    result = DedupResult()

    # Seed an externally-supplied system / skill prompt (e.g. one not present as
    # a message in `messages`) as the canonical first copy. Registered at -1 so
    # later matches are attributed as system-block hits.
    pre_seen = _prescan_system_blocks(system_content, min_block_chars, chunk_modulus)
    for h, origin in pre_seen.items():
        seen_blocks.setdefault(h, origin)

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        if role == "system":
            # Canonical source — register but never shorten. Use -1 so downstream
            # matches count as system-block hits, consistent with `pre_seen`.
            _register_blocks_only(
                msg.get("content", ""), seen_blocks, -1, "system prompt",
                result, min_block_chars, chunk_modulus,
            )

        elif role == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) >= min_content_chars:
                new_content = _dedup_string_field(
                    content, seen_blocks, idx, "user message", result,
                    min_block_chars, chunk_modulus,
                )
                if new_content is not None:
                    msg["content"] = new_content

        elif role == "assistant":
            _dedup_assistant_message(
                msg, idx, seen_blocks, result,
                min_block_chars, min_content_chars, chunk_modulus,
            )

        elif role == "tool":
            _dedup_tool_message(
                msg, idx, tool_names, seen_blocks, result,
                min_block_chars, min_content_chars, chunk_modulus,
            )

    return result


def dedup_responses_api(
    body: dict,
    min_block_chars: int = MIN_BLOCK_CHARS,
    min_content_chars: int = MIN_CONTENT_CHARS,
    chunk_modulus: int = CHUNK_MODULUS,
    system_content: Optional[str] = None,
) -> DedupResult:
    input_items = body.get("input")
    if not isinstance(input_items, list) or not input_items:
        return DedupResult()

    fn_names = _build_tool_name_map_responses(input_items)
    seen_blocks: Dict[str, Tuple[int, str, int]] = {}
    pre_seen = _prescan_system_blocks(system_content, min_block_chars, chunk_modulus)
    result = DedupResult()

    for idx, item in enumerate(input_items):
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue

        output = item.get("output", "")
        if not isinstance(output, str) or len(output) < min_content_chars:
            continue

        call_id = item.get("call_id", "")
        fn_name = fn_names.get(call_id, call_id) or "tool"

        extracted_text, json_key = _extract_text_for_dedup(output)
        dedup_target = extracted_text if extracted_text else output

        new_output = _dedup_text(
            dedup_target,
            seen_blocks,
            idx,
            fn_name,
            result,
            min_block_chars,
            chunk_modulus,
            pre_seen=pre_seen,
        )
        if new_output is not None:
            if json_key and extracted_text:
                original_len = len(output)
                item["output"] = _rebuild_json_content(output, json_key, new_output)
                new_len = len(item["output"])
            else:
                original_len = len(output)
                item["output"] = new_output
                new_len = len(new_output)
            result.chars_before += original_len
            result.chars_after += new_len
            result.chars_saved += original_len - new_len
            logger.info(
                f"Block dedup: input[{idx}] {fn_name} — "
                f"saved {original_len - new_len:,} chars"
            )

    return result
