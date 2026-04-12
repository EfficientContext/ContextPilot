import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MIN_BLOCK_CHARS = 80
MIN_CONTENT_CHARS = 500

CHUNK_MODULUS = 13
CHUNK_MIN_LINES = 5
CHUNK_MAX_LINES = 40


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
        line_hash = int.from_bytes(
            hashlib.md5(line.strip().encode("utf-8", errors="replace")).digest()[:4],
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
    normalized = block.strip()
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


def dedup_chat_completions(
    body: dict,
    min_block_chars: int = MIN_BLOCK_CHARS,
    min_content_chars: int = MIN_CONTENT_CHARS,
    chunk_modulus: int = CHUNK_MODULUS,
    system_content: Optional[str] = None,
) -> DedupResult:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return DedupResult()

    tool_names = _build_tool_name_map_openai(messages)
    seen_blocks: Dict[str, Tuple[int, str, int]] = {}
    pre_seen = _prescan_system_blocks(system_content, min_block_chars, chunk_modulus)
    result = DedupResult()

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue

        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) < min_content_chars:
            continue

        tc_id = msg.get("tool_call_id", "")
        fn_name = tool_names.get(tc_id, msg.get("name", "")) or "tool"

        new_content = _dedup_text(
            content,
            seen_blocks,
            idx,
            fn_name,
            result,
            min_block_chars,
            chunk_modulus,
            pre_seen=pre_seen,
        )
        if new_content is not None:
            original_len = len(content)
            msg["content"] = new_content
            new_len = len(new_content)
            result.chars_before += original_len
            result.chars_after += new_len
            result.chars_saved += original_len - new_len
            logger.info(
                f"Block dedup: msg[{idx}] {fn_name} — "
                f"saved {original_len - new_len:,} chars"
            )

    _dedup_assistant_code_blocks(
        messages,
        seen_blocks,
        result,
        min_block_chars,
        min_content_chars,
        chunk_modulus,
        pre_seen=pre_seen,
    )

    return result


_CODE_BLOCK_RE = re.compile(r"(```[\w]*\n)(.*?)(```)", re.DOTALL)


def _dedup_assistant_code_blocks(
    messages: list,
    seen_blocks: Dict[str, Tuple[int, str, int]],
    result: DedupResult,
    min_block_chars: int,
    min_content_chars: int,
    chunk_modulus: int,
    pre_seen: Optional[Dict[str, Tuple[int, str, int]]] = None,
) -> None:
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        raw_content = msg.get("content", "")

        # Handle both string and list (content blocks) formats
        is_list_content = False
        text_block_idx = -1
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            # OpenClaw sends [{type: "text", text: "..."}, ...]
            # Find the text block that contains code
            content = ""
            for bi, block in enumerate(raw_content):
                if isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text", "")
                    if "```" in t and len(t) > len(content):
                        content = t
                        text_block_idx = bi
                        is_list_content = True
            if not content:
                continue
        else:
            continue

        if len(content) < min_content_chars:
            continue

        code_blocks = list(_CODE_BLOCK_RE.finditer(content))
        if not code_blocks:
            continue

        modified = False
        new_content = content

        for match in reversed(code_blocks):
            code = match.group(2)
            if len(code.strip()) < min_block_chars:
                continue

            new_code = _dedup_text(
                code,
                seen_blocks,
                idx,
                "assistant",
                result,
                min_block_chars,
                chunk_modulus,
                pre_seen=pre_seen,
            )
            if new_code is not None:
                start, end = match.start(2), match.end(2)
                original_len = end - start
                new_content = new_content[:start] + new_code + new_content[end:]
                result.chars_before += original_len
                result.chars_after += len(new_code)
                result.chars_saved += original_len - len(new_code)
                modified = True

        if modified:
            if is_list_content and text_block_idx >= 0:
                msg["content"][text_block_idx]["text"] = new_content
            else:
                msg["content"] = new_content


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

        new_output = _dedup_text(
            output,
            seen_blocks,
            idx,
            fn_name,
            result,
            min_block_chars,
            chunk_modulus,
            pre_seen=pre_seen,
        )
        if new_output is not None:
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
