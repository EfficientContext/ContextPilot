import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def _content_defined_chunking(text: str) -> List[str]:
    lines = text.split("\n")
    if len(lines) <= CHUNK_MIN_LINES:
        return [text]

    blocks: List[str] = []
    current: List[str] = []

    for line in lines:
        current.append(line)
        line_hash = hash(line.strip()) & 0xFFFFFFFF
        is_boundary = (
            line_hash % CHUNK_MODULUS == 0 and len(current) >= CHUNK_MIN_LINES
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


def dedup_chat_completions(
    body: dict,
    min_block_chars: int = MIN_BLOCK_CHARS,
    min_content_chars: int = MIN_CONTENT_CHARS,
) -> DedupResult:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return DedupResult()

    tool_names = _build_tool_name_map_openai(messages)
    seen_blocks: Dict[str, Tuple[int, str, int]] = {}
    result = DedupResult()

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue

        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) < min_content_chars:
            continue

        tc_id = msg.get("tool_call_id", "")
        fn_name = tool_names.get(tc_id, msg.get("name", "")) or "tool"

        blocks = _content_defined_chunking(content)
        if len(blocks) < 2:
            for b in blocks:
                if len(b.strip()) >= min_block_chars:
                    h = _hash_block(b)
                    if h not in seen_blocks:
                        seen_blocks[h] = (idx, fn_name, 0)
            continue

        new_blocks = []
        deduped_in_this = 0

        for block_idx, block in enumerate(blocks):
            if len(block.strip()) < min_block_chars:
                new_blocks.append(block)
                continue

            h = _hash_block(block)
            result.blocks_total += 1

            if h in seen_blocks and seen_blocks[h][0] != idx:
                _, orig_fn, _ = seen_blocks[h]
                first_line = block.strip().split("\n")[0][:80]
                ref = f'[... "{first_line}" — identical to earlier {orig_fn} result, see above ...]'
                chars_saved = len(block) - len(ref)
                if chars_saved > 0:
                    new_blocks.append(ref)
                    deduped_in_this += 1
                    result.blocks_deduped += 1
                else:
                    new_blocks.append(block)
            else:
                if h not in seen_blocks:
                    seen_blocks[h] = (idx, fn_name, block_idx)
                new_blocks.append(block)

        if deduped_in_this > 0:
            original_len = len(content)
            new_content = "\n\n".join(new_blocks)
            msg["content"] = new_content
            new_len = len(new_content)
            result.chars_before += original_len
            result.chars_after += new_len
            result.chars_saved += original_len - new_len
            logger.info(
                f"Block dedup: msg[{idx}] {fn_name} — "
                f"{deduped_in_this}/{len(blocks)} blocks, "
                f"saved {original_len - new_len:,} chars"
            )
        else:
            for block_idx, block in enumerate(blocks):
                if len(block.strip()) >= min_block_chars:
                    h = _hash_block(block)
                    if h not in seen_blocks:
                        seen_blocks[h] = (idx, fn_name, block_idx)

    return result


def dedup_responses_api(
    body: dict,
    min_block_chars: int = MIN_BLOCK_CHARS,
    min_content_chars: int = MIN_CONTENT_CHARS,
) -> DedupResult:
    input_items = body.get("input")
    if not isinstance(input_items, list) or not input_items:
        return DedupResult()

    fn_names = _build_tool_name_map_responses(input_items)
    seen_blocks: Dict[str, Tuple[int, str, int]] = {}
    result = DedupResult()

    for idx, item in enumerate(input_items):
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue

        output = item.get("output", "")
        if not isinstance(output, str) or len(output) < min_content_chars:
            continue

        call_id = item.get("call_id", "")
        fn_name = fn_names.get(call_id, call_id) or "tool"

        blocks = _content_defined_chunking(output)
        if len(blocks) < 2:
            for b in blocks:
                if len(b.strip()) >= min_block_chars:
                    h = _hash_block(b)
                    if h not in seen_blocks:
                        seen_blocks[h] = (idx, fn_name, 0)
            continue

        new_blocks = []
        deduped_in_this = 0

        for block_idx, block in enumerate(blocks):
            if len(block.strip()) < min_block_chars:
                new_blocks.append(block)
                continue

            h = _hash_block(block)
            result.blocks_total += 1

            if h in seen_blocks and seen_blocks[h][0] != idx:
                _, orig_fn, _ = seen_blocks[h]
                first_line = block.strip().split("\n")[0][:80]
                ref = f'[... "{first_line}" — identical to earlier {orig_fn} result, see above ...]'
                chars_saved = len(block) - len(ref)
                if chars_saved > 0:
                    new_blocks.append(ref)
                    deduped_in_this += 1
                    result.blocks_deduped += 1
                else:
                    new_blocks.append(block)
            else:
                if h not in seen_blocks:
                    seen_blocks[h] = (idx, fn_name, block_idx)
                new_blocks.append(block)

        if deduped_in_this > 0:
            original_len = len(output)
            new_output = "\n\n".join(new_blocks)
            item["output"] = new_output
            new_len = len(new_output)
            result.chars_before += original_len
            result.chars_after += new_len
            result.chars_saved += original_len - new_len
            logger.info(
                f"Block dedup: input[{idx}] {fn_name} — "
                f"{deduped_in_this}/{len(blocks)} blocks, "
                f"saved {original_len - new_len:,} chars"
            )
        else:
            for block_idx, block in enumerate(blocks):
                if len(block.strip()) >= min_block_chars:
                    h = _hash_block(block)
                    if h not in seen_blocks:
                        seen_blocks[h] = (idx, fn_name, block_idx)

    return result
