"""General claim→evidence provenance-linking dataset and shadow baseline.

This module is deliberately separate from the artifact-dedup canary.  It is the
training/eval substrate for the more general design: build examples containing
raw source blocks plus assistant claims, run a shadow evidence linker, and report
metrics without changing the online LLM payload.

Privacy contract: committed fixtures should be synthetic.  Real trace exports may
contain raw content and must live under a local/gitignored path (for example
``~/contextpilot/provenance_datasets``).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

PROVENANCE_LINKING_SCHEMA_VERSION = 1
MUTABLE_EVIDENCE_TYPES = {"tool_result", "assistant_context", "worker_output", "file_snippet", "web_result"}
CLAIM_BLOCK_TYPES = {"assistant_context", "assistant", "parent_summary"}
SOURCE_BLOCK_TYPES = {"tool_result", "worker_output", "file_snippet", "web_result", "assistant_context"}
ALLOWED_RELATIONS = {
    "copied",
    "supports",
    "supports_candidate",
    "extracted",
    "extracted_support",
    "summarized_support",
    "aggregated_support",
    "contradicts",
    "insufficient",
}

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]{2,}|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")


@dataclass(frozen=True)
class ProvenanceBlock:
    block_id: str
    block_type: str
    text: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ProvenanceClaim:
    claim_id: str
    block_id: str
    text: str
    start: int
    end: int
    claim_type: str = "factual"


@dataclass(frozen=True)
class ProvenanceLink:
    claim_id: str
    evidence_block_id: str
    start: int
    end: int
    relation: str
    confidence: float
    method: str


@dataclass(frozen=True)
class ProvenanceExample:
    example_id: str
    domain: str
    blocks: list[ProvenanceBlock]
    claims: list[ProvenanceClaim]
    gold_links: list[ProvenanceLink] = field(default_factory=list)


def _stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def extract_claims(block: ProvenanceBlock, *, max_claims: int = 12, min_chars: int = 12) -> list[ProvenanceClaim]:
    """Extract candidate factual claims from an assistant/summary block.

    This is intentionally cheap and recall-oriented; it is a shadow data builder,
    not a production semantic claim extractor.  A trained linker can replace this
    later while keeping the dataset schema stable.
    """
    if block.block_type not in CLAIM_BLOCK_TYPES:
        return []
    claims: list[ProvenanceClaim] = []
    pos = 0
    for part in _SENTENCE_SPLIT_RE.split(block.text):
        text = part.strip()
        if not text or len(text) < min_chars:
            pos += len(part) + 1
            continue
        start = block.text.find(text, pos)
        if start < 0:
            start = block.text.find(text)
        end = start + len(text)
        # Prefer factual-looking claims: numbers, paths, result words, or Chinese
        # technical verbs.  Skip pure planning/hedging lines when possible.
        lower = text.lower()
        factual_cues = (
            any(ch.isdigit() for ch in text)
            or any(tok in lower for tok in ["passed", "failed", "error", "warning", "commit", "pr", "test", "source", "evidence"])
            or any(tok in text for tok in ["通过", "失败", "错误", "证据", "来自", "支持", "省", "测试", "结论"])
        )
        if factual_cues:
            cid = _stable_id("claim", block.block_id, str(start), text)
            claims.append(ProvenanceClaim(cid, block.block_id, text, start, end))
            if len(claims) >= max_claims:
                break
        pos = max(end, pos + len(part))
    return claims


def shadow_link_claims(
    example: ProvenanceExample,
    *,
    top_k: int = 1,
    min_overlap: int = 2,
) -> list[ProvenanceLink]:
    """Cheap shadow provenance linker: claim→top-k earlier evidence snippets.

    This baseline never folds payloads.  It creates training/eval candidates using
    exact substring, numeric/path overlap, and token Jaccard signals.  A future
    trained model should replace the scoring function, not the safety contract.
    """
    by_id = {b.block_id: b for b in example.blocks}
    order = {b.block_id: i for i, b in enumerate(example.blocks)}
    links: list[ProvenanceLink] = []
    for claim in example.claims:
        claim_tokens = _tokens(claim.text)
        if not claim_tokens:
            continue
        scored: list[tuple[float, ProvenanceBlock, int, int, str]] = []
        claim_block_order = order.get(claim.block_id, 10**9)
        for block in example.blocks:
            if block.block_id == claim.block_id or block.block_type not in SOURCE_BLOCK_TYPES:
                continue
            if order.get(block.block_id, 10**9) >= claim_block_order:
                continue
            start = block.text.lower().find(claim.text.lower())
            if start >= 0:
                scored.append((1.0, block, start, start + len(claim.text), "copied"))
                continue
            block_tokens = _tokens(block.text)
            overlap = claim_tokens & block_tokens
            if len(overlap) < min_overlap:
                continue
            score = len(overlap) / max(len(claim_tokens), 1)
            # Pick a compact evidence window around the first overlapping token.
            first_positions = [block.text.lower().find(tok) for tok in overlap if block.text.lower().find(tok) >= 0]
            if first_positions:
                center = min(first_positions)
                start = max(0, center - 220)
                end = min(len(block.text), center + 420)
            else:
                start, end = 0, min(len(block.text), 640)
            relation = "extracted" if score >= 0.55 else "supports_candidate"
            scored.append((score, block, start, end, relation))
        scored.sort(key=lambda row: (-row[0], order[row[1].block_id], row[1].block_id))
        for score, block, start, end, relation in scored[:top_k]:
            links.append(
                ProvenanceLink(
                    claim_id=claim.claim_id,
                    evidence_block_id=block.block_id,
                    start=start,
                    end=end,
                    relation=relation,
                    confidence=round(float(score), 4),
                    method="shadow_lexical_v1",
                )
            )
    return links


def example_from_trace_case(case: dict, *, domain: str = "hermes_trace") -> ProvenanceExample:
    """Convert a trace-validation JSONL case into a provenance-linking example."""
    blocks: list[ProvenanceBlock] = []
    claims: list[ProvenanceClaim] = []
    for idx, msg in enumerate(case.get("messages", [])):
        block_type = str(msg.get("block_type") or "unknown")
        text = str(msg.get("content") or "")
        bid = f"b{idx:04d}_{block_type}"
        block = ProvenanceBlock(
            block_id=bid,
            block_type=block_type,
            text=text,
            metadata={"role": msg.get("role") or "", "index": idx},
        )
        blocks.append(block)
        claims.extend(extract_claims(block))
    return ProvenanceExample(
        example_id=str(case.get("case_id") or _stable_id("ex", json.dumps(case, sort_keys=True)[:1000])),
        domain=domain,
        blocks=blocks,
        claims=claims,
    )


def to_jsonable(example: ProvenanceExample, *, shadow_links: list[ProvenanceLink] | None = None) -> dict:
    out = asdict(example)
    out["schema_version"] = PROVENANCE_LINKING_SCHEMA_VERSION
    if shadow_links is not None:
        out["shadow_links"] = [asdict(link) for link in shadow_links]
    return out


def read_jsonl_examples(path: Path) -> list[ProvenanceExample]:
    examples: list[ProvenanceExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        links = [ProvenanceLink(**l) for l in raw.get("gold_links", [])]
        bad_relations = sorted({link.relation for link in links if link.relation not in ALLOWED_RELATIONS})
        if bad_relations:
            raise ValueError(f"unsupported provenance relation(s): {bad_relations}")
        examples.append(
            ProvenanceExample(
                example_id=raw["example_id"],
                domain=raw.get("domain", "unknown"),
                blocks=[ProvenanceBlock(**b) for b in raw.get("blocks", [])],
                claims=[ProvenanceClaim(**c) for c in raw.get("claims", [])],
                gold_links=links,
            )
        )
    return examples


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def evaluate_shadow(examples: Iterable[ProvenanceExample]) -> dict:
    """Evaluate shadow links against examples that include gold links.

    Matching is intentionally evidence-block level for generality: a model can
    learn better span boundaries later, while the early metric asks whether it
    found the right source evidence for each claim.
    """
    tp = fp = fn = 0
    case_rows = []
    predicted_total = gold_total = 0
    for ex in examples:
        pred = shadow_link_claims(ex)
        pred_pairs = {(l.claim_id, l.evidence_block_id) for l in pred}
        gold_pairs = {(l.claim_id, l.evidence_block_id) for l in ex.gold_links}
        case_tp = len(pred_pairs & gold_pairs)
        case_fp = len(pred_pairs - gold_pairs)
        case_fn = len(gold_pairs - pred_pairs)
        tp += case_tp
        fp += case_fp
        fn += case_fn
        predicted_total += len(pred_pairs)
        gold_total += len(gold_pairs)
        case_rows.append(
            {
                "example_id": ex.example_id,
                "claim_count": len(ex.claims),
                "gold_links": len(gold_pairs),
                "predicted_links": len(pred_pairs),
                "tp": case_tp,
                "fp": case_fp,
                "fn": case_fn,
            }
        )
    return {
        "schema_version": PROVENANCE_LINKING_SCHEMA_VERSION,
        "corpus": "provenance_linking_shadow_eval",
        "claim_scope": "shadow claim→evidence linking; does not mutate online context",
        "example_count": len(case_rows),
        "gold_links": gold_total,
        "predicted_links": predicted_total,
        "shadow_link_tp": tp,
        "shadow_link_fp": fp,
        "shadow_link_fn": fn,
        "shadow_link_precision": tp / (tp + fp) if tp + fp else 1.0,
        "shadow_link_recall": tp / (tp + fn) if tp + fn else 1.0,
        "cases": case_rows,
    }
