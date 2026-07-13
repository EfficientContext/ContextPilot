"""Worker Context Routing — SHADOW MODE (P0 data collection only).

Classifies LLM-bound blocks with a conservative, deterministic heuristic and
emits aggregate counters + salted hashes so the labels can be evaluated offline.
Nothing here ever drops, summarizes, replaces, or mutates context: it is pure
reporting/measurement. ``est_candidate_tokens`` is an ADVISORY upper bound on
what a *future* router might route away -- never a realized saving.
"""
from __future__ import annotations

from typing import Iterable

from .models import (
    RouterCandidateBlock,
    RouterLabelCount,
    RouterReasonCount,
    WorkerRoutingShadow,
    _est_tokens,
    _LLMContent,
)
from .privacy import _salted_hash

# Low-cardinality router labels. These are the *training/eval* labels a future
# small worker-context router would predict. P0 is data-collection only: nothing
# here ever drops, summarizes, or mutates context -- it only classifies blocks
# and emits aggregate counters + salted hashes so the labels can be evaluated
# offline before any online pruning is built.
ROUTER_LABELS = (
    "policy_must_keep",        # never droppable (user/system/skill/safety constraints)
    "direct_task_hint",        # short actionable task signal -- keep
    "likely_relevant",         # default keep; not obviously prunable
    "summarizable_candidate",  # large single block that *might* be summarized later
    "likely_drop_candidate",   # large/repeated tool-like block, candidate to route away
)

# Labels whose blocks a future router might safely route away. Used only to
# tally *advisory* candidate tokens; P0 never acts on them.
_ROUTABLE_LABELS = ("summarizable_candidate", "likely_drop_candidate")

# Block-type priority when one fingerprint spans multiple origins: the most
# "must-keep" origin wins, so cross-origin blocks are classified conservatively.
_TYPE_KEEP_PRIORITY = {
    "user_prompt": 5,
    "system_prompt": 4,
    "skill_prompt": 4,
    "assistant_context": 2,
    "tool_result": 1,
    "unknown": 0,
}

# Cues marking content that must NEVER be dropped even from a tool/assistant
# block: explicit safety / acceptance / hard-constraint language. Matching here
# is intentionally generous -- over-keeping is the safe direction for P0.
_SAFETY_CONSTRAINT_CUES = (
    "must not",
    "must never",
    "never drop",
    "do not delete",
    "do not remove",
    "do not modify",
    "acceptance criteria",
    "acceptance test",
    "safety",
    "must keep",
    "you must",
    "required:",
    "constraint",
    "forbidden",
    "policy",
)

# Cues marking a short, actionable task hint worth keeping verbatim.
_TASK_HINT_CUES = (
    "todo",
    "next step",
    "error:",
    "traceback",
    "failed",
    "fixme",
    "task:",
    "goal:",
    "implement",
    "reproduce",
)


def classify_router_label(
    block_type: str,
    content: str,
    *,
    occurrences: int,
    large_output_chars: int,
    min_repeat: int,
) -> tuple[str, str]:
    """Heuristically assign a worker-routing label + reason code to a block.

    Pure P0 heuristic: no ML, no network, no mutation. Operates on in-memory
    text only and returns two low-cardinality enums (``route_label``,
    ``reason_code``) -- never the text. The bias is deliberately conservative:
    when in doubt, keep. Anything that is a user prompt, a system/skill prompt,
    or carries explicit safety/acceptance-constraint language is pinned to
    ``policy_must_keep`` and can never become a routable candidate.
    """
    low = content.lower()

    # 1. Never-drop by origin: prompts the user/system/skills authored.
    if block_type == "user_prompt":
        return "policy_must_keep", "user_prompt_never_drop"
    if block_type in ("system_prompt", "skill_prompt"):
        return "policy_must_keep", "system_or_skill_constraint_never_drop"

    # 2. Never-drop by content: explicit safety / acceptance / hard constraints,
    #    even inside an assistant or tool block.
    if any(cue in low for cue in _SAFETY_CONSTRAINT_CUES):
        return "policy_must_keep", "safety_or_acceptance_constraint"

    char_len = len(content)
    has_task_hint = any(cue in low for cue in _TASK_HINT_CUES)

    # 3. Short actionable task hints -> keep verbatim. Very large diagnostic
    #    logs often contain "error:"/"failed"/"traceback"; keep collecting
    #    them as summarization candidates instead of pinning the whole log.
    if has_task_hint and char_len < large_output_chars:
        return "direct_task_hint", "actionable_task_signal"

    # 4. Bulky / repeated tool-like material -> routable candidates (advisory).
    if block_type in ("tool_result", "assistant_context", "unknown"):
        if has_task_hint and char_len >= large_output_chars:
            return "summarizable_candidate", "large_actionable_tool_block"
        is_large = char_len >= large_output_chars
        is_repeated = occurrences >= min_repeat
        if is_large and is_repeated:
            return "likely_drop_candidate", "large_repeated_tool_block"
        if is_repeated:
            return "likely_drop_candidate", "repeated_tool_block"
        if is_large:
            return "summarizable_candidate", "large_single_tool_block"

    # 5. Everything else: keep by default.
    return "likely_relevant", "default_keep"


def analyze_worker_routing_shadow(
    contents: Iterable[_LLMContent],
    *,
    salt: str,
    large_output_chars: int,
    min_repeat: int,
    top_n: int,
    enabled: bool = True,
) -> WorkerRoutingShadow:
    """Shadow-mode worker-context routing classifier (P0: data collection only).

    Fingerprints each LLM-bound item, assigns a conservative router label, and
    returns aggregate counters + salted hashes for routable candidates. Emits
    NO raw text and never mutates/drops context. ``est_candidate_tokens`` is an
    advisory upper bound on what a *future* router might route away -- not a
    realized saving.
    """
    if not enabled:
        return WorkerRoutingShadow(
            enabled=False,
            item_count=0,
            classified_block_count=0,
            total_occurrences=0,
            must_keep_block_count=0,
            must_keep_occurrence_count=0,
            est_must_keep_tokens=0,
            est_candidate_tokens_total=0,
            est_drop_candidate_tokens=0,
            est_summarizable_candidate_tokens=0,
            label_counts=[],
            reason_counts=[],
            top_candidate_blocks=[],
            notes=["worker-routing shadow analysis disabled via flag"],
        )

    # Aggregate occurrences per fingerprint, picking the most must-keep origin
    # when one block spans several block types.
    agg: dict[str, dict] = {}
    item_count = 0
    for item in contents:
        content = item.content
        if not content:
            continue
        item_count += 1
        h = _salted_hash(content, salt)
        bt = item.block_type
        entry = agg.get(h)
        if entry is None:
            agg[h] = {
                "block_type": bt,
                "char_length": len(content),
                "occurrences": 1,
                "content": content,
            }
        else:
            entry["occurrences"] += 1
            cur = entry["block_type"]
            bt_pri = _TYPE_KEEP_PRIORITY.get(bt, 0)
            cur_pri = _TYPE_KEEP_PRIORITY.get(cur, 0)
            if bt_pri > cur_pri or (bt_pri == cur_pri and bt < cur):
                entry["block_type"] = bt

    # Classify each unique fingerprint and roll up counters.
    label_agg: dict[str, dict] = {}
    reason_agg: dict[tuple[str, str, str], dict] = {}
    candidates: list[RouterCandidateBlock] = []
    must_keep_blocks = 0
    must_keep_occ = 0
    est_must_keep_tokens = 0
    drop_tokens = 0
    summ_tokens = 0

    for h, entry in agg.items():
        bt = entry["block_type"]
        occ = entry["occurrences"]
        char_len = entry["char_length"]
        est = _est_tokens(char_len)
        total_est = est * occ
        label, reason = classify_router_label(
            bt,
            entry["content"],
            occurrences=occ,
            large_output_chars=large_output_chars,
            min_repeat=min_repeat,
        )
        candidate_tokens = total_est if label in _ROUTABLE_LABELS else 0

        la = label_agg.setdefault(
            label,
            {"block_count": 0, "occ": 0, "total_est": 0, "candidate": 0},
        )
        la["block_count"] += 1
        la["occ"] += occ
        la["total_est"] += total_est
        la["candidate"] += candidate_tokens

        ra = reason_agg.setdefault(
            (bt, label, reason),
            {"block_count": 0, "occ": 0, "total_est": 0, "candidate": 0},
        )
        ra["block_count"] += 1
        ra["occ"] += occ
        ra["total_est"] += total_est
        ra["candidate"] += candidate_tokens

        if label == "policy_must_keep":
            must_keep_blocks += 1
            must_keep_occ += occ
            est_must_keep_tokens += total_est
        if label == "likely_drop_candidate":
            drop_tokens += candidate_tokens
        elif label == "summarizable_candidate":
            summ_tokens += candidate_tokens

        if candidate_tokens > 0:
            candidates.append(
                RouterCandidateBlock(
                    block_hash=h,
                    block_type=bt,
                    route_label=label,
                    reason_code=reason,
                    occurrences=occ,
                    char_length=char_len,
                    est_tokens=est,
                    est_candidate_tokens=candidate_tokens,
                )
            )

    # Deterministic ordering: label_counts follow the canonical label order;
    # reason_counts and candidates sort by a stable key.
    label_counts = [
        RouterLabelCount(
            route_label=lbl,
            block_count=label_agg[lbl]["block_count"],
            occurrence_count=label_agg[lbl]["occ"],
            total_est_tokens=label_agg[lbl]["total_est"],
            est_candidate_tokens=label_agg[lbl]["candidate"],
        )
        for lbl in ROUTER_LABELS
        if lbl in label_agg
    ]
    reason_counts = [
        RouterReasonCount(
            block_type=bt,
            route_label=lbl,
            reason_code=reason,
            block_count=v["block_count"],
            occurrence_count=v["occ"],
            total_est_tokens=v["total_est"],
            est_candidate_tokens=v["candidate"],
        )
        for (bt, lbl, reason), v in sorted(reason_agg.items())
    ]
    candidates.sort(
        key=lambda c: (c.est_candidate_tokens, c.occurrences, c.block_hash),
        reverse=True,
    )

    total_occ = sum(e["occurrences"] for e in agg.values())
    notes = [
        "SHADOW MODE P0: classification only -- no context was dropped, summarized, or mutated",
        "route_label/reason_code/block_type are low-cardinality enums; block_hash is a salted SHA-256 fingerprint",
        "est_candidate_tokens is ADVISORY (an upper bound for a FUTURE router), not a realized saving",
        "user/system/skill prompts and safety/acceptance constraints are pinned to policy_must_keep and never routable",
        "classification is conservative: when uncertain, blocks are kept (likely_relevant)",
    ]

    return WorkerRoutingShadow(
        enabled=True,
        item_count=item_count,
        classified_block_count=len(agg),
        total_occurrences=total_occ,
        must_keep_block_count=must_keep_blocks,
        must_keep_occurrence_count=must_keep_occ,
        est_must_keep_tokens=est_must_keep_tokens,
        est_candidate_tokens_total=drop_tokens + summ_tokens,
        est_drop_candidate_tokens=drop_tokens,
        est_summarizable_candidate_tokens=summ_tokens,
        label_counts=label_counts,
        reason_counts=reason_counts,
        top_candidate_blocks=candidates[:top_n],
        notes=notes,
    )
