#!/usr/bin/env python3
"""
Diagnose *why* (and whether) reordering retrieved frames produces KV-cache
hits on a given engine + model.

Part 1 (offline, no server): for every pair of questions on the same video,
how many leading frames do the two prompts share under each ordering?
This is the upper bound on prefix reuse, independent of the engine.

Part 2 (online): replay a small set of requests and report the engine's
``cached_tokens`` per request, so the theoretical overlap can be compared
with what the engine actually reuses. On hybrid linear-attention models the
two differ: the KV prefix only counts up to the last recorded Mamba
checkpoint, which exists at a previous request's end or at a branch point
recorded by an earlier miss.

    python probe_cache.py --questions q.jsonl --retrieval r.jsonl \
        --frames frames/ --k 16 --api http://host:8000 --model M
"""

import argparse
import contextlib
import io
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

import requests  # noqa: E402

from run_video_rag_bench import (  # noqa: E402
    CONDITIONS,
    all_frame_paths,
    load_jsonl,
    make_requests,
    warm_frame_cache,
)


def shared_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def offline(conditions, questions, retrieval, frames, k, seed, budget=True):
    print("\n== Offline: leading frames shared between question pairs of the same video ==\n")
    print(f"{'condition':<18}{'pairs':>7}{'mean':>8}{'median':>8}{'max':>6}{'zero%':>8}{'shared-set':>12}")
    rows = {}
    for cond in conditions:
        with contextlib.redirect_stdout(io.StringIO()):
            reqs = make_requests(cond, questions, retrieval, frames, k, seed, {}, budget=budget)
        by_vid = defaultdict(list)
        for r in reqs:
            by_vid[r["video_id"]].append(r["display_frames"])
        pref, inter = [], []
        for seqs in by_vid.values():
            for i in range(len(seqs)):
                for j in range(i + 1, len(seqs)):
                    pref.append(shared_prefix(seqs[i], seqs[j]))
                    inter.append(len(set(seqs[i]) & set(seqs[j])))
        if not pref:
            continue
        pref_sorted = sorted(pref)
        rows[cond] = pref
        print(f"{cond:<18}{len(pref):>7}{sum(pref) / len(pref):>8.2f}"
              f"{pref_sorted[len(pref) // 2]:>8}{max(pref):>6}"
              f"{100 * sum(1 for p in pref if p == 0) / len(pref):>7.1f}%"
              f"{sum(inter) / len(inter):>12.2f}")
    return rows


def online(api, model, conditions, questions, retrieval, frames, k, seed, max_tokens, budget=True):
    print("\n== Online: engine cached_tokens per request (concurrency 1) ==\n")
    for cond in conditions:
        with contextlib.redirect_stdout(io.StringIO()):
            reqs = make_requests(cond, questions, retrieval, frames, k, seed, {}, budget=budget)
        requests.post(f"{api}/flush_cache", timeout=60)
        tot_p = tot_c = 0
        per_req = []
        for r in reqs:
            body = {"model": model, "messages": r["messages"], "max_tokens": max_tokens,
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            resp = requests.post(f"{api}/v1/chat/completions", json=body, timeout=600).json()
            u = resp["usage"]
            det = u.get("prompt_tokens_details") or {}
            c = det.get("cached_tokens", u.get("cached_tokens")) or 0
            tot_p += u["prompt_tokens"]
            tot_c += c
            per_req.append((r["video_id"], u["prompt_tokens"], c))
        n = max(1, len(reqs))
        print(f"{cond:<20} hit={tot_c / tot_p:.3f}  cached={tot_c}/{tot_p}  "
              f"prompt/req={tot_p / n:.0f}  uncached/req={(tot_p - tot_c) / n:.0f}")
        for vid, p, c in per_req:
            print(f"    {vid[:14]:<16} prompt={p:>6} cached={c:>6}")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conditions", default="chrono_plain,cp_sentence,canon_sentence,shuffle_sentence")
    ap.add_argument("--no-budget", action="store_true")
    ap.add_argument("--api", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--max-tokens", type=int, default=8)
    a = ap.parse_args()

    questions = load_jsonl(a.questions)
    retrieval = {r["qid"]: r for r in load_jsonl(a.retrieval)}
    questions = [q for q in questions if q["qid"] in retrieval]
    if a.limit:
        questions = questions[: a.limit]
    conds = [c.strip() for c in a.conditions.split(",") if c.strip() in CONDITIONS]
    print(f"{len(questions)} questions, k={a.k}")
    warm_frame_cache(all_frame_paths(questions, retrieval, a.frames, a.k, {}))
    offline(conds, questions, retrieval, a.frames, a.k, a.seed, budget=not a.no_budget)
    if a.api and a.model:
        online(a.api, a.model, conds, questions, retrieval, a.frames, a.k, a.seed, a.max_tokens, budget=not a.no_budget)


if __name__ == "__main__":
    main()
