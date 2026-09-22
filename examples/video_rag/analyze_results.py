#!/usr/bin/env python3
"""
Aggregate video-RAG benchmark runs into markdown tables.

    python analyze_results.py /mnt/cp/results/*/ --questions-dir /mnt/cp/data

For every run directory (containing summary.json + <condition>.jsonl) prints
accuracy, TTFT (mean / p50), cache-hit ratio, uncached prefill tokens and wall
time per condition, plus accuracy broken down by task type (to expose
order-sensitive question types) and a paired comparison of each condition
against chrono_plain (McNemar-style counts: wins / losses on the same
questions).
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def analyze_run(run_dir, questions_by_ds):
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    cfg = summary.get("_config", {})
    name = os.path.basename(os.path.normpath(run_dir))
    conds = [c for c in summary if not c.startswith("_")]
    print(f"\n## {name}\n")
    print(f"model={cfg.get('model')} k={cfg.get('k')} n={cfg.get('n_questions')} "
          f"concurrency={cfg.get('concurrency')} questions={os.path.basename(str(cfg.get('questions')))}\n")
    print("| condition | acc | ttft mean (s) | ttft p50 | latency mean | cache hit | uncached tok/req | wall (s) | errors |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        s = summary[c]
        pt, ct, n = s["prompt_tokens_total"], s["cached_tokens_total"], max(1, s["n"] - s["errors"])
        print(f"| {c} | {s['accuracy']:.4f} | {fmt(s['ttft_mean'])} | {fmt(s['ttft_p50'])} | {fmt(s['latency_mean'])} | "
              f"{fmt(s['cache_hit_ratio'])} | {(pt - ct) / n:.0f} | {s['wall_s']:.0f} | {s['errors']} |")

    # per-question results
    per = {c: {r["qid"]: r for r in load_jsonl(os.path.join(run_dir, f"{c}.jsonl"))} for c in conds
           if os.path.exists(os.path.join(run_dir, f"{c}.jsonl"))}
    base = "chrono_plain" if "chrono_plain" in per else (conds[0] if conds else None)
    if base and len(per) > 1:
        print(f"\nPaired vs {base} (same questions):\n")
        print("| condition | both right | only base right | only cond right | both wrong | delta acc |")
        print("|---|---|---|---|---|---|")
        for c in conds:
            if c == base or c not in per:
                continue
            bb = bo = ob = ww = 0
            for qid, rb in per[base].items():
                rc = per[c].get(qid)
                if rc is None:
                    continue
                a, b = bool(rb["correct"]), bool(rc["correct"])
                if a and b: bb += 1
                elif a: bo += 1
                elif b: ob += 1
                else: ww += 1
            n = bb + bo + ob + ww
            print(f"| {c} | {bb} | {bo} | {ob} | {ww} | {(ob - bo) / n:+.4f} |" if n else f"| {c} | - | - | - | - | - |")

    # breakdown by task type
    qs = questions_by_ds.get(os.path.basename(str(cfg.get("questions"))), {})
    if qs:
        types = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for c, rows in per.items():
            for qid, r in rows.items():
                q = qs.get(qid)
                if not q:
                    continue
                t = q.get("task_type") or q.get("domain") or "?"
                types[t][c][0] += int(bool(r["correct"]))
                types[t][c][1] += 1
        print("\nAccuracy by task type:\n")
        print("| task type | n | " + " | ".join(conds) + " |")
        print("|---|---|" + "---|" * len(conds))
        for t in sorted(types, key=lambda t: -sum(v[1] for v in types[t].values())):
            n = max(v[1] for v in types[t].values())
            cells = [f"{types[t][c][0] / types[t][c][1]:.3f}" if types[t][c][1] else "-" for c in conds]
            print(f"| {t} | {n} | " + " | ".join(cells) + " |")

    # frame-overlap statistics from display_frames
    if base in per:
        rows = list(per[base].values())
        by_vid = defaultdict(list)
        for r in rows:
            by_vid[r["video_id"]].append(set(r["display_frames"]))
        pair_j = []
        for sets in by_vid.values():
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    inter = len(sets[i] & sets[j]); uni = len(sets[i] | sets[j])
                    pair_j.append(inter / uni if uni else 0)
        if pair_j:
            print(f"\nRetrieved-frame overlap between questions of the same video: "
                  f"mean Jaccard {sum(pair_j) / len(pair_j):.3f} over {len(pair_j)} pairs, "
                  f"{len(by_vid)} videos, {len(rows) / len(by_vid):.1f} questions/video")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--questions-dir", default=None)
    a = ap.parse_args()
    qmap = {}
    if a.questions_dir:
        for p in glob.glob(os.path.join(a.questions_dir, "*", "questions*.jsonl")):
            qmap[os.path.basename(p)] = {q["qid"]: q for q in load_jsonl(p)}
    for run in a.runs:
        if os.path.exists(os.path.join(run, "summary.json")):
            analyze_run(run, qmap)


if __name__ == "__main__":
    main()
