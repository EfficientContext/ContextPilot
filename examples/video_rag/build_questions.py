#!/usr/bin/env python3
"""
Build the unified question file + video manifest used by the video-RAG
benchmark from a benchmark's native annotation files.

Unified question row (jsonl):
    {"qid", "video_id", "question", "options": ["A. ...", ...], "answer": "B",
     "duration": <s or bucket>, "task_type": ...}
Video manifest row (jsonl):
    {"video_id", "video_path"}

Supported datasets:
    videomme   lmms-eval/Video-MME (parquet + videos/<videoID>.mp4)
    egoschema  lmms-eval/egoschema (parquet + videos/<video_idx>.mp4)

Usage:
    python build_questions.py videomme --root /mnt/cp/data/videomme \
        --out-questions q.jsonl --out-manifest videos.jsonl [--duration long]
"""

import argparse
import glob
import json
import os
import sys


def _find_video(videos_dir, stem):
    for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
        p = os.path.join(videos_dir, stem + ext)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(videos_dir, "**", stem + ".*"), recursive=True)
    return hits[0] if hits else None


def videomme(root, durations):
    import pandas as pd
    pq = sorted(glob.glob(os.path.join(root, "raw", "**", "*.parquet"), recursive=True))
    if not pq:
        sys.exit(f"no parquet under {root}/raw")
    df = pd.concat([pd.read_parquet(p) for p in pq], ignore_index=True)
    videos_dir = os.path.join(root, "videos")
    qs, vids, missing = [], {}, set()
    for _, r in df.iterrows():
        if durations and str(r["duration"]) not in durations:
            continue
        vid = str(r["videoID"])
        if vid not in vids:
            p = _find_video(videos_dir, vid)
            if p is None:
                missing.add(vid)
                continue
            vids[vid] = p
        opts = list(r["options"])
        qs.append({
            "qid": str(r["question_id"]), "video_id": vid,
            "question": str(r["question"]), "options": [str(o) for o in opts],
            "answer": str(r["answer"]).strip(), "duration": str(r["duration"]),
            "task_type": str(r.get("task_type", "")), "domain": str(r.get("domain", "")),
        })
    print(f"videomme: {len(qs)} questions, {len(vids)} videos, {len(missing)} videos missing", file=sys.stderr)
    return qs, vids


def egoschema(root, durations):
    import pandas as pd
    pq = sorted(glob.glob(os.path.join(root, "raw", "**", "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(p) for p in pq], ignore_index=True)
    videos_dir = os.path.join(root, "videos")
    qs, vids, missing = [], {}, set()
    for _, r in df.iterrows():
        vid = str(r["video_idx"])
        if vid not in vids:
            p = _find_video(videos_dir, vid)
            if p is None:
                missing.add(vid)
                continue
            vids[vid] = p
        opts = list(r["option"])
        opts = [o if o[:2] in ("A.", "B.", "C.", "D.", "E.") else f"{chr(65 + i)}. {o}" for i, o in enumerate(opts)]
        ans = r.get("answer")
        if ans is None or (isinstance(ans, float)):
            continue
        ans = str(ans).strip()
        if ans.isdigit():
            ans = chr(65 + int(ans))
        qs.append({
            "qid": str(r["question_idx"]), "video_id": vid,
            "question": str(r["question"]), "options": opts, "answer": ans,
            "duration": "180", "task_type": "egoschema",
        })
    print(f"egoschema: {len(qs)} questions, {len(vids)} videos, {len(missing)} missing", file=sys.stderr)
    return qs, vids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["videomme", "egoschema"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-questions", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--duration", nargs="*", default=None, help="videomme: short/medium/long filter")
    a = ap.parse_args()
    fn = {"videomme": videomme, "egoschema": egoschema}[a.dataset]
    qs, vids = fn(a.root, a.duration)
    with open(a.out_questions, "w") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(a.out_manifest, "w") as f:
        for vid, p in vids.items():
            f.write(json.dumps({"video_id": vid, "video_path": p}) + "\n")


if __name__ == "__main__":
    main()
