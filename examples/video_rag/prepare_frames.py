#!/usr/bin/env python3
"""
Extract a fixed-rate frame pool from each video (video-RAG corpus prep).

For every video listed in a manifest (jsonl with ``video_id`` and ``video_path``)
this writes ``<out>/<video_id>/NNNNN.jpg`` plus ``<out>/<video_id>/frames.json``
holding the frame timestamps.  Frames are sampled at ``--fps`` and capped at
``--max-frames`` (uniform subsampling), resized so the width is ``--width``.

Usage:
    python prepare_frames.py --manifest videos.jsonl --out /data/frames \
        --fps 1 --max-frames 256 --width 448 --workers 8
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def _seek_frame(path: str, ts: float, out_path: str, width: int, quality: int) -> bool:
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-ss", f"{ts:.3f}", "-i", path,
        "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", str(quality), out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(out_path)


def extract(video_id: str, path: str, out_dir: str, fps: float, max_frames: int, width: int, quality: int = 4):
    vdir = os.path.join(out_dir, video_id)
    done = os.path.join(vdir, "frames.json")
    if os.path.exists(done):
        return video_id, "skip", None
    os.makedirs(vdir, exist_ok=True)
    dur = probe_duration(path)
    if dur <= 0:
        return video_id, "error", "ffprobe failed"
    n_at_fps = int(dur * fps)
    if n_at_fps > max_frames:
        # Long video: uniform sampling via input seeking (decodes only near keyframes)
        stamps = [(i + 0.5) * dur / max_frames for i in range(max_frames)]
        files, kept = [], []
        for i, ts in enumerate(stamps):
            name = f"{i:05d}.jpg"
            if _seek_frame(path, ts, os.path.join(vdir, name), width, quality):
                files.append(name)
                kept.append(ts)
        if not files:
            return video_id, "error", "seek extraction produced no frames"
        stamps = kept
        eff_fps = len(files) / dur
    else:
        # Short video: single decode pass at the requested fps
        n = max(1, n_at_fps)
        eff_fps = n / dur
        tmp_pattern = os.path.join(vdir, "%05d.jpg")
        cmd = [
            "ffmpeg", "-v", "error", "-y", "-i", path,
            "-vf", f"fps={eff_fps:.6f},scale={width}:-2",
            "-q:v", str(quality), "-frames:v", str(n), "-start_number", "0", tmp_pattern,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return video_id, "error", r.stderr[-300:]
        files = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg"))
        stamps = [k / eff_fps for k in range(len(files))]
    with open(done, "w") as f:
        json.dump({"video_id": video_id, "duration": dur, "fps": eff_fps,
                   "frames": files, "timestamps": stamps}, f)
    return video_id, "ok", len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=256)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    vids = [json.loads(l) for l in open(a.manifest) if l.strip()]
    if a.limit:
        vids = vids[: a.limit]
    os.makedirs(a.out, exist_ok=True)
    ok = err = skip = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(extract, v["video_id"], v["video_path"], a.out, a.fps, a.max_frames, a.width) for v in vids]
        for i, fut in enumerate(as_completed(futs), 1):
            vid, status, info = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"[error] {vid}: {info}", file=sys.stderr)
            if i % 20 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} ok={ok} skip={skip} err={err}", flush=True)


if __name__ == "__main__":
    main()
