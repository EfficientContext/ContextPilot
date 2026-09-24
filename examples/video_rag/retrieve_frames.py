#!/usr/bin/env python3
"""
Frame retrieval for video RAG: score every frame of a question's video
against the question text with a CLIP/SigLIP dual encoder and keep the
top-k.  Output is one jsonl row per question:

    {"qid": ..., "video_id": ..., "topk": [frame_idx, ...], "scores": [...]}

``frame_idx`` indexes into ``<frames>/<video_id>/frames.json``.

Usage:
    python retrieve_frames.py --questions q.jsonl --frames /data/frames \
        --out retrieval.jsonl --model google/siglip2-base-patch16-256 --k 64
"""

import argparse
import json
import os
import sys
import time

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image


def load_model(name, device, dtype=torch.float16):
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, dtype=dtype if device == "cuda" else torch.float32)
    return proc, model.to(device).eval()


def _load_image(path):
    return Image.open(path).convert("RGB")


def _as_tensor(feats):
    """transformers >=5 returns a ModelOutput from get_*_features; unwrap it."""
    if isinstance(feats, torch.Tensor):
        return feats
    for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        v = getattr(feats, attr, None)
        if isinstance(v, torch.Tensor):
            return v.mean(1) if attr == "last_hidden_state" and v.dim() == 3 else v
    if isinstance(feats, (tuple, list)) and isinstance(feats[0], torch.Tensor):
        return feats[0]
    raise TypeError(f"cannot extract embedding tensor from {type(feats)}")


@torch.no_grad()
def embed_images(proc, model, paths, device, bs=64, io_workers=16):
    """Embed frames; JPEG decode runs in a thread pool (it dominates on CephFS)."""
    out = []
    pool = ThreadPoolExecutor(io_workers) if io_workers > 1 else None
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        ims = list(pool.map(_load_image, chunk)) if pool else [_load_image(p) for p in chunk]
        inputs = proc(images=ims, return_tensors="pt").to(device)
        if device == "cuda":
            inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}
        feats = _as_tensor(model.get_image_features(**inputs))
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def embed_texts(proc, model, texts, device, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        inputs = proc(text=texts[i:i + bs], padding="max_length", truncation=True,
                      max_length=64, return_tensors="pt").to(device)
        feats = _as_tensor(model.get_text_features(**inputs))
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="jsonl with qid, video_id, question, options")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/siglip2-base-patch16-256")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--cache", default=None, help="dir to cache per-video frame embeddings")
    ap.add_argument("--use-options", action="store_true", help="append options to the query text")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--io-workers", type=int, default=16)
    ap.add_argument("--shard", default="0/1", help="i/n — process only videos with index %% n == i")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    proc, model = load_model(a.model, device)
    cache = a.cache or os.path.join(a.frames, "_emb_" + a.model.replace("/", "_"))
    os.makedirs(cache, exist_ok=True)

    qs = [json.loads(l) for l in open(a.questions) if l.strip()]
    by_vid = {}
    for q in qs:
        by_vid.setdefault(q["video_id"], []).append(q)
    si, sn = (int(x) for x in a.shard.split("/"))
    if sn > 1:
        by_vid = {v: q for i, (v, q) in enumerate(sorted(by_vid.items())) if i % sn == si}
        print(f"shard {si}/{sn}: {len(by_vid)} videos")

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                done.add(json.loads(l)["qid"])
            except Exception:
                pass
    fout = open(a.out, "a")
    t_start = time.time()
    n_done = 0
    for vi, (vid, vqs) in enumerate(by_vid.items(), 1):
        if all(q["qid"] in done for q in vqs):
            continue
        meta_p = os.path.join(a.frames, vid, "frames.json")
        if not os.path.exists(meta_p):
            print(f"[skip] no frames for {vid}", file=sys.stderr)
            continue
        meta = json.load(open(meta_p))
        emb_p = os.path.join(cache, f"{vid}.npy")
        if os.path.exists(emb_p):
            F = np.load(emb_p)
        else:
            paths = [os.path.join(a.frames, vid, f) for f in meta["frames"]]
            F = embed_images(proc, model, paths, device, bs=a.batch_size, io_workers=a.io_workers)
            np.save(emb_p, F)
        texts = []
        for q in vqs:
            t = q["question"]
            if a.use_options and q.get("options"):
                t = t + " " + " ".join(q["options"])
            texts.append(t)
        T = embed_texts(proc, model, texts, device)
        S = T @ F.T  # [nq, nframes]
        for q, s in zip(vqs, S):
            if q["qid"] in done:
                continue
            k = min(a.k, len(s))
            top = np.argsort(-s)[:k]
            fout.write(json.dumps({
                "qid": q["qid"], "video_id": vid,
                "topk": [int(i) for i in top],
                "scores": [float(s[i]) for i in top],
                "n_frames": int(len(s)),
            }) + "\n")
            n_done += 1
        fout.flush()
        if vi % 10 == 0:
            el = time.time() - t_start
            print(f"{vi}/{len(by_vid)} videos, {n_done} questions, "
                  f"{el / max(1, vi):.1f}s/video", flush=True)
    print(f"done: {n_done} questions written")


if __name__ == "__main__":
    main()
