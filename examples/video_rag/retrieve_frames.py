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

import numpy as np
import torch
from PIL import Image


def load_model(name, device):
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()
    return proc, model


@torch.no_grad()
def embed_images(proc, model, paths, device, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        inputs = proc(images=ims, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def embed_texts(proc, model, texts, device, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        inputs = proc(text=texts[i:i + bs], padding="max_length", truncation=True,
                      max_length=64, return_tensors="pt").to(device)
        feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
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

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                done.add(json.loads(l)["qid"])
            except Exception:
                pass
    fout = open(a.out, "a")
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
            F = embed_images(proc, model, paths, device)
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
            print(f"{vi}/{len(by_vid)} videos, {n_done} questions", flush=True)
    print(f"done: {n_done} questions written")


if __name__ == "__main__":
    main()
