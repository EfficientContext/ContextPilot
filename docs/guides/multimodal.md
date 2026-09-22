# Multimodal contexts (images / video frames)

ContextPilot's Context Index is content-agnostic: it reorders *blocks* so that
blocks shared across requests form a common prefix. Since v0.4.3 a block can be
an **image** — typically a video frame retrieved by a video-RAG pipeline — and
ContextPilot will (1) reorder the frames for KV-cache prefix sharing and (2)
tell the model the true chronological order in one sentence placed *after* the
frames, so the shared prefix stays byte-identical across requests.

```
[system]  (constant)
[user]    intro (constant)
          [label] image  ┐  reorderable prefix — shared frames first
          [label] image  │  (labels are per-frame timestamps, not positions)
          ...            ┘
          "Note: the frames above are NOT shown in chronological order.
           Their true chronological order, from earliest to latest, is: …"
          question
```

## Library API

```python
from contextpilot import ContextPilot
from contextpilot.multimodal import video_frames_to_blocks, optimize_multimodal

pilot = ContextPilot(use_gpu=False)

# One block per frame; key = "<video_id>#<frame_index>", order = timestamp
blocks = video_frames_to_blocks(frame_paths, timestamps, video_id="vid42")

# Your retriever picks frames per question (any order, e.g. by CLIP score)
retrieved = [blocks[i] for i in retriever.top_k(question, k=64)]

messages = optimize_multimodal(retrieved, question, pilot=pilot,
                               order_hint="sentence")   # or labels | both | none
client.chat.completions.create(model="Qwen/Qwen3.8-27B", messages=messages,
                               chat_template_kwargs={"enable_thinking": False})
```

* `ImageBlock.from_path / from_bytes / from_url` build blocks with a
  content-derived key (SHA-256 of the payload) unless you pass `key=`.
* `order` is the block's true position (timestamp or index); `label` is an
  optional short text rendered before the image. Labels must describe the
  frame itself (e.g. `[Frame 12 | t=34.5s]`), never its slot in the prompt.
* `optimize_multimodal_batch(all_blocks, all_queries)` reorders a whole batch
  globally and returns the cache-aware execution order, like `optimize_batch`.
* `reorder=False` gives the chronological baseline with identical scaffolding.

Order-hint modes: `sentence` (one sentence listing the true order, the default),
`labels` (per-frame timestamp labels only), `both`, `none` (ablation).

## Proxy (intercept) mode

When the HTTP server proxies `/v1/chat/completions`, a user message whose
content contains two or more `image_url` parts is treated as a frame sequence:
text parts before the first image are the intro, a text part immediately before
each image is its label, and text after the last image is the question. The
images are reordered with the same persistent index used for text documents
(keys are hashes of the `image_url` value) and the order hint is prepended to
the question.

Headers:

| Header | Values | Default |
|---|---|---|
| `X-ContextPilot-Multimodal` | `auto`, `off` | `auto` |
| `X-ContextPilot-MM-Order-Hint` | `sentence`, `labels`, `both`, `none` | `sentence` |

The true order is parsed from labels (`t=12.5s` or `Frame 3`) when every image
has one; otherwise the incoming display order is assumed chronological.

## Engine notes (SGLang)

* SGLang hashes image content into the radix-cache key, so identical frames in
  an identical prefix hit the KV cache **and** skip the vision encoder.
* Hybrid linear-attention models (Qwen3.5 / Qwen3.8 / GLM-5.3-Flash) only hit
  at 64-token-aligned *Mamba checkpoints*, which exist at previous prompt ends
  and at branch points recorded after a miss. A shorter shared prefix therefore
  misses the first time it is seen and hits from the second time on. Full
  attention models do not have this restriction.
* Reordering frames changes the token sequence, so a frame moved to a new
  position is a miss — that is exactly what the reorder avoids across requests.

See `examples/video_rag/` for a complete benchmark (frame extraction, SigLIP
retrieval, and the reorder-vs-chronological accuracy/TTFT study).
