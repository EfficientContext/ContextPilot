# Video RAG: does reordering frames for cache reuse keep accuracy?

Setting: long-video question answering where each question retrieves a subset
of a video's frames. Reordering those frames so that frames shared with earlier
questions form a common prefix raises the engine's prefix-cache hit rate, but
it also destroys the chronological order the model would normally rely on. The
usual fix is to state the true order in one sentence after the frames, where it
does not break the shared prefix.

This page reports what that actually costs and buys, measured end to end.

**Reproduce**: `examples/video_rag/` (frame extraction, SigLIP-2 retrieval,
benchmark runner, analysis). Results below are raw runs, not tuned.

## Setup

| | |
|---|---|
| Benchmark | Video-MME, all three duration splits, 3 questions per video |
| Retrieval | SigLIP-2 base (`google/siglip2-base-patch16-256`) over a 1 fps frame pool (≤256 frames/video, 448 px), top-16 frames per question |
| Engine | SGLang v0.5.20, stock image, `--enable-cache-report` |
| Models | Qwen3.5-4B (1×H100), Qwen3.8-27B (2×H100, tp 2) |
| Prompt | system + intro + frames + question, answer forced with "The best answer is:" |
| Requests | concurrency 4, engine cache flushed before every condition |

Conditions differ only in frame order and in how the true order is conveyed:

| condition | frame order | order information |
|---|---|---|
| `chrono_plain` | chronological | none (baseline) |
| `chrono_labels` | chronological | per-frame `[Frame i \| t=..s]` labels |
| `cp_sentence` | ContextPilot reorder | one sentence listing the true order |
| `canon_*` | canonical per-video core prefix | as the suffix says |
| `shuffle_*` | random | as the suffix says |

## Result 1 — frame order is nearly free, the order sentence is not

Qwen3.5-4B, 285 questions, paired with the `chrono_plain` baseline on the same
questions (McNemar exact test):

| condition | accuracy | Δ | lost / gained | p | cache hit |
|---|---|---|---|---|---|
| `chrono_labels` | 0.5789 | +0.011 | 13 / 16 | 0.71 | 0.007 |
| `chrono_plain` | 0.5684 | — | — | — | 0.006 |
| `shuffle_none` | 0.5579 | −0.011 | 9 / 6 | 0.61 | 0.000 |
| `canon_none` | 0.5544 | −0.014 | 12 / 8 | 0.50 | **0.108** |
| `canon_sentence` | 0.4807 | −0.088 | 37 / 12 | 4.7e−4 | **0.108** |
| `shuffle_sentence` | 0.4807 | −0.088 | 34 / 9 | 1.7e−4 | 0.000 |
| `cp_sentence` | 0.4596 | −0.109 | 39 / 8 | 5.5e−6 | 0.012 |

Reordering frames without saying anything about it costs nothing measurable
(`canon_none`, `shuffle_none`: p ≥ 0.5). Adding the sentence that states the
true chronological order costs 9–11 points, and it costs the same whether the
underlying order is cache-aware or random — so the loss comes from the
sentence, not from the reordering. Per-frame timestamp labels are free.

Practical consequence: reorder frames for cache reuse, convey time with
per-frame labels, and do not append a positional order sentence. Labels also
sit inside the shared prefix, so they are cached; the sentence sits in the
suffix and is recomputed on every request.

## Result 2 — the cache hit needs a canonical prefix

Leading frames shared between two questions on the same video (they share ~4 of
their 16 retrieved frames):

| ordering | mean shared leading frames | pairs sharing none |
|---|---|---|
| chronological | 0.26 | 89 % |
| ContextPilot reorder | 3.04 | 30 % |
| canonical core prefix | 8.89 | 0 % |

Hybrid linear-attention models (Qwen3.5 / Qwen3.8 / GLM-5.3-Flash) only restore
their recurrent state at recorded checkpoints, so a prefix boundary that occurs
once is never reused: per-request reordering produced 1.2 % cached tokens while
the canonical prefix produced 10.8 % at only 3 questions per video, at equal
prompt length. Expect more where a video has more questions.

## Result 3 — cached tokens do not become faster prefill

Same runs, same hardware: TTFT is flat across every condition (11.5–12.1 s at
concurrency 4 on the 4B) regardless of a hit rate between 0 % and 10.8 %.

Isolated measurement on the same server:

| | |
|---|---|
| 16-frame request (1871 prompt tokens), cold | 3.17 s |
| same request, 99 % of tokens KV-cached | 3.06 s |
| 3817-token text-only prompt, cold / cached | 0.14 s / 0.05 s |
| per-frame cost, 64×64 … 896×504 | 191 … 233 ms, flat in resolution |
| throughput at concurrency 1 / 4 / 8 | 0.33 req/s at every level |

GPU utilisation is 0 % during those requests, and the number does not move with
`--image-processor-backend pil`, `--mm-preprocess-cache-size-mb`,
`--mm-io-worker-num`, `--mm-processor-worker-num`, `--mm-feature-transport
cuda_ipc --keep-mm-feature-on-device`, or sending frames as http URLs instead of
`data:` URIs. So on this stack the win from reordering is in tokens, not in
wall time. A wall-time win needs a serving path where the vision stage is not
the bottleneck, for example a deployment that caches vision embeddings across
requests (`--enable-mm-global-cache`, currently only wired into the
disaggregated `--encoder-only` mode).

## Honest limitations

- One benchmark (Video-MME) with 3 questions per video, so the canonical
  prefix is measured in its least favourable regime.
- Retrieval is a single dual-encoder over uniformly sampled frames; a stronger
  retriever changes the overlap between questions and therefore the headroom.
- The accuracy conclusion is established on Qwen3.5-4B; see the table above for
  the models each number comes from.
