# Video RAG: does reordering frames for cache reuse keep accuracy?

Setting: long-video question answering where each question retrieves a subset
of a video's frames. Reordering those frames so that frames shared with earlier
questions form a common prefix raises the engine's prefix-cache hit rate, but
it also destroys the chronological order the model would normally rely on. The
usual remedy is to state the true order in one sentence *after* the frames,
where it does not break the shared prefix.

This page reports what that costs and what it buys, measured end to end.

**Reproduce**: `examples/video_rag/` (frame extraction, retrieval, benchmark
runner, analysis). These are raw runs, not tuned.

## Setup

| | |
|---|---|
| Benchmarks | Video-MME (3 questions per video) and LVBench (hour-long videos, ~11 questions per video) |
| Retrieval | SigLIP-2 base over a 1 fps frame pool (≤256 frames/video, 448 px), top-16 frames per question |
| Models | Qwen3.8-27B (2×H100, tp 2), Qwen3.8-Flash-Next-FP8 180B (4×H100, tp 4 ep 4) |
| Engine | SGLang v0.5.20, stock image, `--enable-cache-report` |
| Prompt | system + intro + frames + question, answer forced with "The best answer is:" |
| Requests | concurrency 4, engine cache flushed before every condition |

Conditions differ only in frame order and in how the true order is conveyed:

| condition | frame order | order information |
|---|---|---|
| `chrono_plain` | chronological | none (baseline) |
| `chrono_labels` | chronological | per-frame `[Frame i \| t=..s]` labels |
| `cp_sentence` | per-request ContextPilot reorder | one sentence listing the true order |
| `canon_*` | canonical per-video core prefix | per the suffix |

## Result 1 — reordering frames does not cost accuracy

**LVBench, Qwen3.8-27B, 1131 questions.** Paired against the chronological
baseline on the same questions (McNemar exact test on discordant pairs):

| condition | accuracy | Δ | p | cached tokens | uncached tokens/req |
|---|---|---|---|---|---|
| `canon_none` (reorder, no order info) | 0.4403 | +0.002 | 0.91 | **8.0 %** | **1804** |
| `chrono_plain` (baseline) | 0.4385 | — | — | 0.6 % | 1949 |
| `canon_labels` | 0.4332 | −0.005 | 0.70 | 7.7 % | 2048 |
| `canon_sentence` | 0.4324 | −0.006 | 0.57 | 7.0 % | 2077 |
| `cp_sentence` (per-request reorder) | 0.4253 | −0.013 | 0.22 | 0.4 % | 2236 |

The canonical reordering matches the baseline accuracy exactly while raising the
cached-token share from 0.6 % to 8.0 % and cutting prefill tokens by 7.4 %.

**Video-MME, Qwen3.8-27B, 633 questions**, baseline 0.6872:

| condition | accuracy | Δ | p |
|---|---|---|---|
| `canon_none` | 0.6888 | +0.002 | 1.00 |
| `canon_sentence` | 0.6777 | −0.010 | 0.54 |
| `cp_sentence` | 0.6746 | −0.013 | 0.38 |
| `canon_labels` | 0.6746 | −0.013 | 0.43 |
| `chrono_labels` | 0.6619 | −0.025 | 0.044 |
| `canon_both` (labels + sentence) | 0.6351 | −0.052 | 0.001 |

**Video-MME, Qwen3.8-Flash-Next-FP8 180B, 708 questions**, baseline 0.7542:
`canon_none` 0.7500, `canon_sentence` 0.7444, `cp_sentence` 0.7260.

Across both benchmarks and both models, reordering the frames is free, and one
sentence stating the true order is free within noise. What does cost accuracy
is supplying two order signals at once: labels *and* the sentence lose 5 points
on Video-MME (p = 0.001). Use one order signal, not two.

Note on model size: the same experiment on a 4B model showed the order sentence
costing 9–11 points (p < 1e−4) while reordering alone stayed free. Following a
positional order statement is what scales with model size, not tolerance for
reordering.

## Result 2 — the cache hit needs a canonical prefix

Leading frames shared between two questions on the same video (they share ~4 of
their 16 retrieved frames):

| ordering | mean shared leading frames | pairs sharing none |
|---|---|---|
| chronological | 0.26 | 89 % |
| ContextPilot reorder | 3.04 | 30 % |
| canonical core prefix | 8.89 | 0 % |

Hybrid linear-attention models (Qwen3.5 / Qwen3.8 / GLM-5.3-Flash) restore
their recurrent state only at recorded checkpoints, so a prefix boundary that
occurs once is never reused. Measured on one such server: a shared boundary
misses on its first two occurrences and is reused from the third on, which is
why per-request reordering yields nothing and a canonical prefix is required.

The payoff scales with how many questions share a video. On LVBench (~11
questions per video) the canonical prefix reached 8.0 % cached tokens against
0.6 % for the chronological baseline. On Video-MME (3 questions per video) the
same mechanism reached 10.8 % on a single-GPU server but under 1 % on the
multi-GPU servers used for the models above, so treat the canonical prefix as
necessary but not sufficient — the engine's checkpoint policy decides how much
of it is actually reused.

## Result 3 — cached tokens do not become faster prefill

TTFT is flat across every condition above, independent of the hit rate: 7.5–7.7 s
at concurrency 4 for the 27B and 9.0–9.5 s for the 180B. A multi-image request is dominated by host-side
per-image work that the KV cache does not cover. Isolated measurement on the
same stack:

| | |
|---|---|
| 16-frame request (1871 prompt tokens), cold | 3.17 s |
| same request, 99 % of tokens KV-cached | 3.06 s |
| 3817-token text-only prompt, cold / cached | 0.14 s / 0.05 s |
| per-frame cost, 64×64 … 896×504 | 191 … 233 ms, flat in resolution |
| throughput at concurrency 1 / 4 / 8 | 0.33 req/s at every level |

GPU utilisation is 0 % during those requests, and the figure does not move with
`--image-processor-backend pil`, `--mm-preprocess-cache-size-mb`,
`--mm-io-worker-num`, `--mm-processor-worker-num`, `--mm-feature-transport
cuda_ipc --keep-mm-feature-on-device`, or with frames sent as http URLs instead
of `data:` URIs. So on this stack the win from reordering is in tokens, not in
wall time. A wall-time win needs a serving path where the vision stage is not
the bottleneck, for example one that caches vision embeddings across requests
(`--enable-mm-global-cache`, currently wired only into the disaggregated
`--encoder-only` mode).

## Honest limitations

- Retrieval is a single dual-encoder over uniformly sampled frames; a stronger
  retriever changes the frame overlap between questions and so the headroom.
- 16 frames is sparse coverage for LVBench's hour-long videos, which is why its
  absolute accuracy is low; the comparison between conditions is unaffected
  because every condition sees the same frames.
- The cache and latency figures are tied to the SGLang version and flags listed
  above, and the cached-token share varied by server configuration.
- The 180B run covers a subset of the conditions; the 27B carries the full grid.
