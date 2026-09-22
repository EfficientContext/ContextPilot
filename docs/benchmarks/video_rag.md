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
| Benchmark | Video-MME, all three duration splits, 3 questions per video |
| Questions | 633 (211 videos) with retrieval available at run time |
| Retrieval | SigLIP-2 base over a 1 fps frame pool (≤256 frames/video, 448 px), top-16 frames per question |
| Model | Qwen3.8-27B, SGLang v0.5.20 stock image, 2×H100, tp 2 |
| Prompt | system + intro + frames + question, answer forced with "The best answer is:" |
| Requests | concurrency 4, engine cache flushed before every condition |

Conditions differ only in frame order and in how the true order is conveyed:

| condition | frame order | order information |
|---|---|---|
| `chrono_plain` | chronological | none (baseline) |
| `chrono_labels` | chronological | per-frame `[Frame i \| t=..s]` labels |
| `cp_sentence` | ContextPilot reorder | one sentence listing the true order |
| `canon_*` | canonical per-video core prefix | per the suffix |

## Result 1 — reordering frames does not cost accuracy

Paired against the `chrono_plain` baseline on the same 633 questions (McNemar
exact test on discordant pairs):

| condition | accuracy | Δ | lost / gained | p |
|---|---|---|---|---|
| `canon_none` (reorder, no order info) | 0.6888 | +0.002 | 26 / 27 | 1.00 |
| `chrono_plain` (baseline) | 0.6872 | — | — | — |
| `canon_sentence` (reorder + sentence) | 0.6777 | −0.010 | 36 / 30 | 0.54 |
| `cp_sentence` | 0.6746 | −0.013 | 36 / 28 | 0.38 |
| `canon_labels` (reorder + labels) | 0.6746 | −0.013 | 43 / 35 | 0.43 |
| `chrono_labels` | 0.6619 | −0.025 | 36 / 20 | 0.044 |
| `canon_both` (reorder + labels + sentence) | 0.6351 | −0.052 | 63 / 30 | 0.001 |

Reordering the frames is free: `canon_none` matches the chronological baseline
exactly. Adding one sentence that states the true chronological order is also
free within noise (p = 0.54). What does cost accuracy is piling both signals on
at once — labels *and* the sentence together lose 5 points (p = 0.001).

So for a model of this size the premise holds: frames can be reordered for
cache friendliness, with or without a one-line order hint. Use one order
signal, not two.

Note on model size: the same experiment on a 4B model showed the order sentence
costing 9–11 points (p < 1e−4) while reordering alone stayed free. The ability
to follow a positional order statement is what scales with model size, not the
tolerance for reordering.

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

Observed cached-token share in the runs above was low (≤ 0.9 %) under the
tp 2 / `--disable-prefill-cuda-graph` configuration used here, against 10.8 %
measured for the same canonical prefix on a single-GPU server. Treat the
canonical prefix as necessary but not sufficient: the engine's checkpoint
policy decides how much of it is actually reused.

## Result 3 — cached tokens do not become faster prefill

TTFT is flat across every condition above (7.4–7.9 s at concurrency 4),
independent of the hit rate. A multi-image request is dominated by host-side
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

- Video-MME gives only 3 questions per video, the least favourable regime for a
  canonical prefix. A long-video benchmark with ~11 questions per video is the
  better test of the cache claim.
- Retrieval is a single dual-encoder over uniformly sampled frames; a stronger
  retriever changes the frame overlap between questions and so the headroom.
- The accuracy numbers come from Qwen3.8-27B; the cache and latency figures are
  tied to the SGLang version and flags listed above.
