# Video-RAG benchmark: frame reordering for KV-cache reuse

Question studied: in large-scale video QA where each question retrieves a
subset of a video's frames, can we **reorder the frames** so that frames shared
across questions form a common prefix (KV-cache hits) and **tell the model the
true chronological order in one sentence**, and thereby cut prefill time
without losing accuracy?

Pipeline:

1. `build_questions.py` — unify benchmark annotations (Video-MME, LVBench,
   EgoSchema) into `questions.jsonl` + a video manifest.
2. `prepare_frames.py` — sample a frame pool per video (1 fps, capped at 256
   uniformly spaced frames, 448 px wide JPEG).
3. `retrieve_frames.py` — SigLIP-2 text→frame retrieval, top-k frames per
   question (`retrieval_k64.jsonl`).
4. `run_video_rag_bench.py` — send the same retrieved frames under several
   prompt conditions to an SGLang OpenAI-compatible server, flushing the cache
   before each condition; records TTFT, cached tokens and MCQ accuracy.
5. `analyze_results.py` — tables, paired comparisons, per-task-type accuracy.

Conditions:

| name | frame order | order information |
|---|---|---|
| `chrono_plain` | chronological | none (natural baseline) |
| `chrono_labels` | chronological | `[Frame i | t=..s]` labels |
| `cp_sentence` | ContextPilot reorder | one sentence with the true order |
| `cp_labels` | ContextPilot reorder | timestamp labels |
| `cp_both` | ContextPilot reorder | labels + sentence |
| `cp_none` | ContextPilot reorder | none (ablation) |
| `shuffle_sentence` | random | sentence (isolates the hint from the cache effect) |

Example (inside the cluster, SGLang service `sglang-qwen38-27b`):

```bash
python run_video_rag_bench.py \
  --questions /mnt/cp/data/videomme/questions_long.jsonl \
  --retrieval /mnt/cp/data/videomme/retrieval_k64.jsonl \
  --frames /mnt/cp/data/videomme/frames \
  --api http://sglang-qwen38-27b:8000 --model Qwen/Qwen3.8-27B \
  --k 64 --concurrency 1 --out /mnt/cp/results/videomme_long_qwen38_27b
```

Notes on measurement: the server is started with `--enable-cache-report` so
`usage.prompt_tokens_details.cached_tokens` is returned; `POST /flush_cache` is
issued before every condition; requests are sent in the order produced by
ContextPilot's scheduler for `cp_*` conditions and in dataset order (grouped by
video) otherwise. See `docs/guides/multimodal.md` for the hybrid-model
checkpoint caveat that limits prefix hits on Qwen3.5/3.8 and GLM-5.3-Flash.
