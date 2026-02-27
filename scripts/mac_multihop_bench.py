#!/usr/bin/env python3
"""
MultihopRAG Benchmark — macOS / Apple Silicon
llama.cpp + ContextPilot eviction proxy stack

Full pipeline:

  Step 1 (one-time): Build retrieval data
    docker run -d --name elasticsearch -p 9200:9200 \\
        -e "discovery.type=single-node" \\
        -e "xpack.security.enabled=false" \\
        -e "xpack.security.http.ssl.enabled=false" \\
        docker.elastic.co/elasticsearch/elasticsearch:8.18.2

    python examples/construct_rag_data/multihopRAG_bm25.py \\
        --corpus_path mulhoprag_corpus.jsonl \\
        --query_path  mulhoprag_queries.jsonl \\
        --output_path mulhoprag_bm25_top20.jsonl

  Step 2 (one-time): Reorder with ContextPilot
    python examples/offline/prepare_batch.py \\
        --context_path mulhoprag_bm25_top20.jsonl \\
        --output_path  mulhoprag_reordered.jsonl

  Step 3: Start services (three terminals)
    # A) llama.cpp
    llama-server -m models/Qwen3-8B-Q4_K_M.gguf \\
        --host 0.0.0.0 --port 8889 \\
        -ngl 99 --cache-reuse 256 --parallel 4 -c 32768

    # B) ContextPilot eviction proxy
    CONTEXTPILOT_INDEX_URL=http://localhost:8765 \\
        python patches/llama_cpp/eviction_proxy.py

    # C) ContextPilot HTTP server
    python -m contextpilot.server.http_server \\
        --port 8765 --infer-api-url http://localhost:8890

  Step 4: Run ContextPilot benchmark
    python scripts/mac_multihop_bench.py \\
        --reordered_path mulhoprag_reordered.jsonl \\
        --corpus_path    mulhoprag_corpus.jsonl \\
        --num_queries    100

  Step 5: Run baseline (restart llama.cpp first for a fair cache comparison)
    python scripts/mac_multihop_bench.py \\
        --reordered_path mulhoprag_reordered.jsonl \\
        --corpus_path    mulhoprag_corpus.jsonl \\
        --num_queries    100 \\
        --baseline

Metrics reported
  - F1 / Exact Match   answer quality  (contextpilot/utils/eval_metrics.py)
  - Cache Hit Rate     avg reuse_ratio_pct from proxy log
  - Prefill TP         prompt_n / prompt_eval_ms * 1000  (tokens/s)
  - Avg Latency        total wall-clock latency per request (ms)
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from openai import OpenAI
from tqdm import tqdm

from contextpilot.utils.eval_metrics import update_answer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONTEXTPILOT_URL = "http://localhost:8765"
PROXY_URL        = "http://localhost:8890"
MODEL            = "qwen3-8b"      # must match llama-server -m filename stem
MAX_TOKENS       = 128
TEMPERATURE      = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: str) -> dict[int, str]:
    """Return {chunk_id: text} map from corpus JSONL."""
    corpus = {}
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            cid = doc.get("chunk_id")
            if cid is not None:
                corpus[int(cid)] = doc.get("text", "")
    print(f"Loaded {len(corpus):,} chunks from corpus")
    return corpus


def load_reordered(path: str, num_queries: int) -> list[dict]:
    """Load output of prepare_batch.py (has top_k_doc_id + orig_top_k_doc_id)."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= num_queries:
                break
    return rows


def build_prompt(
    question: str,
    doc_ids: list[int],
    corpus: dict,
    max_docs: int = 10,
    max_chars_per_doc: int = 800,
) -> str:
    """Build a RAG prompt from doc IDs mapped to text.

    max_docs limits how many docs are included (keeps prompt within context window).
    max_chars_per_doc truncates each document to avoid runaway long chunks.
    """
    docs = []
    for i, did in enumerate(doc_ids[:max_docs], 1):
        text = corpus.get(int(did), f"[doc {did} not found]")
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "..."
        docs.append(f"[{i}] {text}")
    context_block = "\n\n".join(docs)
    return (
        f"Answer the question based on the documents below.\n\n"
        f"<documents>\n{context_block}\n</documents>\n\n"
        f"Question: {question}\nAnswer:"
    )


def reset_proxy():
    """Clear slot state on the eviction proxy between runs."""
    try:
        r = requests.post(f"{PROXY_URL}/reset", timeout=5)
        if r.status_code == 200:
            print("Proxy slot state reset.")
    except Exception as e:
        print(f"Warning: could not reset proxy: {e}")


def check_services():
    """Verify all three services are reachable before starting."""
    ok = True
    for name, url in [
        ("ContextPilot HTTP server", f"{CONTEXTPILOT_URL}/health"),
        ("Eviction proxy /stats",    f"{PROXY_URL}/stats"),
    ]:
        try:
            r = requests.get(url, timeout=3)
            print(f"  {name}: OK ({r.status_code})")
        except Exception:
            print(f"  {name}: NOT REACHABLE at {url}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Metrics from proxy log
# ---------------------------------------------------------------------------

def parse_proxy_log(log_path: str, since_ts: float) -> dict:
    """
    Read query_log.jsonl written by eviction_proxy.py and compute aggregate
    cache / prefill metrics for records written after `since_ts` (unix time).
    """
    if not os.path.exists(log_path):
        return {}

    prompt_tokens_total = 0
    prompt_eval_ms_total = 0.0
    reuse_ratios = []
    latencies = []

    with open(log_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue

            # Filter to records from this run
            ts_str = rec.get("timestamp", "")
            try:
                from datetime import datetime, timezone
                ts = datetime.fromisoformat(ts_str).timestamp()
                if ts < since_ts:
                    continue
            except Exception:
                pass

            cr = rec.get("cache_reuse", {})
            if cr.get("reuse_ratio_pct") is not None:
                reuse_ratios.append(cr["reuse_ratio_pct"])
            if cr.get("prompt_eval_ms") and cr.get("newly_computed"):
                prompt_eval_ms_total += cr["prompt_eval_ms"]
                prompt_tokens_total  += cr["newly_computed"]
            if rec.get("latency_ms"):
                latencies.append(rec["latency_ms"])

    if not latencies:
        return {}

    prefill_tps = (
        prompt_tokens_total / prompt_eval_ms_total * 1000
        if prompt_eval_ms_total > 0 else 0.0
    )
    return {
        "avg_cache_hit_pct": round(sum(reuse_ratios) / len(reuse_ratios), 1)
                             if reuse_ratios else 0.0,
        "prefill_tps":       round(prefill_tps, 1),
        "avg_latency_ms":    round(sum(latencies) / len(latencies), 1),
        "p50_latency_ms":    sorted(latencies)[len(latencies) // 2],
        "n_requests":        len(latencies),
    }


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(
    rows: list[dict],
    corpus: dict,
    use_baseline: bool,
    proxy_log: str,
    max_docs: int = 10,
    max_chars_per_doc: int = 800,
) -> tuple[dict, list[dict]]:
    """
    Send all queries to llama.cpp via the ContextPilot HTTP server.

    use_baseline=True  → original doc IDs in original query order
    use_baseline=False → reordered doc IDs in ContextPilot-scheduled order
                         (rows already come from prepare_batch.py in optimal order)
    """
    client = OpenAI(base_url=f"{CONTEXTPILOT_URL}/v1", api_key="EMPTY")

    # Collect answers keyed by qid for eval_metrics
    answers = []    # [{qid, question, predicted, gold_answers}, ...]
    t_run_start = time.time()

    for row in tqdm(rows, desc="Inference"):
        question = row.get("question") or row.get("text", "")
        gold_answers = row.get("answers") or ([row["answer"]] if row.get("answer") else [])
        qid = row.get("qid", 0)

        if use_baseline:
            doc_ids = row.get("orig_top_k_doc_id") or row.get("top_k_doc_id", [])
        else:
            doc_ids = row.get("top_k_doc_id", [])

        prompt = build_prompt(question, doc_ids, corpus, max_docs, max_chars_per_doc)

        try:
            resp = client.completions.create(
                model=MODEL,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            predicted = resp.choices[0].text.strip()
        except Exception as e:
            predicted = ""
            tqdm.write(f"[qid={qid}] inference error: {e}")

        answers.append({
            "qid":          qid,
            "question":     question,
            "predicted":    predicted,
            "gold_answers": gold_answers,
        })

    # --- F1 / EM ---
    metrics_acc = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    for item in answers:
        if item["gold_answers"]:
            update_answer(metrics_acc, item["predicted"], item["gold_answers"])

    n = len(answers)
    qa_metrics = {k: round(v / n * 100, 2) for k, v in metrics_acc.items()} if n else {}

    # --- Cache / prefill metrics from proxy log ---
    cache_metrics = parse_proxy_log(proxy_log, since_ts=t_run_start)

    return {**qa_metrics, **cache_metrics}, answers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MultihopRAG benchmark on Mac using llama.cpp + ContextPilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reordered_path", required=True,
        help="JSONL output of prepare_batch.py (has top_k_doc_id + orig_top_k_doc_id)",
    )
    parser.add_argument(
        "--corpus_path", required=True,
        help="JSONL corpus file with chunk_id and text fields",
    )
    parser.add_argument(
        "--num_queries", type=int, default=100,
        help="Number of queries to evaluate (default: 100 — full dataset is ~2500)",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Run in baseline mode: original doc order, no ContextPilot scheduling",
    )
    parser.add_argument(
        "--proxy_log", default="query_log.jsonl",
        help="Path to eviction proxy JSONL log (default: query_log.jsonl)",
    )
    parser.add_argument(
        "--max_docs", type=int, default=10,
        help="Max documents per prompt (default: 10). Reduce if getting 400 context-too-long errors.",
    )
    parser.add_argument(
        "--max_chars_per_doc", type=int, default=800,
        help="Truncate each document to this many characters (default: 800 ≈ 200 tokens).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save per-query results as JSONL",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" MultihopRAG Benchmark — Mac / Apple Silicon")
    print("=" * 60)
    mode = "BASELINE (original order)" if args.baseline else "CONTEXTPILOT (reordered)"
    print(f" Mode:        {mode}")
    print(f" Queries:     {args.num_queries}")
    print(f" Max docs:    {args.max_docs}  (chars/doc: {args.max_chars_per_doc})")
    print(f" Corpus:      {args.corpus_path}")
    print(f" Reordered:   {args.reordered_path}")
    print(f" Proxy log:   {args.proxy_log}")
    print()

    # --- Service check ---
    print("Checking services...")
    if not check_services():
        print("\nStart all three services before running.  See module docstring.")
        return

    # --- Reset proxy slot state before run ---
    reset_proxy()

    # --- Load data ---
    corpus = load_corpus(args.corpus_path)
    rows   = load_reordered(args.reordered_path, args.num_queries)
    print(f"Loaded {len(rows)} queries\n")

    if not rows:
        print("No rows loaded — check --reordered_path")
        return

    # --- Run inference ---
    print(f"Running inference ({mode})...")
    results, per_query = run_inference(
        rows=rows,
        corpus=corpus,
        use_baseline=args.baseline,
        proxy_log=args.proxy_log,
        max_docs=args.max_docs,
        max_chars_per_doc=args.max_chars_per_doc,
    )

    # --- Print summary ---
    print("\n" + "=" * 60)
    print(f" Results — {mode}")
    print("=" * 60)
    for k, v in results.items():
        unit = "%" if ("pct" in k or k in ("em", "f1", "prec", "recall")) else \
               " tok/s" if "tps" in k else \
               " ms" if "ms" in k else ""
        print(f"  {k:<25} {v}{unit}")
    print("=" * 60)
    print()
    print("Reference numbers from README (SGLang + Qwen3-32B on 4×A6000):")
    print("  Without ContextPilot:  cache_hit=4.64%  prefill_tps=7,290  F1=60.42")
    print("  With    ContextPilot:  cache_hit=33.97% prefill_tps=14,214 F1=64.39")
    print()
    print("Note: llama.cpp on Apple Silicon will show lower absolute TPS than")
    print("      a GPU server, but the relative cache_hit improvement should")
    print("      still be visible (~5x improvement expected).")

    # --- Save per-query output ---
    if args.output:
        with open(args.output, "w") as f:
            for item in per_query:
                f.write(json.dumps(item) + "\n")
        print(f"Per-query results saved to {args.output}")


if __name__ == "__main__":
    main()
