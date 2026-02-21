"""
Stress test: fill vLLM KV cache via ContextPilot full pipeline.

Flow per request:
  1. POST /reorder with random doc-ID contexts → get reordered_contexts + request_ids
  2. POST /v1/completions (ContextPilot proxy) with request_id → forwards to vLLM
  3. vLLM evicts → POST /evict → ContextPilot prunes index → 200

Usage:
    python examples/stress_vllm_cache.py

Env vars:
    CONTEXTPILOT_URL    ContextPilot server (default http://localhost:8765)
    NUM_WORKERS         concurrent threads (default 8)
    NUM_REQUESTS        total requests (default 200)
    MAX_TOKENS          output tokens per request (default 1)
    NUM_DOCS            docs per context (default 30)
    DOC_POOL_SIZE       total unique doc IDs to sample from (default 200)
    WORDS_PER_DOC       words of filler per doc to bulk up prompts (default 40)
"""

import os
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

CP_URL = os.environ.get("CONTEXTPILOT_URL", "http://localhost:8765")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
NUM_REQUESTS = int(os.environ.get("NUM_REQUESTS", "200"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1"))
NUM_DOCS = int(os.environ.get("NUM_DOCS", "30"))
DOC_POOL_SIZE = int(os.environ.get("DOC_POOL_SIZE", "200"))
WORDS_PER_DOC = int(os.environ.get("WORDS_PER_DOC", "40"))


def get_model():
    """Get model name from inference engine via ContextPilot proxy."""
    r = requests.get(f"{CP_URL}/v1/models", timeout=5)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def build_corpus(pool_size, words_per_doc):
    """Pre-generate a fake corpus: doc_id → text string.
    Each doc is ~words_per_doc words so prompts are large enough to fill cache.
    Deterministic per doc_id so same doc always produces same text (prefix sharing).
    """
    corpus = {}
    for doc_id in range(pool_size):
        rng = random.Random(doc_id)  # deterministic per doc
        words = " ".join(
            "".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 8)))
            for _ in range(words_per_doc)
        )
        corpus[doc_id] = f"Document {doc_id}: {words}"
    return corpus


def make_context(num_docs, pool_size):
    """Generate a random context (list of doc IDs) with partial overlap.
    First 5 docs are from a small shared pool (prefix sharing),
    rest are random from the full pool (unique tails).
    """
    shared_prefix = random.sample(range(10), k=min(5, num_docs))
    tail_count = max(num_docs - 5, 0)
    tail = random.sample(range(pool_size), k=min(tail_count, pool_size))
    return shared_prefix + tail


def do_request(model, idx, corpus):
    """One full round-trip: /reorder → /v1/completions."""
    context = make_context(NUM_DOCS, DOC_POOL_SIZE)

    # Step 1: reorder through ContextPilot
    try:
        rr = requests.post(f"{CP_URL}/reorder", json={
            "contexts": [context],
            "use_gpu": False,
            "linkage_method": "average",
            "alpha": 0.0005,
        }, timeout=15)
        rr.raise_for_status()
        reorder_resp = rr.json()
    except Exception as e:
        return "reorder_fail", idx, str(e)[:120]

    reordered = reorder_resp.get("reordered_contexts", [context])[0]
    request_ids = reorder_resp.get("request_ids", [])
    request_id = request_ids[0] if request_ids else None

    # Build a fat prompt from the reordered doc IDs using corpus text
    context_str = "\n\n".join(corpus.get(d, f"[doc {d}]") for d in reordered)
    prompt = f"Context:\n{context_str}\n\nQuestion: Summarize document {reordered[0]}.\nAnswer:"

    # Step 2: send to vLLM via ContextPilot proxy
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    if request_id:
        payload["request_id"] = request_id

    try:
        cr = requests.post(
            f"{CP_URL}/v1/completions", json=payload, timeout=60
        )
        if cr.status_code != 200:
            return "infer_fail", idx, f"status={cr.status_code} {cr.text[:80]}"
        return "ok", idx, request_id or ""
    except Exception as e:
        return "infer_fail", idx, str(e)[:120]


def main():
    # Preflight checks
    print("Checking ContextPilot...", end=" ", flush=True)
    try:
        h = requests.get(f"{CP_URL}/health", timeout=3)
        print(f"status={h.status_code}")
    except Exception as e:
        print(f"FAILED: {e}")
        return

    model = get_model()
    print(f"Model: {model}")
    print(f"Config: {NUM_WORKERS} workers, {NUM_REQUESTS} requests, "
          f"{MAX_TOKENS} max_tokens, {NUM_DOCS} docs/context, "
          f"pool={DOC_POOL_SIZE}, ~{WORDS_PER_DOC} words/doc")

    # Build corpus
    print("Building corpus...", end=" ", flush=True)
    corpus = build_corpus(DOC_POOL_SIZE, WORDS_PER_DOC)
    sample_prompt_len = sum(len(corpus[i]) for i in range(min(NUM_DOCS, DOC_POOL_SIZE)))
    print(f"{len(corpus)} docs, ~{sample_prompt_len} chars/prompt")

    # Warmup
    print("Warmup...", end=" ", flush=True)
    status, _, msg = do_request(model, -1, corpus)
    print(f"{status} {msg}")
    print()

    ok, reorder_fail, infer_fail = 0, 0, 0
    errors = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {
            pool.submit(do_request, model, i, corpus): i
            for i in range(NUM_REQUESTS)
        }
        for i, future in enumerate(as_completed(futures), 1):
            status, idx, msg = future.result()
            if status == "ok":
                ok += 1
            elif status == "reorder_fail":
                reorder_fail += 1
                if len(errors) < 5:
                    errors.append(f"  [{idx}] reorder: {msg}")
            else:
                infer_fail += 1
                if len(errors) < 5:
                    errors.append(f"  [{idx}] infer: {msg}")

            if i % 25 == 0 or i == NUM_REQUESTS:
                elapsed = time.perf_counter() - t0
                rps = i / elapsed
                print(f"  [{i:>4}/{NUM_REQUESTS}] ok={ok} "
                      f"reorder_fail={reorder_fail} infer_fail={infer_fail} "
                      f"{elapsed:.1f}s {rps:.1f}rps")

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {ok} ok, {reorder_fail} reorder_fail, "
          f"{infer_fail} infer_fail in {elapsed:.1f}s "
          f"({ok/elapsed:.1f} req/s)")

    if errors:
        print("\nFirst few errors:")
        for e in errors:
            print(e)

    # Check index state
    try:
        sr = requests.get(f"{CP_URL}/stats", timeout=3)
        if sr.status_code == 200:
            stats = sr.json()
            print(f"\nContextPilot index stats: {stats}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
