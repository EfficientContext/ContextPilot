#!/usr/bin/env python3
"""
ContextPilot + llama.cpp on Apple Silicon — Sample Queries

Three usage patterns:
  1. Single-query  — one question at a time
  2. Multi-turn    — conversation with cross-turn deduplication
  3. Batch         — offline batch processing in optimal execution order

Setup (two terminals):

  Terminal 1 – llama-server with Metal + prefix caching:
    llama-server -m models/Qwen3-8B-Q4_K_M.gguf \\
        --host 0.0.0.0 --port 8889 \\
        -ngl 99 --cache-reuse 256 --parallel 4 -c 32768

    llama-server ships its own OpenAI-compatible /v1/* API, so no separate
    API server or format-translation proxy is needed.

  Terminal 2 – ContextPilot HTTP server (points directly at llama-server):
    python -m contextpilot.server.http_server --port 8765 \\
        --infer-api-url http://localhost:8889

  Then run:
    pip install -e .
    python examples/mac_llama_cpp_example.py
"""

from openai import OpenAI
import contextpilot as cp

# ---------------------------------------------------------------------------
# Client points at the ContextPilot HTTP server.
# The server injects `rid` automatically and forwards to the eviction proxy.
# ---------------------------------------------------------------------------
client = OpenAI(base_url="http://localhost:8765/v1", api_key="EMPTY")
cp_instance = cp.ContextPilot(use_gpu=False)

MODEL = "qwen3-8b"

# ---------------------------------------------------------------------------
# Sample document corpus (simulates RAG retrieval results)
# ---------------------------------------------------------------------------
DOCS = {
    "transformer": (
        "The Transformer architecture, introduced in 'Attention Is All You Need' (2017), "
        "replaces recurrent layers with multi-head self-attention. It processes all tokens "
        "in parallel during training, enabling massive scale-up on GPU/TPU hardware. "
        "The encoder maps an input sequence to contextual embeddings; the decoder "
        "auto-regressively generates the output sequence using cross-attention over "
        "the encoder output."
    ),
    "attention": (
        "Self-attention computes a weighted sum of all token representations. "
        "For each query token q, the score against key token k is dot(q,k)/sqrt(d_k). "
        "Scores are softmax-normalised to produce attention weights, then applied to "
        "value vectors. Multi-head attention runs h independent heads in parallel "
        "and concatenates their outputs, letting the model attend to different "
        "representation subspaces simultaneously."
    ),
    "kv_cache": (
        "KV-cache stores the key and value tensors computed for all previous tokens "
        "so they do not need to be recomputed during auto-regressive decoding. "
        "For a 7B model with 32 layers, 32 heads, and head-dim 128, each token costs "
        "2 × 32 × 32 × 128 × 2 bytes ≈ 0.5 MB. Prefix caching extends this by "
        "reusing KV tensors for a shared prompt prefix across different requests."
    ),
    "llama_cpp": (
        "llama.cpp is a C++ inference engine for GGUF-quantised LLMs. It supports "
        "Apple Silicon Metal GPU offload via -ngl (number of GPU layers). "
        "The --cache-reuse N flag enables prefix caching: if a new request shares "
        "N or more tokens with the sequence currently cached in a slot, those tokens "
        "are reused without re-evaluation. --parallel K allocates K independent "
        "KV-cache slots for concurrent requests."
    ),
    "rag": (
        "Retrieval-Augmented Generation (RAG) combines a retriever with a generative "
        "LLM. The retriever (BM25 or dense embedding) selects the top-k documents "
        "relevant to a query; the LLM generates an answer conditioned on those "
        "documents. RAG reduces hallucinations by grounding the model in a "
        "dynamically updated knowledge base without fine-tuning."
    ),
    "quantisation": (
        "GGUF quantisation reduces model weight precision from float16 to 4-8 bits, "
        "cutting memory requirements by 50-75%. Q4_K_M uses a mixed 4-bit scheme "
        "with higher precision for attention and feed-forward projection layers, "
        "offering near-fp16 quality at ~4.5 bits/weight. On Apple Silicon the "
        "entire quantised model fits in the unified DRAM shared by CPU and GPU."
    ),
}


def get_contexts(query: str) -> list[str]:
    """Simulate a retriever: return relevant documents for a query.

    In production replace this with BM25Retriever, FAISSRetriever, etc.
    Documents that overlap across queries will share KV-cache tokens.
    """
    keyword_map = {
        "attention":       ["transformer", "attention", "kv_cache"],
        "prefix":          ["kv_cache", "llama_cpp", "transformer"],
        "quantis":         ["quantisation", "llama_cpp", "kv_cache"],
        "rag":             ["rag", "transformer", "attention"],
        "retriev":         ["rag", "kv_cache", "transformer"],
        "throughput":      ["llama_cpp", "quantisation", "kv_cache"],
    }
    q = query.lower()
    for kw, keys in keyword_map.items():
        if kw in q:
            return [DOCS[k] for k in keys]
    # fallback: return all docs
    return list(DOCS.values())[:3]


# ===========================================================================
# Example 1 – Single query
# ===========================================================================

def example_single_query():
    print("\n" + "=" * 60)
    print("Example 1: Single Query")
    print("=" * 60)

    queries = [
        "How does self-attention work in the Transformer?",
        "What is prefix caching and how does it speed up llama.cpp?",
        "How much memory does Q4_K_M quantisation save on Apple Silicon?",
    ]

    for query in queries:
        contexts = get_contexts(query)
        messages = cp_instance.optimize(contexts, query)
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=200,
        )
        answer = response.choices[0].message.content.strip()
        print(f"\nQ: {query}")
        print(f"A: {answer[:200]}{'...' if len(answer) > 200 else ''}")


# ===========================================================================
# Example 2 – Multi-turn conversation
# ===========================================================================

def example_multi_turn():
    """Pass the same conversation_id across turns so ContextPilot can
    deduplicate documents already seen in earlier turns."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-turn Conversation")
    print("=" * 60)

    import uuid
    conversation_id = f"conv-{uuid.uuid4().hex[:8]}"

    turns = [
        "What is the Transformer architecture?",
        "How does the KV cache interact with self-attention during decoding?",
        "What llama.cpp flags maximise throughput on an M3 MacBook Pro?",
    ]

    for turn_idx, query in enumerate(turns, 1):
        contexts = get_contexts(query)
        messages = cp_instance.optimize(contexts, query, conversation_id=conversation_id)
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=200,
        )
        answer = response.choices[0].message.content.strip()
        print(f"\nTurn {turn_idx}: {query}")
        print(f"Answer:  {answer[:200]}{'...' if len(answer) > 200 else ''}")


# ===========================================================================
# Example 3 – Batch (offline)
# ===========================================================================

def example_batch():
    """optimize_batch schedules all queries in the globally optimal order so
    queries that share documents are sent consecutively — maximising prefix
    reuse across the entire batch."""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)

    all_queries = [
        "Explain how multi-head attention works in a Transformer.",
        "How does RAG reduce hallucinations in LLM responses?",
        "What is prefix caching and why does it matter for throughput?",
        "How does Q4_K_M quantisation affect model quality?",
    ]

    all_docs = [get_contexts(q) for q in all_queries]

    messages_batch, original_indices = cp_instance.optimize_batch(all_docs, all_queries)

    print(f"Scheduled execution order: {original_indices}")
    print("(queries with shared docs are grouped for maximum KV-cache reuse)\n")

    answers = [""] * len(all_queries)
    for messages, orig_idx in zip(messages_batch, original_indices):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=200,
        )
        answers[orig_idx] = response.choices[0].message.content.strip()

    for i, (query, answer) in enumerate(zip(all_queries, answers)):
        print(f"Q{i}: {query}")
        print(f"A{i}: {answer[:180]}{'...' if len(answer) > 180 else ''}")
        print()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   ContextPilot + llama.cpp — Apple Silicon Sample        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    example_single_query()
    example_multi_turn()
    example_batch()

    print("\nDone.")
