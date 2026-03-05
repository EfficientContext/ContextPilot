#!/usr/bin/env python3
"""
MultihopRAG Benchmark — macOS / Apple Silicon
openjiuwen WorkflowAgent + ContextPilot

Drop-in replacement for mac_multihop_bench.py that routes inference through
openjiuwen's WorkflowAgent instead of a raw OpenAI client.

Usage (same flags as mac_multihop_bench.py):

  # ContextPilot reordered
  python scripts/mac_multihop_bench_jiuwen.py \\
      --reordered_path mulhoprag_reordered.jsonl \\
      --corpus_path    mulhoprag_corpus.jsonl \\
      --num_queries    100

  # Baseline — restart llama-server first to clear KV cache, then:
  python scripts/mac_multihop_bench_jiuwen.py \\
      --reordered_path mulhoprag_reordered.jsonl \\
      --corpus_path    mulhoprag_corpus.jsonl \\
      --num_queries    100 \\
      --baseline

See test_mac_contextpilot.sh for the full automated pipeline.
"""

import argparse
import asyncio
import json
import os
import time

from tqdm import tqdm

from contextpilot.utils.eval_metrics import update_answer

from openjiuwen.core.workflow import (
    Start, End, LLMComponent, LLMCompConfig,
    Workflow, WorkflowCard, generate_workflow_key,
)
from openjiuwen.core.foundation.llm import ModelRequestConfig, ModelClientConfig
from openjiuwen.core.runner.runner import Runner
from openjiuwen.core.single_agent.legacy import WorkflowAgentConfig
from openjiuwen.core.application.workflow_agent import WorkflowAgent


# ---------------------------------------------------------------------------
# Config — override with env vars or CLI flags
# ---------------------------------------------------------------------------

LLAMA_PORT  = int(os.environ.get("LLAMA_PORT",  "8889"))
MODEL_NAME  = os.environ.get("MODEL_NAME", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
MAX_TOKENS  = 128
TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Build openjiuwen WorkflowAgent (module-level, created once)
# ---------------------------------------------------------------------------

def build_agent(model_name: str, llama_port: int) -> WorkflowAgent:
    model_client_config = ModelClientConfig(
        client_provider="openai",
        api_key="EMPTY",
        api_base=f"http://localhost:{llama_port}/v1",
        verify_ssl=False,
        timeout=120.0,
        max_retries=1,
    )
    model_config = ModelRequestConfig(
        model=model_name,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    workflow_card = WorkflowCard(
        id="multihop_rag_workflow",
        name="multihop_rag",
        version="1.0",
        description="MultihopRAG inference via openjiuwen WorkflowAgent",
        input_params={
            "type": "object",
            "properties": {
                "query":   {"type": "string", "description": "The question to answer"},
                "context": {"type": "string", "description": "Retrieved documents (already reordered by ContextPilot)"},
            },
            "required": ["query", "context"],
        },
    )

    flow = Workflow(card=workflow_card)
    start = Start()
    end = End({"responseTemplate": "{{output}}"})

    llm_config = LLMCompConfig(
        model_client_config=model_client_config,
        model_config=model_config,
        template_content=[
            {
                "role": "system",
                "content": (
                    "Answer the question based only on the documents below. "
                    "Be concise — one sentence or a short phrase.\n\n"
                    "<documents>\n{{context}}\n</documents>"
                ),
            },
            {"role": "user", "content": "Question: {{query}}\nAnswer:"},
        ],
        response_format={"type": "text"},
        output_config={"output": {"type": "string", "description": "Model answer"}},
    )
    llm = LLMComponent(llm_config)

    flow.set_start_comp("start", start, inputs_schema={"query": "${query}", "context": "${context}"})
    flow.add_workflow_comp("llm", llm, inputs_schema={"query": "${start.query}", "context": "${start.context}"})
    flow.set_end_comp("end", end, inputs_schema={"output": "${llm.output}"})
    flow.add_connection("start", "llm")
    flow.add_connection("llm", "end")

    Runner.resource_mgr.add_workflow(
        WorkflowCard(id=generate_workflow_key(flow.card.id, flow.card.version)),
        lambda: flow,
    )

    agent_config = WorkflowAgentConfig(
        id="multihop_rag_agent",
        version="1.0.0",
        description="MultihopRAG WorkflowAgent",
    )
    agent = WorkflowAgent(agent_config)
    agent.add_workflows([flow])
    return agent


# ---------------------------------------------------------------------------
# Data helpers (identical to mac_multihop_bench.py)
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: str) -> dict[int, str]:
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
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= num_queries:
                break
    return rows


def build_context(
    doc_ids: list[int],
    corpus: dict,
    max_docs: int = 10,
    max_chars_per_doc: int = 800,
) -> str:
    """Build context string from doc IDs (query goes separately to the workflow)."""
    docs = []
    for i, did in enumerate(doc_ids[:max_docs], 1):
        text = corpus.get(int(did), f"[doc {did} not found]")
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "..."
        docs.append(f"[{i}] {text}")
    return "\n\n".join(docs)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

async def run_inference(
    agent: WorkflowAgent,
    rows: list[dict],
    corpus: dict,
    use_baseline: bool,
    max_docs: int = 10,
    max_chars_per_doc: int = 800,
) -> tuple[dict, list[dict]]:
    answers = []
    latencies = []

    for row in tqdm(rows, desc="Inference"):
        question    = row.get("question") or row.get("text", "")
        gold_answers = row.get("answers") or ([row["answer"]] if row.get("answer") else [])
        qid         = row.get("qid", 0)

        doc_ids = (
            row.get("orig_top_k_doc_id") or row.get("top_k_doc_id", [])
            if use_baseline
            else row.get("top_k_doc_id", [])
        )

        context = build_context(doc_ids, corpus, max_docs, max_chars_per_doc)

        t0 = time.perf_counter()
        try:
            invoke_result = await Runner.run_agent(
                agent,
                {"query": question, "context": context},
            )
            output_result = invoke_result.get("output").result
            predicted = (output_result.get("response") or str(output_result)).strip()
        except Exception as e:
            predicted = ""
            tqdm.write(f"[qid={qid}] inference error: {e}")
        latencies.append(round((time.perf_counter() - t0) * 1000, 1))

        answers.append({
            "qid":          qid,
            "question":     question,
            "predicted":    predicted,
            "gold_answers": gold_answers,
        })

    # F1 / EM
    metrics_acc = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    for item in answers:
        if item["gold_answers"]:
            update_answer(metrics_acc, item["predicted"], item["gold_answers"])

    n = len(answers)
    qa_metrics = {k: round(v / n * 100, 2) for k, v in metrics_acc.items()} if n else {}
    latency_metrics = {
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "p50_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
    }

    return {**qa_metrics, **latency_metrics}, answers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MultihopRAG benchmark via openjiuwen WorkflowAgent + ContextPilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reordered_path", required=True)
    parser.add_argument("--corpus_path",    required=True)
    parser.add_argument("--num_queries",    type=int, default=100)
    parser.add_argument("--baseline",       action="store_true",
                        help="Use original doc order (no ContextPilot scheduling)")
    parser.add_argument("--max_docs",       type=int, default=10)
    parser.add_argument("--max_chars_per_doc", type=int, default=800)
    parser.add_argument("--output",         default=None,
                        help="Path to save per-query JSONL results")
    parser.add_argument("--model",          default=MODEL_NAME,
                        help=f"Model filename (default: {MODEL_NAME})")
    parser.add_argument("--llama-port",     type=int, default=LLAMA_PORT,
                        help=f"llama-server port (default: {LLAMA_PORT})")
    args = parser.parse_args()

    mode = "BASELINE (original order)" if args.baseline else "CONTEXTPILOT (reordered)"

    print("\n" + "=" * 60)
    print(" MultihopRAG Benchmark — openjiuwen WorkflowAgent")
    print("=" * 60)
    print(f" Mode:        {mode}")
    print(f" Queries:     {args.num_queries}")
    print(f" Max docs:    {args.max_docs}  (chars/doc: {args.max_chars_per_doc})")
    print(f" Model:       {args.model}")
    print(f" llama-server: http://localhost:{args.llama_port}")
    print()

    agent = build_agent(args.model, args.llama_port)

    corpus = load_corpus(args.corpus_path)
    rows   = load_reordered(args.reordered_path, args.num_queries)
    print(f"Loaded {len(rows)} queries\n")

    if not rows:
        print("No rows loaded — check --reordered_path")
        return

    print(f"Running inference ({mode})...")
    results, per_query = asyncio.run(run_inference(
        agent=agent,
        rows=rows,
        corpus=corpus,
        use_baseline=args.baseline,
        max_docs=args.max_docs,
        max_chars_per_doc=args.max_chars_per_doc,
    ))

    print("\n" + "=" * 60)
    print(f" Results — {mode}")
    print("=" * 60)
    for k, v in results.items():
        unit = "%" if k in ("em", "f1", "prec", "recall") else " ms" if "ms" in k else ""
        print(f"  {k:<25} {v}{unit}")
    print("=" * 60)
    print()
    print("Reference numbers from README (SGLang + Qwen3-32B on 4×A6000):")
    print("  Without ContextPilot:  cache_hit=4.64%  prefill_tps=7,290  F1=60.42")
    print("  With    ContextPilot:  cache_hit=33.97% prefill_tps=14,214 F1=64.39")

    if args.output:
        with open(args.output, "w") as f:
            for item in per_query:
                f.write(json.dumps(item) + "\n")
        print(f"\nPer-query results saved to {args.output}")


if __name__ == "__main__":
    main()
