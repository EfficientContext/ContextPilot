#!/usr/bin/env python3
"""
LoCoMo + mem0 + ContextPilot + SGLang benchmark.

Measures TTFT and answer accuracy (token-F1 + LLM judge) with and without
ContextPilot context reordering on a live SGLang server.

    OPENAI_API_KEY=... python examples/mem0_locomo_example.py
"""

import asyncio
import json
import os
import re
import time
import urllib.request
import uuid
from pathlib import Path

import aiohttp
import openai
import requests

from contextpilot.context_index import build_context_index
from contextpilot.context_ordering import InterContextScheduler
from contextpilot.retriever import Mem0Retriever
from contextpilot.utils.eval_metrics import eval_answer

SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:30000")
CONTEXTPILOT_URL = os.environ.get("CONTEXTPILOT_URL", "http://localhost:8765")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-2025-04-14")
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_CACHE = Path(__file__).resolve().parent.parent / "tests" / ".locomo_cache" / "locomo10.json"

CONV_INDEX = int(os.environ.get("LOCOMO_CONV_INDEX", "0"))
MAX_QA = int(os.environ.get("LOCOMO_MAX_QA", "50"))
MAX_GEN = int(os.environ.get("LOCOMO_MAX_TOKENS", "1024"))
NUM_TURNS = int(os.environ.get("LOCOMO_NUM_TURNS", "50"))
TOP_K_LIST = [int(k) for k in os.environ.get("LOCOMO_TOP_K", "20,50").split(",")]


# ---------------------------------------------------------------------------
# Reused helpers
# ---------------------------------------------------------------------------
async def _stream_ttft(prompt, model, max_tokens=512, rid=None):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True}
    if rid:
        payload["rid"] = rid
    result = {"ttft": 0.0, "text": "", "success": False}
    st = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.post(f"{SGLANG_URL}/v1/completions", json=payload) as resp:
            if resp.status != 200:
                result["error"] = await resp.text()
                return result
            ttft, text = 0.0, ""
            async for raw in resp.content:
                for line in raw.decode().split("\n"):
                    line = line.strip()
                    if line.startswith("data: "):
                        line = line[6:]
                    if not line or line == "[DONE]":
                        continue
                    try:
                        tok = json.loads(line)["choices"][0].get("text", "")
                        if tok and ttft == 0.0:
                            ttft = time.perf_counter() - st
                        text += tok
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
    result.update(ttft=ttft, text=text, success=True)
    return result


def run_ttft(prompt, model, max_tokens=512, rid=None):
    return asyncio.run(_stream_ttft(prompt, model, max_tokens, rid=rid))


def build_prompt(question, doc_ids, corpus_map, history=None):
    docs = [corpus_map.get(str(d), {}).get("text", f"[memory {d}]") for d in doc_ids]
    ctx = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
    parts = [f"Memories:\n{ctx}"]
    if history:
        parts.append("Conversation so far:\n" + "\n".join(history))
    parts.append(f"Based on the memories above, concisely answer the following "
                 f"question in as few words as possible.\nQuestion: {question}\nAnswer:")
    return "\n\n".join(parts)


def llm_judge(question, prediction, ground_truth):
    try:
        resp = openai.OpenAI().chat.completions.create(
            model=JUDGE_MODEL, temperature=0, max_tokens=150,
            messages=[
                {"role": "system", "content":
                    "You are an answer evaluator. Given a question, a predicted answer, "
                    "and the ground truth answer, judge whether the prediction is correct.\n"
                    'Output JSON: {"score": <0.0-1.0>, "reason": "<brief reason>"}\n'
                    "Score 1.0 = fully correct, 0.5 = partially correct, 0.0 = wrong.\n"
                    "Be lenient with phrasing differences if the meaning matches."},
                {"role": "user", "content":
                    f"Question: {question}\nPredicted: {prediction}\nGround truth: {ground_truth}"},
            ],
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        return float(parsed.get("score", 0)), parsed.get("reason", "")
    except Exception as e:
        return -1.0, f"judge error: {e}"


def cp_build_index(contexts, initial_tokens_per_context=0, incremental=False):
    r = requests.post(f"{CONTEXTPILOT_URL}/build", json={
        "contexts": contexts, "use_gpu": False, "linkage_method": "average",
        "alpha": 0.005, "initial_tokens_per_context": initial_tokens_per_context,
        "incremental": incremental,
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def strip_thinking(text):
    if not text.lstrip().startswith("<think>"):
        return text
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return stripped or text


def run_benchmark(results, request_ids, label, corpus_map, qa_pairs, model, max_gen):
    outputs = []
    for i, r in enumerate(results):
        p = build_prompt(r["text"], r["top_k_doc_id"], corpus_map)
        rid = request_ids[i] if i < len(request_ids) else None
        out = run_ttft(p, model, max_gen, rid=rid)
        qid = r.get("qid", i)
        qa = qa_pairs[qid]
        gt = str(qa["answer"])

        if out["success"] and out["text"]:
            answer = strip_thinking(out["text"])
            _, f1, _, _ = eval_answer(answer, gt)
            score, _ = llm_judge(qa["question"], answer, gt)
            out.update(f1=f1, judge_score=score)
            if i < 5:
                print(f"  Q{qid}: {qa['question'][:80]}")
                print(f"    Pred: {answer[:150]}  |  Gold: {gt}")
                print(f"    F1={f1:.3f} Judge={score:.1f}")
        else:
            out.update(f1=0.0, judge_score=-1)
        outputs.append(out)

    ok = [o for o in outputs if o["success"]]
    if not ok:
        print(f"\n[{label}] all failed")
        return {"ttft": 0, "f1": 0, "judge": 0, "n": 0}
    avg_ttft = sum(o["ttft"] for o in ok) / len(ok)
    avg_f1 = sum(o["f1"] for o in ok) / len(ok)
    js = [o["judge_score"] for o in ok if o["judge_score"] >= 0]
    avg_j = sum(js) / len(js) if js else 0
    print(f"\n[{label}] {len(ok)}/{len(outputs)} ok  "
          f"TTFT={avg_ttft:.4f}s  F1={avg_f1:.4f}  Judge={avg_j:.3f}")
    return {"ttft": avg_ttft, "f1": avg_f1, "judge": avg_j, "n": len(ok)}
if __name__ == "__main__":
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
    assert requests.get(f"{SGLANG_URL}/health", timeout=3).status_code == 200, \
        f"SGLang not reachable at {SGLANG_URL}"

    model = requests.get(f"{SGLANG_URL}/v1/models", timeout=5).json()["data"][0]["id"]
    print(f"SGLang model: {model}")

    try:
        cp_available = requests.get(f"{CONTEXTPILOT_URL}/health", timeout=3).status_code in (200, 503)
    except Exception:
        cp_available = False
    print(f"ContextPilot: {'available' if cp_available else 'unavailable'}")

    if not LOCOMO_CACHE.exists():
        print(f"Downloading LoCoMo data -> {LOCOMO_CACHE}")
        LOCOMO_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LOCOMO_URL, LOCOMO_CACHE)

    conv = json.loads(LOCOMO_CACHE.read_text())[CONV_INDEX]
    qa_pairs = conv["qa"][:MAX_QA]
    conv = conv["conversation"]
    print(f"LoCoMo conv {CONV_INDEX}: {conv['speaker_a']} & {conv['speaker_b']}, {len(qa_pairs)} QA pairs")

    user_id = f"locomo_{uuid.uuid4().hex[:8]}"
    retriever = Mem0Retriever(config={
        "llm": {"provider": "openai", "config": {"model": "gpt-4.1-mini-2025-04-14"}},
        "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    })

    n = 1
    while f"session_{n}" in conv:
        turns = conv[f"session_{n}"]
        dt = conv.get(f"session_{n}_date_time", "")
        msgs = [{"role": "user" if t["speaker"] == conv["speaker_a"] else "assistant",
                 "content": t["text"]} for t in turns]
        retriever.add_memory(msgs, user_id=user_id)
        print(f"  session {n} ({len(turns)} turns, {dt})")
        n += 1
    print("Waiting for mem0 indexing ...")
    time.sleep(5)

    for _ in range(3):
        run_ttft("Hello, world.", model, max_tokens=4)
    print("Warmup done.\n")

    query_data = [{"qid": i, "text": qa["question"]} for i, qa in enumerate(qa_pairs)]
    rows = []
    bench = lambda res, rids, label, cmap: run_benchmark(res, rids, label, cmap, qa_pairs, model, MAX_GEN)

    try:
        for top_k in TOP_K_LIST:
            print(f"{'#'*60}\n## top_k={top_k}\n{'#'*60}")

            search_results = retriever.search_queries(
                query_data=query_data, user_id=user_id, top_k=top_k)
            corpus_map = retriever.get_corpus_map()

            base = bench(search_results, [], f"baseline k={top_k}", corpus_map)

            contexts = [r["top_k_doc_id"] for r in search_results]
            avg_tok = sum(len(v.get("text", "")) // 4 for v in corpus_map.values()) // max(len(corpus_map), 1)

            if cp_available and len(contexts) >= 2:
                requests.post(f"{CONTEXTPILOT_URL}/reset", timeout=5)
                build_resp = cp_build_index(contexts, initial_tokens_per_context=avg_tok * top_k)
                reordered = build_resp.get("reordered_contexts")
                order = build_resp.get("scheduled_order")
                request_ids = build_resp.get("request_ids", [])
                if reordered and order:
                    opt_results = [dict(search_results[idx], top_k_doc_id=reordered[i])
                                   for i, idx in enumerate(order)]
                else:
                    opt_results, request_ids = search_results, []
            else:
                ci = build_context_index(contexts)
                reordered_lib, _, mapping, _ = InterContextScheduler().schedule_contexts(ci)
                opt_results = [dict(search_results[idx], top_k_doc_id=reordered_lib[i])
                               for i, idx in enumerate(mapping)]
                request_ids = []

            opt = bench(opt_results, request_ids, f"optimized k={top_k}", corpus_map)

            print(f"\nMulti-turn ({NUM_TURNS} turns):")
            history, prev_doc_ids = [], []
            if cp_available:
                requests.post(f"{CONTEXTPILOT_URL}/reset", timeout=5)

            for idx, qa in enumerate(qa_pairs[:NUM_TURNS]):
                s = retriever.search_queries(
                    query_data=[{"qid": idx, "text": qa["question"]}],
                    user_id=user_id, top_k=top_k)
                cmap = retriever.get_corpus_map()
                doc_ids = s[0]["top_k_doc_id"]
                ctx_tokens = sum(len(cmap.get(str(d), {}).get("text", "")) // 4 for d in doc_ids)

                rid, reordered_ids = None, doc_ids
                if cp_available:
                    try:
                        br = cp_build_index([doc_ids], initial_tokens_per_context=ctx_tokens,
                                            incremental=idx > 0)
                        rids = br.get("request_ids", [])
                        rid = rids[0] if rids else None
                        rc = br.get("reordered_contexts")
                        reordered_ids = rc[0] if rc else doc_ids
                    except Exception:
                        pass
                elif len(prev_doc_ids) >= 1:
                    all_ctx = prev_doc_ids + [doc_ids]
                    ci = build_context_index(all_ctx)
                    reordered, _, mapping, _ = InterContextScheduler().schedule_contexts(ci)
                    pos = mapping.index(len(all_ctx) - 1)
                    reordered_ids = reordered[pos]

                prior_set = {d for ids in prev_doc_ids for d in ids}
                overlap = len(set(reordered_ids) & prior_set)

                prompt = build_prompt(qa["question"], reordered_ids, cmap, history[-4:])
                out = run_ttft(prompt, model, MAX_GEN, rid=rid)
                print(f"  T{idx:>2d}: TTFT={out['ttft']:.4f}s  overlap={overlap}/{len(reordered_ids)}")

                prev_doc_ids.append(doc_ids)
                history.append(f"Q: {qa['question']}")
                if out["success"] and out["text"]:
                    history.append(f"A: {out['text'][:200]}")

            delta = (base["ttft"] - opt["ttft"]) / base["ttft"] * 100 if base["ttft"] else 0
            rows.append((top_k, base, opt, delta))

        print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
        print(f"  {'k':>5s}  |  {'Base TTFT':>10s} {'F1':>6s} {'Judge':>6s}  |  "
              f"{'Opt TTFT':>10s} {'F1':>6s} {'Judge':>6s}  |  {'delta':>6s}")
        for k, b, o, d in rows:
            print(f"  k={k:>3d}  |  {b['ttft']:.4f}s  {b['f1']:.3f}  {b['judge']:.3f}  |  "
                  f"{o['ttft']:.4f}s  {o['f1']:.3f}  {o['judge']:.3f}  |  {d:+.1f}%")

        try:
            cp = requests.get(f"{CONTEXTPILOT_URL}/stats", timeout=5).json().get("index_stats", {})
            if cp:
                print(f"\n  ContextPilot: {cp.get('num_requests',0)} reqs, "
                      f"{cp.get('total_tokens',0)} tokens, "
                      f"{cp.get('avg_search_time_us',0):.0f}us avg search")
        except Exception:
            pass

    finally:
        try:
            retriever.delete_all_memories(user_id=user_id)
            print(f"\nCleaned up memories for {user_id}")
        except Exception as e:
            print(f"\nCleanup warning: {e}")
