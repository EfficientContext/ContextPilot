import asyncio, json, os, re, time, urllib.request, uuid
os.environ["TQDM_DISABLE"] = "1"
from pathlib import Path

import aiohttp, openai, requests

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
MAX_GEN = int(os.environ.get("LOCOMO_MAX_TOKENS", "32"))
NUM_TURNS = int(os.environ.get("LOCOMO_NUM_TURNS", "200"))
TOP_K_LIST = [int(k) for k in os.environ.get("LOCOMO_TOP_K", "20,50").split(",")]
BENCH_MODE = os.environ.get("BENCH_MODE", "both")  # baseline, optimized, both
BENCH_USER_ID = os.environ.get("BENCH_USER_ID", "")


async def _stream_ttft(prompt, model, max_tokens=512, rid=None):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True}
    if rid:
        payload["rid"] = rid
    result = {"ttft": 0.0, "text": "", "success": False}
    st = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as sess:
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


def build_prompt(question, doc_ids, corpus_map):
    docs = [corpus_map.get(str(d), {}).get("text", f"[memory {d}]") for d in doc_ids]
    ctx = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
    return (f"Memories:\n{ctx}\n\n"
            f"Based on the memories above, concisely answer the following "
            f"question in as few words as possible.\nQuestion: {question}\nAnswer:")


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
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() or text


def run_multi_turn(retriever, user_id, qa_pairs, model, top_k, optimize, cp_available):
    """Single multi-turn run. Returns dict with avg ttft/f1/judge."""
    label = "contextpilot" if optimize else "baseline"
    print(f"\n--- {label} multi-turn ({NUM_TURNS} turns) ---")

    prev_doc_ids, prev_reordered = [], []
    ttfts, f1s, judges, prefix_matches = [], [], [], []
    if cp_available and optimize:
        requests.post(f"{CONTEXTPILOT_URL}/reset", timeout=5)

    for idx in range(min(NUM_TURNS, len(qa_pairs))):
        qa = qa_pairs[idx % len(qa_pairs)]
        # Retrieve from mem0
        s = retriever.search_queries(
            query_data=[{"qid": idx, "text": qa["question"]}],
            user_id=user_id, top_k=top_k)
        cmap = retriever.get_corpus_map()
        doc_ids = s[0]["top_k_doc_id"]

        # Reorder via ContextPilot (optimized only)
        rid, reordered_ids = None, doc_ids
        if optimize:
            ctx_tokens = sum(len(cmap.get(str(d), {}).get("text", "")) // 4 for d in doc_ids)
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
                reordered_ids = reordered[mapping.index(len(all_ctx) - 1)]

        # Count consecutive matching docs from position 0 (prefix overlap)
        prefix_match = 0
        if prev_reordered:
            for a, b in zip(reordered_ids, prev_reordered):
                if a != b:
                    break
                prefix_match += 1

        # Build prompt and measure TTFT
        prompt = build_prompt(qa["question"], reordered_ids, cmap)
        out = run_ttft(prompt, model, MAX_GEN, rid=rid)
        gt = str(qa["answer"])
        # Skip turn 0 — no prior context, so baseline and optimized are identical
        if idx > 0:
            ttfts.append(out["ttft"])
            prefix_matches.append(prefix_match / len(reordered_ids) if reordered_ids else 0)

        # Score answer
        if out["success"] and out["text"]:
            answer = strip_thinking(out["text"])
            _, f1, _, _ = eval_answer(answer, gt)
            score, _ = llm_judge(qa["question"], answer, gt)
            f1s.append(f1)
            if score >= 0:
                judges.append(score)
            if idx < 5:
                print(f"Q{idx}: {qa['question'][:80]}")
                print(f"  original:  {doc_ids}")
                print(f"  reordered: {reordered_ids}")
                print(f"Pred: {answer[:500]}")
                print(f"Ground Truth: {gt}")
                print(f"F1={f1:.3f} Judge={score:.1f}")

        print(f"  T{idx:>2d}: TTFT={out['ttft']:.4f}s  prefix={prefix_match}/{len(reordered_ids)}")

        prev_doc_ids.append(doc_ids)
        prev_reordered = reordered_ids

    stats = {
        "ttft": sum(ttfts) / len(ttfts) if ttfts else 0,
        "f1": sum(f1s) / len(f1s) if f1s else 0,
        "judge": sum(judges) / len(judges) if judges else 0,
        "prefix": sum(prefix_matches) / len(prefix_matches) if prefix_matches else 0,
    }
    print(f"\n[{label} k={top_k}] TTFT={stats['ttft']:.4f}s  F1={stats['f1']:.4f}  "
          f"Judge={stats['judge']:.3f}  Prefix={stats['prefix']:.1%}")
    return stats


if __name__ == "__main__":
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
    assert requests.get(f"{SGLANG_URL}/health", timeout=3).status_code == 200, \
        f"SGLang not reachable at {SGLANG_URL}"

    model = requests.get(f"{SGLANG_URL}/v1/models", timeout=5).json()["data"][0]["id"]
    print(f"SGLang model: {model}  mode: {BENCH_MODE}")

    try:
        cp_available = requests.get(f"{CONTEXTPILOT_URL}/health", timeout=3).status_code in (200, 503)
    except Exception:
        cp_available = False
    print(f"ContextPilot: {'available' if cp_available else 'unavailable'}")

    # Load LoCoMo
    if not LOCOMO_CACHE.exists():
        print(f"Downloading LoCoMo data -> {LOCOMO_CACHE}")
        LOCOMO_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LOCOMO_URL, LOCOMO_CACHE)
    conv = json.loads(LOCOMO_CACHE.read_text())[CONV_INDEX]
    qa_pairs = conv["qa"][:MAX_QA]
    conv = conv["conversation"]
    print(f"LoCoMo conv {CONV_INDEX}: {conv['speaker_a']} & {conv['speaker_b']}, {len(qa_pairs)} QA pairs")

    # Ingest into mem0 (or reuse existing)
    user_id = BENCH_USER_ID or f"locomo_{uuid.uuid4().hex[:8]}"
    retriever = Mem0Retriever(config={
        "llm": {"provider": "openai", "config": {"model": "gpt-4.1-mini-2025-04-14"}},
        "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    })
    if not BENCH_USER_ID:
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
    print(f"mem0 user_id: {user_id}")

    # Warmup SGLang
    for _ in range(3):
        run_ttft("Hello, world.", model, max_tokens=4)
    print("Warmup done.\n")

    # Benchmark: baseline vs contextpilot, multi-turn for each top_k
    rows = []
    try:
        for top_k in TOP_K_LIST:
            print(f"{'#'*60}\n## top_k={top_k}\n{'#'*60}")

            # Baseline: no reordering
            base = run_multi_turn(retriever, user_id, qa_pairs, model, top_k,
                                  optimize=False, cp_available=cp_available)
            # ContextPilot: reorder for prefix sharing
            opt = run_multi_turn(retriever, user_id, qa_pairs, model, top_k,
                                 optimize=True, cp_available=cp_available)

            delta = (base["ttft"] - opt["ttft"]) / base["ttft"] * 100 if base["ttft"] else 0
            rows.append((top_k, base, opt, delta))

        # Summary
        import pandas as pd
        df = pd.DataFrame([{
            "top_k": k,
            "base_ttft": f"{b['ttft']:.4f}s",
            "base_f1": f"{b['f1']:.3f}",
            "base_judge": f"{b['judge']:.3f}",
            "opt_ttft": f"{o['ttft']:.4f}s",
            "opt_f1": f"{o['f1']:.3f}",
            "opt_judge": f"{o['judge']:.3f}",
            "ttft_delta": f"{d:+.1f}%",
            "base_prefix": f"{b['prefix']:.1%}",
            "opt_prefix": f"{o['prefix']:.1%}",
        } for k, b, o, d in rows])
        print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
        print(df.to_string(index=False))

        try:
            cp = requests.get(f"{CONTEXTPILOT_URL}/stats", timeout=5).json().get("index_stats", {})
            if cp:
                print(f"\nContextPilot: {cp.get('num_requests',0)} reqs, "
                      f"{cp.get('total_tokens',0)} tokens, "
                      f"{cp.get('avg_search_time_us',0):.0f}us avg search")
        except Exception:
            pass

    finally:
        if not BENCH_USER_ID:
            try:
                retriever.delete_all_memories(user_id=user_id)
                print(f"\nCleaned up memories for {user_id}")
            except Exception as e:
                print(f"\nCleanup warning: {e}")
