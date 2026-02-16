import asyncio, json, os, re, time, urllib.request, uuid
os.environ["TQDM_DISABLE"] = "1"
from pathlib import Path

import aiohttp, openai, requests

from contextpilot.retriever import Mem0Retriever
from contextpilot.utils.eval_metrics import eval_answer

SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:30000")
CONTEXTPILOT_URL = os.environ.get("CONTEXTPILOT_URL", "http://localhost:8765")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-2025-04-14")
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_CACHE = Path(__file__).resolve().parent.parent / "tests" / ".locomo_cache" / "locomo10.json"

CONV_INDEX = int(os.environ.get("LOCOMO_CONV_INDEX", "0"))
MAX_QA = int(os.environ.get("LOCOMO_MAX_QA", "150"))
MAX_GEN = int(os.environ.get("LOCOMO_MAX_TOKENS", "32"))
NUM_TURNS = int(os.environ.get("LOCOMO_NUM_TURNS", "150"))
TOP_K_LIST = os.environ.get("LOCOMO_TOP_K_LIST", "20,100")


async def _stream_ttft(prompt, model, max_tokens=512):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True}
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


def run_ttft(prompt, model, max_tokens=512):
    return asyncio.run(_stream_ttft(prompt, model, max_tokens))


def build_prompt(question, context_str):
    return (f"Memories:\n{context_str}\n"
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


def cp_build(contexts, incremental=False):
    r = requests.post(f"{CONTEXTPILOT_URL}/build", json={
        "contexts": contexts, "use_gpu": False, "linkage_method": "average",
        "alpha": 0.0005,
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def cp_reset():
    try:
        requests.post(f"{CONTEXTPILOT_URL}/reset", timeout=5)
    except Exception:
        pass


def strip_thinking(text):
    if not text.lstrip().startswith("<think>"):
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() or text


def build_context_str(doc_ids, corpus_map):
    """Build context string from doc IDs and corpus map."""
    parts = []
    for did in doc_ids:
        entry = corpus_map.get(str(did), {})
        text = entry.get("text", entry.get("content", f"[doc {did}]"))
        parts.append(text)
    return "\n\n".join(parts)


def run_multi_turn(retriever, user_id, qa_pairs, model, top_k,
                   use_reorder=False, cp_available=False):
    """Run multi-turn benchmark: baseline vs reorder.

    Args:
        use_reorder: Reorder docs via ContextPilot /build for prefix sharing.
    """
    label = "reorder" if use_reorder else "baseline"
    print(f"\n--- {label} ({NUM_TURNS} turns, k={top_k}) ---")

    prev_reordered = []
    ttfts, prefix_matches, f1s, judges = [], [], [], []

    for idx in range(min(NUM_TURNS, len(qa_pairs))):
        qa = qa_pairs[idx % len(qa_pairs)]
        cat = qa.get("category", 0)

        # Retrieve from mem0
        s = retriever.search_queries(
            query_data=[{"qid": idx, "text": qa["question"]}],
            user_id=user_id, top_k=top_k)
        cmap = retriever.get_corpus_map()
        doc_ids = s[0]["top_k_doc_id"]

        reordered_ids = doc_ids

        # Reorder via ContextPilot /build
        if use_reorder and cp_available:
            try:
                incremental = idx > 0  # first turn: initial build, rest: incremental
                br = cp_build([doc_ids], incremental=incremental)
                if br.get("reordered_contexts"):
                    reordered_ids = br["reordered_contexts"][0]
                if idx < 5:
                    print(f"    /build mode={br.get('mode')} matched={br.get('matched_count')}")
            except Exception as e:
                print(f"    /build FAILED: {e}")

        # Build context string directly from corpus map
        context_str = build_context_str(reordered_ids, cmap)

        # Prefix match (consecutive matching doc IDs from position 0)
        prefix_match = 0
        if prev_reordered:
            for a, b in zip(reordered_ids, prev_reordered):
                if a != b:
                    break
                prefix_match += 1

        # Build prompt and measure TTFT
        prompt = build_prompt(qa["question"], context_str)
        out = run_ttft(prompt, model, MAX_GEN)
        gt = str(qa["answer"])

        if idx > 0:
            ttfts.append(out["ttft"])
            prefix_matches.append(prefix_match / len(reordered_ids) if reordered_ids else 0)

        # Score answer
        f1, score = 0.0, -1.0
        if out["success"] and out["text"]:
            answer = strip_thinking(out["text"])
            _, f1, _, _ = eval_answer(answer, gt)
            score, _ = llm_judge(qa["question"], answer, gt)
            f1s.append(f1)
            if score >= 0:
                judges.append(score)

        if idx < 5:
            print(f"  Q{idx}: {qa['question']}")
            print(f"    original:  {doc_ids}")
            print(f"    reordered: {reordered_ids}")
            if prev_reordered:
                print(f"    prev:      {prev_reordered}")
            print(f"    prefix_match={prefix_match}/{len(reordered_ids)}"
                  f" ttft={out['ttft']:.4f}s f1={f1:.3f} judge={score:.1f}")

        prev_reordered = reordered_ids

    avg = lambda xs: sum(xs) / len(xs) if xs else 0
    stats = {
        "label": label,
        "ttft": avg(ttfts),
        "prefix": avg(prefix_matches),
        "f1": avg(f1s),
        "judge": avg(judges),
    }
    print(f"  [{label}] TTFT={stats['ttft']:.4f}s  Prefix={stats['prefix']:.1%}"
          f"  F1={stats['f1']:.3f}  Judge={stats['judge']:.3f}")
    return stats


def ingest_conversation(conv_data, retriever, user_id):
    conv = conv_data["conversation"]
    n, total_turns = 1, 0
    while f"session_{n}" in conv:
        turns = conv[f"session_{n}"]
        for t in turns:
            role = "user" if t["speaker"] == conv["speaker_a"] else "assistant"
            retriever.add_memory(t["text"], user_id=user_id)
            total_turns += 1
        n += 1
    print(f"  ingested {total_turns} turns from {n-1} sessions, waiting for indexing ...")
    time.sleep(5)
    all_memories = retriever.memory.get_all(user_id=user_id)
    n_memories = len(all_memories.get("results", []))
    print(f"  {n_memories} memories stored")
    return n_memories


if __name__ == "__main__":
    import pandas as pd

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

    # Load LoCoMo
    if not LOCOMO_CACHE.exists():
        print(f"Downloading LoCoMo data -> {LOCOMO_CACHE}")
        LOCOMO_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LOCOMO_URL, LOCOMO_CACHE)
    all_convs = json.loads(LOCOMO_CACHE.read_text())

    # Warmup SGLang
    for _ in range(3):
        run_ttft("Hello, world.", model, max_tokens=4)
    print("Warmup done.\n")

    retriever = Mem0Retriever(config={
        "llm": {"provider": "openai", "config": {"model": "gpt-4.1-mini-2025-04-14"}},
        "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    })

    conv_data = all_convs[CONV_INDEX]
    qa_pairs = conv_data["qa"][:MAX_QA]
    conv = conv_data["conversation"]
    print(f"\n{'='*70}")
    print(f"CONV {CONV_INDEX}: {conv['speaker_a']} & {conv['speaker_b']}, {len(qa_pairs)} QA pairs")
    print(f"{'='*70}")

    user_id = f"locomo_{CONV_INDEX}_{uuid.uuid4().hex[:6]}"
    n_memories = ingest_conversation(conv_data, retriever, user_id)
    top_k_values = [int(k) for k in TOP_K_LIST.split(",")]

    try:
        all_rows = []
        for top_k in top_k_values:
            print(f"\n## top_k={top_k}")
            results = {}
            for use_reorder in [True, False]:
                cp_reset()  # fresh tree for each mode
                stats = run_multi_turn(
                    retriever, user_id, qa_pairs, model, top_k,
                    use_reorder=use_reorder, cp_available=cp_available)
                results[stats["label"]] = stats

            base_ttft = results["baseline"]["ttft"]

            for name in ["baseline", "reorder"]:
                s = results[name]
                delta = (base_ttft - s["ttft"]) / base_ttft * 100 if base_ttft else 0
                all_rows.append({
                    "k": top_k,
                    "mode": name,
                    "ttft": f"{s['ttft']:.4f}s",
                    "ttft_delta": f"{delta:+.1f}%" if name != "baseline" else "-",
                    "prefix": f"{s['prefix']:.1%}",
                    "f1": f"{s['f1']:.3f}",
                    "judge": f"{s['judge']:.3f}",
                })

        # Summary table
        print(f"\n{'='*70}")
        print(f"RESULTS (conv={CONV_INDEX}, memories={n_memories}, turns={min(NUM_TURNS, len(qa_pairs))})")
        print(f"{'='*70}")
        print(pd.DataFrame(all_rows).to_string(index=False))

    finally:
        try:
            retriever.delete_all_memories(user_id=user_id)
            print(f"\nCleaned up memories for {user_id}")
        except Exception as e:
            print(f"\nCleanup warning: {e}")
        del retriever
        import gc; gc.collect()
