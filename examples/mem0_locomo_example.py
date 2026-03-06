import asyncio, json, os, re, time, urllib.request, uuid
os.environ["TQDM_DISABLE"] = "1"
from pathlib import Path

import aiohttp, openai, requests

from contextpilot.retriever import Mem0Retriever
from contextpilot.utils.eval_metrics import eval_answer

INFERENCE_URL = os.environ.get("INFERENCE_URL", os.environ.get("SGLANG_URL", "http://localhost:30000"))
CONTEXTPILOT_URL = os.environ.get("CONTEXTPILOT_URL", "http://localhost:8765")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-2025-04-14")
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_CACHE = Path(__file__).resolve().parent.parent / "tests" / ".locomo_cache" / "locomo10.json"

CONV_INDEX = os.environ.get("LOCOMO_CONV_INDEX", "all")
MAX_QA = int(os.environ.get("LOCOMO_MAX_QA", "150"))
MAX_GEN = int(os.environ.get("LOCOMO_MAX_TOKENS", "32"))
NUM_TURNS = int(os.environ.get("LOCOMO_NUM_TURNS", "150"))
TOP_K_LIST = os.environ.get("LOCOMO_TOP_K_LIST", "20,50,5x10")


async def _stream_ttft(prompt, model, max_tokens=512, request_id=None):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True}
    if request_id:
        payload["rid"] = request_id          # SGLang
        payload["request_id"] = request_id   # vLLM
    result = {"ttft": 0.0, "text": "", "success": False}
    st = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as sess:
        async with sess.post(f"{INFERENCE_URL}/v1/completions", json=payload) as resp:
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


def run_ttft(prompt, model, max_tokens=512, request_id=None):
    return asyncio.run(_stream_ttft(prompt, model, max_tokens, request_id))


def build_prompt(question, context_str, importance_ranking=None):
    prompt = (f"Memories:\n{context_str}\n"
              f"Based on the memories above, concisely answer the following "
              f"question in as few words as possible.\n")
    if importance_ranking:
        prompt += (f"Please read the documents in the following importance ranking:\n"
                   f"{importance_ranking}\n"
                   f"Prioritize information from higher-ranked documents.\n")
    prompt += f"Question: {question}\nAnswer:"
    return prompt


def build_importance_ranking(original_ids, reordered_ids):
    """Map original retrieval order to positions in the reordered doc list.

    With repeated docs the same doc_id appears multiple times, so we track
    the *first* occurrence of each unique doc in the original order and map
    it to its first position in the reordered list.
    """
    # First occurrence of each doc in reordered list -> its [Doc_N] position
    pos = {}
    for i, did in enumerate(reordered_ids):
        if did not in pos:
            pos[did] = i + 1
    # Deduplicate original_ids while preserving order
    seen = set()
    unique_original = []
    for did in original_ids:
        if did not in seen:
            seen.add(did)
            unique_original.append(did)
    return " > ".join(f"[Doc_{pos[did]}]" for did in unique_original if did in pos)


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
    r = requests.post(f"{CONTEXTPILOT_URL}/reorder", json={
        "contexts": contexts, "use_gpu": False, "linkage_method": "average",
        "alpha": 0.0005,
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def cp_search(context):
    r = requests.post(
        f"{CONTEXTPILOT_URL}/search",
        json={"context": context, "update_access": False},
        timeout=10,
    )
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
    parts = []
    for i, did in enumerate(doc_ids):
        entry = corpus_map.get(str(did), {})
        text = entry.get("text", entry.get("content", f"[doc {did}]"))
        parts.append(f"[Doc_{i+1}] {text}")
    return "\n\n".join(parts)


def run_multi_turn(retriever, user_id, qa_pairs, model, top_k,
                   use_reorder=False, cp_available=False, repeat_times=1):
    """Run multi-turn benchmark: baseline vs reorder."""
    label = "reorder" if use_reorder else "baseline"
    actual_k = top_k * repeat_times if repeat_times > 1 else top_k
    suffix = f" (k={top_k}x{repeat_times}={actual_k} docs)" if repeat_times > 1 else f" (k={top_k})"
    print(f"\n--- {label} ({NUM_TURNS} turns,{suffix}) ---")

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

        # Repeat docs to create long context if requested
        if repeat_times > 1:
            doc_ids = doc_ids * repeat_times

        original_ids = list(doc_ids)  # preserve original retrieval order
        reordered_ids = doc_ids
        req_id = None
        server_prefix_len, server_has_prefix, server_node_id = 0, False, -1

        # Query server's prefix match (same logic used by incremental build search).
        # Only valid for turn>=1, since turn 0 has no existing index.
        if use_reorder and cp_available and idx > 0:
            try:
                sr = cp_search(doc_ids)
                server_prefix_len = int(sr.get("prefix_length", 0) or 0)
                server_has_prefix = bool(sr.get("has_prefix", False))
                server_node_id = int(sr.get("node_id", -1) or -1)
            except Exception as e:
                if idx < 5:
                    print(f"    /search FAILED: {e}")

        # Reorder via ContextPilot /reorder
        if use_reorder and cp_available:
            try:
                incremental = idx > 0  # first turn: initial build, rest: incremental
                br = cp_build([doc_ids], incremental=incremental)
                if br.get("reordered_contexts"):
                    reordered_ids = br["reordered_contexts"][0]
                if br.get("request_ids"):
                    req_id = br["request_ids"][0]
                if idx < 5:
                    print(f"    /reorder mode={br.get('mode')} matched={br.get('matched_count')}")
            except Exception as e:
                print(f"    /reorder FAILED: {e}")

        # Build context string directly from corpus map
        context_str = build_context_str(reordered_ids, cmap)

        # Build importance ranking — always include so prompt length is equal
        # between baseline and reorder (fair TTFT comparison).
        # Baseline: natural order [Doc_1] > [Doc_2] > ...
        # Reorder:  original retrieval order mapped to reordered positions
        if use_reorder and reordered_ids != original_ids:
            importance_ranking = build_importance_ranking(original_ids, reordered_ids)
        else:
            importance_ranking = " > ".join(f"[Doc_{i+1}]" for i in range(len(reordered_ids)))

        # Build prompt and measure TTFT
        prompt = build_prompt(qa["question"], context_str, importance_ranking)
        out = run_ttft(prompt, model, MAX_GEN, request_id=req_id)
        gt = str(qa.get("answer", qa.get("answers", qa.get("gold_answer", ""))))

        if idx > 0:
            ttfts.append(out["ttft"])
            if use_reorder and cp_available:
                prefix_matches.append(server_prefix_len / len(doc_ids) if doc_ids else 0)
            else:
                prefix_matches.append(0.0)

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
            if use_reorder and cp_available and idx > 0:
                print(
                    f"    server_prefix={server_prefix_len}/{len(doc_ids)} "
                    f"has_prefix={server_has_prefix} node={server_node_id}"
                )
            print(f"    ttft={out['ttft']:.4f}s f1={f1:.3f} judge={score:.1f}")

    avg = lambda xs: sum(xs) / len(xs) if xs else 0
    stats = {
        "label": label,
        "ttft": avg(ttfts),
        "prefix": avg(prefix_matches),
        "f1": avg(f1s),
        "judge": avg(judges),
        "repeat": repeat_times,
    }
    print(f"  [{label}] TTFT={stats['ttft']:.4f}s  Prefix={stats['prefix']:.1%}"
          f"  F1={stats['f1']:.3f}  Judge={stats['judge']:.3f}")
    return stats


def ingest_conversation(conv_data, retriever, user_id):
    conv = conv_data["conversation"]
    n, total_turns, add_calls = 1, 0, 0
    while f"session_{n}" in conv:
        turns = conv[f"session_{n}"]
        messages = []
        for t in turns:
            role = "user" if t["speaker"] == conv["speaker_a"] else "assistant"
            messages.append({"role": role, "content": t["text"]})
            total_turns += 1
        if messages:
            # Let mem0's extraction/update pipeline compact overlapping facts.
            retriever.add_memory(
                messages,
                user_id=user_id,
                metadata={"source": "locomo", "session": n},
            )
            add_calls += 1
        n += 1
    print(
        f"  ingested {total_turns} turns from {n-1} sessions "
        f"via {add_calls} mem0 add() calls, waiting for indexing ..."
    )
    time.sleep(5)
    all_memories = retriever.memory.get_all(user_id=user_id, limit=5000)
    n_memories = len(all_memories.get("results", []))
    print(f"  {n_memories} memories stored")
    return n_memories


if __name__ == "__main__":
    import pandas as pd

    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set"
    assert requests.get(f"{INFERENCE_URL}/health", timeout=3).status_code == 200, \
        f"Inference engine not reachable at {INFERENCE_URL}"

    model = requests.get(f"{INFERENCE_URL}/v1/models", timeout=5).json()["data"][0]["id"]
    print(f"Model: {model}")

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

    # Warmup inference engine
    for _ in range(3):
        run_ttft("Hello, world.", model, max_tokens=4)
    print("Warmup done.\n")

    # Parse TOP_K_LIST: supports "20", "50", or "5x10" (k=5, repeat 10 times)
    top_k_configs = []
    for entry in TOP_K_LIST.split(","):
        entry = entry.strip()
        if "x" in entry:
            k_str, r_str = entry.split("x", 1)
            top_k_configs.append((int(k_str), int(r_str)))
        else:
            top_k_configs.append((int(entry), 1))

    # Determine which conversations to run
    if CONV_INDEX == "all":
        conv_indices = list(range(len(all_convs)))
    else:
        conv_indices = [int(CONV_INDEX)]

    grand_rows = []  # aggregate across all conversations

    for ci in conv_indices:
        # Flush SGLang's radix cache between conversations to avoid pressure buildup
        try:
            requests.post(f"{INFERENCE_URL}/flush_cache", timeout=5)
        except Exception:
            pass

        conv_data = all_convs[ci]
        qa_pairs = conv_data["qa"][:MAX_QA]
        conv = conv_data["conversation"]
        print(f"\n{'='*70}")
        print(f"CONV {ci}: {conv['speaker_a']} & {conv['speaker_b']}, {len(qa_pairs)} QA pairs")
        print(f"{'='*70}")

        retriever = Mem0Retriever(config={
            "llm": {"provider": "openai", "config": {"model": "gpt-4.1-mini-2025-04-14"}},
            "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
        })

        user_id = f"locomo_{ci}_{uuid.uuid4().hex[:6]}"
        n_memories = ingest_conversation(conv_data, retriever, user_id)

        try:
            conv_rows = []
            for top_k, repeat_times in top_k_configs:
                label = f"top_k={top_k}" + (f"x{repeat_times}" if repeat_times > 1 else "")
                print(f"\n## {label}")
                results = {}
                for use_reorder in [False, True]:
                    cp_reset()  # fresh tree for each mode
                    stats = run_multi_turn(
                        retriever, user_id, qa_pairs, model, top_k,
                        use_reorder=use_reorder, cp_available=cp_available,
                        repeat_times=repeat_times)
                    results[stats["label"]] = stats

                base_ttft = results["baseline"]["ttft"]

                k_label = f"{top_k}x{repeat_times}" if repeat_times > 1 else str(top_k)
                for name in ["baseline", "reorder"]:
                    s = results[name]
                    delta = (base_ttft - s["ttft"]) / base_ttft * 100 if base_ttft else 0
                    row = {
                        "conv": ci,
                        "k": k_label,
                        "mode": name,
                        "ttft": s["ttft"],
                        "ttft_delta": delta if name != "baseline" else 0,
                        "prefix": s["prefix"],
                        "f1": s["f1"],
                        "judge": s["judge"],
                    }
                    conv_rows.append(row)
                    grand_rows.append(row)

            # Per-conversation summary
            print(f"\n{'='*70}")
            print(f"RESULTS (conv={ci}, memories={n_memories}, turns={min(NUM_TURNS, len(qa_pairs))})")
            print(f"{'='*70}")
            df = pd.DataFrame(conv_rows)
            df_display = df.copy()
            df_display["ttft"] = df_display["ttft"].map(lambda x: f"{x:.4f}s")
            df_display["ttft_delta"] = df.apply(
                lambda r: f"{r['ttft_delta']:+.1f}%" if r["mode"] != "baseline" else "-", axis=1)
            df_display["prefix"] = df_display["prefix"].map(lambda x: f"{x:.1%}")
            df_display["f1"] = df_display["f1"].map(lambda x: f"{x:.3f}")
            df_display["judge"] = df_display["judge"].map(lambda x: f"{x:.3f}")
            print(df_display.drop(columns=["conv"]).to_string(index=False))

        finally:
            try:
                retriever.delete_all_memories(user_id=user_id)
                print(f"\nCleaned up memories for {user_id}")
            except Exception as e:
                print(f"\nCleanup warning: {e}")
            del retriever
            import gc; gc.collect()

    # Grand aggregate table across all conversations
    if len(conv_indices) > 1:
        print(f"\n{'='*70}")
        print(f"AGGREGATE RESULTS ({len(conv_indices)} conversations)")
        print(f"{'='*70}")
        gdf = pd.DataFrame(grand_rows)
        agg = gdf.groupby(["k", "mode"]).agg(
            ttft=("ttft", "mean"),
            prefix=("prefix", "mean"),
            f1=("f1", "mean"),
            judge=("judge", "mean"),
        ).reset_index()
        # Compute delta from baseline per k
        for k_val in agg["k"].unique():
            base = agg.loc[(agg["k"] == k_val) & (agg["mode"] == "baseline"), "ttft"].values[0]
            agg.loc[agg["k"] == k_val, "ttft_delta"] = agg.loc[agg["k"] == k_val, "ttft"].apply(
                lambda x: (base - x) / base * 100 if base else 0)
        agg_display = agg.copy()
        agg_display["ttft"] = agg_display["ttft"].map(lambda x: f"{x:.4f}s")
        agg_display["ttft_delta"] = agg.apply(
            lambda r: f"{r['ttft_delta']:+.1f}%" if r["mode"] != "baseline" else "-", axis=1)
        agg_display["prefix"] = agg_display["prefix"].map(lambda x: f"{x:.1%}")
        agg_display["f1"] = agg_display["f1"].map(lambda x: f"{x:.3f}")
        agg_display["judge"] = agg_display["judge"].map(lambda x: f"{x:.3f}")
        print(agg_display[["k", "mode", "ttft", "ttft_delta", "prefix", "f1", "judge"]].to_string(index=False))
