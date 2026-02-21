"""
End-to-end verifier for ContextPilot + vLLM patch behavior.

What it checks:
1. Two-request reorder works (prefix sharing improves for the crafted pair).
2. The two request_ids remain tracked before stress (no early/weird eviction).
3. After cache pressure, eviction is observed via vLLM callback -> ContextPilot /evict.

Prerequisites:
1. Start ContextPilot server (live mode):
   python -m contextpilot.server.http_server --port 8765

2. Start patched vLLM server with prefix caching enabled:
   CONTEXTPILOT_INDEX_URL=http://localhost:8765 python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen2.5-7B-Instruct --port 8000 --enable-prefix-caching

Run:
  python examples/vllm_patch_e2e_check.py
"""

from __future__ import annotations

import argparse
import random
import string
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, List, Sequence, Set, Tuple

import requests


def lcp_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def require_ok(resp: requests.Response, name: str) -> Dict:
    if resp.status_code >= 400:
        raise RuntimeError(f"{name} failed: HTTP {resp.status_code}, body={resp.text[:500]}")
    return resp.json()


def get_model_id(session: requests.Session, vllm_url: str, timeout: float) -> str:
    data = require_ok(session.get(f"{vllm_url}/v1/models", timeout=timeout), "GET /v1/models")
    items = data.get("data") or []
    if not items:
        raise RuntimeError("No models returned by vLLM /v1/models")
    return items[0]["id"]


def reset_contextpilot(session: requests.Session, cp_url: str, timeout: float) -> None:
    require_ok(session.post(f"{cp_url}/reset", timeout=timeout), "POST /reset")


def reorder_two_contexts(
    session: requests.Session, cp_url: str, timeout: float
) -> Tuple[List[List[int]], List[int], List[str]]:
    # Crafted pair: large overlap + distinct tail docs, so reorder can increase
    # prefix sharing while preserving two distinct requests.
    contexts = [
        [101, 202, 303, 404, 901],
        [303, 404, 101, 202, 902],
    ]
    payload = {
        "contexts": contexts,
        "alpha": 0.001,
        "use_gpu": False,
        "linkage_method": "average",
    }
    data = require_ok(
        session.post(f"{cp_url}/reorder", json=payload, timeout=timeout),
        "POST /reorder",
    )

    reordered = data.get("reordered_contexts")
    original_indices = data.get("original_indices")
    request_ids = data.get("request_ids")
    if not reordered or not original_indices or not request_ids:
        raise RuntimeError(f"/reorder missing required fields: {data}")
    if len(reordered) != 2 or len(original_indices) != 2 or len(request_ids) != 2:
        raise RuntimeError(f"Expected exactly two contexts and request IDs, got: {data}")
    if len(set(request_ids)) != 2:
        raise RuntimeError(
            "Expected two distinct request_ids but got duplicates. "
            f"request_ids={request_ids}, reordered={reordered}"
        )

    for i, orig_idx in enumerate(original_indices):
        if Counter(reordered[i]) != Counter(contexts[orig_idx]):
            raise RuntimeError(
                "Reorder changed document membership unexpectedly: "
                f"reordered[{i}]={reordered[i]} vs original[{orig_idx}]={contexts[orig_idx]}"
            )

    before = lcp_len(contexts[original_indices[0]], contexts[original_indices[1]])
    after = lcp_len(reordered[0], reordered[1])
    if after <= before:
        raise RuntimeError(
            f"Two-request reorder did not improve prefix sharing: before={before}, after={after}. "
            f"reordered={reordered}, original_indices={original_indices}"
        )

    print(f"[PASS] Reorder improved shared prefix: before={before}, after={after}")
    return reordered, original_indices, request_ids


def get_tracked_request_ids(
    session: requests.Session, cp_url: str, timeout: float
) -> Set[str]:
    data = require_ok(session.get(f"{cp_url}/requests", timeout=timeout), "GET /requests")
    return set(data.get("request_ids", []))


def make_prompt(
    doc_ids: Sequence[int],
    approx_words: int,
    tag: str,
    anchor: str,
    family: str,
) -> str:
    base = " ".join([f"doc_{d}" for d in doc_ids])
    words = [f"w{(i * 7919) % 100000}" for i in range(max(0, approx_words - len(doc_ids)))]
    return (
        # Family-specific first tokens reduce accidental shared prefix hashes
        # between seed and pressure requests.
        f"ANCHOR::{family}::{anchor}\n"
        f"Context IDs: {base}\n"
        f"Tag: {tag}\n"
        f"{' '.join(words)}\n"
        "Answer in one short sentence."
    )


def send_vllm_completion(
    session: requests.Session,
    vllm_url: str,
    model: str,
    prompt: str,
    timeout: float,
    request_id: str | None = None,
    max_tokens: int = 8,
    attempts: int = 3,
    retry_backoff: float = 2.0,
    label: str = "request",
    verbose: bool = True,
) -> None:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    if request_id is not None:
        payload["request_id"] = request_id

    last_err: Exception | None = None
    for i in range(1, attempts + 1):
        if verbose:
            print(f"  [{label}] attempt {i}/{attempts} ...")
        try:
            resp = session.post(
                f"{vllm_url}/v1/completions",
                json=payload,
                timeout=timeout,
            )
            require_ok(resp, "POST /v1/completions")
            if verbose:
                print(f"  [{label}] success")
            return
        except (requests.Timeout, requests.ConnectionError, RuntimeError) as e:
            last_err = e
            if verbose:
                print(f"  [{label}] attempt {i} failed: {e}")
            if i == attempts:
                break
            time.sleep(retry_backoff * i)
    raise RuntimeError(f"vLLM completion failed after {attempts} attempts: {last_err}")


def warmup_vllm(
    session: requests.Session,
    vllm_url: str,
    model: str,
    timeout: float,
) -> None:
    send_vllm_completion(
        session=session,
        vllm_url=vllm_url,
        model=model,
        prompt="Warmup request. Reply with one word.",
        timeout=timeout,
        request_id=None,
        max_tokens=4,
        attempts=5,
        retry_backoff=3.0,
        label="warmup",
    )


def run_pressure(
    vllm_url: str,
    model: str,
    requests_count: int,
    workers: int,
    prompt_words: int,
    timeout: float,
    max_tokens: int,
    attempts: int,
    retry_backoff: float,
    progress_every: int,
    heartbeat_seconds: float,
) -> Tuple[int, int]:
    def _task(i: int) -> bool:
        sess = requests.Session()
        try:
            random_tail = "".join(random.choices(string.ascii_lowercase, k=16))
            prompt = make_prompt(
                [i, i + 1, i + 2, i + 3],
                approx_words=prompt_words,
                tag=f"pressure-{i}-{random_tail}",
                anchor=f"p-{i}-{random_tail}",
                family="pressure",
            )
            send_vllm_completion(
                session=sess,
                vllm_url=vllm_url,
                model=model,
                prompt=prompt,
                timeout=timeout,
                request_id=None,
                max_tokens=max_tokens,
                attempts=attempts,
                retry_backoff=retry_backoff,
                label=f"pressure-{i}",
                verbose=False,
            )
            return True
        except Exception:
            return False
        finally:
            sess.close()

    ok, fail = 0, 0
    completed = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_task, i) for i in range(requests_count)}
        while pending:
            done, pending = wait(
                pending,
                timeout=heartbeat_seconds,
                return_when=FIRST_COMPLETED,
            )

            if not done:
                elapsed = time.time() - start
                print(
                    "  pressure heartbeat: "
                    f"completed={completed}/{requests_count} "
                    f"ok={ok} fail={fail} elapsed={elapsed:.1f}s"
                )
                continue

            for fut in done:
                completed += 1
                if fut.result():
                    ok += 1
                else:
                    fail += 1

                if (
                    completed % max(1, progress_every) == 0
                    or completed == requests_count
                ):
                    elapsed = time.time() - start
                    print(
                        "  pressure progress: "
                        f"{completed}/{requests_count} "
                        f"(ok={ok}, fail={fail}, elapsed={elapsed:.1f}s)"
                    )
            if completed == requests_count:
                break
    return ok, fail


def poll_eviction(
    session: requests.Session,
    cp_url: str,
    target_ids: Set[str],
    timeout: float,
    poll_interval: float = 2.0,
) -> Tuple[Set[str], Set[str]]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        tracked = get_tracked_request_ids(session, cp_url, timeout=10.0)
        remaining = tracked.intersection(target_ids)
        evicted = target_ids - remaining
        if evicted:
            return evicted, remaining
        time.sleep(poll_interval)
    tracked = get_tracked_request_ids(session, cp_url, timeout=10.0)
    remaining = tracked.intersection(target_ids)
    evicted = target_ids - remaining
    return evicted, remaining


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ContextPilot + vLLM patch behavior.")
    parser.add_argument("--cp-url", default="http://localhost:8765", help="ContextPilot base URL")
    parser.add_argument("--vllm-url", default="http://localhost:8000", help="vLLM base URL")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="HTTP timeout seconds")
    parser.add_argument("--seed-prompt-words", type=int, default=220, help="Approx words for the first 2 tracked requests")
    parser.add_argument("--pressure-requests", type=int, default=140, help="Number of pressure requests")
    parser.add_argument("--pressure-workers", type=int, default=6, help="Parallel workers for pressure phase")
    parser.add_argument("--pressure-prompt-words", type=int, default=900, help="Approx words per pressure prompt")
    parser.add_argument("--pressure-timeout", type=float, default=20.0, help="HTTP timeout seconds for pressure requests")
    parser.add_argument("--pressure-attempts", type=int, default=1, help="Retry attempts for each pressure request")
    parser.add_argument("--pressure-retry-backoff", type=float, default=1.0, help="Backoff multiplier between pressure retries")
    parser.add_argument("--pressure-progress-every", type=int, default=10, help="Print pressure progress every N completed requests")
    parser.add_argument("--pressure-heartbeat-seconds", type=float, default=15.0, help="Print heartbeat if no pressure request finishes in this interval")
    parser.add_argument("--max-tokens", type=int, default=4, help="max_tokens for completion requests")
    parser.add_argument("--eviction-wait-seconds", type=float, default=120.0, help="How long to wait for eviction callback")
    args = parser.parse_args()

    print("=== ContextPilot + vLLM Patch E2E Check ===")
    print(f"ContextPilot: {args.cp_url}")
    print(f"vLLM:        {args.vllm_url}")
    print(f"Pressure:    {args.pressure_requests} requests, workers={args.pressure_workers}")

    session = requests.Session()
    try:
        model = get_model_id(session, args.vllm_url, timeout=args.request_timeout)
        print(f"Model: {model}")
        warmup_vllm(session, args.vllm_url, model, timeout=args.request_timeout)
        print("[PASS] vLLM warmup request succeeded")

        reset_contextpilot(session, args.cp_url, timeout=args.request_timeout)
        print("Reset ContextPilot index")

        reordered, original_indices, request_ids = reorder_two_contexts(
            session, args.cp_url, timeout=args.request_timeout
        )
        print(f"request_ids: {request_ids}")
        print(f"original_indices: {original_indices}")
        print(f"reordered_contexts: {reordered}")

        target_ids = set(request_ids)
        for i, rid in enumerate(request_ids):
            print(f"Sending tracked request {i + 1}/2 with rid={rid} ...")
            prompt = make_prompt(
                reordered[i],
                approx_words=args.seed_prompt_words,
                tag=f"seed-{i}-{uuid.uuid4().hex[:8]}",
                anchor=rid,
                family="seed",
            )
            send_vllm_completion(
                session=session,
                vllm_url=args.vllm_url,
                model=model,
                prompt=prompt,
                timeout=args.request_timeout,
                request_id=rid,
                max_tokens=args.max_tokens,
                attempts=4,
                retry_backoff=2.0,
                label=f"seed-{i + 1}",
            )
        print("[PASS] Sent two tracked requests to vLLM")

        tracked_before_stress = get_tracked_request_ids(session, args.cp_url, timeout=args.request_timeout)
        if not target_ids.issubset(tracked_before_stress):
            raise RuntimeError(
                "Unexpected early eviction before stress. "
                f"target_ids={sorted(target_ids)} tracked={sorted(tracked_before_stress)}"
            )
        print("[PASS] No weird early eviction before stress phase")

        print("Applying cache pressure...")
        ok, fail = run_pressure(
            vllm_url=args.vllm_url,
            model=model,
            requests_count=args.pressure_requests,
            workers=args.pressure_workers,
            prompt_words=args.pressure_prompt_words,
            timeout=args.pressure_timeout,
            max_tokens=args.max_tokens,
            attempts=args.pressure_attempts,
            retry_backoff=args.pressure_retry_backoff,
            progress_every=args.pressure_progress_every,
            heartbeat_seconds=args.pressure_heartbeat_seconds,
        )
        print(f"Pressure completed: ok={ok}, fail={fail}")

        evicted, remaining = poll_eviction(
            session=session,
            cp_url=args.cp_url,
            target_ids=target_ids,
            timeout=args.eviction_wait_seconds,
        )

        if not evicted:
            raise RuntimeError(
                "No target request_id eviction observed after pressure. "
                "Increase --pressure-requests / --pressure-prompt-words or reduce vLLM KV cache."
            )

        print(f"[PASS] Eviction callback observed for target request_ids: {sorted(evicted)}")
        if remaining:
            print(f"[INFO] Still cached (not yet evicted): {sorted(remaining)}")
        else:
            print("[PASS] Both tracked request_ids were evicted and synced to ContextPilot")

        print("=== E2E CHECK PASSED ===")
        return 0

    except Exception as e:
        print(f"[FAIL] {e}")
        print("=== E2E CHECK FAILED ===")
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
