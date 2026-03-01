import contextpilot as cp
import json
import argparse
import os
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Reorder prompts based on clustering results.")
    parser.add_argument('--context_path', type=str, required=True, help='Path to the JSONL file containing retrieval results.')
    parser.add_argument('--output_path', type=str, default='planning_output.jsonl', help='Path to the JSONL file containing plans.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for distance computation.')
    parser.add_argument('--linkage_method', type=str, default='average', choices=['average', 'complete', 'single'],
                       help='Linkage method for hierarchical clustering.')
    parser.add_argument('--alpha', type=float, default=0.001, help='Weight for position term in distance calculation.')
    parser.add_argument('--force', action='store_true', help='Force rebuild even if output file already exists.')
    args = parser.parse_args()
    return args

def prepare_batch(context_path, args):
    # Load prompts from the specified path
    with open(context_path, 'r') as f:
        prompts = [json.loads(line) for line in f]

    qids = [prompt['qid'] for prompt in prompts]
    questions = [prompt['text'] for prompt in prompts]
    answers = [prompt['answer'] for prompt in prompts]
    topk_doc_ids = [
        [doc_id for doc_id in prompt['top_k_doc_id'] if doc_id is not None]
        for prompt in prompts
    ]

    # Reorder contexts for optimal KV-cache prefix sharing
    reorder_start = time.perf_counter()
    print("Reordering contexts with ContextPilot...")
    engine = cp.ContextPilot(
        use_gpu=args.use_gpu,
        alpha=args.alpha,
        linkage_method=args.linkage_method,
    )
    reordered_contexts, original_indices = engine.reorder(topk_doc_ids)
    reorder_end = time.perf_counter()
    print(f"Reordering took {reorder_end - reorder_start:.2f} seconds")

    # Build output in optimized order
    items = []
    for i, q_idx in enumerate(original_indices):
        items.append({
            "qid": qids[q_idx],
            "question": questions[q_idx],
            "answer": answers[q_idx],
            "top_k_doc_id": reordered_contexts[i],
            "orig_top_k_doc_id": topk_doc_ids[q_idx],
        })

    return items

if __name__ == '__main__':
    args = parse_args()

    if os.path.exists(args.output_path) and not args.force:
        print(f"Output already exists at {args.output_path} — skipping. Use --force to rebuild.")
        sys.exit(0)

    start = time.perf_counter()
    batch_items = prepare_batch(args.context_path, args)
    end = time.perf_counter()
    print(f"Total batch preparation took {end - start:.2f} seconds")
    print(f"Generated {len(batch_items)} reordered items")
    with open(args.output_path, 'w') as f:
        for item in batch_items:
            f.write(json.dumps(item) + "\n")
    print(f"Output written to {args.output_path}")
