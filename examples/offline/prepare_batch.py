from contextpilot.context_index import build_context_index
from contextpilot.context_ordering import InterContextScheduler
import json
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Reorder prompts based on clustering results.")
    parser.add_argument('--context_path', type=str, required=True, help='Path to the JSONL file containing retrieval results.')
    parser.add_argument('--output_path', type=str, default='planning_output.jsonl', help='Path to the JSONL file containing plans.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for distance computation.')
    parser.add_argument('--linkage_method', type=str, default='average', choices=['average', 'complete', 'single'],
                       help='Linkage method for hierarchical clustering.')
    parser.add_argument('--alpha', type=float, default=0.001, help='Weight for position term in distance calculation.')
    args = parser.parse_args()
    return args

args = parse_args()
context_path = args.context_path

def prepare_batch(context_path):
    # Load prompts from the specified path
    with open(context_path, 'r') as f:
        prompts = [json.loads(line) for line in f]

    qids = [prompt['qid'] for prompt in prompts]
    questions = [prompt['text'] for prompt in prompts]
    answers = [prompt['answer'] for prompt in prompts]
    topk_doc_ids = [prompt['top_k_doc_id'] for prompt in prompts]

    scheduler = InterContextScheduler()
    
    # Perform clustering and intra-reordering
    cluster_start = time.perf_counter()
    print("Building context index (clustering + intra-reordering)...")
    result = build_context_index(
        topk_doc_ids,
        linkage_method=args.linkage_method,
        use_gpu=args.use_gpu,
        alpha=args.alpha
    )
    cluster_end = time.perf_counter()
    print(f"Context indexing took {cluster_end - cluster_start:.2f} seconds")

    # Perform inter-context scheduling
    inter_start = time.perf_counter()
    print("Performing inter-context scheduling...")
    organized_reordered_ids, organized_original_ids, final_index_mapping, all_groups_with_scores = scheduler.schedule_contexts(result)
    inter_end = time.perf_counter()
    print(f"Inter-context scheduling took {inter_end - inter_start:.2f} seconds")

    # Build output groups efficiently using final_index_mapping
    groups = []
    for group_id, (score, group_indices) in enumerate(all_groups_with_scores):
        items = [
            {
                "qid": qids[idx],
                "question": questions[idx],
                "answer": answers[idx],
                "top_k_doc_id": organized_reordered_ids[final_index_mapping.index(idx)],
                "orig_top_k_doc_id": organized_original_ids[final_index_mapping.index(idx)]
            }
            for idx in group_indices
        ]
        groups.append({
            "group_id": group_id,
            "group_size": len(items),
            "items": items
        })

    # Sort groups by size (largest to smallest)
    groups.sort(key=lambda x: x['group_size'], reverse=True)

    return groups

start = time.perf_counter()
batch_groups = prepare_batch(context_path)
end = time.perf_counter()
print(f"Total batch preparation took {end - start:.2f} seconds")
print(f"Generated {len(batch_groups)} groups")
with open(args.output_path, 'w') as f:
    for group in batch_groups:
        f.write(json.dumps(group) + "\n")
print(f"Output written to {args.output_path}")
