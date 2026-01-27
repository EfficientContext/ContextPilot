from contextpilot.retriever import FAISSRetriever
from contextpilot.utils.tools import chunk_documents
from datasets import load_dataset
import asyncio
import argparse
import json


parser = argparse.ArgumentParser(description="Process queries and build index")
parser.add_argument("--embedding_model_path", type=str, default="Alibaba-NLP/gte-Qwen2-7B-instruct", help="Path to the embedding model")
parser.add_argument("--corpus_path", type=str, default="mulhoprag_corpus.jsonl", help="Path to the corpus you want to store, ending with .jsonl")
parser.add_argument("--index_path", type=str, default="mulhoprag_corpus_index.faiss", help="Path to save or load the index")
parser.add_argument("--query_path", type=str, default="mulhoprag_queries.jsonl", help="Path to the queries you want to store, ending with .jsonl")
parser.add_argument("--output_path", type=str, default="mulhoprag_faiss_results_top20.jsonl", help="Path to the output results")
parser.add_argument("--port", type=int, default=30000, help="Port for the embedding API")
parser.add_argument("--topk", type=int, default=20, help="Number of top documents to retrieve")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing")
parser.add_argument("--batch_delay", type=float, default=5.0, help="Delay between batches in seconds")
args = parser.parse_args()
base_url = f"http://localhost:{args.port}/v1"


corpus_origin = load_dataset("yixuantt/MultiHopRAG", "corpus")
qa = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")

# === Step 2: Create corpus documents with deduplication ===
corpus_docs = []
seen_paragraph_prefixes = set()

for entry in corpus_origin:
    for p in corpus_origin[entry]:
        paragraph_text = p["body"]
        title = p.get("title", "Untitled")
        
        # Calculate the first 50% of characters for deduplication
        prefix_length = len(paragraph_text) // 2
        paragraph_prefix = paragraph_text[:prefix_length]
        
        # Skip if we've seen this prefix before
        if paragraph_prefix in seen_paragraph_prefixes:
            continue
        
        # Add this prefix to our seen set
        seen_paragraph_prefixes.add(paragraph_prefix)
        
        # Add the full document (not just the prefix)
        corpus_docs.append({
            "title": title,
            "text": paragraph_text
        })

all_chunks = chunk_documents(corpus_docs)

# Save chunks to corpus file
with open(args.corpus_path, "w") as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk) + "\n")
        
# === Step 3: Create queries.jsonl ===
queries = []
query_id = 0
for entry in qa:
    for q in qa[entry]:
        queries.append({
            "id": query_id,
            "question": q["query"],
            "answers": [q["answer"]] + q.get("answer_aliases", [])
        })
        query_id += 1

with open(args.query_path, "w") as f:
    for query in queries:
        f.write(json.dumps(query) + "\n")

# === Step 4: Initialize retriever and run retrieval ===
retriever = FAISSRetriever(
    model_path=args.embedding_model_path,
    base_url=base_url,
    index_path=args.index_path
)

# Example usage:
# `corpus_path` is checked for existence to determine if indexing should run.
# If the corpus file exists, it will be indexed. If not, the program assumes
# a pre-built index exists at `index_path` and proceeds to search.

asyncio.run(retriever.run_retrieval(
    corpus_file=args.corpus_path,
    queries_file=args.query_path,
    output_file=args.output_path,
    top_k=args.topk,
    batch_size=args.batch_size,
    batch_delay=args.batch_delay
))