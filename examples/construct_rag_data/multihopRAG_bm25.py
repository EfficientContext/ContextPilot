from datasets import load_dataset
import json
import os
import sys
from contextpilot.retriever.bm25 import BM25Retriever
from contextpilot.utils.tools import chunk_documents
import argparse

'''
Make sure to have Elasticsearch running on localhost:9200 by running:
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.security.http.ssl.enabled=false" \
  -e "xpack.security.transport.ssl.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms100g -Xmx100g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.18.2
'''

parser = argparse.ArgumentParser(description="Process queries and build index")
parser.add_argument("--gen_model_path", type=str, default="Qwen/Qwen3-32B", help="Path to the generation model")
parser.add_argument("--index_name", type=str, default="mulhoprag_corpus", help="Name of the Elasticsearch index")
parser.add_argument("--corpus_path", type=str, default="mulhoprag_corpus.jsonl", help="Path to the corpus you want to store")
parser.add_argument("--query_path", type=str, default="mulhoprag_queries.jsonl", help="Path to the queries you want to store")
parser.add_argument("--topk", type=int, default=20, help="Number of top documents to retrieve")
parser.add_argument("--output_path", type=str, default="mulhoprag_bm25_results_top20.jsonl", help="Path to the output results")
parser.add_argument("--port", type=int, default=9200, help="Port for Elasticsearch")
parser.add_argument("--force", action="store_true", help="Force rebuild even if output file already exists")
args = parser.parse_args()

# Skip everything if output already exists and --force not set
if os.path.exists(args.output_path) and not args.force:
    print(f"Output already exists at {args.output_path} — skipping. Use --force to rebuild.")
    sys.exit(0)

corpus_origin = load_dataset("yixuantt/MultiHopRAG", "corpus")
qa = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")

# === Step 2: Create corpus.jsonl with title included in text ===
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

        seen_paragraph_prefixes.add(paragraph_prefix)
        corpus_docs.append({"title": title, "text": paragraph_text})

all_chunks = chunk_documents(corpus_docs)

# Save corpus to disk (chunk_documents only returns chunks, doesn't write)
with open(args.corpus_path, "w") as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk) + "\n")
print(f"Corpus saved to {args.corpus_path} ({len(all_chunks)} chunks)")

# === Step 3: Create queries.jsonl ===
queries = []
query_id = 0
for entry in qa:
    for q in qa[entry]:
        queries.append({
            "qid": query_id,
            "text": q["query"],
            "answer": q["answer"],
        })
        query_id += 1

with open(args.query_path, "w") as f:
    for query in queries:
        f.write(json.dumps(query) + "\n")

detailed_chunks = all_chunks

retriever = BM25Retriever(es_host=f"http://localhost:{args.port}", index_name=args.index_name)
retriever.create_index()
retriever.index_corpus(corpus_data=detailed_chunks)
results = retriever.search_queries(query_data=queries, top_k=20)
output_file = args.output_path

# === Step 5: Write output files ===
print(f"Writing results to {output_file}...")
with open(output_file, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + "\n")
            
    print("Done.")