import json
from elasticsearch import Elasticsearch
from tqdm import tqdm

class BM25Retriever:
    def __init__(self, es_host="http://localhost:9200", index_name="bm25_index"):
        """
        Initialize the BM25 Retriever.
        
        Args:
            es_host (str): Elasticsearch host URL
            index_name (str): Name of the Elasticsearch index
        """
        self.es = Elasticsearch(es_host, request_timeout=3600*24)
        self.index_name = index_name
    
    def create_index(self):
        """
        Create or recreate the Elasticsearch index with appropriate mappings.
        """
        print(f"Creating index '{self.index_name}'...")
        self.es.indices.create(
            index=self.index_name,
            body={
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "title": {
                            "type": "keyword"
                        },
                        "chunk_id": {
                            "type": "integer"
                        }
                    }
                }
            }
        )
        print(f"Index '{self.index_name}' created.")
    
    def index_corpus(self, corpus_file=None, corpus_data=None):
        """
        Index the corpus documents into Elasticsearch.
        
        Args:
            corpus_file (str): Path to the corpus JSONL file
        """
        actions = []
        assert corpus_file or corpus_data, "Either corpus_file or corpus_data must be provided."
        if corpus_data:
            for data in tqdm(corpus_data, desc="Indexing corpus data"):
                # Get the ID for the action line
                chunk_id = data.get("chunk_id") or data.get("document_id") or data.get("_id")
                text = data['text']
                title = data.get('title', '')

                # Action line: Specifies the index and the document's unique ID
                actions.append({"index": {"_index": self.index_name, "_id": chunk_id}})

                # Source line: Contains the data fields to be indexed, including chunk_id as integer
                actions.append({"text": text, "title": title, "chunk_id": chunk_id})

                if len(actions) >= 2000:
                    self.es.bulk(body=actions)
                    actions = []
        else:
            with open(corpus_file, 'r') as f:
                for line in tqdm(f, desc="Indexing corpus from file"):
                    doc = json.loads(line)
                    doc_id = doc.get("chunk_id") or doc.get("document_id") or doc.get("_id")

                    # Action line: Specifies the index and the document's unique ID
                    action = {"index": {"_index": self.index_name, "_id": doc_id}}
                    actions.append(action)

                    # Source line: Keep chunk_id as an integer field in the document
                    # Store chunk_id for retrieval as integer
                    if "chunk_id" not in doc and doc_id is not None:
                        doc["chunk_id"] = doc_id
                    
                    # Remove alternative ID fields but keep chunk_id
                    if "_id" in doc:
                        del doc["_id"]
                    if "document_id" in doc:
                        del doc["document_id"]
                    
                    actions.append(doc)

                    if len(actions) >= 2000:
                        self.es.bulk(body=actions)
                        actions = []
        if actions:
            self.es.bulk(body=actions)

        self.es.indices.refresh(index=self.index_name)
    
    def is_existing_index(self):
        """
        Check if the Elasticsearch index exists.
        
        Returns:
            bool: True if index exists, False otherwise
        """
        return self.es.indices.exists(index=self.index_name)
    
    def search_queries(self, queries_file=None, query_data=None, top_k=20):
        """
        Search queries against the indexed corpus.
        """
        results = []
        assert queries_file or query_data, "Either queries_file or query_data must be provided."
        
        # Logic for query_data
        if query_data:
            for query in query_data:
                # Extract text from query dict, or use query directly if it's a string
                if isinstance(query, dict):
                    query_text = query.get("text", query.get("question", ""))
                    qid = query.get("qid")
                    answer = query.get("answer")
                else:
                    query_text = query
                    qid = None
                    answer = None
                    
                response = self.es.search(
                    index=self.index_name,
                    body={
                        "query": { "match": { "text": query_text } },
                        "size": top_k
                    }
                )
                
                # Retrieve chunk_id as integer from document source
                top_k_doc_ids = [hit["_source"]["chunk_id"] for hit in response["hits"]["hits"]]
                
                result = {
                    "text": query_text,
                    "top_k_doc_id": top_k_doc_ids
                }
                
                # Add optional fields if they exist
                if qid is not None:
                    result["qid"] = qid
                if answer is not None:
                    result["answer"] = answer
                
                results.append(result)
                
        # Logic for queries_file
        else:
            with open(queries_file, 'r') as f:
                for line in tqdm(f, desc="Searching queries"):
                    query_data_line = json.loads(line)
                    query_text = query_data_line["question"]
                    
                    response = self.es.search(
                        index=self.index_name,
                        body={
                            "query": { "match": { "text": query_text } },
                            "size": top_k
                        }
                    )
                    
                    # Retrieve chunk_id as integer from document source
                    top_k_doc_ids = [hit["_source"]["chunk_id"] for hit in response["hits"]["hits"]]
                    
                    results.append({
                        "text": query_data_line["question"],
                        "top_k_doc_id": top_k_doc_ids
                    })
                    
        return results
    
    def save_results(self, results, output_file):
        """
        Save search results to a JSONL file.
        
        Args:
            results (list): List of search results
            output_file (str): Path to the output JSONL file
        """
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
    
    def run_retrieval(self, corpus_file, queries_file, output_file, top_k=20):
        """
        Run the complete retrieval pipeline.
        
        Args:
            corpus_file (str): Path to the corpus JSONL file
            queries_file (str): Path to the queries JSONL file
            output_file (str): Path to the output JSONL file
            top_k (int): Number of top documents to retrieve
        """
        print("Creating index...")
        self.create_index()
        
        print("Indexing corpus...")
        self.index_corpus(corpus_file)
        
        print("Searching queries...")
        results = self.search_queries(queries_file, top_k)
        
        print(f"Writing results to {output_file}...")
        self.save_results(results, output_file)
        
        print("Done.")