import faiss
import json
import asyncio
import aiohttp
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import datetime
from tqdm import tqdm

class FAISSRetriever:
    def __init__(self, model_path: str, base_url: str, index_path: str):
        """
        Initialize the FAISS Retriever for semantic search.

        Args:
            model_path (str): The model identifier for embedding generation.
            base_url (str): The base URL for the embedding service.
            index_path (str): The path to the FAISS index file.
        """
        self.model_path = model_path
        self.base_url = base_url
        self.index_path = index_path
        self.es = None  # Placeholder to match the style

    async def _get_embedding(self, session: aiohttp.ClientSession, text: str) -> List[float]:
        """Get embedding for a single text using an async HTTP request."""
        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.model_path,
            "input": text,
        }

        async with session.post(url, json=payload) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}: {await response.text()}")
            result = await response.json()
            return result["data"][0]["embedding"]

    async def _process_corpus_batch(self, session: aiohttp.ClientSession, batch_docs: List[Dict[str, Any]]) -> List[Tuple[int, Optional[List[float]]]]:
        """Process a single batch of corpus documents asynchronously."""
        async def process_single_document(doc: Dict[str, Any]) -> Tuple[int, Optional[List[float]]]:
            try:
                embedding = await self._get_embedding(session, f"(Part of {doc['title']}) {doc['text']}")
                return (doc["chunk_id"], embedding)
            except Exception as e:
                print(f"❌ Failed to get embedding for doc {doc['chunk_id']}: {e}")
                return (doc["chunk_id"], None)

        tasks = [process_single_document(doc) for doc in batch_docs]
        return await asyncio.gather(*tasks)

    async def index_corpus(self, corpus_file: Optional[str] = None, corpus_data: Optional[List[Dict[str, Any]]] = None, **kwargs):
        """
        Build and save a FAISS index from a corpus.

        Args:
            corpus_file (str): Path to the corpus JSONL file.
            corpus_data (list): List of corpus documents.
            **kwargs: Additional parameters for index building (e.g., hnsw_m, ef_construction, batch_size, batch_delay).
        """
        hnsw_m = kwargs.get("hnsw_m", 64)
        ef_construction = kwargs.get("ef_construction", 400)
        ef_search = kwargs.get("ef_search", 200)
        batch_size = kwargs.get("batch_size", 50)
        batch_delay = kwargs.get("batch_delay", 2.0)
        
        assert corpus_file or corpus_data, "Either corpus_file or corpus_data must be provided."

        if corpus_file:
            with open(corpus_file, "r") as f:
                corpus = [json.loads(line) for line in f]
        else:
            corpus = corpus_data
        
        # Pre-allocate list to preserve order
        embeddings = [None] * len(corpus)
        total_successful = 0

        with tqdm(total=len(corpus), desc="Indexing corpus") as pbar:
            for batch_idx in range(0, len(corpus), batch_size):
                batch_docs = corpus[batch_idx:batch_idx + batch_size]
                
                # Process this batch (async within batch)
                results = await self._process_corpus_batch(self.session, batch_docs)
                
                # Store results
                for chunk_id, embedding in results:
                    if embedding is not None:
                        embeddings[chunk_id] = embedding
                        total_successful += 1
                
                pbar.update(len(batch_docs))
                
                # Wait before next batch (sync between batches)
                if batch_idx + batch_size < len(corpus) and batch_delay > 0:
                    await asyncio.sleep(batch_delay)

        if total_successful == 0:
            raise Exception("No embeddings were successfully generated!")
        
        # Create mapping for valid embeddings
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        
        embeddings_array = np.array(valid_embeddings, dtype=np.float32)
        
        dimension = embeddings_array.shape[1]
        index = faiss.IndexHNSWFlat(dimension, hnsw_m)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        
        index.add(embeddings_array)
        
        # Save index and metadata
        faiss_path = self.index_path
        metadata_path = self.index_path.replace('.faiss', '') + '_metadata.json'
        
        os.makedirs(os.path.dirname(faiss_path) or '.', exist_ok=True)
        faiss.write_index(index, faiss_path)
        
        metadata = {
            'created_at': datetime.datetime.now().isoformat(),
            'corpus_size': len(corpus),
            'valid_embeddings': total_successful,
            'failed_embeddings': len(corpus) - total_successful,
            'dimension': dimension,
            'hnsw_m': hnsw_m,
            'ef_construction': ef_construction,
            'ef_search': ef_search,
            'valid_indices': valid_indices
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ FAISS index saved to: {faiss_path}")
        print(f"📋 Metadata saved to: {metadata_path}")
        print(f"📊 Index stats: {index.ntotal} vectors, dimension {dimension}")

    async def _process_query_batch(self, batch_data: List[Tuple[int, Dict[str, Any]]], top_k: int) -> List[Any]:
        """Process a single batch of queries asynchronously."""
        tasks = []
        for original_index, query_doc in batch_data:
            query_text = query_doc["question"]
            if query_doc.get("title"):
                query_text = f"{query_text} in '{query_doc['title']}'"

            tasks.append(self._search_single_query(
                session=self.session,
                index=original_index,
                query_doc=query_doc,
                query_text=query_text,
                top_k=top_k
            ))
        
        return await asyncio.gather(*tasks)

    async def _search_single_query(self, session, index, query_doc, query_text, top_k):
        try:
            embedding = await self._get_embedding(session, query_text)
            embedding_array = np.array([embedding], dtype=np.float32)
            
            D, I = self.indices.search(embedding_array, top_k)
            
            result = {
                "qid": query_doc.get("id", query_doc.get("qid")),
                "text": query_text,
                "answer": query_doc.get("answers", query_doc.get("ans")),
                "top_k_doc_id": [int(i) for i in I[0]]
            }
            
            return (index, result)
        except Exception as e:
            print(f"Error processing query at index {index}: {e}")
            return (index, None)

    async def search_queries(self, queries_file: Optional[str] = None, query_data: Optional[List[Dict[str, Any]]] = None, top_k: int = 20, **kwargs) -> List[Dict[str, Any]]:
        """
        Search queries against the indexed corpus.

        Args:
            queries_file (str): Path to the queries JSONL file.
            query_data (list): List of query documents.
            top_k (int): Number of top documents to retrieve.
            **kwargs: Additional parameters for query processing (e.g., batch_size, batch_delay).
        
        Returns:
            list: List of search results with query info and top-k document IDs.
        """
        batch_size = kwargs.get("batch_size", 100)
        batch_delay = kwargs.get("batch_delay", 0.5)

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        self.indices = faiss.read_index(self.index_path)
        
        # Configure search accuracy
        if hasattr(self.indices, 'hnsw'):
            ef_search = kwargs.get("ef_search", 200)
            self.indices.hnsw.efSearch = ef_search
            print(f"🎯 Search accuracy configured: efSearch = {ef_search}")
        
        assert queries_file or query_data, "Either queries_file or query_data must be provided."

        if queries_file:
            with open(queries_file, 'r') as f:
                queries = [json.loads(line) for line in f]
        else:
            queries = query_data

        results = [None] * len(queries)
        
        with tqdm(total=len(queries), desc="Searching queries") as pbar:
            for batch_idx in range(0, len(queries), batch_size):
                batch_queries = []
                for i in range(batch_idx, min(batch_idx + batch_size, len(queries))):
                    batch_queries.append((i, queries[i]))
                
                batch_results = await self._process_query_batch(batch_queries, top_k)

                for original_index, data in batch_results:
                    if data is not None:
                        results[original_index] = data
                
                pbar.update(len(batch_queries))
                
                if batch_idx + batch_size < len(queries) and batch_delay > 0:
                    await asyncio.sleep(batch_delay)

        return [r for r in results if r is not None]

    def save_results(self, results, output_file):
        """
        Save search results to a JSONL file.
        
        Args:
            results (list): List of search results.
            output_file (str): Path to the output JSONL file.
        """
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    async def run_retrieval(self, corpus_file: str, queries_file: str, output_file: str, top_k: int = 20, **kwargs):
        """
        Run the complete retrieval pipeline.
        
        Args:
            corpus_file (str): Path to the corpus JSONL file.
            queries_file (str): Path to the queries JSONL file.
            output_file (str): Path to the output JSONL file.
            top_k (int): Number of top documents to retrieve.
        """
        connector = aiohttp.TCPConnector(
            limit=kwargs.get("batch_size", 50) * 2,
            limit_per_host=kwargs.get("batch_size", 50)
        )
        timeout = aiohttp.ClientTimeout(
            total=kwargs.get("total_session_timeout", 14400),
            connect=30,
            sock_read=kwargs.get("per_request_timeout", 300)
        )
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            self.session = session

            if os.path.exists(corpus_file):
                print("Building and indexing corpus...")
                await self.index_corpus(corpus_file=corpus_file, **kwargs)
            else:
                print("Corpus file not provided or found. Skipping indexing.")
            
            print("Searching queries...")
            results = await self.search_queries(queries_file=queries_file, top_k=top_k, **kwargs)
            
            print(f"Writing results to {output_file}...")
            self.save_results(results, output_file)
            
            print("Done.")