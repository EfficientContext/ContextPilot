"""
High-level RAG Pipeline abstraction.

Provides a simple interface for building RAG pipelines with or without
ContextPilot optimization.
"""

import json
import asyncio
import sys
import traceback
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time

import aiohttp

from .components import (
    RetrieverConfig,
    OptimizerConfig,
    InferenceConfig,
    PipelineConfig
)
from .multi_turn import MultiTurnManager
from ..retriever import BM25Retriever, FAISSRetriever, FAISS_AVAILABLE
from ..retriever import Mem0Retriever, MEM0_AVAILABLE
from ..retriever import PageIndexRetriever, PAGEINDEX_AVAILABLE
from ..context_index import build_context_index
from ..context_ordering import InterContextScheduler
from ..utils.prompt_generator import prompt_generator

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=100 * 60 * 60)


class RAGPipeline:
    """
    High-level RAG pipeline that handles retrieval, optimization, and inference.
    
    Examples:
        >>> # Simple ContextPilot pipeline (retrieve + optimize)
        >>> pipeline = RAGPipeline(
        ...     retriever="bm25",
        ...     corpus_path="corpus.jsonl"
        ... )
        >>> results = pipeline.run(queries=["What is AI?"])
        
        >>> # With LLM generation
        >>> from contextpilot.pipeline import InferenceConfig
        >>> pipeline = RAGPipeline(
        ...     retriever="bm25",
        ...     corpus_path="corpus.jsonl",
        ...     inference=InferenceConfig(
        ...         model_name="Qwen/Qwen2.5-7B-Instruct",
        ...         base_url="http://localhost:30000"
        ...     )
        ... )
        >>> results = pipeline.run(
        ...     queries=["What is AI?"],
        ...     generate_responses=True
        ... )
        
        >>> # Custom configuration
        >>> from contextpilot.pipeline import RetrieverConfig, OptimizerConfig, InferenceConfig
        >>> pipeline = RAGPipeline(
        ...     retriever=RetrieverConfig(
        ...         retriever_type="faiss",
        ...         top_k=10,
        ...         corpus_path="corpus.jsonl"
        ...     ),
        ...     optimizer=OptimizerConfig(enabled=True, use_gpu=True),
        ...     inference=InferenceConfig(model_name="Qwen/Qwen2.5-7B-Instruct")
        ... )
        
        >>> # Disable ContextPilot optimization
        >>> pipeline = RAGPipeline(
        ...     retriever="bm25",
        ...     corpus_path="corpus.jsonl",
        ...     use_contextpilot=False
        ... )
    """
    
    def __init__(
        self,
        retriever: Union[str, RetrieverConfig, Any] = "bm25",
        optimizer: Optional[Union[bool, OptimizerConfig]] = None,
        inference: Optional[Union[str, InferenceConfig]] = None,
        corpus_path: Optional[str] = None,
        corpus_data: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        use_contextpilot: bool = True,
        **kwargs
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Retriever type ("bm25", "faiss") or RetrieverConfig or custom retriever
            optimizer: OptimizerConfig or bool to enable/disable ContextPilot
            inference: Model name or InferenceConfig
            corpus_path: Path to corpus JSONL file
            corpus_data: List of corpus documents (alternative to corpus_path)
            model: Model name (shorthand for inference config)
            use_contextpilot: Whether to use ContextPilot optimization (shorthand)
            **kwargs: Additional configuration options
        """
        # Parse retriever configuration
        if isinstance(retriever, str):
            self.retriever_config = RetrieverConfig(
                retriever_type=retriever,
                corpus_path=corpus_path,
                corpus_data=corpus_data,
                **{k: v for k, v in kwargs.items() if k in RetrieverConfig.__dataclass_fields__}
            )
        elif isinstance(retriever, RetrieverConfig):
            self.retriever_config = retriever
        else:
            # Custom retriever instance
            self.retriever_config = RetrieverConfig(
                retriever_type="custom",
                custom_retriever=retriever,
                corpus_path=corpus_path,
                corpus_data=corpus_data
            )
        
        # Parse optimizer configuration
        if optimizer is None:
            optimizer = use_contextpilot
        
        if isinstance(optimizer, bool):
            self.optimizer_config = OptimizerConfig(
                enabled=optimizer,
                **{k: v for k, v in kwargs.items() if k in OptimizerConfig.__dataclass_fields__}
            )
        else:
            self.optimizer_config = optimizer
        
        # Parse inference configuration
        model_name = model or (inference if isinstance(inference, str) else None)
        if model_name:
            self.inference_config = InferenceConfig(
                model_name=model_name,
                **{k: v for k, v in kwargs.items() if k in InferenceConfig.__dataclass_fields__}
            )
        elif isinstance(inference, InferenceConfig):
            self.inference_config = inference
        else:
            self.inference_config = None
        
        # Initialize components
        self.retriever = None
        self.corpus = None
        self.corpus_map = None
        self._retrieval_cache = {}
        
        # Multi-turn conversation manager
        self.multi_turn_manager = MultiTurnManager()
        
        # Track if index has been built (for incremental builds)
        self._index_built = False
        
        # Verbose flag
        self.verbose = kwargs.get("verbose", True)
        
    def _log(self, message: str):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def setup(self):
        """
        Setup pipeline components (retriever, corpus loading, etc.).
        
        This is called automatically by run() but can be called explicitly
        for more control.
        """
        self._log("🚀 Setting up RAG pipeline...")
        
        # Setup retriever
        self._setup_retriever()
        
        # Load corpus
        self._load_corpus()
        
        self._log("✅ Pipeline setup complete")
    
    def _setup_retriever(self):
        """Initialize the retriever based on configuration."""
        if self.retriever_config.retriever_type == "bm25":
            self._log("📚 Initializing BM25 retriever...")
            self.retriever = BM25Retriever(
                es_host=self.retriever_config.es_host,
                index_name=self.retriever_config.es_index_name
            )
            
            # Index corpus if index doesn't exist
            if not self.retriever.is_existing_index():
                self._log("  Creating BM25 index...")
                self.retriever.create_index()
                self.retriever.index_corpus(
                    corpus_file=self.retriever_config.corpus_path,
                    corpus_data=self.retriever_config.corpus_data
                )
                self._log("  ✅ BM25 index created")
            else:
                self._log("  ✅ Using existing BM25 index")
                
        elif self.retriever_config.retriever_type == "faiss":
            if not FAISS_AVAILABLE:
                raise ImportError(
                    "FAISS retriever requires faiss-cpu or faiss-gpu. "
                    "Install with: pip install faiss-cpu"
                )
            self._log("🔍 Initializing FAISS retriever...")
            self.retriever = FAISSRetriever(
                model_path=self.retriever_config.embedding_model,
                base_url=self.retriever_config.embedding_base_url,
                index_path=self.retriever_config.index_path
            )
            
            # Build FAISS index if it doesn't exist
            index_path = Path(self.retriever_config.index_path)
            if not index_path.exists():
                self._log("  Building FAISS index...")
                asyncio.run(self.retriever.index_corpus(
                    corpus_file=self.retriever_config.corpus_path,
                    corpus_data=self.retriever_config.corpus_data,
                    hnsw_m=self.retriever_config.hnsw_m,
                    ef_construction=self.retriever_config.ef_construction,
                    ef_search=self.retriever_config.ef_search
                ))
                self._log("  ✅ FAISS index built")
            else:
                self._log("  ✅ Using existing FAISS index")
                
        elif self.retriever_config.retriever_type == "custom":
            self._log("🔧 Using custom retriever...")
            self.retriever = self.retriever_config.custom_retriever
        
        elif self.retriever_config.retriever_type == "pageindex":
            if not PAGEINDEX_AVAILABLE:
                raise ImportError(
                    "PageIndex retriever requires pageindex package. "
                    "Install with: pip install pageindex"
                )
            self._log("🌲 Initializing PageIndex retriever (reasoning-based tree search)...")
            self.retriever = PageIndexRetriever(
                model=self.retriever_config.pageindex_model,
                openai_api_key=self.retriever_config.pageindex_openai_api_key,
                tree_cache_dir=self.retriever_config.pageindex_tree_cache_dir,
                verbose=self.verbose
            )
            
            # Index documents or load tree structures
            if self.retriever_config.pageindex_document_paths:
                self._log("  Building tree structures from PDF documents...")
                self.retriever.index_documents(self.retriever_config.pageindex_document_paths)
                self._log("  ✅ PageIndex trees built")
            elif self.retriever_config.pageindex_tree_paths:
                self._log("  Loading pre-built tree structures...")
                self.retriever.load_tree_structures(self.retriever_config.pageindex_tree_paths)
                self._log("  ✅ Tree structures loaded")
            elif self.retriever_config.corpus_path or self.retriever_config.corpus_data:
                self._log("  Using corpus data directly...")
                self.retriever.index_corpus(
                    corpus_file=self.retriever_config.corpus_path,
                    corpus_data=self.retriever_config.corpus_data
                )
                self._log("  ✅ Corpus loaded")
        
        elif self.retriever_config.retriever_type == "mem0":
            if not MEM0_AVAILABLE:
                raise ImportError(
                    "mem0 retriever requires mem0ai package. "
                    "Install with: pip install mem0ai"
                )
            self._log("🧠 Initializing mem0 retriever...")
            self.retriever = Mem0Retriever(
                memory=self.retriever_config.mem0_memory,
                config=self.retriever_config.mem0_config,
                use_client=self.retriever_config.mem0_use_client,
                api_key=self.retriever_config.mem0_api_key,
            )
            
            # Store mem0 search parameters for later use
            self._mem0_user_id = self.retriever_config.mem0_user_id
            self._mem0_agent_id = self.retriever_config.mem0_agent_id
            self._mem0_run_id = self.retriever_config.mem0_run_id
            self._mem0_threshold = self.retriever_config.mem0_threshold
            self._mem0_rerank = self.retriever_config.mem0_rerank
            
            # Load existing memories as corpus if no corpus_path/data provided
            if not self.retriever_config.corpus_path and not self.retriever_config.corpus_data:
                self._log("  Loading memories from mem0...")
                self.corpus = self.retriever.load_corpus_from_memories(
                    user_id=self._mem0_user_id,
                    agent_id=self._mem0_agent_id,
                    run_id=self._mem0_run_id,
                )
                self._log(f"  ✅ Loaded {len(self.corpus)} memories from mem0")
            else:
                # Index provided corpus into mem0
                self._log("  Indexing corpus into mem0...")
                self.retriever.index_corpus(
                    corpus_file=self.retriever_config.corpus_path,
                    corpus_data=self.retriever_config.corpus_data,
                    user_id=self._mem0_user_id,
                    agent_id=self._mem0_agent_id,
                    run_id=self._mem0_run_id,
                )
                self._log("  ✅ Corpus indexed into mem0")
        
        else:
            raise ValueError(f"Unsupported retriever type: {self.retriever_config.retriever_type}")
    
    def _load_corpus(self):
        """Load corpus documents into memory."""
        # For mem0, corpus may already be loaded during retriever setup
        if self.retriever_config.retriever_type == "mem0" and self.corpus:
            self._log("📖 Using corpus loaded from mem0 memories...")
            # Use mem0's corpus map
            self.corpus_map = self.retriever.get_corpus_map()
            self._log(f"  ✅ {len(self.corpus)} memories available")
        # For PageIndex, use the corpus from the retriever
        elif self.retriever_config.retriever_type == "pageindex":
            self._log("📖 Using corpus from PageIndex tree nodes...")
            self.corpus = self.retriever.get_corpus()
            self.corpus_map = self.retriever.get_corpus_map()
            self._log(f"  ✅ {len(self.corpus)} tree nodes available")
        elif self.retriever_config.corpus_data:
            self.corpus = self.retriever_config.corpus_data
        elif self.retriever_config.corpus_path:
            self._log(f"📖 Loading corpus from {self.retriever_config.corpus_path}...")
            with open(self.retriever_config.corpus_path, 'r') as f:
                self.corpus = [json.loads(line) for line in f]
            self._log(f"  ✅ Loaded {len(self.corpus)} documents")
        elif self.retriever_config.retriever_type == "mem0":
            # mem0 with no corpus - this is fine, memories loaded during setup
            self._log("  Using mem0 memories as corpus")
            self.corpus_map = self.retriever.get_corpus_map()
            return
        else:
            raise ValueError("Either corpus_path or corpus_data must be provided")
        
        # Create mapping from doc_id to document (skip for mem0 which uses memory IDs)
        if self.retriever_config.retriever_type == "mem0":
            # For mem0, merge file corpus into memory cache
            if self.retriever_config.corpus_path or self.retriever_config.corpus_data:
                self.corpus_map = self.retriever.get_corpus_map()
        else:
            self.corpus_map = {}
            for doc in self.corpus:
                doc_id = doc.get("_id") or doc.get("document_id") or doc.get("chunk_id")
                if doc_id is not None:
                    self.corpus_map[str(doc_id)] = doc
    
    def retrieve(
        self,
        queries: Union[str, List[str], List[Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for queries.
        
        Args:
            queries: Single query string, list of query strings, or list of query dicts
            top_k: Number of documents to retrieve (overrides config)
        
        Returns:
            List of retrieval results, each containing:
                - qid: Query ID
                - text: Query text
                - top_k_doc_id: List of retrieved document IDs
                - answer: Ground truth answers (if provided)
        """
        if not self.retriever:
            self.setup()
        
        # Normalize queries to list of dicts
        if isinstance(queries, str):
            queries = [{"qid": 0, "text": queries}]
        elif isinstance(queries, list) and queries and isinstance(queries[0], str):
            queries = [{"qid": i, "text": q} for i, q in enumerate(queries)]
        elif isinstance(queries, list) and queries and isinstance(queries[0], dict):
            # Ensure each query has required fields
            for i, q in enumerate(queries):
                if "qid" not in q:
                    q["qid"] = i
                if "text" not in q and "question" in q:
                    q["text"] = q["question"]
        else:
            raise ValueError("Invalid queries format")
        
        top_k = top_k or self.retriever_config.top_k
        
        self._log(f"🔍 Retrieving documents for {len(queries)} queries (top_k={top_k})...")
        start_time = time.time()
        
        # Perform retrieval
        if self.retriever_config.retriever_type == "bm25":
            results = self.retriever.search_queries(
                query_data=queries,
                top_k=top_k
            )
        elif self.retriever_config.retriever_type == "faiss":
            results = asyncio.run(self.retriever.search_queries(
                query_data=queries,
                top_k=top_k
            ))
        elif self.retriever_config.retriever_type == "mem0":
            # mem0 retriever with user/agent/run scoping
            results = self.retriever.search_queries(
                query_data=queries,
                top_k=top_k,
                user_id=self._mem0_user_id,
                agent_id=self._mem0_agent_id,
                run_id=self._mem0_run_id,
                threshold=self._mem0_threshold,
                rerank=self._mem0_rerank,
            )
            # Update corpus_map with any new memories found during search
            self.corpus_map = self.retriever.get_corpus_map()
        else:
            # Custom retriever
            results = self.retriever.retrieve(queries, top_k=top_k)
        
        elapsed = time.time() - start_time
        self._log(f"  ✅ Retrieval completed in {elapsed:.2f}s")
        
        # Ensure results have all required fields
        for i, result in enumerate(results):
            if "qid" not in result:
                result["qid"] = queries[i]["qid"]
            if "answer" not in result and "answer" in queries[i]:
                result["answer"] = queries[i]["answer"]
        
        return results
    
    def optimize(
        self,
        retrieval_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize context ordering using ContextPilot.
        
        Args:
            retrieval_results: List of retrieval results
        
        Returns:
            Dictionary containing:
                - groups: Optimized groups of queries
                - original_results: Original retrieval results
                - reordered_contexts: Reordered document lists
                - metadata: Optimization metadata
        """
        if not self.optimizer_config.enabled:
            self._log("⚠️  ContextPilot optimization disabled, returning original order")
            # Return results in simple format without grouping
            return {
                "groups": [{
                    "group_id": 0,
                    "group_size": len(retrieval_results),
                    "items": retrieval_results
                }],
                "original_results": retrieval_results,
                "metadata": {"optimized": False}
            }
        
        self._log("🔧 Optimizing context ordering with ContextPilot...")
        start_time = time.time()
        
        # Extract contexts and metadata
        contexts = [result["top_k_doc_id"] for result in retrieval_results]
        qids = [result["qid"] for result in retrieval_results]
        questions = [result.get("text", result.get("question", "")) for result in retrieval_results]
        answers = [result.get("answer", []) for result in retrieval_results]
        
        # Build context index
        self._log("  Building context index...")
        result = build_context_index(
            contexts,
            linkage_method=self.optimizer_config.linkage_method,
            alpha=self.optimizer_config.alpha,
            use_gpu=self.optimizer_config.use_gpu
        )
        
        # Schedule contexts
        self._log("  Scheduling contexts...")
        scheduler = InterContextScheduler()
        organized_reordered_ids, organized_original_ids, final_index_mapping, all_groups_with_scores = scheduler.schedule_contexts(result)
        
        # Build output groups
        groups = []
        for group_id, (score, group_indices) in enumerate(all_groups_with_scores):
            items = []
            for idx in group_indices:
                mapped_idx = final_index_mapping.index(idx)
                items.append({
                    "qid": qids[idx],
                    "text": questions[idx],
                    "question": questions[idx],  # Add 'question' field for prompt_generator
                    "answer": answers[idx],
                    "top_k_doc_id": organized_reordered_ids[mapped_idx],
                    "orig_top_k_doc_id": organized_original_ids[mapped_idx]
                })
            
            groups.append({
                "group_id": group_id,
                "group_size": len(items),
                "items": items
            })
        
        # Sort groups by size
        groups.sort(key=lambda x: x['group_size'], reverse=True)
        
        elapsed = time.time() - start_time
        self._log(f"  ✅ Optimization completed in {elapsed:.2f}s")
        self._log(f"  📊 Created {len(groups)} groups")
        
        return {
            "groups": groups,
            "original_results": retrieval_results,
            "metadata": {
                "optimized": True,
                "num_groups": len(groups),
                "linkage_method": self.optimizer_config.linkage_method,
                "use_gpu": self.optimizer_config.use_gpu,
                "optimization_time": elapsed
            }
        }
    
    async def _async_request_completions(
        self,
        prompt: str,
        api_url: str,
        max_tokens: int,
        extra_request_body: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a single request to the completion API.
        
        Args:
            prompt: The prompt text
            api_url: API endpoint URL
            max_tokens: Maximum tokens to generate
            extra_request_body: Additional request parameters
            request_id: Optional request ID for token tracking
        
        Returns:
            Dictionary with response data including:
                - generated_text: Generated text
                - success: Whether request succeeded
                - latency: Total request latency
                - ttft: Time to first token
                - output_len: Number of output tokens
                - error: Error message if failed
        """
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            payload = {
                "model": self.inference_config.model_name,
                "prompt": prompt,
                "temperature": self.inference_config.temperature,
                "max_tokens": max_tokens,
                "stream": False,
                "ignore_eos": False,
                **extra_request_body,
            }
            
            # Add rid for request tracking in SGLang's radix cache
            # SGLang uses 'rid' field to identify requests
            if request_id:
                payload["rid"] = request_id
            
            output = {
                "generated_text": "",
                "success": False,
                "latency": 0.0,
                "ttft": 0.0,
                "output_len": 0,
                "error": ""
            }
            
            generated_text = ""
            output_len = 0
            ttft = 0.0
            st = time.perf_counter()
            
            try:
                async with session.post(url=api_url, json=payload) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue
                            
                            chunk = chunk_bytes.decode("utf-8")
                            if chunk.startswith("data: "):
                                chunk = chunk[6:]  # Remove "data: " prefix
                            
                            latency = time.perf_counter() - st
                            
                            if chunk == "[DONE]":
                                pass
                            else:
                                data = json.loads(chunk)
                                
                                if data["choices"][0]["text"]:
                                    # First token
                                    if ttft == 0.0:
                                        ttft = time.perf_counter() - st
                                    
                                    generated_text += data["choices"][0]["text"]
                                    output_len = (data.get("usage") or {}).get(
                                        "completion_tokens", output_len
                                    )

                        output["generated_text"] = generated_text
                        output["success"] = True
                        output["latency"] = latency
                        output["ttft"] = ttft
                        output["output_len"] = output_len
                    else:
                        output["error"] = response.reason or ""
                        output["success"] = False
            except Exception:
                output["success"] = False
                exc_info = sys.exc_info()
                output["error"] = "".join(traceback.format_exception(*exc_info))
            
            return output
    
    async def _inference(
        self,
        prompts: List[str],
        api_url: str,
        max_tokens: int,
        batch_size: Optional[int] = None,
        extra_request_body: Optional[Dict[str, Any]] = None,
        request_ids: Optional[List[str]] = None
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Send all prompts to the API and collect results.
        
        Args:
            prompts: List of prompts to send
            api_url: API endpoint URL
            max_tokens: Maximum tokens to generate
            batch_size: Number of requests per batch. If None, send all at once.
                       When set, will send next batch after 50% of current batch completes.
            extra_request_body: Additional parameters for the API request
            request_ids: Optional list of request IDs for token tracking (one per prompt)
        
        Returns:
            Tuple of (results, total_time) where total_time is end-to-end latency in seconds
        """
        if extra_request_body is None:
            extra_request_body = {}
        
        # If no batch_size specified, send all at once
        if batch_size is None or batch_size >= len(prompts):
            tasks = []
            
            for i, prompt in enumerate(prompts):
                req_id = request_ids[i] if request_ids and i < len(request_ids) else None
                tasks.append(
                    self._async_request_completions(
                        prompt=prompt,
                        api_url=api_url,
                        max_tokens=max_tokens,
                        extra_request_body=extra_request_body,
                        request_id=req_id
                    )
                )
            
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_time
            
            return results, total_time
        
        # Batched execution with 50% overlap
        all_results = [None] * len(prompts)
        
        start_time = time.perf_counter()
        
        batch_start_idx = 0
        active_tasks = {}  # Maps task to its original index
        
        while batch_start_idx < len(prompts) or active_tasks:
            # Launch new batch if we haven't processed all prompts yet
            if batch_start_idx < len(prompts):
                batch_end_idx = min(batch_start_idx + batch_size, len(prompts))
                
                for idx in range(batch_start_idx, batch_end_idx):
                    prompt = prompts[idx]
                    req_id = request_ids[idx] if request_ids and idx < len(request_ids) else None
                    task = asyncio.create_task(
                        self._async_request_completions(
                            prompt=prompt,
                            api_url=api_url,
                            max_tokens=max_tokens,
                            extra_request_body=extra_request_body,
                            request_id=req_id
                        )
                    )
                    active_tasks[task] = idx
                
                current_batch_size = batch_end_idx - batch_start_idx
                threshold = current_batch_size // 2  # 50% of current batch
                completed_in_batch = 0
                
                # Wait until 50% of current batch is complete before launching next batch
                while completed_in_batch < threshold and active_tasks:
                    done, pending = await asyncio.wait(
                        active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for task in done:
                        result = await task
                        idx = active_tasks.pop(task)
                        all_results[idx] = result
                        
                        # Count if this task belongs to the current batch
                        if batch_start_idx <= idx < batch_end_idx:
                            completed_in_batch += 1
                
                batch_start_idx = batch_end_idx
            
            else:
                # No more batches to launch, just wait for remaining tasks
                done, pending = await asyncio.wait(
                    active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    result = await task
                    idx = active_tasks.pop(task)
                    all_results[idx] = result
        
        total_time = time.perf_counter() - start_time
        
        return all_results, total_time
    
    def generate(
        self,
        optimized_results: Dict[str, Any],
        prompts: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        **inference_kwargs
    ) -> Dict[str, Any]:
        """
        Generate responses using LLM inference.
        
        By default, this method automatically generates prompts with:
        - Retrieved document context (reordered for optimal cache sharing)
        - Document importance ranking (based on original retrieval order)
        - Formatted instructions for the LLM
        
        Args:
            optimized_results: Results from optimize() method
            prompts: List of prompts (if None, will be auto-generated with full RAG context)
            max_tokens: Maximum tokens to generate (overrides config)
            batch_size: Number of requests per batch (None = send all at once)
            **inference_kwargs: Additional inference parameters
        
        Returns:
            Dictionary containing:
                - results: List of generation results with generated_text
                - metadata: Generation statistics (throughput, latency, etc.)
        """
        if not self.inference_config:
            raise ValueError("Inference configuration not provided. Set 'model' or 'inference' parameter.")
        
        # Generate prompts with document context if not provided
        if prompts is None:
            if not self.corpus_map:
                raise ValueError("Corpus not loaded. Cannot generate prompts with document context.")
            
            # Create chunk_id_text_dict from corpus_map
            chunk_id_text_dict = {
                chunk_id: doc.get("text", doc.get("content", ""))
                for chunk_id, doc in self.corpus_map.items()
            }
            
            # Determine if we should apply chat template
            apply_template = self.inference_config.apply_chat_template if self.inference_config else False
            model_name = self.inference_config.model_name if self.inference_config else None
            system_prompt = self.inference_config.system_prompt if self.inference_config else None
            
            # Generate prompts for all groups using prompt_generator
            prompts = []
            all_qids = []
            all_answers = []
            
            for group in optimized_results["groups"]:
                group_prompts, group_qids, group_answers = prompt_generator(
                    chunk_id_text_dict, 
                    group["items"],
                    model_name=model_name,
                    system_prompt=system_prompt,
                    apply_template=apply_template
                )
                prompts.extend(group_prompts)
                all_qids.extend(group_qids)
                all_answers.extend(group_answers)
        
        if not prompts:
            raise ValueError("No prompts found in optimized_results")
        
        # Initialize request_ids (will be populated from build response)
        request_ids = None
        
        # BUILD INDEX FIRST - extract contexts from optimized results
        try:
            contexts = []
            for group in optimized_results["groups"]:
                for item in group["items"]:
                    # Each item has a context (list of doc IDs)
                    # The key is "top_k_doc_id" from the optimize method
                    if "top_k_doc_id" in item:
                        contexts.append(item["top_k_doc_id"])
                    elif "context" in item:
                        contexts.append(item["context"])
                    elif "deduplicated_docs" in item:
                        contexts.append(item["deduplicated_docs"])
                    elif "retrieved_docs" in item:
                        contexts.append(item["retrieved_docs"])
            
            if contexts:
                build_url = f"{self.inference_config.base_url}/build"
                # Use incremental mode if index might already exist
                incremental = getattr(self, '_index_built', False)
                mode_str = "incremental" if incremental else "initial"
                self._log(f"📦 Building index ({mode_str}) with {len(contexts)} contexts at {build_url}")
                
                import requests
                build_response = requests.post(
                    build_url,
                    json={
                        "contexts": contexts,
                        "initial_tokens_per_context": 100,
                        "alpha": 0.005,
                        "use_gpu": False,
                        "linkage_method": "average",
                        "incremental": incremental
                    },
                    timeout=60
                )
                
                if build_response.status_code == 200:
                    build_result = build_response.json()
                    # Use the ordered request_ids list that matches input contexts order
                    request_ids = build_result.get("request_ids", [])
                    if not request_ids:
                        # Fallback to dict keys if ordered list not available
                        request_id_mapping = build_result.get("request_id_mapping", {})
                        request_ids = list(request_id_mapping.keys())
                    
                    # Log match stats for incremental builds
                    matched = build_result.get("matched_count", 0)
                    inserted = build_result.get("inserted_count", len(request_ids))
                    self._log(f"✅ Index built: {len(request_ids)} request IDs (matched={matched}, inserted={inserted})")
                    
                    # Mark index as built for subsequent calls
                    self._index_built = True
                else:
                    self._log(f"⚠️  Index build failed: {build_response.status_code} - {build_response.text}")
                    request_ids = None
        except Exception as e:
            self._log(f"⚠️  Failed to build index (continuing anyway): {e}")
            request_ids = None
        
        # Set parameters
        max_tokens = max_tokens or self.inference_config.max_tokens
        # If batch_size is not provided, use None to send all at once
        # If batch_size is explicitly set in config or parameter, use it
        if batch_size is None and hasattr(self.inference_config, 'batch_size') and self.inference_config.batch_size > 1:
            batch_size = self.inference_config.batch_size
        
        # Construct API URL
        api_url = f"{self.inference_config.base_url}/v1/completions"
        
        # Run inference with request_ids for token tracking
        results, total_time = asyncio.run(
            self._inference(
                prompts=prompts,
                api_url=api_url,
                max_tokens=max_tokens,
                batch_size=batch_size,
                extra_request_body=inference_kwargs,
                request_ids=request_ids
            )
        )
        
        # Calculate statistics
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        # Calculate throughput metrics
        metadata = {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "total_time": total_time,
        }
        
        if successful_results:
            total_output_tokens = sum(r["output_len"] for r in successful_results)
            total_ttft = sum(r["ttft"] for r in successful_results)
            total_decode_time = sum(r["latency"] - r["ttft"] for r in successful_results)
            
            metadata["total_output_tokens"] = total_output_tokens
            metadata["average_output_length"] = total_output_tokens / len(successful_results)
            
            if total_decode_time > 0:
                metadata["decode_throughput_tokens_per_sec"] = total_output_tokens / total_decode_time
            
            if total_ttft > 0:
                metadata["average_ttft"] = total_ttft / len(successful_results)
        
        return {
            "results": results,
            "metadata": metadata
        }
    
    def run(
        self,
        queries: Union[str, List[str], List[Dict[str, Any]]],
        top_k: Optional[int] = None,
        return_intermediate: bool = False,
        generate_responses: bool = False,
        max_tokens: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline: retrieve, optimize, and (optionally) generate.
        
        Args:
            queries: Queries to process
            top_k: Number of documents to retrieve
            return_intermediate: Whether to return intermediate results
            generate_responses: Whether to generate responses using LLM
            max_tokens: Maximum tokens to generate (if generate_responses=True)
            batch_size: Batch size for generation (if generate_responses=True)
        
        Returns:
            Dictionary containing optimized results and optionally generated responses
        """
        self._log("=" * 60)
        self._log("🚀 Running RAG Pipeline")
        self._log("=" * 60)
        
        pipeline_start = time.time()
        
        # Step 1: Retrieval
        retrieval_results = self.retrieve(queries, top_k=top_k)
        
        # Step 2: Optimization
        optimized_results = self.optimize(retrieval_results)
        
        # Step 3: (Optional) Inference
        inference_results = None
        if generate_responses:
            if not self.inference_config:
                self._log("⚠️  Inference config not provided, skipping generation")
            else:
                inference_results = self.generate(
                    optimized_results,
                    max_tokens=max_tokens,
                    batch_size=batch_size
                )
        
        pipeline_elapsed = time.time() - pipeline_start
        
        self._log("=" * 60)
        self._log(f"✅ Pipeline completed in {pipeline_elapsed:.2f}s")
        self._log("=" * 60)
        
        result = {
            "optimized_batch": optimized_results["groups"],
            "metadata": {
                **optimized_results.get("metadata", {}),
                "total_time": pipeline_elapsed,
                "num_queries": len(queries) if isinstance(queries, list) else 1
            }
        }
        
        if inference_results:
            result["generation_results"] = inference_results["results"]
            result["metadata"]["generation_stats"] = inference_results["metadata"]
        
        if return_intermediate:
            result["retrieval_results"] = retrieval_results
            result["original_results"] = optimized_results["original_results"]
        
        return result
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        format: str = "jsonl"
    ):
        """
        Save pipeline results to file.
        
        Args:
            results: Results from run() method
            output_path: Path to output file
            format: Output format ("jsonl" or "json")
        """
        self._log(f"💾 Saving results to {output_path}...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(output_path, 'w') as f:
                for group in results["optimized_batch"]:
                    f.write(json.dumps(group) + "\n")
        elif format == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self._log(f"  ✅ Results saved")
    
    def process_conversation_turn(
        self,
        conversation_id: str,
        query: Union[str, Dict[str, Any]],
        top_k: Optional[int] = None,
        enable_deduplication: bool = True,
        generate_response: bool = False,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a single turn in a multi-turn conversation with context deduplication.
        
        This method maintains conversation state across turns and applies context
        deduplication to avoid redundant prefill computation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            query: User query (string or dict with 'text'/'question' field)
            top_k: Number of documents to retrieve
            enable_deduplication: Whether to apply context deduplication (True by default)
            generate_response: Whether to generate LLM response
            max_tokens: Maximum tokens for generation
        
        Returns:
            Dictionary containing:
                - query: The original query
                - retrieved_docs: List of retrieved document IDs
                - novel_docs: List of novel document IDs (not in history)
                - deduplicated_docs: List of deduplicated document IDs
                - context: Formatted context string
                - deduplication_stats: Statistics for this turn
                - conversation_stats: Cumulative stats for the conversation
                - response: Generated response (if generate_response=True)
        
        Example:
            >>> pipeline = RAGPipeline(retriever="bm25", corpus_path="corpus.jsonl")
            >>> 
            >>> # Turn 1
            >>> result1 = pipeline.process_conversation_turn(
            ...     conversation_id="conv_1",
            ...     query="What is machine learning?"
            ... )
            >>> print(f"Retrieved: {result1['deduplication_stats']['num_retrieved']}")
            >>> print(f"Novel: {result1['deduplication_stats']['num_novel']}")
            >>> 
            >>> # Turn 2 - deduplication kicks in
            >>> result2 = pipeline.process_conversation_turn(
            ...     conversation_id="conv_1",
            ...     query="How does it work?"
            ... )
            >>> print(f"Deduplicated: {result2['deduplication_stats']['num_deduplicated']}")
        """
        # Ensure pipeline is set up
        if self.corpus_map is None:
            self.setup()
        
        # Extract query text
        if isinstance(query, dict):
            query_text = query.get("text", query.get("question", ""))
            qid = query.get("qid")
        else:
            query_text = query
            qid = None
        
        self._log(f"🔄 Processing turn for conversation '{conversation_id}'")
        
        # Get conversation state to build context-aware query
        conv_state = self.multi_turn_manager.get_conversation(conversation_id)
        
        # Build retrieval query with conversation history
        if conv_state.messages:
            # Include previous Q&A pairs for context-aware retrieval
            history_context = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in conv_state.messages[-4:]  # Last 2 turns (Q&A pairs)
            ])
            retrieval_query = f"{history_context}\nuser: {query_text}"
            self._log(f"  🔗 Using conversation history for retrieval ({len(conv_state.messages)} messages)")
        else:
            retrieval_query = query_text
            self._log(f"  🆕 First turn - no history")
        
        # Step 1: Retrieve documents using context-aware query
        retrieval_result = self.retrieve([retrieval_query], top_k=top_k)[0]
        retrieved_doc_ids = retrieval_result["top_k_doc_id"]
        
        self._log(f"  📚 Retrieved {len(retrieved_doc_ids)} documents")
        
        # Step 2: Apply context deduplication
        context_str, novel_doc_ids, dedup_stats = self.multi_turn_manager.deduplicate_context(
            conversation_id=conversation_id,
            retrieved_doc_ids=retrieved_doc_ids,
            corpus_map=self.corpus_map,
            enable_deduplication=enable_deduplication
        )
        
        if enable_deduplication and dedup_stats["num_deduplicated"] > 0:
            self._log(
                f"  ♻️  Deduplicated {dedup_stats['num_deduplicated']} documents "
                f"({dedup_stats['deduplication_rate']:.1%})"
            )
        
        # Get conversation statistics
        conv_stats = self.multi_turn_manager.get_conversation_stats(conversation_id)
        
        # Build result
        result = {
            "conversation_id": conversation_id,
            "query": query_text,
            "qid": qid,
            "retrieved_docs": retrieved_doc_ids,
            "novel_docs": novel_doc_ids,
            "deduplicated_docs": [
                doc_id for doc_id in retrieved_doc_ids if doc_id not in novel_doc_ids
            ],
            "context": context_str,
            "deduplication_stats": dedup_stats,
            "conversation_stats": conv_stats
        }
        
        # Step 3: (Optional) Generate response
        if generate_response:
            if not self.inference_config:
                self._log("  ⚠️  Inference config not provided, skipping generation")
                result["response"] = None
            else:
                # Build prompt with context
                prompt_template = '''With the chat history and the following context, answer the question:
{context}
Question: {question}
'''
                prompt = prompt_template.format(context=context_str, question=query_text)
                
                # Generate response (single request)
                api_url = f"{self.inference_config.base_url}/v1/completions"
                gen_result = asyncio.run(self._async_request_completions(
                    prompt=prompt,
                    api_url=api_url,
                    max_tokens=max_tokens or self.inference_config.max_tokens,
                    extra_request_body={}
                ))
                
                if gen_result and gen_result["success"]:
                    result["response"] = gen_result["generated_text"]
                    result["generation_metadata"] = {
                        "ttft": gen_result["ttft"],
                        "latency": gen_result["latency"],
                        "output_len": gen_result["output_len"]
                    }
                    self._log(f"  ✅ Generated response ({gen_result['output_len']} tokens)")
                    
                    # Store query and response in conversation history
                    conv_state.add_message("user", query_text)
                    conv_state.add_message("assistant", gen_result["generated_text"])
                else:
                    result["response"] = None
                    result["generation_metadata"] = {"error": gen_result.get("error") if gen_result else "Unknown error"}
                    
                    # Store query even if generation failed
                    conv_state.add_message("user", query_text)
        else:
            # No generation, just store the query
            conv_state.add_message("user", query_text)
        
        return result
    
    def get_conversation_stats(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get deduplication statistics for conversations.
        
        Args:
            conversation_id: Specific conversation ID, or None for all conversations
        
        Returns:
            Statistics dictionary with deduplication metrics
        """
        if conversation_id:
            return self.multi_turn_manager.get_conversation_stats(conversation_id)
        else:
            return self.multi_turn_manager.get_all_stats()
    
    def reset_conversation(self, conversation_id: str):
        """
        Reset a specific conversation's history.
        
        Args:
            conversation_id: Conversation to reset
        """
        self.multi_turn_manager.reset_conversation(conversation_id)
        self._log(f"🔄 Reset conversation '{conversation_id}'")
    
    def reset_all_conversations(self):
        """Reset all conversation histories."""
        self.multi_turn_manager.reset_all()
        self._log("🔄 Reset all conversations")
