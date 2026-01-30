"""
Mem0 Memory Retriever for ContextPilot.

Integrates mem0's Memory system with ContextPilot to enable:
1. Retrieval from user conversation memory history
2. Context deduplication for overlapping memories
3. Reordering opportunities for memory-based contexts

mem0 stores and retrieves user conversation memories, making it ideal for
personalized RAG applications where historical context matters.
"""

import json
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

# mem0 is optional - only import if available
try:
    from mem0 import Memory, MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    Memory = None
    MemoryClient = None
    MEM0_AVAILABLE = False


class Mem0Retriever:
    """
    Retriever that uses mem0's Memory for context retrieval.
    
    This retriever fetches relevant memories from mem0's vector store
    and formats them for use in ContextPilot's RAG pipeline.
    
    Note on ID Mapping:
        mem0 uses UUID strings as memory IDs, but ContextPilot's reordering
        and deduplication work better with integer IDs. This retriever provides
        automatic ID mapping:
        - `use_integer_ids=True` (default): Assigns sequential integer IDs
        - `use_integer_ids=False`: Uses original mem0 UUID strings
        
        The mapping is maintained internally and can be accessed via:
        - `get_id_mapping()`: Returns {int_id: mem0_uuid} mapping
        - `get_reverse_mapping()`: Returns {mem0_uuid: int_id} mapping
    
    Examples:
        >>> # Basic usage with default configuration
        >>> retriever = Mem0Retriever()
        >>> results = retriever.search_queries(
        ...     query_data=[{"qid": 0, "text": "What did we discuss yesterday?"}],
        ...     user_id="user123",
        ...     top_k=10
        ... )
        
        >>> # Using an existing Memory instance
        >>> from mem0 import Memory
        >>> memory = Memory()
        >>> retriever = Mem0Retriever(memory=memory)
        
        >>> # Custom mem0 configuration
        >>> config = {
        ...     "llm": {"provider": "openai", "config": {"model": "gpt-4"}},
        ...     "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}}
        ... }
        >>> retriever = Mem0Retriever(config=config)
    """
    
    def __init__(
        self,
        memory: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        use_client: bool = False,
        api_key: Optional[str] = None,
        use_integer_ids: bool = True,
    ):
        """
        Initialize the Mem0 Retriever.
        
        Args:
            memory: Existing mem0 Memory or MemoryClient instance
            config: mem0 configuration dictionary (used if memory is None)
            use_client: If True, use MemoryClient for cloud API (requires api_key)
            api_key: API key for mem0 cloud service (required if use_client=True)
            use_integer_ids: If True, map mem0 UUIDs to sequential integer IDs
                            for better compatibility with ContextPilot's reordering
        """
        if not MEM0_AVAILABLE:
            raise ImportError(
                "mem0 package is required for Mem0Retriever. "
                "Install it with: pip install mem0ai"
            )
        
        if memory is not None:
            self.memory = memory
        elif use_client:
            if api_key is None:
                raise ValueError("api_key is required when use_client=True")
            self.memory = MemoryClient(api_key=api_key)
        elif config is not None:
            self.memory = Memory.from_config(config)
        else:
            self.memory = Memory()
        
        # ID mapping configuration
        self.use_integer_ids = use_integer_ids
        
        # ID mapping: integer ID -> mem0 UUID
        self._id_to_uuid: Dict[int, str] = {}
        # Reverse mapping: mem0 UUID -> integer ID
        self._uuid_to_id: Dict[str, int] = {}
        # Next available integer ID
        self._next_id: int = 0
        
        # Track indexed memories for corpus_map compatibility
        # Key is the mapped ID (int or str depending on use_integer_ids)
        self._memory_cache: Dict[Any, Dict[str, Any]] = {}
        self._corpus_loaded = False
    
    def _get_or_create_id(self, mem0_uuid: str) -> Union[int, str]:
        """
        Get or create an integer ID for a mem0 UUID.
        
        Args:
            mem0_uuid: The mem0 memory UUID
            
        Returns:
            Integer ID if use_integer_ids=True, otherwise the original UUID
        """
        if not self.use_integer_ids:
            return mem0_uuid
        
        if mem0_uuid in self._uuid_to_id:
            return self._uuid_to_id[mem0_uuid]
        
        # Assign new integer ID
        new_id = self._next_id
        self._next_id += 1
        self._id_to_uuid[new_id] = mem0_uuid
        self._uuid_to_id[mem0_uuid] = new_id
        return new_id
    
    def get_id_mapping(self) -> Dict[int, str]:
        """
        Get the integer ID to mem0 UUID mapping.
        
        Returns:
            dict: Mapping from integer IDs to mem0 UUIDs
        """
        return self._id_to_uuid.copy()
    
    def get_reverse_mapping(self) -> Dict[str, int]:
        """
        Get the mem0 UUID to integer ID mapping.
        
        Returns:
            dict: Mapping from mem0 UUIDs to integer IDs
        """
        return self._uuid_to_id.copy()
    
    def get_mem0_uuid(self, doc_id: Union[int, str]) -> Optional[str]:
        """
        Get the original mem0 UUID for a document ID.
        
        Args:
            doc_id: The document ID (integer or string)
            
        Returns:
            The original mem0 UUID, or None if not found
        """
        if isinstance(doc_id, int):
            return self._id_to_uuid.get(doc_id)
        return doc_id  # Already a UUID string
    
    def get_integer_id(self, mem0_uuid: str) -> Optional[int]:
        """
        Get the integer ID for a mem0 UUID.
        
        Args:
            mem0_uuid: The mem0 UUID
            
        Returns:
            The integer ID, or None if not found
        """
        return self._uuid_to_id.get(mem0_uuid)
    
    def add_memory(
        self,
        messages: Union[str, List[Dict[str, str]]],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a memory to the mem0 store.
        
        This is useful for building a memory corpus from conversation history.
        
        Args:
            messages: Message content or list of messages
            user_id: User identifier for memory scoping
            agent_id: Agent identifier for memory scoping
            run_id: Run identifier for memory scoping
            metadata: Additional metadata to store with the memory
        
        Returns:
            dict: Result from mem0 add operation
        """
        return self.memory.add(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
        )
    
    def index_corpus(
        self,
        corpus_file: Optional[str] = None,
        corpus_data: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        **kwargs
    ):
        """
        Index a corpus into mem0's memory store.
        
        Each document in the corpus is added as a memory entry.
        
        Args:
            corpus_file: Path to JSONL corpus file
            corpus_data: List of corpus documents
            user_id: User identifier for all memories
            agent_id: Agent identifier for all memories
            run_id: Run identifier for all memories
            **kwargs: Additional parameters (ignored for compatibility)
        """
        if not user_id and not agent_id and not run_id:
            raise ValueError(
                "At least one of 'user_id', 'agent_id', or 'run_id' must be provided "
                "for indexing memories"
            )
        
        assert corpus_file or corpus_data, "Either corpus_file or corpus_data must be provided."
        
        if corpus_file:
            with open(corpus_file, 'r') as f:
                corpus = [json.loads(line) for line in f]
        else:
            corpus = corpus_data
        
        print(f"📚 Indexing {len(corpus)} documents into mem0...")
        
        for doc in tqdm(corpus, desc="Indexing into mem0"):
            original_doc_id = doc.get("_id") or doc.get("document_id") or doc.get("chunk_id")
            text = doc.get("text", doc.get("content", ""))
            title = doc.get("title", "")
            
            # Create metadata for the memory
            metadata = {
                "original_doc_id": original_doc_id,  # Store original ID in metadata
                "title": title,
                "source": "corpus",
            }
            
            # Add any extra fields from the document
            for key, value in doc.items():
                if key not in ["_id", "document_id", "chunk_id", "text", "content", "title"]:
                    metadata[key] = value
            
            # Add to mem0
            result = self.memory.add(
                text,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                metadata=metadata,
                infer=False,  # Don't use LLM for fact extraction, store raw
            )
            
            # Cache the memory for later retrieval
            if result and len(result) > 0:
                mem0_uuid = result[0].get("id")
                if mem0_uuid:
                    # Get or create mapped ID
                    mapped_id = self._get_or_create_id(mem0_uuid)
                    cache_key = str(mapped_id)
                    self._memory_cache[cache_key] = {
                        "chunk_id": mapped_id,
                        "memory_id": mem0_uuid,
                        "original_doc_id": original_doc_id,
                        "text": text,
                        "content": text,  # Alias for compatibility
                        "title": title,
                        "metadata": metadata,
                    }
        
        self._corpus_loaded = True
        print(f"✅ Indexed {len(corpus)} documents into mem0")
    
    def load_corpus_from_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Load existing memories as a corpus for ContextPilot.
        
        This fetches all memories for the given user/agent/run and
        creates a corpus_map compatible format.
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            run_id: Run identifier
            limit: Maximum number of memories to fetch
        
        Returns:
            list: List of memory documents in corpus format
        """
        if not user_id and not agent_id and not run_id:
            raise ValueError(
                "At least one of 'user_id', 'agent_id', or 'run_id' must be provided"
            )
        
        result = self.memory.get_all(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit,
        )
        
        memories = result.get("results", [])
        corpus = []
        
        for mem in memories:
            mem0_uuid = mem.get("id")
            # Get or create mapped ID
            mapped_id = self._get_or_create_id(mem0_uuid)
            
            doc = {
                "chunk_id": mapped_id,  # Use mapped ID for ContextPilot compatibility
                "mem0_id": mem0_uuid,   # Keep original mem0 UUID for reference
                "text": mem.get("memory", ""),
                "created_at": mem.get("created_at"),
                "updated_at": mem.get("updated_at"),
            }
            
            # Add metadata fields
            if "metadata" in mem:
                doc.update(mem["metadata"])
            
            # Add user/agent/run info if present
            for key in ["user_id", "agent_id", "run_id", "actor_id", "role"]:
                if key in mem:
                    doc[key] = mem[key]
            
            corpus.append(doc)
            
            # Cache for later using mapped ID as key
            cache_key = str(mapped_id)
            self._memory_cache[cache_key] = {
                "chunk_id": mapped_id,
                "memory_id": mem0_uuid,
                "text": mem.get("memory", ""),
                "content": mem.get("memory", ""),  # Alias for compatibility
                "metadata": mem.get("metadata", {}),
            }
        
        self._corpus_loaded = True
        return corpus
    
    def get_corpus_map(self) -> Dict[Any, Dict[str, Any]]:
        """
        Get the corpus map for ContextPilot deduplication.
        
        The keys are mapped IDs (integers if use_integer_ids=True, UUIDs otherwise).
        Each value contains:
            - chunk_id: The mapped ID (same as key)
            - memory_id: Original mem0 UUID
            - text/content: Memory content
            - metadata: Additional metadata
        
        Returns:
            dict: Mapping from mapped doc_id to document content
        """
        return self._memory_cache
    
    def search_queries(
        self,
        queries_file: Optional[str] = None,
        query_data: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 20,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        threshold: Optional[float] = None,
        rerank: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search queries against mem0's memory store.
        
        This is the main search method compatible with ContextPilot's
        retriever interface.
        
        Args:
            queries_file: Path to queries JSONL file
            query_data: List of query documents
            top_k: Number of top memories to retrieve
            user_id: User identifier for search scoping
            agent_id: Agent identifier for search scoping
            run_id: Run identifier for search scoping
            threshold: Minimum score threshold for results
            rerank: Whether to use mem0's reranker (if configured)
            **kwargs: Additional parameters
        
        Returns:
            list: List of search results with format:
                  [{"qid": ..., "text": ..., "top_k_doc_id": [...], "memories": [...]}]
        """
        if not user_id and not agent_id and not run_id:
            raise ValueError(
                "At least one of 'user_id', 'agent_id', or 'run_id' must be provided"
            )
        
        assert queries_file or query_data, "Either queries_file or query_data must be provided."
        
        if queries_file:
            with open(queries_file, 'r') as f:
                queries = [json.loads(line) for line in f]
        else:
            queries = query_data
        
        results = []
        
        for query in tqdm(queries, desc="Searching mem0 memories"):
            # Extract query text
            if isinstance(query, dict):
                query_text = query.get("text", query.get("question", ""))
                qid = query.get("qid", query.get("id"))
                answer = query.get("answer", query.get("answers", query.get("ans")))
            else:
                query_text = str(query)
                qid = None
                answer = None
            
            # Search mem0
            search_result = self.memory.search(
                query_text,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                limit=top_k,
                threshold=threshold,
                rerank=rerank,
            )
            
            memories = search_result.get("results", [])
            
            # Map mem0 UUIDs to integer IDs for ContextPilot compatibility
            top_k_doc_ids = []
            for mem in memories:
                mem0_uuid = mem.get("id")
                mapped_id = self._get_or_create_id(mem0_uuid)
                top_k_doc_ids.append(mapped_id)
                
                # Cache memories for later corpus_map access using mapped ID
                cache_key = str(mapped_id)
                if cache_key not in self._memory_cache:
                    self._memory_cache[cache_key] = {
                        "chunk_id": mapped_id,
                        "memory_id": mem0_uuid,
                        "text": mem.get("memory", ""),
                        "content": mem.get("memory", ""),  # Alias for compatibility
                        "score": mem.get("score"),
                        "metadata": mem.get("metadata", {}),
                    }
            
            result = {
                "text": query_text,
                "top_k_doc_id": top_k_doc_ids,  # Now contains mapped integer IDs
                "memories": memories,  # Include full memory objects for advanced use
            }
            
            if qid is not None:
                result["qid"] = qid
            if answer is not None:
                result["answer"] = answer
            
            results.append(result)
        
        return results
    
    def is_existing_index(self) -> bool:
        """
        Check if memories exist for compatibility with ContextPilot's pipeline.
        
        For mem0, we always return True since the memory store is persistent.
        Use load_corpus_from_memories() to check if there are actual memories.
        
        Returns:
            bool: Always True for mem0
        """
        return True
    
    def create_index(self):
        """
        Create index method for compatibility with ContextPilot's pipeline.
        
        For mem0, this is a no-op since the memory store is managed internally.
        """
        pass  # mem0 manages its own vector store
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: The memory ID to retrieve
        
        Returns:
            dict: Memory content or None if not found
        """
        return self.memory.get(memory_id)
    
    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: The memory ID to delete
        
        Returns:
            dict: Result of deletion
        """
        return self.memory.delete(memory_id)
    
    def delete_all_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Delete all memories for a user/agent/run.
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            run_id: Run identifier
        
        Returns:
            dict: Result of deletion
        """
        return self.memory.delete_all(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )


def create_mem0_corpus_map(
    memory: Any,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: int = 1000,
    use_integer_ids: bool = True,
) -> Dict[Any, Dict[str, Any]]:
    """
    Utility function to create a corpus_map from mem0 memories.
    
    This is useful when you want to use mem0 memories with
    ContextPilot's deduplication features.
    
    Args:
        memory: mem0 Memory instance
        user_id: User identifier
        agent_id: Agent identifier
        run_id: Run identifier
        limit: Maximum number of memories to fetch
        use_integer_ids: If True, use sequential integer IDs instead of UUIDs
    
    Returns:
        dict: Corpus map suitable for ContextPilot (keys are integers if use_integer_ids=True)
    
    Example:
        >>> from mem0 import Memory
        >>> from contextpilot.retriever import create_mem0_corpus_map
        >>> 
        >>> memory = Memory()
        >>> corpus_map = create_mem0_corpus_map(memory, user_id="user123")
        >>> # corpus_map keys are integers: {0: {...}, 1: {...}, ...}
    """
    if not user_id and not agent_id and not run_id:
        raise ValueError(
            "At least one of 'user_id', 'agent_id', or 'run_id' must be provided"
        )
    
    result = memory.get_all(
        user_id=user_id,
        agent_id=agent_id,
        run_id=run_id,
        limit=limit,
    )
    
    corpus_map = {}
    for idx, mem in enumerate(result.get("results", [])):
        mem0_uuid = mem.get("id")
        # Use integer ID or UUID based on configuration
        mapped_id = idx if use_integer_ids else mem0_uuid
        map_key = str(mapped_id)
        
        corpus_map[map_key] = {
            "chunk_id": mapped_id,
            "memory_id": mem0_uuid,  # Always include original UUID
            "text": mem.get("memory", ""),
            "content": mem.get("memory", ""),  # Alias for compatibility
            "created_at": mem.get("created_at"),
            "updated_at": mem.get("updated_at"),
            "score": mem.get("score"),
        }
        
        # Add metadata fields
        if "metadata" in mem:
            corpus_map[map_key].update(mem["metadata"])
    
    return corpus_map
