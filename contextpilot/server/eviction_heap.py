"""
Eviction Heap for LRU Tracking

Min-heap that tracks requests by last access time for efficient LRU eviction.
Mirrors SGLang's cache behavior with capacity-based eviction triggering.

Key Design:
- Tracks by request_id (leaf nodes only have request_id)
- Capacity-based eviction when total tokens exceed max_tokens
- Synchronized with context index tree structure
"""

import heapq
import time
from typing import Dict, Optional, List, Tuple, Set
from .metadata import NodeMetadata


class EvictionHeap:
    """
    Min-heap for tracking least recently used requests.
    
    Uses last_access_time as the priority key. Only tracks leaf nodes
    (which have request_id). Supports capacity-based eviction to mirror
    SGLang's cache eviction behavior.
    
    Key invariants:
    - Only leaf nodes (with request_id) are tracked
    - The heap and context index remain synchronized
    - Eviction removes tokens until below max_tokens capacity
    """
    
    def __init__(self, max_tokens: Optional[int] = None):
        """
        Initialize eviction heap.
        
        Args:
            max_tokens: Maximum token capacity (triggers eviction when exceeded)
        """
        self._heap: List[Tuple[float, int]] = []  # (access_time, node_id)
        self._metadata: Dict[int, NodeMetadata] = {}  # node_id -> metadata
        self._request_to_node: Dict[str, int] = {}  # request_id -> node_id
        self._in_heap: Dict[int, bool] = {}  # Track which nodes are in heap
        self._max_tokens = max_tokens
        self._total_tokens = 0
    
    @property
    def max_tokens(self) -> Optional[int]:
        """Get maximum token capacity."""
        return self._max_tokens
    
    @max_tokens.setter
    def max_tokens(self, value: int):
        """Set maximum token capacity."""
        self._max_tokens = value
    
    def push(self, metadata: NodeMetadata):
        """
        Add a node to the heap.
        
        Only leaf nodes (with request_id) should be added.
        Tracks extra_tokens (unique to this leaf) not total_tokens,
        since shared prefix tokens are only stored once in the cache.
        
        Args:
            metadata: Node metadata to track (must have request_id for leaf nodes)
        """
        node_id = metadata.node_id
        
        if node_id in self._in_heap and self._in_heap[node_id]:
            # Node already in heap - update access time and recalculate tokens
            old_metadata = self._metadata.get(node_id)
            if old_metadata:
                # Adjust token count for the difference (use extra_tokens)
                self._total_tokens += metadata.extra_tokens - old_metadata.extra_tokens
            self._metadata[node_id] = metadata
            self.update_access_time(node_id, metadata.last_access_time)
            return
        
        heapq.heappush(self._heap, (metadata.last_access_time, node_id))
        self._metadata[node_id] = metadata
        self._in_heap[node_id] = True
        # Track extra_tokens (unique to this leaf), not total_tokens
        self._total_tokens += metadata.extra_tokens
        
        # Track request_id -> node_id mapping (only for leaf nodes)
        if metadata.request_id:
            self._request_to_node[metadata.request_id] = node_id
    
    def pop(self) -> Optional[NodeMetadata]:
        """
        Remove and return the least recently used node.
        
        Returns:
            NodeMetadata of LRU node, or None if heap is empty
        """
        while self._heap:
            access_time, node_id = heapq.heappop(self._heap)
            
            # Skip if node was removed or is stale
            if node_id not in self._metadata:
                continue
            
            metadata = self._metadata[node_id]
            
            # Check if this is the current entry (not stale)
            if metadata.last_access_time == access_time:
                self._in_heap[node_id] = False
                # Subtract extra_tokens when popping (unique tokens only)
                self._total_tokens -= metadata.extra_tokens
                return metadata
            
            # Stale entry, continue to next
        
        return None
    
    def peek(self) -> Optional[NodeMetadata]:
        """
        View the least recently used node without removing.
        
        Returns:
            NodeMetadata of LRU node, or None if heap is empty
        """
        while self._heap:
            access_time, node_id = self._heap[0]
            
            if node_id not in self._metadata:
                heapq.heappop(self._heap)
                continue
            
            metadata = self._metadata[node_id]
            
            if metadata.last_access_time == access_time:
                return metadata
            
            # Stale entry, remove and continue
            heapq.heappop(self._heap)
        
        return None
    
    def update_access_time(self, node_id: int, new_time: Optional[float] = None):
        """
        Update a node's access time (lazy deletion approach).
        
        Args:
            node_id: Node to update
            new_time: New access time (defaults to current time)
        """
        if node_id not in self._metadata:
            return
        
        metadata = self._metadata[node_id]
        
        if new_time is None:
            new_time = time.time()
        
        metadata.last_access_time = new_time
        
        # Push new entry (old one will be filtered as stale)
        heapq.heappush(self._heap, (new_time, node_id))
    
    def remove(self, node_id: int):
        """
        Remove a node from tracking.
        
        Uses lazy deletion - actual removal happens during pop/peek.
        
        Args:
            node_id: Node to remove
        """
        if node_id in self._metadata:
            metadata = self._metadata[node_id]
            # Subtract extra_tokens (unique tokens only)
            self._total_tokens -= metadata.extra_tokens
            
            # Remove request_id mapping
            if metadata.request_id and metadata.request_id in self._request_to_node:
                del self._request_to_node[metadata.request_id]
            
            del self._metadata[node_id]
        
        if node_id in self._in_heap:
            self._in_heap[node_id] = False
    
    def get_node_by_request_id(self, request_id: str) -> Optional[NodeMetadata]:
        """
        Get node metadata by request_id.
        
        Args:
            request_id: The unique request identifier
            
        Returns:
            NodeMetadata if found, None otherwise
        """
        node_id = self._request_to_node.get(request_id)
        if node_id is not None:
            return self._metadata.get(node_id)
        return None
    
    def update_tokens_for_request(self, request_id: str, input_tokens: int, output_tokens: int) -> bool:
        """
        Accumulate tokens for a completed request.
        
        Called when a request completes and we need to track the total tokens
        (input + output) for that request in the eviction heap.
        
        Args:
            request_id: The unique request identifier
            input_tokens: Number of input tokens (prompt)
            output_tokens: Number of output tokens (generation)
            
        Returns:
            True if successful, False if request_id not found
        """
        metadata = self.get_node_by_request_id(request_id)
        if metadata is None:
            return False
        
        # Update total tokens
        old_tokens = metadata.total_tokens
        total_new = input_tokens + output_tokens
        delta = total_new - old_tokens
        
        metadata.total_tokens = total_new
        metadata.extra_tokens = max(0, metadata.extra_tokens + delta)
        metadata.update_access_time()
        
        # Update heap tracking
        self._total_tokens += delta
        heapq.heappush(self._heap, (metadata.last_access_time, metadata.node_id))
        
        return True
    
    def needs_eviction(self) -> bool:
        """
        Check if eviction is needed based on capacity.
        
        Returns:
            True if total_tokens exceeds max_tokens
        """
        if self._max_tokens is None:
            return False
        return self._total_tokens > self._max_tokens
    
    def tokens_to_evict(self) -> int:
        """
        Calculate how many tokens need to be evicted.
        
        Returns:
            Number of tokens to evict to get below capacity
        """
        if self._max_tokens is None or self._total_tokens <= self._max_tokens:
            return 0
        return self._total_tokens - self._max_tokens
    
    def get_metadata(self, node_id: int) -> Optional[NodeMetadata]:
        """Get metadata for a specific node."""
        return self._metadata.get(node_id)
    
    def is_empty(self) -> bool:
        """Check if heap has any active nodes."""
        return self.peek() is None
    
    def size(self) -> int:
        """Get number of active nodes in heap."""
        return len(self._metadata)
    
    def total_tokens(self) -> int:
        """Get total tokens across all tracked nodes."""
        return self._total_tokens
    
    def get_all_request_ids(self) -> Set[str]:
        """Get all tracked request IDs."""
        return set(self._request_to_node.keys())
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the heap.
        
        Returns:
            Dictionary with heap statistics
        """
        if not self._metadata:
            return {
                'size': 0,
                'total_tokens': 0,
                'max_tokens': self._max_tokens,
                'utilization_pct': 0,
                'avg_tokens_per_node': 0,
                'oldest_access_time': None,
                'newest_access_time': None,
                'num_requests': 0
            }
        
        access_times = [m.last_access_time for m in self._metadata.values()]
        utilization = (self._total_tokens / self._max_tokens * 100) if self._max_tokens else 0
        
        return {
            'size': len(self._metadata),
            'total_tokens': self._total_tokens,
            'max_tokens': self._max_tokens,
            'utilization_pct': utilization,
            'avg_tokens_per_node': self._total_tokens / len(self._metadata),
            'oldest_access_time': min(access_times),
            'newest_access_time': max(access_times),
            'num_requests': len(self._request_to_node)
        }
    
    def __len__(self):
        """Get number of active nodes."""
        return len(self._metadata)
    
    def __repr__(self):
        return (f"EvictionHeap(size={len(self._metadata)}, "
                f"total_tokens={self._total_tokens}, "
                f"max_tokens={self._max_tokens})")
