"""
Node Metadata for Live Context Index

Tracks per-node information for dynamic index updates and LRU eviction.

Key Design:
- request_id is ONLY stored on leaf nodes (actual requests)
- Parent/intermediate nodes do NOT have request_id values
- Each request maintains a traversal path through the context index tree
"""

import time
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class NodeMetadata:
    """
    Metadata for each node in the live context index.
    
    Attributes:
        node_id: Unique identifier for the node
        request_id: Unique request identifier (ONLY for leaf nodes, None for internal nodes)
        total_tokens: Current sequence length (full context = input + output tokens)
        extra_tokens: Tokens beyond parent's prefix
        last_access_time: Timestamp for LRU tracking
        search_path: Child indices from root to this node
        is_active: Whether this node is currently in cache
        is_leaf: Whether this is a leaf node (has request_id)
        doc_ids: Document IDs in this node (for prefix matching)
    """
    
    node_id: int
    total_tokens: int = 0
    extra_tokens: int = 0
    last_access_time: float = field(default_factory=time.time)
    search_path: List[int] = field(default_factory=list)
    is_active: bool = True
    is_leaf: bool = False
    doc_ids: Optional[List[int]] = None
    request_id: Optional[str] = None  # Only set for leaf nodes
    
    def update_access_time(self):
        """Update last access time to current timestamp."""
        self.last_access_time = time.time()
    
    def add_tokens(self, delta: int):
        """
        Add tokens to this node (generation).
        
        Args:
            delta: Number of tokens to add
        """
        self.total_tokens += delta
        self.extra_tokens += delta
        self.update_access_time()
    
    def remove_tokens(self, delta: int) -> int:
        """
        Remove tokens from this node (eviction).
        
        Removes from extra_tokens first, then from total_tokens.
        
        Args:
            delta: Number of tokens to remove
            
        Returns:
            Actual number of tokens removed
        """
        if delta <= 0:
            return 0
        
        # Remove from extra tokens first
        tokens_removed = min(delta, self.extra_tokens)
        self.extra_tokens -= tokens_removed
        self.total_tokens -= tokens_removed
        
        # If more needed, remove from total (affects prefix)
        remaining = delta - tokens_removed
        if remaining > 0:
            actual_removed = min(remaining, self.total_tokens)
            self.total_tokens -= actual_removed
            tokens_removed += actual_removed
        
        return tokens_removed
    
    def is_empty(self) -> bool:
        """Check if node has zero tokens."""
        return self.total_tokens <= 0
    
    def __lt__(self, other):
        """
        Comparison for heap ordering (by last access time).
        Earlier access time = higher priority for eviction.
        """
        return self.last_access_time < other.last_access_time
    
    def __repr__(self):
        req_str = f", request_id={self.request_id}" if self.request_id else ""
        return (f"NodeMetadata(id={self.node_id}, "
                f"total_tokens={self.total_tokens}, "
                f"extra_tokens={self.extra_tokens}, "
                f"is_leaf={self.is_leaf}{req_str}, "
                f"active={self.is_active})")
