"""
Conversation Tracker for Multi-Turn Context Deduplication

Tracks document history across conversation turns and provides deduplication
to avoid sending the same documents multiple times in a conversation.

Usage:
    tracker = ConversationTracker()
    
    # Turn 1
    req_a_id = tracker.register_request(docs=[4, 3, 1])
    
    # Turn 2 (continuation of Turn 1)
    result = tracker.deduplicate(
        request_id=req_b_id,
        parent_request_id=req_a_id,
        docs=[4, 3, 2]
    )
    # result.new_docs = [2]
    # result.overlapping_docs = [4, 3]
    # result.reference_hints = ["Refer to Doc 4...", "Refer to Doc 3..."]
"""

import time
import logging
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of context deduplication."""
    
    # Original documents in this turn
    original_docs: List[int]
    
    # Documents that overlap with previous turns (to be deduplicated)
    overlapping_docs: List[int]
    
    # New documents not seen in previous turns
    new_docs: List[int]
    
    # Reference hints for overlapping documents
    reference_hints: List[str]
    
    # The deduplicated context (new docs only, preserving order)
    deduplicated_docs: List[int]
    
    # Map: doc_id -> which turn it first appeared in
    doc_source_turns: Dict[int, str] = field(default_factory=dict)
    
    # Whether this is a new conversation (no parent, Turn 1)
    is_new_conversation: bool = False


@dataclass
class RequestHistory:
    """History of a single request."""
    request_id: str
    docs: List[int]
    parent_request_id: Optional[str] = None
    turn_number: int = 1
    timestamp: float = field(default_factory=time.time)


class ConversationTracker:
    """
    Tracks conversation history for multi-turn context deduplication.
    
    Features:
    - Track documents sent per request
    - Track conversation chains (parent-child relationships)
    - Deduplicate contexts by removing already-seen documents
    - Generate reference hints for deduplicated documents
    """
    
    def __init__(self, hint_template: str = None):
        """
        Initialize the tracker.
        
        Args:
            hint_template: Template for reference hints.
                           Default: "Please refer to [Doc {doc_id}] from turn {turn_number}."
        """
        # request_id -> RequestHistory
        self._requests: Dict[str, RequestHistory] = {}
        
        # Template for generating reference hints
        self._hint_template = hint_template or "Please refer to [Doc {doc_id}] from the previous conversation turn."
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'total_dedup_calls': 0,
            'total_docs_deduplicated': 0,
        }
    
    def register_request(self, 
                         request_id: str, 
                         docs: List[int],
                         parent_request_id: Optional[str] = None) -> RequestHistory:
        """
        Register a request and its documents.
        
        Args:
            request_id: Unique identifier for this request
            docs: List of document IDs sent in this request
            parent_request_id: ID of the previous turn's request (if multi-turn)
            
        Returns:
            RequestHistory object
        """
        # Determine turn number
        turn_number = 1
        if parent_request_id and parent_request_id in self._requests:
            turn_number = self._requests[parent_request_id].turn_number + 1
        
        history = RequestHistory(
            request_id=request_id,
            docs=list(docs),
            parent_request_id=parent_request_id,
            turn_number=turn_number
        )
        
        self._requests[request_id] = history
        self._stats['total_requests'] += 1
        
        logger.debug(f"Registered request {request_id}: {len(docs)} docs, turn {turn_number}")
        
        return history
    
    def get_conversation_chain(self, request_id: str) -> List[RequestHistory]:
        """
        Get the full conversation chain leading to this request.
        
        Args:
            request_id: The current request ID
            
        Returns:
            List of RequestHistory objects from first turn to current, in order
        """
        chain = []
        current_id = request_id
        
        while current_id and current_id in self._requests:
            chain.append(self._requests[current_id])
            current_id = self._requests[current_id].parent_request_id
        
        # Reverse to get chronological order
        chain.reverse()
        return chain
    
    def get_all_previous_docs(self, parent_request_id: str) -> Tuple[Set[int], Dict[int, str]]:
        """
        Get all documents from previous turns in the conversation.
        
        Args:
            parent_request_id: The parent request ID
            
        Returns:
            Tuple of (set of all doc IDs, dict mapping doc_id to request_id where it first appeared)
        """
        all_docs = set()
        doc_sources = {}  # doc_id -> request_id where it first appeared
        
        chain = self.get_conversation_chain(parent_request_id)
        
        for history in chain:
            for doc_id in history.docs:
                if doc_id not in all_docs:
                    all_docs.add(doc_id)
                    doc_sources[doc_id] = history.request_id
        
        return all_docs, doc_sources
    
    def deduplicate(self,
                    request_id: str,
                    docs: List[int],
                    parent_request_id: Optional[str] = None,
                    hint_template: Optional[str] = None) -> DeduplicationResult:
        """
        Deduplicate documents for a new turn based on conversation history.
        
        Args:
            request_id: ID for this request
            docs: Documents retrieved for this turn
            parent_request_id: ID of the previous turn's request
            hint_template: Optional custom template for hints
            
        Returns:
            DeduplicationResult with new docs, overlapping docs, and reference hints
        """
        self._stats['total_dedup_calls'] += 1
        
        # If no parent, this is turn 1 - no deduplication needed
        if not parent_request_id or parent_request_id not in self._requests:
            # Register this request
            self.register_request(request_id, docs, parent_request_id=None)
            
            return DeduplicationResult(
                original_docs=docs,
                overlapping_docs=[],
                new_docs=docs,
                reference_hints=[],
                deduplicated_docs=docs,
                doc_source_turns={},
                is_new_conversation=True
            )
        
        # Get all docs from previous turns
        previous_docs, doc_sources = self.get_all_previous_docs(parent_request_id)
        
        # Separate into overlapping and new
        overlapping_docs = []
        new_docs = []
        doc_source_turns = {}
        
        for doc_id in docs:
            if doc_id in previous_docs:
                overlapping_docs.append(doc_id)
                doc_source_turns[doc_id] = doc_sources[doc_id]
            else:
                new_docs.append(doc_id)
        
        # Generate reference hints for overlapping docs
        template = hint_template or self._hint_template
        reference_hints = []
        
        for doc_id in overlapping_docs:
            source_request = doc_sources.get(doc_id)
            source_history = self._requests.get(source_request) if source_request else None
            turn_number = source_history.turn_number if source_history else "previous"
            
            hint = template.format(
                doc_id=doc_id,
                turn_number=turn_number,
                source_request=source_request or "previous"
            )
            reference_hints.append(hint)
        
        # Register this request with only the new docs (for future deduplication)
        # But store original docs for complete history
        self.register_request(request_id, docs, parent_request_id)
        
        # Update stats
        self._stats['total_docs_deduplicated'] += len(overlapping_docs)
        
        logger.info(f"Deduplication for {request_id}: "
                   f"{len(overlapping_docs)} overlapping, {len(new_docs)} new")
        
        return DeduplicationResult(
            original_docs=docs,
            overlapping_docs=overlapping_docs,
            new_docs=new_docs,
            reference_hints=reference_hints,
            deduplicated_docs=new_docs,
            doc_source_turns=doc_source_turns,
            is_new_conversation=False
        )
    
    def deduplicate_batch(self,
                          request_ids: List[str],
                          docs_list: List[List[int]],
                          parent_request_ids: Optional[List[Optional[str]]] = None,
                          hint_template: Optional[str] = None) -> List[DeduplicationResult]:
        """
        Deduplicate multiple requests at once.
        
        Args:
            request_ids: List of request IDs
            docs_list: List of document lists, one per request
            parent_request_ids: List of parent request IDs (None for turn 1)
            hint_template: Optional custom template for hints
            
        Returns:
            List of DeduplicationResult objects
        """
        if parent_request_ids is None:
            parent_request_ids = [None] * len(request_ids)
        
        results = []
        for req_id, docs, parent_id in zip(request_ids, docs_list, parent_request_ids):
            result = self.deduplicate(req_id, docs, parent_id, hint_template)
            results.append(result)
        
        return results
    
    def remove_request(self, request_id: str) -> bool:
        """
        Remove a request from tracking.
        
        Note: This will NOT update parent references of child requests.
        Use with caution.
        
        Args:
            request_id: The request to remove
            
        Returns:
            True if removed, False if not found
        """
        if request_id in self._requests:
            del self._requests[request_id]
            return True
        return False
    
    def clear_conversation(self, request_id: str) -> int:
        """
        Clear all requests in a conversation chain.
        
        Args:
            request_id: Any request in the conversation
            
        Returns:
            Number of requests removed
        """
        chain = self.get_conversation_chain(request_id)
        count = 0
        
        for history in chain:
            if self.remove_request(history.request_id):
                count += 1
        
        return count
    
    def reset(self):
        """Clear all tracked conversations."""
        self._requests.clear()
        self._stats = {
            'total_requests': 0,
            'total_dedup_calls': 0,
            'total_docs_deduplicated': 0,
        }
        logger.info("ConversationTracker reset")
    
    def get_stats(self) -> Dict:
        """Get tracking statistics."""
        return {
            **self._stats,
            'active_requests': len(self._requests),
        }
    
    def get_request_history(self, request_id: str) -> Optional[RequestHistory]:
        """Get history for a specific request."""
        return self._requests.get(request_id)


# Singleton instance for use across the server
_conversation_tracker: Optional[ConversationTracker] = None


def get_conversation_tracker() -> ConversationTracker:
    """Get the global conversation tracker instance."""
    global _conversation_tracker
    if _conversation_tracker is None:
        _conversation_tracker = ConversationTracker()
    return _conversation_tracker


def reset_conversation_tracker():
    """Reset the global conversation tracker."""
    global _conversation_tracker
    if _conversation_tracker is not None:
        _conversation_tracker.reset()
    else:
        _conversation_tracker = ConversationTracker()
