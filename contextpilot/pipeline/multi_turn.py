"""
Multi-turn conversation support with context deduplication.

Implements efficient multi-turn RAG conversations by:
1. Tracking retrieved documents across conversation turns
2. Deduplicating redundant contexts with location hints
3. Maintaining cumulative context history per conversation
"""

from typing import Dict, Set, List, Tuple, Any, Optional


class ConversationState:
    """
    Tracks state for a single multi-turn conversation.
    
    Maintains:
    - Retrieved document history across turns
    - Conversation messages
    - Deduplication statistics
    """
    
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.retrieved_history: Set[Any] = set()  # Document IDs seen so far
        self.messages: List[Dict[str, str]] = []  # Chat history
        self.turn_count: int = 0
        
        # Statistics
        self.total_retrieved: int = 0
        self.total_novel: int = 0
        self.total_deduplicated: int = 0
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.messages.append({"role": role, "content": content})
    
    def update_history(self, novel_doc_ids: List[Any]):
        """Update retrieved history with novel documents."""
        self.retrieved_history.update(novel_doc_ids)
        self.turn_count += 1
    
    def update_stats(self, num_retrieved: int, num_novel: int, num_deduplicated: int):
        """Update deduplication statistics."""
        self.total_retrieved += num_retrieved
        self.total_novel += num_novel
        self.total_deduplicated += num_deduplicated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            "conversation_id": self.conversation_id,
            "turn_count": self.turn_count,
            "total_retrieved": self.total_retrieved,
            "total_novel": self.total_novel,
            "total_deduplicated": self.total_deduplicated,
            "deduplication_rate": (
                self.total_deduplicated / self.total_retrieved 
                if self.total_retrieved > 0 else 0.0
            )
        }


class MultiTurnManager:
    """
    Manages multiple concurrent conversations with context deduplication.
    
    Each conversation maintains its own:
    - Document retrieval history
    - Message history
    - Deduplication statistics
    """
    
    def __init__(self):
        self.conversations: Dict[str, ConversationState] = {}
    
    def get_conversation(self, conversation_id: str) -> ConversationState:
        """Get or create a conversation state."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationState(conversation_id)
        return self.conversations[conversation_id]
    
    def deduplicate_context(
        self,
        conversation_id: str,
        retrieved_doc_ids: List[Any],
        corpus_map: Dict[Any, Any],
        enable_deduplication: bool = True
    ) -> Tuple[str, List[Any], Dict[str, int]]:
        """
        Build context with deduplication for a conversation turn.
        
        Args:
            conversation_id: Unique conversation identifier
            retrieved_doc_ids: List of document IDs retrieved in current turn
            corpus_map: Dictionary mapping doc IDs to document objects
            enable_deduplication: Whether to apply deduplication (False for baseline)
        
        Returns:
            Tuple of:
            - context_str: Formatted context string with hints/content
            - novel_doc_ids: List of novel document IDs
            - stats: Dictionary with deduplication statistics
        """
        conv = self.get_conversation(conversation_id)
        
        context_str = ""
        novel_doc_ids = []
        num_retrieved = len(retrieved_doc_ids)
        
        if enable_deduplication and conv.retrieved_history:
            # Identify overlapping documents (O(N) time)
            overlap_ids = set(retrieved_doc_ids) & conv.retrieved_history
            
            # Build context with deduplication
            for doc_id in retrieved_doc_ids:
                if doc_id in overlap_ids:
                    # Deduplicated: insert location hint
                    context_str += f"Please refer to [Doc_{doc_id}] in the previous conversation.\n\n"
                else:
                    # Novel: include full content
                    doc = corpus_map.get(str(doc_id))
                    if doc:
                        content = doc.get("text", doc.get("content", ""))
                        context_str += f"[Doc_{doc_id}]: {content}\n\n"
                    novel_doc_ids.append(doc_id)
        else:
            # First turn or baseline mode: no deduplication
            for doc_id in retrieved_doc_ids:
                doc = corpus_map.get(str(doc_id))
                if doc:
                    content = doc.get("text", doc.get("content", ""))
                    context_str += f"[Doc_{doc_id}]: {content}\n\n"
                novel_doc_ids.append(doc_id)
        
        # Calculate statistics
        num_novel = len(novel_doc_ids)
        num_deduplicated = num_retrieved - num_novel
        
        # Update conversation state
        if enable_deduplication:
            conv.update_history(novel_doc_ids)
        else:
            # Baseline: track all documents for comparison
            conv.update_history(retrieved_doc_ids)
        
        conv.update_stats(num_retrieved, num_novel, num_deduplicated)
        
        stats = {
            "num_retrieved": num_retrieved,
            "num_novel": num_novel,
            "num_deduplicated": num_deduplicated,
            "deduplication_rate": num_deduplicated / num_retrieved if num_retrieved > 0 else 0.0
        }
        
        return context_str, novel_doc_ids, stats
    
    def get_conversation_stats(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific conversation."""
        conv = self.conversations.get(conversation_id)
        return conv.get_stats() if conv else None
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics across all conversations."""
        if not self.conversations:
            return {
                "total_conversations": 0,
                "total_turns": 0,
                "total_retrieved": 0,
                "total_novel": 0,
                "total_deduplicated": 0,
                "average_deduplication_rate": 0.0
            }
        
        total_turns = sum(c.turn_count for c in self.conversations.values())
        total_retrieved = sum(c.total_retrieved for c in self.conversations.values())
        total_novel = sum(c.total_novel for c in self.conversations.values())
        total_deduplicated = sum(c.total_deduplicated for c in self.conversations.values())
        
        return {
            "total_conversations": len(self.conversations),
            "total_turns": total_turns,
            "total_retrieved": total_retrieved,
            "total_novel": total_novel,
            "total_deduplicated": total_deduplicated,
            "average_deduplication_rate": (
                total_deduplicated / total_retrieved if total_retrieved > 0 else 0.0
            ),
            "average_turns_per_conversation": (
                total_turns / len(self.conversations) if self.conversations else 0.0
            )
        }
    
    def reset_conversation(self, conversation_id: str):
        """Reset a specific conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def reset_all(self):
        """Reset all conversations."""
        self.conversations.clear()
