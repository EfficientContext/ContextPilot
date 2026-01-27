"""
Tests for Multi-Turn Conversation Support.

Tests the multi-turn conversation handling including
context deduplication, session management, and history tracking.
"""

import pytest
from typing import List, Dict


class TestConversationState:
    """Test ConversationState class."""
    
    def test_initialization(self):
        """Test conversation state initialization."""
        from contextpilot.pipeline.multi_turn import ConversationState
        
        state = ConversationState("conv_001")
        
        assert state.conversation_id == "conv_001"
        assert len(state.retrieved_history) == 0
        assert len(state.messages) == 0
        assert state.turn_count == 0
    
    def test_add_message(self):
        """Test adding messages to conversation."""
        from contextpilot.pipeline.multi_turn import ConversationState
        
        state = ConversationState("conv_001")
        
        state.add_message("user", "Hello!")
        state.add_message("assistant", "Hi there!")
        
        assert len(state.messages) == 2
        assert state.messages[0]["role"] == "user"
        assert state.messages[1]["role"] == "assistant"
    
    def test_update_history(self):
        """Test updating retrieved document history."""
        from contextpilot.pipeline.multi_turn import ConversationState
        
        state = ConversationState("conv_001")
        
        state.update_history([1, 2, 3])
        assert state.retrieved_history == {1, 2, 3}
        assert state.turn_count == 1
        
        state.update_history([4, 5])
        assert state.retrieved_history == {1, 2, 3, 4, 5}
        assert state.turn_count == 2
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        from contextpilot.pipeline.multi_turn import ConversationState
        
        state = ConversationState("conv_001")
        
        state.update_stats(num_retrieved=10, num_novel=8, num_deduplicated=2)
        
        stats = state.get_stats()
        
        assert stats["total_retrieved"] == 10
        assert stats["total_novel"] == 8
        assert stats["total_deduplicated"] == 2
    
    def test_deduplication_rate_calculation(self):
        """Test deduplication rate calculation."""
        from contextpilot.pipeline.multi_turn import ConversationState
        
        state = ConversationState("conv_001")
        
        state.update_stats(num_retrieved=10, num_novel=5, num_deduplicated=5)
        
        stats = state.get_stats()
        
        assert stats["deduplication_rate"] == 0.5
    
    def test_zero_retrieved_no_division_error(self):
        """Test handling of zero retrieved documents."""
        from contextpilot.pipeline.multi_turn import ConversationState
        
        state = ConversationState("conv_001")
        
        stats = state.get_stats()
        
        # Should not raise division by zero
        assert stats["deduplication_rate"] == 0.0


class TestMultiTurnManager:
    """Test MultiTurnManager class."""
    
    def test_initialization(self):
        """Test manager initialization."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        
        assert len(manager.conversations) == 0
    
    def test_get_conversation_creates_new(self):
        """Test that get_conversation creates new state if needed."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        
        conv = manager.get_conversation("new_conv")
        
        assert conv is not None
        assert conv.conversation_id == "new_conv"
    
    def test_get_conversation_returns_existing(self):
        """Test that get_conversation returns existing state."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        
        conv1 = manager.get_conversation("conv_001")
        conv1.add_message("user", "Hello")
        
        conv2 = manager.get_conversation("conv_001")
        
        assert conv1 is conv2
        assert len(conv2.messages) == 1


class TestContextDeduplication:
    """Test context deduplication functionality."""
    
    @pytest.fixture
    def corpus_map(self):
        """Create sample corpus map."""
        return {
            "1": {"text": "Document 1 content"},
            "2": {"text": "Document 2 content"},
            "3": {"text": "Document 3 content"},
            "4": {"text": "Document 4 content"},
            "5": {"text": "Document 5 content"},
        }
    
    def test_first_turn_no_deduplication(self, corpus_map):
        """Test that first turn has no deduplication."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        
        context_str, novel_ids, stats = manager.deduplicate_context(
            conversation_id="conv_001",
            retrieved_doc_ids=[1, 2, 3],
            corpus_map=corpus_map
        )
        
        assert len(novel_ids) == 3
        assert stats["num_deduplicated"] == 0
        assert "Document 1 content" in context_str
    
    def test_subsequent_turn_deduplicates(self, corpus_map):
        """Test that subsequent turns deduplicate overlapping docs."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        
        # First turn
        manager.deduplicate_context(
            conversation_id="conv_001",
            retrieved_doc_ids=[1, 2, 3],
            corpus_map=corpus_map
        )
        
        # Second turn with overlap
        context_str, novel_ids, stats = manager.deduplicate_context(
            conversation_id="conv_001",
            retrieved_doc_ids=[2, 3, 4],  # 2 and 3 overlap
            corpus_map=corpus_map
        )
        
        assert len(novel_ids) == 1  # Only 4 is novel
        assert stats["num_deduplicated"] == 2
        assert "refer to" in context_str.lower() or "previous" in context_str.lower()
    
    def test_deduplication_disabled(self, corpus_map):
        """Test that deduplication can be disabled."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        
        # First turn
        manager.deduplicate_context(
            conversation_id="conv_001",
            retrieved_doc_ids=[1, 2, 3],
            corpus_map=corpus_map
        )
        
        # Second turn with deduplication disabled
        context_str, novel_ids, stats = manager.deduplicate_context(
            conversation_id="conv_001",
            retrieved_doc_ids=[2, 3, 4],
            corpus_map=corpus_map,
            enable_deduplication=False
        )
        
        assert len(novel_ids) == 3  # All are "novel" when disabled
        assert stats["num_deduplicated"] == 0
    
    def test_location_hints_added(self, corpus_map):
        """Test that location hints are added for deduplicated docs."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        
        # First turn
        manager.deduplicate_context(
            conversation_id="conv_001",
            retrieved_doc_ids=[1, 2],
            corpus_map=corpus_map
        )
        
        # Second turn
        context_str, _, _ = manager.deduplicate_context(
            conversation_id="conv_001",
            retrieved_doc_ids=[1, 3],  # 1 overlaps
            corpus_map=corpus_map
        )
        
        # Should have a hint for doc 1
        assert "Doc_1" in context_str or "doc_1" in context_str.lower()


class TestDeduplicationStatistics:
    """Test deduplication statistics across turns."""
    
    @pytest.fixture
    def corpus_map(self):
        return {str(i): {"text": f"Doc {i}"} for i in range(10)}
    
    def test_cumulative_statistics(self, corpus_map):
        """Test that statistics accumulate across turns."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        
        # Turn 1: 5 docs
        manager.deduplicate_context("conv", [0, 1, 2, 3, 4], corpus_map)
        
        # Turn 2: 5 docs, 3 overlap
        manager.deduplicate_context("conv", [2, 3, 4, 5, 6], corpus_map)
        
        # Turn 3: 5 docs, 4 overlap
        manager.deduplicate_context("conv", [3, 4, 5, 6, 7], corpus_map)
        
        conv = manager.get_conversation("conv")
        stats = conv.get_stats()
        
        assert stats["total_retrieved"] == 15
        assert stats["turn_count"] == 3


class TestMultiConversationIsolation:
    """Test isolation between different conversations."""
    
    @pytest.fixture
    def corpus_map(self):
        return {str(i): {"text": f"Doc {i}"} for i in range(20)}
    
    def test_conversations_isolated(self, corpus_map):
        """Test that different conversations don't share state."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        
        # Conversation 1
        manager.deduplicate_context("conv_1", [1, 2, 3], corpus_map)
        
        # Conversation 2 with same docs
        _, novel_ids, stats = manager.deduplicate_context(
            "conv_2", [1, 2, 3], corpus_map
        )
        
        # Conv 2 should not see deduplication from conv 1
        assert len(novel_ids) == 3
        assert stats["num_deduplicated"] == 0
    
    def test_concurrent_conversations(self, corpus_map):
        """Test handling multiple concurrent conversations."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        
        # Simulate concurrent conversations
        for i in range(5):
            conv_id = f"conv_{i}"
            
            # Turn 1
            manager.deduplicate_context(conv_id, [i, i+1, i+2], corpus_map)
            
            # Turn 2
            manager.deduplicate_context(conv_id, [i+1, i+2, i+3], corpus_map)
        
        # All conversations should exist
        assert len(manager.conversations) == 5
        
        # Each conversation should have 2 turns
        for conv in manager.conversations.values():
            assert conv.turn_count == 2


class TestManagerStatistics:
    """Test aggregated statistics from manager."""
    
    @pytest.fixture
    def corpus_map(self):
        return {str(i): {"text": f"Doc {i}"} for i in range(20)}
    
    def test_get_all_stats(self, corpus_map):
        """Test getting aggregated stats across all conversations."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        
        # Create multiple conversations
        manager.deduplicate_context("conv_1", [1, 2, 3], corpus_map)
        manager.deduplicate_context("conv_1", [2, 3, 4], corpus_map)
        manager.deduplicate_context("conv_2", [5, 6, 7], corpus_map)
        
        stats = manager.get_all_stats()
        
        assert stats["total_conversations"] == 2
        assert stats["total_turns"] == 3
        assert stats["total_retrieved"] == 9
    
    def test_reset_conversation(self, corpus_map):
        """Test resetting a specific conversation."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        
        manager.deduplicate_context("conv_1", [1, 2, 3], corpus_map)
        manager.deduplicate_context("conv_2", [4, 5, 6], corpus_map)
        
        manager.reset_conversation("conv_1")
        
        assert "conv_1" not in manager.conversations
        assert "conv_2" in manager.conversations
    
    def test_reset_all(self, corpus_map):
        """Test resetting all conversations."""
        from contextpilot.pipeline.multi_turn import MultiTurnManager
        
        manager = MultiTurnManager()
        
        manager.deduplicate_context("conv_1", [1, 2, 3], corpus_map)
        manager.deduplicate_context("conv_2", [4, 5, 6], corpus_map)
        
        manager.reset_all()
        
        assert len(manager.conversations) == 0
