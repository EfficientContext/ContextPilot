"""
Tests for mem0 integration with ContextPilot.

Tests the Mem0Retriever class and its integration with
ContextPilot's deduplication and reordering features.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json


# Skip all tests if mem0 is not installed
pytest.importorskip("mem0")


class TestMem0RetrieverInit:
    """Test Mem0Retriever initialization."""
    
    def test_init_default(self):
        """Test default initialization creates Memory instance."""
        from contextpilot.retriever import Mem0Retriever
        
        with patch('contextpilot.retriever.mem0_retriever.Memory') as MockMemory:
            MockMemory.return_value = Mock()
            retriever = Mem0Retriever()
            MockMemory.assert_called_once()
            assert retriever.memory is not None
    
    def test_init_with_existing_memory(self):
        """Test initialization with existing Memory instance."""
        from contextpilot.retriever import Mem0Retriever
        
        mock_memory = Mock()
        retriever = Mem0Retriever(memory=mock_memory)
        assert retriever.memory is mock_memory
    
    def test_init_with_config(self):
        """Test initialization with config dictionary."""
        from contextpilot.retriever import Mem0Retriever
        
        config = {"llm": {"provider": "openai"}}
        
        with patch('contextpilot.retriever.mem0_retriever.Memory') as MockMemory:
            MockMemory.from_config.return_value = Mock()
            retriever = Mem0Retriever(config=config)
            MockMemory.from_config.assert_called_once_with(config)
    
    def test_init_with_client(self):
        """Test initialization with cloud client."""
        from contextpilot.retriever import Mem0Retriever
        
        with patch('contextpilot.retriever.mem0_retriever.MemoryClient') as MockClient:
            MockClient.return_value = Mock()
            retriever = Mem0Retriever(use_client=True, api_key="test_key")
            MockClient.assert_called_once_with(api_key="test_key")
    
    def test_init_client_requires_api_key(self):
        """Test that client mode requires API key."""
        from contextpilot.retriever import Mem0Retriever
        
        with pytest.raises(ValueError, match="api_key is required"):
            Mem0Retriever(use_client=True)


class TestMem0RetrieverSearch:
    """Test Mem0Retriever search functionality."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Create a retriever with mocked memory."""
        from contextpilot.retriever import Mem0Retriever
        
        mock_memory = Mock()
        mock_memory.search.return_value = {
            "results": [
                {"id": "mem1", "memory": "User likes pizza", "score": 0.9},
                {"id": "mem2", "memory": "User is vegetarian", "score": 0.8},
            ]
        }
        
        retriever = Mem0Retriever(memory=mock_memory)
        return retriever
    
    def test_search_queries(self, mock_retriever):
        """Test searching queries returns proper format."""
        results = mock_retriever.search_queries(
            query_data=[{"qid": 0, "text": "What food do I like?"}],
            user_id="test_user",
            top_k=5
        )
        
        assert len(results) == 1
        assert results[0]["qid"] == 0
        assert results[0]["text"] == "What food do I like?"
        assert len(results[0]["top_k_doc_id"]) == 2
        assert "memories" in results[0]
    
    def test_search_requires_user_id(self, mock_retriever):
        """Test that search requires user/agent/run ID."""
        with pytest.raises(ValueError, match="At least one of"):
            mock_retriever.search_queries(
                query_data=[{"qid": 0, "text": "test"}],
                top_k=5
            )
    
    def test_search_with_agent_id(self, mock_retriever):
        """Test search with agent_id instead of user_id."""
        results = mock_retriever.search_queries(
            query_data=[{"qid": 0, "text": "test"}],
            agent_id="test_agent",
            top_k=5
        )
        
        mock_retriever.memory.search.assert_called_once()
        call_kwargs = mock_retriever.memory.search.call_args[1]
        assert call_kwargs["agent_id"] == "test_agent"
    
    def test_search_caches_memories(self, mock_retriever):
        """Test that search caches memories for corpus_map."""
        mock_retriever.search_queries(
            query_data=[{"qid": 0, "text": "test"}],
            user_id="test_user",
            top_k=5
        )
        
        corpus_map = mock_retriever.get_corpus_map()
        # With integer ID mapping (default), keys are "0", "1", etc.
        assert "0" in corpus_map
        assert "1" in corpus_map
        assert corpus_map["0"]["text"] == "User likes pizza"
        # Check that original mem0 UUID is preserved
        assert corpus_map["0"]["memory_id"] == "mem1"


class TestMem0RetrieverAddMemory:
    """Test adding memories through Mem0Retriever."""
    
    def test_add_memory(self):
        """Test adding a memory."""
        from contextpilot.retriever import Mem0Retriever
        
        mock_memory = Mock()
        mock_memory.add.return_value = [{"id": "new_mem", "memory": "test", "event": "ADD"}]
        
        retriever = Mem0Retriever(memory=mock_memory)
        result = retriever.add_memory("Test memory", user_id="test_user")
        
        mock_memory.add.assert_called_once()
        assert result is not None


class TestMem0RetrieverIndexCorpus:
    """Test indexing a corpus into mem0."""
    
    def test_index_corpus_data(self):
        """Test indexing corpus data."""
        from contextpilot.retriever import Mem0Retriever
        
        mock_memory = Mock()
        mock_memory.add.return_value = [{"id": "mem1", "memory": "test", "event": "ADD"}]
        
        retriever = Mem0Retriever(memory=mock_memory)
        
        corpus = [
            {"chunk_id": 0, "text": "First doc", "title": "Doc 1"},
            {"chunk_id": 1, "text": "Second doc", "title": "Doc 2"},
        ]
        
        retriever.index_corpus(corpus_data=corpus, user_id="test_user")
        
        assert mock_memory.add.call_count == 2
        assert retriever._corpus_loaded


class TestMem0RetrieverLoadMemories:
    """Test loading existing memories as corpus."""
    
    def test_load_corpus_from_memories(self):
        """Test loading memories as corpus."""
        from contextpilot.retriever import Mem0Retriever
        
        mock_memory = Mock()
        mock_memory.get_all.return_value = {
            "results": [
                {"id": "mem1", "memory": "Memory 1", "created_at": "2024-01-01"},
                {"id": "mem2", "memory": "Memory 2", "created_at": "2024-01-02"},
            ]
        }
        
        retriever = Mem0Retriever(memory=mock_memory)
        corpus = retriever.load_corpus_from_memories(user_id="test_user")
        
        assert len(corpus) == 2
        # With integer ID mapping, chunk_id should be integer
        assert corpus[0]["chunk_id"] == 0
        assert corpus[0]["mem0_id"] == "mem1"
        assert corpus[0]["text"] == "Memory 1"
        assert retriever._corpus_loaded


class TestMem0IDMapping:
    """Test ID mapping functionality for ContextPilot compatibility."""
    
    def test_integer_id_mapping_default(self):
        """Test that integer ID mapping is enabled by default."""
        from contextpilot.retriever import Mem0Retriever
        
        mock_memory = Mock()
        mock_memory.search.return_value = {
            "results": [
                {"id": "uuid-abc-123", "memory": "Test 1", "score": 0.9},
                {"id": "uuid-def-456", "memory": "Test 2", "score": 0.8},
            ]
        }
        
        retriever = Mem0Retriever(memory=mock_memory)
        assert retriever.use_integer_ids is True
        
        results = retriever.search_queries(
            query_data=[{"qid": 0, "text": "test"}],
            user_id="test_user"
        )
        
        doc_ids = results[0]["top_k_doc_id"]
        assert doc_ids == [0, 1]
        assert all(isinstance(x, int) for x in doc_ids)
    
    def test_id_mapping_methods(self):
        """Test ID mapping helper methods."""
        from contextpilot.retriever import Mem0Retriever
        
        mock_memory = Mock()
        mock_memory.search.return_value = {
            "results": [
                {"id": "uuid-abc", "memory": "Test", "score": 0.9},
            ]
        }
        
        retriever = Mem0Retriever(memory=mock_memory)
        retriever.search_queries(
            query_data=[{"qid": 0, "text": "test"}],
            user_id="test_user"
        )
        
        # Test get_id_mapping
        mapping = retriever.get_id_mapping()
        assert mapping[0] == "uuid-abc"
        
        # Test get_reverse_mapping
        reverse = retriever.get_reverse_mapping()
        assert reverse["uuid-abc"] == 0
        
        # Test get_mem0_uuid
        assert retriever.get_mem0_uuid(0) == "uuid-abc"
        
        # Test get_integer_id
        assert retriever.get_integer_id("uuid-abc") == 0
    
    def test_disable_integer_ids(self):
        """Test using original UUID strings when disabled."""
        from contextpilot.retriever import Mem0Retriever
        
        mock_memory = Mock()
        mock_memory.search.return_value = {
            "results": [
                {"id": "uuid-abc", "memory": "Test", "score": 0.9},
            ]
        }
        
        retriever = Mem0Retriever(memory=mock_memory, use_integer_ids=False)
        
        results = retriever.search_queries(
            query_data=[{"qid": 0, "text": "test"}],
            user_id="test_user"
        )
        
        doc_ids = results[0]["top_k_doc_id"]
        assert doc_ids == ["uuid-abc"]
    
    def test_consistent_id_across_searches(self):
        """Test that same memory gets same integer ID across searches."""
        from contextpilot.retriever import Mem0Retriever
        
        mock_memory = Mock()
        retriever = Mem0Retriever(memory=mock_memory)
        
        # First search
        mock_memory.search.return_value = {
            "results": [
                {"id": "uuid-abc", "memory": "Test 1", "score": 0.9},
                {"id": "uuid-def", "memory": "Test 2", "score": 0.8},
            ]
        }
        results1 = retriever.search_queries(
            query_data=[{"qid": 0, "text": "test"}],
            user_id="test_user"
        )
        
        # Second search with overlapping memory
        mock_memory.search.return_value = {
            "results": [
                {"id": "uuid-abc", "memory": "Test 1", "score": 0.95},  # Same UUID
                {"id": "uuid-ghi", "memory": "Test 3", "score": 0.85},  # New UUID
            ]
        }
        results2 = retriever.search_queries(
            query_data=[{"qid": 1, "text": "test2"}],
            user_id="test_user"
        )
        
        # uuid-abc should have same ID (0) in both searches
        assert 0 in results1[0]["top_k_doc_id"]
        assert 0 in results2[0]["top_k_doc_id"]
        
        # New memory uuid-ghi should get ID 2
        assert 2 in results2[0]["top_k_doc_id"]


class TestMem0WithMultiTurn:
    """Test mem0 integration with ContextPilot's multi-turn deduplication."""
    
    def test_deduplication_with_memories(self):
        """Test that mem0 memories work with multi-turn deduplication."""
        from contextpilot.retriever import Mem0Retriever
        from contextpilot.pipeline import MultiTurnManager
        
        # Setup mock retriever
        mock_memory = Mock()
        mock_memory.search.return_value = {
            "results": [
                {"id": "mem1", "memory": "User likes pizza", "score": 0.9},
                {"id": "mem2", "memory": "User is vegetarian", "score": 0.8},
                {"id": "mem3", "memory": "User lives in SF", "score": 0.7},
            ]
        }
        
        retriever = Mem0Retriever(memory=mock_memory)
        multi_turn = MultiTurnManager()
        
        # First turn - search and get memories
        results1 = retriever.search_queries(
            query_data=[{"qid": 0, "text": "What food do I like?"}],
            user_id="test_user",
            top_k=5
        )
        
        corpus_map = retriever.get_corpus_map()
        
        # Deduplicate first turn (no deduplication expected)
        context1, novel1, stats1 = multi_turn.deduplicate_context(
            conversation_id="conv_1",
            retrieved_doc_ids=results1[0]["top_k_doc_id"],
            corpus_map=corpus_map,
            enable_deduplication=True
        )
        
        assert stats1["num_novel"] == 3
        assert stats1["num_deduplicated"] == 0
        
        # Second turn with overlapping memories
        mock_memory.search.return_value = {
            "results": [
                {"id": "mem1", "memory": "User likes pizza", "score": 0.95},  # Overlap
                {"id": "mem4", "memory": "User works at tech company", "score": 0.85},
            ]
        }
        
        results2 = retriever.search_queries(
            query_data=[{"qid": 1, "text": "Where do I work?"}],
            user_id="test_user",
            top_k=5
        )
        
        corpus_map = retriever.get_corpus_map()
        
        # Deduplicate second turn (should deduplicate mem1)
        context2, novel2, stats2 = multi_turn.deduplicate_context(
            conversation_id="conv_1",
            retrieved_doc_ids=results2[0]["top_k_doc_id"],
            corpus_map=corpus_map,
            enable_deduplication=True
        )
        
        assert stats2["num_novel"] == 1  # Only mem4 is novel
        assert stats2["num_deduplicated"] == 1  # mem1 is deduplicated


class TestMem0CorpusMapHelper:
    """Test the create_mem0_corpus_map helper function."""
    
    def test_create_corpus_map(self):
        """Test creating a corpus map from mem0 memories."""
        from contextpilot.retriever import create_mem0_corpus_map
        
        mock_memory = Mock()
        mock_memory.get_all.return_value = {
            "results": [
                {"id": "mem1", "memory": "Test memory", "score": 0.9},
            ]
        }
        
        # Default: use_integer_ids=True
        corpus_map = create_mem0_corpus_map(mock_memory, user_id="test_user")
        
        assert "0" in corpus_map  # Integer ID as string key
        assert corpus_map["0"]["memory_id"] == "mem1"  # Original UUID preserved
        assert corpus_map["0"]["text"] == "Test memory"
        assert corpus_map["0"]["content"] == "Test memory"
    
    def test_create_corpus_map_with_uuid(self):
        """Test creating a corpus map with original UUID keys."""
        from contextpilot.retriever import create_mem0_corpus_map
        
        mock_memory = Mock()
        mock_memory.get_all.return_value = {
            "results": [
                {"id": "mem1", "memory": "Test memory", "score": 0.9},
            ]
        }
        
        corpus_map = create_mem0_corpus_map(
            mock_memory, user_id="test_user", use_integer_ids=False
        )
        
        assert "mem1" in corpus_map
        assert corpus_map["mem1"]["text"] == "Test memory"


class TestRetrieverConfigMem0:
    """Test RetrieverConfig for mem0 type."""
    
    def test_mem0_config_validation(self):
        """Test that mem0 config requires user/agent/run ID."""
        from contextpilot.pipeline import RetrieverConfig
        
        with pytest.raises(ValueError, match="At least one of"):
            RetrieverConfig(retriever_type="mem0")
    
    def test_mem0_config_with_user_id(self):
        """Test mem0 config accepts user_id."""
        from contextpilot.pipeline import RetrieverConfig
        
        config = RetrieverConfig(
            retriever_type="mem0",
            mem0_user_id="test_user"
        )
        assert config.mem0_user_id == "test_user"
    
    def test_mem0_client_requires_api_key(self):
        """Test mem0 client mode requires API key."""
        from contextpilot.pipeline import RetrieverConfig
        
        with pytest.raises(ValueError, match="mem0_api_key is required"):
            RetrieverConfig(
                retriever_type="mem0",
                mem0_user_id="test_user",
                mem0_use_client=True
            )
