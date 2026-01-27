#!/usr/bin/env python3
"""
End-to-End Test for Multi-Turn Conversation Deduplication

This test validates the complete multi-turn deduplication workflow:
1. ConversationTracker unit tests
2. HTTP Server /deduplicate endpoint tests
3. Multi-turn conversation chain tests
4. Reset and eviction tests
"""

import pytest
import requests


class TestConversationTrackerUnit:
    """Test ConversationTracker unit functionality (no server needed)."""
    
    def test_basic_deduplication(self):
        """Test basic Turn 1 - new conversation detection."""
        from contextpilot.server.conversation_tracker import ConversationTracker
        
        tracker = ConversationTracker()
        result1 = tracker.deduplicate('req_a', [4, 3, 1], parent_request_id=None)
        
        assert result1.is_new_conversation == True
        assert result1.new_docs == [4, 3, 1]
        assert result1.overlapping_docs == []
    
    def test_overlap_detection(self):
        """Test Turn 2 - overlap detection."""
        from contextpilot.server.conversation_tracker import ConversationTracker
        
        tracker = ConversationTracker()
        tracker.deduplicate('req_a', [4, 3, 1], parent_request_id=None)
        result2 = tracker.deduplicate('req_b', [4, 3, 2], parent_request_id='req_a')
        
        assert result2.is_new_conversation == False
        assert result2.new_docs == [2]
        assert set(result2.overlapping_docs) == {4, 3}
    
    def test_reference_hints_generation(self):
        """Test Turn 2 - reference hints generated."""
        from contextpilot.server.conversation_tracker import ConversationTracker
        
        tracker = ConversationTracker()
        tracker.deduplicate('req_a', [4, 3, 1], parent_request_id=None)
        result2 = tracker.deduplicate('req_b', [4, 3, 2], parent_request_id='req_a')
        
        assert len(result2.reference_hints) == 2
    
    def test_chain_propagation(self):
        """Test Turn 3 - chain propagation."""
        from contextpilot.server.conversation_tracker import ConversationTracker
        
        tracker = ConversationTracker()
        tracker.deduplicate('req_a', [4, 3, 1], parent_request_id=None)
        tracker.deduplicate('req_b', [4, 3, 2], parent_request_id='req_a')
        result3 = tracker.deduplicate('req_c', [4, 2, 5], parent_request_id='req_b')
        
        assert result3.new_docs == [5]
        assert set(result3.overlapping_docs) == {4, 2}
    
    def test_get_conversation_chain(self):
        """Test getting conversation chain."""
        from contextpilot.server.conversation_tracker import ConversationTracker
        
        tracker = ConversationTracker()
        tracker.deduplicate('req_a', [4, 3, 1], parent_request_id=None)
        tracker.deduplicate('req_b', [4, 3, 2], parent_request_id='req_a')
        tracker.deduplicate('req_c', [4, 2, 5], parent_request_id='req_b')
        
        chain = tracker.get_conversation_chain('req_c')
        assert len(chain) == 3
    
    def test_reset_functionality(self):
        """Test reset clears all requests."""
        from contextpilot.server.conversation_tracker import ConversationTracker
        
        tracker = ConversationTracker()
        tracker.deduplicate('req_a', [4, 3, 1], parent_request_id=None)
        tracker.reset()
        
        assert len(tracker._requests) == 0
    
    def test_orphaned_request(self):
        """Test orphaned request treated as new conversation."""
        from contextpilot.server.conversation_tracker import ConversationTracker
        
        tracker = ConversationTracker()
        result = tracker.deduplicate('orphan', [1, 2, 3], parent_request_id='nonexistent')
        
        assert result.is_new_conversation == True
        assert result.new_docs == [1, 2, 3]
    
    def test_clear_single_conversation(self):
        """Test clearing single request from conversation."""
        from contextpilot.server.conversation_tracker import ConversationTracker
        
        tracker = ConversationTracker()
        tracker.deduplicate('conv1_t1', [1, 2], parent_request_id=None)
        tracker.deduplicate('conv1_t2', [2, 3], parent_request_id='conv1_t1')
        tracker.deduplicate('conv2_t1', [4, 5], parent_request_id=None)
        
        cleared = tracker.clear_conversation('conv1_t1')
        
        assert cleared == 1
        assert len(tracker._requests) == 2


# Integration tests requiring HTTP server
pytestmark_integration = pytest.mark.integration


@pytest.fixture(scope="module")
def index_server_url():
    """Fixture providing the index server URL."""
    return "http://localhost:8765"


@pytest.fixture
def reset_server(index_server_url):
    """Reset the server before each test."""
    try:
        requests.post(f"{index_server_url}/reset", timeout=5)
    except requests.exceptions.ConnectionError:
        pytest.skip("Index server not running")
    yield


@pytest.mark.integration
class TestHTTPDeduplicateEndpoint:
    """Test HTTP /deduplicate endpoint."""
    
    def test_first_turn_new_conversation(self, index_server_url, reset_server):
        """Test first turn creates new conversation."""
        response = requests.post(
            f"{index_server_url}/deduplicate",
            json={
                "request_id": "http_req_1",
                "doc_ids": [10, 20, 30],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("is_new_conversation") == True
        assert data.get("new_docs") == [10, 20, 30]
    
    def test_second_turn_deduplication(self, index_server_url, reset_server):
        """Test second turn performs deduplication."""
        # Turn 1
        requests.post(
            f"{index_server_url}/deduplicate",
            json={
                "request_id": "http_req_1",
                "doc_ids": [10, 20, 30],
            }
        )
        
        # Turn 2
        response = requests.post(
            f"{index_server_url}/deduplicate",
            json={
                "request_id": "http_req_2",
                "doc_ids": [20, 30, 40],
                "parent_request_id": "http_req_1"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("is_new_conversation") == False
        assert data.get("new_docs") == [40]
        assert set(data.get("overlapping_docs", [])) == {20, 30}
        assert len(data.get("reference_hints", [])) == 2


@pytest.mark.integration
class TestHTTPMultiTurnChain:
    """Test HTTP multi-turn conversation chain."""
    
    def test_five_turn_conversation(self, index_server_url, reset_server):
        """Test 5-turn conversation chain."""
        conversation = [
            ("turn_1", [1, 2, 3, 4, 5], None),
            ("turn_2", [3, 4, 5, 6, 7], "turn_1"),       # 3,4,5 overlap
            ("turn_3", [5, 6, 7, 8, 9], "turn_2"),       # 5,6,7 overlap
            ("turn_4", [1, 8, 9, 10, 11], "turn_3"),     # 1,8,9 overlap
            ("turn_5", [10, 11, 12, 13, 14], "turn_4"),  # 10,11 overlap
        ]
        
        expected_new = [
            [1, 2, 3, 4, 5],  # Turn 1: all new
            [6, 7],           # Turn 2: 3,4,5 overlap
            [8, 9],           # Turn 3: 5,6,7 overlap
            [10, 11],         # Turn 4: 1,8,9 overlap
            [12, 13, 14],     # Turn 5: 10,11 overlap
        ]
        
        for i, (request_id, doc_ids, parent_id) in enumerate(conversation):
            payload = {
                "request_id": request_id,
                "doc_ids": doc_ids,
            }
            if parent_id:
                payload["parent_request_id"] = parent_id
            
            response = requests.post(f"{index_server_url}/deduplicate", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data.get("new_docs") == expected_new[i], f"Turn {i+1} mismatch"


@pytest.mark.integration
class TestHTTPResetEndpoint:
    """Test HTTP /reset endpoint."""
    
    def test_reset_clears_conversations(self, index_server_url, reset_server):
        """Test reset endpoint clears conversation data."""
        # Add some data
        requests.post(f"{index_server_url}/deduplicate", json={
            "request_id": "reset_test_1",
            "doc_ids": [1, 2, 3],
        })
        
        # Reset
        response = requests.post(f"{index_server_url}/reset", json={
            "conversation_only": True
        })
        
        assert response.status_code == 200
        
        # Verify reset - should be new conversation
        response2 = requests.post(f"{index_server_url}/deduplicate", json={
            "request_id": "after_reset",
            "doc_ids": [1, 2, 3],
            "parent_request_id": "reset_test_1"  # This should not exist anymore
        })
        
        data = response2.json()
        # After reset, parent doesn't exist, so treated as new conversation
        assert data.get("is_new_conversation") == True
