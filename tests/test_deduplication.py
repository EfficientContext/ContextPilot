"""
Test Multi-Turn Conversation Context Deduplication

Tests the deduplication feature where subsequent turns in a conversation
have overlapping documents removed and replaced with reference hints.

Example scenario:
- req-a (turn 1): [4, 3, 1]
- req-b (turn 2, continues req-a): [4, 3, 2]
- Expected: req-b's context deduplicated to [2] with hints for [4, 3]
"""

import pytest
import requests

# Mark all tests in this module as integration tests requiring server
pytestmark = pytest.mark.integration


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


class TestBasicDeduplication:
    """Test basic multi-turn deduplication scenarios."""
    
    def test_basic_two_turn_deduplication(self, index_server_url, reset_server):
        """Test basic two-turn deduplication."""
        # Turn 1: Initial request with documents [4, 3, 1]
        turn1_contexts = [[4, 3, 1]]
        
        turn1_response = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": turn1_contexts,
                "incremental": False,
                "deduplicate": True,
                "parent_request_ids": [None],
            }
        )
        
        assert turn1_response.status_code == 200, f"Turn 1 failed: {turn1_response.text}"
        turn1_result = turn1_response.json()
        req_a_id = turn1_result["request_ids"][0]
        
        # Turn 1 should have no deduplication (it's the first turn)
        if "deduplication" in turn1_result:
            dedup = turn1_result["deduplication"]["results"][0]
            assert dedup["overlapping_docs"] == [], "Turn 1 should have no overlapping docs"
            assert dedup["deduplicated_docs"] == turn1_contexts[0], "Turn 1 should keep all docs"
        
        # Turn 2: Continuation with documents [4, 3, 2]
        turn2_contexts = [[4, 3, 2]]
        
        turn2_response = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": turn2_contexts,
                "incremental": True,
                "deduplicate": True,
                "parent_request_ids": [req_a_id],
            }
        )
        
        assert turn2_response.status_code == 200, f"Turn 2 failed: {turn2_response.text}"
        turn2_result = turn2_response.json()
        
        # Check deduplication results
        assert "deduplication" in turn2_result, "Turn 2 should have deduplication results"
        dedup = turn2_result["deduplication"]["results"][0]
        
        # Verify deduplication
        assert set(dedup['overlapping_docs']) == {4, 3}, f"Expected [4, 3] to overlap, got {dedup['overlapping_docs']}"
        assert dedup['new_docs'] == [2], f"Expected new docs [2], got {dedup['new_docs']}"
        assert dedup['deduplicated_docs'] == [2], f"Expected deduplicated [2], got {dedup['deduplicated_docs']}"
        assert len(dedup['reference_hints']) == 2, f"Expected 2 hints, got {len(dedup['reference_hints'])}"
    
    def test_three_turn_conversation(self, index_server_url, reset_server):
        """Test deduplication across 3 turns."""
        # Turn 1: [10, 20, 30]
        t1 = requests.post(
            f"{index_server_url}/build",
            json={"contexts": [[10, 20, 30]], "incremental": False, "deduplicate": True}
        ).json()
        req_1 = t1["request_ids"][0]
        
        # Turn 2: [10, 20, 40] - [10, 20] overlap with turn 1
        t2 = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": [[10, 20, 40]], 
                "incremental": True, 
                "deduplicate": True,
                "parent_request_ids": [req_1]
            }
        ).json()
        req_2 = t2["request_ids"][0]
        dedup_2 = t2["deduplication"]["results"][0]
        
        assert set(dedup_2['overlapping_docs']) == {10, 20}
        assert dedup_2['new_docs'] == [40]
        
        # Turn 3: [10, 30, 50] - [10, 30] overlap from the chain
        t3 = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": [[10, 30, 50]], 
                "incremental": True, 
                "deduplicate": True,
                "parent_request_ids": [req_2]
            }
        ).json()
        dedup_3 = t3["deduplication"]["results"][0]
        
        assert set(dedup_3['overlapping_docs']) == {10, 30}, f"Expected [10, 30] overlap, got {dedup_3['overlapping_docs']}"
        assert dedup_3['new_docs'] == [50], f"Expected [50] new, got {dedup_3['new_docs']}"


class TestParallelConversations:
    """Test deduplication with multiple parallel conversations."""
    
    def test_multiple_parallel_conversations(self, index_server_url, reset_server):
        """Test deduplication with multiple parallel conversations."""
        # Conversation A - Turn 1
        ca_t1 = requests.post(
            f"{index_server_url}/build",
            json={"contexts": [[100, 101, 102]], "incremental": False, "deduplicate": True}
        ).json()
        ca_req_1 = ca_t1["request_ids"][0]
        
        # Conversation B - Turn 1 (different conversation)
        cb_t1 = requests.post(
            f"{index_server_url}/build",
            json={"contexts": [[200, 201, 202]], "incremental": True, "deduplicate": True}
        ).json()
        cb_req_1 = cb_t1["request_ids"][0]
        
        # Conversation A - Turn 2
        ca_t2 = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": [[100, 101, 103]], 
                "incremental": True, 
                "deduplicate": True,
                "parent_request_ids": [ca_req_1]
            }
        ).json()
        ca_dedup_2 = ca_t2["deduplication"]["results"][0]
        
        assert set(ca_dedup_2['overlapping_docs']) == {100, 101}
        assert ca_dedup_2['new_docs'] == [103]
        
        # Conversation B - Turn 2
        cb_t2 = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": [[200, 201, 203]], 
                "incremental": True, 
                "deduplicate": True,
                "parent_request_ids": [cb_req_1]
            }
        ).json()
        cb_dedup_2 = cb_t2["deduplication"]["results"][0]
        
        assert set(cb_dedup_2['overlapping_docs']) == {200, 201}
        assert cb_dedup_2['new_docs'] == [203]


class TestEdgeCases:
    """Test edge cases in deduplication."""
    
    def test_no_overlap(self, index_server_url, reset_server):
        """Test when there's no overlap between turns."""
        # Turn 1: [1, 2, 3]
        t1 = requests.post(
            f"{index_server_url}/build",
            json={"contexts": [[1, 2, 3]], "incremental": False, "deduplicate": True}
        ).json()
        req_1 = t1["request_ids"][0]
        
        # Turn 2: [4, 5, 6] - completely different docs
        t2 = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": [[4, 5, 6]], 
                "incremental": True, 
                "deduplicate": True,
                "parent_request_ids": [req_1]
            }
        ).json()
        dedup_2 = t2["deduplication"]["results"][0]
        
        assert dedup_2['overlapping_docs'] == [], "Expected no overlap"
        assert dedup_2['new_docs'] == [4, 5, 6], "All docs should be new"
        assert dedup_2['reference_hints'] == [], "No hints when no overlap"
    
    def test_complete_overlap(self, index_server_url, reset_server):
        """Test when turn 2 is completely overlapping with turn 1."""
        # Turn 1: [1, 2, 3]
        t1 = requests.post(
            f"{index_server_url}/build",
            json={"contexts": [[1, 2, 3]], "incremental": False, "deduplicate": True}
        ).json()
        req_1 = t1["request_ids"][0]
        
        # Turn 2: [1, 2, 3] - exact same docs
        t2 = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": [[1, 2, 3]], 
                "incremental": True, 
                "deduplicate": True,
                "parent_request_ids": [req_1]
            }
        ).json()
        dedup_2 = t2["deduplication"]["results"][0]
        
        assert set(dedup_2['overlapping_docs']) == {1, 2, 3}, "All docs should overlap"
        assert dedup_2['new_docs'] == [], "No new docs"
        assert dedup_2['deduplicated_docs'] == [], "Deduplicated should be empty"
        assert len(dedup_2['reference_hints']) == 3, "Should have 3 hints"


class TestBatchDeduplication:
    """Test batch deduplication scenarios."""
    
    def test_batch_deduplication(self, index_server_url, reset_server):
        """Test deduplication with multiple contexts in one request."""
        # Turn 1: Two contexts
        t1 = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": [[10, 20, 30], [40, 50, 60]], 
                "incremental": False, 
                "deduplicate": True,
                "parent_request_ids": [None, None]
            }
        ).json()
        req_0 = t1["request_ids"][0]
        req_1 = t1["request_ids"][1]
        
        # Turn 2: Two contexts continuing their respective parents
        t2 = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": [[10, 20, 35], [40, 50, 65]], 
                "incremental": True, 
                "deduplicate": True,
                "parent_request_ids": [req_0, req_1]
            }
        ).json()
        
        dedup_0 = t2["deduplication"]["results"][0]
        dedup_1 = t2["deduplication"]["results"][1]
        
        assert set(dedup_0['overlapping_docs']) == {10, 20}
        assert dedup_0['new_docs'] == [35]
        
        assert set(dedup_1['overlapping_docs']) == {40, 50}
        assert dedup_1['new_docs'] == [65]


class TestDeduplicateEndpoint:
    """Test the standalone /deduplicate endpoint."""
    
    def test_deduplicate_endpoint(self, index_server_url, reset_server):
        """Test the standalone /deduplicate endpoint."""
        # Turn 1: Use /build to create initial index
        turn1_response = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": [[4, 3, 1]],
                "incremental": False,
                "deduplicate": True,
                "parent_request_ids": [None],
            }
        )
        
        assert turn1_response.status_code == 200
        turn1_result = turn1_response.json()
        req_a_id = turn1_result["request_ids"][0]
        
        # Turn 2: Use /deduplicate (no index operations!)
        turn2_response = requests.post(
            f"{index_server_url}/deduplicate",
            json={
                "contexts": [[4, 3, 2]],
                "parent_request_ids": [req_a_id],
            }
        )
        
        assert turn2_response.status_code == 200
        turn2_result = turn2_response.json()
        
        result = turn2_result["results"][0]
        
        # Verify deduplication
        assert set(result['overlapping_docs']) == {4, 3}
        assert result['new_docs'] == [2]
        assert result['is_new_conversation'] == False
        assert result['parent_request_id'] == req_a_id
        
        # Check summary
        summary = turn2_result["summary"]
        assert summary['continued_conversations'] == 1
        assert summary['total_docs_deduplicated'] == 2
