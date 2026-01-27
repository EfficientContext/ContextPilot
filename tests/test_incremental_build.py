"""
Test Incremental Build Feature

This tests the incremental build algorithm:
1. First batch: Full build (clustering, scheduling, index creation)
2. Second batch: Incremental build
   - Search existing index for matches
   - Reorder matched contexts to align with existing prefix
   - Build separate index for unmatched contexts
   - Merge new index under global root
"""

import pytest
import requests

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def index_server_url():
    """Fixture providing the index server URL."""
    return "http://localhost:8765"


@pytest.fixture
def reset_server(index_server_url):
    """Reset the server before each test."""
    try:
        response = requests.post(f"{index_server_url}/reset", timeout=5)
        if response.status_code != 200:
            pytest.skip("Index server not responding correctly")
    except requests.exceptions.ConnectionError:
        pytest.skip("Index server not running")
    yield


class TestIncrementalBuildBasic:
    """Test basic incremental build functionality."""
    
    def test_incremental_build_basic(self, index_server_url, reset_server):
        """Test basic incremental build functionality."""
        # Step 1: Initial build with first batch
        batch1_contexts = [
            [1, 2, 3, 4, 5],      # Context A
            [1, 2, 3, 6, 7],      # Context B (shares [1,2,3] with A)
            [1, 2, 8, 9, 10],     # Context C (shares [1,2] with A,B)
            [11, 12, 13, 14, 15], # Context D (different branch)
            [11, 12, 13, 16, 17], # Context E (shares [11,12,13] with D)
        ]
        
        build1_response = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": batch1_contexts,
                "initial_tokens_per_context": 100,
                "incremental": False
            }
        )
        
        assert build1_response.status_code == 200, f"Build failed: {build1_response.text}"
        result1 = build1_response.json()
        
        batch1_request_ids = result1.get('request_ids', [])
        assert len(batch1_request_ids) == 5, "Expected 5 request IDs"
        
        # Check index stats
        stats1 = requests.get(f"{index_server_url}/stats").json()
        
        # Step 2: Incremental build with second batch
        batch2_contexts = [
            [1, 2, 3, 4, 20],     # Matches [1,2,3,4] from Context A
            [1, 2, 3, 21, 22],    # Matches [1,2,3]
            [11, 12, 13, 23, 24], # Matches [11,12,13] from D/E
            [30, 31, 32, 33, 34], # No match
            [30, 31, 32, 35, 36], # No match, but shares with above
        ]
        
        build2_response = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": batch2_contexts,
                "initial_tokens_per_context": 100,
                "incremental": True
            }
        )
        
        assert build2_response.status_code == 200, f"Incremental build failed: {build2_response.text}"
        result2 = build2_response.json()
        
        batch2_request_ids = result2.get('request_ids', [])
        assert len(batch2_request_ids) == 5, "Expected 5 new request IDs"
        
        # Verify all request IDs are NEW (not reused from batch 1)
        for rid in batch2_request_ids:
            assert rid not in batch1_request_ids, f"Request ID {rid} was reused"
        
        # Check index stats after incremental build
        stats2 = requests.get(f"{index_server_url}/stats").json()
        
        # Should have more requests now
        num_requests_1 = stats1.get('index_stats', {}).get('num_requests', 0)
        num_requests_2 = stats2.get('index_stats', {}).get('num_requests', 0)
        assert num_requests_2 > num_requests_1, "Expected more requests after incremental build"
    
    def test_incremental_build_all_matched(self, index_server_url, reset_server):
        """Test incremental build when ALL new contexts match existing ones."""
        # Initial build
        batch1 = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 6, 7],
            [10, 20, 30, 40, 50],
        ]
        
        build1 = requests.post(
            f"{index_server_url}/build",
            json={"contexts": batch1, "incremental": False}
        ).json()
        
        assert build1.get('num_contexts') == 3
        
        # Incremental with contexts that should ALL match
        batch2 = [
            [1, 2, 3, 100, 101],    # Matches [1,2,3]
            [1, 2, 3, 4, 102],      # Matches [1,2,3,4]
            [10, 20, 30, 103, 104], # Matches [10,20,30]
        ]
        
        build2 = requests.post(
            f"{index_server_url}/build",
            json={"contexts": batch2, "incremental": True}
        ).json()
        
        assert build2.get('matched_count') == 3, "Expected all 3 to match"
        assert build2.get('merged_count') == 0, "Expected 0 merged"
    
    def test_incremental_build_none_matched(self, index_server_url, reset_server):
        """Test incremental build when NO new contexts match existing ones."""
        # Initial build
        batch1 = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 6, 7],
        ]
        
        build1 = requests.post(
            f"{index_server_url}/build",
            json={"contexts": batch1, "incremental": False}
        ).json()
        
        # Incremental with completely different contexts
        batch2 = [
            [100, 200, 300, 400, 500],  # No overlap
            [100, 200, 300, 600, 700],  # No overlap with batch1
            [800, 900, 1000],           # No overlap
        ]
        
        build2 = requests.post(
            f"{index_server_url}/build",
            json={"contexts": batch2, "incremental": True}
        ).json()
        
        assert build2.get('matched_count') == 0, "Expected 0 matched"
        assert build2.get('merged_count') == 3, "Expected all 3 merged"


class TestIncrementalBuildWithLLM:
    """Test incremental build with LLM requests (requires SGLang)."""
    
    @pytest.mark.skip(reason="Requires SGLang server running")
    def test_incremental_build_with_llm(self, index_server_url, reset_server):
        """Test incremental build with actual LLM requests."""
        # Initial build
        batch1 = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 6, 7],
            [10, 11, 12, 13, 14],
        ]
        
        build1 = requests.post(
            f"{index_server_url}/build",
            json={"contexts": batch1, "incremental": False}
        ).json()
        
        batch1_ids = build1.get('request_ids', [])
        
        # Send LLM requests for batch 1
        for i, rid in enumerate(batch1_ids[:2]):
            resp = requests.post(
                f"{index_server_url}/v1/completions",
                json={
                    "model": "Qwen/Qwen3-4B",
                    "prompt": f"Test prompt {i}: What is 2+2?",
                    "max_tokens": 20,
                    "temperature": 0.0,
                    "rid": rid
                },
                timeout=30
            )
            assert resp.status_code == 200
        
        # Incremental build
        batch2 = [
            [1, 2, 3, 100, 101],  # Matches existing
            [50, 51, 52, 53, 54], # No match
        ]
        
        build2 = requests.post(
            f"{index_server_url}/build",
            json={"contexts": batch2, "incremental": True}
        ).json()
        
        batch2_ids = build2.get('request_ids', [])
        assert len(batch2_ids) == 2
        
        # Send LLM requests for batch 2
        for i, rid in enumerate(batch2_ids):
            resp = requests.post(
                f"{index_server_url}/v1/completions",
                json={
                    "model": "Qwen/Qwen3-4B",
                    "prompt": f"Test prompt batch2-{i}: What is 3+3?",
                    "max_tokens": 20,
                    "temperature": 0.0,
                    "rid": rid
                },
                timeout=30
            )
            assert resp.status_code == 200
