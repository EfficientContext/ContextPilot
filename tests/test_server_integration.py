"""
ContextPilot End-to-End Pipeline Example Tests

These tests demonstrate and validate the complete RAG pipeline:
1. Retrieve documents using BM25
2. Optimize context ordering with ContextPilot clustering
3. Build index on ContextPilot Index Server
4. Generate responses (proxied through Index Server to SGLang)
5. Automatic token tracking and eviction

These are integration tests requiring external servers to be running.
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


class TestManualWorkflow:
    """Test manual workflow with explicit index operations."""
    
    def test_build_index(self, index_server_url, reset_server):
        """Test building index manually."""
        build_response = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": [
                    [1, 2, 3, 4, 5],      # Context for query 1
                    [1, 2, 3, 6, 7],      # Context for query 2 (shares prefix)
                    [8, 9, 10, 11, 12],   # Context for query 3
                ],
                "initial_tokens_per_context": 100,
                "alpha": 0.005
            }
        )
        
        assert build_response.status_code == 200, f"Build failed: {build_response.text}"
        result = build_response.json()
        request_id_mapping = result.get("request_id_mapping", {})
        assert len(request_id_mapping) == 3, "Expected 3 contexts in mapping"
    
    def test_get_stats(self, index_server_url, reset_server):
        """Test getting index stats."""
        # First build some data
        requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": [[1, 2, 3], [4, 5, 6]],
                "initial_tokens_per_context": 100,
            }
        )
        
        stats_response = requests.get(f"{index_server_url}/stats")
        
        assert stats_response.status_code == 200
        stats = stats_response.json()
        assert "eviction_stats" in stats or "index_stats" in stats


class TestLLMIntegration:
    """Test LLM integration (requires SGLang server)."""
    
    @pytest.mark.skip(reason="Requires SGLang server running")
    def test_completion_with_request_id(self, index_server_url, reset_server):
        """Test sending completion request with request_id tracking."""
        # Build index first
        build_response = requests.post(
            f"{index_server_url}/build",
            json={
                "contexts": [[1, 2, 3, 4, 5]],
                "initial_tokens_per_context": 100,
            }
        )
        
        result = build_response.json()
        request_ids = result.get("request_ids", [])
        
        if not request_ids:
            pytest.skip("No request IDs returned")
        
        first_request_id = request_ids[0]
        
        # Send completion request
        completion_response = requests.post(
            f"{index_server_url}/v1/completions",
            json={
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "prompt": "What is machine learning?",
                "max_tokens": 100,
                "temperature": 0.0,
                "request_id": first_request_id
            },
            timeout=30
        )
        
        assert completion_response.status_code == 200
        result = completion_response.json()
        assert "choices" in result


class TestPipelineAPI:
    """Test RAGPipeline API (unit tests, no server needed)."""
    
    def test_pipeline_import(self):
        """Test that pipeline can be imported."""
        from contextpilot.pipeline import RAGPipeline, InferenceConfig
        assert RAGPipeline is not None
        assert InferenceConfig is not None
    
    def test_inference_config_creation(self):
        """Test InferenceConfig creation."""
        from contextpilot.pipeline import InferenceConfig
        
        config = InferenceConfig(
            model_name="test-model",
            backend="sglang",
            base_url="http://localhost:8765",
            max_tokens=256,
            temperature=0.0
        )
        
        assert config.model_name == "test-model"
        assert config.backend == "sglang"
        assert config.max_tokens == 256
