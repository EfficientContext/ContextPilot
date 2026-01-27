"""
Basic tests for the ContextPilot pipeline abstraction.

These tests verify that the pipeline API works correctly without
requiring external services (Elasticsearch, embedding servers, etc.).
"""

import pytest
from contextpilot.pipeline import (
    RAGPipeline,
    RetrieverConfig,
    OptimizerConfig,
    InferenceConfig,
    PipelineConfig
)


class TestConfigClasses:
    """Test configuration classes."""
    
    def test_retriever_config_bm25(self):
        """Test BM25 retriever configuration."""
        config = RetrieverConfig(
            retriever_type="bm25",
            top_k=20,
            corpus_path="test.jsonl"
        )
        assert config.retriever_type == "bm25"
        assert config.top_k == 20
        assert config.corpus_path == "test.jsonl"
    
    def test_retriever_config_faiss(self):
        """Test FAISS retriever configuration."""
        config = RetrieverConfig(
            retriever_type="faiss",
            top_k=10,
            index_path="test.faiss",
            embedding_model="test-model"
        )
        assert config.retriever_type == "faiss"
        assert config.top_k == 10
        assert config.index_path == "test.faiss"
    
    def test_retriever_config_invalid_type(self):
        """Test that invalid retriever type raises error."""
        with pytest.raises(ValueError):
            RetrieverConfig(retriever_type="invalid")
    
    def test_optimizer_config(self):
        """Test optimizer configuration."""
        config = OptimizerConfig(
            enabled=True,
            use_gpu=True,
            linkage_method="average",
            alpha=0.005
        )
        assert config.enabled is True
        assert config.use_gpu is True
        assert config.linkage_method == "average"
        assert config.alpha == 0.005
    
    def test_optimizer_config_invalid_linkage(self):
        """Test that invalid linkage method raises error."""
        with pytest.raises(ValueError):
            OptimizerConfig(linkage_method="invalid")
    
    def test_inference_config(self):
        """Test inference configuration."""
        config = InferenceConfig(
            model_name="test-model",
            backend="sglang",
            temperature=0.5,
            max_tokens=256
        )
        assert config.model_name == "test-model"
        assert config.backend == "sglang"
        assert config.temperature == 0.5
        assert config.max_tokens == 256
    
    def test_inference_config_invalid_backend(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError):
            InferenceConfig(model_name="test", backend="invalid")
    
    def test_pipeline_config_from_dict(self):
        """Test creating pipeline config from dictionary."""
        config_dict = {
            "retriever": {"retriever_type": "bm25", "top_k": 20},
            "optimizer": {"enabled": True, "use_gpu": True},
            "inference": {"model_name": "test-model"},
            "verbose": True
        }
        config = PipelineConfig.from_dict(config_dict)
        assert config.retriever.retriever_type == "bm25"
        assert config.optimizer.enabled is True
        assert config.inference.model_name == "test-model"


class TestPipelineInitialization:
    """Test pipeline initialization with different configurations."""
    
    def test_simple_string_init(self):
        """Test initialization with simple string arguments."""
        pipeline = RAGPipeline(
            retriever="bm25",
            corpus_data=[{"_id": 1, "text": "test"}],
            use_contextpilot=True
        )
        assert pipeline.retriever_config.retriever_type == "bm25"
        assert pipeline.optimizer_config.enabled is True
    
    def test_config_object_init(self):
        """Test initialization with configuration objects."""
        pipeline = RAGPipeline(
            retriever=RetrieverConfig(
                retriever_type="bm25",
                corpus_data=[{"_id": 1, "text": "test"}]
            ),
            optimizer=OptimizerConfig(enabled=True),
            inference=InferenceConfig(model_name="test-model")
        )
        assert pipeline.retriever_config.retriever_type == "bm25"
        assert pipeline.optimizer_config.enabled is True
        assert pipeline.inference_config.model_name == "test-model"
    
    def test_disable_contextpilot(self):
        """Test disabling ContextPilot optimization."""
        pipeline = RAGPipeline(
            retriever="bm25",
            corpus_data=[{"_id": 1, "text": "test"}],
            use_contextpilot=False
        )
        assert pipeline.optimizer_config.enabled is False
    
    def test_custom_retriever(self):
        """Test initialization with custom retriever."""
        class CustomRetriever:
            def retrieve(self, queries, top_k=20):
                return []
        
        pipeline = RAGPipeline(
            retriever=CustomRetriever(),
            corpus_data=[{"_id": 1, "text": "test"}]
        )
        assert pipeline.retriever_config.retriever_type == "custom"
        assert pipeline.retriever_config.custom_retriever is not None


class TestQueryNormalization:
    """Test query format normalization."""
    
    def test_single_string_query(self):
        """Test that single string query is normalized correctly."""
        pipeline = RAGPipeline(
            retriever="bm25",
            corpus_data=[{"_id": 1, "text": "test"}]
        )
        
        # Mock retriever to test normalization
        class MockRetriever:
            def search_queries(self, query_data, top_k):
                return [
                    {"qid": q["qid"], "text": q["text"], "top_k_doc_id": []}
                    for q in query_data
                ]
        
        pipeline.retriever = MockRetriever()
        pipeline.retriever_config.retriever_type = "bm25"
        
        results = pipeline.retrieve("What is AI?")
        assert len(results) == 1
        assert results[0]["text"] == "What is AI?"
    
    def test_list_of_strings_query(self):
        """Test that list of strings is normalized correctly."""
        pipeline = RAGPipeline(
            retriever="bm25",
            corpus_data=[{"_id": 1, "text": "test"}]
        )
        
        class MockRetriever:
            def search_queries(self, query_data, top_k):
                return [
                    {"qid": q["qid"], "text": q["text"], "top_k_doc_id": []}
                    for q in query_data
                ]
        
        pipeline.retriever = MockRetriever()
        pipeline.retriever_config.retriever_type = "bm25"
        
        results = pipeline.retrieve(["What is AI?", "What is ML?"])
        assert len(results) == 2
        assert results[0]["text"] == "What is AI?"
        assert results[1]["text"] == "What is ML?"


class TestOptimization:
    """Test optimization functionality."""
    
    def test_optimization_disabled(self):
        """Test that optimization can be disabled."""
        pipeline = RAGPipeline(
            retriever="bm25",
            corpus_data=[{"_id": 1, "text": "test"}],
            use_contextpilot=False
        )
        
        retrieval_results = [
            {"qid": 1, "text": "test", "top_k_doc_id": [1, 2, 3]}
        ]
        
        optimized = pipeline.optimize(retrieval_results)
        
        assert optimized["metadata"]["optimized"] is False
        assert len(optimized["groups"]) == 1
        assert len(optimized["groups"][0]["items"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
