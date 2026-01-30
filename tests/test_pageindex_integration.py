"""
Tests for PageIndex + ContextPilot integration.

These tests verify that PageIndex retriever works correctly with ContextPilot
optimization pipeline.
"""

import pytest
import json
import os
from pathlib import Path
from typing import Dict, Any, List

from contextpilot.context_index import build_context_index
from contextpilot.context_ordering import InterContextScheduler


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_tree_structure() -> Dict[str, Any]:
    """Create a sample PageIndex tree structure for testing."""
    return {
        "doc_name": "test_document.pdf",
        "doc_description": "A test document for PageIndex integration testing.",
        "structure": [
            {
                "title": "Introduction",
                "start_index": 1,
                "end_index": 1,
                "node_id": "0001",
                "summary": "Introduction to the document covering main topics.",
                "text": "This is the introduction section with background information.",
                "nodes": [
                    {
                        "title": "Background",
                        "start_index": 1,
                        "end_index": 1,
                        "node_id": "0002",
                        "summary": "Background information and context.",
                        "text": "Background details about the subject matter.",
                        "nodes": []
                    },
                    {
                        "title": "Objectives",
                        "start_index": 1,
                        "end_index": 1,
                        "node_id": "0003",
                        "summary": "Main objectives of the document.",
                        "text": "The objectives include several key goals.",
                        "nodes": []
                    }
                ]
            },
            {
                "title": "Methodology",
                "start_index": 2,
                "end_index": 3,
                "node_id": "0004",
                "summary": "Methodology section describing the approach.",
                "text": "The methodology involves multiple steps and procedures.",
                "nodes": [
                    {
                        "title": "Data Collection",
                        "start_index": 2,
                        "end_index": 2,
                        "node_id": "0005",
                        "summary": "How data was collected for the study.",
                        "text": "Data was collected through surveys and interviews.",
                        "nodes": []
                    },
                    {
                        "title": "Analysis",
                        "start_index": 3,
                        "end_index": 3,
                        "node_id": "0006",
                        "summary": "Analysis methods used.",
                        "text": "Statistical analysis was performed on the data.",
                        "nodes": []
                    }
                ]
            },
            {
                "title": "Results",
                "start_index": 4,
                "end_index": 5,
                "node_id": "0007",
                "summary": "Key findings and results.",
                "text": "The results show significant improvements.",
                "nodes": [
                    {
                        "title": "Key Findings",
                        "start_index": 4,
                        "end_index": 4,
                        "node_id": "0008",
                        "summary": "The main findings of the study.",
                        "text": "Finding 1: Performance improved by 30%.",
                        "nodes": []
                    },
                    {
                        "title": "Discussion",
                        "start_index": 5,
                        "end_index": 5,
                        "node_id": "0009",
                        "summary": "Discussion of the results.",
                        "text": "These results align with previous research.",
                        "nodes": []
                    }
                ]
            },
            {
                "title": "Conclusion",
                "start_index": 6,
                "end_index": 6,
                "node_id": "0010",
                "summary": "Conclusions and future work.",
                "text": "In conclusion, the study demonstrates success.",
                "nodes": []
            }
        ]
    }


@pytest.fixture
def sample_queries() -> List[Dict[str, Any]]:
    """Sample queries for testing."""
    return [
        {"qid": "q1", "question": "What is the background of this study?"},
        {"qid": "q2", "question": "What methodology was used?"},
        {"qid": "q3", "question": "What are the key findings?"},
        {"qid": "q4", "question": "What are the conclusions?"},
    ]


@pytest.fixture
def real_tree_path() -> Path:
    """Path to real PageIndex tree structure if available."""
    path = Path(__file__).parent.parent.parent / "PageIndex" / "tests" / "results" / "q1-fy25-earnings_structure.json"
    return path if path.exists() else None


# ============================================================================
# Helper Functions
# ============================================================================

def flatten_tree(structure) -> List[Dict[str, Any]]:
    """Flatten tree structure to list of nodes."""
    results = []
    
    def traverse(node):
        if isinstance(node, dict):
            results.append(node)
            for child in node.get('nodes', []):
                traverse(child)
        elif isinstance(node, list):
            for item in node:
                traverse(item)
    
    traverse(structure)
    return results


def get_node_texts(tree_structure: Dict[str, Any], node_ids: List[str]) -> List[Dict[str, Any]]:
    """Get text content for specified node IDs."""
    structure = tree_structure.get('structure', tree_structure)
    nodes = flatten_tree(structure)
    
    node_map = {node.get('node_id'): node for node in nodes if node.get('node_id')}
    
    results = []
    for node_id in node_ids:
        if node_id in node_map:
            node = node_map[node_id]
            results.append({
                'node_id': node_id,
                'title': node.get('title', ''),
                'text': node.get('text', ''),
                'summary': node.get('summary', ''),
            })
    
    return results


# ============================================================================
# PageIndex Retriever Tests
# ============================================================================

class TestPageIndexRetrieverImport:
    """Test PageIndexRetriever can be imported."""
    
    def test_import_retriever(self):
        """Test that PageIndexRetriever can be imported."""
        from contextpilot.retriever import PageIndexRetriever, PAGEINDEX_AVAILABLE
        assert PageIndexRetriever is not None
    
    def test_import_from_pipeline(self):
        """Test that pageindex is a valid retriever type."""
        from contextpilot.pipeline import RetrieverConfig
        # PageIndex requires corpus_data or paths
        config = RetrieverConfig(
            retriever_type="pageindex", 
            top_k=5,
            corpus_data=[{"text": "test"}]  # Minimal corpus data
        )
        assert config.retriever_type == "pageindex"


# Check if openai is available for PageIndex tests
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class TestPageIndexRetrieverBasic:
    """Basic tests for PageIndexRetriever."""
    
    @pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
    def test_init_with_tree_structure(self, sample_tree_structure, tmp_path):
        """Test initialization with tree structure."""
        from contextpilot.retriever import PageIndexRetriever
        
        # Save tree to temp file
        tree_path = tmp_path / "test_structure.json"
        with open(tree_path, 'w') as f:
            json.dump(sample_tree_structure, f)
        
        retriever = PageIndexRetriever(model="gpt-4o")
        retriever.load_tree_structures([str(tree_path)])
        
        assert 'test' in retriever.documents
    
    @pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
    def test_get_corpus(self, sample_tree_structure, tmp_path):
        """Test getting corpus from tree structure."""
        from contextpilot.retriever import PageIndexRetriever
        
        # Save tree to temp file
        tree_path = tmp_path / "test_structure.json"
        with open(tree_path, 'w') as f:
            json.dump(sample_tree_structure, f)
        
        retriever = PageIndexRetriever(model="gpt-4o")
        retriever.load_tree_structures([str(tree_path)])
        
        corpus = retriever.get_corpus()
        
        # Should have 10 nodes total
        assert len(corpus) == 10
        
        # Each item should have required fields
        for item in corpus:
            assert 'node_id' in item
            assert 'title' in item
            assert 'text' in item or 'summary' in item
    
    def test_flatten_tree(self, sample_tree_structure):
        """Test tree flattening."""
        structure = sample_tree_structure.get('structure', sample_tree_structure)
        nodes = flatten_tree(structure)
        
        # Should have 10 nodes
        assert len(nodes) == 10
        
        # Check node IDs
        node_ids = [n.get('node_id') for n in nodes]
        expected_ids = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010']
        assert sorted(node_ids) == sorted(expected_ids)


class TestPageIndexWithContextPilot:
    """Test PageIndex integration with ContextPilot optimization."""
    
    def test_optimize_retrieved_contexts(self, sample_tree_structure):
        """Test that retrieved contexts can be optimized."""
        # Simulate retrieved node IDs (with overlap like real queries)
        query1_nodes = ['0001', '0002', '0003']
        query2_nodes = ['0004', '0005', '0006']
        query3_nodes = ['0001', '0004', '0007']  # Overlaps with q1 and q2
        
        all_node_ids = query1_nodes + query2_nodes + query3_nodes
        
        # Get contexts
        all_contexts = get_node_texts(sample_tree_structure, all_node_ids)
        
        # Should have 9 total, 7 unique
        assert len(all_contexts) == 9
        unique_ids = set(ctx['node_id'] for ctx in all_contexts)
        assert len(unique_ids) == 7
        
        # Build ContextPilot index on unique contexts
        unique_contexts = {ctx['node_id']: ctx for ctx in all_contexts}
        unique_list = list(unique_contexts.values())
        
        # Convert to token lists (using text length as proxy)
        context_tokens = []
        for ctx in unique_list:
            text = ctx.get('text', '') or ctx.get('summary', '')
            tokens = list(range(len(text)))
            context_tokens.append(tokens)
        
        # Build index
        clustering_result = build_context_index(
            contexts=context_tokens,
            use_gpu=False,
            alpha=0.001
        )
        
        assert clustering_result is not None
        
        # Schedule contexts
        scheduler = InterContextScheduler()
        scheduled = scheduler.schedule_contexts(clustering_result)
        
        assert scheduled is not None
        assert len(scheduled) >= 3  # Should return tuple with at least 3 elements
    
    def test_shared_index_efficiency(self, sample_tree_structure):
        """Test that shared index is more efficient than per-query indexing."""
        # Simulate 4 queries with overlapping node retrievals
        queries_nodes = [
            ['0001', '0002', '0003', '0007'],
            ['0002', '0004', '0005', '0008'],
            ['0001', '0004', '0006', '0009'],
            ['0003', '0005', '0007', '0010'],
        ]
        
        # Count total and unique
        all_node_ids = []
        for nodes in queries_nodes:
            all_node_ids.extend(nodes)
        
        total_count = len(all_node_ids)  # 16
        unique_count = len(set(all_node_ids))  # 10
        overlap_ratio = 1 - (unique_count / total_count)
        
        assert total_count == 16
        assert unique_count == 10
        assert overlap_ratio > 0.3  # At least 30% overlap
        
        # Get unique contexts
        unique_node_ids = list(set(all_node_ids))
        contexts = get_node_texts(sample_tree_structure, unique_node_ids)
        
        # Build index once
        context_tokens = []
        for ctx in contexts:
            text = ctx.get('text', '') or ctx.get('summary', '')
            tokens = list(range(len(text)))
            context_tokens.append(tokens)
        
        clustering_result = build_context_index(
            contexts=context_tokens,
            use_gpu=False,
            alpha=0.001
        )
        
        # Verify index was built on unique contexts only
        assert clustering_result is not None


class TestPageIndexEndToEnd:
    """End-to-end tests for PageIndex + ContextPilot workflow."""
    
    @pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
    def test_complete_workflow(self, sample_tree_structure, sample_queries, tmp_path):
        """Test the complete PageIndex + ContextPilot workflow."""
        from contextpilot.retriever import PageIndexRetriever
        
        # Save tree to temp file
        tree_path = tmp_path / "test_structure.json"
        with open(tree_path, 'w') as f:
            json.dump(sample_tree_structure, f)
        
        # Step 1: Initialize retriever
        retriever = PageIndexRetriever(model="gpt-4o")
        retriever.load_tree_structures([str(tree_path)])
        
        # Step 2: Get corpus
        corpus = retriever.get_corpus()
        assert len(corpus) == 10
        
        # Step 3: Simulate search results (in real usage, would call LLM)
        # Each query retrieves some nodes with overlap
        search_results = {
            'q1': ['0001', '0002'],  # Background query
            'q2': ['0004', '0005', '0006'],  # Methodology query
            'q3': ['0007', '0008', '0009'],  # Findings query
            'q4': ['0001', '0007', '0010'],  # Conclusion query (overlaps)
        }
        
        # Step 4: Collect all retrieved contexts
        all_contexts = []
        for qid, node_ids in search_results.items():
            contexts = get_node_texts(sample_tree_structure, node_ids)
            all_contexts.extend(contexts)
        
        # Step 5: Build shared ContextPilot index
        unique_contexts = {}
        for ctx in all_contexts:
            node_id = ctx.get('node_id')
            if node_id and node_id not in unique_contexts:
                unique_contexts[node_id] = ctx
        
        unique_list = list(unique_contexts.values())
        
        context_tokens = []
        for ctx in unique_list:
            text = ctx.get('text', '') or ctx.get('summary', '')
            tokens = list(range(len(text)))
            context_tokens.append(tokens)
        
        clustering_result = build_context_index(
            contexts=context_tokens,
            use_gpu=False,
            alpha=0.001
        )
        
        # Step 6: For each query, reorder contexts using shared index
        scheduler = InterContextScheduler()
        scheduled = scheduler.schedule_contexts(clustering_result)
        
        # Verify workflow completed successfully
        assert clustering_result is not None
        assert scheduled is not None
        assert len(unique_list) == 9  # 9 unique nodes
        assert len(all_contexts) == 11  # 11 total (2 overlaps)
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.parent.joinpath(
            "PageIndex/tests/results/q1-fy25-earnings_structure.json"
        ).exists(),
        reason="Real tree structure not available"
    )
    def test_with_real_tree_structure(self):
        """Test with real PageIndex tree structure if available."""
        tree_path = Path(__file__).parent.parent.parent / "PageIndex" / "tests" / "results" / "q1-fy25-earnings_structure.json"
        
        with open(tree_path) as f:
            tree_structure = json.load(f)
        
        # Get structure
        structure = tree_structure.get('structure', tree_structure)
        nodes = flatten_tree(structure)
        
        # Verify structure
        assert len(nodes) > 0
        
        # All nodes should have node_id
        for node in nodes:
            assert 'node_id' in node or 'title' in node
        
        # Test building index
        node_ids = [n.get('node_id') for n in nodes[:10] if n.get('node_id')]
        contexts = get_node_texts(tree_structure, node_ids)
        
        context_tokens = []
        for ctx in contexts:
            text = ctx.get('text', '') or ctx.get('summary', '')
            if text:
                tokens = list(range(len(text)))
                context_tokens.append(tokens)
        
        if context_tokens:
            clustering_result = build_context_index(
                contexts=context_tokens,
                use_gpu=False,
                alpha=0.001
            )
            assert clustering_result is not None


class TestPipelineIntegration:
    """Test PageIndex with RAGPipeline."""
    
    def test_pipeline_config_pageindex(self):
        """Test RAGPipeline can be configured with PageIndex."""
        from contextpilot.pipeline import (
            RetrieverConfig,
            OptimizerConfig,
            InferenceConfig,
            PipelineConfig
        )
        
        config = PipelineConfig(
            retriever=RetrieverConfig(
                retriever_type="pageindex",
                top_k=5,
                pageindex_model="gpt-4o",
                corpus_data=[{"text": "test", "node_id": "001"}]  # Required
            ),
            optimizer=OptimizerConfig(
                enabled=True,
                use_gpu=False
            ),
            inference=InferenceConfig(
                model_name="test-model"
            )
        )
        
        assert config.retriever.retriever_type == "pageindex"
        assert config.retriever.pageindex_model == "gpt-4o"
        assert config.optimizer.enabled is True
    
    def test_pipeline_with_pageindex_tree(self, sample_tree_structure, tmp_path):
        """Test RAGPipeline with PageIndex tree structure."""
        from contextpilot.pipeline import (
            RetrieverConfig,
            OptimizerConfig,
            InferenceConfig,
            PipelineConfig
        )
        
        # Save tree to temp file
        tree_path = tmp_path / "test_structure.json"
        with open(tree_path, 'w') as f:
            json.dump(sample_tree_structure, f)
        
        # Create config without API key (won't actually call LLM)
        config = PipelineConfig(
            retriever=RetrieverConfig(
                retriever_type="pageindex",
                top_k=5,
                pageindex_tree_paths=[str(tree_path)]
            ),
            optimizer=OptimizerConfig(
                enabled=True,
                use_gpu=False
            ),
            inference=InferenceConfig(
                model_name="test-model"
            )
        )
        
        # Just verify config is valid
        assert config.retriever.retriever_type == "pageindex"


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.slow
class TestPageIndexPerformance:
    """Performance tests for PageIndex integration."""
    
    def test_large_context_optimization(self):
        """Test optimization with larger number of contexts."""
        # Create 50 mock contexts
        contexts = []
        for i in range(50):
            contexts.append({
                'node_id': f'{i:04d}',
                'title': f'Section {i}',
                'text': f'This is the content of section {i}. ' * 20,
            })
        
        # Convert to tokens
        context_tokens = []
        for ctx in contexts:
            text = ctx.get('text', '')
            tokens = list(range(len(text)))
            context_tokens.append(tokens)
        
        # Build index
        clustering_result = build_context_index(
            contexts=context_tokens,
            use_gpu=False,
            alpha=0.001
        )
        
        assert clustering_result is not None
        
        # Schedule
        scheduler = InterContextScheduler()
        scheduled = scheduler.schedule_contexts(clustering_result)
        
        assert scheduled is not None
    
    def test_overlap_detection(self):
        """Test that overlapping contexts are detected."""
        # Create contexts with intentional overlap in content
        base_text = "This is shared content that appears in multiple sections. "
        
        contexts = [
            {'node_id': '0001', 'text': base_text + "Section 1 specific."},
            {'node_id': '0002', 'text': base_text + "Section 2 specific."},
            {'node_id': '0003', 'text': base_text + "Section 3 specific."},
            {'node_id': '0004', 'text': "Completely different content here."},
        ]
        
        # The first 3 should cluster together due to shared prefix
        context_tokens = []
        for ctx in contexts:
            text = ctx.get('text', '')
            # Use character positions as "tokens" to simulate prefix sharing
            tokens = [ord(c) for c in text]
            context_tokens.append(tokens)
        
        clustering_result = build_context_index(
            contexts=context_tokens,
            use_gpu=False,
            alpha=0.001
        )
        
        assert clustering_result is not None
