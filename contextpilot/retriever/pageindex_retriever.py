"""
PageIndex Retriever for ContextPilot.

This module provides integration with PageIndex, a reasoning-based RAG framework
that uses tree structures and LLM reasoning for document retrieval instead of
traditional vector similarity search.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Optional imports - only required when using PageIndex
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import PageIndex
PAGEINDEX_AVAILABLE = False
PAGEINDEX_LOCAL = False  # Whether local page_index function is available
PAGEINDEX_CLIENT = False  # Whether PageIndexClient (API) is available

try:
    # Try importing PageIndexClient (PyPI version)
    from pageindex import PageIndexClient
    PAGEINDEX_CLIENT = True
    PAGEINDEX_AVAILABLE = True
except ImportError:
    PageIndexClient = None

try:
    # Try importing local page_index function (local repo version)
    from pageindex import page_index as local_page_index
    PAGEINDEX_LOCAL = True
    PAGEINDEX_AVAILABLE = True
except ImportError:
    local_page_index = None

# Try importing utilities
try:
    from pageindex.utils import remove_fields, print_tree, create_node_mapping
except ImportError:
    remove_fields = None
    print_tree = None
    create_node_mapping = None


def structure_to_list(structure):
    """Convert nested tree structure to flat list of nodes."""
    if isinstance(structure, dict):
        nodes = [structure]
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes
    return []


def _remove_fields(data, fields=None):
    """Remove specified fields from nested data structure."""
    if fields is None:
        fields = ['text']
    if isinstance(data, dict):
        return {k: _remove_fields(v, fields) 
                for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [_remove_fields(item, fields) for item in data]
    return data


def _print_tree(tree, indent=0):
    """Print tree structure."""
    if isinstance(tree, list):
        for node in tree:
            _print_tree(node, indent)
    elif isinstance(tree, dict):
        print('  ' * indent + str(tree.get('title', 'Untitled')))
        if 'nodes' in tree:
            _print_tree(tree['nodes'], indent + 1)


# Use imported functions if available, otherwise use local implementations
if remove_fields is None:
    remove_fields = _remove_fields
if print_tree is None:
    print_tree = _print_tree


class PageIndexRetriever:
    """
    Retriever that uses PageIndex's reasoning-based tree search for document retrieval.
    
    PageIndex builds a hierarchical tree index from documents and uses LLM reasoning
    to navigate the tree and find relevant sections, simulating how human experts
    read and search documents.
    
    This is in contrast to traditional vector-based RAG which relies on semantic
    similarity. PageIndex's reasoning-based approach is particularly effective for:
    - Professional/technical documents
    - Documents with complex structure (financial reports, legal documents, etc.)
    - Multi-hop reasoning questions
    
    Examples:
        >>> # Basic usage with PDF documents
        >>> retriever = PageIndexRetriever(
        ...     model="gpt-4o",
        ...     openai_api_key="your-api-key"
        ... )
        >>> retriever.index_documents(["report.pdf", "manual.pdf"])
        >>> results = retriever.search_queries(
        ...     query_data=[{"question": "What is the revenue for Q1?"}],
        ...     top_k=5
        ... )
        
        >>> # With pre-built tree structures
        >>> retriever = PageIndexRetriever(model="gpt-4o")
        >>> retriever.load_tree_structures(["report_structure.json"])
        >>> results = retriever.search_queries(query_data=[...])
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        openai_api_key: Optional[str] = None,
        tree_cache_dir: Optional[str] = None,
        toc_check_page_num: int = 20,
        max_page_num_each_node: int = 20,
        verbose: bool = True
    ):
        """
        Initialize the PageIndex Retriever.
        
        Args:
            model: OpenAI model to use for tree generation and search reasoning.
            openai_api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            tree_cache_dir: Directory to cache generated tree structures.
            toc_check_page_num: Number of pages to check for table of contents.
            max_page_num_each_node: Maximum pages per tree node.
            verbose: Whether to print progress messages.
        """
        # PageIndex is optional for loading pre-built trees
        # but required for building new trees from PDFs
        
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI is required for PageIndex. Install it with: pip install openai"
            )
        
        self.model = model
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.tree_cache_dir = tree_cache_dir
        self.toc_check_page_num = toc_check_page_num
        self.max_page_num_each_node = max_page_num_each_node
        self.verbose = verbose
        
        # Document storage
        self.documents: Dict[str, Dict[str, Any]] = {}  # doc_id -> tree structure
        self.node_maps: Dict[str, Dict[str, Any]] = {}  # doc_id -> node_id -> node
        self.corpus: List[Dict[str, Any]] = []  # Flat list of nodes as corpus items
        
        if self.tree_cache_dir:
            os.makedirs(self.tree_cache_dir, exist_ok=True)
    
    def _log(self, message: str):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def index_documents(
        self,
        document_paths: List[str],
        force_rebuild: bool = False
    ):
        """
        Build PageIndex tree structures for PDF documents.
        
        Args:
            document_paths: List of paths to PDF documents.
            force_rebuild: If True, rebuild trees even if cached versions exist.
        """
        for doc_path in document_paths:
            self._index_single_document(doc_path, force_rebuild)
        
        self._build_corpus_from_trees()
    
    def _index_single_document(self, doc_path: str, force_rebuild: bool = False):
        """Index a single PDF document."""
        doc_name = Path(doc_path).stem
        
        # Check cache
        cache_path = None
        if self.tree_cache_dir:
            cache_path = os.path.join(self.tree_cache_dir, f"{doc_name}_structure.json")
            if os.path.exists(cache_path) and not force_rebuild:
                self._log(f"📂 Loading cached tree for: {doc_name}")
                with open(cache_path, 'r') as f:
                    result = json.load(f)
                    self.documents[doc_name] = result
                    self._build_node_map(doc_name, result.get('structure', result))
                    return
        
        # Check if we can build trees locally
        if not PAGEINDEX_LOCAL:
            raise ImportError(
                "Local PageIndex tree building is not available. "
                "Either use pre-built tree structures with load_tree_structures(), "
                "or install the full PageIndex package from the repository."
            )
        
        self._log(f"🌲 Building PageIndex tree for: {doc_path}")
        
        # Generate tree structure using local page_index function
        result = local_page_index(
            doc_path,
            model=self.model,
            toc_check_page_num=self.toc_check_page_num,
            max_page_num_each_node=self.max_page_num_each_node,
            if_add_node_id='yes',
            if_add_node_summary='yes',
            if_add_node_text='yes'
        )
        
        self.documents[doc_name] = result
        self._build_node_map(doc_name, result.get('structure', result))
        
        # Save to cache
        if cache_path:
            with open(cache_path, 'w') as f:
                json.dump(result, f, indent=2)
            self._log(f"💾 Cached tree structure: {cache_path}")
    
    def load_tree_structures(self, structure_paths: List[str]):
        """
        Load pre-built PageIndex tree structures from JSON files.
        
        Args:
            structure_paths: List of paths to tree structure JSON files.
        """
        for path in structure_paths:
            doc_name = Path(path).stem.replace('_structure', '')
            self._log(f"📂 Loading tree structure: {path}")
            
            with open(path, 'r') as f:
                result = json.load(f)
            
            self.documents[doc_name] = result
            structure = result.get('structure', result)
            if isinstance(structure, list):
                self._build_node_map(doc_name, structure)
            else:
                self._build_node_map(doc_name, [structure])
        
        self._build_corpus_from_trees()
    
    def _build_node_map(self, doc_name: str, structure: Union[List, Dict]):
        """Build a node_id -> node mapping for quick lookup."""
        if isinstance(structure, dict):
            structure = [structure]
        
        # Flatten structure to list
        nodes = structure_to_list(structure)
        node_map = {}
        
        for node in nodes:
            node_id = node.get('node_id')
            if node_id:
                node_map[node_id] = node
        
        self.node_maps[doc_name] = node_map
    
    def _build_corpus_from_trees(self):
        """Build a flat corpus list from all tree nodes for ContextPilot compatibility."""
        self.corpus = []
        chunk_id = 0
        
        for doc_name, doc_data in self.documents.items():
            structure = doc_data.get('structure', doc_data)
            if isinstance(structure, dict):
                structure = [structure]
            
            nodes = structure_to_list(structure)
            
            for node in nodes:
                corpus_item = {
                    'chunk_id': chunk_id,
                    'doc_name': doc_name,
                    'node_id': node.get('node_id', ''),
                    'title': node.get('title', ''),
                    'text': node.get('text', ''),
                    'summary': node.get('summary', ''),
                    'start_index': node.get('start_index'),
                    'end_index': node.get('end_index'),
                }
                self.corpus.append(corpus_item)
                chunk_id += 1
        
        self._log(f"📊 Built corpus with {len(self.corpus)} nodes from {len(self.documents)} documents")
    
    def index_corpus(
        self,
        corpus_file: Optional[str] = None,
        corpus_data: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Index corpus for compatibility with other retrievers.
        
        For PageIndex, this is typically done via index_documents() or load_tree_structures(),
        but this method allows using pre-processed corpus data directly.
        """
        if corpus_data:
            self.corpus = corpus_data
            self._log(f"📊 Loaded corpus with {len(self.corpus)} items")
        elif corpus_file:
            with open(corpus_file, 'r') as f:
                self.corpus = [json.loads(line) for line in f]
            self._log(f"📊 Loaded corpus from {corpus_file} with {len(self.corpus)} items")
    
    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0
    ) -> str:
        """Call LLM for reasoning."""
        client = AsyncOpenAI(api_key=self.openai_api_key)
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    async def _tree_search_single(
        self,
        query: str,
        doc_name: str,
        top_k: int = 5
    ) -> List[str]:
        """
        Perform tree search for a single query on a single document.
        
        Returns list of node_ids that are relevant to the query.
        """
        doc_data = self.documents.get(doc_name)
        if not doc_data:
            return []
        
        structure = doc_data.get('structure', doc_data)
        
        # Create tree without full text for search
        tree_for_search = remove_fields(structure, fields=['text'])
        
        search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_for_search, indent=2)}

Please reply in the following JSON format:
{{
    "thinking": "<Your thinking process on which nodes are relevant to the question>",
    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}}
Directly return the final JSON structure. Do not output anything else.
"""
        
        try:
            result = await self._call_llm(search_prompt)
            result_json = json.loads(result)
            node_ids = result_json.get('node_list', [])
            
            # Limit to top_k
            return node_ids[:top_k]
        except Exception as e:
            self._log(f"⚠️ Tree search failed: {e}")
            return []
    
    async def _tree_search_all_docs(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform tree search across all indexed documents.
        
        Returns list of corpus items (nodes) relevant to the query.
        """
        all_results = []
        
        # Search each document
        tasks = [
            self._tree_search_single(query, doc_name, top_k)
            for doc_name in self.documents.keys()
        ]
        
        doc_results = await asyncio.gather(*tasks)
        
        # Collect matching nodes
        for doc_name, node_ids in zip(self.documents.keys(), doc_results):
            node_map = self.node_maps.get(doc_name, {})
            for node_id in node_ids:
                if node_id in node_map:
                    node = node_map[node_id]
                    # Find corresponding corpus item
                    for item in self.corpus:
                        if item.get('doc_name') == doc_name and item.get('node_id') == node_id:
                            all_results.append(item)
                            break
        
        return all_results[:top_k]
    
    def search_queries(
        self,
        queries_file: Optional[str] = None,
        query_data: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search queries using PageIndex reasoning-based tree search.
        
        Args:
            queries_file: Path to JSONL file containing queries.
            query_data: List of query dictionaries with 'question' field.
            top_k: Number of top nodes to retrieve per query.
            
        Returns:
            List of result dictionaries with 'text', 'top_k_doc_id', etc.
        """
        if queries_file:
            with open(queries_file, 'r') as f:
                query_data = [json.loads(line) for line in f]
        
        if not query_data:
            return []
        
        async def process_all_queries():
            results = []
            for query in query_data:
                query_text = query.get('question', query.get('text', ''))
                
                # Perform tree search
                matching_nodes = await self._tree_search_all_docs(query_text, top_k)
                
                # Extract chunk_ids
                top_k_doc_ids = [node.get('chunk_id') for node in matching_nodes]
                
                result = {
                    'text': query_text,
                    'top_k_doc_id': top_k_doc_ids,
                }
                
                # Add optional fields
                if 'qid' in query:
                    result['qid'] = query['qid']
                if 'answer' in query:
                    result['answer'] = query['answer']
                
                results.append(result)
            
            return results
        
        return asyncio.run(process_all_queries())
    
    def get_corpus(self) -> List[Dict[str, Any]]:
        """Return the corpus for ContextPilot optimization."""
        return self.corpus
    
    def get_corpus_map(self) -> Dict[int, Dict[str, Any]]:
        """Return chunk_id -> corpus item mapping."""
        return {item['chunk_id']: item for item in self.corpus}
    
    def get_tree_structure(self, doc_name: str) -> Optional[Dict[str, Any]]:
        """Get tree structure for a specific document."""
        return self.documents.get(doc_name)
    
    def print_tree(self, doc_name: str):
        """Print tree structure for a document."""
        doc_data = self.documents.get(doc_name)
        if doc_data:
            structure = doc_data.get('structure', doc_data)
            print_tree(structure)
        else:
            print(f"Document '{doc_name}' not found")
