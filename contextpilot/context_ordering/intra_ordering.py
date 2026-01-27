"""
Intra-Context Ordering Module

This module provides intra-context reordering functionality based on
clustering tree structure to maximize prefix sharing and cache efficiency
within individual contexts.
"""

from typing import List, Dict, Any


class IntraContextOrderer:
    """
    Handles intra-context reordering based on clustering results.
    
    This class analyzes the clustering tree structure to determine optimal
    ordering of elements within individual contexts that maximizes shared 
    prefixes, improving cache hit rates and reducing computational overhead.
    """
    
    def reorder_contexts(self, original_contexts: List[List[int]], 
                        unique_nodes: Dict[int, Any]) -> List[List[int]]:
        """
        Reorder contexts based on clustering tree structure.
        
        Uses a top-down approach:
        1. Traverse from root to leaves
        2. Each node inherits its parent's prefix
        3. Leaf nodes' reordered content becomes the final context ordering
        
        Args:
            original_contexts: List of original context lists (each is a list of chunk IDs)
            unique_nodes: Dictionary of unique tree nodes from clustering
            
        Returns:
            List of reordered contexts with optimized prefix sharing
        """
        # Find root node
        root_node = None
        for node_id, node in unique_nodes.items():
            if node.is_root:
                root_node = node
                break
        
        if not root_node:
            return original_contexts
        
        # First, assign original contexts to leaf nodes
        for node_id, node in unique_nodes.items():
            if node.is_leaf and node.original_indices:
                first_idx = min(node.original_indices)
                if first_idx < len(original_contexts):
                    node.doc_ids = list(original_contexts[first_idx])
        
        # Top-down traversal: reorder each node to start with parent's prefix
        from collections import deque
        queue = deque([root_node.node_id])
        visited = set()
        
        while queue:
            node_id = queue.popleft()
            if node_id in visited or node_id not in unique_nodes:
                continue
            visited.add(node_id)
            
            node = unique_nodes[node_id]
            
            # If not root and has parent, reorder to start with parent's prefix
            if not node.is_root and node.parent is not None:
                parent_node = unique_nodes.get(node.parent)
                if parent_node and parent_node.doc_ids and node.doc_ids:
                    node.doc_ids = self._reorder_with_parent_prefix(
                        node.doc_ids,
                        parent_node.doc_ids
                    )
            
            # Add children to queue
            if node.children:
                for child_id in node.children:
                    if child_id in unique_nodes:
                        queue.append(child_id)
        
        # Extract reordered contexts from leaf nodes
        reordered_contexts = []
        for i, original_context in enumerate(original_contexts):
            leaf_node = self._find_leaf_node(i, unique_nodes)
            if leaf_node and leaf_node.doc_ids:
                reordered_contexts.append(list(leaf_node.doc_ids))
            else:
                reordered_contexts.append(list(original_context))
        
        return reordered_contexts
    
    def _update_tree_and_reorder_nodes(
        self,
        unique_nodes: Dict[int, Any],
        reordered_contexts: List[List[int]]
    ) -> None:
        """
        Update tree node contents and reorder all nodes based on parent prefix.
        
        This method:
        1. Updates leaf nodes with reordered contexts
        2. Reorders each node (except root's children) to start with parent's prefix
        
        Args:
            unique_nodes: Dictionary of unique tree nodes
            reordered_contexts: List of reordered contexts
        """
        # Find root node
        root_node = None
        for node_id, node in unique_nodes.items():
            if node.is_root:
                root_node = node
                break
        
        # Update leaf nodes with reordered contexts
        for node_id, node in unique_nodes.items():
            if node.is_leaf and node.original_indices:
                first_idx = min(node.original_indices)
                if first_idx < len(reordered_contexts):
                    node.doc_ids = reordered_contexts[first_idx]
        
        # Reorder all nodes based on parent prefix using BFS
        if root_node:
            from collections import deque
            queue = deque()
            
            # Start with root's children - they don't get reordered
            if root_node.children:
                for child_id in root_node.children:
                    if child_id in unique_nodes:
                        queue.append((child_id, True))  # (node_id, is_child_of_root)
            
            # Process nodes level by level
            while queue:
                node_id, is_child_of_root = queue.popleft()
                
                if node_id not in unique_nodes:
                    continue
                
                node = unique_nodes[node_id]
                
                # Reorder this node based on parent's prefix (except root's children)
                if not is_child_of_root and node.parent is not None:
                    parent_node = unique_nodes.get(node.parent)
                    if parent_node and parent_node.doc_ids and node.doc_ids:
                        # Reorder to start with parent's prefix
                        node.doc_ids = self._reorder_with_parent_prefix(
                            node.doc_ids,
                            parent_node.doc_ids
                        )
                
                # Add children to queue
                if node.children:
                    for child_id in node.children:
                        if child_id in unique_nodes:
                            queue.append((child_id, False))
    
    def _reorder_with_parent_prefix(
        self,
        node_docs: List[int],
        parent_docs: List[int]
    ) -> List[int]:
        """
        Reorder a node's documents to start with the parent's prefix.
        
        Example: node=[2,3,4,1], parent=[1,2,3] -> result=[1,2,3,4]
        
        Args:
            node_docs: The node's current document list
            parent_docs: The parent's document list (to be used as prefix)
            
        Returns:
            Reordered document list starting with parent's prefix
        """
        if not parent_docs:
            return node_docs
        
        # Create a new list starting with parent's prefix
        result = list(parent_docs)
        
        # Add remaining documents from node_docs that are not in parent_docs
        parent_set = set(parent_docs)
        for doc in node_docs:
            if doc not in parent_set:
                result.append(doc)
        
        return result
    
    def _reorder_context_with_tree_prefix(
        self,
        context_index: int,
        original_context: List[int],
        unique_nodes: Dict[int, Any]
    ) -> List[int]:
        """
        Reorder a context to match its tree path's prefix structure.
        
        This ensures that the context starts with the prefix from its
        parent nodes in the tree, maintaining cache efficiency.
        
        Args:
            context_index: Index of the context
            original_context: Original context list
            unique_nodes: Dictionary of unique tree nodes
            
        Returns:
            Reordered context matching tree structure
        """
        # Find the leaf node for this context
        leaf_node = self._find_leaf_node(context_index, unique_nodes)
        if not leaf_node:
            return list(original_context)
        
        # Build the prefix by traversing from leaf to root
        prefix_docs = []
        visited = set()
        current_node = leaf_node
        
        # Traverse up to root, collecting prefixes
        ancestors = []
        while current_node and not current_node.is_root:
            if id(current_node) in visited:
                break
            visited.add(id(current_node))
            ancestors.append(current_node)
            
            if current_node.parent is not None and current_node.parent in unique_nodes:
                current_node = unique_nodes[current_node.parent]
            else:
                break
        
        # Reverse to get root-to-leaf order
        ancestors.reverse()
        
        # Build prefix from ancestors
        seen_docs = set()
        for ancestor in ancestors:
            if ancestor.doc_ids:
                for doc in ancestor.doc_ids:
                    if doc not in seen_docs:
                        prefix_docs.append(doc)
                        seen_docs.add(doc)
        
        # Add remaining documents from original context
        result = prefix_docs[:]
        for doc in original_context:
            if doc not in seen_docs:
                result.append(doc)
                seen_docs.add(doc)
        
        return result
    
    def extract_search_paths(
        self,
        unique_nodes: Dict[int, Any],
        num_contexts: int
    ) -> List[List[int]]:
        """
        Extract search paths (child indices) for each context.
        
        The search path represents the child indices to follow at each level
        when traversing from root to leaf. For example, [0, 1, 0] means:
        take the 0th child of root, then the 1st child of that node, then
        the 0th child of that node.
        
        Args:
            unique_nodes: Dictionary of unique tree nodes
            num_contexts: Total number of contexts
            
        Returns:
            List of search paths, where each path is a list of child indices
        """
        search_paths = [[] for _ in range(num_contexts)]
        
        # Build a mapping from context index to its leaf node
        context_to_leaf = {}
        for node_id, node in unique_nodes.items():
            if node.is_leaf:
                for orig_idx in node.original_indices:
                    context_to_leaf[orig_idx] = node_id
        
        # Extract search paths for each context
        for context_idx in range(num_contexts):
            if context_idx not in context_to_leaf:
                search_paths[context_idx] = []
                continue
            
            # Trace upward from leaf to root, recording child indices
            child_indices = []
            current_id = context_to_leaf[context_idx]
            visited = set()
            
            while current_id is not None and current_id in unique_nodes:
                if current_id in visited:
                    break
                visited.add(current_id)
                
                current_node = unique_nodes[current_id]
                
                # Find this node's position in parent's children list
                if current_node.parent is not None and current_node.parent in unique_nodes:
                    parent_node = unique_nodes[current_node.parent]
                    try:
                        child_index = parent_node.children.index(current_id)
                        child_indices.append(child_index)
                    except (ValueError, AttributeError):
                        pass
                
                current_id = current_node.parent
            
            # Reverse to get root-to-leaf path of child indices
            search_paths[context_idx] = child_indices[::-1]
        
        return search_paths
    
    def _reorder_single_context(self, context_index: int, 
                               original_context: List[int],
                               unique_nodes: Dict[int, Any]) -> List[int]:
        """
        Reorder a single context based on its position in the clustering tree.
        
        This method finds the leaf node containing this context, then traverses
        up the tree to find the best ancestor node with high frequency (shared
        with other contexts). The content from this ancestor becomes the prefix,
        followed by the remaining unique elements.
        
        Args:
            context_index: Index of the context in the original list
            original_context: Original context as list of chunk IDs
            unique_nodes: Dictionary of unique tree nodes
            
        Returns:
            Reordered context as list of chunk IDs
        """
        original_set = set(original_context)
        
        # Find the leaf node containing this context
        leaf_node = self._find_leaf_node(context_index, unique_nodes)
        if not leaf_node:
            return list(original_context)
        
        # If leaf is root or has frequency > 1, use it directly
        if leaf_node.is_root:
            return sorted(list(leaf_node.content))
        
        if leaf_node.frequency > 1:
            prefix_content = leaf_node.content
            prefix_list = sorted(list(prefix_content))
            remaining_elements = original_set - prefix_content
            remaining_list = sorted(list(remaining_elements))
            return prefix_list + remaining_list
        
        # Find best ancestor with frequency > 1
        best_node = self._find_best_ancestor(leaf_node, unique_nodes)
        
        if best_node:
            prefix_content = best_node.content
            prefix_list = sorted(list(prefix_content))
            remaining_elements = original_set - prefix_content
            remaining_list = sorted(list(remaining_elements))
            return prefix_list + remaining_list
        else:
            return list(original_context)
    
    def _find_leaf_node(self, context_index: int, unique_nodes: Dict[int, Any]):
        """
        Find the leaf node that contains the given context index.
        
        Args:
            context_index: Index of the context to find
            unique_nodes: Dictionary of unique tree nodes
            
        Returns:
            The leaf node containing this context, or None if not found
        """
        for node in unique_nodes.values():
            if context_index in node.original_indices and node.is_leaf:
                return node
        return None
    
    def _find_best_ancestor(self, start_node, unique_nodes: Dict[int, Any]):
        """
        Find the best ancestor by traversing up the tree.
        
        Returns the FIRST ancestor with a frequency > 1 and non-empty content.
        This ancestor represents a shared prefix that appears in multiple contexts,
        making it ideal for cache efficiency.
        
        Args:
            start_node: The starting node (typically a leaf)
            unique_nodes: Dictionary of unique tree nodes
            
        Returns:
            The best ancestor node, or None if no suitable ancestor is found
        """
        current_node = start_node

        while current_node.parent is not None:
            parent_id = current_node.parent
            
            # Handle cases where the parent might not exist in the final tree
            if parent_id not in unique_nodes:
                return None 
                
            parent_node = unique_nodes[parent_id]
            
            # Check if the parent is a valid, shareable cluster
            if parent_node.frequency > 1 and not parent_node.is_empty:
                # Found the first valid ancestor, return it immediately.
                return parent_node 
            
            # Move up to the next parent for the next iteration
            current_node = parent_node
        
        # Reached the root without finding a suitable ancestor
        return None
    
    # Legacy method names for backward compatibility
    def reorder_prompts(self, original_prompts: List[List[int]], 
                       unique_nodes: Dict[int, Any]) -> List[List[int]]:
        """
        Legacy method name for backward compatibility.
        Redirects to reorder_contexts.
        """
        return self.reorder_contexts(original_prompts, unique_nodes)
    
    def _reorder_single_prompt(self, prompt_index: int, 
                              original_prompt: List[int],
                              unique_nodes: Dict[int, Any]) -> List[int]:
        """
        Legacy method name for backward compatibility.
        Redirects to _reorder_single_context.
        """
        return self._reorder_single_context(prompt_index, original_prompt, unique_nodes)
