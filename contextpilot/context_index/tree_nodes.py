"""
Tree Node Management Module

This module provides classes for managing cluster tree nodes in the context index.
It handles node creation, deduplication, and tree structure maintenance.
"""


class ClusterNode:
    """Represents a node in the clustering tree."""
    
    def __init__(self, node_id, content, original_indices=None, 
                 distance=0.0, children=None, parent=None, frequency=1):
        self.node_id = node_id
        self.content = set(content) if not isinstance(content, set) else content
        self.doc_ids = sorted(list(self.content))
        self.original_indices = original_indices or {node_id}
        self.distance = distance
        self.children = children or []
        self.parent = parent
        self.frequency = frequency
        self.merge_distance = distance
        self.search_path = []  # Path from root: list of child indices used to traverse from root to this node
    
    @property
    def is_leaf(self):
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    @property
    def is_root(self):
        """Check if this is a root node."""
        return self.parent is None
    
    @property
    def is_empty(self):
        """Check if this node has empty content."""
        return len(self.content) == 0
    
    def add_child(self, child_id):
        """Add a child to this node."""
        if child_id not in self.children and child_id != self.node_id:
            self.children.append(child_id)
    
    def remove_child(self, child_id):
        """Remove a child from this node."""
        if child_id in self.children:
            self.children.remove(child_id)
    
    def update_frequency(self, additional_frequency):
        """Update the frequency of this node."""
        self.frequency += additional_frequency
    
    def merge_with(self, other_node):
        """Merge content and indices with another node."""
        self.content = self.content.intersection(other_node.content)
        self.doc_ids = sorted(list(self.content))
        self.original_indices.update(other_node.original_indices)
        self.frequency += other_node.frequency
    
    def get_depth(self):
        """
        Get the depth of this node in the tree (distance from root).
        
        Returns:
            Integer depth (number of child index traversals from root).
        """
        return len(self.search_path)
    
    def __repr__(self):
        status = "Leaf" if self.is_leaf else "Internal"
        if self.search_path:
            path_str = "[" + "][".join(map(str, self.search_path)) + "]"
        else:
            path_str = "[] (root)"
        return (f"Node {self.node_id} ({status}): "
                f"content={len(self.content)} items, "
                f"frequency={self.frequency}, "
                f"children={len(self.children)}, "
                f"search_path={path_str}")


class NodeManager:
    """Manages cluster nodes and handles deduplication."""
    
    def __init__(self):
        self.cluster_nodes = {}
        self.unique_nodes = {}
        self.redirects = {}
        self.content_to_node_id = {}
    
    def create_leaf_node(self, node_id, prompt_content):
        """Create a leaf node, handling duplicates."""
        content_key = frozenset(prompt_content)
        
        if content_key in self.content_to_node_id:
            # Duplicate content found
            canonical_id = self.content_to_node_id[content_key]
            canonical_node = self.unique_nodes[canonical_id]
            canonical_node.update_frequency(1)
            canonical_node.original_indices.add(node_id)
            
            self.redirects[node_id] = canonical_id
            self.cluster_nodes[node_id] = canonical_node
            return canonical_node
        else:
            # New content
            node = ClusterNode(node_id, prompt_content)
            self.cluster_nodes[node_id] = node
            self.unique_nodes[node_id] = node
            self.content_to_node_id[content_key] = node_id
            return node
    
    def create_internal_node(self, node_id, child1_id, child2_id, distance):
        """Create an internal node from two children."""
        canonical_child1_id = self.redirects.get(child1_id, child1_id)
        canonical_child2_id = self.redirects.get(child2_id, child2_id)
        
        if canonical_child1_id == canonical_child2_id:
            # Self-reference, redirect to canonical child
            self.redirects[node_id] = canonical_child1_id
            self.cluster_nodes[node_id] = self.unique_nodes[canonical_child1_id]
            return self.unique_nodes[canonical_child1_id]
        
        child1 = self.unique_nodes[canonical_child1_id]
        child2 = self.unique_nodes[canonical_child2_id]
        
        # Create intersection content
        intersection_content = child1.content.intersection(child2.content)
        content_key = frozenset(intersection_content)
        
        # Check for existing content
        if content_key in self.content_to_node_id and len(intersection_content) > 0:
            canonical_id = self.content_to_node_id[content_key]
            if canonical_id not in [canonical_child1_id, canonical_child2_id]:
                # Use existing node
                existing_node = self.unique_nodes[canonical_id]
                existing_node.add_child(canonical_child1_id)
                existing_node.add_child(canonical_child2_id)
                existing_node.frequency = max(existing_node.frequency, 
                                            child1.frequency + child2.frequency)
                existing_node.original_indices.update(child1.original_indices)
                existing_node.original_indices.update(child2.original_indices)
                
                child1.parent = canonical_id
                child2.parent = canonical_id
                
                self.redirects[node_id] = canonical_id
                self.cluster_nodes[node_id] = existing_node
                return existing_node
        
        # Create new internal node
        combined_indices = child1.original_indices.union(child2.original_indices)
        node = ClusterNode(
            node_id, 
            intersection_content,
            combined_indices,
            distance,
            [canonical_child1_id, canonical_child2_id],
            frequency=child1.frequency + child2.frequency
        )
        
        self.cluster_nodes[node_id] = node
        self.unique_nodes[node_id] = node
        
        if len(intersection_content) > 0:
            self.content_to_node_id[content_key] = node_id
        
        child1.parent = node_id
        child2.parent = node_id
        
        return node
    
    def cleanup_empty_nodes(self):
        """
        Remove empty nodes and relink their children to their grandparents,
        preserving the tree hierarchy.
        """
        empty_node_ids = {
            node_id for node_id, node in self.unique_nodes.items() if node.is_empty
        }

        if not empty_node_ids:
            return
        
        # Sort empty nodes by ID to process from newest to oldest.
        # This helps avoid conflicts when dealing with nested empty nodes.
        sorted_empty_ids = sorted(list(empty_node_ids), reverse=True)

        for empty_id in sorted_empty_ids:
            if empty_id not in self.unique_nodes:
                continue # Node might have already been processed as a child

            empty_node = self.unique_nodes[empty_id]
            parent_id = empty_node.parent
            children_ids = empty_node.children

            # 1. Update the parent node (grandparent of the children)
            if parent_id is not None and parent_id in self.unique_nodes:
                parent_node = self.unique_nodes[parent_id]
                parent_node.remove_child(empty_id)
                for child_id in children_ids:
                    if child_id in self.unique_nodes:
                         parent_node.add_child(child_id)

            # 2. Update the children nodes
            for child_id in children_ids:
                if child_id in self.unique_nodes:
                    child_node = self.unique_nodes[child_id]
                    child_node.parent = parent_id # Re-parent to grandparent

            # 3. Delete the empty node
            del self.unique_nodes[empty_id]

        # Final validation pass to ensure all parent pointers are valid
        for node in self.unique_nodes.values():
            if node.parent is not None and node.parent not in self.unique_nodes:
                node.parent = None
    
    def get_node_stats(self):
        """Get statistics about the node tree."""
        total_nodes = len(self.unique_nodes)
        leaf_nodes = sum(1 for node in self.unique_nodes.values() if node.is_leaf)
        root_nodes = sum(1 for node in self.unique_nodes.values() if node.is_root)
        
        return {
            'total_nodes': total_nodes,
            'leaf_nodes': leaf_nodes,
            'root_nodes': root_nodes,
            'internal_nodes': total_nodes - leaf_nodes
        }
    
    def update_search_paths(self):
        """
        Update the search path for all nodes in the tree.
        The search path records the child indices to traverse from root to each node.
        This should be called after the tree structure is finalized.
        
        If there are multiple root nodes (forest), creates a virtual root to unify them.
        """
        # Find all current root nodes
        root_nodes = [node for node in self.unique_nodes.values() if node.is_root]
        
        if len(root_nodes) == 0:
            return  # No nodes in tree
        elif len(root_nodes) == 1:
            # Single root - just update paths from it
            root = root_nodes[0]
            root.search_path = []
            self._update_paths_from_node(root)
        else:
            # Multiple roots - create a virtual root node
            # Find a new ID that doesn't conflict
            virtual_root_id = max(self.unique_nodes.keys()) + 1
            
            # Create virtual root with empty content
            virtual_root = ClusterNode(
                node_id=virtual_root_id,
                content=set(),
                original_indices=set(),
                distance=0.0,
                children=[node.node_id for node in root_nodes],
                parent=None,
                frequency=sum(node.frequency for node in root_nodes)
            )
            virtual_root.search_path = []
            
            # Add virtual root to the tree
            self.unique_nodes[virtual_root_id] = virtual_root
            
            # Update all previous roots to point to this virtual root as parent
            for node in root_nodes:
                node.parent = virtual_root_id
            
            # Now update all paths starting from virtual root
            self._update_paths_from_node(virtual_root)
    
    def _update_paths_from_node(self, node):
        """
        Recursively update search paths starting from a given node.
        
        Args:
            node: The current node whose children's paths will be updated
        """
        # Update paths for all children
        for child_index, child_id in enumerate(node.children):
            if child_id in self.unique_nodes:
                child_node = self.unique_nodes[child_id]
                # Child's path = parent's path + child_index
                child_node.search_path = node.search_path + [child_index]
                # Recursively update this child's descendants
                self._update_paths_from_node(child_node)
