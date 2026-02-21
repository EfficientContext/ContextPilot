"""
Live Context Index

Dynamic context index supporting search, insertion, and eviction notification.
Implements the algorithms described in the ContextPilot paper.

Inherits from ContextIndex to provide:
1. Initial construction and clustering
2. Intra-context reordering  
3. Inter-context scheduling
4. Then becomes live for dynamic updates

Key Design for Request Tracking:
- request_id is ONLY stored on leaf nodes (actual requests)
- Parent/intermediate nodes do NOT have request_id values
- Eviction is driven by SGLang's radix cache callback (not internal heap)
- Tree is automatically pruned when branches become empty

IMPORTANT: Eviction is now handled via callback from SGLang's radix cache.
When SGLang evicts a request from its cache, it calls our callback with
the evicted request_ids. No need to maintain a separate eviction heap.
"""

import time
import uuid
import logging
import warnings
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import deque

from ..context_index.index_construction import ContextIndex
from ..context_index.tree_nodes import ClusterNode
from ..context_index.compute_distance_cpu import compute_distance_single, compute_distances_batch
from ..context_ordering import InterContextScheduler
from .metadata import NodeMetadata

logger = logging.getLogger(__name__)


def compute_prefix_length(list1: List[int], list2: List[int]) -> int:
    """Compute the length of common prefix between two lists."""
    length = 0
    for a, b in zip(list1, list2):
        if a == b:
            length += 1
        else:
            break
    return length


class ContextPilot(ContextIndex):
    """
    Live context index with dynamic updates and request tracking.
    
    Workflow:
    1. Build initial index: build_and_schedule() -> constructs tree, reorders, schedules
    2. Go live: enable dynamic search, insert operations
    3. Track requests: each leaf node has a unique request_id
    4. Eviction: handled via callback from SGLang's radix cache
    
    Key invariants:
    - Only leaf nodes have request_id values
    - Parent/intermediate nodes do NOT have request_id
    - Tree is automatically pruned when branches become empty
    - Eviction is DRIVEN BY SGLANG (not internal heap)
    
    Supports:
    - Context search: O(|C| · log n)
    - Node traversal: O(h)
    - Context insertion: O(1) or O(|C|)
    - Request removal: O(h) for pruning
    """
    
    def __init__(self, alpha: float = 0.001, use_gpu: bool = False,
                 linkage_method: str = "average", batch_size: int = 10000):
        """
        Initialize live context index.
        
        Args:
            alpha: Distance computation parameter
            use_gpu: Whether to use GPU for distance computation
            linkage_method: Linkage method for hierarchical clustering
            batch_size: Batch size for distance computation
        """
        # Initialize parent ContextIndex
        super().__init__(alpha=alpha, use_gpu=use_gpu,
                        linkage_method=linkage_method, batch_size=batch_size)
        
        # Additional components for live operations
        self.metadata: Dict[int, NodeMetadata] = {}
        self.inter_scheduler = InterContextScheduler()
        
        # Request tracking
        self._request_to_node: Dict[str, int] = {}  # request_id -> node_id
        self._next_request_counter: int = 0  # Counter for auto-generating request_ids
        
        # Conversation tracking for multi-turn deduplication
        # conversation_id -> {"seen_docs": set, "turn_count": int}
        self._conversations: Dict[str, Dict] = {}
        self._has_explicit_conversation: bool = False
        
        # Track if index is live
        self.is_live = False
        self.initial_result = None
        self.scheduled_result = None
        
        # Tree structure aliases (for backwards compatibility)
        self.nodes: Dict[int, ClusterNode] = {}
        self.root_id: Optional[int] = None
        self.next_node_id: int = 0
        
        # Statistics for live operations
        self.live_stats = {
            'total_searches': 0,
            'total_insertions': 0,
            'total_evictions': 0,
            'total_search_time_us': 0,
            'total_traversal_time_us': 0,
        }
    
    def get_all_request_ids(self) -> Set[str]:
        """Get all tracked request IDs."""
        return set(self._request_to_node.keys())
    
    def reset(self):
        """
        Reset the index to initial state.
        
        Clears all nodes, metadata, and request tracking.
        Use this to start fresh without creating a new instance.
        """
        self.metadata.clear()
        self._request_to_node.clear()
        self._next_request_counter = 0
        self.is_live = False
        self.initial_result = None
        self.scheduled_result = None
        self.nodes.clear()
        self.root_id = None
        self.next_node_id = 0
        self.live_stats = {
            'total_searches': 0,
            'total_insertions': 0,
            'total_evictions': 0,
            'total_search_time_us': 0,
            'total_traversal_time_us': 0,
        }
        logger.info("Index reset to initial state")

    def build_and_schedule(self, contexts: List[List[int]], 
                          initial_tokens_per_context: int = 0) -> Dict:
        """
        Build index, reorder contexts, schedule execution, and go live.
        
        This is the main entry point that combines:
        1. fit_transform() - build tree and reorder contexts
        2. Inter-context scheduling - optimize execution order
        3. Initialize live metadata - prepare for dynamic updates
        
        Args:
            contexts: List of contexts (each is a list of document IDs)
            initial_tokens_per_context: Initial token count for each context
            
        Returns:
            Dictionary with scheduled results, index info, and request_id mapping
        """
        print("=" * 80)
        print("BUILDING LIVE CONTEXT INDEX")
        print("=" * 80)
        
        # Step 1: Build static index (clustering + reordering)
        print("\n1. Building static index...")
        self.initial_result = self.fit_transform(contexts)
        
        print(f"   ✓ Built tree with {self.initial_result.stats['total_nodes']} nodes")
        print(f"   ✓ Leaf nodes: {self.initial_result.stats['leaf_nodes']}")
        
        # Step 2: Inter-context scheduling
        print("\n2. Scheduling contexts for optimal execution...")
        scheduled_reordered, scheduled_originals, final_mapping, groups = \
            self.inter_scheduler.schedule_contexts(self.initial_result)
        
        print(f"   ✓ Created {len(groups)} execution groups")
        
        self.scheduled_result = {
            'reordered_contexts': scheduled_reordered,
            'original_indices': final_mapping,
            'scheduled_originals': scheduled_originals,
            'groups': groups,
            'clustering_result': self.initial_result
        }
        
        # Step 3: Initialize live metadata (auto-generates request_ids)
        print("\n3. Initializing live metadata...")
        request_id_mapping, request_ids_ordered = self._initialize_live_metadata(
            initial_tokens_per_context, 
            num_input_contexts=len(contexts)
        )
        
        print(f"   ✓ Initialized {len(self.metadata)} nodes with metadata")
        print(f"   ✓ Auto-assigned {len(request_id_mapping)} request IDs")
        
        # Add request_id mapping to result (dict and ordered list)
        self.scheduled_result['request_id_mapping'] = request_id_mapping
        self.scheduled_result['request_ids'] = request_ids_ordered  # Ordered list matching input contexts
        
        # Step 4: Mark as live
        self.is_live = True
        
        print("\n" + "=" * 80)
        print("✓ INDEX IS NOW LIVE - Ready for dynamic operations")
        print("=" * 80 + "\n")
        
        return self.scheduled_result
    
    _DEFAULT_CONVERSATION = "_default"

    def reorder(self, contexts, initial_tokens_per_context: int = 0,
                conversation_id: Optional[str] = None):
        """Reorder contexts for optimal KV-cache prefix sharing.

        This is the primary user-facing method.  It handles both the
        initial (cold-start) build and subsequent incremental updates
        transparently — callers never need to distinguish between them.

        Args:
            contexts: A single context (``List[int]`` / ``List[str]``)
                or a batch of contexts (``List[List[int]]`` /
                ``List[List[str]]``).  A single list is automatically
                wrapped into ``[contexts]``.
            initial_tokens_per_context: Initial token budget per context
                (used for eviction tracking; 0 to ignore).
            conversation_id: Conversation key for multi-turn
                deduplication.  Defaults to a shared internal key so
                single-conversation callers can omit it.  For
                multi-user / multi-session scenarios, pass a unique
                key (e.g. ``user_id`` or ``session_id``).

        Returns:
            A tuple ``(reordered_contexts, original_indices)`` where

            * **reordered_contexts** – contexts with documents reordered so
              that shared prefixes appear first, maximising cache hits.
            * **original_indices** – execution order mapping.
              ``reordered_contexts[i]`` corresponds to
              ``contexts[original_indices[i]]``.
        """
        # Accept a single list and wrap it
        if contexts and not isinstance(contexts[0], list):
            contexts = [contexts]

        result = self.build_incremental(contexts, initial_tokens_per_context)
        reordered = result["reordered_contexts"]

        # Register docs with conversation tracker for future deduplication
        cid = conversation_id or self._DEFAULT_CONVERSATION
        if conversation_id is not None:
            self._has_explicit_conversation = True
        elif self._has_explicit_conversation:
            warnings.warn(
                "This ContextPilot instance already has explicit "
                "conversation_id(s). Calling .reorder() without a "
                "conversation_id will use the shared '_default' "
                "conversation — other users' documents may appear as "
                "duplicates.  Pass an explicit conversation_id to "
                "avoid cross-contamination.",
                stacklevel=2,
            )
        conv = self._conversations.setdefault(
            cid, {"seen_docs": set(), "turn_count": 0}
        )
        for ctx in reordered:
            conv["seen_docs"].update(ctx)
        conv["turn_count"] += 1

        return reordered, result["original_indices"]

    def deduplicate(
        self,
        contexts: List[List],
        conversation_id: str,
        hint_template: Optional[str] = None,
    ) -> List[Dict]:
        """Remove already-seen documents from follow-up conversation turns.

        Lightweight method for Turn 2+ of a multi-turn conversation.
        Compares each context against the document history accumulated
        under ``conversation_id`` by earlier ``.reorder()`` or
        ``.deduplicate()`` calls.

        Example::

            engine = cp.ContextPilot(use_gpu=False)

            # Turn 1 — reorder and register docs
            reordered, indices = engine.reorder(
                turn1_docs, conversation_id="user_123"
            )

            # Turn 2 — deduplicate against Turn 1
            results = engine.deduplicate(
                turn2_docs, conversation_id="user_123"
            )
            for r in results:
                print(r["new_docs"])           # docs not seen before
                print(r["overlapping_docs"])   # already-sent docs
                print(r["reference_hints"])    # hints for the LLM

        Args:
            contexts: ``List[List[int]]`` or ``List[List[str]]`` — each
                inner list is one context (document IDs or text strings).
            conversation_id: **Required.**  Unique key identifying the
                conversation / user / session.  Must match the
                ``conversation_id`` used in a prior ``.reorder()`` call.
            hint_template: Custom hint template with ``{doc_id}``
                placeholder. Default:
                ``"Please refer to [Doc {doc_id}] from the previous conversation."``.

        Returns:
            A list of dicts (one per context) with keys:

            * ``new_docs`` – documents not seen in prior turns.
            * ``overlapping_docs`` – documents already sent.
            * ``reference_hints`` – hint strings for overlapping docs.
            * ``deduplicated_docs`` – alias for ``new_docs``.

        Raises:
            ValueError: If ``conversation_id`` is not provided or has
                no prior ``.reorder()`` history.
        """
        if not conversation_id:
            raise ValueError(
                "conversation_id is required for .deduplicate(). "
                "Pass the same conversation_id used in .reorder() to "
                "identify whose document history to compare against."
            )

        template = hint_template or "Please refer to [Doc {doc_id}] from the previous conversation."

        if conversation_id not in self._conversations:
            raise ValueError(
                f"No prior .reorder() call found for "
                f"conversation_id={conversation_id!r}. Call "
                f".reorder(contexts, conversation_id={conversation_id!r}) "
                f"first to register the initial documents."
            )
        conv = self._conversations[conversation_id]
        seen = conv["seen_docs"]

        results = []
        for ctx in contexts:
            overlapping = [d for d in ctx if d in seen]
            new = [d for d in ctx if d not in seen]
            hints = [template.format(doc_id=d) for d in overlapping]

            results.append({
                "new_docs": new,
                "overlapping_docs": overlapping,
                "reference_hints": hints,
                "deduplicated_docs": new,
            })

            # Register new docs so future turns see them
            seen.update(ctx)

        conv["turn_count"] += 1
        return results

    def build_incremental(self, contexts: List[List[int]], 
                          initial_tokens_per_context: int = 0) -> Dict:
        """
        Incrementally build/update index for a new batch of contexts.
        
        Algorithm:
        1. For each context, search existing index to find best matching node
        2. If match found (shared_prefix > 0): reorder context to align with matched path
        3. For contexts with no match: build a separate temporary index
        4. Merge: children of new temp index become children of global root
        5. Assign new request_ids to all new contexts
        
        Args:
            contexts: List of contexts (each is a list of document IDs)
            initial_tokens_per_context: Initial token count for new contexts
            
        Returns:
            Dictionary with:
                - request_ids: List of NEW request_ids in same order as input contexts
                - reordered_contexts: Contexts reordered for optimal cache reuse
                - original_indices: Optimized execution order
                - groups: Execution groups for cache efficiency
        """
        # Auto-convert string inputs to integer IDs
        contexts = self._convert_to_int(contexts)
        
        if not self.is_live:
            # First batch - do full build
            print("Index not live, performing full build...")
            result = self.build_and_schedule(contexts, initial_tokens_per_context)
            reordered = result.get('reordered_contexts', contexts)
            return {
                'request_ids': result.get('request_ids', []),
                'reordered_contexts': self._convert_to_str(reordered),
                'matched_count': 0,
                'inserted_count': len(contexts),
                'merged_count': 0,
                'original_indices': result.get('original_indices', list(range(len(contexts)))),
                'groups': result.get('groups', []),
            }
        
        print("=" * 80)
        print("INCREMENTAL BUILD - Search, Reorder, Build, Merge")
        print("=" * 80)
        
        matched_contexts = []  # (original_idx, reordered_context, search_path)
        unmatched_contexts = []  # (original_idx, context)
        
        print(f"\n1. Batch searching existing index for {len(contexts)} contexts...")
        
        # Use batch search for efficiency
        search_results = self.search_batch(contexts)
        
        for i, (context, (search_path, matched_node_id, overlap_count, has_prefix)) in enumerate(zip(contexts, search_results)):
            if overlap_count > 0 and matched_node_id >= 0 and matched_node_id != self.root_id:
                # Has a meaningful match (not global root) - reorder context to share docs with matched node
                matched_node = self.nodes.get(matched_node_id)
                node_docs = None
                if matched_node_id in self.metadata and self.metadata[matched_node_id].doc_ids:
                    node_docs = self.metadata[matched_node_id].doc_ids
                elif matched_node and hasattr(matched_node, 'doc_ids') and matched_node.doc_ids:
                    node_docs = matched_node.doc_ids
                
                if node_docs:
                    reordered = self._reorder_with_prefix(context, node_docs)
                    mode = "prefix→child" if has_prefix else "overlap→sibling"
                    print(f"   Context {i}: matched node {matched_node_id} ({mode}) overlap={overlap_count}/{len(context)}")
                    print(f"      Original:  {context[:8]}{'...' if len(context) > 8 else ''}")
                    print(f"      Reordered: {reordered[:8]}{'...' if len(reordered) > 8 else ''}")
                else:
                    reordered = context
                    has_prefix = True  # default to child if no docs available
                matched_contexts.append((i, reordered, search_path, has_prefix))
            else:
                # No match - will build new index for these
                unmatched_contexts.append((i, context))
        
        print(f"   ✓ Found {len(matched_contexts)} contexts with matches")
        print(f"   ✓ Found {len(unmatched_contexts)} contexts without matches")
        
        # Prepare result arrays (will fill in order)
        request_ids = [None] * len(contexts)
        reordered_contexts = [None] * len(contexts)
        context_info = []  # For scheduling: (original_idx, request_id, search_path)
        
        # Step 2: Insert matched contexts into existing tree
        # has_prefix=True  → split matched leaf, share prefix as internal node
        # has_prefix=False → sibling of matched node (insert at parent level)
        print(f"\n2. Inserting {len(matched_contexts)} matched contexts...")
        for orig_idx, reordered, search_path, has_prefix in matched_contexts:
            matched_node = self.traverse(search_path)
            if has_prefix and matched_node and matched_node.is_leaf:
                # Split the leaf: intersection becomes new internal node
                new_node_id, new_search_path, request_id = self._split_leaf_and_insert(
                    reordered, matched_node, search_path, initial_tokens_per_context
                )
            elif has_prefix:
                # Matched an internal node — insert as child
                new_node_id, new_search_path, request_id = self.insert(
                    reordered, search_path, initial_tokens_per_context
                )
            else:
                # No prefix match → sibling (insert at parent level)
                insert_path = search_path[:-1] if search_path else search_path
                new_node_id, new_search_path, request_id = self.insert(
                    reordered, insert_path, initial_tokens_per_context
                )
            request_ids[orig_idx] = request_id
            reordered_contexts[orig_idx] = reordered
            context_info.append((orig_idx, request_id, new_search_path))
        
        # Step 3: Build temporary index for unmatched contexts and merge
        merged_count = 0
        if unmatched_contexts:
            print(f"\n3. Building temporary index for {len(unmatched_contexts)} unmatched contexts...")
            
            # Extract just the contexts for building
            unmatched_only = [ctx for _, ctx in unmatched_contexts]
            
            # Build a temporary index (don't go live)
            temp_index = ContextPilot(
                alpha=self.alpha,
                use_gpu=self.use_gpu,
                linkage_method=self.linkage_method,
                batch_size=self.batch_size
            )
            temp_result = temp_index.fit_transform(unmatched_only)
            
            print(f"   ✓ Built temp index with {temp_result.stats['total_nodes']} nodes")
            
            # Step 4: Merge temp index into global index
            print("\n4. Merging temp index into global index...")
            merged_request_ids, merged_search_paths = self._merge_index(
                temp_index=temp_result,
                unmatched_info=unmatched_contexts,
                initial_tokens=initial_tokens_per_context
            )
            
            # Fill in request_ids and reordered_contexts for unmatched
            for i, (orig_idx, orig_context) in enumerate(unmatched_contexts):
                request_ids[orig_idx] = merged_request_ids[i]
                # Use reordered context from temp index
                if i < len(temp_result.reordered_contexts):
                    reordered_contexts[orig_idx] = temp_result.reordered_contexts[i]
                else:
                    reordered_contexts[orig_idx] = orig_context
                context_info.append((orig_idx, merged_request_ids[i], merged_search_paths[i]))
            
            merged_count = len(unmatched_contexts)
            print(f"   ✓ Merged {merged_count} new subtrees under global root")
        
        # Step 5: Schedule execution order
        print("\n5. Scheduling execution order for cache reuse...")
        scheduled_order = self._schedule_incremental(context_info)
        groups = self._group_by_path_prefix(context_info)
        print(f"   ✓ Scheduled {len(scheduled_order)} contexts into {len(groups)} groups")
        
        print("\n" + "=" * 80)
        print(f"✓ INCREMENTAL BUILD COMPLETE")
        print(f"   Matched & inserted: {len(matched_contexts)}")
        print(f"   Built & merged: {merged_count}")
        print("=" * 80 + "\n")
        
        return {
            'request_ids': request_ids,
            'reordered_contexts': self._convert_to_str(reordered_contexts),
            'matched_count': len(matched_contexts),
            'inserted_count': len(contexts),
            'merged_count': merged_count,
            'original_indices': scheduled_order,
            'groups': groups,
        }
    
    def _reorder_with_prefix(self, context: List[int], prefix: List[int]) -> List[int]:
        """
        Reorder context to start with the matched prefix.
        
        Example: context=[a,b,c], prefix=[c,a] -> result=[c,a,b]
        
        Args:
            context: Original context
            prefix: Prefix to match (from matched node)
            
        Returns:
            Reordered context starting with matched prefix elements
        """
        context_set = set(context)
        result = []
        
        # First, add elements from prefix that exist in context
        prefix_used = set()
        for elem in prefix:
            if elem in context_set and elem not in prefix_used:
                result.append(elem)
                prefix_used.add(elem)
        
        # Then, add remaining elements from context
        for elem in context:
            if elem not in prefix_used:
                result.append(elem)
        
        return result
    
    def _merge_index(self, temp_index, unmatched_info: List[Tuple], 
                     initial_tokens: int) -> Tuple[List[str], List[List[int]]]:
        """
        Merge a temporary index into the global index.
        
        The children of temp_index's root become children of global root.
        
        Args:
            temp_index: IndexResult from fit_transform on unmatched contexts
            unmatched_info: List of (original_idx, context) for unmatched contexts
            initial_tokens: Initial token count for new nodes
            
        Returns:
            Tuple of (request_ids, search_paths) for merged contexts
        """
        request_ids = []
        search_paths = []
        
        # Find temp root
        temp_root = None
        for node_id, node in temp_index.unique_nodes.items():
            if node.is_root:
                temp_root = node
                break
        
        if temp_root is None:
            # No tree built, insert contexts directly
            for orig_idx, context in unmatched_info:
                new_node_id, new_path, req_id = self.insert(context, [], initial_tokens)
                request_ids.append(req_id)
                search_paths.append(new_path)
            return request_ids, search_paths
        
        # Get global root
        global_root = self.nodes.get(self.root_id)
        if global_root is None:
            # No global root, insert directly
            for orig_idx, context in unmatched_info:
                new_node_id, new_path, req_id = self.insert(context, [], initial_tokens)
                request_ids.append(req_id)
                search_paths.append(new_path)
            return request_ids, search_paths
        
        # Map from temp node_id to new node_id in global index
        node_id_map = {}
        
        # Children of temp root become children of global root
        # We need to copy the entire subtree under each child
        base_child_idx = len(global_root.children)
        
        for child_idx, temp_child_id in enumerate(temp_root.children):
            new_child_idx = base_child_idx + child_idx
            self._copy_subtree(
                temp_index.unique_nodes,
                temp_child_id,
                self.root_id,
                node_id_map,
                initial_tokens,
                [new_child_idx]  # Base search path for this subtree
            )
        
        # Now map original contexts to their request_ids
        for i, (orig_idx, context) in enumerate(unmatched_info):
            # Find which leaf node this context belongs to
            temp_leaf_id = None
            for node_id, node in temp_index.unique_nodes.items():
                if node.is_leaf and i in node.original_indices:
                    temp_leaf_id = node_id
                    break
            
            if temp_leaf_id is not None and temp_leaf_id in node_id_map:
                new_node_id = node_id_map[temp_leaf_id]
                if new_node_id in self.metadata:
                    req_id = self.metadata[new_node_id].request_id
                    search_path = self.metadata[new_node_id].search_path
                    request_ids.append(req_id)
                    search_paths.append(search_path)
                    continue
            
            # Fallback: insert directly
            new_node_id, new_path, req_id = self.insert(context, [], initial_tokens)
            request_ids.append(req_id)
            search_paths.append(new_path)
        
        return request_ids, search_paths
    
    def _copy_subtree(self, source_nodes: Dict, source_node_id: int, 
                      parent_id: int, node_id_map: Dict, 
                      initial_tokens: int, search_path: List[int]):
        """
        Copy a subtree from source index into the global index.
        
        Args:
            source_nodes: Source index's unique_nodes
            source_node_id: Root of subtree to copy
            parent_id: Parent in global index
            node_id_map: Mapping from source node_id to new node_id
            initial_tokens: Initial token count for leaf nodes
            search_path: Search path to this node in global index
        """
        source_node = source_nodes.get(source_node_id)
        if source_node is None:
            return
        
        # Create new node in global index
        new_node_id = self.next_node_id
        self.next_node_id += 1
        
        # Copy node
        new_node = ClusterNode(
            node_id=new_node_id,
            content=source_node.doc_ids if hasattr(source_node, 'doc_ids') else [],
            children=[],
            parent=parent_id,
            original_indices=set(source_node.original_indices) if hasattr(source_node, 'original_indices') else set()
        )
        # ClusterNode.__init__ does doc_ids = sorted(content), losing the
        # reordered order from fit_transform.  Restore the source order.
        if hasattr(source_node, 'doc_ids') and source_node.doc_ids:
            new_node.doc_ids = list(source_node.doc_ids)
        
        self.nodes[new_node_id] = new_node
        node_id_map[source_node_id] = new_node_id
        
        # Add to parent's children
        parent_node = self.nodes.get(parent_id)
        if parent_node:
            parent_node.add_child(new_node_id)
        
        # Create metadata
        is_leaf = source_node.is_leaf
        request_id = f"req-{uuid.uuid4().hex[:12]}" if is_leaf else None
        
        parent_tokens = self.metadata[parent_id].total_tokens if parent_id in self.metadata else 0
        metadata = NodeMetadata(
            node_id=new_node_id,
            total_tokens=initial_tokens if is_leaf else 0,
            extra_tokens=max(0, initial_tokens - parent_tokens) if is_leaf else 0,
            search_path=search_path,
            doc_ids=source_node.doc_ids if hasattr(source_node, 'doc_ids') else None,
            is_leaf=is_leaf,
            request_id=request_id
        )
        self.metadata[new_node_id] = metadata
        
        if is_leaf and request_id:
            self._request_to_node[request_id] = new_node_id
        
        # Recursively copy children
        if source_node.children:
            for child_idx, child_id in enumerate(source_node.children):
                child_search_path = search_path + [child_idx]
                self._copy_subtree(
                    source_nodes, child_id, new_node_id, 
                    node_id_map, initial_tokens, child_search_path
                )
    
    def _schedule_incremental(self, context_info: List[Tuple]) -> List[int]:
        """
        Schedule contexts for optimal execution based on search paths.
        
        Groups contexts by path prefix and orders by path length descending
        to maximize cache reuse.
        
        Args:
            context_info: List of (context_idx, request_id, search_path)
            
        Returns:
            List of context indices in scheduled execution order
        """
        # Group by first element of search path
        from collections import defaultdict
        groups = defaultdict(list)
        
        for ctx_idx, req_id, path in context_info:
            if path:
                group_key = path[0]
            else:
                group_key = -1  # Empty path
            groups[group_key].append((ctx_idx, len(path)))
        
        # Sort within each group by path length descending (longer paths first for cache reuse)
        scheduled = []
        for group_key in sorted(groups.keys()):
            items = groups[group_key]
            items.sort(key=lambda x: -x[1])
            scheduled.extend([item[0] for item in items])
        
        return scheduled
    
    def _group_by_path_prefix(self, context_info: List[Tuple]) -> List[Tuple[int, List[int]]]:
        """
        Group contexts by search path prefix for execution.
        
        Args:
            context_info: List of (context_idx, request_id, search_path)
            
        Returns:
            List of (group_score, [context_indices])
        """
        from collections import defaultdict
        groups = defaultdict(list)
        
        for ctx_idx, req_id, path in context_info:
            if path:
                group_key = path[0]
            else:
                group_key = -1
            groups[group_key].append(ctx_idx)
        
        # Convert to list of (score, indices) tuples
        result = []
        for group_key, indices in groups.items():
            score = len(indices)  # Simple score based on group size
            result.append((score, indices))
        
        # Sort by score descending
        result.sort(key=lambda x: -x[0])
        return result

    def schedule_only(self, contexts: List[List[int]]) -> Dict:
        """
        Schedule contexts for optimal execution (STATELESS MODE).
        
        This performs clustering and scheduling WITHOUT initializing live metadata.
        Use this for batch processing where you don't need cache tracking.
        
        Workflow:
        1. fit_transform() - build tree and reorder contexts
        2. Inter-context scheduling - optimize execution order
        3. Return results WITHOUT going live
        
        Args:
            contexts: List of contexts (each is a list of document IDs)
            
        Returns:
            Dictionary with scheduled results (no request_id mapping)
        """
        print("=" * 80)
        print("SCHEDULING BATCH (STATELESS MODE)")
        print("=" * 80)
        
        # Step 1: Build static index (clustering + reordering)
        print("\n1. Building static index...")
        result = self.fit_transform(contexts)
        
        print(f"   ✓ Built tree with {result.stats['total_nodes']} nodes")
        print(f"   ✓ Leaf nodes: {result.stats['leaf_nodes']}")
        
        # Step 2: Inter-context scheduling
        print("\n2. Scheduling contexts for optimal execution...")
        scheduled_reordered, scheduled_originals, final_mapping, groups = \
            self.inter_scheduler.schedule_contexts(result)
        
        print(f"   ✓ Created {len(groups)} execution groups")
        
        # Return results without going live (stateless)
        scheduled_result = {
            'reordered_contexts': scheduled_reordered,
            'original_indices': final_mapping,
            'scheduled_originals': scheduled_originals,
            'groups': groups,
            'stats': {
                'total_nodes': result.stats['total_nodes'],
                'leaf_nodes': result.stats['leaf_nodes'],
                'num_contexts': len(contexts),
                'num_groups': len(groups),
            }
        }
        
        print("\n" + "=" * 80)
        print("✓ BATCH SCHEDULED (Stateless - no cache tracking)")
        print("=" * 80 + "\n")
        
        return scheduled_result
    
    def _initialize_live_metadata(self, initial_tokens_per_context: int, num_input_contexts: int = None) -> Tuple[Dict[str, int], List[str]]:
        """
        Initialize metadata for all nodes after static index is built.
        
        Auto-generates request_id for each leaf node during construction.
        Returns both a mapping dict and an ordered list of request_ids.
        
        Args:
            initial_tokens_per_context: Initial token count for each context
            num_input_contexts: Number of input contexts (for generating request_ids list)
            
        Returns:
            Tuple of:
                - Dictionary mapping request_id -> node_id for all leaf nodes
                - List of request_ids in same order as input contexts
        """
        if not self.initial_result:
            raise RuntimeError("Must call fit_transform() before initializing metadata")
        
        unique_nodes = self.initial_result.unique_nodes
        # Reordered contexts preserve the actual order used for cache prefix sharing.
        # node.doc_ids is sorted (from ClusterNode.__init__) and loses ordering info.
        reordered_contexts = self.initial_result.reordered_contexts
        request_id_mapping = {}  # request_id -> node_id
        
        # Set up node aliases
        self.nodes = unique_nodes
        
        # Find root
        for node_id, node in unique_nodes.items():
            if hasattr(node, 'is_root') and node.is_root:
                self.root_id = node_id
                break
        
        # Set next node ID
        self.next_node_id = max(unique_nodes.keys()) + 1 if unique_nodes else 0
        
        # Counter for auto-generating request_ids
        leaf_counter = 0
        
        # Track original context index -> request_id mapping
        original_index_to_request_id = {}
        
        # Initialize metadata for all nodes
        for node_id, node in unique_nodes.items():
            search_path = self._compute_search_path(node_id)
            
            # Determine if this is a leaf node
            is_leaf = hasattr(node, 'is_leaf') and node.is_leaf
            
            # Compute token counts (only leaf nodes have initial tokens)
            if is_leaf:
                total_tokens = initial_tokens_per_context
                # Auto-generate request_id for leaf nodes using UUID
                request_id = f"req-{uuid.uuid4().hex[:12]}"
                leaf_counter += 1
                
                # Track which original context index this leaf represents
                if hasattr(node, 'original_indices') and node.original_indices:
                    for orig_idx in node.original_indices:
                        original_index_to_request_id[orig_idx] = request_id
            else:
                # Internal node: no direct tokens, no request_id
                total_tokens = 0
                request_id = None
            
            # Compute extra tokens (beyond parent)
            parent_tokens = 0
            if node.parent is not None and node.parent in self.metadata:
                parent_tokens = self.metadata[node.parent].total_tokens
            extra_tokens = max(0, total_tokens - parent_tokens)
            
            # Create metadata with auto-generated request_id for leaf nodes
            # For leaf nodes, use the reordered context order (not node.doc_ids which is sorted).
            # node.doc_ids comes from ClusterNode.__init__ which does sorted(content),
            # losing the intra-context reordering that maximizes prefix sharing.
            if is_leaf and hasattr(node, 'original_indices') and node.original_indices:
                first_orig_idx = min(node.original_indices)
                if reordered_contexts and first_orig_idx < len(reordered_contexts):
                    leaf_doc_ids = reordered_contexts[first_orig_idx]
                else:
                    leaf_doc_ids = node.doc_ids if hasattr(node, 'doc_ids') else None
            else:
                leaf_doc_ids = node.doc_ids if hasattr(node, 'doc_ids') else None
            
            metadata = NodeMetadata(
                node_id=node_id,
                total_tokens=total_tokens,
                extra_tokens=extra_tokens,
                search_path=search_path,
                is_leaf=is_leaf,
                doc_ids=leaf_doc_ids,
                request_id=request_id
            )
            
            self.metadata[node_id] = metadata
            
            # Track leaf nodes
            if is_leaf and request_id:
                self._request_to_node[request_id] = node_id
                request_id_mapping[request_id] = node_id
        
        self.next_node_id = max(self.nodes.keys()) + 1 if self.nodes else 0
        self._next_request_counter = leaf_counter  # Track for future inserts
        
        # Build ordered list of request_ids matching INPUT context order
        # Use num_input_contexts if provided, otherwise use original_index_to_request_id length
        if num_input_contexts is not None:
            num_contexts = num_input_contexts
        else:
            num_contexts = len(original_index_to_request_id)
        
        request_ids_ordered = [
            original_index_to_request_id.get(i) for i in range(num_contexts)
        ]
        
        return request_id_mapping, request_ids_ordered
    
    # =========================================================================
    # Request Eviction (Called by inference engine's eviction callback)
    # =========================================================================
    
    def remove_requests(self, request_ids: Set[str]) -> Dict[str, Any]:
        """
        Remove requests from the context index.

        Called by the inference engine's eviction callback (SGLang or vLLM).

        When the engine's cache evicts requests, it calls a callback
        with the set of evicted request_ids. That callback invokes
        this method to keep the context index in sync.

        Args:
            request_ids: Set of request IDs to remove (from engine callback)
            
        Returns:
            Dictionary with eviction results
        """
        evicted_nodes = []
        not_found = []
        
        for request_id in request_ids:
            node_id = self._request_to_node.get(request_id)
            if node_id is None:
                not_found.append(request_id)
                continue
            
            # Remove from tracking
            del self._request_to_node[request_id]
            evicted_nodes.append(node_id)
            
            # Remove node and prune empty parents
            self._remove_node_and_prune(node_id)
        
        self.live_stats['total_evictions'] += len(evicted_nodes)
        
        logger.info(
            f"Removed {len(evicted_nodes)} requests from context index, "
            f"{len(not_found)} not found"
        )
        
        return {
            'removed_count': len(evicted_nodes),
            'evicted_node_ids': evicted_nodes,
            'evicted_request_ids': list(set(request_ids) - set(not_found)),
            'not_found': not_found,
            'nodes_remaining': len(self.nodes),
            'requests_remaining': len(self._request_to_node)
        }
    
    def remove_request_by_id(self, request_id: str) -> bool:
        """
        Remove a single request from the context index.
        
        Convenience method for removing one request at a time.
        
        Args:
            request_id: The request ID to remove
            
        Returns:
            True if request was found and removed, False otherwise
        """
        result = self.remove_requests({request_id})
        return len(result['evicted_node_ids']) > 0
    
    def get_request_node(self, request_id: str) -> Optional[int]:
        """
        Get the node_id for a request.
        
        Args:
            request_id: The unique request identifier
            
        Returns:
            node_id if found, None otherwise
        """
        return self._request_to_node.get(request_id)
    
    def _collect_all_node_docs(self) -> Tuple[List[int], List[List[int]], Dict[int, List[int]]]:
        """
        Collect doc_ids from all nodes in the tree.
        
        Returns:
            Tuple of (node_ids, node_docs_list, node_id_to_path)
        """
        node_ids = []
        node_docs_list = []
        node_id_to_path = {}
        
        # BFS to collect all nodes
        queue = deque([(self.root_id, [])])
        
        while queue:
            node_id, path = queue.popleft()
            
            if node_id not in self.nodes:
                continue
            
            node = self.nodes[node_id]
            node_meta = self.metadata.get(node_id)
            
            if node_meta and node_meta.doc_ids:
                docs = node_meta.doc_ids
            elif hasattr(node, 'doc_ids') and node.doc_ids:
                docs = node.doc_ids
            else:
                docs = None
            
            if docs:
                node_ids.append(node_id)
                node_docs_list.append(docs)
                node_id_to_path[node_id] = path
            
            # Add children to queue
            if not node.is_leaf and node.children:
                for idx, child_id in enumerate(node.children):
                    queue.append((child_id, path + [idx]))
        
        return node_ids, node_docs_list, node_id_to_path
    
    def _get_node_docs(self, node_id: int) -> Optional[List[int]]:
        """Get doc_ids for a node from metadata or node attributes."""
        meta = self.metadata.get(node_id)
        if meta and meta.doc_ids:
            return meta.doc_ids
        node = self.nodes.get(node_id)
        if node and hasattr(node, 'doc_ids') and node.doc_ids:
            return node.doc_ids
        return None

    def _search_single_hierarchical(self, context: List[int]) -> Tuple[List[int], int, int, bool]:
        """
        Hierarchical top-down search for a single context.
        
        Algorithm:
        1. Start at root's children
        2. At each level, find the best matching child (overlap > 0, min distance)
        3. If best child's docs[0] is in the query (has_prefix):
           → descend into that child's children and repeat
        4. If best child has overlap but docs[0] NOT in query (no prefix):
           → stop, return this child as a sibling match (has_prefix=False)
        5. If no children have overlap → no match
        
        Returns:
            (search_path, matched_node_id, overlap_count, has_prefix)
        """
        context_set = set(context)
        current_id = self.root_id
        current_path: List[int] = []
        
        while True:
            current_node = self.nodes.get(current_id)
            if current_node is None or current_node.is_leaf or not current_node.children:
                # Reached a leaf or node with no children — return current as match
                docs = self._get_node_docs(current_id)
                if docs and current_id != self.root_id:
                    overlap = len(context_set & set(docs))
                    has_prefix = docs[0] in context_set if overlap > 0 else False
                    return (current_path, current_id, overlap, has_prefix)
                return ([], -1, 0, False)
            
            # Collect children info for distance computation
            child_ids = []
            child_docs_list = []
            child_indices = []  # index within parent.children
            
            for idx, child_id in enumerate(current_node.children):
                docs = self._get_node_docs(child_id)
                if docs:
                    child_ids.append(child_id)
                    child_docs_list.append(docs)
                    child_indices.append(idx)
            
            if not child_ids:
                return ([], -1, 0, False)
            
            # Compute distances from this context to all children at this level
            distances = compute_distances_batch([context], child_docs_list, self.alpha)
            
            # Find best matching child with overlap > 0
            best_j = -1
            best_distance = float('inf')
            best_overlap = 0
            
            for j, (child_id, docs) in enumerate(zip(child_ids, child_docs_list)):
                overlap = len(context_set & set(docs))
                if overlap == 0:
                    continue
                dist = distances[0, j]
                if dist < best_distance:
                    best_distance = dist
                    best_overlap = overlap
                    best_j = j
            
            if best_j < 0:
                # No child has overlap at this level
                if current_id != self.root_id:
                    # We descended here because this node had prefix match.
                    # Return it as the match — insert as child of this node.
                    docs = self._get_node_docs(current_id)
                    if docs:
                        overlap = len(context_set & set(docs))
                        return (current_path, current_id, overlap, True)
                return ([], -1, 0, False)
            
            best_child_id = child_ids[best_j]
            best_child_idx = child_indices[best_j]
            best_docs = child_docs_list[best_j]
            child_path = current_path + [best_child_idx]
            
            if best_docs[0] in context_set:
                # has_prefix=True: descend into this child
                best_child_node = self.nodes.get(best_child_id)
                if best_child_node and not best_child_node.is_leaf and best_child_node.children:
                    # Has children to explore — go deeper
                    current_id = best_child_id
                    current_path = child_path
                    continue
                else:
                    # Leaf or no children — this is the match
                    return (child_path, best_child_id, best_overlap, True)
            else:
                # has_prefix=False: stop here, insert as sibling
                return (child_path, best_child_id, best_overlap, False)

    def search_batch(self, contexts: List[List[int]]) -> List[Tuple[List[int], int, int, bool]]:
        """
        Batch search using hierarchical top-down traversal.
        
        For each context, traverses the tree from root downward:
        - At each level, finds the best matching child (overlap > 0, min distance)
        - If the child's first doc is in the query (has_prefix=True), descends deeper
        - If overlap exists but first doc is NOT in query, stops (sibling match)
        - If no overlap at a level, returns no match
        
        Args:
            contexts: List of query contexts
            
        Returns:
            List of (search_path, matched_node_id, overlap_count, has_prefix)
            for each context.  ``has_prefix`` is True when the matched node's
            first doc exists in the query (KV-cache prefix reuse possible).
        """
        start_time = time.perf_counter()
        
        if self.root_id is None or not contexts:
            return [([], -1, 0, False) for _ in contexts]
        
        results = [self._search_single_hierarchical(ctx) for ctx in contexts]
        
        # Update statistics
        elapsed_us = (time.perf_counter() - start_time) * 1_000_000
        self.live_stats['total_searches'] += len(contexts)
        self.live_stats['total_search_time_us'] += elapsed_us
        
        return results
    
    def search(self, context: List[int], update_access: bool = True) -> Tuple[List[int], int, int, bool]:
        """
        Search for best matching node for a single context.
        For multiple contexts, use search_batch() instead.
        
        Args:
            context: Query context (list of document IDs)
            update_access: Whether to update LRU timestamp
            
        Returns:
            Tuple of (search_path, matched_node_id, overlap_count, has_prefix)
        """
        results = self.search_batch([context])
        search_path, node_id, overlap, has_prefix = results[0]
        
        # Update access time
        if update_access and node_id >= 0 and node_id in self.metadata:
            self.metadata[node_id].update_access_time()
        
        return (search_path, node_id, overlap, has_prefix)
    
    def traverse(self, search_path: List[int]) -> Optional[ClusterNode]:
        """
        Traverse to a node using its search path.
        
        Complexity: O(h) where h is tree height
        
        Args:
            search_path: List of child indices from root
            
        Returns:
            ClusterNode at the end of the path, or None if invalid
        """
        start_time = time.perf_counter()
        
        if self.root_id is None:
            return None
        
        current_id = self.root_id
        
        for child_idx in search_path:
            if current_id not in self.nodes:
                return None
            
            current_node = self.nodes[current_id]
            
            if not current_node.children or child_idx >= len(current_node.children):
                return None
            
            current_id = current_node.children[child_idx]
        
        # Update statistics
        elapsed_us = (time.perf_counter() - start_time) * 1_000_000
        self.live_stats['total_traversal_time_us'] += elapsed_us
        
        return self.nodes.get(current_id)
    
    def insert(self, context: List[int], search_path: List[int], 
               total_tokens: int = 0) -> Tuple[int, List[int], str]:
        """
        Insert a new context into the index.
        
        Two cases:
        1. Matched internal node: Append as child - O(1)
        2. Matched leaf: Insert as sibling (child of leaf's parent) - O(1)
        
        Auto-generates a unique request_id for the new leaf node.
        
        Args:
            context: New context to insert
            search_path: Search path from search() operation
            total_tokens: Initial token count
            
        Returns:
            Tuple of (new_node_id, new_search_path, request_id)
        """
        start_time = time.perf_counter()
        
        # Find matched node
        matched_node = self.traverse(search_path)
        
        if matched_node is None:
            # Invalid path, insert at root
            matched_node = self.nodes[self.root_id]
            search_path = []
        
        matched_id = matched_node.node_id
        
        if matched_node.is_leaf:
            # Case 2: Matched a leaf, insert as sibling (child of leaf's parent)
            new_node_id, new_search_path, request_id = self._insert_at_leaf(
                context, matched_node, search_path, total_tokens
            )
        else:
            # Case 1: Matched internal node, append as child
            new_node_id, new_search_path, request_id = self._insert_at_internal(
                context, matched_node, search_path, total_tokens
            )
        
        # Update statistics
        elapsed_us = (time.perf_counter() - start_time) * 1_000_000
        self.live_stats['total_insertions'] += 1
        
        return (new_node_id, new_search_path, request_id)
    
    def _insert_at_internal(self, context: List[int], parent_node: ClusterNode,
                           search_path: List[int], total_tokens: int) -> Tuple[int, List[int], str]:
        """Insert new context as child of internal node."""
        # Auto-generate request_id using UUID
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        
        # Create new leaf node
        new_node = ClusterNode(
            node_id=self.next_node_id,
            content=context,
            children=[],
            parent=parent_node.node_id,
            original_indices={self.next_node_id}
        )
        
        self.nodes[self.next_node_id] = new_node
        parent_node.add_child(self.next_node_id)
        
        # Create metadata with auto-generated request_id
        parent_tokens = self.metadata[parent_node.node_id].total_tokens
        metadata = NodeMetadata(
            node_id=self.next_node_id,
            total_tokens=total_tokens,
            extra_tokens=max(0, total_tokens - parent_tokens),
            search_path=search_path + [len(parent_node.children) - 1],
            doc_ids=context,
            is_leaf=True,
            request_id=request_id
        )
        
        self.metadata[self.next_node_id] = metadata
        self._request_to_node[request_id] = self.next_node_id
        
        new_search_path = search_path + [len(parent_node.children) - 1]
        new_node_id = self.next_node_id
        self.next_node_id += 1
        
        return (new_node_id, new_search_path, request_id)
    
    def _insert_at_leaf(self, context: List[int], leaf_node: ClusterNode,
                       search_path: List[int], total_tokens: int) -> Tuple[int, List[int], str]:
        """
        Insert new context as sibling of the matched leaf node.
        
        Instead of creating a new internal node, we simply insert the new context
        as another child of the leaf's parent node.
        """
        # Auto-generate request_id using UUID
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        
        # Get parent node
        if leaf_node.parent is None:
            # Leaf is directly under root, use root as parent
            parent_node = self.nodes[self.root_id]
            parent_search_path = []
        else:
            parent_node = self.nodes[leaf_node.parent]
            # Parent's search path is the leaf's search path without the last element
            parent_search_path = search_path[:-1] if search_path else []
        
        # Create new leaf node as sibling of matched leaf
        new_leaf = ClusterNode(
            node_id=self.next_node_id,
            content=context,
            children=[],
            parent=parent_node.node_id,
            original_indices={self.next_node_id}
        )
        
        self.nodes[self.next_node_id] = new_leaf
        parent_node.add_child(self.next_node_id)
        new_leaf_id = self.next_node_id
        self.next_node_id += 1
        
        # New search path: parent's path + index of new child
        new_search_path = parent_search_path + [len(parent_node.children) - 1]
        
        # Create metadata for new leaf with auto-generated request_id
        parent_tokens = self.metadata[parent_node.node_id].total_tokens if parent_node.node_id in self.metadata else 0
        new_metadata = NodeMetadata(
            node_id=new_leaf_id,
            total_tokens=total_tokens,
            extra_tokens=max(0, total_tokens - parent_tokens),
            search_path=new_search_path,
            doc_ids=context,
            is_leaf=True,
            request_id=request_id
        )
        self.metadata[new_leaf_id] = new_metadata
        self._request_to_node[request_id] = new_leaf_id
        
        return (new_leaf_id, new_search_path, request_id)
    
    def _split_leaf_and_insert(self, context: List[int], leaf_node: ClusterNode,
                               search_path: List[int], total_tokens: int) -> Tuple[int, List[int], str]:
        """
        Split a matched leaf node and insert a new context.
        
        Creates a new internal node whose doc_ids is the contiguous shared
        prefix between the existing leaf and the new context.  Both the
        original leaf and the new leaf become children of this internal node.
        
        Before:
            parent → leaf_A [a,b,c,d,e]
        
        After:
            parent → new_internal [a,b,c]          (shared prefix)
                      ├── leaf_A [a,b,c,d,e]       (original)
                      └── leaf_B [a,b,c,x,y]       (new)
        
        Falls back to sibling insertion if no contiguous shared prefix.
        
        Args:
            context: Reordered new context (already starts with matched prefix)
            leaf_node: The matched leaf node to split
            search_path: Search path to the matched leaf
            total_tokens: Initial token count for the new context
            
        Returns:
            Tuple of (new_leaf_id, new_search_path, request_id)
        """
        matched_docs = self._get_node_docs(leaf_node.node_id)
        
        if not matched_docs:
            return self._insert_at_leaf(context, leaf_node, search_path, total_tokens)
        
        # Compute contiguous shared prefix
        shared_prefix = []
        for a, b in zip(matched_docs, context):
            if a == b:
                shared_prefix.append(a)
            else:
                break
        
        if not shared_prefix:
            # No contiguous prefix — just insert as sibling
            return self._insert_at_leaf(context, leaf_node, search_path, total_tokens)
        
        if shared_prefix == list(matched_docs) and set(matched_docs) == set(context):
            # Identical docs — just insert as sibling
            return self._insert_at_leaf(context, leaf_node, search_path, total_tokens)
        
        # --- Tree restructuring: split the leaf ---
        
        # 1. Identify parent
        parent_id = leaf_node.parent
        if parent_id is None:
            parent_id = self.root_id
        parent_node = self.nodes[parent_id]
        parent_search_path = search_path[:-1] if search_path else []
        
        # Find leaf's index in parent.children
        leaf_child_idx = parent_node.children.index(leaf_node.node_id)
        
        # 2. Create new internal node with shared prefix as doc_ids
        new_internal_id = self.next_node_id
        self.next_node_id += 1
        
        all_content = set(leaf_node.content) | set(context)
        new_internal = ClusterNode(
            node_id=new_internal_id,
            content=all_content,
            children=[leaf_node.node_id],  # original leaf as first child
            parent=parent_id,
            original_indices=set()
        )
        # Override sorted doc_ids with the actual shared prefix
        new_internal.doc_ids = list(shared_prefix)
        
        self.nodes[new_internal_id] = new_internal
        
        # 3. Replace leaf in parent's children list with new internal node
        parent_node.children[leaf_child_idx] = new_internal_id
        
        # 4. Update original leaf's parent pointer
        leaf_node.parent = new_internal_id
        
        # 5. Create metadata for new internal node
        parent_tokens = self.metadata[parent_id].total_tokens if parent_id in self.metadata else 0
        leaf_meta = self.metadata.get(leaf_node.node_id)
        leaf_total = leaf_meta.total_tokens if leaf_meta else 0
        
        # Estimate internal node tokens proportionally from shared prefix
        if matched_docs and len(matched_docs) > 0:
            prefix_ratio = len(shared_prefix) / len(matched_docs)
            internal_tokens = int(parent_tokens + (leaf_total - parent_tokens) * prefix_ratio)
        else:
            internal_tokens = parent_tokens
        
        internal_path = parent_search_path + [leaf_child_idx]
        
        internal_meta = NodeMetadata(
            node_id=new_internal_id,
            total_tokens=internal_tokens,
            extra_tokens=max(0, internal_tokens - parent_tokens),
            search_path=internal_path,
            doc_ids=list(shared_prefix),
            is_leaf=False,
            request_id=None
        )
        self.metadata[new_internal_id] = internal_meta
        
        # 6. Update original leaf's metadata (now child 0 of new internal)
        if leaf_meta:
            leaf_meta.extra_tokens = max(0, leaf_total - internal_tokens)
            leaf_meta.search_path = internal_path + [0]
        
        # 7. Create new leaf for the inserted context
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        
        new_leaf_id = self.next_node_id
        self.next_node_id += 1
        
        new_leaf = ClusterNode(
            node_id=new_leaf_id,
            content=context,
            children=[],
            parent=new_internal_id,
            original_indices={new_leaf_id}
        )
        new_leaf.doc_ids = list(context)  # Preserve insertion order
        
        self.nodes[new_leaf_id] = new_leaf
        new_internal.add_child(new_leaf_id)  # becomes child 1
        
        new_leaf_path = internal_path + [1]
        
        new_leaf_meta = NodeMetadata(
            node_id=new_leaf_id,
            total_tokens=total_tokens,
            extra_tokens=max(0, total_tokens - internal_tokens),
            search_path=new_leaf_path,
            doc_ids=list(context),
            is_leaf=True,
            request_id=request_id
        )
        self.metadata[new_leaf_id] = new_leaf_meta
        self._request_to_node[request_id] = new_leaf_id
        
        print(f"      Split: leaf {leaf_node.node_id} → internal {new_internal_id} "
              f"(prefix={len(shared_prefix)}) + leaf {new_leaf_id}")
        
        return (new_leaf_id, new_leaf_path, request_id)

    def update_node(self, search_path: List[int], token_delta: int) -> bool:
        """
        Update a node's token count.
        
        Args:
            search_path: Path to the node
            token_delta: Tokens to add (positive) or remove (negative)
            
        Returns:
            True if update successful, False otherwise
        """
        node = self.traverse(search_path)
        
        if node is None or node.node_id not in self.metadata:
            return False
        
        metadata = self.metadata[node.node_id]
        
        if token_delta > 0:
            metadata.add_tokens(token_delta)
        else:
            metadata.remove_tokens(abs(token_delta))
        
        return True
    
    def _remove_node(self, node_id: int):
        """
        Remove a node and recursively delete empty parents.
        
        Args:
            node_id: Node to remove
        """
        self._remove_node_and_prune(node_id)
    
    def _remove_node_and_prune(self, node_id: int) -> int:
        """
        Remove a node and recursively delete empty parents.
        
        This mirrors SGLang's radix cache eviction behavior:
        - When a leaf is evicted, SGLang checks if the parent becomes childless
        - If parent has no children and lock_ref==0, it's pushed to eviction heap
        - Parent is then evicted in the same evict() cycle
        
        So when we receive notification that a request was evicted, we should:
        1. Remove that request's node
        2. Prune any parent nodes that become childless
        
        This keeps the context index in sync with SGLang's actual cache state.
        
        Args:
            node_id: Node to remove
            
        Returns:
            Number of additional nodes pruned (parent nodes that became empty)
        """
        if node_id not in self.nodes:
            return 0
        
        nodes_pruned = 0
        node = self.nodes[node_id]
        parent_id = node.parent
        
        # Remove from parent's child list
        if parent_id and parent_id in self.nodes:
            parent = self.nodes[parent_id]
            if node_id in parent.children:
                parent.children.remove(node_id)
            
            # Recursively remove empty parents (not root)
            # This mirrors SGLang behavior: when a leaf is evicted and parent
            # becomes childless, SGLang evicts the parent too in the same cycle
            if not parent.children and not parent.is_root:
                nodes_pruned += 1
                nodes_pruned += self._remove_node_and_prune(parent_id)
        
        # Delete node
        del self.nodes[node_id]
        
        if node_id in self.metadata:
            del self.metadata[node_id]
        
        return nodes_pruned
    
    def _compute_search_path(self, node_id: int) -> List[int]:
        """Compute search path from root to node."""
        if node_id == self.root_id:
            return []
        
        path = []
        current_id = node_id
        visited = set()
        
        while current_id != self.root_id and current_id is not None:
            if current_id in visited:
                break
            visited.add(current_id)
            
            node = self.nodes.get(current_id)
            if node is None or node.parent is None:
                break
            
            parent = self.nodes.get(node.parent)
            if parent is None:
                break
            
            try:
                child_idx = parent.children.index(current_id)
                path.append(child_idx)
            except (ValueError, AttributeError):
                break
            
            current_id = node.parent
        
        return path[::-1]
    
    def _find_common_prefix(self, list1: List[int], list2: List[int]) -> List[int]:
        """Find common prefix of two lists."""
        prefix = []
        for a, b in zip(list1, list2):
            if a == b:
                prefix.append(a)
            else:
                break
        return prefix
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        avg_search_time = (self.live_stats['total_search_time_us'] / self.live_stats['total_searches'] 
                          if self.live_stats['total_searches'] > 0 else 0)
        
        # Calculate total extra tokens from metadata
        total_tokens = sum(m.extra_tokens for m in self.metadata.values())
        
        return {
            'num_nodes': len(self.nodes),
            'active_nodes': len(self.metadata),
            'total_tokens': total_tokens,
            'num_requests': len(self._request_to_node),
            'total_searches': self.live_stats['total_searches'],
            'total_insertions': self.live_stats['total_insertions'],
            'total_removals': self.live_stats.get('total_removals', 0),
            'avg_search_time_us': avg_search_time,
        }

