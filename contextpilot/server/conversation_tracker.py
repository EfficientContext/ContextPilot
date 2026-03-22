"""
Conversation Tracker for Multi-Turn Context Deduplication

Tracks document history across conversation turns and provides deduplication
to avoid sending the same documents multiple times in a conversation.

Usage:
    tracker = ConversationTracker()

    # Turn 1
    req_a_id = tracker.register_request(docs=[4, 3, 1])

    # Turn 2 (continuation of Turn 1)
    result = tracker.deduplicate(
        request_id=req_b_id,
        parent_request_id=req_a_id,
        docs=[4, 3, 2]
    )
    # result.new_docs = [2]
    # result.overlapping_docs = [4, 3]
    # result.reference_hints = ["Refer to Doc 4...", "Refer to Doc 3..."]
"""

import time
import logging
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    original_docs: List[int]
    overlapping_docs: List[int]
    new_docs: List[int]
    reference_hints: List[str]
    deduplicated_docs: List[int]
    doc_source_turns: Dict[int, str] = field(default_factory=dict)
    is_new_conversation: bool = False
    blocks_deduped: int = 0
    blocks_total: int = 0
    block_chars_saved: int = 0


@dataclass
class RequestHistory:
    """History of a single request."""

    request_id: str
    docs: List[int]
    parent_request_id: Optional[str] = None
    turn_number: int = 1
    timestamp: float = field(default_factory=time.time)


class ConversationTracker:
    """
    Tracks conversation history for multi-turn context deduplication.

    Features:
    - Track documents sent per request
    - Track conversation chains (parent-child relationships)
    - Deduplicate contexts by removing already-seen documents
    - Generate reference hints for deduplicated documents
    """

    def __init__(self, hint_template: str = None):
        """
        Initialize the tracker.

        Args:
            hint_template: Template for reference hints.
                           Default: "Please refer to [Doc {doc_id}] from turn {turn_number}."
        """
        # request_id -> RequestHistory
        self._requests: Dict[str, RequestHistory] = {}

        # Template for generating reference hints
        self._hint_template = (
            hint_template
            or "Please refer to [Doc {doc_id}] from the previous conversation turn."
        )

        # Statistics
        self._stats = {
            "total_requests": 0,
            "total_dedup_calls": 0,
            "total_docs_deduplicated": 0,
        }

    def register_request(
        self, request_id: str, docs: List[int], parent_request_id: Optional[str] = None
    ) -> RequestHistory:
        """
        Register a request and its documents.

        Args:
            request_id: Unique identifier for this request
            docs: List of document IDs sent in this request
            parent_request_id: ID of the previous turn's request (if multi-turn)

        Returns:
            RequestHistory object
        """
        # Determine turn number
        turn_number = 1
        if parent_request_id and parent_request_id in self._requests:
            turn_number = self._requests[parent_request_id].turn_number + 1

        history = RequestHistory(
            request_id=request_id,
            docs=list(docs),
            parent_request_id=parent_request_id,
            turn_number=turn_number,
        )

        self._requests[request_id] = history
        self._stats["total_requests"] += 1

        logger.debug(
            f"Registered request {request_id}: {len(docs)} docs, turn {turn_number}"
        )

        return history

    def get_conversation_chain(self, request_id: str) -> List[RequestHistory]:
        """
        Get the full conversation chain leading to this request.

        Args:
            request_id: The current request ID

        Returns:
            List of RequestHistory objects from first turn to current, in order
        """
        chain = []
        current_id = request_id

        while current_id and current_id in self._requests:
            chain.append(self._requests[current_id])
            current_id = self._requests[current_id].parent_request_id

        # Reverse to get chronological order
        chain.reverse()
        return chain

    def get_all_previous_docs(
        self, parent_request_id: str
    ) -> Tuple[Set[int], Dict[int, str]]:
        """
        Get all documents from previous turns in the conversation.

        Args:
            parent_request_id: The parent request ID

        Returns:
            Tuple of (set of all doc IDs, dict mapping doc_id to request_id where it first appeared)
        """
        all_docs = set()
        doc_sources = {}  # doc_id -> request_id where it first appeared

        chain = self.get_conversation_chain(parent_request_id)

        for history in chain:
            for doc_id in history.docs:
                if doc_id not in all_docs:
                    all_docs.add(doc_id)
                    doc_sources[doc_id] = history.request_id

        return all_docs, doc_sources

    def deduplicate(
        self,
        request_id: str,
        docs: Optional[List[int]] = None,
        parent_request_id: Optional[str] = None,
        hint_template: Optional[str] = None,
        doc_contents: Optional[Dict[int, str]] = None,
    ) -> DeduplicationResult:
        if docs is None and doc_contents is not None:
            docs = list(doc_contents.keys())
        elif docs is None:
            docs = []
        self._stats["total_dedup_calls"] += 1

        if not parent_request_id or parent_request_id not in self._requests:
            self.register_request(request_id, docs, parent_request_id=None)
            result = DeduplicationResult(
                original_docs=docs,
                overlapping_docs=[],
                new_docs=docs,
                reference_hints=[],
                deduplicated_docs=docs,
                doc_source_turns={},
                is_new_conversation=True,
            )
            if doc_contents:
                self._apply_block_dedup(doc_contents, result)
            return result

        previous_docs, doc_sources = self.get_all_previous_docs(parent_request_id)

        overlapping_docs = []
        new_docs = []
        doc_source_turns = {}

        for doc_id in docs:
            if doc_id in previous_docs:
                overlapping_docs.append(doc_id)
                doc_source_turns[doc_id] = doc_sources[doc_id]
            else:
                new_docs.append(doc_id)

        template = hint_template or self._hint_template
        reference_hints = []

        for doc_id in overlapping_docs:
            source_request = doc_sources.get(doc_id)
            source_history = (
                self._requests.get(source_request) if source_request else None
            )
            turn_number = source_history.turn_number if source_history else "previous"
            hint = template.format(
                doc_id=doc_id,
                turn_number=turn_number,
                source_request=source_request or "previous",
            )
            reference_hints.append(hint)

        self.register_request(request_id, docs, parent_request_id)
        self._stats["total_docs_deduplicated"] += len(overlapping_docs)

        logger.info(
            f"Deduplication for {request_id}: "
            f"{len(overlapping_docs)} overlapping, {len(new_docs)} new"
        )

        result = DeduplicationResult(
            original_docs=docs,
            overlapping_docs=overlapping_docs,
            new_docs=new_docs,
            reference_hints=reference_hints,
            deduplicated_docs=new_docs,
            doc_source_turns=doc_source_turns,
            is_new_conversation=False,
        )

        if doc_contents:
            self._apply_block_dedup(doc_contents, result)

        return result

    def _apply_block_dedup(
        self, doc_contents: Dict[int, str], result: DeduplicationResult
    ) -> None:
        from contextpilot.dedup.block_dedup import (
            _content_defined_chunking,
            _hash_block,
            MIN_BLOCK_CHARS,
            MIN_CONTENT_CHARS,
        )

        seen_blocks: Dict[str, int] = {}

        for doc_id in result.original_docs:
            content = doc_contents.get(doc_id, "")
            if len(content) < MIN_CONTENT_CHARS:
                continue

            blocks = _content_defined_chunking(content)
            if len(blocks) < 2:
                for b in blocks:
                    if len(b.strip()) >= MIN_BLOCK_CHARS:
                        h = _hash_block(b)
                        if h not in seen_blocks:
                            seen_blocks[h] = doc_id
                continue

            new_blocks = []
            deduped_count = 0

            for block in blocks:
                if len(block.strip()) < MIN_BLOCK_CHARS:
                    new_blocks.append(block)
                    continue

                h = _hash_block(block)
                result.blocks_total += 1

                if h in seen_blocks and seen_blocks[h] != doc_id:
                    first_line = block.strip().split("\n")[0][:80]
                    ref = f'[... "{first_line}" — identical to earlier result, see above ...]'
                    if len(block) > len(ref):
                        new_blocks.append(ref)
                        deduped_count += 1
                        result.blocks_deduped += 1
                        result.block_chars_saved += len(block) - len(ref)
                    else:
                        new_blocks.append(block)
                else:
                    if h not in seen_blocks:
                        seen_blocks[h] = doc_id
                    new_blocks.append(block)

            if deduped_count > 0:
                doc_contents[doc_id] = "\n\n".join(new_blocks)

        if result.blocks_deduped > 0:
            logger.info(
                f"Block dedup: {result.blocks_deduped}/{result.blocks_total} blocks, "
                f"saved {result.block_chars_saved:,} chars"
            )

    def deduplicate_batch(
        self,
        request_ids: List[str],
        docs_list: List[List[int]],
        parent_request_ids: Optional[List[Optional[str]]] = None,
        hint_template: Optional[str] = None,
        doc_contents_list: Optional[List[Optional[Dict[int, str]]]] = None,
    ) -> List[DeduplicationResult]:
        if parent_request_ids is None:
            parent_request_ids = [None] * len(request_ids)
        if doc_contents_list is None:
            doc_contents_list = [None] * len(request_ids)

        results = []
        for req_id, docs, parent_id, doc_contents in zip(
            request_ids, docs_list, parent_request_ids, doc_contents_list
        ):
            result = self.deduplicate(
                req_id, docs, parent_id, hint_template, doc_contents=doc_contents
            )
            results.append(result)

        return results

    def remove_request(self, request_id: str) -> bool:
        """
        Remove a request from tracking.

        Note: This will NOT update parent references of child requests.
        Use with caution.

        Args:
            request_id: The request to remove

        Returns:
            True if removed, False if not found
        """
        if request_id in self._requests:
            del self._requests[request_id]
            return True
        return False

    def clear_conversation(self, request_id: str) -> int:
        """
        Clear all requests in a conversation chain.

        Args:
            request_id: Any request in the conversation

        Returns:
            Number of requests removed
        """
        chain = self.get_conversation_chain(request_id)
        count = 0

        for history in chain:
            if self.remove_request(history.request_id):
                count += 1

        return count

    def reset(self):
        """Clear all tracked conversations."""
        self._requests.clear()
        self._stats = {
            "total_requests": 0,
            "total_dedup_calls": 0,
            "total_docs_deduplicated": 0,
        }
        logger.info("ConversationTracker reset")

    def get_stats(self) -> Dict:
        """Get tracking statistics."""
        return {
            **self._stats,
            "active_requests": len(self._requests),
        }

    def get_request_history(self, request_id: str) -> Optional[RequestHistory]:
        """Get history for a specific request."""
        return self._requests.get(request_id)


# Singleton instance for use across the server
_conversation_tracker: Optional[ConversationTracker] = None


def get_conversation_tracker() -> ConversationTracker:
    """Get the global conversation tracker instance."""
    global _conversation_tracker
    if _conversation_tracker is None:
        _conversation_tracker = ConversationTracker()
    return _conversation_tracker


def reset_conversation_tracker():
    """Reset the global conversation tracker."""
    global _conversation_tracker
    if _conversation_tracker is not None:
        _conversation_tracker.reset()
    else:
        _conversation_tracker = ConversationTracker()
