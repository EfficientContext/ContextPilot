
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from .base import BasePlugin

logger = logging.getLogger(__name__)

class ContextDedupPlugin(BasePlugin):
    """
    Plugin for deduplicating redundant conversational history in multi-turn requests.
    Uses ContextPilot's ConversationTracker to replace repeated messages with reference hints.
    """
    
    def __init__(self, hint_template: str = "[Reference to Turn {turn_number}]"):
        super().__init__("context_dedup")
        from contextpilot.server.conversation_tracker import ConversationTracker
        
        self.tracker = ConversationTracker(hint_template=hint_template)
        self._content_to_id = {}
        self._next_id = 0
        
        # Telemetry
        self.total_chars_saved = 0
        self.total_requests_processed = 0
        self.last_execution_time_ms = 0.0

    def _get_id(self, content: str) -> int:
        """Map message content to a unique integer ID."""
        if content not in self._content_to_id:
            self._content_to_id[content] = self._next_id
            self._next_id += 1
        return self._content_to_id[content]

    async def process(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deduplicate a single OpenAI request.
        """
        messages = request_data.get("messages", [])
        if not messages:
            return request_data

        start_time = time.perf_counter()
        
        # Calculate original char length for telemetry
        original_len = sum(len(m.get("content", "")) for m in messages)

        conv_id = request_data.get("user_id", "default_session")
        parent_id = request_data.get("parent_id")

        # 1. Convert messages to IDs
        message_ids = [self._get_id(m.get("content", "")) for m in messages]

        # 2. Run Deduplication
        current_req_id = str(uuid.uuid4())
        result = self.tracker.deduplicate(
            request_id=current_req_id,
            docs=message_ids,
            parent_request_id=parent_id
        )

        # 3. Reconstruct messages with hints
        new_messages = []
        for i, m in enumerate(messages):
            msg_id = message_ids[i]
            if msg_id in result.overlapping_docs:
                hint_idx = result.overlapping_docs.index(msg_id)
                hint_text = result.reference_hints[hint_idx]
                new_messages.append({"role": m.get("role"), "content": hint_text})
            else:
                new_messages.append(m)

        # Update Request
        optimized_request = dict(request_data)
        optimized_request["messages"] = new_messages
        optimized_request["current_id"] = current_req_id
        
        # Update Telemetry
        dedup_len = sum(len(m.get("content", "")) for m in new_messages)
        self.total_chars_saved += (original_len - dedup_len)
        self.total_requests_processed += 1
        self.last_execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"Deduplicated request in {self.last_execution_time_ms:.2f}ms. Saved {original_len - dedup_len} chars.")
        return optimized_request

    def get_plugin_metrics(self) -> Dict[str, float]:
        """Return deduplication metrics."""
        return {
            "total_chars_saved": float(self.total_chars_saved),
            "total_requests_processed": float(self.total_requests_processed),
            "last_execution_time_ms": self.last_execution_time_ms
        }
