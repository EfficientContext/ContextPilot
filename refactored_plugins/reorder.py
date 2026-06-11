import logging
import time
from typing import Any, Dict, List, Optional
from .base import BasePlugin

logger = logging.getLogger(__name__)


class ContextReorderPlugin(BasePlugin):
    """
    Plugin for reordering prompts to maximize KV Cache prefix sharing.
    Optimized for OpenAI-formatted request batches.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", alpha: float = 0.001, use_gpu: bool = False):
        super().__init__("context_reorder")
        from contextpilot.server.live_index import ContextPilot
        from contextpilot.utils.prompt_generator import get_tokenizer

        self.pilot = ContextPilot(alpha=alpha, use_gpu=use_gpu, linkage_method="single")
        self.pilot.num_workers = 1
        self.tokenizer = get_tokenizer(model_name)
        if self.tokenizer is None:
            logger.warning(f"Could not load tokenizer for {model_name}. Using fallback char-split.")

        # Telemetry
        self.total_processed_batches = 0
        self.last_execution_time_ms = 0.0

    def _tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs using the configured tokenizer."""
        if self.tokenizer:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return [ord(c) for c in text]

    async def process(self, request_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of OpenAI requests.
        """
        if not request_batch:
            return []

        start_time = time.perf_counter()

        # 1. Extract and Tokenize
        tokenized_contexts = []
        for req in request_batch:
            full_text = "\n".join([m.get("content", "") for m in req.get("messages", [])])
            tokenized_contexts.append(self._tokenize(full_text))

        # 2. Run ContextPilot Scheduling
        result = self.pilot.build_and_schedule(tokenized_contexts)

        # 3. Reorder the original JSON objects
        new_order_indices = result["original_indices"]
        reordered_batch = [request_batch[i] for i in new_order_indices]

        # Update Telemetry
        self.last_execution_time_ms = (time.perf_counter() - start_time) * 1000
        self.total_processed_batches += 1

        logger.info(f"Reordered batch of {len(request_batch)} requests in {self.last_execution_time_ms:.2f}ms")
        return reordered_batch

    def get_plugin_metrics(self) -> Dict[str, float]:
        """Return reordering metrics."""
        return {
            "total_processed_batches": float(self.total_processed_batches),
            "last_execution_time_ms": self.last_execution_time_ms,
        }
