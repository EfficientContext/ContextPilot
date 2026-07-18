import logging
import time
from typing import Any, Dict, List
import torch
import torch.nn.functional as F

from .base import BasePlugin

logger = logging.getLogger(__name__)

class DynamicPruningPlugin(BasePlugin):
    """
    Plugin for aggressively pruning redundant conversational history using Semantic Similarity.
    """

    def __init__(self, similarity_threshold: float = 0.3):
        super().__init__("dynamic_pruning")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = similarity_threshold

        # Telemetry
        self.total_original_chars = 0
        self.total_chars_saved = 0
        self.total_requests_processed = 0
        self.last_execution_time_ms = 0.0

    async def process(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        messages = request_data.get("messages", [])
        if not messages or len(messages) <= 2:
            # Nothing to prune if only system + user
            return request_data

        start_time = time.perf_counter()

        original_len = sum(len(m.get("content", "")) for m in messages)

        # 1. Identify system prompt and current query
        system_idx = 0 if messages[0].get("role") == "system" else -1
        current_query_idx = len(messages) - 1
        current_query_text = messages[current_query_idx].get("content", "")

        # 2. Extract intermediate historical messages
        start_idx = 1 if system_idx == 0 else 0
        history_indices = list(range(start_idx, current_query_idx))

        if not history_indices:
            return request_data

        history_texts = [messages[i].get("content", "") for i in history_indices]

        # 3. Calculate Cosine Similarity
        # Encode current query and history texts
        query_embedding = self.model.encode(current_query_text, convert_to_tensor=True)
        history_embeddings = self.model.encode(history_texts, convert_to_tensor=True)

        # Compute cosine similarities
        # query_embedding is (dim,), so unsqueeze to (1, dim) for broadcasting
        cosine_scores = F.cosine_similarity(query_embedding.unsqueeze(0), history_embeddings)

        # 4. Filter messages
        new_messages = []
        if system_idx == 0:
            new_messages.append(messages[0])

        for i, score in enumerate(cosine_scores):
            if score.item() >= self.similarity_threshold:
                new_messages.append(messages[history_indices[i]])

        new_messages.append(messages[current_query_idx])

        # Update Request
        optimized_request = dict(request_data)
        optimized_request["messages"] = new_messages

        # Update Telemetry
        pruned_len = sum(len(m.get("content", "")) for m in new_messages)
        self.total_original_chars += original_len
        self.total_chars_saved += (original_len - pruned_len)
        self.total_requests_processed += 1
        self.last_execution_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Pruned request in {self.last_execution_time_ms:.2f}ms. Saved {original_len - pruned_len} chars."
        )
        return optimized_request

    def get_plugin_metrics(self) -> Dict[str, float]:
        saving_percentage = (self.total_chars_saved / self.total_original_chars * 100) if self.total_original_chars > 0 else 0.0
        return {
            "total_original_chars": float(self.total_original_chars),
            "total_chars_saved": float(self.total_chars_saved),
            "chars_saved_percentage": saving_percentage,
            "total_requests_processed": float(self.total_requests_processed),
            "last_execution_time_ms": self.last_execution_time_ms,
        }
