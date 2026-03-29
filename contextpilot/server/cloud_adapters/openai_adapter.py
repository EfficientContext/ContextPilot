"""
OpenAI Cloud Provider Adapter.

Only supports explicit prompt_cache_retention="24h" (extended caching).
In-memory caching (5-10 min, auto-adjusted) is NOT supported because the
TTL is non-deterministic.
"""

import copy
import logging
from typing import Any, Dict, FrozenSet, Set

from .base import CacheMetrics, CloudProviderAdapter, TTLTier

logger = logging.getLogger(__name__)

_OPENAI_API_BASE = "https://api.openai.com"


class OpenAIAdapter(CloudProviderAdapter):
    @property
    def provider_name(self) -> str:
        return "openai"

    def get_api_url(self, path: str = "") -> str:
        return f"{_OPENAI_API_BASE}{path}"

    def get_auth_headers(self, api_key: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def get_target_path(self) -> str:
        return "/v1/chat/completions"

    def get_default_ttl_seconds(self) -> int:
        return 3600

    def get_extended_ttl_seconds(self):
        return 86400

    def inject_cache_control(
        self, body: Dict[str, Any], cached_hashes: Set[str]
    ) -> Dict[str, Any]:
        if self.configured_ttl == TTLTier.LONG:
            body = copy.deepcopy(body)
            body["prompt_cache_retention"] = "24h"
        return body

    def parse_cache_metrics(self, response_body: Dict[str, Any]) -> CacheMetrics:
        usage = response_body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # OpenAI reports cached tokens in prompt_tokens_details
        details = usage.get("prompt_tokens_details", {})
        cached_tokens = 0
        if isinstance(details, dict):
            cached_tokens = details.get("cached_tokens", 0)

        return CacheMetrics(
            cache_creation_tokens=max(0, prompt_tokens - cached_tokens)
            if cached_tokens
            else 0,
            cache_read_tokens=cached_tokens,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
