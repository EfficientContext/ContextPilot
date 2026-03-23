"""
MiniMax Cloud Provider Adapter.

MiniMax provides an Anthropic-compatible API at api.minimax.io/anthropic.
Uses the same cache_control: {"type": "ephemeral"} format and response
metrics as Anthropic.
"""

import copy
import logging
from typing import Any, Dict, FrozenSet, Set

from .base import CacheMetrics, CloudProviderAdapter, TTLTier
from .anthropic_adapter import (
    _CACHE_CONTROL_DEFAULT,
    _inject_system_cache_control,
    _inject_tool_result_cache_control,
)

logger = logging.getLogger(__name__)

_MINIMAX_API_BASE = "https://api.minimax.io/anthropic"


class MiniMaxAdapter(CloudProviderAdapter):
    """Adapter for MiniMax Anthropic-compatible API with prompt caching."""

    @property
    def provider_name(self) -> str:
        return "minimax"

    def get_api_url(self, path: str = "") -> str:
        return f"{_MINIMAX_API_BASE}{path}"

    def get_auth_headers(self, api_key: str) -> Dict[str, str]:
        return {
            "x-api-key": api_key,
            "content-type": "application/json",
        }

    def get_target_path(self) -> str:
        return "/v1/messages"

    def get_default_ttl_seconds(self) -> int:
        return 300

    def get_extended_ttl_seconds(self):
        return None

    def inject_cache_control(
        self, body: Dict[str, Any], cached_hashes: Set[str]
    ) -> Dict[str, Any]:
        """Inject cache_control using Anthropic-compatible format."""
        body = copy.deepcopy(body)
        body = _inject_system_cache_control(body, _CACHE_CONTROL_DEFAULT)
        body = _inject_tool_result_cache_control(body, _CACHE_CONTROL_DEFAULT)
        return body

    def parse_cache_metrics(self, response_body: Dict[str, Any]) -> CacheMetrics:
        """Parse cache metrics — same format as Anthropic."""
        usage = response_body.get("usage", {})
        return CacheMetrics(
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
