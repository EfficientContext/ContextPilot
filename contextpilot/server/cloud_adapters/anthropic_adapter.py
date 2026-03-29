"""
Anthropic Cloud Provider Adapter.

Handles Anthropic Messages API specifics:
- cache_control: {"type": "ephemeral"} injection on content blocks
- x-api-key authentication
- Cache metrics parsing from response.usage
"""

import copy
import logging
from typing import Any, Dict, FrozenSet, List, Set

from .base import CacheMetrics, CloudProviderAdapter, TTLTier

logger = logging.getLogger(__name__)

_ANTHROPIC_API_BASE = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"
_MIN_CONTENT_LENGTH_FOR_CACHE = 1024

_CACHE_CONTROL_DEFAULT = {"type": "ephemeral"}
_CACHE_CONTROL_EXTENDED = {"type": "ephemeral", "ttl": "1h"}


class AnthropicAdapter(CloudProviderAdapter):
    """Adapter for Anthropic Messages API with prompt caching support."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def get_api_url(self, path: str = "") -> str:
        return f"{_ANTHROPIC_API_BASE}{path}"

    def get_auth_headers(self, api_key: str) -> Dict[str, str]:
        return {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def get_target_path(self) -> str:
        return "/v1/messages"

    def get_default_ttl_seconds(self) -> int:
        return 300

    def get_extended_ttl_seconds(self):
        return 3600

    @property
    def _cache_control_value(self) -> Dict[str, str]:
        if self._configured_ttl == TTLTier.LONG:
            return _CACHE_CONTROL_EXTENDED
        return _CACHE_CONTROL_DEFAULT

    def inject_cache_control(
        self, body: Dict[str, Any], cached_hashes: Set[str]
    ) -> Dict[str, Any]:
        body = copy.deepcopy(body)
        cc = self._cache_control_value
        body = _inject_system_cache_control(body, cc)
        body = _inject_tool_result_cache_control(body, cc)
        return body

    def parse_cache_metrics(self, response_body: Dict[str, Any]) -> CacheMetrics:
        usage = response_body.get("usage", {})
        return CacheMetrics(
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )


# ---------------------------------------------------------------------------
# Helpers (shared with MiniMaxAdapter via import)
# ---------------------------------------------------------------------------


def _inject_system_cache_control(
    body: Dict[str, Any], cc: Dict[str, str]
) -> Dict[str, Any]:
    system = body.get("system")
    if system is None:
        return body

    if isinstance(system, str):
        body["system"] = [{"type": "text", "text": system, "cache_control": cc}]
    elif isinstance(system, list) and system:
        last_block = system[-1]
        if isinstance(last_block, dict):
            last_block["cache_control"] = cc
    return body


_MAX_TOOL_RESULT_BREAKPOINTS = 3  # Anthropic allows 4 total; 1 reserved for system


def _inject_tool_result_cache_control(
    body: Dict[str, Any], cc: Dict[str, str]
) -> Dict[str, Any]:
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return body

    breakpoints_used = 0
    for msg in messages:
        if breakpoints_used >= _MAX_TOOL_RESULT_BREAKPOINTS:
            break
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if breakpoints_used >= _MAX_TOOL_RESULT_BREAKPOINTS:
                break
            if not isinstance(block, dict):
                continue
            if block.get("type") not in ("tool_result", "toolResult"):
                continue
            _maybe_add_cache_control_to_tool_result(block, cc)
            breakpoints_used += 1

    return body


def _maybe_add_cache_control_to_tool_result(
    block: Dict[str, Any], cc: Dict[str, str]
) -> None:
    tr_content = block.get("content", "")

    if isinstance(tr_content, str):
        if len(tr_content) >= _MIN_CONTENT_LENGTH_FOR_CACHE:
            block["cache_control"] = cc
    elif isinstance(tr_content, list):
        total_chars = sum(
            len(inner.get("text", ""))
            for inner in tr_content
            if isinstance(inner, dict) and inner.get("type") == "text"
        )
        if total_chars >= _MIN_CONTENT_LENGTH_FOR_CACHE and tr_content:
            last_text_block = None
            for inner in reversed(tr_content):
                if isinstance(inner, dict) and inner.get("type") == "text":
                    last_text_block = inner
                    break
            if last_text_block is not None:
                last_text_block["cache_control"] = cc
