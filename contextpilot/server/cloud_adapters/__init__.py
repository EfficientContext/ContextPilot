"""
Cloud Provider Adapters for ContextPilot Prompt Cache Proxy.

Provides adapters for Anthropic, OpenAI, and MiniMax cloud LLM APIs,
handling API-specific auth, cache control injection, and response parsing.
"""

from .base import CloudProviderAdapter, CacheMetrics, TTLTier
from .anthropic_adapter import AnthropicAdapter
from .openai_adapter import OpenAIAdapter
from .minimax_adapter import MiniMaxAdapter


_ADAPTERS = {
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
    "minimax": MiniMaxAdapter,
}


def get_cloud_adapter(provider: str) -> CloudProviderAdapter:
    """Factory: create adapter by provider name.

    Args:
        provider: One of 'anthropic', 'openai', 'minimax'

    Returns:
        CloudProviderAdapter instance

    Raises:
        ValueError: If provider is not recognized
    """
    cls = _ADAPTERS.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown cloud provider: {provider!r}. "
            f"Choose from: {list(_ADAPTERS.keys())}"
        )
    return cls()


__all__ = [
    "CloudProviderAdapter",
    "CacheMetrics",
    "TTLTier",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "MiniMaxAdapter",
    "get_cloud_adapter",
]
