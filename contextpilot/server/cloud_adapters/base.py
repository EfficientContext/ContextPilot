"""
Base classes and shared types for cloud provider adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, FrozenSet, Optional, Set

from contextpilot.server.ttl_eviction import TTLTier, CacheMetrics


class CloudProviderAdapter(ABC):
    """Abstract base for cloud LLM API provider adapters.

    Each adapter handles provider-specific details:
    - API URL construction
    - Authentication headers
    - Cache control annotation injection
    - Response cache metrics parsing
    """

    def __init__(self):
        self._configured_ttl: Optional[TTLTier] = None

    @property
    def configured_ttl(self) -> Optional[TTLTier]:
        return self._configured_ttl

    @configured_ttl.setter
    def configured_ttl(self, value: TTLTier):
        self._configured_ttl = value

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier (e.g. 'anthropic', 'openai')."""
        ...

    @abstractmethod
    def get_api_url(self, path: str = "") -> str:
        """Build full API URL for the given path."""
        ...

    @abstractmethod
    def get_auth_headers(self, api_key: str) -> Dict[str, str]:
        """Build authentication headers for the provider."""
        ...

    @abstractmethod
    def inject_cache_control(
        self, body: Dict[str, Any], cached_hashes: Set[str]
    ) -> Dict[str, Any]:
        """Add provider-specific cache control annotations to the request body.

        Args:
            body: Request body (will be deep-copied internally if modified)
            cached_hashes: Set of content hashes currently in cache

        Returns:
            Modified request body with cache control annotations
        """
        ...

    @abstractmethod
    def parse_cache_metrics(self, response_body: Dict[str, Any]) -> CacheMetrics:
        """Extract cache usage metrics from the API response."""
        ...

    @abstractmethod
    def get_default_ttl_seconds(self) -> int:
        """Default local index TTL in seconds."""
        ...

    @abstractmethod
    def get_extended_ttl_seconds(self) -> Optional[int]:
        """Extended TTL in seconds, or None if not supported."""
        ...

    @property
    def supports_extended_cache(self) -> bool:
        return self.get_extended_ttl_seconds() is not None

    @abstractmethod
    def get_target_path(self) -> str:
        """Get the API endpoint path (e.g. '/v1/messages')."""
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(provider={self.provider_name!r})"
