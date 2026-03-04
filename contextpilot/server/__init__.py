"""
ContextPilot Server - Live Context Index Server

Provides a persistent server for maintaining dynamic context indices
with support for search, insertion, updates, and eviction.

Includes HTTP server/client for remote index access from SGLang.
"""

from .metadata import NodeMetadata
from .eviction_heap import EvictionHeap
from .live_index import ContextPilot

# HTTP client (optional - requires requests)
try:
    from .http_client import ContextPilotIndexClient, evict_tokens
    _HTTP_AVAILABLE = True
except ImportError:
    _HTTP_AVAILABLE = False
    ContextPilotIndexClient = None
    evict_tokens = None

__all__ = [
    'NodeMetadata',
    'EvictionHeap',
    'ContextPilot',
    'ContextPilotIndexClient',
    'evict_tokens',
]
