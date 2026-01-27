"""
SGLang Cache Init Params - ContextPilot Patched Version

This file adds eviction callback support for ContextPilot integration.
Copy this file to: sglang/srt/mem_cache/cache_init_params.py

Changes from original:
1. Added EvictionCallback type alias
2. Added eviction_callback field to CacheInitParams
3. Added __post_init__ to auto-create callback from RAGBOOST_INDEX_URL
"""
from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Callable, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


# Type alias for eviction callback: callable that receives a set of evicted request IDs
EvictionCallback = Optional[Callable[[set], None]]


@dataclasses.dataclass
class CacheInitParams:
    disable: bool
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int

    is_eagle: bool = False
    tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    eviction_policy: str = "lru"
    disable_finished_insert: bool = False

    enable_metrics: bool = False
    enable_kv_cache_events: bool = False

    # ContextPilot Integration:
    # Callback invoked when requests' extra tokens are fully evicted from cache.
    # The callback receives a set of request IDs whose unique tokens have been evicted.
    # If None and RAGBOOST_INDEX_URL is set, auto-creates a callback in __post_init__.
    eviction_callback: EvictionCallback = None

    def __post_init__(self):
        """Auto-create eviction callback from RAGBOOST_INDEX_URL if not provided."""
        if self.eviction_callback is None:
            from sglang.srt.mem_cache.common import create_contextpilot_eviction_callback
            self.eviction_callback = create_contextpilot_eviction_callback()
