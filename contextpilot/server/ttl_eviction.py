"""
TTL-based Eviction Policy for Cloud Prompt Cache Proxy

Models the cache state of cloud LLM providers (Anthropic, OpenAI, MiniMax)
locally. Cloud providers evict cached prompts after a TTL window:
- Anthropic ephemeral: ~5 minutes
- OpenAI automatic: ~5-10 minutes
- MiniMax: configurable (5 min default)

This module tracks what content is currently cached in the cloud provider
so ContextPilot can optimize document ordering to maximize cache hits.

Key Design:
- TTL-based expiry (not capacity-based like EvictionHeap)
- Thread-safe for concurrent request handling
- Tracks hit/miss statistics for monitoring
- Supports two TTL tiers: SHORT (5 min) and LONG (1 hr)
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TTLTier(str, Enum):
    SHORT = "5m"  # 300 seconds
    MEDIUM = "1h"  # 3600 seconds
    LONG = "24h"  # 86400 seconds

    @property
    def seconds(self) -> int:
        return {"5m": 300, "1h": 3600, "24h": 86400}[self.value]


@dataclass
class CacheEntry:
    """A single cached content entry with TTL tracking."""

    content_hash: str
    created_at: float
    last_accessed_at: float
    ttl_seconds: int
    token_count: int = 0

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if this entry has expired."""
        if now is None:
            now = time.time()
        return (now - self.last_accessed_at) >= self.ttl_seconds

    def time_remaining(self, now: Optional[float] = None) -> float:
        """Get seconds remaining before expiry. Negative if expired."""
        if now is None:
            now = time.time()
        return self.ttl_seconds - (now - self.last_accessed_at)

    def __repr__(self):
        remaining = self.time_remaining()
        status = f"{remaining:.0f}s left" if remaining > 0 else "EXPIRED"
        return (
            f"CacheEntry(hash={self.content_hash[:12]}..., "
            f"tokens={self.token_count}, {status})"
        )


@dataclass
class CacheMetrics:
    """Cache usage metrics from a cloud API response."""

    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class TTLEvictionPolicy:
    """
    TTL-based cache eviction policy for cloud prompt cache proxy.

    Models the cloud provider's cache state locally. Entries expire
    after their TTL window (measured from last access time), mirroring
    how cloud providers evict cached prompts.

    Thread-safe for concurrent request handling.

    Usage:
        policy = TTLEvictionPolicy(default_ttl=TTLTier.SHORT)

        # Track cached content
        policy.add_entry("abc123", token_count=5000)

        # Check if content is still cached
        if policy.is_cached("abc123"):
            print("Cache hit!")

        # Update from API response
        policy.update_from_response(metrics, "abc123")

        # Periodic cleanup
        evicted = policy.evict_expired()
    """

    def __init__(
        self,
        default_ttl: TTLTier = TTLTier.SHORT,
        default_ttl_seconds: Optional[int] = None,
    ):
        self._default_ttl = default_ttl
        self._default_ttl_seconds = default_ttl_seconds or default_ttl.seconds
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

        self._total_hits = 0
        self._total_misses = 0
        self._total_evictions = 0
        self._total_additions = 0

    @property
    def default_ttl(self) -> TTLTier:
        """Get default TTL tier."""
        return self._default_ttl

    @default_ttl.setter
    def default_ttl(self, value: TTLTier):
        """Set default TTL tier."""
        self._default_ttl = value

    def add_entry(self, content_hash: str, token_count: int = 0) -> CacheEntry:
        ttl_secs = self._default_ttl_seconds
        now = time.time()

        with self._lock:
            if content_hash in self._entries:
                entry = self._entries[content_hash]
                entry.last_accessed_at = now
                entry.token_count = token_count or entry.token_count
                logger.debug(f"Cache entry refreshed: {content_hash[:12]}...")
            else:
                entry = CacheEntry(
                    content_hash=content_hash,
                    created_at=now,
                    last_accessed_at=now,
                    ttl_seconds=ttl_secs,
                    token_count=token_count,
                )
                self._entries[content_hash] = entry
                self._total_additions += 1
                logger.debug(
                    f"Cache entry added: {content_hash[:12]}... "
                    f"(ttl={ttl_secs}s, tokens={token_count})"
                )

        return entry

    def touch_entry(self, content_hash: str) -> bool:
        """
        Refresh last_accessed_at timestamp for an entry.

        Args:
            content_hash: Hash of the cached content

        Returns:
            True if entry was found and refreshed, False if not found or expired
        """
        with self._lock:
            entry = self._entries.get(content_hash)
            if entry is None:
                return False
            if entry.is_expired():
                # Already expired, remove it
                del self._entries[content_hash]
                self._total_evictions += 1
                return False
            entry.last_accessed_at = time.time()
            self._total_hits += 1
            return True

    def is_cached(self, content_hash: str) -> bool:
        """
        Check if content is still within its TTL window.

        Args:
            content_hash: Hash of the content to check

        Returns:
            True if cached and not expired
        """
        with self._lock:
            entry = self._entries.get(content_hash)
            if entry is None:
                self._total_misses += 1
                return False
            if entry.is_expired():
                del self._entries[content_hash]
                self._total_evictions += 1
                self._total_misses += 1
                return False
            self._total_hits += 1
            return True

    def evict_expired(self) -> List[CacheEntry]:
        """
        Remove all entries that have exceeded their TTL.

        Returns:
            List of evicted CacheEntry objects
        """
        now = time.time()
        evicted = []

        with self._lock:
            expired_hashes = [
                h for h, entry in self._entries.items() if entry.is_expired(now)
            ]
            for h in expired_hashes:
                evicted.append(self._entries.pop(h))
            self._total_evictions += len(evicted)

        if evicted:
            total_tokens = sum(e.token_count for e in evicted)
            logger.info(
                f"TTL eviction: removed {len(evicted)} expired entries "
                f"({total_tokens:,} tokens)"
            )

        return evicted

    def get_cached_hashes(self) -> Set[str]:
        """
        Get set of all currently-cached content hashes (non-expired).

        Returns:
            Set of content hashes that are still within TTL
        """
        now = time.time()
        with self._lock:
            return {
                h for h, entry in self._entries.items() if not entry.is_expired(now)
            }

    def get_cached_count(self) -> int:
        """Get number of active (non-expired) cache entries."""
        now = time.time()
        with self._lock:
            return sum(
                1 for entry in self._entries.values() if not entry.is_expired(now)
            )

    def get_total_cached_tokens(self) -> int:
        """Get total tokens across all active cache entries."""
        now = time.time()
        with self._lock:
            return sum(
                entry.token_count
                for entry in self._entries.values()
                if not entry.is_expired(now)
            )

    def update_from_response(self, metrics: CacheMetrics, content_hash: str) -> None:
        """
        Update cache state based on cloud API response metrics.

        If cache_read_tokens > 0: content was served from cache -> touch entry.
        If cache_creation_tokens > 0: new content was cached -> add entry.

        Args:
            metrics: Cache metrics from the cloud API response
            content_hash: Hash of the content that was sent
        """
        if metrics.cache_read_tokens > 0:
            # Cache hit — refresh TTL
            self.touch_entry(content_hash)
            logger.debug(
                f"Cache hit confirmed: {content_hash[:12]}... "
                f"({metrics.cache_read_tokens} tokens read from cache)"
            )

        if metrics.cache_creation_tokens > 0:
            # New content cached — add/update entry
            self.add_entry(content_hash, token_count=metrics.cache_creation_tokens)
            logger.debug(
                f"Cache write confirmed: {content_hash[:12]}... "
                f"({metrics.cache_creation_tokens} tokens cached)"
            )

    def reset(self) -> None:
        """Clear all entries and reset statistics."""
        with self._lock:
            self._entries.clear()
            self._total_hits = 0
            self._total_misses = 0
            self._total_evictions = 0
            self._total_additions = 0
        logger.info("TTL eviction policy reset")

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with hit/miss counts, active entries, total tokens, etc.
        """
        now = time.time()
        with self._lock:
            active_entries = [
                e for e in self._entries.values() if not e.is_expired(now)
            ]
            total_tokens = sum(e.token_count for e in active_entries)
            total_requests = self._total_hits + self._total_misses
            hit_rate = (
                self._total_hits / total_requests * 100 if total_requests > 0 else 0
            )

            return {
                "active_entries": len(active_entries),
                "total_entries": len(self._entries),
                "total_cached_tokens": total_tokens,
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "total_evictions": self._total_evictions,
                "total_additions": self._total_additions,
                "hit_rate_pct": round(hit_rate, 2),
                "default_ttl": self._default_ttl.value,
                "default_ttl_seconds": self._default_ttl_seconds,
            }

    def __len__(self):
        """Get number of entries (including possibly expired)."""
        return len(self._entries)

    def __repr__(self):
        active = self.get_cached_count()
        return (
            f"TTLEvictionPolicy(active={active}, "
            f"total={len(self._entries)}, "
            f"default_ttl={self._default_ttl.value})"
        )
