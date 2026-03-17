"""Tests for TTL-based eviction policy."""

import time
import threading
from unittest.mock import patch

import pytest

from contextpilot.server.ttl_eviction import (
    TTLEvictionPolicy,
    TTLTier,
    CacheEntry,
    CacheMetrics,
)


class TestTTLTier:
    def test_short_tier_seconds(self):
        assert TTLTier.SHORT.seconds == 300

    def test_medium_tier_seconds(self):
        assert TTLTier.MEDIUM.seconds == 3600

    def test_long_tier_seconds(self):
        assert TTLTier.LONG.seconds == 86400

    def test_tier_values(self):
        assert TTLTier.SHORT.value == "5m"
        assert TTLTier.MEDIUM.value == "1h"
        assert TTLTier.LONG.value == "24h"

    def test_tier_from_string(self):
        assert TTLTier("5m") is TTLTier.SHORT
        assert TTLTier("1h") is TTLTier.MEDIUM
        assert TTLTier("24h") is TTLTier.LONG


class TestCacheEntry:
    def test_not_expired_when_fresh(self):
        now = time.time()
        entry = CacheEntry(
            content_hash="abc",
            created_at=now,
            last_accessed_at=now,
            ttl_seconds=300,
        )
        assert not entry.is_expired(now)

    def test_expired_after_ttl(self):
        now = time.time()
        entry = CacheEntry(
            content_hash="abc",
            created_at=now - 400,
            last_accessed_at=now - 400,
            ttl_seconds=300,
        )
        assert entry.is_expired(now)

    def test_time_remaining_positive(self):
        now = time.time()
        entry = CacheEntry(
            content_hash="abc",
            created_at=now,
            last_accessed_at=now,
            ttl_seconds=300,
        )
        assert entry.time_remaining(now) == pytest.approx(300, abs=1)

    def test_time_remaining_negative_when_expired(self):
        now = time.time()
        entry = CacheEntry(
            content_hash="abc",
            created_at=now - 400,
            last_accessed_at=now - 400,
            ttl_seconds=300,
        )
        assert entry.time_remaining(now) < 0


class TestCacheMetrics:
    def test_defaults(self):
        m = CacheMetrics()
        assert m.cache_creation_tokens == 0
        assert m.cache_read_tokens == 0
        assert m.input_tokens == 0
        assert m.output_tokens == 0

    def test_custom_values(self):
        m = CacheMetrics(
            cache_creation_tokens=1000,
            cache_read_tokens=500,
            input_tokens=1500,
            output_tokens=200,
        )
        assert m.cache_creation_tokens == 1000
        assert m.cache_read_tokens == 500


class TestTTLEvictionPolicy:
    def test_add_entry(self):
        policy = TTLEvictionPolicy()
        entry = policy.add_entry("hash1", token_count=5000)
        assert entry.content_hash == "hash1"
        assert entry.token_count == 5000
        assert policy.get_cached_count() == 1

    def test_add_entry_with_custom_ttl_seconds(self):
        policy = TTLEvictionPolicy(default_ttl=TTLTier.LONG, default_ttl_seconds=86400)
        entry = policy.add_entry("hash1")
        assert entry.ttl_seconds == 86400

    def test_add_existing_refreshes(self):
        policy = TTLEvictionPolicy()
        e1 = policy.add_entry("hash1", token_count=100)
        t1 = e1.last_accessed_at
        time.sleep(0.01)
        e2 = policy.add_entry("hash1", token_count=200)
        assert e2.last_accessed_at > t1
        assert e2.token_count == 200
        assert policy.get_cached_count() == 1

    def test_is_cached_true(self):
        policy = TTLEvictionPolicy()
        policy.add_entry("hash1")
        assert policy.is_cached("hash1")

    def test_is_cached_false_not_found(self):
        policy = TTLEvictionPolicy()
        assert not policy.is_cached("nonexistent")

    def test_is_cached_false_after_expiry(self):
        policy = TTLEvictionPolicy()
        policy.add_entry("hash1")
        now = time.time() + 600
        with patch("contextpilot.server.ttl_eviction.time.time", return_value=now):
            assert not policy.is_cached("hash1")

    def test_touch_entry(self):
        policy = TTLEvictionPolicy()
        policy.add_entry("hash1")
        assert policy.touch_entry("hash1")
        assert not policy.touch_entry("nonexistent")

    def test_touch_expired_returns_false(self):
        policy = TTLEvictionPolicy()
        policy.add_entry("hash1")
        now = time.time() + 600
        with patch("contextpilot.server.ttl_eviction.time.time", return_value=now):
            assert not policy.touch_entry("hash1")

    def test_evict_expired(self):
        policy = TTLEvictionPolicy()
        policy.add_entry("hash1", token_count=100)
        policy.add_entry("hash2", token_count=200)

        now = time.time()
        evicted = policy.evict_expired()
        assert len(evicted) == 0

        future = now + 600
        with patch("contextpilot.server.ttl_eviction.time.time", return_value=future):
            evicted = policy.evict_expired()
        assert len(evicted) == 2

    def test_evict_only_expired_short_vs_long(self):
        short_policy = TTLEvictionPolicy(
            default_ttl=TTLTier.SHORT, default_ttl_seconds=300
        )
        short_policy.add_entry("short_hash")

        long_policy = TTLEvictionPolicy(
            default_ttl=TTLTier.LONG, default_ttl_seconds=3600
        )
        long_policy.add_entry("long_hash")

        future_6min = time.time() + 360
        with patch(
            "contextpilot.server.ttl_eviction.time.time", return_value=future_6min
        ):
            evicted_short = short_policy.evict_expired()
            evicted_long = long_policy.evict_expired()
        assert len(evicted_short) == 1
        assert evicted_short[0].content_hash == "short_hash"
        assert len(evicted_long) == 0

    def test_get_cached_hashes(self):
        policy = TTLEvictionPolicy()
        policy.add_entry("a")
        policy.add_entry("b")
        policy.add_entry("c")
        hashes = policy.get_cached_hashes()
        assert hashes == {"a", "b", "c"}

    def test_get_total_cached_tokens(self):
        policy = TTLEvictionPolicy()
        policy.add_entry("a", token_count=100)
        policy.add_entry("b", token_count=200)
        assert policy.get_total_cached_tokens() == 300

    def test_update_from_response_cache_hit(self):
        policy = TTLEvictionPolicy()
        policy.add_entry("hash1", token_count=100)
        metrics = CacheMetrics(cache_read_tokens=100)
        policy.update_from_response(metrics, "hash1")
        stats = policy.get_stats()
        assert stats["total_hits"] >= 1

    def test_update_from_response_cache_write(self):
        policy = TTLEvictionPolicy()
        metrics = CacheMetrics(cache_creation_tokens=5000)
        policy.update_from_response(metrics, "new_hash")
        assert policy.is_cached("new_hash")

    def test_reset(self):
        policy = TTLEvictionPolicy()
        policy.add_entry("a")
        policy.add_entry("b")
        policy.reset()
        assert policy.get_cached_count() == 0
        stats = policy.get_stats()
        assert stats["total_hits"] == 0
        assert stats["total_misses"] == 0

    def test_get_stats(self):
        policy = TTLEvictionPolicy()
        policy.add_entry("a", token_count=100)
        policy.is_cached("a")
        policy.is_cached("missing")

        stats = policy.get_stats()
        assert stats["active_entries"] == 1
        assert stats["total_cached_tokens"] == 100
        assert stats["total_hits"] >= 1
        assert stats["total_misses"] >= 1
        assert stats["default_ttl"] == "5m"
        assert stats["default_ttl_seconds"] == 300

    def test_default_ttl_property(self):
        policy = TTLEvictionPolicy(default_ttl=TTLTier.LONG)
        assert policy.default_ttl == TTLTier.LONG
        policy.default_ttl = TTLTier.SHORT
        assert policy.default_ttl == TTLTier.SHORT

    def test_len(self):
        policy = TTLEvictionPolicy()
        assert len(policy) == 0
        policy.add_entry("a")
        policy.add_entry("b")
        assert len(policy) == 2

    def test_repr(self):
        policy = TTLEvictionPolicy()
        policy.add_entry("a")
        r = repr(policy)
        assert "active=1" in r
        assert "default_ttl=5m" in r

    def test_thread_safety(self):
        policy = TTLEvictionPolicy()
        errors = []

        def writer(prefix, count):
            try:
                for i in range(count):
                    policy.add_entry(f"{prefix}_{i}", token_count=i)
            except Exception as e:
                errors.append(e)

        def reader(count):
            try:
                for _ in range(count):
                    policy.get_cached_hashes()
                    policy.get_stats()
                    policy.evict_expired()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("a", 100)),
            threading.Thread(target=writer, args=("b", 100)),
            threading.Thread(target=reader, args=(50,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
        assert policy.get_cached_count() == 200
