"""Integration tests for cloud prompt cache proxy flow.

Tests the end-to-end interaction between:
- TTLEvictionPolicy (cache state tracking)
- CloudProviderAdapters (cache control injection + metrics parsing)
- The combined flow: request → inject → forward → parse → update state
"""

import copy
import time
from unittest.mock import patch

import pytest

from contextpilot.server.ttl_eviction import TTLEvictionPolicy, TTLTier, CacheMetrics
from contextpilot.server.cloud_adapters import (
    get_cloud_adapter,
    AnthropicAdapter,
    OpenAIAdapter,
    MiniMaxAdapter,
)


class TestEndToEndCacheFlow:
    """Test the full cache lifecycle: inject → forward → parse → track."""

    def _simulate_request_response(self, adapter, policy, body, response_body):
        """Simulate one request-response cycle through the cloud proxy."""
        import hashlib
        import json

        cached_hashes = policy.get_cached_hashes()
        modified_body = adapter.inject_cache_control(body, cached_hashes)

        content_hash = hashlib.sha256(
            json.dumps(
                modified_body.get("system", ""), sort_keys=True, ensure_ascii=False
            ).encode()
        ).hexdigest()[:24]

        metrics = adapter.parse_cache_metrics(response_body)
        policy.update_from_response(metrics, content_hash)

        return modified_body, metrics, content_hash

    def test_anthropic_first_request_creates_cache(self):
        adapter = get_cloud_adapter("anthropic")
        policy = TTLEvictionPolicy(default_ttl=TTLTier.SHORT)

        body = {
            "system": "You are a helpful AI assistant.",
            "messages": [{"role": "user", "content": "Hello!"}],
        }
        response = {
            "usage": {
                "cache_creation_input_tokens": 500,
                "cache_read_input_tokens": 0,
                "input_tokens": 510,
                "output_tokens": 50,
            }
        }

        modified_body, metrics, content_hash = self._simulate_request_response(
            adapter, policy, body, response
        )

        assert metrics.cache_creation_tokens == 500
        assert metrics.cache_read_tokens == 0
        assert policy.is_cached(content_hash)
        assert policy.get_cached_count() == 1

    def test_anthropic_second_request_hits_cache(self):
        adapter = get_cloud_adapter("anthropic")
        policy = TTLEvictionPolicy(default_ttl=TTLTier.SHORT)

        body = {
            "system": "You are a helpful AI assistant.",
            "messages": [{"role": "user", "content": "Hello!"}],
        }

        # First request: cache write
        resp1 = {
            "usage": {
                "cache_creation_input_tokens": 500,
                "cache_read_input_tokens": 0,
                "input_tokens": 510,
                "output_tokens": 50,
            }
        }
        _, _, hash1 = self._simulate_request_response(adapter, policy, body, resp1)

        # Second request: cache hit
        body2 = copy.deepcopy(body)
        body2["messages"] = [{"role": "user", "content": "Follow-up question."}]
        resp2 = {
            "usage": {
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 500,
                "input_tokens": 510,
                "output_tokens": 60,
            }
        }
        _, metrics2, hash2 = self._simulate_request_response(
            adapter, policy, body2, resp2
        )

        assert metrics2.cache_read_tokens == 500
        assert metrics2.cache_creation_tokens == 0
        stats = policy.get_stats()
        assert stats["total_hits"] >= 1

    def test_ttl_expiry_triggers_recache(self):
        adapter = get_cloud_adapter("anthropic")
        policy = TTLEvictionPolicy(default_ttl=TTLTier.SHORT)

        body = {
            "system": "You are a coding assistant.",
            "messages": [{"role": "user", "content": "hi"}],
        }

        # First request: create cache
        resp1 = {
            "usage": {
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 0,
                "input_tokens": 1010,
                "output_tokens": 50,
            }
        }
        _, _, hash1 = self._simulate_request_response(adapter, policy, body, resp1)
        assert policy.is_cached(hash1)

        # Simulate 6 minutes passing (TTL=5min)
        future = time.time() + 360
        with patch("contextpilot.server.ttl_eviction.time.time", return_value=future):
            policy.evict_expired()
            assert not policy.is_cached(hash1)
            assert policy.get_cached_count() == 0

        # Third request: must re-cache
        resp3 = {
            "usage": {
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 0,
                "input_tokens": 1010,
                "output_tokens": 50,
            }
        }
        _, metrics3, hash3 = self._simulate_request_response(
            adapter, policy, body, resp3
        )
        assert metrics3.cache_creation_tokens == 1000
        assert policy.is_cached(hash3)

    def test_openai_extended_caching_tracking(self):
        adapter = get_cloud_adapter("openai")
        adapter.configured_ttl = TTLTier.LONG
        policy = TTLEvictionPolicy(
            default_ttl_seconds=adapter.get_extended_ttl_seconds(),
        )

        body = {
            "model": "gpt-4.1",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        modified = adapter.inject_cache_control(body, policy.get_cached_hashes())
        assert modified["prompt_cache_retention"] == "24h"
        assert "prompt_cache_retention" not in body

        # First response: no cache yet
        resp1 = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        }
        metrics1 = adapter.parse_cache_metrics(resp1)
        assert metrics1.cache_read_tokens == 0

        # Second response: cached
        resp2 = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 800},
            }
        }
        metrics2 = adapter.parse_cache_metrics(resp2)
        assert metrics2.cache_read_tokens == 800
        assert metrics2.cache_creation_tokens == 200

    def test_minimax_cache_flow(self):
        adapter = get_cloud_adapter("minimax")
        policy = TTLEvictionPolicy(default_ttl=TTLTier.SHORT)

        body = {
            "system": "You are a literary analysis assistant.",
            "messages": [{"role": "user", "content": "Analyze themes in this book."}],
        }

        modified, metrics, content_hash = self._simulate_request_response(
            adapter,
            policy,
            body,
            {
                "usage": {
                    "cache_creation_input_tokens": 188086,
                    "cache_read_input_tokens": 0,
                    "input_tokens": 21,
                    "output_tokens": 393,
                }
            },
        )

        # System should have cache_control injected
        assert isinstance(modified["system"], list)
        assert modified["system"][0]["cache_control"] == {"type": "ephemeral"}

        # Cache state updated
        assert policy.is_cached(content_hash)
        assert policy.get_total_cached_tokens() == 188086


class TestMultiProviderCacheIsolation:
    """Verify each provider's cache state is independent."""

    def test_separate_policies_per_provider(self):
        policies = {
            name: TTLEvictionPolicy(
                default_ttl_seconds=get_cloud_adapter(name).get_default_ttl_seconds()
            )
            for name in ["anthropic", "openai", "minimax"]
        }

        policies["anthropic"].add_entry("shared_hash", token_count=100)
        assert policies["anthropic"].is_cached("shared_hash")
        assert not policies["openai"].is_cached("shared_hash")
        assert not policies["minimax"].is_cached("shared_hash")


class TestLongTTLTier:
    """Test 1-hour TTL tier."""

    def test_long_ttl_survives_5min(self):
        policy = TTLEvictionPolicy(default_ttl=TTLTier.LONG)
        policy.add_entry("long_lived", token_count=5000)

        future_6min = time.time() + 360
        with patch(
            "contextpilot.server.ttl_eviction.time.time", return_value=future_6min
        ):
            policy.evict_expired()
            assert policy.is_cached("long_lived")

    def test_long_ttl_expires_after_24hr(self):
        policy = TTLEvictionPolicy(default_ttl=TTLTier.LONG, default_ttl_seconds=86400)
        policy.add_entry("long_lived", content_hash="long_lived", token_count=5000)

        future_25hr = time.time() + 90000
        with patch(
            "contextpilot.server.ttl_eviction.time.time", return_value=future_25hr
        ):
            evicted = policy.evict_expired()
            assert len(evicted) == 1
            assert evicted[0].content_hash == "long_lived"


class TestCacheStatsAccumulation:
    """Test that cache statistics accumulate correctly over multiple requests."""

    def test_stats_accumulate_across_requests(self):
        adapter = get_cloud_adapter("anthropic")
        policy = TTLEvictionPolicy()

        for i in range(5):
            policy.add_entry(f"hash_{i}", token_count=100 * (i + 1))

        for i in range(5):
            policy.is_cached(f"hash_{i}")
        policy.is_cached("nonexistent_1")
        policy.is_cached("nonexistent_2")

        stats = policy.get_stats()
        assert stats["active_entries"] == 5
        assert stats["total_hits"] == 5
        assert stats["total_misses"] == 2
        assert stats["total_additions"] == 5
        assert stats["total_cached_tokens"] == 100 + 200 + 300 + 400 + 500
        assert stats["hit_rate_pct"] == pytest.approx(5 / 7 * 100, abs=0.1)
