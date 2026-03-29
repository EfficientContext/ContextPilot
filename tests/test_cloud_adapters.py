"""Tests for cloud provider adapters."""

import copy

import pytest

from contextpilot.server.cloud_adapters import (
    get_cloud_adapter,
    AnthropicAdapter,
    OpenAIAdapter,
    MiniMaxAdapter,
    CacheMetrics,
    TTLTier,
)


class TestGetCloudAdapter:
    def test_anthropic(self):
        adapter = get_cloud_adapter("anthropic")
        assert isinstance(adapter, AnthropicAdapter)

    def test_openai(self):
        adapter = get_cloud_adapter("openai")
        assert isinstance(adapter, OpenAIAdapter)

    def test_minimax(self):
        adapter = get_cloud_adapter("minimax")
        assert isinstance(adapter, MiniMaxAdapter)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown cloud provider"):
            get_cloud_adapter("unknown")


class TestAnthropicAdapter:
    @pytest.fixture
    def adapter(self):
        return AnthropicAdapter()

    def test_provider_name(self, adapter):
        assert adapter.provider_name == "anthropic"

    def test_api_url(self, adapter):
        assert (
            adapter.get_api_url("/v1/messages")
            == "https://api.anthropic.com/v1/messages"
        )

    def test_target_path(self, adapter):
        assert adapter.get_target_path() == "/v1/messages"

    def test_default_ttl_seconds(self, adapter):
        assert adapter.get_default_ttl_seconds() == 300

    def test_extended_ttl_seconds(self, adapter):
        assert adapter.get_extended_ttl_seconds() == 3600

    def test_auth_headers(self, adapter):
        headers = adapter.get_auth_headers("sk-ant-test-key")
        assert headers["x-api-key"] == "sk-ant-test-key"
        assert "anthropic-version" in headers
        assert headers["content-type"] == "application/json"

    def test_inject_cache_control_string_system(self, adapter):
        body = {
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = adapter.inject_cache_control(body, set())
        assert isinstance(result["system"], list)
        assert result["system"][0]["type"] == "text"
        assert result["system"][0]["text"] == "You are a helpful assistant."
        assert result["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_inject_cache_control_list_system(self, adapter):
        body = {
            "system": [
                {"type": "text", "text": "First block"},
                {"type": "text", "text": "Second block"},
            ],
            "messages": [],
        }
        result = adapter.inject_cache_control(body, set())
        assert "cache_control" not in result["system"][0]
        assert result["system"][1]["cache_control"] == {"type": "ephemeral"}

    def test_inject_cache_control_no_system(self, adapter):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        result = adapter.inject_cache_control(body, set())
        assert "system" not in result

    def test_inject_cache_control_tool_result_large(self, adapter):
        large_content = "x" * 2000
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": large_content,
                        }
                    ],
                }
            ]
        }
        result = adapter.inject_cache_control(body, set())
        tool_block = result["messages"][0]["content"][0]
        assert tool_block["cache_control"] == {"type": "ephemeral"}

    def test_inject_cache_control_tool_result_small_unchanged(self, adapter):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "short",
                        }
                    ],
                }
            ]
        }
        result = adapter.inject_cache_control(body, set())
        tool_block = result["messages"][0]["content"][0]
        assert "cache_control" not in tool_block

    def test_inject_does_not_mutate_original(self, adapter):
        body = {
            "system": "test",
            "messages": [{"role": "user", "content": "hi"}],
        }
        original = copy.deepcopy(body)
        adapter.inject_cache_control(body, set())
        assert body == original

    def test_parse_cache_metrics(self, adapter):
        response = {
            "usage": {
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 500,
                "input_tokens": 1500,
                "output_tokens": 200,
            }
        }
        metrics = adapter.parse_cache_metrics(response)
        assert metrics.cache_creation_tokens == 1000
        assert metrics.cache_read_tokens == 500
        assert metrics.input_tokens == 1500
        assert metrics.output_tokens == 200

    def test_parse_cache_metrics_empty_usage(self, adapter):
        metrics = adapter.parse_cache_metrics({})
        assert metrics.cache_creation_tokens == 0
        assert metrics.cache_read_tokens == 0


class TestOpenAIAdapter:
    @pytest.fixture
    def adapter(self):
        return OpenAIAdapter()

    def test_provider_name(self, adapter):
        assert adapter.provider_name == "openai"

    def test_api_url(self, adapter):
        assert (
            adapter.get_api_url("/v1/chat/completions")
            == "https://api.openai.com/v1/chat/completions"
        )

    def test_target_path(self, adapter):
        assert adapter.get_target_path() == "/v1/chat/completions"

    def test_default_ttl_seconds(self, adapter):
        assert adapter.get_default_ttl_seconds() == 3600

    def test_extended_ttl_seconds(self, adapter):
        assert adapter.get_extended_ttl_seconds() == 86400

    def test_auth_headers(self, adapter):
        headers = adapter.get_auth_headers("sk-test-key")
        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["Content-Type"] == "application/json"

    def test_inject_no_retention_by_default(self, adapter):
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = adapter.inject_cache_control(body, set())
        assert "prompt_cache_retention" not in result

    def test_inject_24h_when_extended(self, adapter):
        adapter.configured_ttl = TTLTier.LONG
        body = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = adapter.inject_cache_control(body, set())
        assert result["prompt_cache_retention"] == "24h"

    def test_parse_cache_metrics_with_cached_tokens(self, adapter):
        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 800},
            }
        }
        metrics = adapter.parse_cache_metrics(response)
        assert metrics.cache_read_tokens == 800
        assert metrics.cache_creation_tokens == 200
        assert metrics.input_tokens == 1000
        assert metrics.output_tokens == 200

    def test_parse_cache_metrics_no_cached(self, adapter):
        response = {"usage": {"prompt_tokens": 500, "completion_tokens": 100}}
        metrics = adapter.parse_cache_metrics(response)
        assert metrics.cache_read_tokens == 0
        assert metrics.cache_creation_tokens == 0
        assert metrics.input_tokens == 500

    def test_parse_cache_metrics_empty(self, adapter):
        metrics = adapter.parse_cache_metrics({})
        assert metrics.cache_read_tokens == 0


class TestMiniMaxAdapter:
    @pytest.fixture
    def adapter(self):
        return MiniMaxAdapter()

    def test_provider_name(self, adapter):
        assert adapter.provider_name == "minimax"

    def test_api_url(self, adapter):
        url = adapter.get_api_url("/v1/messages")
        assert "minimax.io" in url

    def test_target_path(self, adapter):
        assert adapter.get_target_path() == "/v1/messages"

    def test_default_ttl_seconds(self, adapter):
        assert adapter.get_default_ttl_seconds() == 300

    def test_no_extended_cache(self, adapter):
        assert adapter.get_extended_ttl_seconds() is None
        assert not adapter.supports_extended_cache

    def test_auth_headers(self, adapter):
        headers = adapter.get_auth_headers("mm-key-123")
        assert headers["x-api-key"] == "mm-key-123"

    def test_inject_cache_control_same_as_anthropic(self, adapter):
        body = {
            "system": "You are an assistant",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = adapter.inject_cache_control(body, set())
        assert isinstance(result["system"], list)
        assert result["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_inject_does_not_mutate_original(self, adapter):
        body = {"system": "test", "messages": []}
        original = copy.deepcopy(body)
        adapter.inject_cache_control(body, set())
        assert body == original

    def test_parse_cache_metrics(self, adapter):
        response = {
            "usage": {
                "cache_creation_input_tokens": 2000,
                "cache_read_input_tokens": 1000,
                "input_tokens": 3000,
                "output_tokens": 500,
            }
        }
        metrics = adapter.parse_cache_metrics(response)
        assert metrics.cache_creation_tokens == 2000
        assert metrics.cache_read_tokens == 1000
