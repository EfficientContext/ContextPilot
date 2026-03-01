"""Integration tests for HTTP intercept proxy endpoints.

Uses httpx AsyncClient with FastAPI's TestClient (no real server needed)
and patches aiohttp to mock the backend LLM server.
"""

import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from contextpilot.server.http_server import app, _init_config

import contextpilot.server.http_server as http_mod


# ============================================================================
# Fixtures
# ============================================================================


class FakeHeaders:
    """Fake multidict-like headers for aiohttp responses."""

    def __init__(self, headers=None):
        self._headers = headers or {"content-type": "application/json"}

    def items(self):
        return self._headers.items()

    def get(self, key, default=None):
        return self._headers.get(key, default)


class FakeResponse:
    """Fake aiohttp response for mocking."""

    def __init__(self, json_body, status=200):
        self._json = json_body
        self.status = status
        self.content = FakeStreamContent()
        self.headers = FakeHeaders()

    async def json(self):
        return self._json

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeStreamContent:
    """Fake async iterator for streaming response."""

    def __init__(self, chunks=None):
        self._chunks = chunks or [b"data: {}\n\n", b"data: [DONE]\n\n"]

    async def iter_any(self):
        for c in self._chunks:
            yield c


class FakeStreamResponse(FakeResponse):
    """Fake response that yields SSE chunks."""

    def __init__(self, chunks=None, status=200):
        super().__init__({}, status)
        self.content = FakeStreamContent(chunks)
        self.headers = FakeHeaders({"content-type": "text/event-stream"})


class FakeSession:
    """Fake aiohttp.ClientSession."""

    def __init__(self, response=None):
        self._response = response or FakeResponse(
            {"choices": [{"message": {"content": "Hello"}}], "usage": {"total_tokens": 10}}
        )

    def post(self, url, json=None, headers=None):
        self._last_url = url
        self._last_json = json
        self._last_headers = headers
        return self._response

    def get(self, url):
        self._last_url = url
        return self._response


def _cp_meta(resp):
    """Extract ContextPilot metadata from the X-ContextPilot-Result response header."""
    raw = resp.headers.get("x-contextpilot-result")
    if raw is None:
        return {}
    return json.loads(raw)


@pytest.fixture
def mock_session():
    """Provide a FakeSession and patch it into http_server globals."""
    session = FakeSession()
    return session


@pytest.fixture
def client(mock_session):
    """FastAPI test client with mocked backend."""
    # Patch the module-level globals
    original_session = http_mod._aiohttp_session
    original_url = http_mod._infer_api_url
    http_mod._aiohttp_session = mock_session
    http_mod._infer_api_url = "http://mock-backend:30000"
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        http_mod._aiohttp_session = original_session
        http_mod._infer_api_url = original_url


# ============================================================================
# Full intercept flow
# ============================================================================


class TestOpenAIIntercept:
    def test_basic_intercept_reorders(self, client, mock_session):
        """Full intercept: docs extracted, reordered, forwarded."""
        body = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "<documents>\n"
                        "<document>Doc A content</document>\n"
                        "<document>Doc B content</document>\n"
                        "<document>Doc C content</document>\n"
                        "</documents>"
                    ),
                },
                {"role": "user", "content": "Summarize the documents."},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        # Backend was called
        assert mock_session._last_url == "http://mock-backend:30000/v1/chat/completions"
        # Metadata in response header (not body)
        meta = _cp_meta(resp)
        assert meta.get("intercepted") is True
        assert "_contextpilot" not in resp.json()
        # The forwarded body should have reordered docs (still in XML format)
        forwarded = mock_session._last_json
        sys_content = forwarded["messages"][0]["content"]
        assert "<document>" in sys_content
        assert "</document>" in sys_content

    def test_bypass_when_disabled(self, client, mock_session):
        """When X-ContextPilot-Enabled: false, forward unmodified."""
        body = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": "<documents><document>A</document><document>B</document></documents>",
                },
                {"role": "user", "content": "Hello"},
            ],
        }
        resp = client.post(
            "/v1/chat/completions",
            json=body,
            headers={"X-ContextPilot-Enabled": "false"},
        )
        assert resp.status_code == 200
        # Body forwarded unmodified
        forwarded = mock_session._last_json
        assert forwarded["messages"][0]["content"] == body["messages"][0]["content"]
        # No contextpilot metadata
        assert _cp_meta(resp) == {}

    def test_bypass_when_no_docs(self, client, mock_session):
        """When system message has no extractable docs, forward unmodified."""
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        forwarded = mock_session._last_json
        assert forwarded["messages"][0]["content"] == "You are a helpful assistant."

    def test_bypass_when_single_doc(self, client, mock_session):
        """Single doc => nothing to reorder, forward unmodified."""
        body = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": "<documents><document>Only one</document></documents>",
                },
                {"role": "user", "content": "Hello"},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        assert _cp_meta(resp) == {}

    def test_numbered_format(self, client, mock_session):
        """Numbered format extraction and forwarding."""
        body = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": "[1] First document [2] Second document [3] Third document",
                },
                {"role": "user", "content": "Summarize"},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        forwarded = mock_session._last_json
        sys_content = forwarded["messages"][0]["content"]
        # Should contain numbered items (reordered)
        assert "[1]" in sys_content
        assert "[2]" in sys_content


class TestAnthropicIntercept:
    def test_basic_intercept(self, client, mock_session):
        body = {
            "model": "claude-3-opus-20240229",
            "system": "<documents><document>A</document><document>B</document><document>C</document></documents>",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        resp = client.post("/v1/messages", json=body)
        assert resp.status_code == 200
        assert mock_session._last_url == "http://mock-backend:30000/v1/messages"
        forwarded = mock_session._last_json
        assert "<document>" in forwarded["system"]

    def test_bypass_no_system(self, client, mock_session):
        body = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        resp = client.post("/v1/messages", json=body)
        assert resp.status_code == 200


# ============================================================================
# Streaming
# ============================================================================


class TestStreaming:
    def test_streaming_passthrough(self, mock_session):
        """Streaming responses are passed through."""
        chunks = [b"data: {\"id\":\"1\"}\n\n", b"data: [DONE]\n\n"]
        stream_resp = FakeStreamResponse(chunks)
        session = FakeSession(stream_resp)

        original_session = http_mod._aiohttp_session
        original_url = http_mod._infer_api_url
        http_mod._aiohttp_session = session
        http_mod._infer_api_url = "http://mock-backend:30000"
        try:
            client = TestClient(app, raise_server_exceptions=False)
            body = {
                "model": "gpt-4",
                "stream": True,
                "messages": [
                    {
                        "role": "system",
                        "content": "<documents><document>A</document><document>B</document></documents>",
                    },
                    {"role": "user", "content": "Hello"},
                ],
            }
            resp = client.post("/v1/chat/completions", json=body)
            assert resp.status_code == 200
            # Streaming response content
            content = resp.content
            assert b"data:" in content
        finally:
            http_mod._aiohttp_session = original_session
            http_mod._infer_api_url = original_url


# ============================================================================
# Catch-all still works
# ============================================================================


class TestCatchAllProxy:
    def test_other_v1_paths_still_proxied(self, client, mock_session):
        """Other /v1/* paths still go to the catch-all proxy."""
        mock_session._response = FakeResponse({"models": [{"id": "gpt-4"}]})
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert mock_session._last_url == "http://mock-backend:30000/v1/models"

    def test_completions_still_works(self, client, mock_session):
        """POST /v1/completions still handled by existing proxy."""
        mock_session._response = FakeResponse(
            {"choices": [{"text": "hello"}], "usage": {"total_tokens": 5}}
        )
        body = {"model": "gpt-4", "prompt": "Hello"}
        resp = client.post("/v1/completions", json=body)
        assert resp.status_code == 200

    def test_completions_metadata_in_header(self, client, mock_session):
        """proxy_completions puts metadata in X-ContextPilot-Result header, not body."""
        mock_session._response = FakeResponse(
            {"choices": [{"text": "hello"}], "usage": {"total_tokens": 5}}
        )
        body = {"model": "gpt-4", "prompt": "Hello", "request_id": "req-abc123"}
        resp = client.post("/v1/completions", json=body)
        assert resp.status_code == 200
        # Metadata should be in header
        meta = _cp_meta(resp)
        assert meta.get("request_id") == "req-abc123"
        assert meta.get("tokens_reported") == 5
        # Body should NOT contain _contextpilot
        assert "_contextpilot" not in resp.json()


# ============================================================================
# Header forwarding
# ============================================================================


class TestHeaderForwarding:
    def test_auth_headers_forwarded(self, client, mock_session):
        """Authorization headers are forwarded, X-ContextPilot-* stripped."""
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "Plain text."},
                {"role": "user", "content": "Hello"},
            ],
        }
        resp = client.post(
            "/v1/chat/completions",
            json=body,
            headers={
                "Authorization": "Bearer sk-test",
                "X-ContextPilot-Mode": "auto",
            },
        )
        assert resp.status_code == 200
        outbound = mock_session._last_headers or {}
        # Auth should be forwarded
        assert outbound.get("authorization") or outbound.get("Authorization")
        # X-ContextPilot-* should be stripped
        for k in outbound:
            assert not k.lower().startswith("x-contextpilot-")


# ============================================================================
# Tool result intercept
# ============================================================================


class TestToolResultIntercept:
    def test_openai_tool_result_reordered(self, client, mock_session):
        """OpenAI tool results with docs get reordered."""
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "Search for X"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "tc1", "type": "function",
                                 "function": {"name": "search", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "tc1",
                 "content": (
                     "<documents>\n"
                     "<document>Tool Doc A</document>\n"
                     "<document>Tool Doc B</document>\n"
                     "<document>Tool Doc C</document>\n"
                     "</documents>"
                 )},
                {"role": "user", "content": "Now summarize."},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        meta = _cp_meta(resp)
        assert meta.get("intercepted") is True
        assert meta.get("documents_reordered") is True
        assert meta.get("sources", {}).get("tool_results", 0) >= 1
        # Forwarded body should have reordered tool content
        forwarded = mock_session._last_json
        tool_content = forwarded["messages"][3]["content"]
        assert "<document>" in tool_content

    def test_anthropic_tool_result_reordered(self, client, mock_session):
        """Anthropic tool_result content blocks get reordered."""
        body = {
            "model": "claude-3-opus-20240229",
            "system": "You are a helper.",
            "messages": [
                {"role": "user", "content": "Search for X"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu1", "name": "search",
                     "input": {"query": "X"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu1",
                     "content": (
                         "<documents>\n"
                         "<document>Anthropic Doc A</document>\n"
                         "<document>Anthropic Doc B</document>\n"
                         "<document>Anthropic Doc C</document>\n"
                         "</documents>"
                     )},
                ]},
                {"role": "user", "content": "Now summarize."},
            ],
        }
        resp = client.post("/v1/messages", json=body)
        assert resp.status_code == 200
        meta = _cp_meta(resp)
        assert meta.get("intercepted") is True
        assert meta.get("sources", {}).get("tool_results", 0) >= 1
        # Forwarded body should have reordered tool_result content
        forwarded = mock_session._last_json
        tr_content = forwarded["messages"][2]["content"][0]["content"]
        assert "<document>" in tr_content

    def test_openai_system_and_tool_results(self, client, mock_session):
        """Both system docs and tool result docs get reordered."""
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": (
                    "<documents>\n"
                    "<document>Sys A</document>\n"
                    "<document>Sys B</document>\n"
                    "<document>Sys C</document>\n"
                    "</documents>"
                )},
                {"role": "user", "content": "Search"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "tc1"}]},
                {"role": "tool", "tool_call_id": "tc1",
                 "content": "[1] Tool A [2] Tool B [3] Tool C"},
                {"role": "user", "content": "Summarize"},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        meta = _cp_meta(resp)
        assert meta.get("intercepted") is True
        assert meta["sources"]["system"] == 1
        assert meta["sources"]["tool_results"] == 1
        assert meta["total_documents"] == 6

    def test_scope_system_skips_tool_results(self, client, mock_session):
        """scope=system should only reorder system docs, not tool results."""
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": (
                    "<documents>\n"
                    "<document>Sys A</document>\n"
                    "<document>Sys B</document>\n"
                    "</documents>"
                )},
                {"role": "tool", "tool_call_id": "tc1",
                 "content": "<documents><document>T1</document><document>T2</document></documents>"},
            ],
        }
        resp = client.post(
            "/v1/chat/completions",
            json=body,
            headers={"X-ContextPilot-Scope": "system"},
        )
        assert resp.status_code == 200
        meta = _cp_meta(resp)
        assert meta.get("intercepted") is True
        assert meta["sources"]["system"] == 1
        assert meta["sources"]["tool_results"] == 0

    def test_scope_tool_results_skips_system(self, client, mock_session):
        """scope=tool_results should only reorder tool result docs."""
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": (
                    "<documents>\n"
                    "<document>Sys A</document>\n"
                    "<document>Sys B</document>\n"
                    "</documents>"
                )},
                {"role": "tool", "tool_call_id": "tc1",
                 "content": "<documents><document>T1</document><document>T2</document></documents>"},
            ],
        }
        resp = client.post(
            "/v1/chat/completions",
            json=body,
            headers={"X-ContextPilot-Scope": "tool_results"},
        )
        assert resp.status_code == 200
        meta = _cp_meta(resp)
        assert meta.get("intercepted") is True
        assert meta["sources"]["system"] == 0
        assert meta["sources"]["tool_results"] == 1


# ============================================================================
# RID injection in intercept (stateful mode)
# ============================================================================


class TestInterceptRidInjection:
    def test_rid_injected_in_stateful_mode(self, mock_session):
        """In stateful mode, intercept injects rid into forwarded body."""
        original_session = http_mod._aiohttp_session
        original_url = http_mod._infer_api_url
        original_stateless = http_mod._stateless_mode
        original_index = http_mod._index
        http_mod._aiohttp_session = mock_session
        http_mod._infer_api_url = "http://mock-backend:30000"
        http_mod._stateless_mode = False
        http_mod._index = MagicMock()  # non-None → stateful
        try:
            client = TestClient(app, raise_server_exceptions=False)
            body = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "Plain text."},
                    {"role": "user", "content": "Hello"},
                ],
            }
            resp = client.post("/v1/chat/completions", json=body)
            assert resp.status_code == 200
            forwarded = mock_session._last_json
            assert "rid" in forwarded
            assert forwarded["rid"].startswith("req-")
        finally:
            http_mod._aiohttp_session = original_session
            http_mod._infer_api_url = original_url
            http_mod._stateless_mode = original_stateless
            http_mod._index = original_index

    def test_no_rid_in_stateless_mode(self, client, mock_session):
        """In stateless mode (default for tests), no rid is injected."""
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "Plain text."},
                {"role": "user", "content": "Hello"},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        forwarded = mock_session._last_json
        assert "rid" not in forwarded
