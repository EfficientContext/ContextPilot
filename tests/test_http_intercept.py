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


class FakeResponse:
    """Fake aiohttp response for mocking."""

    def __init__(self, json_body, status=200):
        self._json = json_body
        self.status = status
        self.content = FakeStreamContent()

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
        result = resp.json()
        # Backend was called
        assert mock_session._last_url == "http://mock-backend:30000/v1/chat/completions"
        # Metadata injected
        assert result.get("_contextpilot", {}).get("intercepted") is True
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
        result = resp.json()
        assert "_contextpilot" not in result

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
        result = resp.json()
        assert "_contextpilot" not in result

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
