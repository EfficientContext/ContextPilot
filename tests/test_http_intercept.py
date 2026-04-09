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
            {
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {"total_tokens": 10},
            }
        )

    def post(self, url, json=None, headers=None):
        self._last_url = url
        self._last_json = json
        self._last_headers = headers
        return self._response

    def get(self, url, headers=None):
        self._last_url = url
        self._last_headers = headers
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
    original_intercept_index = http_mod._intercept_index
    original_states = http_mod._intercept_states.copy()
    http_mod._aiohttp_session = mock_session
    http_mod._infer_api_url = "http://mock-backend:30000"
    http_mod._intercept_index = None  # reset so each test starts fresh
    http_mod._intercept_states.clear()
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        http_mod._aiohttp_session = original_session
        http_mod._infer_api_url = original_url
        http_mod._intercept_index = original_intercept_index
        http_mod._intercept_states.clear()
        http_mod._intercept_states.update(original_states)


# ============================================================================
# Helpers
# ============================================================================


def _warmup(client, path, body):
    """Prime the intercept index so subsequent calls use build_incremental.

    After priming, resets conversation state so the actual test request
    sees a clean slate (but the clustering index remains primed).
    """
    resp = client.post(path, json=body)
    assert resp.status_code == 200
    # Keep _intercept_index primed, but reset conversation tracking.
    http_mod._intercept_states.clear()
    return resp


# Documents with clear clustering signal: two auth docs share words
# (token, authentication, secure, for) while the database doc shares none.
# schedule_only produces [0, 2, 1] — auth docs grouped, database last.
_AUTH_DOC = "JWT token validation and rotation policy for secure authentication"
_DB_DOC = "Database connection pooling sharding replication backup strategy"
_OAUTH_DOC = "OAuth2 authentication token refresh flow for secure login sessions"


# ============================================================================
# Full intercept flow
# ============================================================================


class TestOpenAIIntercept:
    def test_first_request_builds_index_no_reorder(self, client, mock_session):
        """First request builds index but does NOT reorder (index empty, 1 context)."""
        body = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"<documents>\n"
                        f"<document>{_AUTH_DOC}</document>\n"
                        f"<document>{_DB_DOC}</document>\n"
                        f"<document>{_OAUTH_DOC}</document>\n"
                        f"</documents>"
                    ),
                },
                {"role": "user", "content": "Summarize the documents."},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        assert mock_session._last_url == "http://mock-backend:30000/v1/chat/completions"
        # First request: index empty → no reorder, no header.
        meta = _cp_meta(resp)
        assert meta == {}
        # Body forwarded unmodified.
        forwarded = mock_session._last_json
        sys_content = forwarded["messages"][0]["content"]
        assert "<document>" in sys_content

    def test_second_session_may_reorder(self, client, mock_session):
        """After index is built (session 1), session 2 uses build_incremental."""
        body = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "<documents>\n"
                        "<document>JWT token validation and rotation policy for secure authentication</document>\n"
                        "<document>Database connection pooling sharding replication backup strategy</document>\n"
                        "<document>OAuth2 authentication token refresh flow for secure login sessions</document>\n"
                        "</documents>"
                    ),
                },
                {"role": "user", "content": "Summarize."},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        # Body is forwarded (may or may not be reordered depending on index state).
        assert "_contextpilot" not in resp.json()
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
        chunks = [b'data: {"id":"1"}\n\n', b"data: [DONE]\n\n"]
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
    def test_openai_tool_result_forwarded(self, client, mock_session):
        """OpenAI tool results with docs are extracted and forwarded."""
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "Search for X"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc1",
                    "content": (
                        f"<documents>\n"
                        f"<document>{_AUTH_DOC}</document>\n"
                        f"<document>{_DB_DOC}</document>\n"
                        f"<document>{_OAUTH_DOC}</document>\n"
                        f"</documents>"
                    ),
                },
                {"role": "user", "content": "Now summarize."},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        forwarded = mock_session._last_json
        tool_content = forwarded["messages"][3]["content"]
        assert "<document>" in tool_content

    def test_anthropic_tool_result_forwarded(self, client, mock_session):
        """Anthropic tool_result content blocks are extracted and forwarded."""
        body = {
            "model": "claude-3-opus-20240229",
            "system": "You are a helper.",
            "messages": [
                {"role": "user", "content": "Search for X"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu1",
                            "name": "search",
                            "input": {"query": "X"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu1",
                            "content": (
                                f"<documents>\n"
                                f"<document>{_AUTH_DOC}</document>\n"
                                f"<document>{_DB_DOC}</document>\n"
                                f"<document>{_OAUTH_DOC}</document>\n"
                                f"</documents>"
                            ),
                        },
                    ],
                },
                {"role": "user", "content": "Now summarize."},
            ],
        }
        resp = client.post("/v1/messages", json=body)
        assert resp.status_code == 200
        forwarded = mock_session._last_json
        tr_content = forwarded["messages"][2]["content"][0]["content"]
        assert "<document>" in tr_content

    def test_openai_json_tool_result_forwarded(self, client, mock_session):
        """OpenClaw-style JSON tool results are forwarded correctly."""
        import json as _json

        results = [
            {"path": "auth.md", "snippet": _AUTH_DOC, "score": 0.9},
            {"path": "db.md", "snippet": _DB_DOC, "score": 0.8},
            {"path": "oauth.md", "snippet": _OAUTH_DOC, "score": 0.7},
        ]
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "Search for auth"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "type": "function",
                            "function": {"name": "memory_search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc1",
                    "content": _json.dumps(
                        {"results": results, "citations": "auto"}, indent=2
                    ),
                },
                {"role": "user", "content": "Now summarize."},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        forwarded = mock_session._last_json
        tool_content = _json.loads(forwarded["messages"][3]["content"])
        assert "results" in tool_content
        assert len(tool_content["results"]) == 3
        assert tool_content["citations"] == "auto"

    def test_anthropic_json_tool_result_forwarded(self, client, mock_session):
        """Anthropic-format JSON tool results are forwarded correctly."""
        import json as _json

        results = [
            {"title": "Auth guide", "url": "https://a.com", "description": _AUTH_DOC},
            {"title": "DB guide", "url": "https://b.com", "description": _DB_DOC},
            {"title": "OAuth guide", "url": "https://c.com", "description": _OAUTH_DOC},
        ]
        body = {
            "model": "claude-3-opus-20240229",
            "system": "You are a helper.",
            "messages": [
                {"role": "user", "content": "Search the web"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu1",
                            "name": "web_search",
                            "input": {"query": "auth"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu1",
                            "content": _json.dumps(
                                {"results": results, "provider": "brave"}, indent=2
                            ),
                        },
                    ],
                },
                {"role": "user", "content": "Summarize."},
            ],
        }
        resp = client.post("/v1/messages", json=body)
        assert resp.status_code == 200
        forwarded = mock_session._last_json
        tr_content = _json.loads(forwarded["messages"][2]["content"][0]["content"])
        assert "results" in tr_content
        assert len(tr_content["results"]) == 3
        assert tr_content["provider"] == "brave"


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


# ============================================================================
# Conversation-aware intercept (skip old, dedup new)
# ============================================================================


# Distinct document sets for multi-turn dedup testing.
_DOC_CACHE = "Redis cache invalidation strategy with TTL and LRU eviction"
_DOC_DEPLOY = "Kubernetes deployment rolling update blue green canary strategy"


class TestConversationAwareIntercept:
    """Tests for multi-turn skip / dedup / reorder behaviour."""

    def _make_body(self, system, tool_results=None):
        """Build an OpenAI chat body with optional tool result messages."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Hello"},
        ]
        if tool_results:
            for i, content in enumerate(tool_results):
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"tc{i}",
                                "type": "function",
                                "function": {"name": "search", "arguments": "{}"},
                            }
                        ],
                    }
                )
                messages.append(
                    {"role": "tool", "tool_call_id": f"tc{i}", "content": content}
                )
                messages.append({"role": "user", "content": f"Follow-up {i}"})
        return {"model": "gpt-4", "messages": messages}

    def test_old_tool_result_skipped_on_second_turn(self, client, mock_session):
        tool_content = (
            f"<documents>\n"
            f"<document>{_AUTH_DOC}</document>\n"
            f"<document>{_DB_DOC}</document>\n"
            f"<document>{_OAUTH_DOC}</document>\n"
            f"</documents>"
        )
        system = "You are a helpful assistant."
        # Turn 1: first tool result, builds index (no reorder on first call).
        body1 = self._make_body(system, [tool_content])
        resp1 = client.post("/v1/chat/completions", json=body1)
        assert resp1.status_code == 200

        # Turn 2: same body again — tool result should be skipped.
        resp2 = client.post("/v1/chat/completions", json=body1)
        assert resp2.status_code == 200
        meta2 = _cp_meta(resp2)
        assert meta2 == {}
        forwarded = mock_session._last_json
        assert forwarded["messages"][3]["content"] == tool_content

    def test_new_tool_result_processed_old_skipped(self, client, mock_session):
        system = "You are a helpful assistant."
        tool1 = (
            f"<documents>\n"
            f"<document>{_AUTH_DOC}</document>\n"
            f"<document>{_DB_DOC}</document>\n"
            f"<document>{_OAUTH_DOC}</document>\n"
            f"</documents>"
        )
        # New tool result with 3 distinct docs for reliable clustering.
        tool2 = (
            f"<documents>\n"
            f"<document>{_DOC_CACHE}</document>\n"
            f"<document>{_DOC_DEPLOY}</document>\n"
            f"<document>API rate limiting throttling circuit breaker backpressure</document>\n"
            f"</documents>"
        )
        # Turn 1: one tool result.
        body1 = self._make_body(system, [tool1])
        resp1 = client.post("/v1/chat/completions", json=body1)
        assert resp1.status_code == 200

        # Turn 2: old tool result + new one.
        body2 = self._make_body(system, [tool1, tool2])
        resp2 = client.post("/v1/chat/completions", json=body2)
        assert resp2.status_code == 200
        meta2 = _cp_meta(resp2)
        assert meta2 == {}
        forwarded = mock_session._last_json
        tool2_forwarded = forwarded["messages"][6]["content"]
        assert tool2_forwarded.count("<document>") == 3
        assert _DOC_CACHE in tool2_forwarded
        assert _DOC_DEPLOY in tool2_forwarded

    def test_cross_tool_result_dedup(self, client, mock_session):
        """Documents seen in a previous tool result get deduped in new ones."""
        system = "You are a helpful assistant."
        tool1 = (
            f"<documents>\n"
            f"<document>{_AUTH_DOC}</document>\n"
            f"<document>{_DB_DOC}</document>\n"
            f"<document>{_OAUTH_DOC}</document>\n"
            f"</documents>"
        )
        # Second search returns 2 OLD docs + 2 NEW docs.
        tool2 = (
            f"<documents>\n"
            f"<document>{_AUTH_DOC}</document>\n"
            f"<document>{_OAUTH_DOC}</document>\n"
            f"<document>{_DOC_CACHE}</document>\n"
            f"<document>{_DOC_DEPLOY}</document>\n"
            f"</documents>"
        )
        # Turn 1
        body1 = self._make_body(system, [tool1])
        resp1 = client.post("/v1/chat/completions", json=body1)
        assert resp1.status_code == 200

        # Turn 2 with overlapping docs
        body2 = self._make_body(system, [tool1, tool2])
        resp2 = client.post("/v1/chat/completions", json=body2)
        assert resp2.status_code == 200
        meta2 = _cp_meta(resp2)
        assert meta2.get("intercepted") is True
        assert (
            meta2.get("documents_deduplicated", 0) == 2
        )  # AUTH + OAUTH deduped in tool2

    def test_system_prompt_processed_once(self, client, mock_session):
        """System prompt docs are only processed on the first turn.

        First turn: index empty → no reorder (system_processed set to True).
        Second turn: system_processed=True → skipped.
        """
        system = (
            f"<documents>\n"
            f"<document>{_AUTH_DOC}</document>\n"
            f"<document>{_DB_DOC}</document>\n"
            f"<document>{_OAUTH_DOC}</document>\n"
            f"</documents>"
        )
        body = self._make_body(system)
        # Turn 1: system processed (no reorder, but flag set).
        resp1 = client.post("/v1/chat/completions", json=body)
        assert resp1.status_code == 200

        # Turn 2: system NOT re-processed.
        resp2 = client.post("/v1/chat/completions", json=body)
        assert resp2.status_code == 200
        meta2 = _cp_meta(resp2)
        assert meta2 == {} or meta2.get("sources", {}).get("system", 0) == 0

    def test_new_session_resets_state(self, client, mock_session):
        """A shorter messages array (new session) resets all intercept state."""
        system = "You are a helpful assistant."
        tool_content = (
            f"<documents>\n"
            f"<document>{_AUTH_DOC}</document>\n"
            f"<document>{_DB_DOC}</document>\n"
            f"<document>{_OAUTH_DOC}</document>\n"
            f"</documents>"
        )
        tool_overlap = (
            f"<documents>\n"
            f"<document>{_AUTH_DOC}</document>\n"
            f"<document>{_OAUTH_DOC}</document>\n"
            f"<document>{_DOC_CACHE}</document>\n"
            f"<document>{_DOC_DEPLOY}</document>\n"
            f"</documents>"
        )
        # Session 1: long conversation with tool result.
        body_long = self._make_body(system, [tool_content])
        resp1 = client.post("/v1/chat/completions", json=body_long)
        assert resp1.status_code == 200

        body_overlap = self._make_body(system, [tool_content, tool_overlap])
        resp2 = client.post("/v1/chat/completions", json=body_overlap)
        meta2 = _cp_meta(resp2)
        assert meta2.get("documents_deduplicated", 0) == 2

        # Session 2: shorter messages → triggers state reset.
        body_short = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "Fresh session"},
            ],
        }
        client.post("/v1/chat/completions", json=body_short)

        resp3 = client.post(
            "/v1/chat/completions", json=self._make_body(system, [tool_overlap])
        )
        assert resp3.status_code == 200
        meta3 = _cp_meta(resp3)
        assert meta3.get("documents_deduplicated", 0) == 0
        forwarded = mock_session._last_json
        tool3_content = forwarded["messages"][3]["content"]
        assert _AUTH_DOC in tool3_content
        assert _OAUTH_DOC in tool3_content

    def test_json_tool_result_dedup(self, client, mock_session):
        """JSON results format also gets cross-turn dedup.

        Note: dedup compares the full serialised JSON document string, so
        entries must be byte-identical to be considered duplicates.
        """
        import json as _json

        system = "You are a helper."
        # Shared entry — identical dict so serialised form matches.
        auth_entry = {"path": "auth.md", "snippet": _AUTH_DOC, "score": 0.9}
        results1 = [
            auth_entry,
            {"path": "db.md", "snippet": _DB_DOC, "score": 0.8},
            {"path": "oauth.md", "snippet": _OAUTH_DOC, "score": 0.7},
        ]
        results2 = [
            auth_entry,  # exact duplicate
            {"path": "cache.md", "snippet": _DOC_CACHE, "score": 0.85},
            {"path": "deploy.md", "snippet": _DOC_DEPLOY, "score": 0.75},
        ]
        tool1 = _json.dumps({"results": results1}, indent=2)
        tool2 = _json.dumps({"results": results2}, indent=2)

        # Turn 1
        body1 = self._make_body(system, [tool1])
        resp1 = client.post("/v1/chat/completions", json=body1)
        assert resp1.status_code == 200

        # Turn 2: old tool + new tool with overlapping auth entry.
        body2 = self._make_body(system, [tool1, tool2])
        resp2 = client.post("/v1/chat/completions", json=body2)
        assert resp2.status_code == 200
        meta2 = _cp_meta(resp2)
        assert meta2.get("intercepted") is True
        assert (
            meta2.get("documents_deduplicated", 0) == 1
        )  # auth entry deduped in tool2
        forwarded = mock_session._last_json
        tool2_forwarded = _json.loads(forwarded["messages"][6]["content"])
        assert len(tool2_forwarded["results"]) == 2


# ============================================================================
# External content marker stripping
# ============================================================================


class TestExternalContentIdStripping:
    """EXTERNAL_UNTRUSTED_CONTENT random ids are stripped before forwarding."""

    def _wrap(self, text, marker_id="ab12cd34"):
        """Simulate OpenClaw's wrapWebContent."""
        return (
            f'\n<<<EXTERNAL_UNTRUSTED_CONTENT id="{marker_id}">>>\n'
            f"Source: Web Search\n---\n{text}\n"
            f'<<<END_EXTERNAL_UNTRUSTED_CONTENT id="{marker_id}">>>'
        )

    def test_ids_stripped_from_forwarded_body(self, client, mock_session):
        """Random marker ids are removed so identical content shares KV prefix."""
        wrapped_title = self._wrap("Example Title", "aabbccdd11223344")
        wrapped_desc = self._wrap("Example description text", "1122334455667788")
        import json as _json

        results = [
            {
                "title": wrapped_title,
                "url": "https://a.com",
                "description": wrapped_desc,
            },
        ]
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "Search"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc1",
                    "content": _json.dumps({"results": results}),
                },
                {"role": "user", "content": "Summarize"},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        forwarded = mock_session._last_json
        tool_content = forwarded["messages"][3]["content"]
        # Random ids should be stripped
        assert 'id="aabbccdd11223344"' not in tool_content
        assert 'id="1122334455667788"' not in tool_content
        # Markers themselves preserved (without id)
        assert "<<<EXTERNAL_UNTRUSTED_CONTENT>>>" in tool_content
        assert "<<<END_EXTERNAL_UNTRUSTED_CONTENT>>>" in tool_content

    def test_different_ids_produce_same_forwarded_content(self, client, mock_session):
        """Two requests with different random ids produce identical forwarded content."""
        import json as _json

        def _make_body(marker_id):
            wrapped = self._wrap("Same content", marker_id)
            results = [
                {"title": wrapped, "url": "https://a.com", "description": "plain"}
            ]
            return {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a helper."},
                    {"role": "user", "content": "Hello"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc1",
                                "type": "function",
                                "function": {"name": "s", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "tc1",
                        "content": _json.dumps({"results": results}),
                    },
                    {"role": "user", "content": "Go"},
                ],
            }

        # Request 1 with id "aaaa"
        resp1 = client.post("/v1/chat/completions", json=_make_body("aaaa0000bbbb1111"))
        assert resp1.status_code == 200
        content1 = mock_session._last_json["messages"][3]["content"]

        http_mod._intercept_states.clear()

        # Request 2 with different id "bbbb"
        resp2 = client.post("/v1/chat/completions", json=_make_body("cccc2222dddd3333"))
        assert resp2.status_code == 200
        content2 = mock_session._last_json["messages"][3]["content"]

        assert content1 == content2
