import importlib.util
from types import SimpleNamespace
import sys
import types
from pathlib import Path


def _load_plugin_module(monkeypatch):
    agent_pkg = types.ModuleType("agent")
    context_engine_mod = types.ModuleType("agent.context_engine")
    context_compressor_mod = types.ModuleType("agent.context_compressor")

    class FakeContextEngine:
        last_prompt_tokens = 0
        last_completion_tokens = 0
        last_total_tokens = 0
        threshold_tokens = 0
        context_length = 0
        compression_count = 0
        threshold_percent = 0.75
        protect_first_n = 3
        protect_last_n = 6

        def get_status(self):
            return {}

        def on_session_reset(self):
            return None

    class FakeContextCompressor(FakeContextEngine):
        def __init__(self, **kwargs):
            self.threshold_tokens = 0
            self.context_length = kwargs.get("config_context_length") or 0
            self.threshold_percent = 0.75
            self.protect_first_n = 3
            self.protect_last_n = 6
            self.compression_count = 0

        def update_from_response(self, usage):
            self.last_prompt_tokens = usage.get("prompt_tokens", 0)
            self.last_completion_tokens = usage.get("completion_tokens", 0)
            self.last_total_tokens = usage.get("total_tokens", 0)

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(self, messages, current_tokens=None, **kwargs):
            return messages

        def on_session_start(self, session_id, **kwargs):
            return None

        def update_model(self, **kwargs):
            self.context_length = kwargs.get("context_length", self.context_length)
            self.threshold_tokens = int(self.context_length * self.threshold_percent)

        def get_tool_schemas(self):
            return []

        def handle_tool_call(self, name, args, **kwargs):
            return "{}"

    context_engine_mod.ContextEngine = FakeContextEngine
    context_compressor_mod.ContextCompressor = FakeContextCompressor
    agent_pkg.context_engine = context_engine_mod
    agent_pkg.context_compressor = context_compressor_mod

    monkeypatch.setitem(sys.modules, "agent", agent_pkg)
    monkeypatch.setitem(sys.modules, "agent.context_engine", context_engine_mod)
    monkeypatch.setitem(sys.modules, "agent.context_compressor", context_compressor_mod)

    run_agent_mod = types.ModuleType("run_agent")

    class FakeAIAgent:
        @staticmethod
        def _sanitize_api_messages(messages):
            return messages + [{"role": "tool", "content": "sanitized"}]

    run_agent_mod.AIAgent = FakeAIAgent
    monkeypatch.setitem(sys.modules, "run_agent", run_agent_mod)

    module_path = Path(__file__).resolve().parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location("contextpilot_hermes_plugin_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module, run_agent_mod


def test_patch_routes_instance_sanitization_through_contextpilot(monkeypatch):
    module, run_agent_mod = _load_plugin_module(monkeypatch)
    module._patch_hermes_sanitizer()

    engine = module.ContextPilotEngine()
    calls = []

    def optimize(messages, **kwargs):
        calls.append((messages, kwargs))
        return messages + [{"role": "assistant", "content": "optimized"}], {"chars_saved": 1}

    engine.optimize_api_messages = optimize

    agent = run_agent_mod.AIAgent()
    agent.context_compressor = engine
    agent._cached_system_prompt = "system prompt"

    out = agent._sanitize_api_messages([{"role": "user", "content": "hello"}])

    assert out == [
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "sanitized"},
        {"role": "assistant", "content": "optimized"},
    ]
    assert calls == [
        (
            [
                {"role": "user", "content": "hello"},
                {"role": "tool", "content": "sanitized"},
            ],
            {"system_content": "system prompt"},
        )
    ]


def test_patch_preserves_class_level_sanitizer_usage(monkeypatch):
    module, run_agent_mod = _load_plugin_module(monkeypatch)
    module._patch_hermes_sanitizer()

    out = run_agent_mod.AIAgent._sanitize_api_messages([{"role": "user", "content": "hello"}])

    assert out == [
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "sanitized"},
    ]


def test_prefix_replay_matches_original_and_reuses_optimized_prefix(monkeypatch):
    module, _ = _load_plugin_module(monkeypatch)
    monkeypatch.setattr(module, "_check_reorder", lambda: False)
    monkeypatch.setattr(module, "_CONTEXTPILOT_AVAILABLE", False)

    calls = []

    def dedup(body, **kwargs):
        messages = body["messages"]
        calls.append([m.copy() for m in messages])
        saved = 0
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("content") == "FULL TOOL RESULT":
                msg["content"] = "DEDUPED TOOL RESULT"
                saved += len("FULL TOOL RESULT") - len("DEDUPED TOOL RESULT")
        return SimpleNamespace(
            chars_saved=saved,
            blocks_deduped=1 if saved else 0,
            blocks_total=1,
            system_blocks_matched=0,
        )

    monkeypatch.setattr(module, "dedup_chat_completions", dedup)

    engine = module.ContextPilotEngine()
    first = [
        {"role": "user", "content": "read file"},
        {"role": "tool", "tool_call_id": "call_1", "content": "FULL TOOL RESULT"},
    ]
    first_out, _ = engine.optimize_api_messages(first)

    assert first_out[1]["content"] == "DEDUPED TOOL RESULT"
    assert engine._cached_original_messages[1]["content"] == "FULL TOOL RESULT"
    assert engine._cached_messages[1]["content"] == "DEDUPED TOOL RESULT"

    already_optimized = {"messages": [m.copy() for m in first_out]}
    engine._intercept_chat_kwargs(already_optimized)
    assert engine._cached_original_messages[1]["content"] == "FULL TOOL RESULT"
    assert already_optimized["messages"][1]["content"] == "DEDUPED TOOL RESULT"

    second = [
        {"role": "user", "content": "read file"},
        {"role": "tool", "tool_call_id": "call_1", "content": "FULL TOOL RESULT"},
        {"role": "user", "content": "now summarize it"},
    ]
    second_out, _ = engine.optimize_api_messages(second)

    assert second_out[1]["content"] == "DEDUPED TOOL RESULT"
    assert second_out[2]["content"] == "now summarize it"
    assert calls[-1][1]["content"] == "DEDUPED TOOL RESULT"


def _saving_dedup(body, **kwargs):
    saved = 0
    for msg in body["messages"]:
        if msg.get("role") == "tool" and msg.get("content") == "FULL TOOL RESULT":
            msg["content"] = "REF"
            saved += len("FULL TOOL RESULT") - len("REF")
    return SimpleNamespace(
        chars_saved=saved,
        blocks_deduped=1 if saved else 0,
        blocks_total=1,
        system_blocks_matched=0,
    )


def test_optimize_writes_metadata_only_telemetry_line(monkeypatch, tmp_path):
    import json

    module, _ = _load_plugin_module(monkeypatch)
    monkeypatch.setattr(module, "_check_reorder", lambda: False)
    monkeypatch.setattr(module, "_CONTEXTPILOT_AVAILABLE", False)
    monkeypatch.setattr(module, "dedup_chat_completions", _saving_dedup)

    telemetry = tmp_path / "nested" / "telemetry.jsonl"
    monkeypatch.setenv("CONTEXTPILOT_TELEMETRY_FILE", str(telemetry))

    engine = module.ContextPilotEngine()
    engine.on_session_start("session-XYZ", model="test-model")

    secret = "SUPER SECRET USER PROMPT — must never be written to telemetry"
    messages = [
        {"role": "user", "content": secret},
        {"role": "tool", "tool_call_id": "call_1", "content": "FULL TOOL RESULT"},
    ]
    engine.optimize_api_messages(messages)

    assert telemetry.exists()
    lines = [l for l in telemetry.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])

    # Numeric/metadata only — savings recorded.
    assert record["chars_saved"] > 0
    assert record["tokens_saved"] == record["chars_saved"] // 4
    assert record["turn"] == 1
    assert record["session_hash"] == module._hash_text("session-XYZ")
    assert "session" not in record
    assert isinstance(record["ts"], (int, float))

    # Privacy: no message/prompt/tool-payload content may appear anywhere.
    raw = telemetry.read_text(encoding="utf-8")
    assert secret not in raw
    assert "FULL TOOL RESULT" not in raw
    forbidden = {"content", "messages", "prompt", "system_prompt", "text", "tool_calls"}
    assert forbidden.isdisjoint(record.keys())


def test_telemetry_records_payload_chars_and_derived_token_method(monkeypatch, tmp_path):
    """Before/after payload chars are actual; the chars/4 counter is labelled derived."""
    import json

    module, _ = _load_plugin_module(monkeypatch)
    monkeypatch.setattr(module, "_check_reorder", lambda: False)
    monkeypatch.setattr(module, "_CONTEXTPILOT_AVAILABLE", False)
    monkeypatch.setattr(module, "dedup_chat_completions", _saving_dedup)
    # Force the exact tokenizer OFF so this case is deterministic everywhere.
    monkeypatch.setenv("CONTEXTPILOT_DISABLE_EXACT_TOKENIZER", "1")

    telemetry = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("CONTEXTPILOT_TELEMETRY_FILE", str(telemetry))

    engine = module.ContextPilotEngine()
    messages = [
        {"role": "user", "content": "read file"},
        {"role": "tool", "tool_call_id": "call_1", "content": "FULL TOOL RESULT"},
    ]
    _out, stats = engine.optimize_api_messages(messages)

    record = json.loads(telemetry.read_text(encoding="utf-8").splitlines()[0])

    # Actual processed-payload before/after char measurement.
    assert record["payload_chars_before"] > record["payload_chars_after"]
    assert (
        record["payload_chars_saved"]
        == record["payload_chars_before"] - record["payload_chars_after"]
    )
    # The legacy token counter is explicitly tagged as a derived chars/4 estimate.
    assert record["tokens_saved"] == record["chars_saved"] // 4
    assert record["tokens_saved_method"] == "estimated_chars_div_4"
    # No exact tokenizer -> a clear status and NO fabricated token numbers.
    assert record["actual_token_status"] == "unavailable"
    assert "actual_tokens_before" not in record
    assert "actual_tokens_after" not in record
    assert "actual_tokens_saved" not in record
    # Returned stats expose the same payload-char measurement.
    assert stats["payload_chars_saved"] == record["payload_chars_saved"]


def test_telemetry_records_exact_tokens_when_backend_available(monkeypatch, tmp_path):
    """When an exact tokenizer backend is present, actual token fields are emitted."""
    import json

    module, _ = _load_plugin_module(monkeypatch)
    monkeypatch.setattr(module, "_check_reorder", lambda: False)
    monkeypatch.setattr(module, "_CONTEXTPILOT_AVAILABLE", False)
    monkeypatch.setattr(module, "dedup_chat_completions", _saving_dedup)

    # Inject a deterministic fake exact tokenizer (1 token per 3 chars).
    def fake_counter(text):
        return len(text) // 3

    fake_counter._backend = "fake:test-encoding"
    monkeypatch.setattr(module, "_get_exact_tokenizer", lambda: fake_counter)

    telemetry = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("CONTEXTPILOT_TELEMETRY_FILE", str(telemetry))

    engine = module.ContextPilotEngine()
    messages = [
        {"role": "user", "content": "read file"},
        {"role": "tool", "tool_call_id": "call_1", "content": "FULL TOOL RESULT"},
    ]
    engine.optimize_api_messages(messages)

    record = json.loads(telemetry.read_text(encoding="utf-8").splitlines()[0])

    assert record["actual_token_status"] == "available"
    assert record["actual_tokenizer_backend"] == "fake:test-encoding"
    assert record["actual_tokens_before"] >= record["actual_tokens_after"]
    assert (
        record["actual_tokens_saved"]
        == record["actual_tokens_before"] - record["actual_tokens_after"]
    )
    # Actual tokens are distinct from the legacy chars/4 estimate.
    assert "tokens_saved_method" in record


def test_optimize_telemetry_skipped_when_nothing_saved(monkeypatch, tmp_path):
    module, _ = _load_plugin_module(monkeypatch)
    monkeypatch.setattr(module, "_check_reorder", lambda: False)
    monkeypatch.setattr(module, "_CONTEXTPILOT_AVAILABLE", False)
    monkeypatch.setattr(
        module,
        "dedup_chat_completions",
        lambda body, **kw: SimpleNamespace(
            chars_saved=0, blocks_deduped=0, blocks_total=0, system_blocks_matched=0
        ),
    )

    telemetry = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("CONTEXTPILOT_TELEMETRY_FILE", str(telemetry))

    engine = module.ContextPilotEngine()
    engine.optimize_api_messages([{"role": "user", "content": "hello"}])

    # No save -> no telemetry noise.
    assert not telemetry.exists()


def test_optimize_survives_unwritable_telemetry_path(monkeypatch, tmp_path):
    module, _ = _load_plugin_module(monkeypatch)
    monkeypatch.setattr(module, "_check_reorder", lambda: False)
    monkeypatch.setattr(module, "_CONTEXTPILOT_AVAILABLE", False)
    monkeypatch.setattr(module, "dedup_chat_completions", _saving_dedup)

    # Point telemetry at a path whose parent is an existing *file*, so mkdir fails.
    blocker = tmp_path / "iam_a_file"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("CONTEXTPILOT_TELEMETRY_FILE", str(blocker / "telemetry.jsonl"))

    engine = module.ContextPilotEngine()
    messages = [
        {"role": "user", "content": "read file"},
        {"role": "tool", "tool_call_id": "call_1", "content": "FULL TOOL RESULT"},
    ]
    # Must not raise despite the unwritable telemetry destination.
    out, stats = engine.optimize_api_messages(messages)
    assert out[1]["content"] == "REF"
    assert stats["chars_saved"] > 0


def _zero_dedup(body, **kwargs):
    return SimpleNamespace(
        chars_saved=0,
        blocks_deduped=0,
        blocks_total=0,
        system_blocks_matched=0,
    )


def test_prompt_dedup_canary_default_off_does_not_mutate_runtime(monkeypatch, tmp_path):
    module, _ = _load_plugin_module(monkeypatch)
    monkeypatch.setattr(module, "_check_reorder", lambda: False)
    monkeypatch.setattr(module, "_CONTEXTPILOT_AVAILABLE", False)
    monkeypatch.setattr(module, "dedup_chat_completions", _zero_dedup)
    telemetry = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("CONTEXTPILOT_TELEMETRY_FILE", str(telemetry))
    monkeypatch.delenv("CONTEXTPILOT_PROMPT_DEDUP_MODE", raising=False)

    repeated = (
        "Reusable examples paragraph for skill notes with enough descriptive filler "
        "to make the reference shorter than the duplicate body in this test."
    )
    content = f"Use this skill when testing.\n{repeated}\n{repeated}"
    engine = module.ContextPilotEngine()
    out, stats = engine.optimize_api_messages([{"role": "system", "content": content}])

    assert out[0]["content"] == content
    assert stats["prompt_dedup_mode"] == "off"
    assert stats["prompt_dedup_chars_saved"] == 0
    assert not telemetry.exists()


def test_prompt_dedup_canary_mutates_only_skill_prompt_runtime(monkeypatch, tmp_path):
    import json

    module, _ = _load_plugin_module(monkeypatch)
    monkeypatch.setattr(module, "_check_reorder", lambda: False)
    monkeypatch.setattr(module, "_CONTEXTPILOT_AVAILABLE", False)
    monkeypatch.setattr(module, "dedup_chat_completions", _zero_dedup)
    monkeypatch.setenv("CONTEXTPILOT_PROMPT_DEDUP_MODE", "canary")
    telemetry = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("CONTEXTPILOT_TELEMETRY_FILE", str(telemetry))

    repeated = (
        "Reusable examples paragraph for skill notes with enough descriptive filler "
        "to make the reference shorter than the duplicate body in this test."
    )
    skill_content = f"Use this skill when testing.\n{repeated}\n{repeated}"
    ordinary_system = "ordinary system heading\nordinary system text stays untouched"
    user_content = f"{repeated}\n{repeated}"

    engine = module.ContextPilotEngine()
    out, stats = engine.optimize_api_messages(
        [
            {"role": "system", "content": skill_content},
            {"role": "system", "content": ordinary_system},
            {"role": "user", "content": user_content},
        ]
    )

    assert repeated in out[0]["content"]  # first occurrence kept
    assert out[0]["content"].count(repeated) == 1
    assert "ContextPilot dedup: duplicate skill_prompt block omitted" in out[0]["content"]
    # Ordinary system and user content are untouched.
    assert out[1]["content"] == ordinary_system
    assert out[2]["content"] == user_content
    assert stats["prompt_dedup_mode"] == "canary"
    assert stats["prompt_dedup_blocks_replaced"] == 1
    assert stats["prompt_dedup_chars_saved"] > 0
    assert stats["chars_saved"] == stats["prompt_dedup_chars_saved"]

    record = json.loads(telemetry.read_text(encoding="utf-8").splitlines()[0])
    assert record["prompt_dedup_mode"] == "canary"
    assert record["prompt_dedup_class"] == "same_type_skill_prompt_only"
    assert record["prompt_dedup_blocks_replaced"] == 1
    assert record["prompt_dedup_chars_saved"] == stats["prompt_dedup_chars_saved"]
    raw = telemetry.read_text(encoding="utf-8")
    assert repeated not in raw
    assert "Use this skill" not in raw


def test_prompt_dedup_canary_does_not_replace_cross_type_or_denylisted_runtime(monkeypatch):
    module, _ = _load_plugin_module(monkeypatch)
    monkeypatch.setattr(module, "_check_reorder", lambda: False)
    monkeypatch.setattr(module, "_CONTEXTPILOT_AVAILABLE", False)
    monkeypatch.setattr(module, "dedup_chat_completions", _zero_dedup)
    monkeypatch.setenv("CONTEXTPILOT_PROMPT_DEDUP_MODE", "canary")

    cross = (
        "Shared examples paragraph across prompts with enough descriptive filler "
        "to be tempting but cross hierarchy should stay unchanged."
    )
    denied = (
        "This duplicate line contains secret handling details and enough filler "
        "to be long but should be blocked by denylist."
    )
    skill_content = f"Use this skill when testing.\n{cross}\n{denied}\n{denied}"
    ordinary_system = f"ordinary system heading\n{cross}"

    engine = module.ContextPilotEngine()
    out, stats = engine.optimize_api_messages(
        [
            {"role": "system", "content": skill_content},
            {"role": "system", "content": ordinary_system},
        ]
    )

    assert out[0]["content"] == skill_content
    assert out[1]["content"] == ordinary_system
    assert stats["prompt_dedup_chars_saved"] == 0
    assert stats["prompt_dedup_blocks_replaced"] == 0
