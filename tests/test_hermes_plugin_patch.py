import importlib.util
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
