"""ContextPilot context engine plugin for Hermes Agent.

Install: hermes plugins install <org>/ContextPilot
Enable:  hermes plugins → Context Engine → contextpilot
"""

import copy
import hashlib
import importlib
import importlib.util as _ilu
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("contextpilot.hermes_plugin")

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from agent.context_engine import ContextEngine

    _HERMES_AVAILABLE = True
except ImportError:
    ContextEngine = object
    _HERMES_AVAILABLE = False

def _load_submodule(name: str, file_path: Path):
    """Load a .py file directly, bypassing contextpilot/__init__.py."""
    spec = _ilu.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

dedup_chat_completions = None
dedup_responses_api = None
DedupResult = None
get_format_handler = None
InterceptConfig = None
_CONTEXTPILOT_AVAILABLE = False
_CONTEXTPILOT_IMPORT_ERROR = None

_has_reorder = None
_intercept_index = None
_hermes_sanitizer_patched = False
_bootstrap_attempted = False


def _import_contextpilot_submodules():
    global dedup_chat_completions
    global dedup_responses_api
    global DedupResult
    global get_format_handler
    global InterceptConfig
    global _CONTEXTPILOT_AVAILABLE
    global _CONTEXTPILOT_IMPORT_ERROR

    try:
        _cp_root = _REPO_ROOT / "contextpilot"
        _dedup_mod = _load_submodule(
            "_contextpilot_block_dedup", _cp_root / "dedup" / "block_dedup.py"
        )
        _parser_mod = _load_submodule(
            "_contextpilot_intercept_parser", _cp_root / "server" / "intercept_parser.py"
        )

        dedup_chat_completions = _dedup_mod.dedup_chat_completions
        dedup_responses_api = _dedup_mod.dedup_responses_api
        DedupResult = _dedup_mod.DedupResult
        get_format_handler = _parser_mod.get_format_handler
        InterceptConfig = _parser_mod.InterceptConfig
        _CONTEXTPILOT_AVAILABLE = True
        _CONTEXTPILOT_IMPORT_ERROR = None
        return True
    except Exception as e:
        dedup_chat_completions = None
        dedup_responses_api = None
        DedupResult = None
        get_format_handler = None
        InterceptConfig = None
        _CONTEXTPILOT_AVAILABLE = False
        _CONTEXTPILOT_IMPORT_ERROR = e
        logger.debug("[ContextPilot] Could not import submodules: %s", e)
        return False


def _bootstrap_contextpilot_install():
    global _bootstrap_attempted
    if _bootstrap_attempted or os.environ.get("CONTEXTPILOT_PLUGIN_BOOTSTRAP") == "1":
        return False
    if not (_REPO_ROOT / "pyproject.toml").exists():
        return False

    _bootstrap_attempted = True
    logger.info("[ContextPilot] Installing plugin package into Hermes environment")
    env = os.environ.copy()
    env["CONTEXTPILOT_PLUGIN_BOOTSTRAP"] = "1"

    def _run(cmd: List[str]):
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )

    try:
        result = _run([sys.executable, "-m", "pip", "install", "-e", str(_REPO_ROOT)])
    except Exception as e:
        logger.warning("[ContextPilot] Self-install failed: %s", e)
        return False

    if result.returncode != 0 and "No module named pip" in ((result.stderr or "") + (result.stdout or "")):
        logger.info("[ContextPilot] pip missing in Hermes environment, bootstrapping with ensurepip")
        try:
            ensurepip_result = _run([sys.executable, "-m", "ensurepip", "--upgrade"])
        except Exception as e:
            logger.warning("[ContextPilot] ensurepip failed: %s", e)
            return False
        if ensurepip_result.returncode != 0:
            stderr = (ensurepip_result.stderr or "").strip()
            stdout = (ensurepip_result.stdout or "").strip()
            detail = stderr or stdout or f"exit code {ensurepip_result.returncode}"
            logger.warning("[ContextPilot] ensurepip failed: %s", detail)
            return False
        result = _run([sys.executable, "-m", "pip", "install", "-e", str(_REPO_ROOT)])

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        logger.warning("[ContextPilot] Self-install failed: %s", detail)
        return False

    importlib.invalidate_caches()
    logger.info("[ContextPilot] Self-install completed")
    return True


def _ensure_contextpilot_available():
    if _CONTEXTPILOT_AVAILABLE:
        return True
    if _import_contextpilot_submodules():
        return True
    if _bootstrap_contextpilot_install() and _import_contextpilot_submodules():
        return True
    return False


_import_contextpilot_submodules()


def _check_reorder():
    global _has_reorder
    if _has_reorder is not None:
        return _has_reorder
    try:
        from contextpilot.server.live_index import ContextPilot as _CP  # noqa: F401

        _has_reorder = True
    except Exception as e:
        if _bootstrap_contextpilot_install():
            try:
                from contextpilot.server.live_index import ContextPilot as _CP  # noqa: F401

                _has_reorder = True
                return _has_reorder
            except Exception as retry_error:
                e = retry_error
        _has_reorder = False
        logger.debug("[ContextPilot] Reorder unavailable, dedup-only mode: %s", e)
    return _has_reorder


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _telemetry_path() -> "Path | None":
    """Resolve the metadata-only telemetry file, or None if disabled.

    Lets the monitor read ContextPilot savings without depending on gateway log
    lines. Override with CONTEXTPILOT_TELEMETRY_FILE; disable with
    CONTEXTPILOT_DISABLE_TELEMETRY=1.
    """
    if os.environ.get("CONTEXTPILOT_DISABLE_TELEMETRY") == "1":
        return None
    override = os.environ.get("CONTEXTPILOT_TELEMETRY_FILE")
    if override:
        return Path(override)
    return Path.home() / ".hermes" / "contextpilot" / "telemetry.jsonl"


def _write_telemetry(record: Dict[str, Any]) -> None:
    """Append one metadata-only JSON line. Never raises; best-effort only.

    Privacy contract: callers must pass numeric counters / timestamps / session
    / turn metadata only — never message bodies, prompts, or tool payloads.
    """
    try:
        path = _telemetry_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception as e:  # noqa: BLE001 - telemetry must never break optimization
        logger.debug("[ContextPilot] telemetry write skipped: %s", e)


def _iter_message_text(messages: List[Dict[str, Any]]):
    """Yield text fragments from an LLM-bound payload for in-memory measurement.

    Used only to *size* the payload (chars / exact tokens). Fragments are never
    stored or emitted -- callers consume them immediately to produce integer
    counts, then discard them.
    """
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    yield block
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        yield text
                    inner = block.get("content")
                    if isinstance(inner, str):
                        yield inner


def _payload_chars(messages: List[Dict[str, Any]]) -> int:
    """Total character count of an LLM-bound payload (metadata-only measure)."""
    return sum(len(frag) for frag in _iter_message_text(messages))


# Sentinel so the (possibly None) tokenizer is resolved at most once per process.
_exact_tokenizer_cache: Any = "unset"


def _get_exact_tokenizer():
    """Return a callable ``(text) -> int`` for EXACT token counting, or None.

    Optional and best-effort: an exact tokenizer is used only when a backend is
    installed and not disabled. This never raises and never installs anything;
    when no backend is available the caller records an ``unavailable`` status
    rather than emitting a fake (chars/4) token count.

    Backend selection via ``CONTEXTPILOT_EXACT_TOKENIZER`` = ``off`` (default)
    | ``tiktoken``. It is opt-in so merely having a tokenizer library installed
    never creates a misleading provider/tokenizer mismatch. The separate
    disable environment flag also returns ``None`` immediately.
    """

    global _exact_tokenizer_cache
    if _exact_tokenizer_cache != "unset":
        return _exact_tokenizer_cache
    _exact_tokenizer_cache = None
    if os.environ.get("CONTEXTPILOT_DISABLE_EXACT_TOKENIZER") == "1":
        return None
    backend = os.environ.get("CONTEXTPILOT_EXACT_TOKENIZER", "off").lower()
    if backend in ("off", "none", "disabled", "auto"):
        return None
    if backend == "tiktoken":
        try:
            import tiktoken  # optional dependency; never a hard requirement

            encoding_name = os.environ.get(
                "CONTEXTPILOT_TIKTOKEN_ENCODING", "cl100k_base"
            )
            enc = tiktoken.get_encoding(encoding_name)

            def _count(text: str, _enc=enc) -> int:
                return len(_enc.encode(text, disallowed_special=()))

            _count._backend = f"tiktoken:{encoding_name}"  # type: ignore[attr-defined]
            _exact_tokenizer_cache = _count
        except Exception as e:  # noqa: BLE001 - tokenizer is strictly optional
            logger.debug("[ContextPilot] exact tokenizer unavailable: %s", e)
            _exact_tokenizer_cache = None
    return _exact_tokenizer_cache


def _measure_actual_tokens(
    original_messages: List[Dict[str, Any]],
    optimized_messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Metadata-only EXACT before/after token measurement of the payload.

    Returns a dict carrying ``actual_token_status`` of ``available`` or
    ``unavailable``. When unavailable (no exact tokenizer backend), it emits NO
    token numbers -- callers must not substitute a chars/4 estimate for these
    fields. Raw text is counted in-memory only and never stored.
    """
    counter = _get_exact_tokenizer()
    if counter is None:
        return {"actual_token_status": "unavailable"}
    try:
        before = sum(counter(frag) for frag in _iter_message_text(original_messages))
        after = sum(counter(frag) for frag in _iter_message_text(optimized_messages))
    except Exception as e:  # noqa: BLE001 - a measurement must never break optimization
        logger.debug("[ContextPilot] exact token measurement failed: %s", e)
        return {"actual_token_status": "unavailable"}
    return {
        "actual_token_status": "available",
        "actual_tokenizer_backend": getattr(counter, "_backend", "unknown"),
        "actual_tokens_before": before,
        "actual_tokens_after": after,
        "actual_tokens_saved": before - after,
    }


def _classify_prompt_content_for_canary(text: str) -> str:
    """Conservatively classify runtime system text for prompt-dedup canary.

    Runtime API payloads usually expose both system and skill instructions as
    role='system' messages. The canary may only rewrite clearly skill-like text;
    ordinary/unclear system content stays system_prompt and is therefore never
    eligible for the same_type_skill_prompt_only canary class.
    """
    low = text.lower()
    stripped = low.lstrip()
    if stripped.startswith("---") and "name:" in low[:300]:
        return "skill_prompt"
    # Runtime canary is stricter than the offline analyzer: only obvious skill
    # documents whose leading text says "use this skill" are writable. Broader
    # cues such as "available skills" remain system_prompt at runtime.
    if "use this skill" in low[:500]:
        return "skill_prompt"
    return "system_prompt"


def _apply_prompt_dedup_canary_to_api_messages(
    api_messages: List[Dict[str, Any]], *, salt: str = "contextpilot-runtime-prompt-dedup-v1"
):
    """Apply the default-off skill-prompt canary to runtime API messages.

    This is a narrow adapter from Hermes/OpenAI-style messages to the analyzer
    package's in-memory _LLMContent carrier. It mutates api_messages only when
    CONTEXTPILOT_PROMPT_DEDUP_MODE=canary and the canary module replaces a
    same_type_skill_prompt_only duplicate. User/assistant/tool and ordinary
    system content are never passed as writable skill_prompt items.
    """
    try:
        from contextpilot.hermes_opportunities.models import _LLMContent
        from contextpilot.hermes_opportunities.prompt_dedup_canary import (
            apply_prompt_dedup_canary,
        )
    except Exception as e:  # noqa: BLE001 - canary must never break requests
        logger.debug("[ContextPilot] prompt dedup canary unavailable: %s", e)
        return None

    llm_items = []
    message_indexes = []
    for idx, msg in enumerate(api_messages):
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        block_type = _classify_prompt_content_for_canary(content)
        llm_items.append(_LLMContent(block_type=block_type, content=content))
        message_indexes.append(idx)

    if not llm_items:
        return None

    result = apply_prompt_dedup_canary(
        llm_items,
        salt=salt,
        min_block_chars=40,
    )
    if result and result.mutated:
        for item, idx in zip(llm_items, message_indexes):
            if item.block_type == "skill_prompt":
                api_messages[idx]["content"] = item.content
    return result


def _reorder_docs(docs: List[str], alpha: float = 0.001) -> List[str]:
    global _intercept_index
    if len(docs) < 2:
        return docs
    from contextpilot.server.live_index import ContextPilot as CP

    contexts = [docs]
    if _intercept_index is None:
        _intercept_index = CP(alpha=alpha, use_gpu=False, linkage_method="average")
        _intercept_index.build_and_schedule(contexts=contexts)
        return docs
    result = _intercept_index.build_incremental(contexts=contexts)
    reordered = result.get("reordered_contexts", [docs])[0]
    doc_to_orig = {}
    for i, doc in enumerate(docs):
        doc_to_orig.setdefault(doc, []).append(i)
    order = []
    used = set()
    for doc in reordered:
        for idx in doc_to_orig.get(doc, []):
            if idx not in used:
                order.append(idx)
                used.add(idx)
                break
    return [docs[i] for i in order]


def _patch_hermes_sanitizer():
    global _hermes_sanitizer_patched
    if _hermes_sanitizer_patched:
        return
    try:
        import run_agent
    except Exception as e:
        logger.debug("[ContextPilot] Could not import run_agent for patching: %s", e)
        return

    AIAgent = getattr(run_agent, "AIAgent", None)
    if AIAgent is None:
        return

    current = getattr(AIAgent, "_sanitize_api_messages", None)
    if current is None:
        return
    if getattr(current, "_contextpilot_patched", False):
        _hermes_sanitizer_patched = True
        return

    original = current

    def _patched_sanitize_api_messages(self_or_messages, maybe_messages=None):
        if maybe_messages is None:
            agent = None
            messages = self_or_messages
        else:
            agent = self_or_messages
            messages = maybe_messages

        sanitized = original(messages)
        if agent is None:
            return sanitized

        engine = getattr(agent, "context_compressor", None)
        optimize = getattr(engine, "optimize_api_messages", None)
        if not callable(optimize):
            return sanitized

        try:
            optimized, _stats = optimize(
                sanitized,
                system_content=getattr(agent, "_cached_system_prompt", "") or "",
            )
        except Exception as e:
            logger.debug("[ContextPilot] Hermes sanitize hook failed: %s", e)
            return sanitized

        return optimized if isinstance(optimized, list) else sanitized

    _patched_sanitize_api_messages._contextpilot_patched = True
    _patched_sanitize_api_messages._contextpilot_original = original
    AIAgent._sanitize_api_messages = _patched_sanitize_api_messages
    _hermes_sanitizer_patched = True
    logger.info("[ContextPilot] Installed Hermes API-message hook")


class ContextPilotEngine(ContextEngine):
    @property
    def name(self) -> str:
        return "contextpilot"

    def __init__(self):
        self._compressor = None
        self._cached_messages: list = []
        self._cached_original_messages: list = []
        self._seen_doc_hashes: set = set()
        self._single_doc_hashes: dict = {}
        self._first_tool_result_done = False
        self._system_processed = False
        self._total_chars_saved = 0
        self._total_reordered = 0
        self._total_docs_deduped = 0
        self._optimize_count = 0
        self._session_id = None
        self.threshold_percent = 0.75

    @staticmethod
    def is_available() -> bool:
        return True

    def _ensure_compressor(self):
        if self._compressor is not None:
            return
        from agent.context_compressor import ContextCompressor

        self._compressor = ContextCompressor(
            model=getattr(self, "_model", ""),
            base_url=getattr(self, "_base_url", ""),
            api_key=getattr(self, "_api_key", ""),
            provider=getattr(self, "_provider", ""),
            config_context_length=getattr(self, "_config_context_length", None),
            quiet_mode=True,
        )
        self._sync_compressor_state()

    def _sync_compressor_state(self):
        if self._compressor is None:
            return
        self.threshold_tokens = self._compressor.threshold_tokens
        self.context_length = self._compressor.context_length
        self.threshold_percent = self._compressor.threshold_percent
        self.protect_first_n = self._compressor.protect_first_n
        self.protect_last_n = self._compressor.protect_last_n

    def _activate_openai_hook(self):
        """Patch OpenAI SDK calls that bypass Hermes' sanitizer path."""
        engine = self

        def _patch_chat_module(module):
            for class_name, is_async in (
                ("Completions", False),
                ("AsyncCompletions", True),
            ):
                cls = getattr(module, class_name, None)
                if cls is None or getattr(cls, "_contextpilot_patched", False):
                    continue
                original = cls.create
                if is_async:
                    async def _patched(self, *args, _orig=original, **kwargs):
                        engine._intercept_chat_kwargs(kwargs)
                        return await _orig(self, *args, **kwargs)
                else:
                    def _patched(self, *args, _orig=original, **kwargs):
                        engine._intercept_chat_kwargs(kwargs)
                        return _orig(self, *args, **kwargs)

                cls.create = _patched
                cls._contextpilot_patched = True
                logger.info("[ContextPilot] Patched OpenAI %s.create", class_name)

        def _patch_responses_module(module):
            for class_name, is_async in (
                ("Responses", False),
                ("AsyncResponses", True),
            ):
                cls = getattr(module, class_name, None)
                if cls is None or getattr(cls, "_contextpilot_patched", False):
                    continue
                original_create = getattr(cls, "create", None)
                original_stream = getattr(cls, "stream", None)
                if original_create is not None:
                    if is_async:
                        async def _patched_create(self, *args, _orig=original_create, **kwargs):
                            engine._intercept_responses_kwargs(kwargs)
                            return await _orig(self, *args, **kwargs)
                    else:
                        def _patched_create(self, *args, _orig=original_create, **kwargs):
                            engine._intercept_responses_kwargs(kwargs)
                            return _orig(self, *args, **kwargs)
                    cls.create = _patched_create
                if original_stream is not None:
                    def _patched_stream(self, *args, _orig=original_stream, **kwargs):
                        engine._intercept_responses_kwargs(kwargs)
                        return _orig(self, *args, **kwargs)
                    cls.stream = _patched_stream
                cls._contextpilot_patched = True
                logger.info("[ContextPilot] Patched OpenAI %s", class_name)

        for module_name, patcher in (
            ("openai.resources.chat.completions", _patch_chat_module),
            ("openai.resources.responses.responses", _patch_responses_module),
            ("openai.resources.responses", _patch_responses_module),
        ):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            patcher(module)

    def _matches_cached_optimized_payload(self, api_messages: List[Dict[str, Any]]) -> bool:
        if len(api_messages) != len(self._cached_messages):
            return False
        for current, cached in zip(api_messages, self._cached_messages):
            current_h = _hash_text(json.dumps(current, sort_keys=True, default=str))
            cached_h = _hash_text(json.dumps(cached, sort_keys=True, default=str))
            if current_h != cached_h:
                return False
        return bool(api_messages)

    def _intercept_chat_kwargs(self, kwargs: Dict[str, Any]) -> None:
        messages = kwargs.get("messages")
        if not messages or not isinstance(messages, list):
            return
        if self._matches_cached_optimized_payload(messages):
            return
        try:
            optimized, _stats = self.optimize_api_messages(messages, system_content="")
        except Exception as e:
            logger.debug("[ContextPilot] OpenAI chat hook failed: %s", e)
            return
        if isinstance(optimized, list):
            kwargs["messages"] = optimized

    def _intercept_responses_kwargs(self, kwargs: Dict[str, Any]) -> None:
        items = kwargs.get("input")
        if not items or not isinstance(items, list) or not callable(dedup_responses_api):
            return
        system_content = kwargs.get("instructions")
        if not isinstance(system_content, str):
            system_content = None
        try:
            result = dedup_responses_api({"input": items}, system_content=system_content)
        except Exception as e:
            logger.debug("[ContextPilot] OpenAI Responses hook failed: %s", e)
            return
        if getattr(result, "chars_saved", 0) > 0:
            self._total_chars_saved += result.chars_saved
            logger.info(
                "[ContextPilot] Responses hook: %d chars saved, %d blocks deduped",
                result.chars_saved,
                result.blocks_deduped,
            )

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self._ensure_compressor()
        self._compressor.update_from_response(usage)
        self.last_prompt_tokens = self._compressor.last_prompt_tokens
        self.last_completion_tokens = self._compressor.last_completion_tokens

    def should_compress(self, prompt_tokens: int = None) -> bool:
        self._ensure_compressor()
        return self._compressor.should_compress(prompt_tokens)

    def compress(
        self, messages: List[Dict[str, Any]], current_tokens: int = None, **kwargs
    ) -> List[Dict[str, Any]]:
        self._ensure_compressor()
        result = self._compressor.compress(
            messages, current_tokens=current_tokens, **kwargs
        )
        self.compression_count = self._compressor.compression_count
        return result

    def optimize_api_messages(
        self,
        api_messages: List[Dict[str, Any]],
        *,
        system_content: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self._optimize_count += 1
        if self._optimize_count == 1:
            logger.info("[ContextPilot] Per-turn API optimizer active")
        if self._matches_cached_optimized_payload(api_messages):
            return api_messages, {
                "chars_saved": 0,
                "doc_chars_saved": 0,
                "block_chars_saved": 0,
                "blocks_deduped": 0,
                "blocks_total": 0,
                "docs_deduped": self._total_docs_deduped,
                "system_blocks_matched": 0,
                "cumulative_chars_saved": self._total_chars_saved,
            }
        has_reorder = _check_reorder()
        turn_reordered = 0
        original_messages = copy.deepcopy(api_messages)
        replayed_count = 0

        # Step 1: Prefix replay
        old_count = min(len(self._cached_original_messages), len(self._cached_messages))
        if old_count > 0 and len(api_messages) >= old_count:
            prefix_ok = True
            for i in range(old_count):
                cached_h = _hash_text(json.dumps(
                    self._cached_original_messages[i],
                    sort_keys=True,
                    default=str,
                ))
                current_h = _hash_text(
                    json.dumps(api_messages[i], sort_keys=True, default=str)
                )
                if cached_h != current_h:
                    prefix_ok = False
                    break
            if prefix_ok:
                api_messages[:old_count] = copy.deepcopy(self._cached_messages)
                replayed_count = old_count

        # Step 2-4: Extract, reorder & dedup
        # Extraction and dedup always run (pure Python, no numpy needed).
        # Reordering only runs when has_reorder is True (requires numpy + live_index).
        doc_chars_saved = 0
        if _CONTEXTPILOT_AVAILABLE:
            try:
                def _tool_chars(msgs):
                    return sum(
                        len(m.get("content", "") or "")
                        for m in msgs if isinstance(m, dict) and m.get("role") == "tool"
                    )

                config = InterceptConfig(
                    enabled=True,
                    mode="auto",
                    tag="document",
                    separator="---",
                    alpha=0.001,
                    linkage_method="average",
                    scope="all",
                )
                handler = get_format_handler("openai_chat")
                body = {"messages": api_messages}
                chars_before_extract = _tool_chars(body["messages"])
                multi = handler.extract_all(body, config)

                if has_reorder:
                    if multi.system_extraction and not self._system_processed:
                        extraction, sys_idx = multi.system_extraction
                        if len(extraction.documents) >= 2:
                            reordered = _reorder_docs(extraction.documents)
                            if reordered != extraction.documents:
                                handler.reconstruct_system(
                                    body, extraction, reordered, sys_idx
                                )
                        self._system_processed = True

                for extraction, location in multi.tool_extractions:
                    if location.msg_index < replayed_count:
                        continue
                    if len(extraction.documents) < 2:
                        continue
                    if not self._first_tool_result_done:
                        self._first_tool_result_done = True
                        if has_reorder:
                            reordered = _reorder_docs(extraction.documents)
                        else:
                            reordered = extraction.documents
                        for doc in extraction.documents:
                            self._seen_doc_hashes.add(_hash_text(doc))
                        if has_reorder and reordered != extraction.documents:
                            handler.reconstruct_tool_result(
                                body, extraction, reordered, location
                            )
                            turn_reordered += len(extraction.documents)
                            self._total_reordered += len(extraction.documents)
                    else:
                        new_docs = []
                        deduped = 0
                        for doc in extraction.documents:
                            h = _hash_text(doc)
                            if h in self._seen_doc_hashes:
                                deduped += 1
                            else:
                                self._seen_doc_hashes.add(h)
                                new_docs.append(doc)
                        if deduped > 0:
                            self._total_docs_deduped += deduped
                            if not new_docs:
                                new_docs = [
                                    f"[All {deduped} documents identical to a previous tool result. "
                                    f"Refer to the earlier result above.]"
                                ]
                            handler.reconstruct_tool_result(
                                body, extraction, new_docs, location
                            )

                for single_doc, location in multi.single_doc_extractions:
                    if location.msg_index < replayed_count:
                        continue
                    if single_doc.content_hash in self._single_doc_hashes:
                        prev_id = self._single_doc_hashes[single_doc.content_hash]
                        if (
                            single_doc.tool_call_id != prev_id
                            and handler.tool_call_present(body, prev_id)
                        ):
                            self._total_docs_deduped += 1
                            handler.replace_single_doc(
                                body,
                                location,
                                (
                                    f"[Duplicate — identical to previous tool result ({prev_id}). "
                                    f"Refer to the earlier result above.]"
                                ),
                            )
                    else:
                        self._single_doc_hashes[single_doc.content_hash] = (
                            single_doc.tool_call_id
                        )

                api_messages = body["messages"]
                doc_chars_saved = chars_before_extract - _tool_chars(api_messages)
            except Exception as e:
                logger.debug("[ContextPilot] Extract/reorder failed: %s", e)

        # Step 5: Optional prompt-dedup canary (default off). This is the only
        # runtime prompt mutation path and is limited to same_type_skill_prompt_only.
        prompt_dedup_result = _apply_prompt_dedup_canary_to_api_messages(api_messages)
        prompt_dedup_chars_saved = (
            prompt_dedup_result.chars_saved
            if prompt_dedup_result is not None and prompt_dedup_result.mutated
            else 0
        )

        # Step 6: Block-level dedup
        sys_content = None
        for msg in api_messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                sc = msg.get("content", "")
                if isinstance(sc, str):
                    sys_content = sc
                break

        dedup_result: DedupResult = dedup_chat_completions(
            {"messages": api_messages},
            system_content=sys_content,
        )
        turn_chars_saved = doc_chars_saved + dedup_result.chars_saved + prompt_dedup_chars_saved
        self._total_chars_saved += turn_chars_saved

        # Actual before/after of the full LLM-bound payload (chars). These are
        # measured directly from the original input vs the optimized output, so
        # they reflect the realized processed-payload delta -- not a duplicate
        # opportunity count. Cheap (string length only); always computed.
        payload_chars_before = _payload_chars(original_messages)
        payload_chars_after = _payload_chars(api_messages)
        payload_chars_saved = payload_chars_before - payload_chars_after

        # Step 6: Cache for next turn
        self._cached_messages = copy.deepcopy(api_messages)
        self._cached_original_messages = original_messages

        if turn_chars_saved > 0:
            logger.info(
                "[ContextPilot] Turn %d: saved %d chars by processing | cumulative: %d chars",
                self._optimize_count,
                turn_chars_saved,
                self._total_chars_saved,
            )
            # Metadata-only telemetry so the monitor does not depend solely on
            # gateway log lines. No content, prompts, or tool payloads here.
            #
            # Token fields are deliberately separated by provenance:
            #   * ``tokens_saved`` is the LEGACY DERIVED estimate (chars/4); the
            #     ``tokens_saved_method`` tag makes that explicit so it is never
            #     mistaken for a tokenizer/API measurement.
            #   * ``actual_tokens_*`` come from an EXACT tokenizer and are present
            #     only when ``actual_token_status == "available"``. When no exact
            #     tokenizer backend is configured the status is ``unavailable``
            #     and no token numbers are emitted (no fake counts).
            telemetry_record = {
                "ts": time.time(),
                "type": "turn",
                "session_hash": (
                    _hash_text(str(self._session_id))
                    if self._session_id is not None else None
                ),
                "turn": self._optimize_count,
                # Actual processed-payload char delta (doc + block dedup).
                "chars_saved": turn_chars_saved,
                # Actual before/after of the full LLM-bound payload (chars).
                "payload_chars_before": payload_chars_before,
                "payload_chars_after": payload_chars_after,
                "payload_chars_saved": payload_chars_saved,
                # Legacy DERIVED token estimate (chars/4) -- NOT exact tokens.
                "tokens_saved": turn_chars_saved // 4,
                "tokens_saved_method": "estimated_chars_div_4",
                "doc_chars_saved": doc_chars_saved,
                "block_chars_saved": dedup_result.chars_saved,
                "prompt_dedup_mode": (
                    prompt_dedup_result.mode if prompt_dedup_result is not None else "off"
                ),
                "prompt_dedup_class": (
                    prompt_dedup_result.prompt_dedup_class
                    if prompt_dedup_result is not None else "same_type_skill_prompt_only"
                ),
                "prompt_dedup_blocks_replaced": (
                    prompt_dedup_result.blocks_replaced
                    if prompt_dedup_result is not None and prompt_dedup_result.mutated else 0
                ),
                "prompt_dedup_chars_saved": prompt_dedup_chars_saved,
                "blocks_deduped": dedup_result.blocks_deduped,
                "blocks_total": dedup_result.blocks_total,
                "docs_deduped": self._total_docs_deduped,
                "system_blocks_matched": dedup_result.system_blocks_matched,
                "cumulative_chars_saved": self._total_chars_saved,
            }
            # Optional EXACT token measurement (only computed on a saving turn).
            telemetry_record.update(
                _measure_actual_tokens(original_messages, api_messages)
            )
            _write_telemetry(telemetry_record)

        return api_messages, {
            "chars_saved": turn_chars_saved,
            "payload_chars_before": payload_chars_before,
            "payload_chars_after": payload_chars_after,
            "payload_chars_saved": payload_chars_saved,
            "doc_chars_saved": doc_chars_saved,
            "block_chars_saved": dedup_result.chars_saved,
            "prompt_dedup_mode": (
                prompt_dedup_result.mode if prompt_dedup_result is not None else "off"
            ),
            "prompt_dedup_chars_saved": prompt_dedup_chars_saved,
            "prompt_dedup_blocks_replaced": (
                prompt_dedup_result.blocks_replaced
                if prompt_dedup_result is not None and prompt_dedup_result.mutated else 0
            ),
            "blocks_deduped": dedup_result.blocks_deduped,
            "blocks_total": dedup_result.blocks_total,
            "docs_deduped": self._total_docs_deduped,
            "system_blocks_matched": dedup_result.system_blocks_matched,
            "cumulative_chars_saved": self._total_chars_saved,
        }

    def on_context_compressed(self, old_count: int, new_count: int) -> None:
        global _intercept_index
        self._cached_messages.clear()
        self._cached_original_messages.clear()
        self._seen_doc_hashes.clear()
        self._single_doc_hashes.clear()
        self._first_tool_result_done = False
        if _intercept_index is not None:
            _intercept_index = None

    def on_session_start(self, session_id: str, **kwargs) -> None:
        _patch_hermes_sanitizer()
        self._session_id = session_id
        self._model = kwargs.get("model", "")
        self._base_url = ""
        self._api_key = ""
        self._provider = ""
        self._config_context_length = kwargs.get("context_length", None)
        self._ensure_compressor()
        if self._compressor and hasattr(self._compressor, "on_session_start"):
            self._compressor.on_session_start(session_id, **kwargs)
        self._activate_openai_hook()

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        if self._compressor and hasattr(self._compressor, "on_session_end"):
            self._compressor.on_session_end(session_id, messages)
        if self._total_chars_saved > 0:
            logger.info(
                "[ContextPilot] Session %s: %d turns, %d chars saved by processing",
                session_id,
                self._optimize_count,
                self._total_chars_saved,
            )

    def on_session_reset(self) -> None:
        if hasattr(super(), "on_session_reset"):
            super().on_session_reset()
        if self._compressor:
            self._compressor.on_session_reset()
        self.on_context_compressed(0, 0)
        self._total_chars_saved = 0
        self._total_reordered = 0
        self._total_docs_deduped = 0
        self._optimize_count = 0

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        **kwargs,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._provider = provider
        if self._compressor:
            self._compressor.update_model(
                model=model,
                context_length=context_length,
                base_url=base_url,
                api_key=api_key,
                provider=provider,
                **kwargs,
            )
            self._sync_compressor_state()
        else:
            self.context_length = context_length
            self.threshold_tokens = int(context_length * self.threshold_percent)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = []
        if self._compressor:
            schemas.extend(self._compressor.get_tool_schemas())
        return schemas

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._compressor:
            return self._compressor.handle_tool_call(name, args, **kwargs)
        return json.dumps({"error": f"Unknown tool: {name}"})

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["engine"] = "contextpilot"
        status["contextpilot_chars_saved"] = self._total_chars_saved
        status["contextpilot_docs_reordered"] = self._total_reordered
        return status


def _auto_set_context_engine():
    """Set context.engine to 'contextpilot' on first install only."""
    try:
        from hermes_cli.config import load_config, save_config
        config = load_config()
        ctx = config.get("context", {})
        current = ctx.get("engine", "compressor")
        if current == "contextpilot":
            return  # Already set
        if current != "compressor":
            return  # User chose a different engine — don't override
        if ctx.get("_contextpilot_offered"):
            return  # Offered before, user switched back to compressor — respect that
        config.setdefault("context", {})["engine"] = "contextpilot"
        config["context"]["_contextpilot_offered"] = True
        save_config(config)
        logger.info("[ContextPilot] Auto-configured as active context engine")
    except Exception as e:
        logger.debug("[ContextPilot] Could not auto-set config: %s", e)


def register(ctx):
    """Hermes plugin entry point — called by PluginManager.discover_and_load()."""
    if not _HERMES_AVAILABLE:
        return
    if not _ensure_contextpilot_available():
        logger.warning(
            "[ContextPilot] contextpilot package not importable after self-install: %s",
            _CONTEXTPILOT_IMPORT_ERROR,
        )
        return
    _patch_hermes_sanitizer()
    _auto_set_context_engine()
    ctx.register_context_engine(ContextPilotEngine())
