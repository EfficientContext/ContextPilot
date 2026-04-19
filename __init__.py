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
        DedupResult = _dedup_mod.DedupResult
        get_format_handler = _parser_mod.get_format_handler
        InterceptConfig = _parser_mod.InterceptConfig
        _CONTEXTPILOT_AVAILABLE = True
        _CONTEXTPILOT_IMPORT_ERROR = None
        return True
    except Exception as e:
        dedup_chat_completions = None
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
        logger.warning("[ContextPilot] Reorder unavailable, dedup-only mode: %s", e)
    return _has_reorder


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


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
        self._seen_doc_hashes: set = set()
        self._single_doc_hashes: dict = {}
        self._first_tool_result_done = False
        self._system_processed = False
        self._total_chars_saved = 0
        self._total_reordered = 0
        self._optimize_count = 0
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
        has_reorder = _check_reorder()
        turn_reordered = 0

        # Step 1: Prefix replay
        old_count = len(self._cached_messages)
        if old_count > 0 and len(api_messages) >= old_count:
            prefix_ok = True
            for i in range(old_count):
                cached_h = _hash_text(
                    json.dumps(self._cached_messages[i], sort_keys=True, default=str)
                )
                current_h = _hash_text(
                    json.dumps(api_messages[i], sort_keys=True, default=str)
                )
                if cached_h != current_h:
                    prefix_ok = False
                    break
            if prefix_ok:
                api_messages[:old_count] = copy.deepcopy(self._cached_messages)

        # Step 2-4: Extract, reorder & dedup
        # Extraction and dedup always run (pure Python, no numpy needed).
        # Reordering only runs when has_reorder is True (requires numpy + live_index).
        if _CONTEXTPILOT_AVAILABLE:
            try:
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
                    if location.msg_index < old_count:
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
                            if not new_docs:
                                new_docs = [
                                    f"[All {deduped} documents identical to a previous tool result. "
                                    f"Refer to the earlier result above.]"
                                ]
                            handler.reconstruct_tool_result(
                                body, extraction, new_docs, location
                            )

                for single_doc, location in multi.single_doc_extractions:
                    if location.msg_index < old_count:
                        continue
                    if single_doc.content_hash in self._single_doc_hashes:
                        prev_id = self._single_doc_hashes[single_doc.content_hash]
                        if (
                            single_doc.tool_call_id != prev_id
                            and handler.tool_call_present(body, prev_id)
                        ):
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
            except Exception as e:
                logger.debug("[ContextPilot] Extract/reorder failed: %s", e)

        # Step 5: Block-level dedup
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
        self._total_chars_saved += dedup_result.chars_saved

        # Step 6: Cache for next turn
        self._cached_messages = copy.deepcopy(api_messages)

        # Count tool results for diagnostics
        tool_count = 0
        tool_chars = 0
        for msg in api_messages:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                tool_count += 1
                c = msg.get("content", "")
                tool_chars += len(c) if isinstance(c, str) else 0

        single_doc_deduped = len([
            h for h in self._single_doc_hashes
        ]) if self._single_doc_hashes else 0

        logger.info(
            "[ContextPilot] Turn %d: %d chars saved, %d blocks deduped, %d docs reordered "
            "(cumulative: %d chars, %d docs) | %d tool results (%d chars), "
            "%d single-doc hashes tracked, reorder=%s",
            self._optimize_count,
            dedup_result.chars_saved,
            dedup_result.blocks_deduped,
            turn_reordered,
            self._total_chars_saved,
            self._total_reordered,
            tool_count,
            tool_chars,
            single_doc_deduped,
            has_reorder,
        )

        return api_messages, {
            "chars_saved": dedup_result.chars_saved,
            "blocks_deduped": dedup_result.blocks_deduped,
            "blocks_total": dedup_result.blocks_total,
            "system_blocks_matched": dedup_result.system_blocks_matched,
            "cumulative_chars_saved": self._total_chars_saved,
        }

    def on_context_compressed(self, old_count: int, new_count: int) -> None:
        global _intercept_index
        self._cached_messages.clear()
        self._seen_doc_hashes.clear()
        self._single_doc_hashes.clear()
        self._first_tool_result_done = False
        if _intercept_index is not None:
            _intercept_index = None

    def on_session_start(self, session_id: str, **kwargs) -> None:
        _patch_hermes_sanitizer()
        self._model = kwargs.get("model", "")
        self._base_url = ""
        self._api_key = ""
        self._provider = ""
        self._config_context_length = kwargs.get("context_length", None)
        self._ensure_compressor()
        if self._compressor and hasattr(self._compressor, "on_session_start"):
            self._compressor.on_session_start(session_id, **kwargs)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        if self._compressor and hasattr(self._compressor, "on_session_end"):
            self._compressor.on_session_end(session_id, messages)
        if self._total_chars_saved > 0:
            logger.info(
                "[ContextPilot] Session %s: %d turns, %d chars saved (~%d tokens)",
                session_id,
                self._optimize_count,
                self._total_chars_saved,
                self._total_chars_saved // 4,
            )

    def on_session_reset(self) -> None:
        if hasattr(super(), "on_session_reset"):
            super().on_session_reset()
        if self._compressor:
            self._compressor.on_session_reset()
        self.on_context_compressed(0, 0)
        self._total_chars_saved = 0
        self._total_reordered = 0
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
    ctx.register_context_engine(ContextPilotEngine())
