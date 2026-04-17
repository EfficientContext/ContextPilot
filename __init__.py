"""ContextPilot context engine plugin for Hermes Agent.

Install: hermes plugins install <org>/ContextPilot
Enable:  hermes plugins → Context Engine → contextpilot
"""

import copy
import hashlib
import json
import logging
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

from contextpilot.dedup import dedup_chat_completions, DedupResult
from contextpilot.server.intercept_parser import get_format_handler, InterceptConfig

_has_reorder = None
_intercept_index = None


def _check_reorder():
    global _has_reorder
    if _has_reorder is not None:
        return _has_reorder
    try:
        from contextpilot.server.live_index import ContextPilot as _CP  # noqa: F401

        _has_reorder = True
    except (ImportError, Exception):
        _has_reorder = False
        logger.info("[ContextPilot] Reorder unavailable (numpy?), dedup-only mode")
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
        has_reorder = _check_reorder()

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

        # Step 2-4: Extract & reorder
        if has_reorder:
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
                        reordered = _reorder_docs(extraction.documents)
                        for doc in extraction.documents:
                            self._seen_doc_hashes.add(_hash_text(doc))
                        if reordered != extraction.documents:
                            handler.reconstruct_tool_result(
                                body, extraction, reordered, location
                            )
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

        if dedup_result.chars_saved > 0 or self._total_reordered > 0:
            logger.info(
                "[ContextPilot] Turn %d: %d chars saved, %d blocks deduped, %d docs reordered",
                self._optimize_count,
                dedup_result.chars_saved,
                dedup_result.blocks_deduped,
                self._total_reordered,
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
    ctx.register_context_engine(ContextPilotEngine())
