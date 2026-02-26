"""Unit tests for contextpilot.server.intercept_parser."""

import pytest
from contextpilot.server.intercept_parser import (
    InterceptConfig,
    ExtractionResult,
    parse_intercept_headers,
    extract_documents,
    extract_from_openai_chat,
    extract_from_anthropic_messages,
    reconstruct_openai_chat,
    reconstruct_anthropic_messages,
    reconstruct_content,
)


# ============================================================================
# Header parsing
# ============================================================================


class TestParseHeaders:
    def test_defaults(self):
        config = parse_intercept_headers({})
        assert config.enabled is True
        assert config.mode == "auto"
        assert config.tag == "document"
        assert config.separator == "---"
        assert config.alpha == pytest.approx(0.001)
        assert config.linkage_method == "average"

    def test_explicit_mode(self):
        headers = {
            "X-ContextPilot-Mode": "xml_tag",
            "X-ContextPilot-Tag": "passage",
            "X-ContextPilot-Alpha": "0.01",
            "X-ContextPilot-Linkage": "complete",
        }
        config = parse_intercept_headers(headers)
        assert config.mode == "xml_tag"
        assert config.tag == "passage"
        assert config.alpha == pytest.approx(0.01)
        assert config.linkage_method == "complete"

    def test_disabled(self):
        for val in ("false", "0", "no", "False", "NO"):
            config = parse_intercept_headers({"X-ContextPilot-Enabled": val})
            assert config.enabled is False

    def test_case_insensitive_keys(self):
        headers = {"x-contextpilot-mode": "numbered"}
        config = parse_intercept_headers(headers)
        assert config.mode == "numbered"

    def test_separator_header(self):
        headers = {"X-ContextPilot-Separator": "==="}
        config = parse_intercept_headers(headers)
        assert config.separator == "==="


# ============================================================================
# XML tag extraction
# ============================================================================


class TestXmlExtraction:
    def test_basic_documents_wrapper(self):
        text = "<documents>\n<document>Doc A</document>\n<document>Doc B</document>\n</documents>"
        config = InterceptConfig()
        result = extract_documents(text, config)
        assert result is not None
        assert result.mode == "xml_tag"
        assert result.documents == ["Doc A", "Doc B"]
        assert result.wrapper_tag == "documents"
        assert result.item_tag == "document"

    def test_contexts_wrapper(self):
        text = "<contexts><context>First</context><context>Second</context></contexts>"
        config = InterceptConfig()
        result = extract_documents(text, config)
        assert result is not None
        assert result.documents == ["First", "Second"]
        assert result.wrapper_tag == "contexts"

    def test_docs_wrapper(self):
        text = "<docs><doc>A</doc><doc>B</doc><doc>C</doc></docs>"
        config = InterceptConfig()
        result = extract_documents(text, config)
        assert result is not None
        assert result.documents == ["A", "B", "C"]

    def test_passages_wrapper(self):
        text = "<passages><passage>X</passage><passage>Y</passage></passages>"
        config = InterceptConfig()
        result = extract_documents(text, config)
        assert result is not None
        assert result.documents == ["X", "Y"]

    def test_references_wrapper(self):
        text = "<references><reference>Ref1</reference><reference>Ref2</reference></references>"
        config = InterceptConfig()
        result = extract_documents(text, config)
        assert result is not None
        assert result.documents == ["Ref1", "Ref2"]

    def test_custom_tag_explicit_mode(self):
        text = "<snippets><snippet>Code A</snippet><snippet>Code B</snippet></snippets>"
        config = InterceptConfig(mode="xml_tag", tag="snippet")
        result = extract_documents(text, config)
        assert result is not None
        assert result.documents == ["Code A", "Code B"]

    def test_prefix_suffix_preserved(self):
        text = "Here are the docs:\n<documents>\n<document>A</document>\n<document>B</document>\n</documents>\nPlease answer."
        config = InterceptConfig()
        result = extract_documents(text, config)
        assert result is not None
        assert result.prefix == "Here are the docs:\n"
        assert result.suffix == "\nPlease answer."

    def test_no_wrapper_multiple_items(self):
        text = "<document>Doc 1</document>\n<document>Doc 2</document>"
        config = InterceptConfig()
        result = extract_documents(text, config)
        assert result is not None
        assert result.documents == ["Doc 1", "Doc 2"]
        assert result.wrapper_tag == ""

    def test_multiline_content(self):
        text = "<documents>\n<document>Line 1\nLine 2</document>\n<document>Line 3\nLine 4</document>\n</documents>"
        config = InterceptConfig()
        result = extract_documents(text, config)
        assert result is not None
        assert len(result.documents) == 2
        assert "Line 1\nLine 2" in result.documents[0]

    def test_single_item_returns_none_without_wrapper(self):
        text = "<document>Only one doc</document>"
        config = InterceptConfig()
        result = extract_documents(text, config)
        # No wrapper and only 1 item -> None (need >=2 for reordering)
        assert result is None


# ============================================================================
# Numbered extraction
# ============================================================================


class TestNumberedExtraction:
    def test_basic_numbered(self):
        text = "[1] First document [2] Second document [3] Third document"
        config = InterceptConfig(mode="numbered")
        result = extract_documents(text, config)
        assert result is not None
        assert result.mode == "numbered"
        assert result.documents == ["First document", "Second document", "Third document"]

    def test_numbered_with_prefix(self):
        text = "Retrieved documents:\n[1] Doc A [2] Doc B"
        config = InterceptConfig(mode="numbered")
        result = extract_documents(text, config)
        assert result is not None
        assert result.prefix == "Retrieved documents:\n"
        assert result.documents == ["Doc A", "Doc B"]

    def test_numbered_with_newlines(self):
        text = "[1] First doc\n[2] Second doc\n[3] Third doc"
        config = InterceptConfig(mode="numbered")
        result = extract_documents(text, config)
        assert result is not None
        assert len(result.documents) == 3

    def test_single_numbered_returns_none(self):
        text = "[1] Only one document"
        config = InterceptConfig(mode="numbered")
        result = extract_documents(text, config)
        assert result is None


# ============================================================================
# Separator extraction
# ============================================================================


class TestSeparatorExtraction:
    def test_basic_separator(self):
        text = "Doc A\n---\nDoc B\n---\nDoc C"
        config = InterceptConfig(mode="separator")
        result = extract_documents(text, config)
        assert result is not None
        assert result.mode == "separator"
        assert result.documents == ["Doc A", "Doc B", "Doc C"]

    def test_equals_separator(self):
        text = "Doc A\n===\nDoc B\n===\nDoc C"
        config = InterceptConfig(mode="separator", separator="===")
        result = extract_documents(text, config)
        assert result is not None
        assert result.documents == ["Doc A", "Doc B", "Doc C"]

    def test_auto_detects_triple_dash(self):
        text = "First\n---\nSecond\n---\nThird"
        config = InterceptConfig(mode="auto")
        result = extract_documents(text, config)
        assert result is not None
        assert result.mode == "separator"
        assert result.documents == ["First", "Second", "Third"]

    def test_auto_detects_triple_equals(self):
        text = "First\n===\nSecond\n===\nThird"
        config = InterceptConfig(mode="auto")
        result = extract_documents(text, config)
        assert result is not None
        assert result.mode == "separator"

    def test_single_separator_returns_none(self):
        # Only one separator -> only 2 parts, still need >=2 docs
        text = "Only one\n---\nTwo parts"
        config = InterceptConfig(mode="separator")
        result = extract_documents(text, config)
        # 2 docs is fine (>=2)
        assert result is not None
        assert len(result.documents) == 2

    def test_no_separator_returns_none(self):
        text = "Just plain text with no separators"
        config = InterceptConfig(mode="separator")
        result = extract_documents(text, config)
        assert result is None


# ============================================================================
# Auto detection priority
# ============================================================================


class TestAutoDetection:
    def test_xml_takes_priority(self):
        text = "<documents><document>A</document><document>B</document></documents>"
        config = InterceptConfig(mode="auto")
        result = extract_documents(text, config)
        assert result is not None
        assert result.mode == "xml_tag"

    def test_numbered_before_separator(self):
        text = "[1] Doc A [2] Doc B"
        config = InterceptConfig(mode="auto")
        result = extract_documents(text, config)
        assert result is not None
        assert result.mode == "numbered"

    def test_separator_as_fallback(self):
        text = "First doc\n---\nSecond doc\n---\nThird doc"
        config = InterceptConfig(mode="auto")
        result = extract_documents(text, config)
        assert result is not None
        assert result.mode == "separator"

    def test_nothing_returns_none(self):
        text = "Just a plain system message with no documents."
        config = InterceptConfig(mode="auto")
        result = extract_documents(text, config)
        assert result is None


# ============================================================================
# Reconstruction roundtrips
# ============================================================================


class TestReconstruction:
    def test_xml_roundtrip(self):
        text = "Prefix\n<documents>\n<document>A</document>\n<document>B</document>\n<document>C</document>\n</documents>\nSuffix"
        config = InterceptConfig()
        result = extract_documents(text, config)
        assert result is not None
        rebuilt = reconstruct_content(result, ["C", "A", "B"])
        assert "<document>C</document>" in rebuilt
        assert "<document>A</document>" in rebuilt
        assert "<document>B</document>" in rebuilt
        assert rebuilt.startswith("Prefix\n")
        assert rebuilt.endswith("\nSuffix")
        assert "<documents>" in rebuilt
        assert "</documents>" in rebuilt

    def test_numbered_roundtrip(self):
        text = "[1] Alpha [2] Beta [3] Gamma"
        config = InterceptConfig(mode="numbered")
        result = extract_documents(text, config)
        rebuilt = reconstruct_content(result, ["Gamma", "Alpha", "Beta"])
        assert "[1] Gamma" in rebuilt
        assert "[2] Alpha" in rebuilt
        assert "[3] Beta" in rebuilt

    def test_separator_roundtrip(self):
        text = "Doc A\n---\nDoc B\n---\nDoc C"
        config = InterceptConfig(mode="separator")
        result = extract_documents(text, config)
        rebuilt = reconstruct_content(result, ["Doc C", "Doc A", "Doc B"])
        parts = rebuilt.split("\n---\n")
        assert parts == ["Doc C", "Doc A", "Doc B"]


# ============================================================================
# OpenAI chat format
# ============================================================================


class TestOpenAIChatFormat:
    def test_extract_from_system_message(self):
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "<documents><document>A</document><document>B</document></documents>"},
                {"role": "user", "content": "What is A?"},
            ],
        }
        config = InterceptConfig()
        result = extract_from_openai_chat(body, config)
        assert result is not None
        extraction, idx = result
        assert extraction.documents == ["A", "B"]
        assert idx == 0

    def test_no_system_message(self):
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }
        result = extract_from_openai_chat(body, InterceptConfig())
        assert result is None

    def test_system_without_docs(self):
        body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = extract_from_openai_chat(body, InterceptConfig())
        assert result is None

    def test_reconstruct_roundtrip(self):
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "<documents>\n<document>A</document>\n<document>B</document>\n<document>C</document>\n</documents>"},
                {"role": "user", "content": "Summarize"},
            ],
        }
        config = InterceptConfig()
        extraction, idx = extract_from_openai_chat(body, config)
        new_body = reconstruct_openai_chat(body, extraction, ["C", "A", "B"], idx)
        # Original body not modified
        assert "<document>A</document>" in body["messages"][0]["content"]
        # New body has reordered docs
        content = new_body["messages"][0]["content"]
        assert "<document>C</document>" in content
        # User message preserved
        assert new_body["messages"][1]["content"] == "Summarize"
        # Model preserved
        assert new_body["model"] == "gpt-4"

    def test_content_blocks_format(self):
        body = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "<documents><document>X</document><document>Y</document></documents>"}
                    ],
                },
                {"role": "user", "content": "Question"},
            ],
        }
        config = InterceptConfig()
        result = extract_from_openai_chat(body, config)
        assert result is not None
        extraction, idx = result
        assert extraction.documents == ["X", "Y"]

    def test_empty_messages(self):
        result = extract_from_openai_chat({"messages": []}, InterceptConfig())
        assert result is None

    def test_no_messages_key(self):
        result = extract_from_openai_chat({"prompt": "hello"}, InterceptConfig())
        assert result is None


# ============================================================================
# Anthropic messages format
# ============================================================================


class TestAnthropicMessagesFormat:
    def test_extract_string_system(self):
        body = {
            "model": "claude-3-opus-20240229",
            "system": "<documents><document>A</document><document>B</document></documents>",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        config = InterceptConfig()
        result = extract_from_anthropic_messages(body, config)
        assert result is not None
        assert result.documents == ["A", "B"]

    def test_extract_content_blocks_system(self):
        body = {
            "model": "claude-3-opus-20240229",
            "system": [
                {"type": "text", "text": "<documents><document>X</document><document>Y</document></documents>"}
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        config = InterceptConfig()
        result = extract_from_anthropic_messages(body, config)
        assert result is not None
        assert result.documents == ["X", "Y"]

    def test_no_system_field(self):
        body = {"messages": [{"role": "user", "content": "Hello"}]}
        result = extract_from_anthropic_messages(body, InterceptConfig())
        assert result is None

    def test_reconstruct_string_system(self):
        body = {
            "system": "<documents>\n<document>A</document>\n<document>B</document>\n</documents>",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        config = InterceptConfig()
        extraction = extract_from_anthropic_messages(body, config)
        new_body = reconstruct_anthropic_messages(body, extraction, ["B", "A"])
        assert "<document>B</document>" in new_body["system"]
        assert "<document>A</document>" in new_body["system"]
        # Original not modified
        assert body["system"].index("<document>A</document>") < body["system"].index("<document>B</document>")

    def test_reconstruct_content_blocks_system(self):
        body = {
            "system": [
                {"type": "text", "text": "<documents><document>P</document><document>Q</document></documents>"}
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        config = InterceptConfig()
        extraction = extract_from_anthropic_messages(body, config)
        new_body = reconstruct_anthropic_messages(body, extraction, ["Q", "P"])
        text_block = new_body["system"][0]["text"]
        assert "<document>Q</document>" in text_block
        assert "<document>P</document>" in text_block
