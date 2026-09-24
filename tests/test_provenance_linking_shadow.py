import json
import subprocess
import sys
from pathlib import Path

from contextpilot.provenance_linking import (
    ProvenanceBlock,
    ProvenanceExample,
    evaluate_shadow,
    example_from_trace_case,
    extract_claims,
    read_jsonl_examples,
    shadow_link_claims,
)
from scripts.build_provenance_linker_dataset import synthetic_examples

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = REPO_ROOT / "datasets/provenance_linking/synthetic_v1.jsonl"


def test_claim_extraction_and_shadow_linking_are_offline_only():
    tool = ProvenanceBlock("tool_1", "tool_result", "pytest: 592 passed, 37 skipped")
    assistant = ProvenanceBlock(
        "assistant_1",
        "assistant_context",
        "Full suite passed: 592 passed, 37 skipped.",
    )
    claims = extract_claims(assistant)
    assert len(claims) == 1

    example = ProvenanceExample("ex", "coding", [tool, assistant], claims)
    links = shadow_link_claims(example)
    assert len(links) == 1
    assert links[0].evidence_block_id == "tool_1"
    assert links[0].method == "shadow_lexical_v1"
    # The linker only emits metadata; it never mutates source/claim text.
    assert tool.text == "pytest: 592 passed, 37 skipped"
    assert assistant.text == "Full suite passed: 592 passed, 37 skipped."


def test_synthetic_provenance_dataset_is_training_ready_and_shadow_measured():
    examples = read_jsonl_examples(SYNTHETIC)
    assert len(examples) == 3
    assert sum(len(ex.gold_links) for ex in examples) == 4
    assert {ex.domain for ex in examples} == {"coding", "research", "ops"}

    report = evaluate_shadow(examples)
    assert report["claim_scope"] == "shadow claim→evidence linking; does not mutate online context"
    assert report["shadow_link_tp"] >= 3
    assert report["shadow_link_precision"] >= 0.7
    assert report["shadow_link_recall"] >= 0.7
    # This is deliberately a baseline, not a solved model: enough signal to train,
    # but not pretending rules solve the general provenance problem.
    assert report["shadow_link_precision"] < 1.0 or report["shadow_link_recall"] < 1.0


def test_dataset_builder_and_shadow_eval_clis(tmp_path):
    dataset = tmp_path / "prov.jsonl"
    built = subprocess.run(
        [
            sys.executable,
            "scripts/build_provenance_linker_dataset.py",
            "--mode",
            "synthetic",
            "--with-shadow",
            "--output",
            str(dataset),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    manifest = json.loads(built.stdout)
    assert manifest["dataset_kind"] == "provenance_linking"
    assert manifest["gold_link_count"] == 4
    assert dataset.exists()
    assert dataset.with_suffix(dataset.suffix + ".manifest.json").exists()

    report_path = tmp_path / "report.json"
    evaluated = subprocess.run(
        [sys.executable, "scripts/evaluate_provenance_shadow.py", str(dataset), "--output", str(report_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(evaluated.stdout) == json.loads(report_path.read_text())


def test_trace_case_converts_to_general_provenance_example():
    raw = {
        "case_id": "case_1",
        "messages": [
            {"role": "tool", "block_type": "tool_result", "content": "pytest says tests passed: 46 passed"},
            {"role": "assistant", "block_type": "assistant_context", "content": "Tests passed: 46 passed."},
        ],
    }
    example = example_from_trace_case(raw)
    assert example.example_id == "case_1"
    assert len(example.blocks) == 2
    assert len(example.claims) == 1
    assert shadow_link_claims(example)


def test_synthetic_examples_match_committed_dataset_shape():
    generated = synthetic_examples()
    committed = read_jsonl_examples(SYNTHETIC)
    assert [ex.example_id for ex in generated] == [ex.example_id for ex in committed]
    assert [len(ex.gold_links) for ex in generated] == [len(ex.gold_links) for ex in committed]
