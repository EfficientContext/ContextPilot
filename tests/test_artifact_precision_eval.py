import json
import subprocess
import sys
from pathlib import Path

from scripts.evaluate_artifact_precision import build_report


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_artifact_precision_report_is_namespaced_and_reproducible():
    report = build_report()

    assert report["corpus"] == "synthetic_labeled_artifact_precision_v1"
    assert "precision" not in report
    assert "recall" not in report
    assert report["synthetic_event_precision"] == 1.0
    assert report["synthetic_event_recall"] == 1.0
    assert report["synthetic_case_accuracy"] == 1.0
    assert report["case_count"] >= 15
    assert report["synthetic_event_tp"] == report["expected_replacements"]
    assert report["synthetic_event_fp"] == 0
    assert report["synthetic_event_fn"] == 0
    assert report["mode_gate_checks"] == {
        "off_no_mutation": True,
        "shadow_no_mutation": True,
        "disable_env_no_mutation": True,
    }
    assert report["synthetic_negative_case_fpr"] == 0.0
    assert report["validation_gate_checks"]["forged_reference_detected"] is True


def test_synthetic_artifact_precision_cli_writes_json(tmp_path):
    out = tmp_path / "precision.json"
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_artifact_precision.py", "--output", str(out)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    stdout_report = json.loads(completed.stdout)
    file_report = json.loads(out.read_text())
    assert file_report == stdout_report
    assert stdout_report["synthetic_event_precision"] == 1.0
