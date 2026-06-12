"""Validation tests for the contextpilot-self-evolve Hermes skill.

These guard the SKILL.md packaging contract: valid YAML frontmatter, required
Hermes fields, size limits, and the presence of the safety/privacy phrases that
make this skill safe to ship (it must stay proposal-only and never promise to
auto-apply risky context changes).
"""
from pathlib import Path

import pytest

import yaml


SKILL_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "contextpilot-self-evolve"
    / "SKILL.md"
)

MAX_SKILL_CHARS = 100_000
MAX_DESCRIPTION_CHARS = 1024


def _read_skill():
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    # Split on the closing frontmatter fence.
    _, frontmatter, body = text.split("---\n", 2)
    meta = yaml.safe_load(frontmatter)
    return text, meta, body


def test_skill_file_exists():
    assert SKILL_PATH.is_file(), f"missing skill file: {SKILL_PATH}"


def test_skill_size_under_limit():
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert len(text) <= MAX_SKILL_CHARS, (
        f"SKILL.md is {len(text)} chars, exceeds {MAX_SKILL_CHARS}"
    )


def test_frontmatter_parses_and_has_required_fields():
    _, meta, _ = _read_skill()
    assert isinstance(meta, dict), "frontmatter must parse to a mapping"
    for field in ("name", "description", "version", "author", "license", "metadata"):
        assert field in meta, f"frontmatter missing required field: {field}"


def test_name_matches():
    _, meta, _ = _read_skill()
    assert meta["name"] == "contextpilot-self-evolve"


def test_description_is_use_when_and_within_limit():
    _, meta, _ = _read_skill()
    description = meta["description"]
    assert isinstance(description, str) and description.strip()
    assert description.lstrip().lower().startswith("use when"), (
        "description should start with 'Use when' per Hermes convention"
    )
    assert len(description) <= MAX_DESCRIPTION_CHARS, (
        f"description is {len(description)} chars, exceeds {MAX_DESCRIPTION_CHARS}"
    )


def test_metadata_has_tags():
    _, meta, _ = _read_skill()
    metadata = meta["metadata"]
    assert isinstance(metadata, dict)
    hermes_meta = metadata.get("hermes")
    assert isinstance(hermes_meta, dict), "metadata.hermes must be present"
    assert hermes_meta.get("tags"), "metadata.hermes.tags must be a non-empty list"
    assert isinstance(hermes_meta["tags"], list)


@pytest.mark.parametrize(
    "phrase",
    [
        # proposal-only / no auto-apply of risky changes
        "propose",
        "independent review",
        # privacy boundary
        "raw",
        "salted",
        "session ids",
        # realized vs advisory separation
        "advisory",
        "realized",
        # safe install convention
        "--force",
        # change-gate requirements
        "branch",
        "tests",
    ],
)
def test_required_safety_phrases_present(phrase):
    text, _, _ = _read_skill()
    assert phrase.lower() in text.lower(), f"SKILL.md missing safety phrase: {phrase!r}"


def test_does_not_promise_auto_apply():
    """The skill must keep its proposal-only stance for risky changes."""
    text, _, _ = _read_skill()
    lowered = text.lower()
    # Must explicitly disclaim auto-applying routing/drop/summarization.
    assert "never auto-apply" in lowered or "do not auto-apply" in lowered or (
        "not" in lowered and "auto-enable" in lowered
    ), "SKILL.md must state it never auto-applies risky context changes"
