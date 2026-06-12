"""Validation tests for the contextpilot-savings Hermes skill.

These guard the SKILL.md packaging contract: valid YAML frontmatter, the
required Hermes fields, the description size limit, and the presence of the
user-facing behavior (how to run the savings script, the plugin-path fallback,
and the privacy boundary). They also assert the skill stays narrow and
read-only: it must never reintroduce the removed self-evolve workflow or any
cron / branch / pull-request automation instructions.
"""
from pathlib import Path

import pytest

import yaml


SKILL_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "contextpilot-savings"
    / "SKILL.md"
)

MAX_SKILL_CHARS = 100_000
MAX_DESCRIPTION_CHARS = 1024


def _read_skill():
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    # Split on the closing frontmatter fence: ['', frontmatter, body].
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
    assert meta["name"] == "contextpilot-savings"


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


def test_metadata_hermes_tags_and_related_skills():
    _, meta, _ = _read_skill()
    metadata = meta["metadata"]
    assert isinstance(metadata, dict)
    hermes_meta = metadata.get("hermes")
    assert isinstance(hermes_meta, dict), "metadata.hermes must be present"
    assert isinstance(hermes_meta.get("tags"), list) and hermes_meta["tags"], (
        "metadata.hermes.tags must be a non-empty list"
    )
    assert "related_skills" in hermes_meta, (
        "metadata.hermes.related_skills must be present"
    )
    assert isinstance(hermes_meta["related_skills"], list)


@pytest.mark.parametrize(
    "phrase",
    [
        # how to run the reporter
        "scripts/contextpilot_savings.py",
        "python3 <script> --since-hours 24",
        "--all-time",
        "--format json",
        # plugin-path fallback
        "~/.hermes/plugins/ContextPilot/scripts/contextpilot_savings.py",
        # no-events guidance
        "Restart Hermes",
        # install convention note
        "--force",
    ],
)
def test_body_includes_required_invocation(phrase):
    _, _, body = _read_skill()
    assert phrase.lower() in body.lower(), f"SKILL.md body missing: {phrase!r}"


@pytest.mark.parametrize(
    "phrase",
    [
        "privacy boundary",
        "metadata-only",
        "state.db",
        "system prompts",
        "raw session ids",
    ],
)
def test_body_includes_privacy_boundary(phrase):
    _, _, body = _read_skill()
    assert phrase.lower() in body.lower(), (
        f"SKILL.md body missing privacy boundary phrase: {phrase!r}"
    )


@pytest.mark.parametrize(
    "forbidden",
    [
        # must not reintroduce the removed self-evolve workflow
        "self-evolve",
        "self_evolve",
        "self-envolve",
        "evolve",
        # must not instruct cron / branch / PR automation
        "cron",
        "git branch",
        "branch",
        "pull request",
    ],
)
def test_body_excludes_self_evolve_and_automation(forbidden):
    _, _, body = _read_skill()
    assert forbidden.lower() not in body.lower(), (
        f"SKILL.md body must not contain forbidden term: {forbidden!r}"
    )
