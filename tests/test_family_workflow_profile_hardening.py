from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import family_workflow


ROOT = Path(__file__).resolve().parents[1]


def _copied_profile_paths(tmp_path: Path) -> family_workflow.FamilyWorkflowPaths:
    framework = tmp_path / "framework"
    framework.mkdir()
    for name in (
        "family_pilot_profile.json",
        "family_split_profile.json",
        "family_model_profile.json",
    ):
        (framework / name).write_bytes((ROOT / "framework" / name).read_bytes())
    return family_workflow.resolve_family_workflow_paths(
        project_root=tmp_path,
        data_root="data/family_pilot_v1",
        generation_profile_path="framework/family_pilot_profile.json",
        split_profile_path="framework/family_split_profile.json",
        model_profile_path="framework/family_model_profile.json",
        report_dir="docs",
    )


@pytest.mark.parametrize("newline", (b"\n", b"\r\n"), ids=("lf", "crlf"))
def test_locked_split_profile_hash_accepts_git_newline_materializations(
    tmp_path,
    newline,
):
    paths = _copied_profile_paths(tmp_path)
    raw = paths.split_profile_path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    paths.split_profile_path.write_bytes(normalized.replace(b"\n", newline))

    generation, split, model = family_workflow._load_profiles(paths)

    assert split["profile_id"] == model["data_identity"]["split_profile_id"]
    assert split["expected_generation_profile"] == generation["profile_id"]


def test_locked_split_profile_hash_rejects_same_id_content_mutation(tmp_path):
    paths = _copied_profile_paths(tmp_path)
    split = json.loads(paths.split_profile_path.read_text(encoding="utf-8"))
    split["test_only_same_id_mutation"] = True
    paths.split_profile_path.write_text(
        json.dumps(split, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as caught:
        family_workflow._load_profiles(paths)

    assert caught.value.args == (
        "The locked model profile refers to a modified family split profile.",
    )
