from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

from src import family_workflow
from src.family_runner import status


def test_family_status_facade_preserves_contract_and_required_dependencies():
    expected_signatures = {
        "_load_profiles": (
            "(paths: 'FamilyWorkflowPaths') -> 'tuple[dict, dict, dict]'"
        ),
        "_family_split_summary": (
            "(generation: 'dict', split: 'dict') -> 'dict[str, Any]'"
        ),
        "family_workflow_status": (
            "(paths: 'FamilyWorkflowPaths | None' = None, **path_options: "
            "'str | Path') -> 'dict[str, Any]'"
        ),
    }
    assert {
        name: str(inspect.signature(getattr(family_workflow, name)))
        for name in expected_signatures
    } == expected_signatures

    expected_exports = {
        "family_split_summary",
        "family_workflow_status",
        "load_profiles",
    }
    assert set(status.__all__) == expected_exports
    assert len(status.__all__) == len(expected_exports)
    assert family_workflow._load_profiles is not status.load_profiles
    assert family_workflow._family_split_summary is not status.family_split_summary
    assert family_workflow.family_workflow_status is not status.family_workflow_status

    required_dependencies = {
        status.load_profiles: (
            "load_generation_profile",
            "load_split_profile",
            "load_model_profile",
            "file_matches_sha256",
        ),
        status.family_split_summary: ("profile_entries",),
        status.family_workflow_status: (
            "load_profiles",
            "family_split_summary",
            "profile_entries",
            "development_artifacts",
            "development_evidence_summary",
            "final_test_status",
            "model_set_claim",
            "supported_reporting_models",
            "exploratory_models",
        ),
    }
    for function, dependency_names in required_dependencies.items():
        parameters = inspect.signature(function).parameters
        for name in dependency_names:
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is inspect.Parameter.empty


def test_family_status_facade_dispatches_to_reloaded_leaf_with_live_dependencies(
    monkeypatch,
):
    importlib.reload(status)
    paths = object()
    generation = object()
    split = object()
    loaded = object()
    summarized = object()
    projected = object()
    events = []

    def generation_loader(path):
        return None

    def split_loader(path):
        return None

    def model_loader(path):
        return None

    def hash_matcher(path, expected):
        return True

    def entries_provider(profile):
        return ()

    def artifact_provider(value):
        return {}

    def evidence_provider(value):
        return {}

    def final_provider(value, model=None):
        return {}

    monkeypatch.setattr(family_workflow, "load_family_pilot_profile", generation_loader)
    monkeypatch.setattr(family_workflow, "load_family_split_profile", split_loader)
    monkeypatch.setattr(family_workflow, "load_locked_model_profile", model_loader)
    monkeypatch.setattr(family_workflow, "_file_matches_sha256", hash_matcher)
    monkeypatch.setattr(family_workflow, "profile_entries", entries_provider)
    monkeypatch.setattr(family_workflow, "_development_artifacts", artifact_provider)
    monkeypatch.setattr(
        family_workflow,
        "_development_evidence_summary",
        evidence_provider,
    )
    monkeypatch.setattr(family_workflow, "_final_test_status", final_provider)
    monkeypatch.setattr(family_workflow, "MODEL_SET_CLAIM", "live claim")
    monkeypatch.setattr(
        family_workflow,
        "SUPPORTED_REPORTING_MODELS",
        ("live supported",),
    )
    monkeypatch.setattr(
        family_workflow,
        "EXPLORATORY_MODELS",
        ("live exploratory",),
    )

    def fake_load_profiles(value, **dependencies):
        events.append(("load", value, dependencies))
        return loaded

    def fake_split_summary(generation_value, split_value, **dependencies):
        events.append(("split", generation_value, split_value, dependencies))
        return summarized

    def fake_status(value, **dependencies):
        events.append(("status", value, dependencies))
        return projected

    monkeypatch.setattr(status, "load_profiles", fake_load_profiles)
    monkeypatch.setattr(status, "family_split_summary", fake_split_summary)
    monkeypatch.setattr(status, "family_workflow_status", fake_status)

    assert family_workflow._load_profiles(paths) is loaded
    assert family_workflow._family_split_summary(generation, split) is summarized
    assert family_workflow.family_workflow_status(paths) is projected

    assert events[0] == (
        "load",
        paths,
        {
            "load_generation_profile": generation_loader,
            "load_split_profile": split_loader,
            "load_model_profile": model_loader,
            "file_matches_sha256": hash_matcher,
        },
    )
    assert events[1] == (
        "split",
        generation,
        split,
        {"profile_entries": entries_provider},
    )
    assert events[2] == (
        "status",
        paths,
        {
            "load_profiles": family_workflow._load_profiles,
            "family_split_summary": family_workflow._family_split_summary,
            "profile_entries": entries_provider,
            "development_artifacts": artifact_provider,
            "development_evidence_summary": evidence_provider,
            "final_test_status": final_provider,
            "model_set_claim": "live claim",
            "supported_reporting_models": ("live supported",),
            "exploratory_models": ("live exploratory",),
        },
    )


def test_family_status_path_options_use_live_facade_resolver(monkeypatch):
    resolved = object()
    projected = object()
    calls = []

    def resolve_paths(**options):
        calls.append(("resolve", options))
        return resolved

    def project_status(paths, **dependencies):
        calls.append(("status", paths, dependencies))
        return projected

    monkeypatch.setattr(family_workflow, "resolve_family_workflow_paths", resolve_paths)
    monkeypatch.setattr(status, "family_workflow_status", project_status)

    assert (
        family_workflow.family_workflow_status(
            data_root="relative-data",
            report_dir="relative-report",
        )
        is projected
    )
    assert calls[0] == (
        "resolve",
        {
            "data_root": "relative-data",
            "report_dir": "relative-report",
        },
    )
    assert calls[1][0:2] == ("status", resolved)


def test_real_family_status_leaf_projects_live_facade_reporting_constants(
    tmp_path,
    monkeypatch,
):
    paths = SimpleNamespace(
        generation_profile_path=tmp_path / "generation.json",
        split_profile_path=tmp_path / "split.json",
    )
    generation = {
        "profile_id": "generation-v1",
        "hadronic_eos_ids": [],
        "quark_eos_ids": [],
        "deformation": {
            "amplitudes": [0.0],
            "epsilon0_mev_fm3": 500.0,
            "sigma_mev_fm3": 100.0,
        },
    }
    split = {"profile_id": "split-v1"}
    model = {"profile_id": "model-v1"}

    def load_profiles(value):
        return generation, split, model

    def summarize_splits(generation_value, split_value):
        return {}

    def list_entries(generation_value):
        return ()

    def empty_status(value, model_value=None):
        return {}

    monkeypatch.setattr(family_workflow, "_load_profiles", load_profiles)
    monkeypatch.setattr(family_workflow, "_family_split_summary", summarize_splits)
    monkeypatch.setattr(family_workflow, "profile_entries", list_entries)
    monkeypatch.setattr(family_workflow, "_development_artifacts", empty_status)
    monkeypatch.setattr(
        family_workflow,
        "_development_evidence_summary",
        empty_status,
    )
    monkeypatch.setattr(family_workflow, "_final_test_status", empty_status)
    monkeypatch.setattr(family_workflow, "MODEL_SET_CLAIM", "live claim")
    monkeypatch.setattr(
        family_workflow,
        "SUPPORTED_REPORTING_MODELS",
        ("live supported",),
    )
    monkeypatch.setattr(
        family_workflow,
        "EXPLORATORY_MODELS",
        ("live exploratory",),
    )

    projected = family_workflow.family_workflow_status(paths)

    assert projected["scientific_scope"] == "live claim"
    assert projected["reporting_model_policy"] == {
        "supported": ["live supported"],
        "exploratory_not_run_by_workflow": ["live exploratory"],
        "locked_model_profile_id": "model-v1",
    }
