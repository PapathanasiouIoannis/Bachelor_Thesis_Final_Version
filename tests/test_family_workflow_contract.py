from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import family_workflow


EXPECTED_EXPORTS = {
    "EXPLORATORY_MODELS",
    "MODEL_SET_CLAIM",
    "SUPPORTED_REPORTING_MODELS",
    "FamilyDevelopmentStageError",
    "FamilyWorkflowError",
    "FamilyWorkflowPaths",
    "FinalTestAlreadyOpenedError",
    "assert_final_evaluation_not_opened",
    "family_workflow_status",
    "refuse_final_evaluation_request",
    "resolve_family_workflow_paths",
    "run_family_development",
}


def _paths(tmp_path: Path) -> family_workflow.FamilyWorkflowPaths:
    return family_workflow.FamilyWorkflowPaths(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_profile_path=tmp_path / "profiles" / "generation.json",
        split_profile_path=tmp_path / "profiles" / "split.json",
        model_profile_path=tmp_path / "profiles" / "model.json",
        report_dir=tmp_path / "reports",
    )


def test_family_workflow_public_surface_and_signatures_are_characterized():
    assert set(family_workflow.__all__) == EXPECTED_EXPORTS
    assert len(family_workflow.__all__) == len(EXPECTED_EXPORTS)
    assert issubclass(family_workflow.FamilyWorkflowError, RuntimeError)
    assert issubclass(
        family_workflow.FamilyDevelopmentStageError,
        family_workflow.FamilyWorkflowError,
    )
    assert issubclass(
        family_workflow.FinalTestAlreadyOpenedError,
        family_workflow.FamilyWorkflowError,
    )
    assert family_workflow.FamilyWorkflowPaths.__dataclass_params__.frozen is True
    assert tuple(
        field.name for field in fields(family_workflow.FamilyWorkflowPaths)
    ) == (
        "project_root",
        "data_root",
        "generation_profile_path",
        "split_profile_path",
        "model_profile_path",
        "report_dir",
    )

    resolve_signature = inspect.signature(family_workflow.resolve_family_workflow_paths)
    assert tuple(resolve_signature.parameters) == (
        "data_root",
        "generation_profile_path",
        "split_profile_path",
        "model_profile_path",
        "report_dir",
        "project_root",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in resolve_signature.parameters.values()
    )
    assert resolve_signature.parameters["data_root"].default is (
        family_workflow.DEFAULT_FAMILY_DATA_ROOT
    )
    assert resolve_signature.parameters["generation_profile_path"].default is (
        family_workflow.DEFAULT_PROFILE_PATH
    )
    assert resolve_signature.parameters["split_profile_path"].default is (
        family_workflow.DEFAULT_FAMILY_SPLIT_PROFILE
    )
    assert resolve_signature.parameters["model_profile_path"].default is (
        family_workflow.DEFAULT_MODEL_PROFILE
    )
    assert resolve_signature.parameters["report_dir"].default is (
        family_workflow.DEFAULT_REPORT_DIR
    )
    assert resolve_signature.parameters["project_root"].default is (
        family_workflow.PROJECT_ROOT
    )

    assert str(inspect.signature(family_workflow.family_workflow_status)) == (
        "(paths: 'FamilyWorkflowPaths | None' = None, "
        "**path_options: 'str | Path') -> 'dict[str, Any]'"
    )
    assert str(
        inspect.signature(family_workflow.assert_final_evaluation_not_opened)
    ) == (
        "(paths: 'FamilyWorkflowPaths | None' = None, "
        "**path_options: 'str | Path') -> 'None'"
    )
    assert str(inspect.signature(family_workflow.run_family_development)) == (
        "(paths: 'FamilyWorkflowPaths | None' = None, *, jobs: 'int' = 1, "
        "force_regenerate: 'bool' = False, permutations: 'int' = 0, "
        "requested_stage: 'str' = 'development', "
        "**path_options: 'str | Path') -> 'dict[str, Any]'"
    )
    assert str(inspect.signature(family_workflow.refuse_final_evaluation_request)) == (
        "(paths: 'FamilyWorkflowPaths | None' = None, "
        "**path_options: 'str | Path') -> 'NoReturn'"
    )

    assert family_workflow.SUPPORTED_REPORTING_MODELS == (
        "dummy baseline",
        "logistic regression",
    )
    assert family_workflow.EXPLORATORY_MODELS == (
        "XGBoost",
        "multilayer perceptron",
    )
    assert family_workflow.MODEL_SET_CLAIM == (
        "Repository hadronic-surrogate versus analytic CFL MIT-bag model-set "
        "discrimination on unseen EoS families; not universal matter-phase "
        "identification and not observational deployment."
    )


def test_path_resolution_is_absolute_and_has_no_filesystem_side_effects(tmp_path):
    root = tmp_path / "new-project"

    paths = family_workflow.resolve_family_workflow_paths(
        project_root=root,
        data_root="runtime/data",
        generation_profile_path="profiles/generation.json",
        split_profile_path="profiles/split.json",
        model_profile_path="profiles/model.json",
        report_dir="reports",
    )

    assert paths == family_workflow.FamilyWorkflowPaths(
        project_root=root.resolve(),
        data_root=(root / "runtime/data").resolve(),
        generation_profile_path=(root / "profiles/generation.json").resolve(),
        split_profile_path=(root / "profiles/split.json").resolve(),
        model_profile_path=(root / "profiles/model.json").resolve(),
        report_dir=(root / "reports").resolve(),
    )
    assert paths.family_ml_dir == (root / "runtime/data/family_ml").resolve()
    assert (
        paths.final_test_marker_path
        == (root / "runtime/data/family_ml/LOCKED_TEST_OPENED.json").resolve()
    )
    assert (
        paths.final_test_result_path
        == (root / "reports/family_final_test.json").resolve()
    )
    assert not root.exists()

    defaults_with_another_root = family_workflow.resolve_family_workflow_paths(
        project_root=root
    )
    assert defaults_with_another_root.data_root == (
        family_workflow.DEFAULT_FAMILY_DATA_ROOT
    )
    assert defaults_with_another_root.generation_profile_path == (
        family_workflow.DEFAULT_PROFILE_PATH
    )
    assert defaults_with_another_root.split_profile_path == (
        family_workflow.DEFAULT_FAMILY_SPLIT_PROFILE
    )
    assert defaults_with_another_root.model_profile_path == (
        family_workflow.DEFAULT_MODEL_PROFILE
    )
    assert defaults_with_another_root.report_dir == family_workflow.DEFAULT_REPORT_DIR
    assert not root.exists()


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("valid", None),
        (
            "split_generation_mismatch",
            "The family split profile expects generation profile 'other', but "
            "'generation-a' was selected.",
        ),
        (
            "model_generation_mismatch",
            "The locked model profile refers to another generation profile.",
        ),
        (
            "model_split_mismatch",
            "The locked model profile refers to another family split profile.",
        ),
        (
            "all_mismatch",
            "The family split profile expects generation profile 'other', but "
            "'generation-a' was selected.",
        ),
        (
            "missing_split_hash",
            "The locked model profile does not record a valid family split profile "
            "hash.",
        ),
        (
            "non_string_split_hash",
            "The locked model profile does not record a valid family split profile "
            "hash.",
        ),
        (
            "split_hash_mismatch",
            "The locked model profile refers to a modified family split profile.",
        ),
    ],
)
def test_profile_loading_order_and_identity_validation_precedence(
    tmp_path,
    monkeypatch,
    case,
    expected_error,
):
    paths = _paths(tmp_path)
    generation = {"profile_id": "generation-a"}
    split = {
        "profile_id": "split-a",
        "expected_generation_profile": "generation-a",
    }
    model = {
        "data_identity": {
            "generation_profile_id": "generation-a",
            "split_profile_id": "split-a",
            "split_profile_sha256": "expected-split-hash",
        }
    }
    if case in {"split_generation_mismatch", "all_mismatch"}:
        split["expected_generation_profile"] = "other"
    if case in {"model_generation_mismatch", "all_mismatch"}:
        model["data_identity"]["generation_profile_id"] = "other"
    if case in {"model_split_mismatch", "all_mismatch"}:
        model["data_identity"]["split_profile_id"] = "other"
    if case == "missing_split_hash":
        model["data_identity"].pop("split_profile_sha256")
    if case == "non_string_split_hash":
        model["data_identity"]["split_profile_sha256"] = ["not", "a", "hash"]

    events = []

    def generation_loader(path):
        events.append(("generation", path))
        return generation

    def split_loader(path):
        events.append(("split", path))
        return split

    def model_loader(path):
        events.append(("model", path))
        return model

    def matches_hash(path, expected_hash):
        events.append(("split_hash", path, expected_hash))
        return case != "split_hash_mismatch"

    monkeypatch.setattr(family_workflow, "load_family_pilot_profile", generation_loader)
    monkeypatch.setattr(family_workflow, "load_family_split_profile", split_loader)
    monkeypatch.setattr(family_workflow, "load_locked_model_profile", model_loader)
    monkeypatch.setattr(family_workflow, "_file_matches_sha256", matches_hash)

    if expected_error is None:
        loaded = family_workflow._load_profiles(paths)
        assert loaded[0] is generation
        assert loaded[1] is split
        assert loaded[2] is model
    else:
        with pytest.raises(ValueError) as raised:
            family_workflow._load_profiles(paths)
        assert str(raised.value) == expected_error

    expected_events = [
        ("generation", paths.generation_profile_path),
        ("split", paths.split_profile_path),
        ("model", paths.model_profile_path),
    ]
    if case in {"valid", "split_hash_mismatch"}:
        expected_events.append(
            (
                "split_hash",
                paths.split_profile_path,
                "expected-split-hash",
            )
        )
    assert events == expected_events


def test_family_split_summary_preserves_split_and_group_order(monkeypatch):
    generation = {"deformation": {"amplitudes": [-0.1, 0.0, 0.1]}}
    split = {
        "splits": {
            "train": ["H_ALPHA", "Q_ALPHA", "UNKNOWN"],
            "val": ["Q_BETA"],
            "test": ["H_BETA"],
        }
    }
    entries = [
        SimpleNamespace(family_group_id="H_ALPHA"),
        SimpleNamespace(family_group_id="H_ALPHA"),
        SimpleNamespace(family_group_id="Q_ALPHA"),
        SimpleNamespace(family_group_id="Q_BETA"),
        SimpleNamespace(family_group_id="H_BETA"),
    ]
    calls = []

    def fake_profile_entries(value):
        calls.append(value)
        return entries

    monkeypatch.setattr(family_workflow, "profile_entries", fake_profile_entries)

    summary = family_workflow._family_split_summary(generation, split)

    assert tuple(summary) == ("train", "val", "test")
    assert summary == {
        "train": {
            "family_groups": 3,
            "hadronic_family_groups": 1,
            "quark_family_groups": 1,
            "eos_baselines": 3,
            "expected_curves": 9,
            "group_ids": ["H_ALPHA", "Q_ALPHA", "UNKNOWN"],
        },
        "val": {
            "family_groups": 1,
            "hadronic_family_groups": 0,
            "quark_family_groups": 1,
            "eos_baselines": 1,
            "expected_curves": 3,
            "group_ids": ["Q_BETA"],
        },
        "test": {
            "family_groups": 1,
            "hadronic_family_groups": 1,
            "quark_family_groups": 0,
            "eos_baselines": 1,
            "expected_curves": 3,
            "group_ids": ["H_BETA"],
        },
    }
    assert calls == [generation]


def test_status_schema_and_provider_order_are_characterized(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    generation = {
        "profile_id": "generation-v1",
        "hadronic_eos_ids": ["H1", "H2"],
        "quark_eos_ids": ["Q1"],
        "deformation": {
            "amplitudes": ["-0.25", 0, "0.5"],
            "epsilon0_mev_fm3": "500.5",
            "sigma_mev_fm3": 100,
        },
    }
    split = {"profile_id": "split-v1"}
    model = {"profile_id": "model-v1"}
    split_summary = {"train": {"sentinel": "split-summary"}}
    artifacts = {"sentinel": "artifacts"}
    development_evidence = {"sentinel": "development-evidence"}
    final_test = {"sentinel": "final-test"}
    entries = [
        SimpleNamespace(family_group_id="H_GROUP"),
        SimpleNamespace(family_group_id="H_GROUP"),
        SimpleNamespace(family_group_id="Q_GROUP"),
    ]
    events = []

    def load_profiles(value):
        events.append(("load_profiles", value))
        return generation, split, model

    def summarize_splits(generation_value, split_value):
        events.append(("split_summary", generation_value, split_value))
        return split_summary

    def list_entries(generation_value):
        events.append(("profile_entries", generation_value))
        return entries

    def development_artifacts(value):
        events.append(("development_artifacts", value))
        return artifacts

    def evidence_summary(value):
        events.append(("development_evidence", value))
        return development_evidence

    def final_status(value, model_value):
        events.append(("final_test", value, model_value))
        return final_test

    monkeypatch.setattr(family_workflow, "_load_profiles", load_profiles)
    monkeypatch.setattr(family_workflow, "_family_split_summary", summarize_splits)
    monkeypatch.setattr(family_workflow, "profile_entries", list_entries)
    monkeypatch.setattr(
        family_workflow, "_development_artifacts", development_artifacts
    )
    monkeypatch.setattr(
        family_workflow, "_development_evidence_summary", evidence_summary
    )
    monkeypatch.setattr(family_workflow, "_final_test_status", final_status)

    status = family_workflow.family_workflow_status(paths)

    assert tuple(status) == (
        "workflow",
        "scientific_scope",
        "generation_profile",
        "split_profile",
        "reporting_model_policy",
        "development_artifacts",
        "development_evidence",
        "final_test",
    )
    assert status == {
        "workflow": "family_classification",
        "scientific_scope": family_workflow.MODEL_SET_CLAIM,
        "generation_profile": {
            "profile_id": "generation-v1",
            "path": str(paths.generation_profile_path),
            "hadronic_eos_baselines": 2,
            "quark_eos_baselines": 1,
            "family_groups": 2,
            "deformation": {
                "amplitude_symbol": "A",
                "amplitudes": [-0.25, 0.0, 0.5],
                "center_energy_density_symbol": "epsilon_0",
                "center_energy_density_mev_fm3": 500.5,
                "width_symbol": "sigma",
                "width_mev_fm3": 100.0,
            },
            "expected_curves": 9,
        },
        "split_profile": {
            "profile_id": "split-v1",
            "path": str(paths.split_profile_path),
            "primary_split_unit": "physical EoS family",
            "splits": split_summary,
        },
        "reporting_model_policy": {
            "supported": ["dummy baseline", "logistic regression"],
            "exploratory_not_run_by_workflow": [
                "XGBoost",
                "multilayer perceptron",
            ],
            "locked_model_profile_id": "model-v1",
        },
        "development_artifacts": artifacts,
        "development_evidence": development_evidence,
        "final_test": final_test,
    }
    assert events == [
        ("load_profiles", paths),
        ("split_summary", generation, split),
        ("profile_entries", generation),
        ("development_artifacts", paths),
        ("development_evidence", paths),
        ("final_test", paths, model),
    ]


def test_status_rejects_paths_and_path_options_before_provider_calls(
    tmp_path,
    monkeypatch,
):
    paths = _paths(tmp_path)
    calls = []
    monkeypatch.setattr(
        family_workflow,
        "_load_profiles",
        lambda value: calls.append(value),
    )

    with pytest.raises(TypeError) as raised:
        family_workflow.family_workflow_status(paths, report_dir="elsewhere")

    assert str(raised.value) == (
        "Pass either 'paths' or individual path options, not both."
    )
    assert calls == []


@pytest.mark.parametrize("use_resolved_paths", (True, False))
def test_final_refusal_forwards_arguments_and_propagates_guard_error_by_identity(
    tmp_path,
    monkeypatch,
    use_resolved_paths,
):
    paths = _paths(tmp_path)
    path_options = {} if use_resolved_paths else {"report_dir": tmp_path / "reports"}
    passed_paths = paths if use_resolved_paths else None
    signal = family_workflow.FinalTestAlreadyOpenedError("guard refusal")
    calls = []

    def guard(value=None, **options):
        calls.append((value, options))
        raise signal

    monkeypatch.setattr(
        family_workflow,
        "assert_final_evaluation_not_opened",
        guard,
    )

    with pytest.raises(family_workflow.FinalTestAlreadyOpenedError) as caught:
        family_workflow.refuse_final_evaluation_request(
            passed_paths,
            **path_options,
        )

    assert caught.value is signal
    assert calls == [(passed_paths, path_options)]
