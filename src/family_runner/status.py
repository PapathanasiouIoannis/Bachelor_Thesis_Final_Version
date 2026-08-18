"""Read-only profile validation and status assembly for the family workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, Sequence


ProfileLoader = Callable[[Path], dict[str, Any]]
HashMatcher = Callable[[Path, str], bool]
ProfileEntries = Callable[[dict[str, Any]], Sequence[Any]]
StatusProvider = Callable[..., dict[str, Any]]


class _StatusPaths(Protocol):
    @property
    def generation_profile_path(self) -> Path: ...

    @property
    def split_profile_path(self) -> Path: ...

    @property
    def model_profile_path(self) -> Path: ...


def load_profiles(
    paths: _StatusPaths,
    *,
    load_generation_profile: ProfileLoader,
    load_split_profile: ProfileLoader,
    load_model_profile: ProfileLoader,
    file_matches_sha256: HashMatcher,
) -> tuple[dict, dict, dict]:
    generation = load_generation_profile(paths.generation_profile_path)
    split = load_split_profile(paths.split_profile_path)
    model = load_model_profile(paths.model_profile_path)

    if split["expected_generation_profile"] != generation["profile_id"]:
        raise ValueError(
            "The family split profile expects generation profile "
            f"'{split['expected_generation_profile']}', but "
            f"'{generation['profile_id']}' was selected."
        )
    identity = model["data_identity"]
    if identity["generation_profile_id"] != generation["profile_id"]:
        raise ValueError(
            "The locked model profile refers to another generation profile."
        )
    if identity["split_profile_id"] != split["profile_id"]:
        raise ValueError(
            "The locked model profile refers to another family split profile."
        )
    expected_split_hash = identity.get("split_profile_sha256")
    if not isinstance(expected_split_hash, str) or not expected_split_hash:
        raise ValueError(
            "The locked model profile does not record a valid family split profile "
            "hash."
        )
    if not file_matches_sha256(paths.split_profile_path, expected_split_hash):
        raise ValueError(
            "The locked model profile refers to a modified family split profile."
        )
    return generation, split, model


def family_split_summary(
    generation: dict,
    split: dict,
    *,
    profile_entries: ProfileEntries,
) -> dict[str, Any]:
    amplitudes = generation["deformation"]["amplitudes"]
    entries = profile_entries(generation)
    entries_by_group: dict[str, list[Any]] = {}
    for entry in entries:
        entries_by_group.setdefault(str(entry.family_group_id), []).append(entry)

    summary: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        group_ids = [str(value) for value in split["splits"][split_name]]
        hadronic_groups = [value for value in group_ids if value.startswith("H_")]
        quark_groups = [value for value in group_ids if value.startswith("Q_")]
        eos_count = sum(
            len(entries_by_group.get(group_id, [])) for group_id in group_ids
        )
        summary[split_name] = {
            "family_groups": len(group_ids),
            "hadronic_family_groups": len(hadronic_groups),
            "quark_family_groups": len(quark_groups),
            "eos_baselines": eos_count,
            "expected_curves": eos_count * len(amplitudes),
            "group_ids": group_ids,
        }
    return summary


def family_workflow_status(
    paths: _StatusPaths,
    *,
    load_profiles: Callable[[_StatusPaths], tuple[dict, dict, dict]],
    family_split_summary: Callable[[dict, dict], dict[str, Any]],
    profile_entries: ProfileEntries,
    development_artifacts: StatusProvider,
    development_evidence_summary: StatusProvider,
    final_test_status: StatusProvider,
    model_set_claim: str,
    supported_reporting_models: Sequence[str],
    exploratory_models: Sequence[str],
) -> dict[str, Any]:
    generation, split, model = load_profiles(paths)
    amplitudes = [float(value) for value in generation["deformation"]["amplitudes"]]
    split_summary = family_split_summary(generation, split)
    return {
        "workflow": "family_classification",
        "scientific_scope": model_set_claim,
        "generation_profile": {
            "profile_id": generation["profile_id"],
            "path": str(paths.generation_profile_path),
            "hadronic_eos_baselines": len(generation["hadronic_eos_ids"]),
            "quark_eos_baselines": len(generation["quark_eos_ids"]),
            "family_groups": len(
                {str(entry.family_group_id) for entry in profile_entries(generation)}
            ),
            "deformation": {
                "amplitude_symbol": "A",
                "amplitudes": amplitudes,
                "center_energy_density_symbol": "epsilon_0",
                "center_energy_density_mev_fm3": float(
                    generation["deformation"]["epsilon0_mev_fm3"]
                ),
                "width_symbol": "sigma",
                "width_mev_fm3": float(generation["deformation"]["sigma_mev_fm3"]),
            },
            "expected_curves": (
                len(generation["hadronic_eos_ids"]) + len(generation["quark_eos_ids"])
            )
            * len(amplitudes),
        },
        "split_profile": {
            "profile_id": split["profile_id"],
            "path": str(paths.split_profile_path),
            "primary_split_unit": "physical EoS family",
            "splits": split_summary,
        },
        "reporting_model_policy": {
            "supported": list(supported_reporting_models),
            "exploratory_not_run_by_workflow": list(exploratory_models),
            "locked_model_profile_id": model["profile_id"],
        },
        "development_artifacts": development_artifacts(paths),
        "development_evidence": development_evidence_summary(paths),
        "final_test": final_test_status(paths, model),
    }


__all__ = [
    "family_split_summary",
    "family_workflow_status",
    "load_profiles",
]
