"""One professional command-line interface for the supported EoS workflows."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path
from typing import Any

from framework.eos_catalog import CFL_CATALOG, HADRONIC_CATALOG
from src.eoslab_runtime import PROJECT_ROOT, export_summary, run_status
from src.experiment_config import (
    ConfigurationError,
    FamilyClassificationSpec,
    PairExperimentSpec,
    load_experiment_config,
    resolve_pair_experiment,
)

DEFAULT_FAMILY_CONFIG = PROJECT_ROOT / "configs" / "family_classification.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eoslab.py",
        description=(
            "Configure, validate, and run controlled compact-star EoS sensitivity "
            "experiments and the audited family-development workflow."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "doctor", help="check Python, dependencies, inputs, and output access"
    )
    commands.add_parser("list-eos", help="list supported hadronic and CFL baselines")

    validate = commands.add_parser(
        "validate", help="validate and display a configuration without generating stars"
    )
    validate.add_argument("config", type=Path)

    run = commands.add_parser("run", help="run one paired EoS sensitivity experiment")
    run.add_argument("config", type=Path)
    run.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="override only the number of parallel jobs; the override is recorded",
    )
    run.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="optional run root; defaults to the ignored repository runs directory",
    )

    status = commands.add_parser("status", help="show completion and artifact status")
    status.add_argument("run_directory", type=Path)

    export = commands.add_parser(
        "export-summary",
        help="copy compact review artifacts from an artifact-complete terminal run",
    )
    export.add_argument("run_directory", type=Path)
    export.add_argument("--reports-root", type=Path, default=None)

    family = commands.add_parser(
        "family", help="operate the audited family-level development workflow"
    )
    family_commands = family.add_subparsers(dest="family_command", required=True)
    develop = family_commands.add_parser(
        "develop", help="run development stages without opening the locked final test"
    )
    develop.add_argument("config", type=Path)
    _add_family_path_arguments(develop)
    develop.add_argument("--jobs", type=int, default=1)
    develop.add_argument("--permutations", type=int, default=0)
    develop.add_argument("--force-regenerate", action="store_true")

    family_status_parser = family_commands.add_parser(
        "status", help="show profiles, development artifacts, and final-test lock state"
    )
    family_status_parser.add_argument(
        "config", type=Path, nargs="?", default=DEFAULT_FAMILY_CONFIG
    )
    _add_family_path_arguments(family_status_parser)
    return parser


def _add_family_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=Path("data/family_pilot_v1"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs"))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor()
        if args.command == "list-eos":
            _list_eos()
            return 0
        if args.command == "validate":
            return _validate(args.config)
        if args.command == "run":
            from src.physics.experiment_runner import run_pair_experiment

            specification = load_experiment_config(args.config)
            if not isinstance(specification, PairExperimentSpec):
                raise ConfigurationError(
                    "The selected profile is for family classification. Use "
                    "'eoslab.py family develop' instead."
                )
            layout = run_pair_experiment(
                resolve_pair_experiment(specification),
                parallel_jobs=args.jobs,
                runs_root=args.runs_root,
            )
            print(f"Run completed: {layout.root}")
            print(f"Report: {layout.report}")
            return 0
        if args.command == "status":
            _print_pair_status(run_status(args.run_directory))
            return 0
        if args.command == "export-summary":
            destination = export_summary(
                args.run_directory, reports_root=args.reports_root
            )
            print(f"Compact summary exported to: {destination}")
            return 0
        if args.command == "family":
            return _family(args)
    except (ConfigurationError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    parser.error("Unsupported command.")
    return 2


def _doctor() -> int:
    checks: list[tuple[str, bool, str]] = []
    version = sys.version_info[:2]
    supported = (3, 11) <= version <= (3, 13)
    checks.append(
        (
            "Python version",
            supported,
            f"{platform.python_version()} at {Path(sys.executable).resolve()}",
        )
    )
    imports = {
        "NumPy": "numpy",
        "pandas": "pandas",
        "SciPy": "scipy",
        "Matplotlib": "matplotlib",
        "PyArrow": "pyarrow",
        "joblib": "joblib",
        "scikit-learn": "sklearn",
        "Numba": "numba",
    }
    for display, module_name in imports.items():
        try:
            module = importlib.import_module(module_name)
            version_text = getattr(module, "__version__", "installed")
            checks.append((display, True, str(version_text)))
        except Exception as error:
            checks.append((display, False, str(error)))
    try:
        from src.physics.get_eos_library import get_eos_library

        available_hadronic = set(get_eos_library()[0])
        missing_inputs = sorted(
            entry.eos_id
            for entry in HADRONIC_CATALOG
            if entry.eos_id not in available_hadronic
        )
    except Exception as error:
        missing_inputs = [f"library load failed: {error}"]
    checks.append(
        (
            "Hadronic baseline library",
            not missing_inputs,
            "all analytic baselines available"
            if not missing_inputs
            else "missing: " + ", ".join(missing_inputs),
        )
    )
    writable = os.access(PROJECT_ROOT, os.W_OK)
    checks.append(("Repository output access", writable, str(PROJECT_ROOT)))
    _print_table(
        ("Check", "Status", "Details"),
        [
            (name, "PASS" if passed else "FAIL", detail)
            for name, passed, detail in checks
        ],
    )
    passed = all(record[1] for record in checks)
    print("\nReady." if passed else "\nOne or more required checks failed.")
    return 0 if passed else 1


def _list_eos() -> None:
    print("Hadronic repository surrogates")
    _print_table(
        ("EoS", "Model family", "Formula provenance"),
        [
            (
                entry.eos_id,
                entry.model_family,
                "primary formula verified"
                if entry.exact_formula_primary_verified
                else "repository fit; see provenance note",
            )
            for entry in HADRONIC_CATALOG
        ],
    )
    print("\nPublished analytic CFL MIT-bag tuples")
    _print_table(
        ("EoS", "B [MeV fm^-3]", "Delta [MeV]", "m_s [MeV]"),
        [
            (
                entry.eos_id,
                f"{entry.bag_b_mev_fm3:g}",
                f"{entry.gap_delta_mev:g}",
                f"{entry.strange_mass_mev:g}",
            )
            for entry in CFL_CATALOG
        ],
    )


def _validate(config_path: Path) -> int:
    specification = load_experiment_config(config_path)
    if isinstance(specification, FamilyClassificationSpec):
        print(
            "Configuration is valid: audited family-classification development workflow."
        )
        status = _family_status_from_spec(
            specification, Path("data/family_pilot_v1"), Path("docs")
        )
        _print_family_status(status)
        return 0
    from src.physics.experiment_runner import validate_pair_experiment

    preflight = validate_pair_experiment(resolve_pair_experiment(specification))
    _print_preflight(preflight.to_dict())
    return 0


def _print_preflight(preflight: dict[str, Any]) -> None:
    deformation = preflight["deformation"]
    quark = preflight["quark_eos"]
    screens = preflight["physical_requirements"]
    numerical = preflight["resolved_numerical_settings"]
    intervals = preflight["admissible_amplitude_intervals"]
    print("Configuration is valid.\n")
    rows = [
        ("Experiment", preflight["experiment_name"]),
        ("Workflow", preflight["workflow"]),
        ("Mode", preflight["mode"]),
        (
            "Hadronic EoS",
            preflight["hadronic_eos"]["baseline"] + " repository surrogate",
        ),
        ("Quark EoS", quark["catalog_identifier"] + " analytic CFL MIT-bag"),
        ("Bag constant B", f"{quark['bag_constant_mev_fm3']:g} MeV fm^-3"),
        ("Pairing gap Delta", f"{quark['pairing_gap_mev']:g} MeV"),
        ("Strange-quark mass m_s", f"{quark['strange_quark_mass_mev']:g} MeV"),
        (
            "Deformation method",
            "additive Gaussian change to squared sound speed",
        ),
        (
            "Deformation centre epsilon_0",
            f"{deformation['center_energy_density_mev_fm3']:g} MeV fm^-3",
        ),
        ("Deformation width sigma", f"{deformation['width_mev_fm3']:g} MeV fm^-3"),
        (
            "Amplitude values A",
            ", ".join(f"{value:g}" for value in deformation["amplitudes"]),
        ),
        ("Amplitude A units", "dimensionless"),
        ("Numerical preset", preflight["numerical_settings"]["preset"]),
        ("EoS grid points", str(numerical["eos_grid_points"])),
        (
            "Central-pressure points",
            str(numerical["central_pressure_points"]),
        ),
        (
            "TOV relative tolerance",
            f"{numerical['tov_relative_tolerance']:.3g}",
        ),
        (
            "TOV absolute tolerance",
            f"{numerical['tov_absolute_tolerance']:.3g}",
        ),
        (
            "Convergence check",
            preflight["numerical_settings"]["convergence_check"],
        ),
        ("Random seed", str(preflight["execution"]["random_seed"])),
        ("Parallel jobs", str(preflight["execution"]["parallel_jobs"])),
        (
            "Amplitudes per worker batch",
            str(preflight["execution"]["amplitudes_per_batch"]),
        ),
        ("Expected amplitude pairs", str(len(deformation["amplitudes"]))),
        ("Expected curves", str(preflight["expected_curves"])),
        (
            "Maximum-mass requirement",
            f"{screens['minimum_maximum_mass_msun']:g} to {screens['maximum_maximum_mass_msun']:g} M_sun",
        ),
        (
            "Radius at 1.4 M_sun",
            f"{screens['radius_1p4_min_km']:g} to {screens['radius_1p4_max_km']:g} km",
        ),
        (
            f"A = 0 {preflight['hadronic_eos']['baseline']} pressure recovery",
            f"{preflight['baseline_recovery_maximum_relative_pressure_error']['hadronic']:.3g} maximum relative error",
        ),
        (
            "A = 0 CFL pressure recovery",
            f"{preflight['baseline_recovery_maximum_relative_pressure_error']['quark']:.3g} maximum relative error",
        ),
        (
            "Hadronic permitted A interval",
            f"({intervals['hadronic'][0]:.6g}, {intervals['hadronic'][1]:.6g}]",
        ),
        (
            "Quark permitted A interval",
            f"({intervals['quark'][0]:.6g}, {intervals['quark'][1]:.6g}]",
        ),
        (
            "Common permitted A interval",
            f"({intervals['common'][0]:.6g}, {intervals['common'][1]:.6g}]",
        ),
        (
            "Output location",
            str(
                PROJECT_ROOT
                / "runs"
                / preflight["experiment_name"]
                / "<timestamp>-<hash>"
            ),
        ),
    ]
    _print_table(("Setting", "Resolved value"), rows)
    print("\nScientific interpretation:")
    print(preflight["permitted_scientific_interpretation"])
    print(
        "\nClassification: disabled for this one-baseline-per-class sensitivity study."
    )
    hadronic = preflight["provenance"]["hadronic"]
    quark_provenance = preflight["provenance"]["quark"]
    print(f"\nHadronic fit source: {hadronic['fit_source_title']}")
    print(f"Hadronic provenance note: {hadronic['provenance_note']}")
    quark_source = quark_provenance.get(
        "underlying_primary_title", "custom exploratory tuple; no named catalog source"
    )
    print(f"Quark source: {quark_source}")
    print(f"Quark provenance note: {quark_provenance['provenance_note']}")


def _print_pair_status(status: dict[str, Any]) -> None:
    print(f"Experiment: {status.get('experiment_name')}")
    print(f"Status: {status.get('status')}")
    print(f"Run directory: {status.get('run_directory')}")
    integrity = status["artifact_integrity"]
    print(f"Artifact integrity: {integrity['state']}")
    source_match = status.get("source_tree_matches_run")
    source_text = "not recorded" if source_match is None else str(source_match)
    print(f"Current source matches run: {source_text}")
    if integrity["state"] == "invalid":
        for key in ("missing", "mismatched", "unsafe_paths"):
            if integrity[key]:
                print(
                    f"{key.replace('_', ' ').capitalize()}: {', '.join(integrity[key])}"
                )
    rows = [
        (name, "present" if record["present"] else "missing", record["path"])
        for name, record in status["artifacts"].items()
    ]
    print()
    _print_table(("Artifact", "State", "Path"), rows)


def _family(args: argparse.Namespace) -> int:
    from src.family_workflow import family_workflow_status, run_family_development

    specification = load_experiment_config(args.config)
    if not isinstance(specification, FamilyClassificationSpec):
        raise ConfigurationError(
            "The selected profile is a paired sensitivity experiment, not the "
            "family-classification workflow."
        )
    paths = _family_paths_from_spec(specification, args.data_root, args.report_dir)
    if args.family_command == "status":
        _print_family_status(family_workflow_status(paths))
        return 0
    result = run_family_development(
        paths,
        jobs=args.jobs,
        permutations=args.permutations,
        force_regenerate=args.force_regenerate,
    )
    print("Family development stages completed without opening the final test.")
    for stage in result["completed_stages"]:
        print(f"- {stage['stage']}: completed")
    return 0


def _family_status_from_spec(
    specification: FamilyClassificationSpec, data_root: Path, report_dir: Path
) -> dict[str, Any]:
    from src.family_workflow import family_workflow_status

    return family_workflow_status(
        _family_paths_from_spec(specification, data_root, report_dir)
    )


def _family_paths_from_spec(
    specification: FamilyClassificationSpec,
    data_root: Path,
    report_dir: Path,
):
    from src.family_workflow import resolve_family_workflow_paths

    return resolve_family_workflow_paths(
        data_root=data_root,
        generation_profile_path=specification.profiles.generation_profile,
        split_profile_path=specification.profiles.split_profile,
        model_profile_path=specification.profiles.model_profile,
        report_dir=report_dir,
    )


def _print_family_status(status: dict[str, Any]) -> None:
    generation = status["generation_profile"]
    final = status["final_test"]
    print("Family-classification development workflow\n")
    rows = [
        ("Hadronic baselines", generation["hadronic_eos_baselines"]),
        ("Quark baselines", generation["quark_eos_baselines"]),
        ("Configured split-family groups", generation["family_groups"]),
        ("Expected complete curves", generation["expected_curves"]),
        ("Final-test state", final["state"]),
        ("Final-test open count", final["open_count"]),
        ("Final-test integrity", final["integrity"]),
        ("Final-test rerun permitted", final["rerun_permitted"]),
    ]
    _print_table(("Item", "Value"), rows)
    split_rows = []
    for split_name, record in status["split_profile"]["splits"].items():
        split_rows.append(
            (
                split_name,
                record["family_groups"],
                record["hadronic_family_groups"],
                record["quark_family_groups"],
                record["expected_curves"],
            )
        )
    print("\nFamily-level partitions")
    _print_table(
        ("Partition", "Family groups", "Hadronic", "Quark", "Expected curves"),
        split_rows,
    )
    evidence = status["development_evidence"]
    print("\nRecorded development safeguards")
    _print_table(
        ("Safeguard", "Recorded value"),
        [
            ("Shortcut audit passed", evidence["shortcut_audit_passed"]),
            ("Selected reporting model", evidence["selected_reporting_model"]),
            (
                "Family-label permutation empirical p-value",
                evidence["family_permutation_empirical_p_value"],
            ),
            (
                "Locked-test rows used by development",
                evidence["test_rows_used_by_development"],
            ),
        ],
    )
    if final["integrity_errors"]:
        print("\nFinal-test integrity findings")
        for finding in final["integrity_errors"]:
            print(f"- {finding}")
    print("\nReporting models: dummy baseline and logistic regression.")
    print("XGBoost and the multilayer perceptron remain exploratory.")
    result = final.get("result")
    if result:
        print("\nRecorded one-time final-test result")
        _print_table(
            ("Metric", "Recorded value"),
            [
                ("Complete curves", result.get("samples")),
                ("Independent families", result.get("independent_family_units")),
                ("Balanced accuracy", result.get("balanced_accuracy")),
                (
                    "Equal-family balanced accuracy",
                    result.get("family_balanced_accuracy"),
                ),
                ("ROC AUC", result.get("roc_auc")),
                (
                    "Strict 2.08-solar-mass screen applicable",
                    result.get("strict_2p08_test_applicable"),
                ),
            ],
        )
        per_family = result.get("per_family", [])
        if per_family:
            print("\nExact held-out-family results")
            _print_table(
                (
                    "EoS",
                    "Matter type",
                    "Curves",
                    "Accuracy",
                    "Mean quark-class model score",
                    "Score range across A",
                ),
                [
                    (
                        record.get("eos_id"),
                        record.get("matter_type"),
                        record.get("curves"),
                        record.get("accuracy"),
                        record.get("mean_quark_class_model_score"),
                        record.get("model_score_range_across_amplitude"),
                    )
                    for record in per_family
                ],
            )
        print(
            "\nThese classifier scores are model outputs for this fixed repository "
            "experiment; they are not physical probabilities of quark matter."
        )
    print("\nScientific interpretation:")
    print(status["scientific_scope"])


def _print_table(headers: tuple[str, ...], rows) -> None:
    normalized = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in normalized:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    format_string = "  ".join(f"{{:<{width}}}" for width in widths)
    print(format_string.format(*headers))
    print(format_string.format(*(("-" * width) for width in widths)))
    for row in normalized:
        print(format_string.format(*row))


if __name__ == "__main__":
    raise SystemExit(main())
