"""Stress-test the one-week family-pilot candidates on a shared A grid.

The candidate list is taken from the checked-in A=0 feasibility summary.  This
second gate evaluates whether every selected fixed EoS remains physical and
covers the same 1--2 Msun observable support after Gaussian deformation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from framework.eos_catalog import CFL_CATALOG, HADRONIC_CATALOG
from framework.eos_sweep import (
    GaussianDeformation,
    QuarkParameters,
    build_hadronic_eos,
    build_quark_eos,
)
from src.config import CONFIG
from src.physics.feature_extraction import extract_features
from src.physics.solve_sequence import solve_sequence
from src.physics.verification import verify_eos_physical_validity


PILOT_MMAX_FLOOR = 2.0
DEFAULT_AMPLITUDES = (-0.05, -0.02, 0.0, 0.02, 0.05, 0.09)


def selected_catalog_entries(summary_path: Path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = set(summary["recommended_2p0_eos_ids"])
    entries = [
        *(entry for entry in HADRONIC_CATALOG if entry.eos_id in selected),
        *(entry for entry in CFL_CATALOG if entry.eos_id in selected),
    ]
    discovered = {entry.eos_id for entry in entries}
    if discovered != selected:
        raise ValueError(
            f"Feasibility summary/catalog mismatch: missing={sorted(selected - discovered)}, "
            f"unexpected={sorted(discovered - selected)}"
        )
    if any(entry.eos_id == "PS" for entry in entries):
        raise ValueError("The provenance-screened family pilot must not include PS.")
    return entries


def _evaluate(entry, amplitude: float) -> dict:
    is_quark = hasattr(entry, "bag_b_mev_fm3")
    matter_class = "quark" if is_quark else "hadronic"
    record = {
        "matter_class": matter_class,
        "eos_id": entry.eos_id,
        "family_group_id": entry.family_group_id,
        "amplitude": float(amplitude),
        "build_ok": False,
        "physical_trace_ok": False,
        "minimum_mass_msun": np.nan,
        "maximum_mass_msun": np.nan,
        "radius_1p4_km": np.nan,
        "covers_ml_mass_grid_1_to_2": False,
        "passes_mmax_floor": False,
        "passes_mmax_ceiling": False,
        "passes_r14_screen": False,
        "acceptable": False,
        "reason": "",
    }
    try:
        deformation = GaussianDeformation(
            amplitude,
            CONFIG["CONTROLLED_PERTURB_EPS0"],
            CONFIG["CONTROLLED_PERTURB_SIGMA"],
        )
        if is_quark:
            parameters = QuarkParameters(
                entry.bag_b_mev_fm3,
                entry.gap_delta_mev,
                entry.strange_mass_mev,
            )
            eos = build_quark_eos(
                parameters,
                deformation,
                maximum_surface_energy_per_baryon=CONFIG["M_N"],
            )
        else:
            eos = build_hadronic_eos(entry.eos_id, deformation)
        record["build_ok"] = True
        curve, _, maximum_mass = solve_sequence(
            eos.eos_callable,
            is_quark=is_quark,
            p_max_causal=eos.p_max_causal,
            rtol=CONFIG["TOV_RTOL"],
            atol=CONFIG["TOV_ATOL"],
        )
        if not curve:
            raise RuntimeError("TOV integration returned no stable sequence.")
        curve_array = np.asarray(curve, dtype=float)
        record["minimum_mass_msun"] = float(np.min(curve_array[:, 0]))
        record["maximum_mass_msun"] = float(maximum_mass)
        record["physical_trace_ok"] = bool(verify_eos_physical_validity(curve_array))
        features = extract_features(curve_array, maximum_mass)
        if features is not None:
            record["radius_1p4_km"] = float(features["r_14"])
        record["covers_ml_mass_grid_1_to_2"] = bool(
            record["minimum_mass_msun"] <= CONFIG["ML_MASS_GRID_MIN"]
            and maximum_mass >= CONFIG["ML_MASS_GRID_MAX"]
        )
        record["passes_mmax_floor"] = bool(maximum_mass >= PILOT_MMAX_FLOOR)
        record["passes_mmax_ceiling"] = bool(maximum_mass <= 3.0)
        r14 = record["radius_1p4_km"]
        record["passes_r14_screen"] = bool(
            np.isfinite(r14)
            and CONFIG["CONTROLLED_R14_MIN"] <= r14 <= CONFIG["CONTROLLED_R14_MAX"]
        )
        record["acceptable"] = bool(
            record["physical_trace_ok"]
            and record["minimum_mass_msun"] <= CONFIG["BH_LIMIT"]
            and record["covers_ml_mass_grid_1_to_2"]
            and record["passes_mmax_floor"]
            and record["passes_mmax_ceiling"]
            and record["passes_r14_screen"]
        )
        reasons = []
        if not record["physical_trace_ok"]:
            reasons.append("physical trace")
        if record["minimum_mass_msun"] > CONFIG["BH_LIMIT"]:
            reasons.append("low-mass trace")
        if not record["covers_ml_mass_grid_1_to_2"]:
            reasons.append("1-2 Msun support")
        if not record["passes_mmax_floor"]:
            reasons.append("Mmax<2.0")
        if not record["passes_mmax_ceiling"]:
            reasons.append("Mmax>3.0")
        if not record["passes_r14_screen"]:
            reasons.append("R1.4 screen")
        record["reason"] = "; ".join(reasons)
    except Exception as exc:
        record["reason"] = f"{type(exc).__name__}: {exc}"
    return record


def _plot_matrix(results: pd.DataFrame, output_path: Path) -> None:
    order = (
        results[["matter_class", "eos_id"]]
        .drop_duplicates()
        .sort_values(["matter_class", "eos_id"])
    )
    eos_ids = order["eos_id"].tolist()
    amplitudes = sorted(results["amplitude"].unique())
    matrix = (
        results.pivot(index="eos_id", columns="amplitude", values="acceptable")
        .reindex(index=eos_ids, columns=amplitudes)
        .to_numpy(dtype=float)
    )
    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    del image
    axis.set_xticks(np.arange(len(amplitudes)))
    axis.set_xticklabels([f"{value:.2f}" for value in amplitudes])
    axis.set_yticks(np.arange(len(eos_ids)))
    axis.set_yticklabels(eos_ids)
    axis.set_xlabel("Gaussian amplitude A")
    axis.set_ylabel("Fixed EoS baseline")
    axis.set_title("Family-pilot physical viability (green = all screens pass)")
    for row_index, eos_id in enumerate(eos_ids):
        for column_index, amplitude in enumerate(amplitudes):
            row = results[(results["eos_id"] == eos_id) & np.isclose(results["amplitude"], amplitude)].iloc[0]
            if not row["acceptable"]:
                axis.text(column_index, row_index, "x", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _persist_results(results: pd.DataFrame, output_dir: Path) -> dict:
    results = results.sort_values(
        ["matter_class", "eos_id", "amplitude"], ignore_index=True
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "eos_family_amplitude_scan.csv", index=False)
    per_amplitude = (
        results.groupby("amplitude")
        .agg(accepted=("acceptable", "sum"), total=("acceptable", "size"))
        .reset_index()
    )
    per_amplitude["all_families_pass"] = per_amplitude["accepted"].eq(
        per_amplitude["total"]
    )
    summary = {
        "candidate_profile": "recommended_2p0",
        "mmax_floor_msun": PILOT_MMAX_FLOOR,
        "eos_count": int(results["eos_id"].nunique()),
        "amplitudes": sorted(float(value) for value in results["amplitude"].unique()),
        "per_amplitude": per_amplitude.to_dict(orient="records"),
        "amplitudes_passing_all_families": per_amplitude.loc[
            per_amplitude["all_families_pass"], "amplitude"
        ].tolist(),
        "rejected_combinations": int((~results["acceptable"]).sum()),
    }
    (output_dir / "eos_family_amplitude_scan_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot_matrix(results, output_dir / "EOS_FAMILY_AMPLITUDE_VIABILITY.png")
    return summary


def consolidate_scan_results(csv_paths: list[Path], output_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Combine already-computed scan shards without repeating TOV solves."""

    if not csv_paths:
        raise ValueError("At least one scan CSV is required for consolidation.")
    results = pd.concat(
        [pd.read_csv(path) for path in csv_paths], ignore_index=True
    ).drop_duplicates(subset=["matter_class", "eos_id", "amplitude"], keep="last")
    expected_eos = results["eos_id"].nunique()
    counts = results.groupby("amplitude")["eos_id"].nunique()
    incomplete = counts[counts != expected_eos]
    if not incomplete.empty:
        raise ValueError(
            "Consolidated scan has incomplete amplitude slices: "
            f"{incomplete.to_dict()} (expected {expected_eos} EoSs each)."
        )
    return results, _persist_results(results, output_dir)


def run_scan(
    *,
    summary_path: Path,
    output_dir: Path,
    amplitudes: tuple[float, ...],
    jobs: int,
) -> tuple[pd.DataFrame, dict]:
    entries = selected_catalog_entries(summary_path)
    audited = Parallel(n_jobs=jobs, verbose=10)(
        delayed(_evaluate)(entry, amplitude)
        for entry in entries
        for amplitude in amplitudes
    )
    results = pd.DataFrame.from_records(audited)
    return results, _persist_results(results, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/eos_feasibility_summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=list(DEFAULT_AMPLITUDES),
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--merge-results",
        type=Path,
        nargs="+",
        help="Consolidate existing scan CSV shards instead of solving TOV again.",
    )
    args = parser.parse_args()
    if args.merge_results:
        _, summary = consolidate_scan_results(args.merge_results, args.output_dir)
    else:
        _, summary = run_scan(
            summary_path=args.summary,
            output_dir=args.output_dir,
            amplitudes=tuple(args.amplitudes),
            jobs=args.jobs,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
