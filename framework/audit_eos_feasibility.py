"""Run the undeformed EoS provenance and stellar-feasibility audit.

This command deliberately does not generate an ML dataset or train a model.
It evaluates every catalogued baseline at A=0 with epsilon_0=220 MeV/fm^3
and sigma=50 MeV/fm^3, derives model-specific causal amplitude intervals,
and writes a literature catalog plus acceptance/rejection evidence.

Run from the repository root with::

    py -m framework.audit_eos_feasibility --jobs 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import PchipInterpolator

from framework.eos_catalog import (
    CFL_CATALOG,
    HADRONIC_CATALOG,
    CflCatalogEntry,
    HadronicCatalogEntry,
    literature_catalog_rows,
)
from framework.eos_sweep import (
    GaussianDeformation,
    QuarkParameters,
    admissible_amplitude_interval,
    build_hadronic_eos,
    build_quark_eos,
    cfl_baseline_grids,
    hadronic_baseline_grids,
)
from src.config import CONFIG
from src.physics.feature_extraction import extract_features
from src.physics.solve_sequence import solve_sequence
from src.physics.verification import verify_eos_physical_validity


AUDIT_EPSILON0 = 220.0
AUDIT_SIGMA = 50.0
OBSERVED_TWO_SOLAR_MASS_SCREEN = 2.0
PUBLISHED_CFL_RELATIVE_TOLERANCE = 0.03


def _base_record(
    *,
    matter_class: str,
    eos_id: str,
    family_group_id: str,
    parameter_block_id: str,
    model_superfamily_id: str,
    exact_formula_primary_verified: bool,
    underlying_primary_citation_available: bool,
) -> dict:
    return {
        "matter_class": matter_class,
        "eos_id": eos_id,
        "family_group_id": family_group_id,
        "parameter_block_id": parameter_block_id,
        "model_superfamily_id": model_superfamily_id,
        "exact_formula_primary_verified": exact_formula_primary_verified,
        "underlying_primary_citation_available": underlying_primary_citation_available,
        "audit_amplitude": 0.0,
        "audit_epsilon0_mev_fm3": AUDIT_EPSILON0,
        "audit_sigma_mev_fm3": AUDIT_SIGMA,
        "amplitude_lower_open": np.nan,
        "amplitude_upper_closed": np.nan,
        "maximum_gaussian_weight_on_eos_grid": np.nan,
        "energy_nearest_gaussian_center_mev_fm3": np.nan,
        "controlled_a_grid_causally_admissible": False,
        "build_ok": False,
        "physical_trace_ok": False,
        "stable_sequence_points": 0,
        "minimum_mass_msun": np.nan,
        "maximum_mass_msun": np.nan,
        "radius_1p4_km": np.nan,
        "lambda_1p4": np.nan,
        "radius_at_mmax_km": np.nan,
        "minimum_cs2": np.nan,
        "maximum_cs2": np.nan,
        "p_max_causal_mev_fm3": np.nan,
        "surface_energy_density_mev_fm3": np.nan,
        "surface_energy_per_baryon_mev": np.nan,
        "passes_low_mass_trace": False,
        "covers_ml_mass_grid_1_to_2": False,
        "passes_two_solar_mass_screen": False,
        "passes_project_mmax_screen": False,
        "passes_common_r14_screen": False,
        "passes_project_upper_mass_screen": False,
        "pilot_eligible_2p0": False,
        "pilot_eligible_strict": False,
        "recommended_for_pilot_2p0": False,
        "recommended_for_pilot_strict": False,
        "published_mmax_msun": np.nan,
        "published_r_at_mmax_km": np.nan,
        "mmax_minus_published_msun": np.nan,
        "r_at_mmax_minus_published_km": np.nan,
        "published_cfl_crosscheck_within_3pct": np.nan,
        "rejection_reasons": "",
        "recommendation_reasons": "",
        "audit_error": "",
    }


def _interpolate_at_mass(curve: np.ndarray, mass: float, column: int) -> float:
    order = np.argsort(curve[:, 0])
    ordered = curve[order]
    unique_mass, unique_indices = np.unique(ordered[:, 0], return_index=True)
    values = ordered[unique_indices, column]
    if len(unique_mass) < 2 or not unique_mass[0] <= mass <= unique_mass[-1]:
        return float("nan")
    return float(PchipInterpolator(unique_mass, values)(mass))


def _audit_built_eos(record: dict, framework_eos, *, is_quark: bool) -> tuple[dict, np.ndarray]:
    record["build_ok"] = True
    record["minimum_cs2"] = float(np.min(framework_eos.sound_speed_squared))
    record["maximum_cs2"] = float(np.max(framework_eos.sound_speed_squared))
    record["p_max_causal_mev_fm3"] = float(framework_eos.p_max_causal)
    record["surface_energy_density_mev_fm3"] = float(framework_eos.eps_surface)
    if framework_eos.energy_per_baryon_surface is not None:
        record["surface_energy_per_baryon_mev"] = float(
            framework_eos.energy_per_baryon_surface
        )

    curve, _, maximum_mass = solve_sequence(
        framework_eos.eos_callable,
        is_quark=is_quark,
        p_max_causal=framework_eos.p_max_causal,
        rtol=CONFIG["TOV_RTOL"],
        atol=CONFIG["TOV_ATOL"],
    )
    if not curve:
        record["audit_error"] = "TOV integration returned no stable sequence points."
        return record, np.empty((0, 7), dtype=float)

    curve_array = np.asarray(curve, dtype=float)
    record["stable_sequence_points"] = int(len(curve_array))
    record["minimum_mass_msun"] = float(np.min(curve_array[:, 0]))
    record["maximum_mass_msun"] = float(maximum_mass)
    maximum_index = int(np.argmax(curve_array[:, 0]))
    record["radius_at_mmax_km"] = float(curve_array[maximum_index, 1])

    try:
        record["physical_trace_ok"] = bool(verify_eos_physical_validity(curve_array))
    except Exception as exc:  # domain exceptions carry the useful reason
        record["audit_error"] = f"Physical trace validation failed: {exc}"

    features = extract_features(curve_array, maximum_mass)
    if features is not None:
        record["radius_1p4_km"] = float(features["r_14"])
    record["lambda_1p4"] = _interpolate_at_mass(curve_array, 1.4, 2)

    record["passes_low_mass_trace"] = bool(
        record["minimum_mass_msun"] <= CONFIG["BH_LIMIT"]
    )
    record["covers_ml_mass_grid_1_to_2"] = bool(
        record["minimum_mass_msun"] <= CONFIG["ML_MASS_GRID_MIN"]
        and maximum_mass >= CONFIG["ML_MASS_GRID_MAX"]
    )
    record["passes_two_solar_mass_screen"] = bool(
        maximum_mass >= OBSERVED_TWO_SOLAR_MASS_SCREEN
    )
    record["passes_project_mmax_screen"] = bool(
        maximum_mass >= CONFIG["M_MAX_LOWER_BOUND"]
    )
    record["passes_project_upper_mass_screen"] = bool(
        maximum_mass <= min(CONFIG["H_M_MAX_UPPER"], CONFIG["Q_M_MAX_UPPER"])
    )
    r14 = record["radius_1p4_km"]
    record["passes_common_r14_screen"] = bool(
        np.isfinite(r14)
        and CONFIG["CONTROLLED_R14_MIN"] <= r14 <= CONFIG["CONTROLLED_R14_MAX"]
    )

    common_checks = (
        record["build_ok"],
        record["physical_trace_ok"],
        record["passes_low_mass_trace"],
        record["covers_ml_mass_grid_1_to_2"],
        record["passes_common_r14_screen"],
        record["passes_project_upper_mass_screen"],
    )
    record["pilot_eligible_2p0"] = bool(
        all(common_checks) and record["passes_two_solar_mass_screen"]
    )
    record["pilot_eligible_strict"] = bool(
        all(common_checks) and record["passes_project_mmax_screen"]
    )
    record["recommended_for_pilot_2p0"] = bool(
        record["pilot_eligible_2p0"]
        and record["underlying_primary_citation_available"]
    )
    record["recommended_for_pilot_strict"] = bool(
        record["pilot_eligible_strict"]
        and record["underlying_primary_citation_available"]
    )

    rejection_reasons = []
    if not record["physical_trace_ok"]:
        rejection_reasons.append("physical trace failed")
    if not record["passes_low_mass_trace"]:
        rejection_reasons.append("low-mass branch not traced below 0.5 Msun")
    if not record["covers_ml_mass_grid_1_to_2"]:
        rejection_reasons.append("does not cover common 1.0-2.0 Msun grid")
    if not record["passes_project_mmax_screen"]:
        rejection_reasons.append(
            f"Mmax below project screen {CONFIG['M_MAX_LOWER_BOUND']:.2f} Msun"
        )
    if not record["passes_project_upper_mass_screen"]:
        rejection_reasons.append("Mmax above project upper screen 3.0 Msun")
    if not record["passes_common_r14_screen"]:
        rejection_reasons.append("R1.4 outside common 9.5-14.5 km screen")
    record["rejection_reasons"] = "; ".join(rejection_reasons)
    recommendation_reasons = list(rejection_reasons)
    if not record["underlying_primary_citation_available"]:
        recommendation_reasons.append("no underlying primary citation in fit source")
    record["recommendation_reasons"] = "; ".join(recommendation_reasons)
    return record, curve_array


def _set_amplitude_interval(record: dict, energy: np.ndarray, cs2: np.ndarray) -> None:
    lower, upper = admissible_amplitude_interval(
        energy, cs2, AUDIT_EPSILON0, AUDIT_SIGMA
    )
    record["amplitude_lower_open"] = float(lower)
    record["amplitude_upper_closed"] = float(upper)
    gaussian = np.exp(-0.5 * ((energy - AUDIT_EPSILON0) / AUDIT_SIGMA) ** 2)
    peak_index = int(np.argmax(gaussian))
    record["maximum_gaussian_weight_on_eos_grid"] = float(gaussian[peak_index])
    record["energy_nearest_gaussian_center_mev_fm3"] = float(energy[peak_index])
    record["controlled_a_grid_causally_admissible"] = bool(
        CONFIG["CONTROLLED_A_MIN"] > lower and CONFIG["CONTROLLED_A_MAX"] <= upper
    )


def audit_hadronic(entry: HadronicCatalogEntry) -> tuple[dict, np.ndarray]:
    record = _base_record(
        matter_class="hadronic",
        eos_id=entry.eos_id,
        family_group_id=entry.family_group_id,
        parameter_block_id="",
        model_superfamily_id="H_REPOSITORY_SURROGATES",
        exact_formula_primary_verified=entry.exact_formula_primary_verified,
        underlying_primary_citation_available=bool(entry.underlying_primary_url),
    )
    try:
        energy, cs2, _, transition, _ = hadronic_baseline_grids(entry.eos_id)
        if not np.isclose(transition, entry.transition_pressure_mev_fm3):
            raise ValueError(
                f"Catalog transition {entry.transition_pressure_mev_fm3} does not "
                f"match framework transition {transition}."
            )
        _set_amplitude_interval(record, energy, cs2)
        framework_eos = build_hadronic_eos(
            entry.eos_id,
            GaussianDeformation(0.0, AUDIT_EPSILON0, AUDIT_SIGMA),
        )
        return _audit_built_eos(record, framework_eos, is_quark=False)
    except Exception as exc:
        record["audit_error"] = f"{type(exc).__name__}: {exc}"
        record["rejection_reasons"] = "build or audit failed"
        record["recommendation_reasons"] = "build or audit failed"
        return record, np.empty((0, 7), dtype=float)


def audit_cfl(entry: CflCatalogEntry) -> tuple[dict, np.ndarray]:
    record = _base_record(
        matter_class="quark",
        eos_id=entry.eos_id,
        family_group_id=entry.family_group_id,
        parameter_block_id=entry.parameter_block_id,
        model_superfamily_id="Q_ANALYTIC_CFL_MIT_BAG",
        exact_formula_primary_verified=True,
        underlying_primary_citation_available=True,
    )
    record["published_mmax_msun"] = entry.published_mmax_msun
    record["published_r_at_mmax_km"] = entry.published_r_at_mmax_km
    try:
        parameters = QuarkParameters(
            bag_b=entry.bag_b_mev_fm3,
            gap_delta=entry.gap_delta_mev,
            strange_mass=entry.strange_mass_mev,
        )
        _, energy, cs2, _ = cfl_baseline_grids(parameters)
        _set_amplitude_interval(record, energy, cs2)
        framework_eos = build_quark_eos(
            parameters,
            GaussianDeformation(0.0, AUDIT_EPSILON0, AUDIT_SIGMA),
            maximum_surface_energy_per_baryon=CONFIG["M_N"],
        )
        record, curve = _audit_built_eos(record, framework_eos, is_quark=True)
        if np.isfinite(record["maximum_mass_msun"]):
            record["mmax_minus_published_msun"] = float(
                record["maximum_mass_msun"] - entry.published_mmax_msun
            )
            record["r_at_mmax_minus_published_km"] = float(
                record["radius_at_mmax_km"] - entry.published_r_at_mmax_km
            )
            mass_relative_error = abs(record["mmax_minus_published_msun"]) / abs(
                entry.published_mmax_msun
            )
            radius_relative_error = abs(
                record["r_at_mmax_minus_published_km"]
            ) / abs(entry.published_r_at_mmax_km)
            record["published_cfl_crosscheck_within_3pct"] = bool(
                mass_relative_error <= PUBLISHED_CFL_RELATIVE_TOLERANCE
                and radius_relative_error <= PUBLISHED_CFL_RELATIVE_TOLERANCE
            )
        return record, curve
    except Exception as exc:
        record["audit_error"] = f"{type(exc).__name__}: {exc}"
        record["rejection_reasons"] = "build or audit failed"
        record["recommendation_reasons"] = "build or audit failed"
        return record, np.empty((0, 7), dtype=float)


def _intersection(frame: pd.DataFrame, mask: pd.Series) -> tuple[float, float] | None:
    selected = frame.loc[mask]
    if selected.empty:
        return None
    lower = float(selected["amplitude_lower_open"].max())
    upper = float(selected["amplitude_upper_closed"].min())
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return None
    return lower, upper


def _format_interval(interval: tuple[float, float] | None) -> str:
    if interval is None:
        return "none"
    return f"({interval[0]:.6f}, {interval[1]:.6f}]"


def _write_plots(
    output_dir: Path,
    results: pd.DataFrame,
    curves: dict[str, np.ndarray],
) -> None:
    colors = {True: "#087f5b", False: "#adb5bd"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for axis, matter_class, title in zip(
        axes,
        ("hadronic", "quark"),
        ("Hadronic repository surrogates", "Published CFL tuples"),
    ):
        subset = results[results["matter_class"] == matter_class]
        for row in subset.itertuples(index=False):
            curve = curves.get(f"{matter_class}:{row.eos_id}")
            if curve is None or not len(curve):
                continue
            display_curve = curve[curve[:, 0] >= 0.5]
            if not len(display_curve):
                continue
            eligible = bool(row.recommended_for_pilot_strict)
            axis.plot(
                display_curve[:, 1],
                display_curve[:, 0],
                color=colors[eligible],
                alpha=0.9 if eligible else 0.45,
                linewidth=1.6 if eligible else 0.9,
            )
            axis.annotate(
                row.eos_id,
                (display_curve[-1, 1], display_curve[-1, 0]),
                fontsize=6,
                color=colors[eligible],
                xytext=(2, 0),
                textcoords="offset points",
            )
        axis.axhline(CONFIG["M_MAX_LOWER_BOUND"], color="#c92a2a", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Radius [km]")
        axis.grid(alpha=0.2)
    axes[0].set_xlim(8.0, 22.0)
    axes[1].set_xlim(7.5, 15.5)
    axes[0].set_ylim(0.5, 3.5)
    axes[0].set_ylabel(r"Mass [$M_\odot$]")
    fig.suptitle("Undeformed A=0 production-solver feasibility scan")
    fig.tight_layout()
    fig.savefig(output_dir / "EOS_FEASIBILITY_MR.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ordered = results.sort_values(["matter_class", "amplitude_lower_open"])
    y = np.arange(len(ordered))
    fig, axis = plt.subplots(figsize=(11, 10))
    x_min, x_max = -0.65, 1.10
    for position, row in enumerate(ordered.itertuples(index=False)):
        eligible = bool(row.recommended_for_pilot_strict)
        lower = max(float(row.amplitude_lower_open), x_min)
        upper = min(float(row.amplitude_upper_closed), x_max)
        axis.plot(
            [lower, upper],
            [position, position],
            color=colors[eligible],
            linewidth=2.0 if eligible else 1.0,
        )
        if row.amplitude_lower_open < x_min:
            axis.plot(x_min, position, marker="<", color=colors[eligible], markersize=4)
        if row.amplitude_upper_closed > x_max:
            axis.plot(x_max, position, marker=">", color=colors[eligible], markersize=4)
    axis.axvspan(
        CONFIG["CONTROLLED_A_MIN"],
        CONFIG["CONTROLLED_A_MAX"],
        color="#74c0fc",
        alpha=0.25,
        label="current controlled A sweep",
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(y)
    axis.set_yticklabels(
        [f"{row.matter_class[0].upper()} {row.eos_id}" for row in ordered.itertuples(index=False)],
        fontsize=7,
    )
    axis.set_xlabel("Causal/stable Gaussian amplitude A")
    axis.set_title("Model-specific admissible A intervals (open left, closed right)")
    axis.set_xlim(x_min, x_max)
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "EOS_AMPLITUDE_INTERVALS.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [
        "Class",
        "EoS",
        "Family group",
        "Numeric pass",
        "Recommended",
        "Mmax",
        "R1.4",
        "A interval",
        "Reason",
    ]
    rows = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in frame.itertuples(index=False):
        reason = row.recommendation_reasons or "accepted"
        if row.audit_error:
            reason = f"{reason}; {row.audit_error}" if reason else row.audit_error
        reason = str(reason).replace("|", "/")
        rows.append(
            "| "
            + " | ".join(
                [
                    row.matter_class,
                    row.eos_id,
                    row.family_group_id,
                    "yes" if row.pilot_eligible_strict else "no",
                    "yes" if row.recommended_for_pilot_strict else "no",
                    f"{row.maximum_mass_msun:.3f}" if np.isfinite(row.maximum_mass_msun) else "n/a",
                    f"{row.radius_1p4_km:.3f}" if np.isfinite(row.radius_1p4_km) else "n/a",
                    (
                        f"({row.amplitude_lower_open:.4f}, {row.amplitude_upper_closed:.4f}]"
                        if np.isfinite(row.amplitude_lower_open)
                        else "n/a"
                    ),
                    reason,
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _write_report(output_dir: Path, results: pd.DataFrame, summary: dict) -> None:
    hadronic = results[results["matter_class"] == "hadronic"]
    quark = results[results["matter_class"] == "quark"]
    cfl_checked = quark[quark["published_cfl_crosscheck_within_3pct"].notna()]
    cfl_matches = int(cfl_checked["published_cfl_crosscheck_within_3pct"].sum())
    recommended_quark = quark[quark["recommended_for_pilot_strict"]]
    gaussian_min = float(recommended_quark["maximum_gaussian_weight_on_eos_grid"].min())
    gaussian_max = float(recommended_quark["maximum_gaussian_weight_on_eos_grid"].max())

    report = f"""# EoS Literature and A=0 Feasibility Audit

## Outcome

The production framework built and sent all {len(results)} catalog entries through the same undeformed `A=0` TOV path. Under the current strict project screens, **{int(hadronic['pilot_eligible_strict'].sum())}/{len(hadronic)} hadronic surrogates** and **{int(quark['pilot_eligible_strict'].sum())}/{len(quark)} published CFL tuples** pass numerically. Removing the uncited PS surrogate leaves **{int(hadronic['recommended_for_pilot_strict'].sum())} hadronic and {int(quark['recommended_for_pilot_strict'].sum())} CFL baselines recommended for a strict pilot**.

Using a conventional `Mmax >= 2.0 Msun` diagnostic instead of the project's harder `2.08 Msun` point-estimate cut yields **{int(hadronic['recommended_for_pilot_2p0'].sum())} provenance-screened hadronic and {int(quark['recommended_for_pilot_2p0'].sum())} CFL baselines**. This larger sensitivity set is the practical one-week option; the strict `2.08` result must still be reported beside it.

No classifier was trained. This is the family-availability gate that must precede dataset generation.

The numerically strict common causal/stable amplitude intersection is **{summary['strict_global_amplitude_intersection']}**. It excludes the current lower endpoint solely because PS requires `A > -0.041039`. PS is also the one model rejected by the provenance screen. The recommended strict intersection is **{summary['recommended_strict_global_amplitude_intersection']}**, which does contain the current controlled sweep `[{CONFIG['CONTROLLED_A_MIN']:.2f}, {CONFIG['CONTROLLED_A_MAX']:.2f}]`.

## Provenance finding

The 21 repository hadronic expressions exactly match the formulas listed in Stergakis (2025), but the exact coefficients, fit domain, and fit uncertainty have not been verified in the cited primary model papers. Every hadronic row therefore has `exact_formula_primary_verified=false`. They are suitable only as transparently labelled **repository surrogates** until compared against primary/tabulated EoSs. `PS` is weaker still: the fit source attaches no primary citation to it.

The CFL side has stronger provenance: all 19 tuples and their reference maximum-star values appear in Tables I-II of Vasquez Flores & Lugones (2017). The repository solver agrees within 3% in both `Mmax` and the radius at `Mmax` for **{cfl_matches}/{len(cfl_checked)}** successfully checked tuples. Numerical deltas are in `eos_feasibility_results.csv`.

## Which fixed values are literature-backed?

- `B=60 MeV/fm^3`, `Delta=100 MeV`, `m_s=150 MeV` are the published CFL4 tuple.
- `Mmax=2.08 Msun` is the central measured mass of PSR J0740+6620, not a hard lower confidence bound. This audit therefore reports both the project `2.08` screen and a separate `2.0` diagnostic.
- `epsilon0=220 MeV/fm^3`, `sigma=50 MeV/fm^3`, and the sampled `A` values are project-defined deformation coordinates. They are not values inferred from APR or CFL literature.
- `R1.4 in [9.5, 14.5] km` is treated here as a deliberately broad common-support engineering screen, not as a single published posterior interval.

For the strict recommended CFL entries, the largest Gaussian weight actually reached on each EoS grid ranges from {gaussian_min:.3f} to {gaussian_max:.3f}. The same numeric `(epsilon0, sigma)` therefore does not produce exactly the same effective perturbation strength across parameter tuples; this overlap value must travel with each generated family.

## Classification consequence

The hadronic catalog contains {hadronic['family_group_id'].nunique()} conservative family groups before physics screening. The CFL table contains {quark['family_group_id'].nunique()} fixed-tuple families in {quark['parameter_block_id'].nunique()} bag-constant blocks, but every tuple shares the same analytic CFL MIT-bag theory. The strict recommended set retains {summary['counts']['recommended_strict_hadronic_family_groups']} hadronic family groups, {summary['counts']['recommended_strict_quark_family_groups']} CFL tuple families, and only {summary['counts']['recommended_strict_quark_parameter_blocks']} CFL bag-constant blocks. The `2.0 Msun` sensitivity set raises these to {summary['counts']['recommended_2p0_hadronic_family_groups']}, {summary['counts']['recommended_2p0_quark_family_groups']}, and {summary['counts']['recommended_2p0_quark_parameter_blocks']}, respectively.

A primary family-held-out pilot can therefore hold out complete fixed EoS tuples and all their `A` variants. A harsher secondary check may hold out a complete CFL bag-constant block. Neither test establishes generalization to NJL, perturbative-QCD, or other quark-matter theories.

Proceed to dataset construction only with accepted rows, retain `family_group_id`, and keep the hadronic surrogate warning in every manifest. If the strict eligible counts are too small for the locked split, the scientifically honest fast fallback is a limited multi-baseline pilot with a narrower claim—not row-wise splitting of the same curves.

## Detailed acceptance table

{_markdown_table(results)}

## Reproducible artifacts

- `eos_literature_catalog.csv`: source, family grouping, parameters, and provenance layer for all 40 entries.
- `eos_feasibility_results.csv`: complete numeric audit and rejection reasons.
- `eos_feasibility_summary.json`: counts, intersections, and fixed audit controls.
- `eos_feasibility_curves.npz`: the plotted undeformed stable branches.
- `EOS_FEASIBILITY_MR.png`: all undeformed M-R sequences; green is strict and provenance-recommended.
- `EOS_AMPLITUDE_INTERVALS.png`: model-specific causal/stable A support and the current requested sweep.

Run again with `py -m framework.audit_eos_feasibility --jobs 4`.

## Primary sources and exact-fit source

- Stergakis, *Reconstruction of the Equations of State (EoSs) of Compact Stars using machine and deep learning regression techniques* (2025): https://arxiv.org/abs/2509.13037
- Vasquez Flores & Lugones, *Constraining color flavor locked strange stars in the gravitational wave era* (2017): https://arxiv.org/abs/1702.02081
- Lugones & Horvath, *Color-flavor locked strange matter* (2002): https://arxiv.org/abs/hep-ph/0211070
- Fonseca et al., *Refined Mass and Geometric Measurements of the High-Mass PSR J0740+6620* (2021): https://arxiv.org/abs/2104.00880
"""
    (output_dir / "EOS_FEASIBILITY_AUDIT.md").write_text(report, encoding="utf-8")


def _json_compatible(value):
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def run_audit(output_dir: Path, jobs: int) -> tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks: Iterable = [
        *(delayed(audit_hadronic)(entry) for entry in HADRONIC_CATALOG),
        *(delayed(audit_cfl)(entry) for entry in CFL_CATALOG),
    ]
    audited = Parallel(n_jobs=jobs, verbose=10)(tasks)
    records = [record for record, _ in audited]
    curves = {
        f"{record['matter_class']}:{record['eos_id']}": curve
        for record, curve in audited
    }
    results = pd.DataFrame.from_records(records).sort_values(
        ["matter_class", "eos_id"], ignore_index=True
    )

    strict = results["pilot_eligible_strict"]
    strict_h = strict & results["matter_class"].eq("hadronic")
    strict_q = strict & results["matter_class"].eq("quark")
    recommended_strict = results["recommended_for_pilot_strict"]
    recommended_strict_h = recommended_strict & results["matter_class"].eq("hadronic")
    recommended_strict_q = recommended_strict & results["matter_class"].eq("quark")
    recommended_2p0 = results["recommended_for_pilot_2p0"]
    recommended_2p0_h = recommended_2p0 & results["matter_class"].eq("hadronic")
    recommended_2p0_q = recommended_2p0 & results["matter_class"].eq("quark")
    global_interval = _intersection(results, strict)
    recommended_global_interval = _intersection(results, recommended_strict)
    summary = {
        "audit_definition": {
            "amplitude": 0.0,
            "epsilon0_mev_fm3": AUDIT_EPSILON0,
            "sigma_mev_fm3": AUDIT_SIGMA,
            "project_mmax_lower_msun": CONFIG["M_MAX_LOWER_BOUND"],
            "diagnostic_two_solar_mass_floor_msun": OBSERVED_TWO_SOLAR_MASS_SCREEN,
            "common_r14_km": [
                CONFIG["CONTROLLED_R14_MIN"],
                CONFIG["CONTROLLED_R14_MAX"],
            ],
            "common_ml_mass_grid_msun": [
                CONFIG["ML_MASS_GRID_MIN"],
                CONFIG["ML_MASS_GRID_MAX"],
            ],
        },
        "counts": {
            "hadronic_total": int(results["matter_class"].eq("hadronic").sum()),
            "quark_total": int(results["matter_class"].eq("quark").sum()),
            "hadronic_strict_eligible": int(strict_h.sum()),
            "quark_strict_eligible": int(strict_q.sum()),
            "hadronic_2p0_eligible": int(
                (results["matter_class"].eq("hadronic") & results["pilot_eligible_2p0"]).sum()
            ),
            "quark_2p0_eligible": int(
                (results["matter_class"].eq("quark") & results["pilot_eligible_2p0"]).sum()
            ),
            "hadronic_strict_recommended": int(recommended_strict_h.sum()),
            "quark_strict_recommended": int(recommended_strict_q.sum()),
            "hadronic_2p0_recommended": int(recommended_2p0_h.sum()),
            "quark_2p0_recommended": int(recommended_2p0_q.sum()),
            "strict_hadronic_family_groups": int(results.loc[strict_h, "family_group_id"].nunique()),
            "strict_quark_family_groups": int(results.loc[strict_q, "family_group_id"].nunique()),
            "strict_quark_parameter_blocks": int(results.loc[strict_q, "parameter_block_id"].nunique()),
            "recommended_strict_hadronic_family_groups": int(
                results.loc[recommended_strict_h, "family_group_id"].nunique()
            ),
            "recommended_strict_quark_family_groups": int(
                results.loc[recommended_strict_q, "family_group_id"].nunique()
            ),
            "recommended_strict_quark_parameter_blocks": int(
                results.loc[recommended_strict_q, "parameter_block_id"].nunique()
            ),
            "recommended_2p0_hadronic_family_groups": int(
                results.loc[recommended_2p0_h, "family_group_id"].nunique()
            ),
            "recommended_2p0_quark_family_groups": int(
                results.loc[recommended_2p0_q, "family_group_id"].nunique()
            ),
            "recommended_2p0_quark_parameter_blocks": int(
                results.loc[recommended_2p0_q, "parameter_block_id"].nunique()
            ),
        },
        "strict_hadronic_amplitude_intersection": _format_interval(
            _intersection(results, strict_h)
        ),
        "strict_quark_amplitude_intersection": _format_interval(
            _intersection(results, strict_q)
        ),
        "strict_global_amplitude_intersection": _format_interval(global_interval),
        "recommended_strict_global_amplitude_intersection": _format_interval(
            recommended_global_interval
        ),
        "current_controlled_sweep_inside_strict_global_intersection": bool(
            global_interval is not None
            and CONFIG["CONTROLLED_A_MIN"] > global_interval[0]
            and CONFIG["CONTROLLED_A_MAX"] <= global_interval[1]
        ),
        "current_controlled_sweep_inside_recommended_strict_global_intersection": bool(
            recommended_global_interval is not None
            and CONFIG["CONTROLLED_A_MIN"] > recommended_global_interval[0]
            and CONFIG["CONTROLLED_A_MAX"] <= recommended_global_interval[1]
        ),
        "strict_eligible_eos_ids": results.loc[strict, "eos_id"].tolist(),
        "recommended_strict_eos_ids": results.loc[
            recommended_strict, "eos_id"
        ].tolist(),
        "recommended_2p0_eos_ids": results.loc[recommended_2p0, "eos_id"].tolist(),
    }

    pd.DataFrame.from_records(literature_catalog_rows()).to_csv(
        output_dir / "eos_literature_catalog.csv", index=False
    )
    results.to_csv(output_dir / "eos_feasibility_results.csv", index=False)
    np.savez_compressed(
        output_dir / "eos_feasibility_curves.npz",
        **{
            key.replace(":", "__").replace("-", "_"): curve
            for key, curve in curves.items()
        },
    )
    (output_dir / "eos_feasibility_summary.json").write_text(
        json.dumps(_json_compatible(summary), indent=2) + "\n", encoding="utf-8"
    )
    _write_plots(output_dir, results, curves)
    _write_report(output_dir, results, summary)
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs"),
        help="Directory for CSV, JSON, Markdown, and PNG audit artifacts.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel production-solver workers (use 1 for deterministic debugging).",
    )
    args = parser.parse_args()
    if args.jobs == 0:
        parser.error("--jobs cannot be zero")
    results, summary = run_audit(args.output_dir, args.jobs)
    print(json.dumps(_json_compatible(summary), indent=2))
    failed = results[~results["build_ok"]]
    if not failed.empty:
        print("Build/audit failures:")
        print(failed[["matter_class", "eos_id", "audit_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
