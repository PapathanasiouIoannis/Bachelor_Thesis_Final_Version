"""Markdown report assembly for controlled EoS experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.physics.reporting.frames import build_causal_domain_table
from src.physics.reporting.schemas import (
    CAUSAL_DOMAIN_HEADINGS,
    CONVERGENCE_HEADINGS,
    REJECTION_HEADINGS,
    SUMMARY_COLUMNS,
    SUMMARY_HEADINGS,
)


def _markdown_table(frame: pd.DataFrame, headings: dict[str, str] | None = None) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy().rename(columns=headings or {})
    for column in display.select_dtypes(include=["float", "float32", "float64"]):
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_markdown_report(
    eos_tables: pd.DataFrame,
    summary: pd.DataFrame,
    rejections: pd.DataFrame,
    convergence: pd.DataFrame,
    resolved_configuration: dict[str, Any],
    output_path: Path,
    *,
    run_status: str,
    causal_builder=build_causal_domain_table,
    table_renderer=_markdown_table,
) -> Path:
    """Write a concise report with the experiment's scientific boundary."""

    deformation = resolved_configuration["deformation"]
    hadronic = resolved_configuration["hadronic_eos"]
    quark = resolved_configuration["quark_eos"]
    accepted = int(len(summary))
    rejected = int(len(rejections))
    requirements = resolved_configuration["physical_requirements"]
    execution = resolved_configuration["execution"]
    numerical = resolved_configuration["resolved_numerical_settings"]
    causal_domains = causal_builder(eos_tables)
    rejection_note = ""
    if (
        not rejections.empty
        and rejections["reason"]
        .astype(str)
        .str.contains("Maximum mass is not bracketed", regex=False)
        .any()
    ):
        rejection_note = (
            "At least one sequence reached the causal EoS boundary before a resolved "
            "post-peak mass decrease. Such an endpoint is not reported as a maximum "
            "mass."
        )
    elif (
        not rejections.empty
        and rejections["reason"]
        .astype(str)
        .str.contains("energy-density grid is not strictly increasing", regex=False)
        .any()
    ):
        rejection_note = (
            "The complete hadronic table contains a downward energy-density step "
            "inside the legacy crust fit. The runner preserved the diagnostic table "
            "and rejected the complete amplitude pair without smoothing the boundary."
        )
    interpretation = (
        "The retained stellar curves end at the first resolved mass turning point. "
        "This is a turning-point stability estimate and is not a radial-oscillation "
        "calculation."
        if not summary.empty
        else (
            "No stellar curve passed the preceding EoS and pair-level gates. If an "
            "EoS reaches stellar integration, it must still bracket a first mass "
            "turning point; that estimate is not a radial-oscillation calculation."
        )
    )
    lines = [
        "# Controlled EoS sensitivity report",
        "",
        "## Experiment",
        "",
        f"**Terminal run status: `{run_status}`**",
        "",
        f"This run compares one repository {hadronic['baseline']} surrogate with one analytic CFL "
        "MIT-bag baseline under the same additive Gaussian sound-speed deformation. "
        "It is a controlled model-pair sensitivity study, not a universal matter-phase classifier.",
        "",
        f"- Deformation centre, $\\epsilon_0$: {deformation['center_energy_density_mev_fm3']} MeV fm^-3",
        f"- Deformation width, $\\sigma$: {deformation['width_mev_fm3']} MeV fm^-3",
        f"- Bag constant, $B$: {quark['bag_constant_mev_fm3']} MeV fm^-3",
        f"- Pairing gap, $\\Delta$: {quark['pairing_gap_mev']} MeV",
        f"- Strange-quark mass, $m_s$: {quark['strange_quark_mass_mev']} MeV",
        f"- Random seed: {execution['random_seed']}",
        f"- Parallel jobs: {execution['parallel_jobs']}",
        f"- Amplitudes per worker batch: {execution['amplitudes_per_batch']}",
        f"- EoS grid points: {numerical['eos_grid_points']}",
        f"- Central-pressure points: {numerical['central_pressure_points']}",
        f"- TOV relative tolerance: {numerical['tov_relative_tolerance']}",
        f"- TOV absolute tolerance: {numerical['tov_absolute_tolerance']}",
        f"- Accepted curves: {accepted}",
        f"- Rejected amplitude pairs: {rejected}",
        "",
        "## Physical acceptance requirements",
        "",
        (
            "- Maximum mass: "
            f"{requirements['minimum_maximum_mass_msun']} to "
            f"{requirements['maximum_maximum_mass_msun']} $M_\\odot$"
        ),
        (
            "- Radius at $1.4M_\\odot$: "
            f"{requirements['radius_1p4_min_km']} to "
            f"{requirements['radius_1p4_max_km']} km"
        ),
        "",
        "## Canonical observables",
        "",
        table_renderer(summary.loc[:, SUMMARY_COLUMNS], SUMMARY_HEADINGS),
        "",
        "## EoS validation and causal domains",
        "",
        (
            "The framework retains values only through the last causal, stable grid "
            "point and records any discarded suffix below. Values are not clipped or "
            "silently repaired. For hadronic models, the complete table also includes "
            "the crust domain used by the stellar solver."
        ),
        "",
        table_renderer(causal_domains, CAUSAL_DOMAIN_HEADINGS),
        "",
        "## Rejected amplitude pairs",
        "",
        rejection_note,
        "",
        table_renderer(rejections, REJECTION_HEADINGS),
        "",
        "## Numerical convergence",
        "",
        table_renderer(convergence, CONVERGENCE_HEADINGS)
        if not convergence.empty
        else (
            "Convergence checks were disabled by this exploratory configuration."
            if resolved_configuration["numerical_settings"]["convergence_check"]
            == "none"
            else "No convergence results were produced."
        ),
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


__all__ = ["write_markdown_report"]
