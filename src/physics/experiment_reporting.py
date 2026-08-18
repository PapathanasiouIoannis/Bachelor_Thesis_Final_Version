"""Professional output tables, figures, and reports for controlled EoS runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as _plt
import pandas as pd

from framework.eos_sweep import tabulate_complete_eos
from src.physics.reporting.frames import (
    build_causal_domain_table,
    build_summary_table as _build_summary_table_impl,
    eos_to_frame as _eos_to_frame_impl,
    serialize_eos_table as _serialize_eos_table_impl,
    stellar_curve_to_frame as stellar_curve_to_frame,
    summarize_stellar_curve as summarize_stellar_curve,
    validate_eos_frame,
)
from src.physics.reporting.markdown import (
    _markdown_table as _markdown_table_impl,
    write_markdown_report as _write_markdown_report_impl,
)
from src.physics.reporting.plots import create_standard_plots as create_standard_plots
from src.physics.reporting.schemas import (
    CAUSAL_DOMAIN_COLUMNS as CAUSAL_DOMAIN_COLUMNS,
    CAUSAL_DOMAIN_HEADINGS as CAUSAL_DOMAIN_HEADINGS,
    CONVERGENCE_HEADINGS as CONVERGENCE_HEADINGS,
    EOS_COLUMNS as EOS_COLUMNS,
    REJECTION_HEADINGS as REJECTION_HEADINGS,
    STELLAR_COLUMNS as STELLAR_COLUMNS,
    SUMMARY_COLUMNS as SUMMARY_COLUMNS,
    SUMMARY_HEADINGS as SUMMARY_HEADINGS,
)


plt = _plt


def serialize_eos_table(
    framework_eos: Any, matter_type: str, sweep_id: str
) -> pd.DataFrame:
    """Serialize the complete solved EoS domain before validity screening."""

    return _serialize_eos_table_impl(
        framework_eos,
        matter_type,
        sweep_id,
        tabulate_eos=tabulate_complete_eos,
    )


def eos_to_frame(framework_eos: Any, matter_type: str, sweep_id: str) -> pd.DataFrame:
    """Serialize and validate the complete framework EoS table."""

    return _eos_to_frame_impl(
        framework_eos,
        matter_type,
        sweep_id,
        serializer=serialize_eos_table,
        validator=validate_eos_frame,
    )


def build_summary_table(stellar_curves: pd.DataFrame) -> pd.DataFrame:
    return _build_summary_table_impl(
        stellar_curves,
        summarizer=summarize_stellar_curve,
    )


def write_markdown_report(
    eos_tables: pd.DataFrame,
    summary: pd.DataFrame,
    rejections: pd.DataFrame,
    convergence: pd.DataFrame,
    resolved_configuration: dict[str, Any],
    output_path: Path,
    *,
    run_status: str,
) -> Path:
    """Write a concise report with the experiment's scientific boundary."""

    return _write_markdown_report_impl(
        eos_tables,
        summary,
        rejections,
        convergence,
        resolved_configuration,
        output_path,
        run_status=run_status,
        causal_builder=build_causal_domain_table,
        table_renderer=_markdown_table,
    )


def _markdown_table(
    frame: pd.DataFrame,
    headings: dict[str, str] | None = None,
) -> str:
    return _markdown_table_impl(frame, headings)
