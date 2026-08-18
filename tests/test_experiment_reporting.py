from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.physics import experiment_reporting
from src.physics.experiment_reporting import (
    CAUSAL_DOMAIN_COLUMNS,
    EOS_COLUMNS,
    SUMMARY_COLUMNS,
    STELLAR_COLUMNS,
    build_causal_domain_table,
    eos_to_frame,
    serialize_eos_table,
    stellar_curve_to_frame,
    summarize_stellar_curve,
    write_markdown_report,
)


def test_stellar_curve_serialization_and_summary_happy_path():
    framework_eos = SimpleNamespace(
        baseline_name="APR-1[A=0.1]",
        catalog_identifier="APR-1",
        deformation=SimpleNamespace(amplitude=0.1),
    )
    curve = [
        [1.0, 12.5, 600.0, 10.0, 100.0, 0.50, 0.0],
        [1.2, 12.4, 520.0, 20.0, 200.0, 0.51, 0.0],
        [1.4, 12.3, 440.0, 30.0, 300.0, 0.52, 0.0],
        [1.6, 12.2, 360.0, 40.0, 400.0, 0.53, 0.0],
        [2.0, 12.0, 200.0, 50.0, 500.0, 0.54, 0.0],
    ]

    frame = stellar_curve_to_frame(
        curve,
        framework_eos,
        matter_type="hadronic",
        sweep_id="sweep-001",
        curve_id="curve-001",
    )

    assert tuple(frame.columns) == STELLAR_COLUMNS
    assert frame["matter_type"].unique().tolist() == ["hadronic"]
    assert frame["baseline_name"].unique().tolist() == ["APR-1"]
    assert frame["model_identifier"].unique().tolist() == ["APR-1[A=0.1]"]
    assert frame["sweep_id"].unique().tolist() == ["sweep-001"]
    assert frame["curve_id"].unique().tolist() == ["curve-001"]
    assert frame["mass_msun"].tolist() == pytest.approx(
        [1.0, 1.2, 1.4, 1.6, 2.0]
    )
    assert frame["central_pressure_mev_fm3"].tolist() == pytest.approx(
        [10.0, 20.0, 30.0, 40.0, 50.0]
    )

    summary = summarize_stellar_curve(frame)

    assert summary == {
        "matter_type": "hadronic",
        "baseline_name": "APR-1",
        "sweep_id": "sweep-001",
        "deformation_amplitude": pytest.approx(0.1),
        "maximum_mass_msun": pytest.approx(2.0),
        "radius_1p4_km": pytest.approx(12.3),
        "tidal_deformability_1p4": pytest.approx(440.0),
        "turning_point_stability_estimate": True,
        "status": "accepted",
    }


def test_eos_serialization_validation_and_causal_summary_contract(monkeypatch):
    framework_eos = SimpleNamespace(
        baseline_name="APR-1[A=0.1]",
        catalog_identifier="APR-1",
        deformation=SimpleNamespace(amplitude=0.1),
        discarded_suffix_points=2,
        first_discarded_sound_speed_squared=1.05,
        pressure=np.array([1.0, 20.0, 50.0, 90.0]),
        energy_density=np.array([100.0, 200.0, 300.0, 400.0]),
    )
    monkeypatch.setattr(
        experiment_reporting,
        "tabulate_complete_eos",
        lambda eos: (
            eos.pressure,
            eos.energy_density,
            np.array([0.2, 0.3, 0.4, 0.5]),
            np.array(["crust", "core", "core", "core"]),
        ),
    )

    serialized = serialize_eos_table(framework_eos, "hadronic", "sweep-001")
    validated = eos_to_frame(framework_eos, "hadronic", "sweep-001")

    assert tuple(serialized.columns) == EOS_COLUMNS
    assert serialized["baseline_name"].unique().tolist() == ["APR-1"]
    assert serialized["model_identifier"].unique().tolist() == ["APR-1[A=0.1]"]
    assert serialized["eos_validation_passed"].isna().all()
    assert serialized["eos_validation_reason"].unique().tolist() == ["not_checked"]
    assert validated["eos_validation_passed"].astype(bool).all()
    assert validated["eos_validation_reason"].unique().tolist() == ["passed"]
    assert validated["causal_prefix_applied"].astype(bool).all()
    assert validated["discarded_suffix_points"].unique().tolist() == [2]

    causal = build_causal_domain_table(pd.concat([validated, validated]))

    assert tuple(causal.columns) == CAUSAL_DOMAIN_COLUMNS
    assert len(causal) == 1
    assert causal.loc[0, "causal_cutoff_pressure_mev_fm3"] == pytest.approx(90.0)
    assert causal.loc[0, "causal_cutoff_energy_density_mev_fm3"] == pytest.approx(
        400.0
    )
    assert causal.loc[0, "first_discarded_sound_speed_squared"] == pytest.approx(
        1.05
    )


@pytest.mark.parametrize(
    ("convergence_check", "expected_convergence_text"),
    [
        ("none", "Convergence checks were disabled by this exploratory configuration."),
        ("endpoints_and_zero", "No convergence results were produced."),
    ],
)
def test_markdown_report_preserves_semantic_sections_and_empty_run_branches(
    tmp_path, convergence_check, expected_convergence_text
):
    eos_tables = pd.DataFrame(columns=EOS_COLUMNS)
    summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    rejections = pd.DataFrame.from_records(
        [
            {
                "sweep_id": "A00000",
                "deformation_amplitude": 0.0,
                "matter_type": "hadronic",
                "stage": "eos_validation",
                "exception_type": "ValueError",
                "reason": "The generated energy-density grid is not strictly increasing.",
            }
        ]
    )
    runtime = {
        "deformation": {
            "center_energy_density_mev_fm3": 220.0,
            "width_mev_fm3": 50.0,
        },
        "hadronic_eos": {"baseline": "APR-1"},
        "quark_eos": {
            "bag_constant_mev_fm3": 60.0,
            "pairing_gap_mev": 100.0,
            "strange_quark_mass_mev": 150.0,
        },
        "physical_requirements": {
            "minimum_maximum_mass_msun": 2.08,
            "maximum_maximum_mass_msun": 3.2,
            "radius_1p4_min_km": 9.0,
            "radius_1p4_max_km": 15.0,
        },
        "execution": {
            "random_seed": 20260804,
            "parallel_jobs": 1,
            "amplitudes_per_batch": 3,
        },
        "resolved_numerical_settings": {
            "eos_grid_points": 5000,
            "central_pressure_points": 80,
            "tov_relative_tolerance": 1.0e-7,
            "tov_absolute_tolerance": 1.0e-9,
        },
        "numerical_settings": {"convergence_check": convergence_check},
    }
    output_path = tmp_path / "report.md"

    write_markdown_report(
        eos_tables,
        summary,
        rejections,
        pd.DataFrame(),
        runtime,
        output_path,
        run_status="completed_with_rejections",
    )
    report = output_path.read_text(encoding="utf-8")

    headings = [
        "## Experiment",
        "## Physical acceptance requirements",
        "## Canonical observables",
        "## EoS validation and causal domains",
        "## Rejected amplitude pairs",
        "## Numerical convergence",
        "## Interpretation",
    ]
    assert [report.index(heading) for heading in headings] == sorted(
        report.index(heading) for heading in headings
    )
    assert "Terminal run status: `completed_with_rejections`" in report
    assert "not a universal matter-phase classifier" in report
    assert "No stellar curve passed" in report
    assert "downward energy-density step" in report
    assert expected_convergence_text in report
