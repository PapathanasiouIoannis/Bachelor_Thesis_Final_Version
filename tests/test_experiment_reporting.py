from types import SimpleNamespace

import pytest

from src.physics.experiment_reporting import (
    STELLAR_COLUMNS,
    stellar_curve_to_frame,
    summarize_stellar_curve,
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
