import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from framework.eos_sweep import (
    GaussianDeformation,
    QuarkParameters,
    SweepPoint,
    amplitude_grid,
    build_quark_eos,
    cfl_baseline_grids,
)
from physics_main import _validate_controlled_dataset
from src.config import CONFIG
from src.physics import worker_hadronic_gen as hadronic_worker_module
from src.physics import worker_quark_gen as quark_worker_module
from src.physics import controlled_generation as controlled_generation_module
from src.physics.worker_quark_gen import controlled_quark_parameters


def _fake_sequence_result(framework_eos, *, is_quark):
    del framework_eos, is_quark
    curve = [
        [1.0, 12.0, 500.0, 10.0, 200.0, 0.2, 0.0],
        [1.4, 11.5, 200.0, 40.0, 350.0, 0.3, 0.0],
        [2.0, 11.0, 30.0, 100.0, 600.0, 0.5, 0.0],
    ]
    features = {
        "cs2_at_14": 0.3,
        "r_14": 11.5,
        "slopes": {1.4: -0.5, 1.6: -0.6, 1.8: -0.7, 2.0: -0.8},
    }
    return curve, features, 2.2


def test_default_amplitude_grid_is_deterministic_and_contains_control():
    points = amplitude_grid(
        CONFIG["CONTROLLED_A_MIN"],
        CONFIG["CONTROLLED_A_MAX"],
        CONFIG["CONTROLLED_A_POINTS"],
    )
    assert points[0].amplitude == pytest.approx(-0.05)
    assert points[-1].amplitude == pytest.approx(0.09)
    assert any(point.amplitude == pytest.approx(0.0) for point in points)
    assert [point.sweep_id for point in points] == [
        f"A{index:05d}" for index in range(len(points))
    ]


def test_published_cfl4_surface_and_pressure_reconstruction():
    parameters = controlled_quark_parameters()
    assert parameters.bag_b == 60.0
    assert parameters.gap_delta == 100.0
    assert parameters.strange_mass == 150.0

    _, energy, baseline_cs2, energy_per_baryon = cfl_baseline_grids(parameters)
    assert energy[0] == pytest.approx(215.8982059, rel=1e-7)
    assert baseline_cs2[0] == pytest.approx(0.36013584, rel=1e-7)
    assert energy_per_baryon == pytest.approx(791.6274292, rel=1e-7)

    eos = build_quark_eos(
        parameters,
        GaussianDeformation(0.05, 220.0, 50.0),
        maximum_surface_energy_per_baryon=CONFIG["M_N"],
    )
    reconstructed = np.gradient(eos.pressure, eos.energy_density)
    assert np.allclose(
        reconstructed[2:-2], eos.sound_speed_squared[2:-2], rtol=2e-4, atol=2e-5
    )
    assert eos.eos_callable.eps_surf == pytest.approx(eos.eps_surface)


def test_published_cfl_catalog_can_include_massless_strange_limit():
    parameters = QuarkParameters(bag_b=60.0, gap_delta=50.0, strange_mass=0.0)
    _, energy, sound_speed, energy_per_baryon = cfl_baseline_grids(parameters)

    assert np.all(np.isfinite(energy))
    assert np.all((sound_speed > 0.0) & (sound_speed <= 1.0))
    assert energy_per_baryon <= CONFIG["M_N"]

    with pytest.raises(ValueError, match="non-negative"):
        QuarkParameters(bag_b=60.0, gap_delta=50.0, strange_mass=-1.0)


def test_workers_emit_class_paired_fixed_metadata(monkeypatch):
    monkeypatch.setattr(
        hadronic_worker_module, "solve_and_validate_sequence", _fake_sequence_result
    )
    monkeypatch.setattr(
        quark_worker_module, "solve_and_validate_sequence", _fake_sequence_result
    )
    point = SweepPoint(index=5, amplitude=0.0)

    hadronic = hadronic_worker_module.worker_hadronic_gen([point])
    quark = quark_worker_module.worker_quark_gen([point])
    combined = pd.concat([hadronic, quark], ignore_index=True)

    assert set(combined["Sweep_ID"]) == {point.sweep_id}
    assert set(combined["Label"]) == {0, 1}
    assert set(combined["Perturb_eps0"]) == {220.0}
    assert set(combined["Perturb_sigma"]) == {50.0}
    assert set(combined["Perturb_A"]) == {0.0}
    assert set(hadronic["Baseline_Name"]) == {"APR-1"}
    assert set(quark["Bag_B"]) == {60.0}
    assert set(quark["Gap_Delta"]) == {100.0}
    assert set(quark["Mass_Strange"]) == {150.0}

    _validate_controlled_dataset(combined, [point])

    wrong_deformation = combined.copy()
    wrong_deformation["Perturb_eps0"] = 221.0
    with pytest.raises(RuntimeError, match="Perturb_eps0"):
        _validate_controlled_dataset(wrong_deformation, [point])

    second_curve = hadronic.iloc[[0]].copy()
    second_curve["Curve_ID"] = "H_duplicate_curve"
    duplicate_pair = pd.concat([combined, second_curve], ignore_index=True)
    with pytest.raises(RuntimeError, match="exactly one Curve_ID"):
        _validate_controlled_dataset(duplicate_pair, [point])


def test_sequence_validation_supports_declared_mass_sensitivity_profile(monkeypatch):
    curve = [
        [0.2, 15.0, 1000.0, 1.0, 100.0, 0.1, 0.0],
        [1.4, 12.0, 200.0, 40.0, 350.0, 0.3, 0.0],
        [2.04, 11.0, 20.0, 120.0, 700.0, 0.5, 0.0],
    ]
    monkeypatch.setattr(
        controlled_generation_module,
        "solve_sequence",
        lambda *args, **kwargs: (curve, [], 2.04),
    )
    monkeypatch.setattr(
        controlled_generation_module, "verify_eos_physical_validity", lambda _: True
    )
    monkeypatch.setattr(
        controlled_generation_module,
        "extract_features",
        lambda *args: {
            "r_14": 12.0,
            "cs2_at_14": 0.3,
            "slopes": {1.4: -0.5, 1.6: -0.5, 1.8: -0.5, 2.0: -0.5},
        },
    )
    eos = SimpleNamespace(eos_callable=lambda _: (1.0, 0.3), p_max_causal=1000.0)

    with pytest.raises(RuntimeError, match="2.08"):
        controlled_generation_module.solve_and_validate_sequence(eos, is_quark=False)

    _, _, maximum_mass = controlled_generation_module.solve_and_validate_sequence(
        eos,
        is_quark=False,
        minimum_maximum_mass=2.0,
        maximum_maximum_mass=3.0,
        radius_14_bounds=(9.5, 14.5),
    )
    assert maximum_mass == pytest.approx(2.04)
