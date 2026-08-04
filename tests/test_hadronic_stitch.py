import numpy as np
import pytest

from framework.eos_sweep import GaussianDeformation, build_hadronic_eos
from src.config import CONFIG
from src.physics.worker_hadronic_gen import resolve_density_shifted_transition


def _constant_crust(offset=100.0, slope=10.0):
    def eps(pressure):
        return offset + slope * np.asarray(pressure)

    def deps(pressure):
        return np.full_like(np.asarray(pressure, dtype=float), slope, dtype=float)

    return {key: (eps, deps) for key in ["c1", "c2", "c3", "c4"]}


def test_hadronic_stitch_removes_density_jump_with_core_shift():
    crust = _constant_crust()
    core_anchor = (
        lambda pressure: 150.0 + 10.0 * np.asarray(pressure),
        lambda pressure: np.full_like(
            np.asarray(pressure, dtype=float), 10.0, dtype=float
        ),
    )

    transition, energy_shift = resolve_density_shifted_transition(
        crust, core_anchor, CONFIG["P_TRANS_DEFAULT"]
    )

    assert transition == pytest.approx(CONFIG["P_TRANS_DEFAULT"])
    assert energy_shift == pytest.approx(50.0)


def test_framework_apr1_is_density_continuous_and_does_not_clip_cs2():
    eos = build_hadronic_eos(
        "APR-1",
        GaussianDeformation(
            amplitude=0.0,
            epsilon0=CONFIG["CONTROLLED_PERTURB_EPS0"],
            sigma=CONFIG["CONTROLLED_PERTURB_SIGMA"],
        ),
    )

    assert eos.transition_pressure == pytest.approx(CONFIG["P_TRANS_DEFAULT"])
    assert eos.energy_shift == pytest.approx(14.7431739111, rel=1e-9)
    crust_energy, _ = eos.eos_callable(eos.transition_pressure)
    assert eos.energy_density[0] == pytest.approx(crust_energy, rel=1e-10)
    assert np.all(eos.sound_speed_squared > 0.0)
    assert np.all(eos.sound_speed_squared <= 1.0)


def test_framework_uses_the_documented_ps_transition_pressure():
    eos = build_hadronic_eos(
        "PS",
        GaussianDeformation(
            amplitude=0.0,
            epsilon0=CONFIG["CONTROLLED_PERTURB_EPS0"],
            sigma=CONFIG["CONTROLLED_PERTURB_SIGMA"],
        ),
    )

    assert eos.transition_pressure == pytest.approx(0.696)
    crust_energy, _ = eos.eos_callable(eos.transition_pressure)
    assert eos.energy_density[0] == pytest.approx(crust_energy, rel=1e-10)
