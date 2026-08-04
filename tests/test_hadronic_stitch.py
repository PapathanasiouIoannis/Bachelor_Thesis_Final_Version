import numpy as np
import pytest

from src.config import CONFIG
from src.physics.worker_hadronic_gen import (
    build_anchored_sos_spline,
    resolve_density_shifted_transition,
)
def _constant_crust(offset=100.0, slope=10.0):
    eps = lambda p: offset + slope * np.asarray(p)
    deps = lambda p: np.full_like(np.asarray(p, dtype=float), slope, dtype=float)
    return {key: (eps, deps) for key in ["c1", "c2", "c3", "c4"]}


def test_hadronic_stitch_removes_density_jump_with_core_shift():
    crust = _constant_crust()
    core_anchor = (
        lambda p: 150.0 + 10.0 * np.asarray(p),
        lambda p: np.full_like(np.asarray(p, dtype=float), 10.0, dtype=float),
    )

    p_trans, eps_shift = resolve_density_shifted_transition(
        crust, core_anchor, CONFIG["P_TRANS_DEFAULT"]
    )

    assert p_trans == pytest.approx(CONFIG["P_TRANS_DEFAULT"])
    assert eps_shift == pytest.approx(50.0)

    np.random.seed(42)
    _, _, eps_sliced, _, _, _, _, _ = build_anchored_sos_spline(
        crust, core_anchor, CONFIG["P_TRANS_DEFAULT"]
    )
    crust_eps = float(crust["c1"][0](np.array([p_trans]))[0])
    assert float(eps_sliced[0]) == pytest.approx(crust_eps)


def test_hadronic_stitch_accepts_only_continuous_transition():
    crust = _constant_crust()
    core_anchor = (
        lambda p: 100.0 + 10.0 * np.asarray(p),
        lambda p: np.full_like(np.asarray(p, dtype=float), 10.0, dtype=float),
    )

    np.random.seed(42)
    p_trans, _, eps_sliced, _, _, _, _, _ = build_anchored_sos_spline(
        crust, core_anchor, CONFIG["P_TRANS_DEFAULT"]
    )

    crust_eps = float(crust["c1"][0](np.array([p_trans]))[0])
    tol = max(
        CONFIG["CRUST_CORE_EPS_ABS_TOL"],
        CONFIG["CRUST_CORE_EPS_REL_TOL"] * max(abs(crust_eps), abs(float(eps_sliced[0])), 1.0),
    )
    assert abs(float(eps_sliced[0]) - crust_eps) <= tol
