import numpy as np

from src.config import CONFIG
from src.physics.solve_sequence import solve_sequence


class RecordingRejectedEos:
    def __init__(self):
        self.pressures = []
        self.eps_surf = 0.0

    def __call__(self, pressure):
        self.pressures.append(float(pressure))
        return np.nan, 0.5


def test_hadronic_pressure_grid_contract_without_running_ode_solver():
    eos = RecordingRejectedEos()

    result = solve_sequence(
        eos,
        is_quark=False,
        p_max_causal=500.0,
        n_points=10,
    )

    expected = np.geomspace(CONFIG["GRID_P_MIN_LOG"], 500.0, 10)
    np.testing.assert_array_equal(eos.pressures, expected)
    assert len(eos.pressures) == 10
    assert result == ([], [], 0.0)


def test_quark_pressure_grid_contract_without_running_ode_solver():
    eos = RecordingRejectedEos()

    result = solve_sequence(
        eos,
        is_quark=True,
        p_max_causal=10_000.0,
        n_points=10,
    )

    low_count = int(10 * CONFIG["SOLVER_N_LOW_RATIO"])
    expected = np.concatenate(
        (
            np.logspace(-1.0, 2.0, low_count, endpoint=False),
            np.logspace(2.0, 4.0, 10 - low_count),
        )
    )
    np.testing.assert_array_equal(eos.pressures, expected)
    assert len(eos.pressures) == 10
    assert eos.pressures[low_count - 1] < 100.0
    assert eos.pressures[low_count] == 100.0
    assert eos.pressures[-1] == 10_000.0
    assert result == ([], [], 0.0)
