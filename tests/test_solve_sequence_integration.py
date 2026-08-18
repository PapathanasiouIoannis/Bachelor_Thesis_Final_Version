import numpy as np
import pytest

from src.config import CONFIG
from src.physics import solve_sequence as sequence_module
from src.physics.solve_sequence import solve_sequence


class RecordingEos:
    def __init__(self, *, eps_surf=0.0):
        self.pressures = []
        self.eps_surf = eps_surf

    def __call__(self, pressure):
        self.pressures.append(float(pressure))
        return float(250.0 + pressure), 0.42


class SyntheticSolution:
    def __init__(
        self,
        *,
        initial_state,
        mass=1.0,
        radius=12.0,
        y_surface=0.8,
        surfaced=True,
        status=None,
    ):
        self.status = (1 if surfaced else 0) if status is None else status
        self.mass = mass
        self.radius = radius
        self.y_surface = y_surface
        self.initial_state = initial_state
        self.sampled_radii = None
        if surfaced:
            self.t_events = [np.array([radius])]
            self.y_events = [
                np.array(
                    [
                        [
                            mass,
                            CONFIG["SURFACE_PRESSURE_EVENT_CUTOFF"],
                            y_surface,
                        ]
                    ]
                )
            ]
        else:
            self.t_events = [np.array([])]
            self.y_events = [np.empty((0, 3))]

    def sol(self, radii):
        self.sampled_radii = np.asarray(radii)
        sample_count = len(self.sampled_radii)
        return np.vstack(
            (
                np.linspace(self.initial_state[0], self.mass, sample_count),
                np.linspace(
                    self.initial_state[1],
                    CONFIG["SURFACE_PRESSURE_EVENT_CUTOFF"],
                    sample_count,
                ),
                np.linspace(self.initial_state[2], self.y_surface, sample_count),
            )
        )


def _capture_log(records):
    def capture(*args, **kwargs):
        records.append((args, kwargs))

    return capture


def test_mixed_solver_outcomes_preserve_curve_and_profile_assembly(
    monkeypatch,
):
    eos = RecordingEos(eps_surf=5.0)
    solver_calls = []
    solutions = []
    logged = []
    masses = [0.5, 1.0, 1.5, 1.4]
    radii = [12.2, 12.3, 12.4, 12.5]

    def fake_solve_ivp(**arguments):
        call_index = len(solver_calls)
        solver_calls.append(arguments)
        if call_index == 0:
            raise ValueError("synthetic convergence failure")
        if call_index == 1:
            return SyntheticSolution(
                initial_state=arguments["y0"],
                surfaced=False,
            )
        solution = SyntheticSolution(
            initial_state=arguments["y0"],
            mass=masses[call_index - 2],
            radius=radii[call_index - 2],
        )
        solutions.append(solution)
        return solution

    monkeypatch.setattr(sequence_module, "solve_ivp", fake_solve_ivp)
    monkeypatch.setattr(
        sequence_module.logger,
        "exception",
        _capture_log(logged),
    )

    curve, profiles, maximum_mass = solve_sequence(
        eos,
        is_quark=False,
        p_max_causal=600.0,
        rtol=2.0e-8,
        atol=3.0e-10,
        n_points=6,
    )

    expected_pressures = np.geomspace(CONFIG["GRID_P_MIN_LOG"], 600.0, 6)
    np.testing.assert_array_equal(eos.pressures, expected_pressures)
    assert len(solver_calls) == 6
    assert logged == [(("ODE Solver failed due to domain error",), {})]

    assert isinstance(curve, list)
    assert isinstance(profiles, list)
    assert len(curve) == len(profiles) == 3
    assert all(isinstance(row, list) and len(row) == 7 for row in curve)
    np.testing.assert_allclose([row[0] for row in curve], [0.5, 1.0, 1.5])
    np.testing.assert_allclose([row[1] for row in curve], radii[:3])
    np.testing.assert_allclose(
        [row[2] for row in curve],
        [120897.57119496442, 2643.05727192278, 249.8413534873957],
        rtol=1.0e-12,
    )
    np.testing.assert_array_equal(
        [row[3] for row in curve],
        expected_pressures[2:5],
    )
    np.testing.assert_allclose(
        [row[4] for row in curve],
        250.0 + expected_pressures[2:5],
    )
    np.testing.assert_array_equal([row[5] for row in curve], [0.42] * 3)
    np.testing.assert_array_equal([row[6] for row in curve], [5.0] * 3)
    assert maximum_mass == pytest.approx(1.5)

    for row, profile, solution in zip(curve, profiles, solutions[:3]):
        radii_profile, masses_profile = profile
        assert len(radii_profile) == len(masses_profile) == CONFIG[
            "DENSE_PROFILES_POINTS"
        ]
        assert radii_profile[0] == pytest.approx(CONFIG["TOV_R_MIN"])
        assert radii_profile[-1] == pytest.approx(row[1])
        assert masses_profile[-1] == pytest.approx(row[0])
        np.testing.assert_array_equal(solution.sampled_radii, radii_profile)

    for pressure, arguments in zip(expected_pressures, solver_calls):
        assert arguments["fun"] is sequence_module.tov_rhs
        assert arguments["t_span"] == (
            CONFIG["TOV_R_MIN"],
            CONFIG["TOV_R_MAX"],
        )
        assert arguments["args"] == (eos,)
        assert arguments["events"] is sequence_module._surface_event
        assert arguments["method"] == "RK45"
        assert arguments["dense_output"] is True
        assert arguments["rtol"] == 2.0e-8
        assert arguments["atol"] == 3.0e-10
        np.testing.assert_allclose(
            arguments["y0"],
            [
                CONFIG["TOV_R_MIN"] ** 3
                * (250.0 + pressure)
                * (CONFIG["G_CONV"] / 3.0),
                pressure,
                2.0,
            ],
            rtol=1.0e-14,
        )


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError, ArithmeticError])
def test_expected_integrator_errors_are_logged_and_skipped(
    monkeypatch,
    exception_type,
):
    eos = RecordingEos()
    solver_calls = []
    logged = []

    def failing_solve_ivp(**arguments):
        solver_calls.append(arguments)
        raise exception_type("synthetic integrator failure")

    monkeypatch.setattr(sequence_module, "solve_ivp", failing_solve_ivp)
    monkeypatch.setattr(
        sequence_module.logger,
        "exception",
        _capture_log(logged),
    )

    result = solve_sequence(eos, n_points=4)

    assert result == ([], [], 0.0)
    assert len(solver_calls) == len(logged) == 4
    assert all(call["rtol"] == CONFIG["ODE_RTOL"] for call in solver_calls)
    assert all(call["atol"] == CONFIG["ODE_ATOL"] for call in solver_calls)
    assert all(
        entry == (("ODE Solver failed due to domain error",), {}) for entry in logged
    )


def test_unexpected_integrator_errors_propagate(monkeypatch):
    solver_calls = []

    def failing_solve_ivp(**arguments):
        solver_calls.append(arguments)
        raise TypeError("synthetic programming error")

    monkeypatch.setattr(sequence_module, "solve_ivp", failing_solve_ivp)

    with pytest.raises(TypeError, match="synthetic programming error"):
        solve_sequence(RecordingEos(), n_points=4)

    assert len(solver_calls) == 1


def test_eos_errors_propagate_before_integration(monkeypatch):
    solver_calls = []

    def fail_if_called(**arguments):
        solver_calls.append(arguments)

    def failing_eos(pressure):
        raise ValueError(f"synthetic EoS failure at {pressure}")

    monkeypatch.setattr(sequence_module, "solve_ivp", fail_if_called)

    with pytest.raises(ValueError, match="synthetic EoS failure"):
        solve_sequence(failing_eos, n_points=4)

    assert solver_calls == []


@pytest.mark.parametrize(
    "rejection_case",
    [
        "unsuccessful_status",
        "missing_surface",
        "small_mass",
        "small_radius",
        "buchdahl_limit",
        "missing_lambda",
    ],
)
def test_unusable_surface_results_are_silently_rejected(
    monkeypatch,
    rejection_case,
):
    solver_calls = []
    mass = 1.0
    radius = 12.0
    surfaced = True
    status = 1
    if rejection_case == "unsuccessful_status":
        status = 0
    elif rejection_case == "missing_surface":
        surfaced = False
    elif rejection_case == "small_mass":
        mass = CONFIG["MIN_MASS_CUTOFF"] / 2.0
    elif rejection_case == "small_radius":
        mass = 2.0 * CONFIG["MIN_MASS_CUTOFF"]
        radius = CONFIG["MIN_RADIUS_CUTOFF"] / 2.0
    elif rejection_case == "buchdahl_limit":
        mass = (
            1.01
            * CONFIG["BUCHDAHL_LIMIT"]
            * radius
            / CONFIG["A_CONV"]
        )
    elif rejection_case == "missing_lambda":
        monkeypatch.setattr(
            sequence_module,
            "_tidal_lambda_from_y",
            lambda compactness, y_surface: None,
        )

    def fake_solve_ivp(**arguments):
        solver_calls.append(arguments)
        return SyntheticSolution(
            initial_state=arguments["y0"],
            mass=mass,
            radius=radius,
            surfaced=surfaced,
            status=status,
        )

    monkeypatch.setattr(sequence_module, "solve_ivp", fake_solve_ivp)

    result = solve_sequence(RecordingEos(), n_points=4)

    assert result == ([], [], 0.0)
    assert len(solver_calls) == 4
