import importlib

import numpy as np
import pytest

from src.physics import solve_sequence as sequence_module
from src.physics.solve_sequence import solve_sequence
from tests.stellar_integration_support import (
    RecordingEos,
    SyntheticSolution,
    capture_log,
)


def test_initial_state_arithmetic_errors_propagate_before_integration(monkeypatch):
    class FailingEnergyDensity(float):
        def __rmul__(self, other):
            raise FloatingPointError("synthetic initialization failure")

    eos_calls = []
    solver_calls = []
    logged = []

    def failing_energy_eos(pressure):
        eos_calls.append(pressure)
        return FailingEnergyDensity(250.0), 0.42

    def fail_if_integrated(**arguments):
        solver_calls.append(arguments)

    monkeypatch.setattr(sequence_module, "solve_ivp", fail_if_integrated)
    monkeypatch.setattr(
        sequence_module.logger,
        "exception",
        capture_log(logged),
    )

    with pytest.raises(FloatingPointError, match="synthetic initialization failure"):
        solve_sequence(failing_energy_eos, n_points=4)

    assert len(eos_calls) == 1
    assert solver_calls == []
    assert logged == []


@pytest.mark.parametrize(
    ("mass", "radius", "message"),
    [
        (np.nan, np.inf, "NaN detected in TOV Mass or Radius!"),
        (np.inf, 0.0, "Inf detected in TOV Mass or Radius!"),
        (0.0, 0.0, "Unphysical mass detected"),
        (1.0, 0.0, "Unphysical radius detected"),
    ],
)
def test_surface_boundary_invariants_propagate_in_order(
    monkeypatch,
    mass,
    radius,
    message,
):
    eos = RecordingEos()
    solver_calls = []
    logged = []

    def fake_solve_ivp(**arguments):
        solver_calls.append(arguments)
        return SyntheticSolution(
            initial_state=arguments["y0"],
            mass=mass,
            radius=radius,
        )

    monkeypatch.setattr(sequence_module, "solve_ivp", fake_solve_ivp)
    monkeypatch.setattr(
        sequence_module.logger,
        "exception",
        capture_log(logged),
    )

    with pytest.raises(AssertionError, match=message):
        solve_sequence(eos, n_points=4)

    assert len(eos.pressures) == len(solver_calls) == 1
    assert logged == []


@pytest.mark.parametrize(
    ("exception_type", "is_expected"),
    [
        (ValueError, True),
        (ArithmeticError, True),
        (TypeError, False),
    ],
)
def test_dense_postprocessing_preserves_exception_policy(
    monkeypatch,
    exception_type,
    is_expected,
):
    eos = RecordingEos()
    solver_calls = []
    logged = []

    def fake_solve_ivp(**arguments):
        solver_calls.append(arguments)
        solution = SyntheticSolution(initial_state=arguments["y0"])

        def failing_dense_solution(radii):
            raise exception_type("synthetic dense-output failure")

        solution.sol = failing_dense_solution
        return solution

    monkeypatch.setattr(sequence_module, "solve_ivp", fake_solve_ivp)
    monkeypatch.setattr(
        sequence_module.logger,
        "exception",
        capture_log(logged),
    )

    if is_expected:
        assert solve_sequence(eos, n_points=4) == ([], [], 0.0)
        assert len(eos.pressures) == len(solver_calls) == len(logged) == 4
    else:
        with pytest.raises(TypeError, match="synthetic dense-output failure"):
            solve_sequence(eos, n_points=4)
        assert len(eos.pressures) == len(solver_calls) == 1
        assert logged == []


def test_sequence_consumer_distinguishes_accept_skip_and_stop(monkeypatch):
    eos = RecordingEos()
    profile = (np.array([0.1, 12.0]), np.array([0.01, 1.0]))
    accepted_row = [1.0, 12.0, 100.0, 1.0, 250.0, 0.42, 0.0]
    decisions = [
        sequence_module._integration.IntegratedStellarModel(
            curve_row=accepted_row,
            dense_profile=profile,
        ),
        sequence_module._integration.IntegrationDecision.SKIP,
        sequence_module._integration.IntegrationDecision.STOP,
    ]
    integration_calls = []
    extraction_calls = []

    def fake_integrate(**arguments):
        integration_calls.append(arguments)
        if not decisions:
            pytest.fail("integration continued after STOP")
        return decisions.pop(0)

    def fake_extract(curve_data, dense_profiles):
        extraction_calls.append((curve_data, dense_profiles))
        return curve_data, dense_profiles, 1.0

    monkeypatch.setattr(
        sequence_module._integration,
        "integrate_stellar_model",
        fake_integrate,
    )
    monkeypatch.setattr(
        sequence_module,
        "_extract_first_turning_point",
        fake_extract,
    )

    result = solve_sequence(eos, n_points=4)

    assert len(eos.pressures) == len(integration_calls) == 3
    assert decisions == []
    assert len(extraction_calls) == 1
    extracted_rows, extracted_profiles = extraction_calls[0]
    assert extracted_rows == [accepted_row]
    assert len(extracted_profiles) == 1
    np.testing.assert_array_equal(extracted_profiles[0][0], profile[0])
    np.testing.assert_array_equal(extracted_profiles[0][1], profile[1])

    result_rows, result_profiles, result_maximum = result
    assert result_rows == [accepted_row]
    assert len(result_profiles) == 1
    np.testing.assert_array_equal(result_profiles[0][0], profile[0])
    np.testing.assert_array_equal(result_profiles[0][1], profile[1])
    assert result_maximum == 1.0


def test_reloaded_integration_module_keeps_facade_decisions_aligned(monkeypatch):
    importlib.reload(sequence_module._integration)
    solver_calls = []

    def unsuccessful_solve_ivp(**arguments):
        solver_calls.append(arguments)
        return SyntheticSolution(
            initial_state=arguments["y0"],
            surfaced=True,
            status=0,
        )

    monkeypatch.setattr(sequence_module, "solve_ivp", unsuccessful_solve_ivp)

    assert solve_sequence(RecordingEos(), n_points=4) == ([], [], 0.0)
    assert len(solver_calls) == 4
