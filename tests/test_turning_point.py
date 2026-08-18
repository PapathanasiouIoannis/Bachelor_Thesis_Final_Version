import numpy as np
import pytest

from src.physics import solve_sequence as sequence_module
from src.physics.solve_sequence import (
    TurningPointError,
    _extract_first_turning_point,
    solve_sequence,
)
from src.physics.stellar import turning_point


def _curve(masses, pressures=None):
    pressures = pressures or list(range(1, len(masses) + 1))
    return [
        [mass, 12.0, 100.0, pressure, 300.0, 0.3, 0.0]
        for mass, pressure in zip(masses, pressures)
    ]


def test_legacy_module_reexports_turning_point_objects_by_identity():
    assert sequence_module.TurningPointError is turning_point.TurningPointError
    assert (
        sequence_module._extract_first_turning_point
        is turning_point._extract_first_turning_point
    )
    assert sequence_module._MASS_NOISE_RTOL is turning_point._MASS_NOISE_RTOL
    assert sequence_module._MASS_NOISE_ATOL is turning_point._MASS_NOISE_ATOL


def test_extracts_branch_through_first_resolved_mass_peak():
    curve = _curve([0.5, 1.0, 1.5, 1.4, 1.45, 1.2])
    profiles = [f"profile-{index}" for index in range(len(curve))]

    stable_curve, stable_profiles, maximum_mass = _extract_first_turning_point(
        curve, profiles
    )

    assert [point[0] for point in stable_curve] == [0.5, 1.0, 1.5]
    assert stable_profiles == profiles[:3]
    assert maximum_mass == pytest.approx(1.5)


def test_ignores_tiny_mass_noise_around_peak():
    curve = _curve([0.5, 1.0, 1.0000005, 1.0000002, 1.0000004, 0.999])
    profiles = list(range(len(curve)))

    stable_curve, stable_profiles, maximum_mass = _extract_first_turning_point(
        curve, profiles
    )

    assert len(stable_curve) == 3
    assert stable_profiles == profiles[:3]
    assert maximum_mass == pytest.approx(1.0000005)


@pytest.mark.parametrize(
    ("masses", "message"),
    [
        ([1.5, 1.4, 1.3], "lower-pressure boundary"),
        ([0.5, 1.0, 1.5], "not bracketed"),
        ([1.0, 1.0, 1.0], "no resolved increasing-mass branch"),
    ],
)
def test_rejects_boundary_or_unresolved_maxima(masses, message):
    curve = _curve(masses)

    with pytest.raises(TurningPointError, match=message):
        _extract_first_turning_point(curve, list(range(len(curve))))


def test_rejects_unordered_central_pressures():
    curve = _curve([0.5, 1.0, 0.9], pressures=[1.0, 3.0, 2.0])

    with pytest.raises(TurningPointError, match="strictly increasing central pressure"):
        _extract_first_turning_point(curve, [object(), object(), object()])


def test_rejects_nonfinite_data_and_profile_count_mismatch():
    nonfinite_curve = _curve([0.5, np.nan, 0.4])
    with pytest.raises(TurningPointError, match="non-finite"):
        _extract_first_turning_point(nonfinite_curve, [object()] * 3)

    curve = _curve([0.5, 1.0, 0.9])
    with pytest.raises(TurningPointError, match="profile count"):
        _extract_first_turning_point(curve, [object()] * 2)


@pytest.mark.parametrize("n_points", [True, np.bool_(False), 3, 4.5])
def test_solve_sequence_rejects_invalid_pressure_sample_count(n_points):
    with pytest.raises(ValueError, match="n_points must be an integer"):
        solve_sequence(lambda pressure: (pressure, 0.5), n_points=n_points)
