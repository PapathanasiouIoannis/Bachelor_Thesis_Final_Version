import math

from src.config import CONFIG
from src.physics.solve_sequence import _apply_surface_density_correction, _tidal_lambda_from_y


def test_surface_density_correction_uses_r_cubed_and_changes_lambda():
    y_raw = 0.8
    radius_km = 12.0
    mass_msun = 1.4
    eps_surf = 250.0
    compactness = CONFIG["A_CONV"] * mass_msun / radius_km

    corrected_y = _apply_surface_density_correction(y_raw, radius_km, mass_msun, eps_surf)
    expected = y_raw - CONFIG["G_CONV"] * radius_km**3 * eps_surf / mass_msun
    assert math.isclose(corrected_y, expected, rel_tol=1e-12, abs_tol=1e-12)

    lambda_raw = _tidal_lambda_from_y(compactness, y_raw)
    lambda_corrected = _tidal_lambda_from_y(compactness, corrected_y)
    assert lambda_raw is not None
    assert lambda_corrected is not None
    assert math.isfinite(lambda_raw)
    assert math.isfinite(lambda_corrected)
    assert not math.isclose(lambda_raw, lambda_corrected, rel_tol=1e-8, abs_tol=0.0)
