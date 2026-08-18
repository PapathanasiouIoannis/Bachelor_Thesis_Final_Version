import math

from src.config import CONFIG
from src.physics import solve_sequence as sequence_module
from src.physics.solve_sequence import (
    _apply_surface_density_correction,
    _tidal_lambda_from_y,
)
from src.physics.stellar import tidal


def test_legacy_module_reexports_pure_tidal_helper_by_identity():
    assert sequence_module._tidal_lambda_from_y is tidal._tidal_lambda_from_y


def test_surface_correction_facade_passes_its_conversion_snapshot(monkeypatch):
    calls = []

    def fake_correction(
        y_surface,
        radius,
        mass,
        surface_density,
        *,
        gravitational_conversion,
    ):
        calls.append(
            (
                y_surface,
                radius,
                mass,
                surface_density,
                gravitational_conversion,
            )
        )
        return 123.0

    monkeypatch.setattr(tidal, "apply_surface_density_correction", fake_correction)

    result = sequence_module._apply_surface_density_correction(0.8, 12.0, 1.4, 5.0)

    assert result == 123.0
    assert calls == [(0.8, 12.0, 1.4, 5.0, sequence_module._G_CONV)]


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
