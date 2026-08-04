# src/physics/worker_hadronic_gen.py

"""
  Generates Hadronic Star EoS using an Anchored Speed-of-Sound generator.

Refactored:
  - REMOVED STRATIFIED BUCKETS: Removed the mathematically impossible constraint
    of forcing uniformly distributed M_max up to 3.6 M_sun while restricting
    R_1.4 to <= 14.5 km.
  - NATURAL SAMPLING: Reverted to natural forward sampling. The generator rolls
    random microphysics, enforces the viability cut (M_max >= 2.08) and the
    observational radius bounds (9.5 <= R_1.4 <= 14.5), and accepts the resulting
    valid stars. This runs extremely fast and reflects true physical probability.
"""

import os
import time

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from src.config import CONFIG
from src.physics.feature_extraction import extract_features
from src.physics.get_eos_library import get_eos_library
from src.physics.solve_sequence import solve_sequence
from src.physics.verification import verify_eos_physical_validity
from src.utils.exceptions import AcausalEosError, ThermodynamicInstabilityError, CrustStitchingError
import logging

logger = logging.getLogger(__name__)


def _eval_crust(crusts: dict, p_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P_c1 = CONFIG["P_C1"]
    P_c2 = CONFIG["P_C2"]
    P_c3 = CONFIG["P_C3"]

    conds = [
        p_arr > P_c1,
        (p_arr <= P_c1) & (p_arr > P_c2),
        (p_arr <= P_c2) & (p_arr > P_c3),
        p_arr <= P_c3,
    ]
    eps_funcs = [crusts["c1"][0], crusts["c2"][0], crusts["c3"][0], crusts["c4"][0]]
    dedp_funcs = [
        crusts["c1"][1],
        crusts["c2"][1],
        crusts["c3"][1],
        crusts["c4"][1],
    ]
    return np.piecewise(p_arr, conds, eps_funcs), np.piecewise(
        p_arr, conds, dedp_funcs
    )


def _eps_tolerance(e_core: float, e_crust: float) -> float:
    eps_abs_tol = CONFIG["CRUST_CORE_EPS_ABS_TOL"]
    eps_rel_tol = CONFIG["CRUST_CORE_EPS_REL_TOL"]
    return max(eps_abs_tol, eps_rel_tol * max(abs(e_core), abs(e_crust), 1.0))


def _anchor_gap(crusts: dict, core_anchor: tuple, p: float, eps_shift: float = 0.0) -> tuple[float, float, float, float]:
    fA_e, fA_de = core_anchor
    p = float(p)
    e_core = float(fA_e(p)) - float(eps_shift)
    de_core = float(fA_de(p))
    e_crust, _ = _eval_crust(crusts, np.array([p], dtype=float))
    e_crust = float(e_crust[0])

    if np.isnan(e_core) or np.isnan(de_core) or np.isinf(e_core) or np.isinf(de_core):
        raise ValueError(f"Anchor model produced non-finite values at P={p}.")
    if e_core <= 0.0 or de_core <= 0.0:
        raise ValueError(f"Anchor model is non-physical at P={p}.")
    if np.isnan(e_crust) or np.isinf(e_crust) or e_crust <= 0.0:
        raise ValueError(f"Crust model produced non-physical density at P={p}.")

    return e_core - e_crust, e_core, de_core, e_crust


def resolve_density_shifted_transition(crusts: dict, core_anchor: tuple, P_trans_default: float) -> tuple[float, float]:
    """Return a fixed transition pressure and the core epsilon shift needed for continuity.

    The shift is additive in energy-density space only:
        eps_core_shifted(P) = eps_core_raw(P) - eps_shift

    d eps / dP is unchanged, so the baseline sound-speed structure is preserved
    while the crust/core density jump is removed at the anchor pressure.
    """

    fA_e, fA_de = core_anchor
    p_trans = float(P_trans_default)
    try:
        raw_core_eps = float(fA_e(p_trans))
        core_deps_dp = float(fA_de(p_trans))
        crust_eps, _ = _eval_crust(crusts, np.array([p_trans], dtype=float))
        crust_eps = float(crust_eps[0])
    except Exception as e:
        raise CrustStitchingError(
            p_trans=p_trans,
            message=f"Anchor model invalid at transition pressure: {e}",
        ) from e

    if (
        np.isnan(raw_core_eps)
        or np.isnan(core_deps_dp)
        or np.isnan(crust_eps)
        or np.isinf(raw_core_eps)
        or np.isinf(core_deps_dp)
        or np.isinf(crust_eps)
    ):
        raise CrustStitchingError(
            p_trans=p_trans,
            message="Anchor or crust model produced non-finite transition values.",
        )
    if raw_core_eps <= 0.0 or crust_eps <= 0.0:
        raise CrustStitchingError(
            p_trans=p_trans,
            message="Anchor or crust model produced non-positive transition density.",
        )
    if core_deps_dp <= 0.0:
        raise CrustStitchingError(
            p_trans=p_trans,
            message="Anchor model is thermodynamically unstable at transition pressure.",
        )

    eps_shift = raw_core_eps - crust_eps
    stitch_gap, shifted_core_eps, _, shifted_crust_eps = _anchor_gap(
        crusts, core_anchor, p_trans, eps_shift=eps_shift
    )
    if abs(stitch_gap) > _eps_tolerance(shifted_core_eps, shifted_crust_eps):
        raise CrustStitchingError(
            p_trans=p_trans,
            message=(
                f"Density-shift continuity failed: residual gap {stitch_gap:.6e} "
                f"after applying eps_shift={eps_shift:.6e}."
            ),
        )

    return p_trans, eps_shift


def resolve_crust_core_transition(crusts: dict, core_anchor: tuple, P_trans_default: float) -> float:
    try:
        default_gap, default_e_core, _, default_e_crust = _anchor_gap(
            crusts, core_anchor, P_trans_default
        )
    except Exception as e:
        raise CrustStitchingError(
            p_trans=P_trans_default,
            message=f"Anchor model invalid at default transition pressure: {e}",
        ) from e

    default_tol = _eps_tolerance(default_e_core, default_e_crust)
    if abs(default_gap) <= default_tol:
        return float(P_trans_default)

    from scipy.optimize import root_scalar

    bracket = None
    prev_p = P_trans_default
    prev_gap = default_gap

    for p_test in np.linspace(P_trans_default, CONFIG["DYNAMIC_SEARCH_MAX"], CONFIG["DYNAMIC_SEARCH_POINTS"])[1:]:
        try:
            gap, e_core, _, e_crust = _anchor_gap(crusts, core_anchor, p_test)
        except Exception:
            continue

        if abs(gap) <= _eps_tolerance(e_core, e_crust):
            return float(p_test)
        if np.sign(prev_gap) != np.sign(gap):
            bracket = (prev_p, float(p_test))
            break

        prev_p = float(p_test)
        prev_gap = gap

    if bracket is None:
        raise CrustStitchingError(
            p_trans=P_trans_default,
            message=(
                "No crust/core density-continuity root found. "
                f"Default gap={default_gap:.6e} exceeds tolerance={default_tol:.6e}."
            ),
        )

    try:
        sol = root_scalar(lambda p: _anchor_gap(crusts, core_anchor, p)[0], bracket=bracket)
        if not sol.converged:
            raise RuntimeError("root solver did not converge")
        P_trans_actual = float(sol.root)
    except Exception as e:
        raise CrustStitchingError(
            p_trans=P_trans_default,
            message=f"Could not solve crust/core continuity root: {e}",
        ) from e

    try:
        stitch_gap, anchor_e_trans, anchor_de_trans, crust_e_trans = _anchor_gap(
            crusts, core_anchor, P_trans_actual
        )
        stitch_tol = _eps_tolerance(anchor_e_trans, crust_e_trans)
        if anchor_de_trans <= 0:
            raise ValueError(f"Anchor model is thermodynamically unstable at P_trans_actual ({P_trans_actual}).")
        if abs(stitch_gap) > stitch_tol:
            raise ValueError(
                f"Anchor/crust density discontinuity {stitch_gap:.6e} exceeds tolerance {stitch_tol:.6e}."
            )
    except Exception as e:
        raise CrustStitchingError(
            p_trans=P_trans_actual,
            message=f"Explicitly dropping anchor model: {e}",
        ) from e

    return P_trans_actual


def _density_shiftable_anchor_names(core_lib: dict, crust_funcs: dict, worker_logger) -> list[str]:
    usable = []
    excluded = []
    shifts = []
    for name, core_anchor in core_lib.items():
        try:
            _, eps_shift = resolve_density_shifted_transition(crust_funcs, core_anchor, CONFIG["P_TRANS_DEFAULT"])
            usable.append(name)
            shifts.append(eps_shift)
        except Exception as e:
            excluded.append((name, str(e)))

    worker_logger.info(
        f"Density-shift continuity precheck: {len(usable)}/{len(core_lib)} hadronic anchors usable."
    )
    if shifts:
        worker_logger.info(
            f"Core density shifts at P_trans: min={min(shifts):.3e}, "
            f"median={float(np.median(shifts)):.3e}, max={max(shifts):.3e} MeV/fm^3."
        )
    if excluded:
        excluded_names = ", ".join(name for name, _ in excluded)
        worker_logger.info(f"Excluding anchors that cannot be density-shifted safely: {excluded_names}.")
    if not usable:
        raise RuntimeError("No hadronic anchor EoS can satisfy density-shifted crust/core continuity.")
    return usable


def build_anchored_sos_spline(crusts: dict, core_anchor: tuple, P_trans_default: float) -> tuple:
    """
    Constructs a C^1 continuous EoS by keeping the analytic crust, anchoring the
    low-density core to a dynamically chosen nuclear baseline, and generating a
    smooth random Speed-of-Sound (c_s^2) spline for the deep core.
    """
    fA_e, fA_de = core_anchor
    P_trans_actual, eps_shift = resolve_density_shifted_transition(crusts, core_anchor, P_trans_default)

    A = np.random.uniform(*CONFIG["GAUSSIAN_AMP_RANGE"])
    eps_0 = np.random.uniform(*CONFIG["PERTURB_LOC_RANGE"])
    sigma = np.random.uniform(*CONFIG["GAUSSIAN_SIGMA_RANGE"])

    p_grid_ext = np.linspace(P_trans_actual, CONFIG["P_GRID_MAX"], CONFIG["P_GRID_POINTS"])
    eps_ext = fA_e(p_grid_ext) - eps_shift
    deps_ext = fA_de(p_grid_ext)

    if np.any(~np.isfinite(eps_ext)) or np.any(~np.isfinite(deps_ext)):
        raise CrustStitchingError(p_trans=P_trans_actual, message="Shifted anchor produced non-finite core grid values.")
    if np.any(eps_ext <= 0.0):
        raise CrustStitchingError(p_trans=P_trans_actual, message="Shifted anchor produced non-positive core densities.")
    if np.any(np.diff(eps_ext) <= 0.0):
        raise ThermodynamicInstabilityError(
            deps=float(np.min(np.diff(eps_ext))),
            dp=float(np.mean(np.diff(p_grid_ext))),
            message="Shifted anchor energy-density grid is not strictly increasing.",
        )

    cs2_ext = np.zeros_like(deps_ext)
    v_idx = deps_ext != 0
    cs2_ext[v_idx] = 1.0 / deps_ext[v_idx]

    bump = A * np.exp(-0.5 * ((eps_ext - eps_0) / sigma)**2)
    cs2_pert = cs2_ext + bump

    # guillotine logic
    viol_pert = np.where(cs2_pert > 1.0)[0]
    if len(viol_pert) > 0:
        s_idx = viol_pert[0]
        if s_idx == 0:
            raise CrustStitchingError(p_trans=P_trans_actual, message="Causality violated immediately at anchor.")
        eps_sliced = eps_ext[:s_idx]
        cs2_sliced = cs2_pert[:s_idx]
    else:
        eps_sliced = eps_ext
        cs2_sliced = cs2_pert

    # avoid negative or exactly zero cs2 to ensure cumulative integral strictly increases
    cs2_sliced = np.clip(cs2_sliced, CONFIG["THERMO_FLOOR"], None)

    if len(eps_sliced) < 4:
        raise CrustStitchingError(p_trans=P_trans_actual, message="Shifted core grid is too short after causality slicing.")

    # re-integrate P = int cs2 d_eps
    P_pert = P_trans_actual + cumulative_trapezoid(cs2_sliced, eps_sliced, initial=0)
    if np.any(~np.isfinite(P_pert)) or np.any(np.diff(P_pert) <= 0.0):
        raise CrustStitchingError(p_trans=P_trans_actual, message="Reconstructed pressure grid is not strictly increasing.")

    P_max_causal = P_pert[-1]

    return P_trans_actual, P_pert, eps_sliced, cs2_sliced, A, eps_0, sigma, P_max_causal


def worker_hadronic_gen(n_curves_to_gen: int, seed_offset: int, batch_idx: int) -> pd.DataFrame:
    """
    Worker process for generating unbiased, dynamically anchored Hadronic EoS curves.
    """
    # entropy Injection: combine seed_offset, batch_idx, pid, and time
    seed_val = (seed_offset + batch_idx + os.getpid() + int(time.time() * 1e6)) % (
        2**32
    )
    np.random.seed(seed_val)

    core_lib, crust_funcs = get_eos_library()
    from src.utils.logger import get_logger
    logger = get_logger("HADRONIC")
    model_names = _density_shiftable_anchor_names(core_lib, crust_funcs, logger)
    logger.info(
        f"Operating in density-shifted continuity mode: Mixing {len(model_names)} EoS models."
    )

    valid_data = []
    curves_found = 0
    attempts = 0

    m_min_save = CONFIG["M_MIN_SAVE"]
    m_max_lower = CONFIG["M_MAX_LOWER_BOUND"]
    m_max_upper = CONFIG["H_M_MAX_UPPER"]

    max_attempts = n_curves_to_gen * CONFIG["ATTEMPT_MULTIPLIER"]

    while curves_found < n_curves_to_gen and attempts < max_attempts:
        attempts += 1

        # 1. Select dynamic anchor
        anchor_name = np.random.choice(model_names)
        core_anchor = core_lib[anchor_name]

        P_trans = CONFIG["P_TRANS_DEFAULT"]

        try:
            P_trans_actual, P_pert, eps_sliced, cs2_sliced, pert_A, pert_eps0, pert_sigma, P_max_causal = build_anchored_sos_spline(
                crust_funcs, core_anchor, P_trans
            )

            if P_max_causal <= P_trans_actual:
                continue

            eps_spline = PchipInterpolator(P_pert, eps_sliced, extrapolate=True)
            cs2_spline = PchipInterpolator(P_pert, cs2_sliced, extrapolate=True)
        except (AssertionError, CrustStitchingError, ThermodynamicInstabilityError) as e:
            logger.warning(f"Rejected EoS '{anchor_name}' during crust stitching: {e}")
            continue
        except ValueError as e:
            logger.warning(f"Rejected EoS '{anchor_name}' due to interpolation error: {e}")
            continue

        def eos_callable(p):
            p = float(p)
            if p > P_trans_actual:
                if p > P_max_causal:
                    return -1.0, -1.0
                return float(eps_spline(p)), float(cs2_spline(p))
            else:
                if p > CONFIG["P_C1"]:
                    e = float(crust_funcs["c1"][0](p))
                    de = float(crust_funcs["c1"][1](p))
                elif p > CONFIG["P_C2"]:
                    e = float(crust_funcs["c2"][0](p))
                    de = float(crust_funcs["c2"][1](p))
                elif p > CONFIG["P_C3"]:
                    e = float(crust_funcs["c3"][0](p))
                    de = float(crust_funcs["c3"][1](p))
                else:
                    e = float(crust_funcs["c4"][0](p))
                    de = float(crust_funcs["c4"][1](p))
                return e, float(1.0/de if de > 0 else 0.0)

        eos_callable.eps_surf = 0.0

        # 3. Solve structure
        curve, dense_profiles, max_m = solve_sequence(
            eos_callable,
            is_quark=False,
            p_max_causal=P_max_causal,
            rtol=CONFIG["TOV_RTOL"],
            atol=CONFIG["TOV_ATOL"]
        )

        c_arr = np.array(curve)
        if len(c_arr) == 0 or c_arr[0, 0] > CONFIG["BH_LIMIT"]:
            if len(c_arr) == 0:
                logger.warning(f"Rejected EoS '{anchor_name}': Sequence returned empty profiles.")
            continue

        # post-Integration Physical Verification
        # ensures strict causality (cs2 <= 1) and thermodynamic stability (dP/dEps > 0)
        try:
            verify_eos_physical_validity(c_arr)
        except (AcausalEosError, ThermodynamicInstabilityError) as e:
            logger.warning(f"Rejected EoS '{anchor_name}': {e}")
            continue

        # 5. Extract and validate
        features = extract_features(c_arr, max_m)
        if features is None:
            continue

        # Observational constraints (matching quark worker filters)
        if max_m < m_max_lower or max_m > m_max_upper:
            continue
        if features["r_14"] < CONFIG["OBS_R14_HADRONIC_MIN"] or features["r_14"] > CONFIG["OBS_R14_HADRONIC_MAX"]:
            continue

        # 6. Save Data
        curves_found += 1
        curve_id = f"H_{batch_idx}_{attempts}"

        for pt in curve:
            m_val = pt[0]
            if m_val >= m_min_save and m_val <= max_m:
                valid_data.append(
                    [
                        m_val,  # mass
                        pt[1],  # radius
                        pt[2],  # lambda
                        0,  # label (0 = Hadronic)
                        curve_id,  # group ID
                        pt[3],  # P_Central
                        pt[4],  # eps_Central
                        0.0,  # eps_Surface
                        pt[5],  # CS2_Central
                        features["cs2_at_14"],
                        features["r_14"],
                        features["slopes"].get(1.4, 0.0),
                        features["slopes"].get(1.6, 0.0),
                        features["slopes"].get(1.8, 0.0),
                        features["slopes"].get(2.0, 0.0),
                        0.0,
                        0.0,
                        0.0,  # quark Params (Bag_B, Gap_Delta, Mass_Strange)
                        seed_val,  # generation_Seed
                        pert_A, pert_eps0, pert_sigma,
                        anchor_name  # baseline_Name
                    ]
                )

    # convert to DataFrame
    if curves_found < n_curves_to_gen:
        logger.warning(
            f"Hadronic worker generated {curves_found}/{n_curves_to_gen} requested curves "
            f"after {attempts} attempts."
        )

    cols = CONFIG["COLUMN_SCHEMA"]
    df = pd.DataFrame(valid_data, columns=cols)

    # downcast all numerical columns to float32 to prevent Parquet memory bloat
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")


    return df
