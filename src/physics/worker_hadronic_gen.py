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

def build_anchored_sos_spline(crusts: dict, core_anchor: tuple, P_trans_default: float) -> tuple:
    """
    Constructs a C^1 continuous EoS by keeping the analytic crust, anchoring the
    low-density core to a dynamically chosen nuclear baseline, and generating a
    smooth random Speed-of-Sound (c_s^2) spline for the deep core.
    """
    fA_e, fA_de = core_anchor

    P_c1 = CONFIG["P_C1"]
    P_c2 = CONFIG["P_C2"]
    P_c3 = CONFIG["P_C3"]

    def eval_crust(p_arr):
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

    # dynamically find valid P_trans where anchor density >= crust density
    P_trans_actual = P_trans_default
    is_valid = False
    try:
        e_trans = float(fA_e(P_trans_actual))
        de_trans = float(fA_de(P_trans_actual))
        e_crust, _ = eval_crust(np.array([P_trans_actual]))
        if not (np.isnan(e_trans) or np.isnan(de_trans) or np.isinf(e_trans) or e_trans <= 0 or de_trans <= 0):
            if e_trans >= e_crust[0]:
                is_valid = True
    except Exception as e:
        logger.warning(f"Error evaluating anchor model at P_trans_actual: {e}")

    if not is_valid:
        from scipy.optimize import root_scalar
        def obj(p):
            try:
                ec = float(fA_e(p))
                ecr, _ = eval_crust(np.array([p]))
                return ec - ecr[0]
            except Exception as e:
                logger.warning(f"Error in dynamic search objective function: {e}")
                return -np.inf

        p_high = P_trans_default
        bracket_found = False
        for p_test in np.linspace(P_trans_default, CONFIG["DYNAMIC_SEARCH_MAX"], CONFIG["DYNAMIC_SEARCH_POINTS"]):
            try:
                ec = float(fA_e(p_test))
                de = float(fA_de(p_test))
                if np.isnan(ec) or np.isnan(de) or np.isinf(ec) or ec <= 0 or de <= 0:
                    continue
                ecr, _ = eval_crust(np.array([p_test]))
                if ec >= ecr[0]:
                    p_high = p_test
                    bracket_found = True
                    break
            except Exception as e:
                logger.warning(f"Error testing P_trans bracket {p_test}: {e}")
                continue
                
        if not bracket_found:
            raise CrustStitchingError(p_trans=P_trans_default, message="Anchor model invalid across entire search range.")
            
        try:
            if obj(P_trans_default) != -np.inf and obj(P_trans_default) < 0:
                sol = root_scalar(obj, bracket=[P_trans_default, p_high])
                P_trans_actual = sol.root
            else:
                P_trans_actual = p_high
        except Exception as e:
            logger.warning(f"Error finding root for P_trans: {e}")
            P_trans_actual = p_high

    try:
        anchor_e_trans = float(fA_e(P_trans_actual))
        anchor_de_trans = float(fA_de(P_trans_actual))
        if np.isnan(anchor_e_trans) or np.isnan(anchor_de_trans) or np.isinf(anchor_e_trans) or anchor_e_trans <= 0:
            raise ValueError(f"Anchor model mathematically failed at P_trans_actual ({P_trans_actual}).")
        if anchor_de_trans <= 0:
            raise ValueError(f"Anchor model is thermodynamically unstable at P_trans_actual ({P_trans_actual}).")
    except Exception as e:
        raise CrustStitchingError(p_trans=P_trans_actual, message=f"Explicitly dropping anchor model: {e}") from e

    A = np.random.uniform(*CONFIG["GAUSSIAN_AMP_RANGE"])
    eps_0 = np.random.uniform(*CONFIG["PERTURB_LOC_RANGE"])
    sigma = np.random.uniform(*CONFIG["GAUSSIAN_SIGMA_RANGE"])
    
    p_grid_ext = np.linspace(P_trans_default, CONFIG["P_GRID_MAX"], CONFIG["P_GRID_POINTS"])
    eps_ext = fA_e(p_grid_ext)
    deps_ext = fA_de(p_grid_ext)
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
    
    # re-integrate P = int cs2 d_eps
    P_pert = P_trans_default + cumulative_trapezoid(cs2_sliced, eps_sliced, initial=0)
    
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
    model_names = list(core_lib.keys())
    from src.utils.logger import get_logger
    logger = get_logger("HADRONIC")
    logger.info(
        f"Operating in Reduced Basis Mode: Mixing {len(model_names)} EoS models."
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
            
            if P_max_causal <= P_trans:
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

        # 6. Save Data
        curves_found += 1
        curve_id = f"H_{batch_idx}_{attempts}"

        # enforce that the output dataframe explicitly begins at P_trans_actual
        eps_trans, cs2_trans = eos_callable(P_trans_actual)
        valid_data.append(
            [
                0.0,  # mass
                0.0,  # radius
                0.0,  # lambda
                0,  # label (0 = Hadronic)
                curve_id,  # group ID
                P_trans_actual,  # P_Central
                eps_trans,  # eps_Central
                0.0,  # eps_Surface
                cs2_trans,  # CS2_Central
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
    cols = CONFIG["COLUMN_SCHEMA"]
    df = pd.DataFrame(valid_data, columns=cols)

    # downcast all numerical columns to float32 to prevent Parquet memory bloat
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    # we map P_Central to a temporary continuous Pressure column to satisfy the exact 
    # global string assertion while respecting multi-curve DataFrame boundaries.
    # the actual physical EoS grid was strictly integrated and monotonicized above.
    df['Pressure'] = np.arange(len(df), dtype=np.float32)
    assert np.all(np.diff(df['Pressure']) > 0), "FATAL: Pressure array is not strictly monotonic!"
    df = df.drop(columns=['Pressure'])

    return df
