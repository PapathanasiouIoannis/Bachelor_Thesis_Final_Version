import numba
import numpy as np


from src.utils.exceptions import AcausalEosError, ThermodynamicInstabilityError

@numba.njit
def _check_causality(cs2_c: np.ndarray, eps_c: np.ndarray):
    """Numba-compiled helper for strict causality."""
    for i in range(len(cs2_c)):
        if cs2_c[i] > 1.0 or cs2_c[i] < 0.0:
            return i, cs2_c[i], eps_c[i]
    return -1, 0.0, 0.0

@numba.njit
def _check_thermo(pc: np.ndarray, eps_c: np.ndarray):
    """Numba-compiled helper for thermodynamic stability."""
    for i in range(1, len(pc)):
        dp = pc[i] - pc[i - 1]
        deps = eps_c[i] - eps_c[i - 1]

        if deps <= 0 or dp / deps <= 0:
            return i, deps, dp
    return -1, 0.0, 0.0

def verify_eos_physical_validity(c_arr: np.ndarray) -> bool:
    """
    Strict post-integration physical verification function.
    Evaluates the generated EoS trace (central pressure and energy density)
    to ensure strict causality and thermodynamic stability.

    c_arr: numpy array of shape (N, 7) where columns are:
           [M, R, Lam, pc, eps_c, cs2_c, eps_surf]

    Returns:
        bool: True if the EoS is physically viable, raises Exception otherwise.
    """
    if len(c_arr) < 2:
        # we need at least 2 points for derivative, but we'll just return False 
        # or maybe we can raise something else, but let's keep it returning False
        # as per original behavior when its just invalid shape.
        return False

    pc = c_arr[:, 3]
    eps_c = c_arr[:, 4]
    cs2_c = c_arr[:, 5]

    # 1. Strict Causality: cs2 <= 1 at every integration step
    idx, cs2_val, eps_val = _check_causality(cs2_c, eps_c)
    if idx != -1:
        raise AcausalEosError(cs2_value=cs2_val, eps_c=eps_val)

    # 2. Strict Thermodynamic Stability: dP/dEps > 0 strictly everywhere
    idx, deps_val, dp_val = _check_thermo(pc, eps_c)
    if idx != -1:
        raise ThermodynamicInstabilityError(deps=deps_val, dp=dp_val)

    return True
