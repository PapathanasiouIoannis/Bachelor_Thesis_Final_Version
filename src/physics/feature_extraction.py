# src/physics/feature_extraction.py

"""
  Unified DRY module for extracting Machine Learning features from TOV curves.

Refactored:
  - CENTRALIZED EXTRACTION: Both Hadronic and Quark generation workers now use
    this single function, ensuring consistent interpolation.
  - NUMERICAL STABILITY FIX: Implemented strict boundary checks before computing
    the derivative `dR/dM`. It strictly verifies `max_m > m_step + small_step`
    before evaluating the PchipInterpolator to prevent massive extrapolation spikes.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.config import CONFIG


def extract_features(c_arr: np.ndarray, max_m: float) -> dict:
    """
    Extracts macroscopic features and topological slopes from a generated EoS curve.

    Parameters:
    - c_arr: NumPy array of the solved sequence.
             Expected columns -> 0: Mass, 1: Radius, 5: CS2_Central
    - max_m: Float, maximum stable mass of the sequence.

    Returns:
    - Dictionary containing 'cs2_at_14', 'r_14', and 'slopes' (dict).
      Returns None if the extraction fails due to numerical noise.
    """
    try:
        # sort sequence strictly by Mass to ensure monotonic interpolation
        c_arr = c_arr[c_arr[:, 0].argsort()]

        # build interpolators
        # PCHIP is used for M-R curves to preserve shape and avoid ringing (Runge's phenomenon).
        f_R = PchipInterpolator(c_arr[:, 0], c_arr[:, 1])
        # linear interpolation is sufficient for core sound speed, but we use PCHIP to strictly obey the instruction.
        f_CS2 = PchipInterpolator(c_arr[:, 0], c_arr[:, 5], extrapolate=True)

        features = {"cs2_at_14": np.nan, "r_14": np.nan, "slopes": {}}

        # only evaluate canonical features if the sequence actually reaches 1.4 M_sun
        if max_m >= 1.4:
            features["cs2_at_14"] = float(f_CS2(1.4))
            features["r_14"] = float(f_R(1.4))

            small_step = CONFIG["SMALL_STEP_M"]

            # evaluate topological slopes (dR/dM) at specific mass targets
            for m_step in [1.4, 1.6, 1.8, 2.0]:
                # CRITICAL STABILITY FIX:
                # do not evaluate the right-sided finite difference if it exceeds
                # the maximum mass limit. This prevents the spline from extrapolating
                # into unphysical phase space, which previously caused O(1e6) slope spikes.
                if max_m > (m_step + small_step):
                    r_minus = f_R(m_step)
                    r_plus = f_R(m_step + small_step)
                    slope = (r_plus - r_minus) / small_step
                    features["slopes"][m_step] = float(slope)
                else:
                    features["slopes"][m_step] = np.nan
        else:
            # valid low-mass sequence (e.g. M_max = 1.2 M_sun), fill with NaNs
            for m_step in [1.4, 1.6, 1.8, 2.0]:
                features["slopes"][m_step] = np.nan

        return features

    except Exception:
        # catch any interpolation domain errors and flag the sequence as invalid
        return None
