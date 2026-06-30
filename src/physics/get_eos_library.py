# src/physics/get_eos_library.py

"""
  Provides a library of analytic Equation of State (EoS) parameterizations.
  - Core Models: Based on Read et al. (2009) piecewise fits.
  - Crust Model: Based on Douchin & Haensel (SLy) parameterization.

Refactored:
  - FIXED SYMPY BOTTLENECK: Implemented a lazy-loaded global cache. `lambdify`
    is now strictly executed ONCE per process life-cycle. When parallel workers
    import this module, they retrieve the compiled, lightning-fast NumPy C-funcs
    without triggering the extremely expensive SymPy recompilation.
"""

from sympy import Symbol, exp, lambdify
from sympy import log as sym_log

# global Cache for compiled functions
_CORE_CACHE = None
_CRUST_CACHE = None


def get_eos_library() -> tuple:
    """
    Returns dictionary of Analytic Hadronic Core Models + 4-Layer Crust.
    Compiles SymPy expressions to NumPy functions on the first call and
    caches them for all subsequent calls.

    Returns:
    - core_funcs: Dict mapping ModelName -> (func_epsilon(P), func_dEps_dP(P))
    - crust_funcs: Dict mapping 'c1'..'c4' -> (func_epsilon(P), func_dEps_dP(P))
    """
    global _CORE_CACHE, _CRUST_CACHE

    # return cached compiled functions if they already exist (Fixes Bottleneck)
    if _CORE_CACHE is not None and _CRUST_CACHE is not None:
        return _CORE_CACHE, _CRUST_CACHE

    p = Symbol("p")

    # ==========================================
    # 1. CORE MODELS (High Density)
    # ==========================================
    # analytic fits for Energy Density epsilon(p)[MeV/fm^3]
    core_exprs = {
         "MDI-1": 4.1844 * p**0.81449 + 95.00135 * p**0.31736,
         "MDI-2": 5.97365 * p**0.77374 + 89.24 * p**0.30993,
         "MDI-3": 15.55 * p**0.666 + 76.71 * p**0.247,
         "MDI-4": 25.99587 * p**0.61209 + 65.62193 * p**0.15512,
         "NLD": 119.05736
         + 304.80445 * (1 - exp(-p / 48.61465))
         + 33722.34448 * (1 - exp(-p / 17499.47411)),
         "HHJ-1": 1.78429 * p**0.93761 + 106.93652 * p**0.31715,
         "HHJ-2": 1.18961 * p**0.96539 + 108.40302 * p**0.31264,
         "Ska": 0.53928 * p**1.01394 + 94.31452 * p**0.35135,
         "SkI4": 4.75668 * p**0.76537 + 105.722 * p**0.2745,
        "HLPS-2": 161.553
        + 172.858 * (1 - exp(-p / 22.8644))
        + 2777.75 * (1 - exp(-p / 1909.97)),
        "HLPS-3": 81.5682
        + 131.811 * (1 - exp(-p / 4.41577))
        + 924.143 * (1 - exp(-p / 523.736)),
         "SCVBB": 0.371414 * p**1.08004 + 109.258 * p**0.351019,
         "WFF-1": 0.00127717 * p**1.69617 + 135.233 * p**0.331471,
         "WFF-2": 0.00244523 * p**1.62692 + 122.076 * p**0.340401,
         "PS": 1.69483 + 9805.95 * (1 - exp(-p * 0.000193624))
               + 212.072 * (1 - exp(-p * 0.401508)),
         "W": 0.261822 * p**1.16851 + 92.4893 * p**0.307728,
         "BGP": 0.0112475 * p**1.59689 + 102.302 * p**0.335526,
         "BL-1": 0.488686 * p**1.01457 + 102.26 * p**0.355095,
         "BL-2": 1.34241 * p**0.910079 + 100.756 * p**0.354129,
         "DH": 39.5021 * p**0.541485 + 96.0528 * p**0.00401285,
         "APR-1": 0.000719964 * p**1.85898 + 108.975 * p**0.340074,
    }

    # ==========================================
    # 2. MULTI-LAYER CRUST (Low Density)
    # ==========================================
    # constants for Crust 4 (Log-Polynomial Envelope)
    c = [31.93753, 10.82611, 1.29312, 0.08014, 0.00242, 0.000028]
    logP = sym_log(p, 10)

    # crust 1: High Density
    crust1 = (
        0.00873
        + 103.17338 * (1 - exp(-p / 0.38527))
        + 7.34979 * (1 - exp(-p / 0.01211))
    )

    # crust 2: Mid Density
    crust2 = (
        0.00015
        + 0.00203 * (1 - exp(-p * 344827.5))
        + 0.10851 * (1 - exp(-p * 7692.3076))
    )

    # crust 3: Low Density
    crust3 = 0.0000051 * (1 - exp(-p * 0.2373e10)) + 0.00014 * (1 - exp(-p * 0.4020e8))

    # crust 4: Envelope
    crust4 = 10 ** (
        c[0]
        + c[1] * logP
        + c[2] * logP**2
        + c[3] * logP**3
        + c[4] * logP**4
        + c[5] * logP**5
    )

    # ==========================================
    # 3. LAMBDIFICATION (Symbolic -> Fast Numpy)
    # ==========================================
    # core_funcs: Returns tuple (eps_val, dedp_val)
    core_funcs = {
        k: (lambdify(p, e, "numpy"), lambdify(p, e.diff(p), "numpy"))
        for k, e in core_exprs.items()
    }

    # crust_funcs: Returns tuple (eps_val, dedp_val)
    crust_funcs = {
        "c1": (lambdify(p, crust1, "numpy"), lambdify(p, crust1.diff(p), "numpy")),
        "c2": (lambdify(p, crust2, "numpy"), lambdify(p, crust2.diff(p), "numpy")),
        "c3": (lambdify(p, crust3, "numpy"), lambdify(p, crust3.diff(p), "numpy")),
        "c4": (lambdify(p, crust4, "numpy"), lambdify(p, crust4.diff(p), "numpy")),
    }

    # save to global cache
    _CORE_CACHE = core_funcs
    _CRUST_CACHE = crust_funcs

    return _CORE_CACHE, _CRUST_CACHE
