import numpy as np

from src.config import CONFIG as CONSTANTS
from src.physics.solve_sequence import solve_sequence
from src.utils.logger import get_logger

logger = get_logger("TEST_TOV")

def test_tov_n1_polytrope():
    """
    Tests the TOV solver against the known analytical solution for an N=1
    Newtonian polytrope (P = K * eps^2) in the weak-field limit.
    """
    K = 1e-3  # polytropic constant

    A_CONV = CONSTANTS["A_CONV"]
    G_CONV = CONSTANTS["G_CONV"]

    # analytical Newtonian Radius for N=1 Polytrope
    expected_R = np.pi * np.sqrt(2.0 * K / (A_CONV * G_CONV))

    def eos_n1(P):
        eps = np.sqrt(max(P, 0) / K)
        cs2 = 2.0 * K * eps
        return float(eps), float(cs2)

    eos_n1.eps_surf = 0.0

    curve, _, _ = solve_sequence(eos_n1, is_quark=False)

    # sort by mass and take the smallest mass point that passed the viability cuts (M > 0.05)
    # the smallest mass point will have the weakest relativistic effects, thus matching
    # the Newtonian analytical solution most accurately.
    curve.sort(key=lambda x: x[0])
    assert len(curve) > 0, "No valid stars generated"

    lowest_m_pt = curve[0]

    M_num = lowest_m_pt[0]
    R_num = lowest_m_pt[1]
    eps_c_num = lowest_m_pt[4]

    # analytical Newtonian Mass for this specific central density
    expected_M = np.pi * G_CONV * eps_c_num * (2.0 * K / (A_CONV * G_CONV)) ** 1.5

    # we tolerate 1% error because even at M=0.05, there are slight relativistic corrections
    # to the Newtonian analytical limits.
    assert abs(R_num - expected_R) / expected_R < 0.01, (
        f"Radius mismatch: {R_num:.3f} vs {expected_R:.3f}"
    )
    assert abs(M_num - expected_M) / expected_M < 0.01, (
        f"Mass mismatch: {M_num:.5f} vs {expected_M:.5f}"
    )

    logger.info(f"N=1 Polytrope TOV Test Passed! R_num={R_num:.3f}, M_num={M_num:.6f}")


if __name__ == "__main__":
    test_tov_n1_polytrope()
