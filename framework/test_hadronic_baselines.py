import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath('.'))

from src.config import CONFIG
from src.physics.get_eos_library import get_eos_library
from src.physics.tov_rhs import tov_rhs
from src.utils.exceptions import TovConvergenceError

# custom TOV solver tailored for the Causality Guillotine
def bounded_solve_sequence(eos_callable, p_max_causal):
    r_min = CONFIG["TOV_R_MIN"]
    _R_MAX = CONFIG["TOV_R_MAX"]
    _G_CONV = CONFIG["G_CONV"]
    _A_CONV = CONFIG["A_CONV"]
    _BUCHDAHL_LIMIT = CONFIG["BUCHDAHL_LIMIT"]

    def _surface_event(t, y, *args):
        return y[1] - CONFIG["SURFACE_PRESSURE_EVENT_CUTOFF"]
    _surface_event.terminal = True
    _surface_event.direction = -1

    # strict bounds specified by the user
    pressures = np.geomspace(0.10, p_max_causal, 150)
    curve_data = []

    for pc in pressures:
        eps_init, cs2_init = eos_callable(pc)
        if np.isnan(eps_init) or eps_init < 0:
            continue

        m_init = (r_min**3) * eps_init * (_G_CONV / 3.0)
        y0 = [m_init, pc, 2.0]

        try:
            sol = solve_ivp(
                fun=tov_rhs,
                t_span=(r_min, _R_MAX),
                y0=y0,
                args=(eos_callable,),
                events=_surface_event,
                method="RK45",
                rtol=1e-8,
                atol=1e-10,
            )

            if sol.status == 1 and len(sol.t_events[0]) > 0:
                R = sol.t_events[0][0]
                M = sol.y_events[0][0][0]
                if R < 3.0 or M < 0.05:
                    continue
                
                C = (M * _A_CONV) / R
                if C >= _BUCHDAHL_LIMIT:
                    continue
                    
                curve_data.append([M, R])
        except Exception:
            continue

    if not curve_data:
        return []

    curve_arr = np.array(curve_data)
    mass_array = curve_arr[:, 0]
    radius_array = curve_arr[:, 1]
    
    max_mass_idx = int(np.argmax(mass_array))
    
    mass_stable = mass_array[: max_mass_idx + 1]
    radius_stable = radius_array[: max_mass_idx + 1]
    
    return list(zip(mass_stable, radius_stable))



def main():
    core_lib, crust_funcs = get_eos_library()
    P_trans_default = CONFIG["P_TRANS_DEFAULT"]

    fig_cs2, ax_cs2 = plt.subplots(figsize=(8, 6))
    fig_mr, ax_mr = plt.subplots(figsize=(8, 6))

    for name, core_anchor in core_lib.items():
        fA_e, fA_de = core_anchor
        
        # 1. The Causality Slicer
        p_grid = np.linspace(P_trans_default, 1200.0, 5000)
        eps_grid = fA_e(p_grid)
        deps_dp = fA_de(p_grid)
        
        cs2_grid = np.zeros_like(deps_dp)
        valid_idx = deps_dp != 0
        cs2_grid[valid_idx] = 1.0 / deps_dp[valid_idx]
        
        # find where cs2 > 1.0 and slice
        violation_idx = np.where(cs2_grid > 1.0)[0]
        if len(violation_idx) > 0:
            slice_idx = violation_idx[0]
            p_grid_sliced = p_grid[:slice_idx]
            cs2_grid_sliced = cs2_grid[:slice_idx]
        else:
            p_grid_sliced = p_grid
            cs2_grid_sliced = cs2_grid
            
        if len(p_grid_sliced) == 0:
            P_max_causal = P_trans_default + 0.1
        else:
            P_max_causal = max(p_grid_sliced)
            
        # plot sliced cs2
        line, = ax_cs2.plot(p_grid_sliced, cs2_grid_sliced, label=name)
        color = line.get_color()

        # 2. The Bounded TOV Solver
        P_trans_actual = P_trans_default
        try:
            def obj(p):
                # evaluate the first crust layer at p
                e_crust = float(crust_funcs["c1"][0](p))
                return float(fA_e(p)) - e_crust
            if obj(P_trans_default) < 0:
                p_high = P_trans_default
                for p_test in np.linspace(P_trans_default, 10.0, 100):
                    if obj(p_test) >= 0:
                        p_high = p_test
                        break
                sol = root_scalar(obj, bracket=[P_trans_default, p_high])
                P_trans_actual = sol.root
        except:
            pass

        def eos_callable(p):
            p = float(p)
            if p > P_trans_actual:
                e = float(fA_e(p))
                de = float(fA_de(p))
                return e, float(1.0/de if de > 0 else 0.0)
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
        
        try:
            curve = bounded_solve_sequence(eos_callable, P_max_causal)
            if len(curve) > 0:
                masses = [pt[0] for pt in curve]
                radii = [pt[1] for pt in curve]
                
                max_mass = masses[-1]
                
                ax_mr.plot(radii, masses, color=color, label=name)
                # plot causality death X
                ax_mr.plot(radii[-1], masses[-1], marker='X', color=color, markersize=8, markeredgecolor='black')
                
                print(f"[DIAGNOSTIC] {name} terminated at M_max = {max_mass:.2f} M_sun (P_causal = {P_max_causal:.2f})")
            else:
                print(f"[DIAGNOSTIC] {name} returned empty curve.")
        except Exception as e:
            print(f"[ERROR] TOV Failed for {name}: {e}")
            continue

    # 3. Visuals
    ax_cs2.set_xlabel("Pressure P [MeV/fm³]")
    ax_cs2.set_ylabel("Speed of Sound squared $c_s^2$")
    ax_cs2.set_title("Hadronic Baselines $c_s^2$ vs P (Causality Guillotine)")
    ax_cs2.set_xlim(0, 1200)
    ax_cs2.set_ylim(0, 1.5)
    
    ax_cs2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Causality Limit (1.0)')
    ax_cs2.axhline(1.0/3.0, color='gray', linestyle='--', linewidth=2, label='Conformal Limit (1/3)')
    
    if len(core_lib) <= 25:
        ax_cs2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_cs2.tight_layout()
    fig_cs2.savefig("test_diagnostics_cs2.pdf")

    ax_mr.set_xlabel("Radius R [km]")
    ax_mr.set_ylabel("Mass M [$M_\odot$]")
    ax_mr.set_title("Hadronic Baselines M-R (Causality Guillotine)")
    ax_mr.set_xlim(8, 20)
    ax_mr.set_ylim(0.0, 3.2)
    if len(core_lib) <= 25:
        ax_mr.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_mr.tight_layout()
    fig_mr.savefig("test_diagnostics_mr.pdf")

if __name__ == "__main__":
    print("Starting Causality Guillotine...")
    main()
