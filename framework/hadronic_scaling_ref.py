import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

sys.path.append(os.path.abspath('.'))

from src.config import CONFIG
from src.physics.get_eos_library import get_eos_library
from src.physics.tov_rhs import tov_rhs

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

    pressures = np.geomspace(1.0, p_max_causal, 100)
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
                rtol=1e-6,
                atol=1e-8,
            )

            if sol.status == 1 and len(sol.t_events[0]) > 0:
                R = sol.t_events[0][0]
                M = sol.y_events[0][0][0]
                if R < 5.0 or M < 0.05:
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
    os.makedirs('plots/scaled_diagnostics', exist_ok=True)
    
    core_lib, crust_funcs = get_eos_library()
    P_trans_default = CONFIG["P_TRANS_DEFAULT"]

    for name, core_anchor in core_lib.items():
        fA_e, fA_de = core_anchor
        
        fig_cs2, ax_cs2 = plt.subplots(figsize=(8, 6))
        fig_mr, ax_mr = plt.subplots(figsize=(8, 6))
        
        # -----------------------------------------------------
        # unperturbed Baseline
        # -----------------------------------------------------
        p_grid_base = np.linspace(P_trans_default, 1200.0, 5000)
        eps_grid_base = fA_e(p_grid_base)
        deps_dp = fA_de(p_grid_base)
        cs2_grid_base = np.zeros_like(deps_dp)
        valid_idx = deps_dp != 0
        cs2_grid_base[valid_idx] = 1.0 / deps_dp[valid_idx]
        
        # baseline crust transition
        P_trans_actual = P_trans_default
        try:
            def obj(p):
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

        def eos_baseline(p):
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
                
        eos_baseline.eps_surf = 0.0

        # find causality death for unperturbed baseline
        viol_idx = np.where(cs2_grid_base > 1.0)[0]
        if len(viol_idx) > 0:
            slice_idx = viol_idx[0]
            p_max_causal_base = p_grid_base[slice_idx - 1] if slice_idx > 0 else P_trans_default + 0.1
        else:
            p_max_causal_base = 1200.0

        ax_cs2.plot(p_grid_base, cs2_grid_base, color='black', linestyle='--', linewidth=3, label='Baseline')

        curve_base = bounded_solve_sequence(eos_baseline, p_max_causal_base)
        if len(curve_base) > 0:
            m_b = [pt[0] for pt in curve_base]
            r_b = [pt[1] for pt in curve_base]
            ax_mr.plot(r_b, m_b, color='black', linestyle='--', linewidth=3, label='Baseline')
            ax_mr.plot(r_b[-1], m_b[-1], marker='X', color='black', markersize=8)

        # -----------------------------------------------------
        # perturbations
        # -----------------------------------------------------
        print(f"Generating perturbed scaling for {name}...")
        for i in range(50):
            A = np.random.uniform(-0.1, 0.1)
            eps_0 = np.random.uniform(300, 1200)
            sigma = np.random.uniform(20, 120)
            
            p_grid_ext = np.linspace(P_trans_default, 2500.0, 10000)
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
                    continue
                eps_sliced = eps_ext[:s_idx]
                cs2_sliced = cs2_pert[:s_idx]
            else:
                eps_sliced = eps_ext
                cs2_sliced = cs2_pert
                
            # avoid negative or exactly zero cs2 to ensure cumulative integral strictly increases
            cs2_sliced = np.clip(cs2_sliced, 1e-6, None)
            
            # re-integrate P = int cs2 d_eps
            P_pert = P_trans_default + cumulative_trapezoid(cs2_sliced, eps_sliced, initial=0)
            
            P_max_pert = P_pert[-1]
            if P_max_pert <= P_trans_default:
                continue
                
            ax_cs2.plot(P_pert, cs2_sliced, alpha=0.3, linewidth=1)
            
            # spline interpolators
            interp_eps = PchipInterpolator(P_pert, eps_sliced)
            interp_cs2 = PchipInterpolator(P_pert, cs2_sliced)
            
            def eos_pert(p):
                p = float(p)
                if p > P_trans_actual:
                    if p > P_max_pert:
                        return -1.0, -1.0
                    return float(interp_eps(p)), float(interp_cs2(p))
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
            
            eos_pert.eps_surf = 0.0
            
            curve_pert = bounded_solve_sequence(eos_pert, P_max_pert)
            if len(curve_pert) > 0:
                m_p = [pt[0] for pt in curve_pert]
                r_p = [pt[1] for pt in curve_pert]
                ax_mr.plot(r_p, m_p, alpha=0.3, linewidth=1)
                ax_mr.plot(r_p[-1], m_p[-1], marker='X', color='red', markersize=4, alpha=0.5)

        # -----------------------------------------------------
        # formatting
        # -----------------------------------------------------
        ax_cs2.set_xlim(0, 1200)
        ax_cs2.set_ylim(0, 1.2)
        ax_cs2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Causality Limit (1.0)')
        ax_cs2.axhline(1.0/3.0, color='gray', linestyle='--', linewidth=2, label='Conformal Limit (1/3)')
        ax_cs2.set_xlabel("Pressure P [MeV/fm³]")
        ax_cs2.set_ylabel("Speed of Sound squared $c_s^2$")
        ax_cs2.set_title(f"Parametric Scaling ($c_s^2$ vs P) - {name}")
        ax_cs2.legend(loc='upper right')
        fig_cs2.tight_layout()
        fig_cs2.savefig(f"plots/scaled_diagnostics/cs2_scaled_{name}.pdf")
        
        ax_mr.set_xlim(8, 20)
        ax_mr.set_ylim(0.1, 3.2)
        ax_mr.set_xlabel("Radius R [km]")
        ax_mr.set_ylabel(r"Mass M [$M_\odot$]")
        ax_mr.set_title(f"Parametric Scaling (M-R) - {name}")
        ax_mr.legend(loc='upper left')
        fig_mr.tight_layout()
        fig_mr.savefig(f"plots/scaled_diagnostics/MR_scaled_{name}.pdf")
        
        plt.close(fig_cs2)
        plt.close(fig_mr)

if __name__ == "__main__":
    main()
