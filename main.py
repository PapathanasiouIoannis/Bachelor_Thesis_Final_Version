import subprocess
import sys
import os
import argparse

def run_phase(name, script_path):
    print(f"\n===============================================================")
    print(f"--- starting {name} ---")
    print(f"===============================================================")
    if not os.path.exists(script_path):
        print(f"error: couldnt find {script_path}. skipping.")
        return
        
    # use py if sys.executable is weird, but sys.executable is usually safest
    exe = sys.executable if sys.executable else "py"
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run([exe, script_path], env=env)
        if result.returncode != 0:
            print(f"crashed at {name}. check logs for details.")
            sys.exit(result.returncode)
        print(f"done: {name}")
    except Exception as e:
        print(f"failed to run {name}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="master thesis orchestrator (phases 3-5)")
    parser.add_argument("--smoke-test", action="store_true", help="run smoke test (if supported by scripts)")
    args = parser.parse_args()

    print("===============================================================")
    print("    PHASES 3, 4, and 5    ")
    print("===============================================================")
    
    # define execution sequence
    phases = [
        ("Data Pipeline", os.path.join("src", "ml", "data_pipeline.py")),
        ("Leakage Audit", os.path.join("src", "ml", "audit_leakage.py")),
        ("Optimize XGBoost", os.path.join("src", "ml", "optimize_xgboost.py")),
        ("Optimize MLP", os.path.join("src", "ml", "optimize_mlp.py")),
        ("Run Final XGBoost", os.path.join("src", "ml", "run_xgboost.py")),
        ("Run Final MLP", os.path.join("src", "ml", "run_mlp.py")),
        ("Advanced Eval: ROC", os.path.join("src", "ml", "advanced", "run_roc.py")),
        ("Advanced Eval: Calibration", os.path.join("src", "ml", "advanced", "run_calibration.py")),
        ("Advanced Eval: UMAP Topology", os.path.join("src", "ml", "advanced", "run_umap.py")),
        ("Advanced Eval: Uncertainty", os.path.join("src", "ml", "advanced", "eval_uncertainty.py")),
        ("Advanced Eval: MC Inference", os.path.join("src", "ml", "advanced", "run_mc_inference.py")),
        ("Advanced Eval: Confusion Matrix", os.path.join("src", "ml", "advanced", "run_confusion_matrix.py")),
        ("Advanced Eval: Feature Importance", os.path.join("src", "ml", "advanced", "run_feature_importance.py")),
        ("Advanced Eval: Topology", os.path.join("src", "ml", "advanced", "run_dataset_topology.py")),
        ("Advanced Eval: Noise Degradation", os.path.join("src", "ml", "advanced", "run_noise_degradation.py")),
        ("Advanced Eval: Raw Curves", os.path.join("src", "ml", "advanced", "run_plot_all_curves.py")),
        ("Final Stage: Latex Tables Generation", os.path.join("src", "visualize", "generate_tables.py"))
    ]

    for name, script_path in phases:
        run_phase(name, script_path)

    print_final_summary()

def print_final_summary():
    import json
    print("\n" + "="*63)
    print("                 PHASE 7 FINAL SUMMARY")
    print("="*63)
    
    # XGBoost
    try:
        with open(os.path.join("outputs", "xgboost_best_params.json"), "r") as f:
            xgb_params = json.load(f)
        with open(os.path.join("outputs", "xgboost", "metrics.json"), "r") as f:
            xgb_metrics = json.load(f)
            
        print("\n[ XGBoost Final Architecture ]")
        for k, v in xgb_params.items():
            if isinstance(v, float):
                print(f"  - {k:<20}: {v:.4f}")
            else:
                print(f"  - {k:<20}: {v}")
        print("  * Weights strictly saved to : outputs/xgboost/xgboost_weights.json")
        print("\n  >> Held-out Test Metrics:")
        for k, v in xgb_metrics.items():
            print(f"     - {k:<10}: {v:.4f}")
    except Exception:
        pass
        
    print("-" * 63)
        
    # MLP
    try:
        with open(os.path.join("outputs", "mlp_best_params.json"), "r") as f:
            mlp_params = json.load(f)
        with open(os.path.join("outputs", "mlp", "metrics.json"), "r") as f:
            mlp_metrics = json.load(f)
            
        print("\n[ MLP Final Architecture ]")
        for k, v in mlp_params.items():
            if isinstance(v, float):
                print(f"  - {k:<20}: {v:.4f}")
            else:
                print(f"  - {k:<20}: {v}")
        print("  * Weights strictly saved to : outputs/mlp/mlp_weights.pth")
        print("\n  >> Held-out Test Metrics:")
        for k, v in mlp_metrics.items():
            print(f"     - {k:<10}: {v:.4f}")
    except Exception:
        pass
        
    print("-" * 63)
    print("\n[ Generated Visualizations & Artifacts ]")
    print("  To view the results, check the following files:")
    print("  >> Optimization Plots : plots/ml_optimization/")
    print("  >> Advanced Plots     : plots/ml_advanced/")
    print("       - ... (calibration, umap, uncertainty, etc)")
    print("       - confusion_matrix_{xgboost|mlp}.pdf")
    print("       - feature_importance_{xgboost|mlp}.pdf")
    print("  >> Model Weights      : outputs/")
    print("\n  >> Advanced Physics Plots (plots/ml_advanced/):")
    print("     - mr_envelopes_ci.pdf          (68% & 95% Density Contours)")
    print("     - probability_density_1D.pdf   (1D Parameter Distributions)")
    print("     - roc_pr_curves.pdf            (XGBoost vs MLP ROC/PR curves)")
    print("     - calibration_curve.pdf        (Reliability Diagram)")
    print("     - umap_topology.pdf            (2D UMAP Projections)")
    print("     - uncertainty_calibration.pdf  (MC Dropout Epistemic Uncertainty)")
    print("     - mc_observational_noise_combined.pdf (Simulated Inference)")
    print("     - noise_degradation_accuracy.pdf (Performance Drop)")
    print("     - all_curves_raw.png           (Raw M-R Trajectories)")
    print("\n  >> Model Weights & SHAP (outputs/):")
    print("     - xgboost/shap_summary.pdf")
    print("     - xgboost/xgboost_weights.json")
    print("     - mlp/mlp_weights.pth")
        
    print("\n" + "="*63)
    print("  Pipeline complete. All artifacts are ready for presentation.")
    print("="*63 + "\n")

if __name__ == "__main__":
    main()
