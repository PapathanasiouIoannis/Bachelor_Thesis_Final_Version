import os
import sys
import subprocess
import json

def run_phase(name, script_path):
    print(f"\nRunning {name}...")
    try:
        result = subprocess.run([sys.executable, script_path], check=False)
        if result.returncode != 0:
            print(f"crashed at {name}. check logs for details.")
            sys.exit(result.returncode)
        print(f"done: {name}")
    except Exception as e:
        print(f"Failed to execute {name}: {e}")
        sys.exit(1)

def print_final_summary():
    print("\n" + "="*63)
    print("      PERTURBED ML PIPELINE COMPLETE (NOISY DATA)      ")
    print("="*63)
    
    out_dir = "outputs_perturb"
    for feature_set in ["MR", "MRL"]:
        print(f"\n[ Feature Set: {feature_set} ]")
        
        xgb_metrics_path = os.path.join(out_dir, f"xgboost_{feature_set}", "metrics.json")
        mlp_metrics_path = os.path.join(out_dir, f"mlp_{feature_set}", "metrics.json")
        
        if os.path.exists(xgb_metrics_path):
            with open(xgb_metrics_path, "r") as f:
                xgb_metrics = json.load(f)
            print(f"  XGBoost PR-AUC: {xgb_metrics.get('PR-AUC', 0):.4f}")
            
        if os.path.exists(mlp_metrics_path):
            with open(mlp_metrics_path, "r") as f:
                mlp_metrics = json.load(f)
            print(f"  MLP PR-AUC:     {mlp_metrics.get('PR-AUC', 0):.4f}")

    print("-" * 63)
    print("\n[ Generated Visualizations & Artifacts ]")
    print("  To view the results, check the following directories:")
    print("  >> Optimization Plots : plots_perturb/ml_optimization/")
    print("  >> Advanced Plots     : plots_perturb/ml_advanced/")
    print("       - calibration_curve_{MR|MRL}.pdf")
    print("       - umap_topology_{MR|MRL}.pdf")
    print("       - uncertainty_calibration_{MR|MRL}.pdf")
    print("       - mc_observational_noise_combined_{MR|MRL}.pdf")
    print("       - confusion_matrix_{xgboost|mlp}_{MR|MRL}.pdf")
    print("       - feature_importance_{xgboost|mlp}_{MR|MRL}.pdf")
    print("       - perturbation_effect_contours.pdf")
    print("  >> Model Weights      : outputs_perturb/")
        
    print("\n" + "="*63)
    print("  Perturbed pipeline complete. All artifacts are ready for presentation.")
    print("="*63 + "\n")

def main():
    print("="*63)
    print("      INITIALIZING PERTURBED OBSERVATIONAL ML PIPELINE      ")
    print("="*63 + "\n")

    phases = [
        ("Noisy Data Pipeline", os.path.join("src", "ml_perturb", "data_pipeline.py")),
        ("Optimize XGBoost (MR & MRL)", os.path.join("src", "ml_perturb", "optimize_xgboost.py")),
        ("Optimize MLP (MR & MRL)", os.path.join("src", "ml_perturb", "optimize_mlp.py")),
        ("Run Final XGBoost (MR & MRL)", os.path.join("src", "ml_perturb", "run_xgboost.py")),
        ("Run Final MLP (MR & MRL)", os.path.join("src", "ml_perturb", "run_mlp.py")),
        ("Advanced Eval: Calibration", os.path.join("src", "ml_perturb", "advanced", "run_calibration.py")),
        ("Advanced Eval: UMAP Topology", os.path.join("src", "ml_perturb", "advanced", "run_umap.py")),
        ("Advanced Eval: Uncertainty", os.path.join("src", "ml_perturb", "advanced", "eval_uncertainty.py")),
        ("Advanced Eval: MC Inference", os.path.join("src", "ml_perturb", "advanced", "run_mc_inference.py")),
        ("Advanced Eval: Confusion Matrix", os.path.join("src", "ml_perturb", "advanced", "run_confusion_matrix.py")),
        ("Advanced Eval: Feature Importance", os.path.join("src", "ml_perturb", "advanced", "run_feature_importance.py")),
        ("Advanced Eval: Perturbation Effects", os.path.join("src", "visualize", "plot_perturbation_effect.py"))
    ]

    for name, script_path in phases:
        run_phase(name, script_path)
        
    print_final_summary()

if __name__ == "__main__":
    main()
