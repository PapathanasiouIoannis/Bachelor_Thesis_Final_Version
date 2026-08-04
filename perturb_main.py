import argparse
import json
import os
import subprocess
import sys

from src.runtime import add_runtime_args, configure_runtime_from_args, require_paths, runtime_paths


def run_phase(name, script_path):
    print(f"\nRunning {name}...")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Required pipeline script not found: {script_path}")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, script_path], env=env, check=False)
    if result.returncode != 0:
        print(f"crashed at {name}. check logs for details.")
        sys.exit(result.returncode)
    print(f"done: {name}")


def print_final_summary():
    paths = runtime_paths()
    print("\n" + "=" * 63)
    print("      PERTURBED ML PIPELINE COMPLETE (NOISY DATA)      ")
    print("=" * 63)

    required = []
    for feature_set in ["MR", "MRL"]:
        required.extend(
            [
                paths.outputs_perturb_root / f"xgboost_{feature_set}" / "metrics.json",
                paths.outputs_perturb_root / f"mlp_{feature_set}" / "metrics.json",
            ]
        )
    require_paths(required, "Perturbed pipeline final summary")

    for feature_set in ["MR", "MRL"]:
        print(f"\n[ Feature Set: {feature_set} ]")
        xgb_metrics_path = paths.outputs_perturb_root / f"xgboost_{feature_set}" / "metrics.json"
        mlp_metrics_path = paths.outputs_perturb_root / f"mlp_{feature_set}" / "metrics.json"
        with open(xgb_metrics_path, "r", encoding="utf-8") as f:
            xgb_metrics = json.load(f)
        with open(mlp_metrics_path, "r", encoding="utf-8") as f:
            mlp_metrics = json.load(f)
        print(
            "  XGBoost PR-AUC: "
            f"{xgb_metrics.get('PR-AUC-Trapezoidal', 0):.4f}"
        )
        print(
            "  MLP PR-AUC:     "
            f"{mlp_metrics.get('PR-AUC-Trapezoidal', 0):.4f}"
        )

    print("-" * 63)
    print("\n[ Generated Visualizations & Artifacts ]")
    print(f"  >> Optimization Plots : {paths.plots_perturb_root / 'ml_optimization'}")
    print(f"  >> Advanced Plots     : {paths.plots_perturb_root / 'ml_advanced'}")
    print(f"  >> Model Weights      : {paths.outputs_perturb_root}")
    print("\n" + "=" * 63)
    print("  Perturbed pipeline complete. All artifacts are ready.")
    print("=" * 63 + "\n")


def main():
    parser = argparse.ArgumentParser(description="perturbed observational ML pipeline")
    add_runtime_args(parser)
    parser.add_argument("--smoke-test", action="store_true", help="run readiness-sized training/evaluation defaults")
    parser.add_argument("--fast", action="store_true", help="run readiness-sized training/evaluation defaults")
    parser.add_argument("--skip-hpo", action="store_true", help="reuse existing best params instead of running HPO")
    parser.add_argument("--use-cuda-xgb", action="store_true", help="request CUDA-backed XGBoost")
    parser.add_argument(
        "--run-test-diagnostics",
        action="store_true",
        help="unlock repeated held-out-test diagnostics for a declared final run",
    )
    args = parser.parse_args()
    if args.smoke_test:
        args.fast = True
    configure_runtime_from_args(args)
    if args.use_cuda_xgb:
        os.environ["THESIS_XGB_DEVICE"] = "cuda"

    print("=" * 63)
    print("      INITIALIZING PERTURBED OBSERVATIONAL ML PIPELINE      ")
    print("=" * 63 + "\n")
    print(f"Active data root: {runtime_paths().data_root}")

    phases = [
        ("Noisy Data Pipeline", os.path.join("src", "ml_perturb", "data_pipeline.py")),
        ("Perturbed Leakage Audit", os.path.join("src", "ml", "audit_leakage.py")),
    ]
    os.environ["THESIS_AUDIT_PERTURBED_ONLY"] = "1"
    if not args.skip_hpo:
        phases.extend(
            [
                ("Optimize XGBoost (MR & MRL)", os.path.join("src", "ml_perturb", "optimize_xgboost.py")),
                ("Optimize MLP (MR & MRL)", os.path.join("src", "ml_perturb", "optimize_mlp.py")),
            ]
        )
    phases.extend(
        [
            ("Run Final XGBoost (MR & MRL)", os.path.join("src", "ml_perturb", "run_xgboost.py")),
            ("Run Final MLP (MR & MRL)", os.path.join("src", "ml_perturb", "run_mlp.py")),
        ]
    )
    if args.run_test_diagnostics:
        os.environ["THESIS_ALLOW_TEST_DIAGNOSTICS"] = "1"
        phases.extend(
            [
            ("Advanced Eval: Calibration", os.path.join("src", "ml_perturb", "advanced", "run_calibration.py")),
            ("Advanced Eval: UMAP Topology", os.path.join("src", "ml_perturb", "advanced", "run_umap.py")),
            ("Advanced Eval: Uncertainty", os.path.join("src", "ml_perturb", "advanced", "eval_uncertainty.py")),
            ("Advanced Eval: MC Inference", os.path.join("src", "ml_perturb", "advanced", "run_mc_inference.py")),
            ("Advanced Eval: Confusion Matrix", os.path.join("src", "ml_perturb", "advanced", "run_confusion_matrix.py")),
            ("Advanced Eval: Feature Importance", os.path.join("src", "ml_perturb", "advanced", "run_feature_importance.py")),
            ("Advanced Eval: Perturbation Effects", os.path.join("src", "visualize", "plot_perturbation_effect.py")),
            ]
        )
    else:
        print(
            "Held-out test diagnostics are locked. Pass --run-test-diagnostics only "
            "for a declared final analysis run."
        )

    for name, script_path in phases:
        if name == "Perturbed Leakage Audit":
            env_args = os.environ.copy()
            env_args["PYTHONPATH"] = os.getcwd() + os.pathsep + env_args.get("PYTHONPATH", "")
            result = subprocess.run([sys.executable, script_path, "--perturbed-only"], env=env_args, check=False)
            if result.returncode != 0:
                sys.exit(result.returncode)
            print(f"done: {name}")
        else:
            run_phase(name, script_path)

    print_final_summary()


if __name__ == "__main__":
    main()
