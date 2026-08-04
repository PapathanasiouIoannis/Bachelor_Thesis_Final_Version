import argparse
import json
import os
import subprocess
import sys

from src.runtime import add_runtime_args, configure_runtime_from_args, require_paths, runtime_paths


def run_phase(name, script_path):
    print("\n===============================================================")
    print(f"--- starting {name} ---")
    print("===============================================================")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Required pipeline script not found: {script_path}")

    exe = sys.executable if sys.executable else "py"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([exe, script_path], env=env, check=False)
    if result.returncode != 0:
        print(f"crashed at {name}. check logs for details.")
        sys.exit(result.returncode)
    print(f"done: {name}")


def main():
    parser = argparse.ArgumentParser(description="master thesis orchestrator (clean ML and advanced evaluation)")
    add_runtime_args(parser)
    parser.add_argument("--smoke-test", action="store_true", help="run readiness-sized training/evaluation defaults")
    parser.add_argument("--fast", action="store_true", help="run readiness-sized training/evaluation defaults")
    parser.add_argument("--skip-hpo", action="store_true", help="reuse existing best params instead of running HPO")
    parser.add_argument("--use-cuda-xgb", action="store_true", help="request CUDA-backed XGBoost")
    parser.add_argument(
        "--run-test-diagnostics",
        action="store_true",
        help=(
            "explicitly unlock repeated advanced diagnostics on held-out test labels; "
            "omit during model development"
        ),
    )
    args = parser.parse_args()
    if args.smoke_test:
        args.fast = True
    configure_runtime_from_args(args)
    if args.use_cuda_xgb:
        os.environ["THESIS_XGB_DEVICE"] = "cuda"

    print("===============================================================")
    print("    PHASES 3, 4, and 5    ")
    print("===============================================================")
    print(f"Active data root: {runtime_paths().data_root}")

    phases = [
        ("Data Pipeline", os.path.join("src", "ml", "data_pipeline.py")),
        ("Leakage Audit", os.path.join("src", "ml", "audit_leakage.py")),
    ]
    if not args.skip_hpo:
        phases.extend(
            [
                ("Optimize XGBoost", os.path.join("src", "ml", "optimize_xgboost.py")),
                ("Optimize MLP", os.path.join("src", "ml", "optimize_mlp.py")),
            ]
        )
    phases.extend(
        [
            ("Run Final XGBoost", os.path.join("src", "ml", "run_xgboost.py")),
            ("Run Final MLP", os.path.join("src", "ml", "run_mlp.py")),
        ]
    )
    if args.run_test_diagnostics:
        os.environ["THESIS_ALLOW_TEST_DIAGNOSTICS"] = "1"
        phases.extend(
            [
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
            ("Final Stage: Tables Generation", os.path.join("src", "visualize", "generate_tables.py")),
            ]
        )
    else:
        print(
            "Held-out test diagnostics are locked. Pass --run-test-diagnostics only "
            "for a declared final analysis run."
        )

    for name, script_path in phases:
        run_phase(name, script_path)

    print_final_summary()


def print_final_summary():
    paths = runtime_paths()
    xgb_params_path = paths.outputs_root / "xgboost_best_params.json"
    xgb_metrics_path = paths.outputs_root / "xgboost" / "metrics.json"
    mlp_params_path = paths.outputs_root / "mlp_best_params.json"
    mlp_metrics_path = paths.outputs_root / "mlp" / "metrics.json"
    require_paths([xgb_params_path, xgb_metrics_path, mlp_params_path, mlp_metrics_path], "Clean pipeline final summary")

    print("\n" + "=" * 63)
    print("                 PHASE 7 FINAL SUMMARY")
    print("=" * 63)

    with open(xgb_params_path, "r", encoding="utf-8") as f:
        xgb_params = json.load(f)
    with open(xgb_metrics_path, "r", encoding="utf-8") as f:
        xgb_metrics = json.load(f)
    print("\n[ XGBoost Final Architecture ]")
    for key, value in xgb_params.items():
        print(f"  - {key:<20}: {value:.4f}" if isinstance(value, float) else f"  - {key:<20}: {value}")
    print(f"  * Weights saved to : {paths.outputs_root / 'xgboost' / 'xgboost_weights.json'}")
    print("\n  >> Held-out Test Metrics:")
    for key, value in xgb_metrics.items():
        print(f"     - {key:<10}: {value:.4f}")

    print("-" * 63)
    with open(mlp_params_path, "r", encoding="utf-8") as f:
        mlp_params = json.load(f)
    with open(mlp_metrics_path, "r", encoding="utf-8") as f:
        mlp_metrics = json.load(f)
    print("\n[ MLP Final Architecture ]")
    for key, value in mlp_params.items():
        print(f"  - {key:<20}: {value:.4f}" if isinstance(value, float) else f"  - {key:<20}: {value}")
    print(f"  * Weights saved to : {paths.outputs_root / 'mlp' / 'mlp_weights.pth'}")
    print("\n  >> Held-out Test Metrics:")
    for key, value in mlp_metrics.items():
        print(f"     - {key:<10}: {value:.4f}")

    print("-" * 63)
    print("\n[ Generated Visualizations & Artifacts ]")
    print(f"  >> Optimization Plots : {paths.plots_root / 'ml_optimization'}")
    print(f"  >> Advanced Plots     : {paths.plots_root / 'ml_advanced'}")
    print(f"  >> Model Weights      : {paths.outputs_root}")
    print("\n" + "=" * 63)
    print("  Pipeline complete. All artifacts are ready.")
    print("=" * 63 + "\n")


if __name__ == "__main__":
    main()
