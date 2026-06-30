# Advanced Diagnostics and Interpretability

Because neural networks and gradient boosters are essentially "black boxes," the framework executes several advanced mathematical diagnostics in `src/ml/advanced/` to formally prove the algorithmic logic.

- **UMAP Projections:** Uniform Manifold Approximation and Projection (UMAP) mathematically collapses the high-dimensional $M-R-\Lambda$ space into a 2D plane. This visually proves that the classifiers have successfully isolated distinct, non-linear class boundaries rather than just memorizing theoretical noise.
- **Brier Score Calibration:** To ensure the models are trustworthy, Brier scores evaluate calibration reliability. Using reliability diagrams, the script verifies whether a predicted output probability (e.g., $P(\text{Quark}) = 0.85$) accurately aligns with an 85% real-world observational frequency, confirming the model is neither underconfident nor overconfident.
- **SHAP Feature Importance:** Leveraging cooperative game theory, Shapley Additive exPlanations (Lundberg & Lee, 2017) are used to formally quantify the exact marginal predictive contribution of every single variable. The resulting SHAP beeswarm plots unequivocally prove that $\log_{10}\Lambda$ acts as the dominant feature driving the model's boundary decisions, especially when subjected to the observational noise pipeline.
