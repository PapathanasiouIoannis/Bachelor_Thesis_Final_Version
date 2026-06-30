# Advanced ML Diagnostics (src/ml/advanced/)

To really prove that the ML models were working, I couldn't just rely on standard accuracy scores. 

I wrote several scripts to do deep diagnostics:
- **UMAP Topology**: Reduces the 3D physics data into 2D space so we can visually see how the XGBoost model separates the Quark and Hadronic classes.
- **Calibration**: Uses Brier Scores to check if the model's output probabilities are honest (e.g., if it says 80% confident, is it actually right 80% of the time?).
- **Feature Importance**: Uses SHAP (SHapley Additive exPlanations) to mathematically prove which features (Mass, Radius, or $\Lambda$) were the most important for the model's decisions (Lundberg & Lee, 2017).
