# Macroscopic Feature Extraction

Once the TOV equations yield a discrete array of mass and radius points, `feature_extraction.py` processes the curve to extract the inputs that the machine learning models will use.

The primary features are the total mass $M$, surface radius $R$, and the tidal deformability $\Lambda$. Because the integration points don't always land exactly on specific targets like 1.4 solar masses, the script uses a `PchipInterpolator`. This avoids the numerical ringing effects that happen with standard cubic splines.

An earlier version of the code used Compactness as a feature. However I realized during auditing that feeding $C = M/R$ alongside $M$ and $R$ caused data leakage and artificially boosted the classifier metrics, so it was removed to keep the evaluation strict.
