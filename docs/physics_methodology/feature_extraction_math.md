# Feature Extraction (`src/physics/feature_extraction.py`)

After the integration of a physical sequence, the framework extracts the exact macroscopic observables that correspond to multi-messenger astronomical measurements.

The topological input space used by the machine learning ensembles includes:
1. **Total Mass** ($M$)
2. **Surface Radius** ($R$)
3. **Tidal Deformability** ($\Lambda$)

The extraction process heavily utilizes `PchipInterpolator` to smoothly process the discrete integration steps without introducing Runge's phenomenon.

**Note on Compactness:** Previous iterations of this architecture utilized Compactness ($C = \frac{M}{R}$) as a discrete input feature. This was formally removed after auditing revealed that it induced severe data leakage, artificially inflating classifier performance by feeding mathematically redundant information.
