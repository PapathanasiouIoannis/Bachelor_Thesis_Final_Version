# Feature Extraction (src/physics/feature_extraction.py)

Once the TOV solver finishes, we have raw arrays of Pressure, Density, Mass, and Radius. 

In eature_extraction.py, I take these raw arrays and extract the exact macroscopic observables that a real telescope would see:
1. **Total Mass** ($)
2. **Surface Radius** ($)
3. **Tidal Deformability** ($\Lambda$), which is crucial for Gravitational Wave astronomy.

We originally used Compactness ( = M/R$), but I ended up removing it from the pipeline because it was mathematically redundant and caused data leakage during the ML training phase.
