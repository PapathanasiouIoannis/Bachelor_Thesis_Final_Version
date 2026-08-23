# Macroscopic Feature Extraction

After the TOV equations are fully integrated into a discrete sequence of mass and radius coordinates, `src/physics/feature_extraction.py` processes the raw curve to extract the exact macroscopic observables required by the machine learning classifiers.

The primary topological input vector is $X = \{M, R, \log_{10}\Lambda\}$.

Because the RK45 integration steps are dynamically sized, the solver rarely lands exactly on canonical targets like $1.4 M_\odot$. To extract these specific features, the script employs SciPy's `PchipInterpolator`. I explicitly chose Piecewise Cubic Hermite Interpolating Polynomials over standard cubic splines because PCHIP enforces strict monotonicity, completely eliminating the aggressive numerical ringing (Runge's phenomenon) that plagued earlier versions of the pipeline.

### The Removal of Compactness
In an earlier iteration of the project, the compactness parameter $C = \frac{M}{R}$ was provided to the XGBoost algorithms as an explicit input feature. 

During the later audit, compactness was identified as a deterministic function
of features already supplied to the classifier. Retaining it could make model
performance depend on an unnecessary engineered proxy. Compactness was
therefore removed from the approved feature schema. This control narrows one
shortcut risk; it does not by itself establish independent physical-family
generalization or observational validity.
