# Development robustness and ablation

This analysis uses only training and validation families; the locked test
pair remains unopened.

- Best single radius: 1.00 M_sun, inner accuracy 1.000, validation accuracy 1.000.
- Best single tidal feature: 1.95 M_sun, inner accuracy 0.667, validation accuracy 0.917.
- Whole-family permutation null: observed 1.000, null mean 0.490, maximum 1.000, empirical p=0.0006 (1716 permutations).

The low-mass radius separation is already sufficient in this selected
surrogate/CFL catalog. Therefore the defensible claim is model-set
discrimination driven mainly by low-mass radius topology, not a universal
or opaque machine-learning discovery.
