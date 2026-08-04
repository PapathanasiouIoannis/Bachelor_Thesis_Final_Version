# Family-pilot classification report

## Outcome

The frozen model is L2-regularized logistic regression on 21 radii sampled
from 1.0 to 2.0 M_sun. Mass is implicit and no generation, provenance,
central-density, surface-density, maximum-mass, or quark-parameter metadata
is exposed to the classifier.

| Evaluation | Independent families | Curve variants | Balanced accuracy | ROC AUC | Brier |
|---|---:|---:|---:|---:|---:|
| Training-family OOF | 13 | 84 | 1.000 | 1.000 | 0.0153 |
| Validation families | 2 | 12 | 1.000 | 1.000 | 0.0013 |
| One-time locked test | 2 | 12 | 1.000 | 1.000 | 0.0001 |
| Strict-2.08 development OOF | 12 | 78 | 1.000 | 1.000 | 0.0141 |

## Integrity findings

- A alone scores 0.50, and every A value is class-balanced in development.
- Raw sequence geometry, global physics summaries, provenance flags, and
  quark-parameter presence each score 1.00 and are therefore hard-forbidden.
- All A variants and related EoSs remain within one physical-family split.
- The fixed-specification exhaustive family-label null gives p=1/1716.
- The final test was opened once at the committed pre-test lock recorded in
  `docs/family_final_test.json`.

## Interpretation and limitations

A single low-mass radius already separates the development catalog; tidal
features alone are weaker. The result therefore reflects low-mass radius
topology of these repository surrogates versus this analytic CFL model, not
an opaque or universal matter-phase classifier. The final test has only two
independent families; its six A variants per family are correlated. All CFL
families share one analytic MIT-bag superfamily, exact hadronic fit
coefficients remain verified through a secondary thesis source, and this
theoretical full-curve input is not a direct observational deployment setup.
The independent test is a 2.0-M_sun-screen result; strict 2.08-M_sun evidence
is development OOF only.
