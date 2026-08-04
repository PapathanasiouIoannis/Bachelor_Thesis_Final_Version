# One-time locked family test

Locked commit: `85e3a26059ed26a7af0b7be38ae02bfbf703ca88`.

The frozen radius-only logistic model classified 12 of 12 curve variants with balanced accuracy 1.000 and ROC AUC 1.000.

| EoS | Label | Accuracy | Mean P(CFL) | P range across A | Minimum Mmax |
|---|---:|---:|---:|---:|---:|
| CFL14 | 1 | 1.000 | 0.9861 | 0.0000 | 2.0705 |
| MDI-2 | 0 | 1.000 | 0.0096 | 0.0053 | 2.0044 |

Only two independent physical families are present in this final test; the
six A variants per EoS are correlated sensitivity variants, not twelve
independent validation objects. No family-level confidence interval is
claimed. The CFL test family is also an unseen B=100 parameter block.

Neither test EoS reaches the 2.08 M_sun sensitivity threshold over the
shared A grid. Consequently this is a 2.0 M_sun-screen result; strict-2.08
performance must be reported separately as development-family OOF only.
