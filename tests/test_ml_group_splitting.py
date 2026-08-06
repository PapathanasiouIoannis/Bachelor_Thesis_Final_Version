import numpy as np
import pandas as pd

from src.config import CONFIG
from src.ml.dataset import resample_curves_to_common_mass_grid
from src.ml.hpo_groups import grouped_cv_indices, scale_inner_fold
from src.ml.splitting import attach_split_assignments, build_split_manifest


def _paired_sweep_frame(sweeps=10, rows_per_curve=3):
    records = []
    for sweep_index, amplitude in enumerate(np.linspace(-0.05, 0.09, sweeps)):
        sweep_id = f"A{sweep_index:05d}"
        for label in (0, 1):
            curve_id = f"{'H' if label == 0 else 'Q'}_{sweep_id}"
            for row_index in range(rows_per_curve):
                records.append(
                    {
                        "Row_ID": f"{curve_id}:{row_index}",
                        "Curve_ID": curve_id,
                        "Sweep_ID": sweep_id,
                        "Perturb_A": amplitude,
                        "Baseline_Name": "APR-1" if label == 0 else "CFL4",
                        "Mass": 1.0 + 0.1 * row_index,
                        "Radius": 12.0 - label,
                        "log10_Lambda": 2.5 - 0.1 * row_index,
                        "Label": label,
                    }
                )
    return pd.DataFrame.from_records(records)


def test_paired_sweeps_stay_together_in_contiguous_a_blocks():
    frame = _paired_sweep_frame()
    manifest = build_split_manifest(frame)
    assigned = attach_split_assignments(frame, manifest)

    assert assigned.groupby("Sweep_ID")["Split"].nunique().max() == 1
    assert assigned.groupby("Sweep_ID")["Label"].agg(set).map(lambda x: x == {0, 1}).all()

    ordered = manifest.sort_values("Perturb_A").reset_index(drop=True)
    for split in ("train", "val", "test"):
        positions = np.flatnonzero(ordered["Split"].to_numpy() == split)
        assert len(positions) > 0
        assert np.all(np.diff(positions) == 1)


def test_common_mass_resampling_retains_identical_rows_from_distinct_curves():
    records = []
    source_masses = [0.8, 1.0, 1.5, 2.0, 2.2]
    for curve_index in range(2):
        for mass in source_masses:
            records.append(
                {
                    "Curve_ID": f"H_{curve_index}",
                    "Sweep_ID": f"A{curve_index:05d}",
                    "Perturb_A": 0.01 * curve_index,
                    "Baseline_Name": "APR-1",
                    "Mass": mass,
                    "Radius": 12.0 - 0.2 * mass,
                    "log10_Lambda": 3.0 - mass,
                    "Label": 0,
                }
            )
    result = resample_curves_to_common_mass_grid(pd.DataFrame.from_records(records))

    expected = CONFIG["ML_MASS_GRID_POINTS"]
    assert len(result) == 2 * expected
    assert (result.groupby("Curve_ID").size() == expected).all()
    assert result["Row_ID"].nunique() == 2 * expected


def test_inner_hpo_folds_hold_out_contiguous_sweep_blocks():
    groups = pd.Series(
        [f"A{index:05d}" for index in range(9) for _ in range(4)]
    )
    labels = pd.Series([label for _ in range(9) for label in (0, 0, 1, 1)])
    folds = grouped_cv_indices(groups, labels)
    ordered_groups = sorted(groups.unique())
    for fit_indices, score_indices in folds:
        score_groups = sorted(groups.iloc[score_indices].unique())
        positions = [ordered_groups.index(group) for group in score_groups]
        assert np.all(np.diff(positions) == 1)
        assert set(labels.iloc[fit_indices]) == {0, 1}
        assert set(labels.iloc[score_indices]) == {0, 1}


def test_legacy_inner_hpo_folds_are_grouped_and_stratified():
    groups = pd.Series(
        [f"H_{index:02d}" for index in range(6) for _ in range(2)]
        + [f"Q_{index:02d}" for index in range(6) for _ in range(2)]
    )
    labels = pd.Series([0] * 12 + [1] * 12)

    for fit_indices, score_indices in grouped_cv_indices(groups, labels):
        assert set(groups.iloc[fit_indices]).isdisjoint(groups.iloc[score_indices])
        assert set(labels.iloc[fit_indices]) == {0, 1}
        assert set(labels.iloc[score_indices]) == {0, 1}


def test_inner_fold_scaler_never_fits_on_score_rows():
    features = pd.DataFrame({"Mass": [0.0, 2.0, 100.0], "Radius": [10.0, 12.0, 50.0]})
    fit, score = scale_inner_fold(
        features,
        np.array([0, 1]),
        np.array([2]),
    )

    assert np.allclose(fit.mean(axis=0), 0.0)
    assert np.all(np.abs(score) > 10.0)
