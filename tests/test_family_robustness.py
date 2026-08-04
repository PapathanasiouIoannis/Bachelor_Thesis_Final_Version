import pandas as pd

from src.ml.family_robustness import family_label_permutation_null


def test_family_permutation_preserves_whole_groups():
    records = []
    for label, prefix in ((0, "H"), (1, "Q")):
        for group_index in range(3):
            group_id = f"{prefix}_{group_index}"
            for sample_index in range(3):
                records.append(
                    {
                        "Sample_ID": f"{group_id}:{sample_index}",
                        "EoS_ID": group_id,
                        "Group_ID": group_id,
                        "Perturb_A": 0.01 * sample_index,
                        "Radius_M1p00": float(label * 2 + sample_index / 10),
                        "Label": label,
                    }
                )
    training = pd.DataFrame.from_records(records)

    result = family_label_permutation_null(
        training,
        ["Radius_M1p00"],
        c_value=0.1,
        permutations=20,
    )

    assert result["unit"] == "physical family"
    assert len(result["null_scores"]) == 20
    assert 0.0 <= result["empirical_p_value"] <= 1.0
