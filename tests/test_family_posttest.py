import json

import pytest

from src.ml.family_final import file_sha256
from src.ml.family_posttest import load_completed_final_result


def test_completed_result_requires_matching_one_shot_marker(tmp_path):
    result_path = tmp_path / "result.json"
    marker_path = tmp_path / "marker.json"
    result = {
        "test_open_count": 1,
        "locked_git_commit": "abc",
        "model_profile_sha256": "profile",
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    marker = {
        "status": "COMPLETED",
        "result_sha256": file_sha256(result_path),
        "locked_git_commit": "abc",
        "model_profile_sha256": "profile",
    }
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert load_completed_final_result(result_path, marker_path) == result

    marker["result_sha256"] = "tampered"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash"):
        load_completed_final_result(result_path, marker_path)
