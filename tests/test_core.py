
import importlib.util
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "OR_EBM_complete_analysis.py"
spec = importlib.util.spec_from_file_location("analysis", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_threshold_search_is_strictly_post_threshold():
    x = np.arange(9, dtype=float)
    y = np.array([0, 0, 0, 1, 4, 4.05, 4.1, 4.12, 4.14])
    result = m.compute_threshold(x, y)
    assert result["threshold_value"] == 4.0
    assert result["saturation_value"] == 4.0


def test_early_flat_segment_not_called_saturation():
    x = np.arange(10, dtype=float)
    y = np.array([0, .01, .02, .03, 4, 5, 6, 7, 8, 9])
    result = m.compute_threshold(x, y)
    assert result["threshold_value"] == 4.0
    assert np.isnan(result["saturation_value"])


def test_mape_units_and_perfect_fit():
    scores = m.metrics([10, 20, 30], [10, 20, 30], nominal_p=1)
    assert scores["R2"] == 1.0
    assert scores["MAE"] == 0.0
    assert scores["MAPE_fraction"] == 0.0
    assert scores["MAPE_complement_percent"] == 100.0


def test_excludes_target_proxy_items():
    assert "Var_Oper_Readiness" not in m.FEATURES
    assert "Var_Project_Success" not in m.FEATURES
    assert not any(item.startswith("PS") for item in m.FEATURES)
