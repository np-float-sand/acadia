import pandas as pd

from research.xsec_alpha_probe.tilt import weights


def test_weights_sum_to_one_and_tilt_toward_high_z():
    z = pd.Series({"A": 2.0, "B": 0.0, "C": -3.0})
    w = weights(["A", "B", "C"], z)
    for v in w.values():
        assert abs(v.sum() - 1) < 1e-12
    assert w["EW"].nunique() == 1
    assert w["TILT_0.5"]["A"] > w["TILT_0.5"]["B"] > w["TILT_0.5"]["C"]
    assert w["TILT_1"]["C"] == 0          # 1 + 1.0 * -3 floored at zero
    assert list(w["TOP_HALF"].index) == ["A"]


def test_missing_score_is_neutral():
    w = weights(["A", "B"], pd.Series({"A": 1.0}))
    assert w["TILT_0.5"]["B"] < w["TILT_0.5"]["A"]
    assert abs(w["TILT_0.5"]["B"] - 1 / 2.5) < 1e-12
