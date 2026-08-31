from transmission_rate_base import gate as G


def _pass_kwargs():
    return dict(
        ic={"mean_ic": 0.05, "t_stat": 3.0},
        spread={"sharpe": 0.6, "quintile_means": [-0.02, -0.01, 0.0, 0.01, 0.03],
                "positive_years_frac": 0.73},
        additivity={"alpha_ann": 0.05, "alpha_t": 2.4, "r2": 0.45},
        pre_thesis={"t_stat": -0.4},
    )


def test_all_pass():
    out = G.evaluate_gate(**_pass_kwargs())
    assert out["passed"] is True
    assert all(out["conditions"][k] for k in ("rank_ic", "quintile_spread", "additivity"))


def test_weak_ic_fails_overall():
    kw = _pass_kwargs()
    kw["ic"] = {"mean_ic": 0.01, "t_stat": 1.1}
    out = G.evaluate_gate(**kw)
    assert out["passed"] is False and out["conditions"]["rank_ic"] is False


def test_high_additivity_r2_fails():
    kw = _pass_kwargs()
    kw["additivity"] = {"alpha_ann": 0.05, "alpha_t": 2.4, "r2": 0.72}
    assert G.evaluate_gate(**kw)["passed"] is False


def test_low_positive_years_fails_spread():
    kw = _pass_kwargs()
    kw["spread"] = {**kw["spread"], "positive_years_frac": 0.4}
    out = G.evaluate_gate(**kw)
    assert out["passed"] is False and out["conditions"]["quintile_spread"] is False


def test_pre_thesis_negative_is_recorded_not_fatal():
    kw = _pass_kwargs()
    kw["pre_thesis"] = {"t_stat": -3.5}
    out = G.evaluate_gate(**kw)
    assert out["passed"] is True
    assert out["conditions"]["pre_thesis_sign"] is False


def test_quintile_inversions_counts_min_direction():
    assert G.quintile_inversions([1, 2, 3, 4, 5]) == 0
    assert G.quintile_inversions([5, 4, 3, 2, 1]) == 0
    assert G.quintile_inversions([1, 3, 2, 4, 5]) == 1


def test_nan_inputs_fail_cleanly():
    out = G.evaluate_gate(
        ic={"mean_ic": float("nan"), "t_stat": float("nan")},
        spread={"sharpe": float("nan"), "quintile_means": [float("nan")] * 5,
                "positive_years_frac": float("nan")},
        additivity={"alpha_ann": float("nan"), "alpha_t": float("nan"), "r2": float("nan")},
        pre_thesis={"t_stat": 0.0},
    )
    assert out["passed"] is False
