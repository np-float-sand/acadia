from __future__ import annotations

import numpy as np

from transmission_rate_base.config import GATE


def quintile_inversions(quintile_means: list[float]) -> int:
    """Adjacent-pair inversions against the better-fitting monotone direction
    (so a cleanly decreasing sequence also scores 0). A NaN bucket mean means
    the spread could not be formed -> return a large count so the gate fails."""
    a = np.asarray(quintile_means, dtype=float)
    if not np.isfinite(a).all():
        return 99
    d = np.diff(a)
    return int(min((d < 0).sum(), (d > 0).sum()))


def evaluate_gate(ic: dict, spread: dict, additivity: dict, pre_thesis: dict) -> dict:
    """Apply spec §11. Conditions 1-3 are hard (any False -> passed=False);
    condition 4 (pre-thesis sign) is recorded but does not flip passed."""
    reasons: list[str] = []

    rank_ic_ok = bool(
        np.isfinite(ic["mean_ic"]) and np.isfinite(ic["t_stat"])
        and ic["mean_ic"] >= GATE["min_mean_ic"] and ic["t_stat"] >= GATE["min_ic_t"]
    )
    reasons.append(
        f"rank-IC mean={ic['mean_ic']:.4f} (>= {GATE['min_mean_ic']}), "
        f"t={ic['t_stat']:.2f} (>= {GATE['min_ic_t']}) -> {'OK' if rank_ic_ok else 'FAIL'}"
    )

    inv = quintile_inversions(spread["quintile_means"])
    spread_ok = bool(
        np.isfinite(spread["sharpe"]) and spread["sharpe"] >= GATE["min_ls_sharpe"]
        and inv <= GATE["max_quintile_inversions"]
        and np.isfinite(spread["positive_years_frac"])
        and spread["positive_years_frac"] >= GATE["min_positive_years_frac"]
    )
    reasons.append(
        f"Q5-Q1 Sharpe={spread['sharpe']:.2f} (>= {GATE['min_ls_sharpe']}), "
        f"inversions={inv} (<= {GATE['max_quintile_inversions']}), "
        f"pos-years={spread['positive_years_frac']:.2f} (>= {GATE['min_positive_years_frac']}) "
        f"-> {'OK' if spread_ok else 'FAIL'}"
    )

    add_ok = bool(
        np.isfinite(additivity["alpha_ann"]) and np.isfinite(additivity["alpha_t"])
        and np.isfinite(additivity["r2"])
        and additivity["alpha_ann"] >= GATE["min_alpha_ann"]
        and additivity["alpha_t"] >= GATE["min_alpha_t"]
        and additivity["r2"] < GATE["max_additivity_r2"]
    )
    reasons.append(
        f"additivity alpha={additivity['alpha_ann']:.3f}/yr (>= {GATE['min_alpha_ann']}), "
        f"t={additivity['alpha_t']:.2f} (>= {GATE['min_alpha_t']}), "
        f"R2={additivity['r2']:.2f} (< {GATE['max_additivity_r2']}) "
        f"-> {'OK' if add_ok else 'FAIL'}"
    )

    pre_ok = bool(pre_thesis["t_stat"] > GATE["pre_thesis_min_t"])
    reasons.append(
        f"pre-thesis one-sided t={pre_thesis['t_stat']:.2f} "
        f"(> {GATE['pre_thesis_min_t']}) -> {'OK' if pre_ok else 'CAVEAT'}"
    )

    passed = bool(rank_ic_ok and spread_ok and add_ok)
    reasons.append(f"GATE {'PASSED' if passed else 'FAILED'}")
    return {
        "passed": passed,
        "conditions": {"rank_ic": rank_ic_ok, "quintile_spread": spread_ok,
                       "additivity": add_ok, "pre_thesis_sign": pre_ok},
        "reasons": reasons,
    }
