from __future__ import annotations

import pandas as pd


def window_sensitivity_report(period_returns: pd.Series) -> pd.DataFrame:
    """
    Report the point estimate under the full sample and under three
    trimmed variants, so a short-sample result's stability to the exact
    start/end date is visible next to the headline number, not hidden
    behind it.
    """
    variants = {
        "full": period_returns,
        "drop_first": period_returns.iloc[1:],
        "drop_last": period_returns.iloc[:-1],
        "drop_first_and_last": period_returns.iloc[1:-1],
    }
    rows = [
        {"variant": name, "n_periods": len(series), "mean_return": series.mean() if len(series) else float("nan")}
        for name, series in variants.items()
    ]
    return pd.DataFrame(rows)


def sample_size_caveat(n_periods: int, minimum_trusted: int = 12) -> str:
    """Plain-English caveat for reporting alongside any short-sample result."""
    if n_periods >= minimum_trusted:
        return ""
    return (
        f"Only {n_periods} independent period(s) observed (below the "
        f"{minimum_trusted}-period bar treated as minimally trustworthy here) — "
        "treat this result as directional, not confirmed."
    )
