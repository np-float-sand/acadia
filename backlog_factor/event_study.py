from __future__ import annotations

"""Backlog-surprise event study (spec 2026-08-31 §5). Abnormal CAR = name simple
return minus its industry-group ETF simple return, cumulative-summed over a window
of `w` trading days measured from the close of day 0 (day 0 = the first trading day
strictly after the SEC filing date)."""

import numpy as np
import pandas as pd

from backlog_factor import config

_MAJOR_GROUPS = ["machinery", "electrical_equipment", "aerospace_defense",
                 "engineering_construction", "building_products"]


def _window_abnormal_car(px: pd.Series, etf: pd.Series, filing: pd.Timestamp, w: int) -> float:
    r = px.pct_change()
    b = etf.pct_change().reindex(r.index)
    day0 = px.index.searchsorted(pd.Timestamp(filing), side="right")   # first bar strictly after filing
    seg = (r - b).iloc[day0 + 1: day0 + 1 + w]                          # close-to-close from day 0
    if len(seg) < max(2, w // 2):
        return np.nan
    return float(seg.sum())


def event_pairs(sp, prices, group_etf_prices, industry_group_map, group_etf_map, car_windows):
    out = []
    for _, row in sp.iterrows():
        t = row["ticker"]
        grp = industry_group_map.get(t)
        etf_t = group_etf_map.get(grp)
        if t not in prices.columns or etf_t not in group_etf_prices.columns:
            continue
        rec = {"ticker": t, "availability_date": pd.Timestamp(row["availability_date"]),
               "month": pd.Timestamp(row["availability_date"]).to_period("M"),
               "industry_group": grp, "surprise": float(row["surprise"])}
        for w in car_windows:
            rec[f"car_{w}"] = _window_abnormal_car(prices[t].dropna(),
                                                  group_etf_prices[etf_t].dropna(),
                                                  row["availability_date"], w)
        out.append(rec)
    return pd.DataFrame(out)


def _quintile(s: pd.Series) -> pd.Series:
    try:
        return pd.qcut(s, 5, labels=[1, 2, 3, 4, 5]).astype("Int64")
    except ValueError:
        return pd.Series(pd.NA, index=s.index, dtype="Int64")


def quintile_car_table(pairs: pd.DataFrame, car_windows) -> pd.DataFrame:
    p = pairs.copy()
    p["q"] = p.groupby("month")["surprise"].transform(_quintile)
    p = p.dropna(subset=["q"])
    rows = {q: {f"car_{w}_mean": p.loc[p["q"] == q, f"car_{w}"].mean() for w in car_windows}
            for q in [1, 2, 3, 4, 5]}
    tbl = pd.DataFrame(rows).T
    tbl.loc["Q5-Q1"] = tbl.loc[5] - tbl.loc[1]
    return tbl


def _monthly_spread(pairs: pd.DataFrame, value_col: str) -> pd.Series:
    p = pairs.copy()
    p["q"] = p.groupby("month")["surprise"].transform(_quintile)
    p = p.dropna(subset=["q"])
    hi = p[p["q"] == 5].groupby("month")[value_col].mean()
    lo = p[p["q"] == 1].groupby("month")[value_col].mean()
    return (hi - lo).dropna()


def _spread_tstat(m: pd.Series) -> float:
    if len(m) < 3 or m.std() == 0:
        return np.nan
    return float(m.mean() / (m.std() / np.sqrt(len(m))))


def clustered_tstat(pairs: pd.DataFrame, value_col: str, cluster_col: str = "month") -> float:
    return _spread_tstat(_monthly_spread(pairs, value_col))


def monotonic(quintile_means) -> bool:
    v = list(quintile_means)
    up = all(v[i] <= v[i + 1] for i in range(len(v) - 1))
    down = all(v[i] >= v[i + 1] for i in range(len(v) - 1))
    return bool(up or down)


def subperiod_split(pairs: pd.DataFrame):
    months = pairs["month"].sort_values().reset_index(drop=True)
    med = months.iloc[len(months) // 2]
    return pairs[pairs["month"] <= med], pairs[pairs["month"] > med]


def by_industry_signs(pairs: pd.DataFrame, value_col: str) -> pd.Series:
    out = {}
    for g, gp in pairs.groupby("industry_group"):
        m = _monthly_spread(gp, value_col)
        out[g] = float(np.sign(m.mean())) if len(m) else np.nan
    return pd.Series(out)


def evaluate_gate(pairs: pd.DataFrame, car_windows=None) -> dict:
    car_windows = list(car_windows or config.CAR_WINDOWS)
    tbl = quintile_car_table(pairs, car_windows)
    peak_w = max(car_windows, key=lambda w: abs(tbl.loc["Q5-Q1", f"car_{w}_mean"]))
    col = f"car_{peak_w}"
    q_means = [tbl.loc[q, f"{col}_mean"] for q in [1, 2, 3, 4, 5]]
    q5q1 = float(tbl.loc["Q5-Q1", f"{col}_mean"])
    t = clustered_tstat(pairs, col)
    s1, s2 = subperiod_split(pairs)
    m1, m2 = _monthly_spread(s1, col), _monthly_spread(s2, col)
    t1, t2 = _spread_tstat(m1), _spread_tstat(m2)
    signs = by_industry_signs(pairs, col)
    want = float(np.sign(q5q1))
    n_right = int((signs.reindex(_MAJOR_GROUPS).dropna() == want).sum())

    passed = bool(
        q5q1 > 0 and monotonic(q_means)
        and (t == t and t > 2)
        and (len(m1) >= 3 and float(np.sign(m1.mean())) == want and t1 == t1 and t1 > 1)
        and (len(m2) >= 3 and float(np.sign(m2.mean())) == want and t2 == t2 and t2 > 1)
        and n_right >= 3
    )
    return {
        "peak_window": int(peak_w), "q5_q1_car": q5q1, "monotone": monotonic(q_means),
        "tstat": float(t) if t == t else np.nan,
        "sub1_sign": float(np.sign(m1.mean())) if len(m1) else np.nan,
        "sub1_tstat": float(t1) if t1 == t1 else np.nan,
        "sub2_sign": float(np.sign(m2.mean())) if len(m2) else np.nan,
        "sub2_tstat": float(t2) if t2 == t2 else np.nan,
        "industry_signs": {k: (float(v) if v == v else None) for k, v in signs.items()},
        "n_groups_right_sign": n_right, "passed": passed, "drift_horizon_days": int(peak_w),
    }
