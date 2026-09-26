"""
Approach C -- 2026-YTD out-of-sample holdout of the "RECENT window" GBM/RF cells
that looked strong (RECENT_2023_2025: rank-IC t up to +3.7) but failed the prior
pre-registered both-windows bar. User ask 2026-09-08: "see if they hold up for 2026 ytd."

THROWAWAY. Faithful re-derivation of research/narrative_mispricing/approach_c/{build_panel,model}.py
with the feature panel extended through 2026-08. Not byte-identical to the prior run
(macro intermediates were deleted -> recomputed from source), so the re-derived
RECENT_2023_2025 row is printed as a sanity check against the prior results.csv.

Reports, for GBM & RF x {full,lean} x {fwd_ret_1m, fwd_ret_3m}:
  A. RECENT_2023_2025  -- re-derived (compare to prior results.csv)
  B. RECENT_2023_2026  -- same, +8 fresh months
  C. HOLDOUT           -- train on months <= 2025-12, predict each month 2026-01..2026-08
                          (fwd_3m: ..2026-06); per-month cross-sectional rank-IC path + HAC t
  + feature importances (train<=2025-12 fit) and narrative_z's rank.
"""
from __future__ import annotations
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

REPO = Path(__file__).resolve().parents[2]
NM = REPO / "research/narrative_mispricing"
AC = NM / "approach_c"
TICKERS = ['NRG','VST','CNP','AEP','EXC','PPL','FE','CEG','TLN','ETR','WEC','DTE',
           'CMS','PCG','EIX','XEL','ES','ED','D','PEG','AEE','EVRG','DUK','SO','NEE']
M0, M1 = "2018-06-30", "2026-08-31"


def months_grid():
    return pd.date_range(M0, M1, freq="ME")


def _mz(s: pd.Series, win=36, minp=12) -> pd.Series:
    return (s - s.rolling(win, min_periods=minp).mean()) / s.rolling(win, min_periods=minp).std()


def load_narrative(months):
    f = NM / "composite_monthly_refined_2026.parquet"
    if not f.exists():
        f = NM / "composite_monthly_refined.parquet"
        print(f"  [warn] extended composite missing -> using {f.name} (ends 2025-09)")
    d = pd.read_parquet(f)
    d.index = pd.to_datetime(d.index).to_period("M").to_timestamp("M")
    return d.reindex(months)


def load_icr(months):
    d = pd.read_parquet(REPO / "research/icr_edgar_retest/icr_edgar_panel.parquet")
    d.index = pd.to_datetime(d.index)
    dd = d.reindex(pd.date_range(d.index.min(), months.max(), freq="D")).ffill()
    return dd.resample("ME").last().reindex(months)


def load_gsi(months):
    f = AC / "factor_scores_2026.parquet"
    if f.exists():
        fs = pd.read_parquet(f).reset_index()
    else:
        alt = REPO / "output/factor_scores.csv"
        print(f"  [warn] factor_scores_2026 missing -> {alt} (ends 2025-12; 2026 rows NaN)")
        fs = pd.read_csv(alt)
    fs.columns = [c.lower() for c in fs.columns]
    dcol = [c for c in fs.columns if "date" in c][0]
    vcol = [c for c in fs.columns if "score" in c or "factor" in c][-1]
    fs[dcol] = pd.to_datetime(fs[dcol]).dt.to_period("M").dt.to_timestamp("M")
    piv = fs.groupby([dcol, "ticker"])[vcol].mean().unstack("ticker")
    return piv.reindex(months)


def load_beta(months):
    f = REPO / "research/merchant_beta_orthogonalization/rolling_betas_baseline.parquet"
    sb = pd.read_parquet(f).reset_index()
    sb["date"] = pd.to_datetime(sb["date"]).dt.to_period("M").dt.to_timestamp("M")
    piv = sb.groupby(["date", "ticker"])["stress_beta"].mean().unstack("ticker")
    if piv.index.max() < months.max():
        print(f"  [warn] stress_beta ends {piv.index.max().date()} -> 2026 rows ffilled")
    return piv.reindex(piv.index.union(months)).sort_index().ffill().reindex(months)


def load_prices(months):
    import yfinance as yf
    cf = NM / "_holdout_prices.parquet"
    if cf.exists():
        px = pd.read_parquet(cf)
    else:
        px = yf.download(TICKERS, start="2015-01-01", end="2026-09-30",
                         interval="1d", auto_adjust=True, progress=False)["Close"]
        px.to_parquet(cf)
    px.index = pd.to_datetime(px.index)
    mpx = px.resample("ME").last()
    mom6 = np.log(mpx).diff(6).reindex(months)
    fwd1 = mpx.pct_change(1).shift(-1).reindex(months)
    fwd3 = mpx.pct_change(3).shift(-3).reindex(months)
    return mom6, fwd1, fwd3


def load_realyield(months):
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10", timeout=30)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))
        d.columns = ["date", "v"]
        d["date"] = pd.to_datetime(d["date"])
        s = pd.to_numeric(d.set_index("date")["v"], errors="coerce").dropna()
        m = s.resample("ME").last()
        return m.diff(6).reindex(months)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] real yield fetch failed ({e}) -> feature dropped")
        return pd.Series(np.nan, index=months)


def load_ng(months):
    import yfinance as yf
    try:
        ng = yf.download("NG=F", start="2015-01-01", end="2026-09-30",
                         interval="1d", auto_adjust=True, progress=False)["Close"]
        ng.index = pd.to_datetime(ng.index)
        m = ng.resample("ME").last()
        out = np.log(m).diff(6)
        out = out.reindex(months)
        return out.iloc[:, 0] if isinstance(out, pd.DataFrame) else out
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] NG fetch failed ({e}) -> feature dropped")
        return pd.Series(np.nan, index=months)


def load_catdem(months):
    d = pd.read_parquet(REPO / "grid_equipment_basket/data/cache/category_demand.parquet")
    d.index = pd.to_datetime(d.index).to_period("M").to_timestamp("M")
    mom = np.log(d["ee_new_orders"]).diff(6)
    return _mz(mom).reindex(months)


def load_capex(months):
    try:
        import sys
        sys.path.insert(0, str(REPO))
        from grid_resilience.data.utility_capex_guidance import (
            load_capex_guidance, aggregate_revision_series)
        df = load_capex_guidance()
        s = aggregate_revision_series(df)  # event-dated trailing-4Q $ revision sum
        s.index = pd.to_datetime(s.index)
        daily = s.reindex(pd.date_range(s.index.min(), months.max(), freq="D")).ffill()
        m = daily.resample("ME").last()
        return _mz(m).reindex(months)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] capex guidance failed ({e}) -> feature dropped")
        return pd.Series(np.nan, index=months)


def build_panel():
    months = months_grid()
    narr = load_narrative(months)
    icr = load_icr(months)
    gsi = load_gsi(months)
    beta = load_beta(months)
    mom6, fwd1, fwd3 = load_prices(months)
    realyld = load_realyield(months)
    ng = load_ng(months)
    catdem = load_catdem(months)
    capex = load_capex(months)

    rows = []
    for t in TICKERS:
        if t not in narr.columns:
            continue
        d = pd.DataFrame(index=months)
        d["month"] = months
        d["ticker"] = t
        d["narrative_z"] = narr[t]
        d["icr"] = icr[t] if t in icr.columns else np.nan
        d["stress_beta"] = beta[t] if t in beta.columns else np.nan
        d["gsi_factor"] = gsi[t] if t in gsi.columns else np.nan
        d["own_mom6m"] = mom6[t] if t in mom6.columns else np.nan
        d["capex_guidance_z"] = capex.values
        d["category_demand_z"] = catdem.values
        d["real_yield_chg6m"] = realyld.values
        d["ng_mom6m"] = ng.values
        d["fwd_ret_1m"] = fwd1[t] if t in fwd1.columns else np.nan
        d["fwd_ret_3m"] = fwd3[t] if t in fwd3.columns else np.nan
        rows.append(d)
    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(NM / "_holdout_panel.parquet")
    return panel


FULL = ["narrative_z", "icr", "stress_beta", "gsi_factor", "own_mom6m",
        "capex_guidance_z", "category_demand_z", "real_yield_chg6m", "ng_mom6m"]
LEAN = ["narrative_z", "icr", "gsi_factor", "capex_guidance_z"]


def hac_t(x, lags=3):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, n
    mean = x.mean(); r = x - mean
    var = (r @ r) / n
    for l in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - l / (lags + 1)) * (r[l:] @ r[:-l]) / n
    se = np.sqrt(max(var, 1e-12) / n)
    return mean, (mean / se if se > 0 else np.nan), n


def fit_predict_ic(train, test, feats, target, kind, seed=0):
    feats = [f for f in feats if train[f].notna().any()]
    Xtr, ytr = train[feats].copy(), train[target]
    med = Xtr.median()
    Xtr = Xtr.fillna(med).fillna(0.0)
    ok = ytr.notna()
    Xtr, ytr = Xtr[ok], ytr[ok]
    if len(ytr) < 40:
        return {}, None, feats
    mdl = (GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=seed) if kind == "gbm"
           else RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=5,
                                      random_state=seed))
    mdl.fit(Xtr, ytr)
    ic_by_m = {}
    for m, tm in test.groupby("month"):
        y = tm[target]
        v = y.notna()
        if v.sum() < 5:
            continue
        Xte = tm[feats].fillna(med).fillna(0.0)
        ic, _ = spearmanr(mdl.predict(Xte[v]), y[v])
        ic_by_m[m] = ic
    imp = pd.Series(mdl.feature_importances_, index=feats).sort_values(ascending=False)
    return ic_by_m, imp, feats


def walk_forward(panel_sub, feats, target, kind, n_splits=5):
    from sklearn.model_selection import TimeSeriesSplit
    ms = np.sort(panel_sub["month"].unique())
    if len(ms) < n_splits + 5:
        n_splits = max(2, len(ms) // 6)
    ic = {}
    for tr_i, te_i in TimeSeriesSplit(n_splits=n_splits).split(ms):
        tr = panel_sub[panel_sub["month"].isin(ms[tr_i])]
        te = panel_sub[panel_sub["month"].isin(ms[te_i])]
        d, _, _ = fit_predict_ic(tr, te, feats, target, kind)
        ic.update(d)
    return pd.Series(ic).sort_index()


def main():
    print("building extended panel (through 2026-08) ...")
    panel = build_panel()
    cov = panel.groupby("month").size()
    print(f"  months {panel.month.min().date()}..{panel.month.max().date()} ({panel.month.nunique()})")
    print("  feature non-null by era:")
    for f in FULL:
        pre = panel.loc[panel.month <= '2025-12-31', f].notna().mean()
        post = panel.loc[panel.month >= '2026-01-31', f].notna().mean()
        print(f"    {f:20s} <=2025-12 {pre:5.1%}   2026 {post:5.1%}")

    rows = []
    for feat_name, feats in [("full", FULL), ("lean", LEAN)]:
        for target in ["fwd_ret_1m", "fwd_ret_3m"]:
            for kind in ["gbm", "rf"]:
                tag = f"{feat_name:4s} {target:10s} {kind:3s}"
                # A. re-derived RECENT_2023_2025
                a = walk_forward(panel[(panel.month >= "2023-01-01") & (panel.month <= "2025-12-31")],
                                 feats, target, kind)
                ma, ta, na = hac_t(a.values)
                # B. RECENT_2023_2026
                b = walk_forward(panel[panel.month >= "2023-01-01"], feats, target, kind)
                mb, tb, nb = hac_t(b.values)
                # C. HOLDOUT: train <= 2025-12, predict 2026
                tr = panel[panel.month <= "2025-12-31"]
                te = panel[panel.month >= "2026-01-01"]
                cdict, imp, used = fit_predict_ic(tr, te, feats, target, kind)
                c = pd.Series(cdict).sort_index()
                mc, tc, nc = hac_t(c.values, lags=2)
                nrank = (list(imp.index).index("narrative_z") + 1) if (imp is not None and "narrative_z" in imp.index) else None
                pm = ", ".join(f"{k.strftime('%y-%m')}:{v:+.2f}" for k, v in c.items())
                print(f"\n[{tag}]")
                print(f"  A RECENT_2023_2025 (re-derived) IC={ma:+.3f} t={ta:+.2f} n={na}")
                print(f"  B RECENT_2023_2026            IC={mb:+.3f} t={tb:+.2f} n={nb}")
                print(f"  C HOLDOUT 2026-YTD           IC={mc:+.3f} t={tc:+.2f} n={nc}   per-month: {pm}")
                if imp is not None:
                    print(f"    importances: {dict(imp.round(3))}  | narrative_z rank = {nrank}/{len(used)}")
                rows.append(dict(feats=feat_name, target=target, model=kind,
                                 A_ic=ma, A_t=ta, B_ic=mb, B_t=tb, C_ic=mc, C_t=tc, C_n=nc,
                                 narr_rank=nrank))
    out = pd.DataFrame(rows)
    out.to_csv(NM / "spike_c_2026_holdout_results.csv", index=False)
    print(f"\nwrote {NM / 'spike_c_2026_holdout_results.csv'}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
