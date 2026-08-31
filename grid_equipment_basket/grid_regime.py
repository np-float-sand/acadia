"""Layer-2 grid-congestion *regime* signal for the equal-weight grid-equipment
basket.

The basket's bull case is "the electricity grid is the bottleneck for the AI
buildout." Whether that bottleneck is **tightening or easing** is measurable in
the physical wholesale-power market -- transmission congestion and reserve-margin
tightness in the ISO zones where data-center load concentrates -- *before* it
shows up in equipment-maker margins, and it is not in prices or momentum. This
module turns that read into one **monthly exposure multiplier** on the basket
(step 8 of the spec), which layer 1's `apply_overlay_l2` applies in place of the
price trend gate:

    exposure = regime_multiplier(grid_regime)  x  vol_target_scalar(basket_returns)

Design / gate / honesty caveats:
    docs/superpowers/specs/2026-08-31-grid-regime-layer2-design.md

Data is the already-cached PJM zone-level LMP (real congestion component) and
per-zone load, read through `grid_resilience.data.grid_data` public functions --
NOT the heavy `build_multi_iso_gsi` pipeline. The gated backtest does zero
network I/O.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from grid_equipment_basket import config, overlay
from grid_equipment_basket.backtest import calendar_year_returns, compute_metrics
from grid_equipment_basket.overlay import _month_hold

_ANN = config.ANN_FACTOR


# ── per-zone daily signals ───────────────────────────────────────────────────

def _daily_zone_congestion(lmp_df: pd.DataFrame) -> pd.DataFrame:
    """Hourly zone LMP rows -> daily mean of ``|congestion $|`` per zone.

    Absolute dollars, *not* the ``|congestion| / |LMP|`` ratio -- the ratio
    blows up to >300% on divide-by-near-zero LMP hours (seen at COMED). Returns
    a frame indexed by ``date`` with one sorted column per zone; empty in ->
    empty out.
    """
    if lmp_df is None or lmp_df.empty:
        return pd.DataFrame()
    df = lmp_df.copy()
    df["_date"] = pd.to_datetime(df["time"]).dt.normalize()
    df["_abs_cong"] = df["congestion"].abs()
    daily = (df.groupby(["_date", "location"])["_abs_cong"].mean()
               .unstack("location").sort_index(axis=1))
    daily.index.name = "date"
    return daily


# PJM zone codes in the LMP cache that are aggregates / non-zone entities, not
# load zones -- excluded from the "rest of PJM" congestion benchmark (option A).
_PJM_NON_ZONE = {"PJM-RTO", "MID-ATL/APS", "OVEC"}


def _rest_of_pjm_congestion(lmp_df: pd.DataFrame, dc_zones: list[str]) -> pd.Series:
    """Daily mean ``|congestion $|`` across every PJM zone that is *not* one of
    ``dc_zones`` and not an aggregate (`_PJM_NON_ZONE`). This is the common-mode
    benchmark: a system-wide demand swing (COVID, a mild March) moves it and the
    DC zones together, so the DC-minus-rest spread nets it out."""
    daily = _daily_zone_congestion(lmp_df)
    if daily.empty:
        return pd.Series(dtype=float, name="rest_cong")
    keep = [c for c in daily.columns if c not in set(dc_zones) and c not in _PJM_NON_ZONE]
    if not keep:
        return pd.Series(dtype=float, name="rest_cong")
    return daily[keep].mean(axis=1).rename("rest_cong")


def _daily_zone_peak_load(load_df: pd.DataFrame) -> pd.DataFrame:
    """Hourly per-zone load rows -> daily peak MW per zone (date x zone)."""
    if load_df is None or load_df.empty:
        return pd.DataFrame()
    df = load_df.copy()
    df["_date"] = pd.to_datetime(df["time"]).dt.normalize()
    daily = (df.groupby(["_date", "zone"])["load_mw"].max()
               .unstack("zone").sort_index(axis=1))
    daily.index.name = "date"
    return daily


def _trailing_zscore(s: pd.Series, window: int, min_periods: int,
                     winsor: float) -> pd.Series:
    """``clip((s - trailing_mean) / trailing_std, +/- winsor)`` using a rolling
    window that includes only past-and-current observations (point-in-time). A
    zero-variance window -> NaN, not inf."""
    s = s.astype(float)
    roll = s.rolling(window, min_periods=min_periods)
    mean = roll.mean()
    std = roll.std().replace(0.0, np.nan)
    return ((s - mean) / std).clip(-winsor, winsor)


def _combine_subsignals(subsignals: dict[str, tuple[pd.Series, float]]) -> pd.Series:
    """Weight-normalised mean of named series. Entries with weight <= 0 are
    dropped. At each timestamp the weights are renormalised over the subsignals
    that are present (non-NaN) there, so a not-yet-warm subsignal simply doesn't
    contribute rather than NaN-ing the whole composite. All subsignals absent ->
    NaN at that timestamp."""
    active = {k: (s, float(w)) for k, (s, w) in subsignals.items() if w > 0}
    if not active:
        return pd.Series(dtype=float)
    frame = pd.DataFrame({k: s.astype(float) for k, (s, _) in active.items()})
    w = pd.Series({k: wt for k, (_, wt) in active.items()}).reindex(frame.columns)
    present = frame.notna()
    wsum = present.mul(w, axis=1).sum(axis=1)
    num = frame.mul(w, axis=1).sum(axis=1, min_count=1)
    return (num / wsum.replace(0.0, np.nan)).rename("composite")


# ── daily cross-zone composite ───────────────────────────────────────────────

def _default_lmp_fn(zones, start, end):
    from grid_resilience.data.grid_data import fetch_lmp
    df = fetch_lmp("PJM", start, end, location_type="ZONE", use_cache=True)
    if not df.empty and "location" in df.columns:
        df = df[df["location"].isin(zones)]
    return df


def _default_load_fn(zones, start, end):
    from grid_resilience.data.grid_data import fetch_zonal_load
    df = fetch_zonal_load(start, end, use_cache=True)
    if not df.empty and "zone" in df.columns:
        df = df[df["zone"].isin(zones)]
    return df


def regime_composite(start: str, end: str, *, zones: list[str],
                     w_cong: float = config.REGIME_W_CONG,
                     w_reserve: float = config.REGIME_W_RESERVE,
                     zone_weight: str = "equal",
                     zscore_window: int = config.REGIME_ZSCORE_WINDOW,
                     zscore_minp: int = config.REGIME_ZSCORE_MINP,
                     winsor: float = config.REGIME_ZSCORE_WINSOR,
                     lmp_fn=None, load_fn=None) -> pd.Series:
    """Daily cross-zone congestion/scarcity composite z (spec s4.1 steps 1-5).

    Per zone: z-scored daily ``|congestion $|`` (weight ``w_cong``) blended with
    z-scored daily reserve-tightness = peak-load / trailing-p99-peak-load
    (weight ``w_reserve``). Zones are then combined with equal weights, or
    weights proportional to trailing mean peak load when ``zone_weight="load"``
    (falls back to equal, with a warning, when load data is unavailable).

    ``lmp_fn(zones, start, end) -> hourly LMP frame`` and
    ``load_fn(zones, start, end) -> hourly load frame`` are injectable for tests;
    the defaults read the cached PJM parquet via ``grid_resilience.data.grid_data``.
    """
    lmp_fn = lmp_fn or _default_lmp_fn
    load_fn = load_fn or _default_load_fn

    cong = _daily_zone_congestion(lmp_fn(zones, start, end))
    if cong.empty:
        warnings.warn("grid_regime: no LMP/congestion data for the window; "
                      "composite is empty (signal inactive).", RuntimeWarning, stacklevel=2)
        return pd.Series(dtype=float, name="composite")
    cong_z = cong.apply(lambda col: _trailing_zscore(col, zscore_window, zscore_minp, winsor))

    peak = pd.DataFrame()
    reserve_z = pd.DataFrame()
    if w_reserve > 0 or zone_weight == "load":
        peak = _daily_zone_peak_load(load_fn(zones, start, end))
    if w_reserve > 0:
        if peak.empty:
            warnings.warn("grid_regime: w_reserve>0 but no load data; using congestion only.",
                          RuntimeWarning, stacklevel=2)
        else:
            tight = peak / peak.rolling(zscore_window, min_periods=zscore_minp).quantile(0.99)
            reserve_z = tight.apply(
                lambda col: _trailing_zscore(col, zscore_window, zscore_minp, winsor))

    zone_cols = list(cong_z.columns)
    per_zone = {}
    for z in zone_cols:
        subs = {"cong": (cong_z[z], w_cong)}
        if z in getattr(reserve_z, "columns", []):
            subs["reserve"] = (reserve_z[z], w_reserve)
        per_zone[z] = _combine_subsignals(subs)

    if zone_weight == "load" and not peak.empty:
        wt = peak.reindex(columns=zone_cols).mean()
        wt = (wt / wt.sum()).to_dict()
    else:
        if zone_weight == "load":
            warnings.warn("grid_regime: zone_weight='load' but no load data; equal-weighting zones.",
                          RuntimeWarning, stacklevel=2)
        wt = {z: 1.0 / len(zone_cols) for z in zone_cols}

    return _combine_subsignals({z: (per_zone[z], wt[z]) for z in zone_cols}).rename("regime_composite")


# ── option A: DC-zone congestion RELATIVE to the rest of PJM ────────────────

def relative_regime_composite(start: str, end: str, *,
                              dc_zones: list[str] = None,
                              zscore_window: int = config.REGIME_ZSCORE_WINDOW,
                              zscore_minp: int = config.REGIME_ZSCORE_MINP,
                              winsor: float = config.REGIME_ZSCORE_WINSOR,
                              lmp_fn=None) -> pd.Series:
    """Daily z-score of ``mean|congestion $|(DC zones) - mean|congestion $|(rest
    of PJM)`` (spec option A). A demand collapse that drags every zone's
    congestion down together leaves this ~unchanged; only congestion
    *concentrating* in the data-center zones moves it. Feeds `regime_multiplier`
    unchanged."""
    dc_zones = list(dc_zones or config.REGIME_DC_ZONES)
    lmp_fn = lmp_fn or _default_lmp_fn_all
    df = lmp_fn(start, end)
    dc = _daily_zone_congestion(df)
    if dc.empty:
        warnings.warn("grid_regime.relative: no PJM zone LMP for the window.",
                      RuntimeWarning, stacklevel=2)
        return pd.Series(dtype=float, name="rel_composite")
    dc_mean = dc[[c for c in dc_zones if c in dc.columns]].mean(axis=1)
    rest_mean = _rest_of_pjm_congestion(df, dc_zones)
    spread = (dc_mean - rest_mean.reindex(dc_mean.index)).rename("dc_minus_rest")
    return _trailing_zscore(spread, zscore_window, zscore_minp, winsor).rename("rel_composite")


def _default_lmp_fn_all(start, end):
    """All PJM zones (no zone filter) -- the relative signal needs the rest-of-PJM set."""
    from grid_resilience.data.grid_data import fetch_lmp
    return fetch_lmp("PJM", start, end, location_type="ZONE", use_cache=True)


def _eval_windows(series: pd.Series, primary, prior, rf: float) -> dict:
    out = {}
    for wk, (ws, we) in (("primary", primary), ("prior", prior)):
        seg = series.loc[ws:we].dropna()
        out[wk] = {"metrics": compute_metrics(seg, rf, _ANN),
                   "calendar": calendar_year_returns(seg)}
    return out


def relative_signal_report(price_fn=None, lmp_fn=None,
                           signal_start: str = config.REGIME_SIGNAL_START,
                           primary=config.REGIME_PRIMARY_WINDOW,
                           prior=config.REGIME_PRIOR_WINDOW,
                           thresh: float = config.REGIME_THRESH,
                           zscore_window: int = config.REGIME_ZSCORE_WINDOW,
                           zscore_minp: int = config.REGIME_ZSCORE_MINP,
                           winsor: float = config.REGIME_ZSCORE_WINSOR,
                           month_lookback: int = config.REGIME_MONTH_LOOKBACK) -> dict:
    """Option A: rung-1 overlay driven by the DC-minus-rest-of-PJM congestion
    spread instead of the absolute DC-zone composite. Reports baselines, the
    current ABSOLUTE rung-1, and the RELATIVE version on both windows, with the
    frozen gate and a threshold plateau."""
    price_fn = price_fn or overlay._default_price_fn
    rf = config.RISK_FREE_RATE
    ret, lvl = overlay._basket_series(prior[0], primary[1], price_fn)

    bh = _eval_windows(ret, primary, prior, rf)
    vt = _eval_windows(overlay.apply_overlay(ret, lvl, rf, ma_days=10 ** 9), primary, prior, rf)
    l1 = _eval_windows(overlay.apply_overlay(ret, lvl, rf), primary, prior, rf)

    abs_comp = regime_composite(signal_start, primary[1], zones=list(config.REGIME_DC_ZONES),
                                w_cong=1.0, w_reserve=0.0, zone_weight="equal",
                                zscore_window=zscore_window, zscore_minp=zscore_minp,
                                winsor=winsor, lmp_fn=(lambda z, s, e: lmp_fn(s, e)) if lmp_fn else None)
    rel_comp = relative_regime_composite(signal_start, primary[1],
                                         dc_zones=list(config.REGIME_DC_ZONES),
                                         zscore_window=zscore_window, zscore_minp=zscore_minp,
                                         winsor=winsor, lmp_fn=lmp_fn)

    def _run(comp, th):
        m = regime_multiplier(comp, mode="discrete", thresh=th, month_lookback=month_lookback)
        return _eval_windows(overlay.apply_overlay_l2(ret, m, rf), primary, prior, rf), m

    out = {"windows": {"primary": tuple(primary), "prior": tuple(prior)},
           "baselines": {"buy_and_hold": bh, "vol_target_only": vt, "layer1_only": l1}}
    for key, comp in (("absolute", abs_comp), ("relative", rel_comp)):
        block, m = _run(comp, thresh)
        gate = gate_check(block["primary"], block["prior"],
                          l1["primary"], l1["prior"], bh["primary"], bh["prior"])
        nb = []
        for th in (0.25, 0.75):
            nblk, _ = _run(comp, th)
            ng = gate_check(nblk["primary"], nblk["prior"], l1["primary"], l1["prior"],
                            bh["primary"], bh["prior"])
            nb.append(bool(ng["G1"] and ng["G2"]))
        out[key] = {"block": block, "gate": gate, "neighbours_pass": nb,
                    "mechanics": _mechanics(m, ret)}

    rg = out["relative"]["gate"]
    rel_dd_p = out["relative"]["block"]["prior"]["metrics"]["max_dd"]
    abs_dd_p = out["absolute"]["block"]["prior"]["metrics"]["max_dd"]
    out["fixes_prior_dd"] = bool(rel_dd_p >= abs_dd_p)          # less negative = shallower
    out["verdict"] = final_verdict(rg, out["relative"]["neighbours_pass"])
    return out


def relative_signal_table(rep: dict) -> str:
    def row(lab, blk):
        p, q = blk["primary"]["metrics"], blk["prior"]["metrics"]
        return (f"  {lab:<26}{_pct(p['cagr']):>8}{_f(p['sharpe']):>7}{_calmar_s(p):>7}{_pct(p['max_dd']):>8}   |"
                f"{_pct(q['cagr']):>8}{_f(q['sharpe']):>7}{_calmar_s(q):>7}{_pct(q['max_dd']):>8}")
    L = ["OPTION A -- DC-zone congestion RELATIVE to rest of PJM",
         f"  primary {rep['windows']['primary'][0]}..{rep['windows']['primary'][1]}   "
         f"prior {rep['windows']['prior'][0]}..{rep['windows']['prior'][1]}", "",
         f"  {'':<26}{'CAGR':>8}{'Shrp':>7}{'Clmr':>7}{'MaxDD':>8}   |{'CAGR':>8}{'Shrp':>7}{'Clmr':>7}{'MaxDD':>8}"]
    for k, lab in [("buy_and_hold", "buy & hold"), ("vol_target_only", "vol-target only"),
                   ("layer1_only", "layer-1 only")]:
        L.append(row(lab, rep["baselines"][k]))
    L.append("")
    for k in ("absolute", "relative"):
        e = rep[k]
        L.append(row(f"rung-1 ({k})", e["block"]))
        g = e["gate"]
        L.append(f"  {'':<26}G1 {_b(g['G1'])}  G2 {_b(g['G2'])}  G3 {_b(g['G3'])}  "
                 f"plateau {sum(e['neighbours_pass'])}/{len(e['neighbours_pass'])}   "
                 f"[avg x{e['mechanics']['avg_mult']}, back {e['mechanics']['pct_stepped_back']*100:.0f}%]")
    L += ["", f"  fixes prior-window drawdown vs absolute: {rep['fixes_prior_dd']}",
          f"  verdict: {rep['verdict']}"]
    return "\n".join(L)


# ── daily -> monthly exposure multiplier ─────────────────────────────────────

def regime_multiplier(composite: pd.Series, *, mode: str = "discrete",
                      thresh: float = config.REGIME_THRESH,
                      hi: float = config.REGIME_HI, lo: float = config.REGIME_LO,
                      k: float = config.REGIME_K,
                      month_lookback: int = config.REGIME_MONTH_LOOKBACK) -> pd.Series:
    """Daily exposure multiplier (spec s4.2 steps 6-9): smooth the composite with
    a trailing ``month_lookback``-day mean, map to a multiplier, then hold each
    month-end's verdict through the *following* calendar month (decision known at
    the prior month-end). Warm-up / missing months -> 1.0 (neutral, invested)."""
    comp = composite.astype(float).sort_index()
    if comp.empty:
        return pd.Series(dtype=float, name="regime_mult")
    smoothed = comp.rolling(month_lookback, min_periods=1).mean()

    if mode == "continuous":
        raw = (1.0 + k * smoothed).clip(0.5, 1.5)
    elif mode == "discrete":
        raw = pd.Series(1.0, index=smoothed.index)
        raw[smoothed >= thresh] = hi
        raw[smoothed <= -thresh] = lo
        raw[smoothed.isna()] = np.nan
    else:
        raise ValueError(f"mode must be 'discrete' or 'continuous', got {mode!r}")

    return _month_hold(raw, comp.index, fill=1.0).rename("regime_mult")


# ── the pre-registered gate (spec s6) ────────────────────────────────────────
#
# A ladder rung PASSES iff all four hold:
#   G1  beats layer-1-only on Sharpe AND Calmar, primary window (2023-2025)
#   G2  beats layer-1-only on Sharpe AND Calmar, prior window   (2018-2022)
#   G3  no bigger drag than layer 1 -- mean(BH_yr - rung_yr) <= mean(BH_yr - L1_yr)
#       on BOTH windows (BH_yr cancels: this is rung's mean annual return >= L1's)
#   G4  plateau -- every parameter neighbour also satisfies G1 and G2
# A pass where any G1/G2 gap is < GATE_MARGIN is reported "marginal" and does
# NOT stop the ladder; only a non-marginal plateau-pass stops it.

GATE_MARGIN: float = 0.05


def _calmar(m: dict) -> float:
    dd = abs(float(m.get("max_dd", float("nan"))))
    if dd == 0 or dd != dd:
        return float("inf") if float(m.get("cagr", 0.0)) > 0 else float("nan")
    return float(m["cagr"]) / dd


def _mean_annual(block: dict) -> float:
    cal = block.get("calendar")
    if cal is None or len(cal) == 0:
        return float("nan")
    return float(pd.Series(cal).mean())


def _beats(rung: dict, l1: dict) -> tuple[bool, float, float]:
    """(rung beats l1 on Sharpe AND Calmar, sharpe_gap, calmar_gap)."""
    s_gap = float(rung["metrics"]["sharpe"]) - float(l1["metrics"]["sharpe"])
    c_gap = _calmar(rung["metrics"]) - _calmar(l1["metrics"])
    return (s_gap > 0 and c_gap > 0), s_gap, c_gap


def gate_check(rung_primary: dict, rung_prior: dict,
               l1_primary: dict, l1_prior: dict,
               bh_primary: dict, bh_prior: dict) -> dict:
    """G1-G3 + the marginal flag for one ladder rung. G4 (plateau) is folded in
    later by `final_verdict` from the neighbour runs."""
    g1, s1, c1 = _beats(rung_primary, l1_primary)
    g2, s2, c2 = _beats(rung_prior, l1_prior)

    # G3: mean(BH_yr - rung_yr) <= mean(BH_yr - L1_yr) on both windows.
    def _no_worse_drag(rung, l1, bh):
        return (_mean_annual(bh) - _mean_annual(rung)) <= (_mean_annual(bh) - _mean_annual(l1)) + 1e-12
    g3 = _no_worse_drag(rung_primary, l1_primary, bh_primary) and \
         _no_worse_drag(rung_prior, l1_prior, bh_prior)

    gaps = {"sharpe_primary": s1, "calmar_primary": c1,
            "sharpe_prior": s2, "calmar_prior": c2}
    passing_gaps = [g for k, g in gaps.items()
                    if (k.endswith("primary") and g1) or (k.endswith("prior") and g2)]
    marginal = any(0 < g < GATE_MARGIN for g in passing_gaps)

    return {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3),
            "gaps": gaps, "marginal": bool(marginal)}


def final_verdict(gate: dict, neighbours_pass: list[bool]) -> str:
    """Fold the plateau requirement (G4) and the marginal flag into a verdict:
    'FAIL' if any of G1-G3 is false; 'knife-edge' if a neighbour fails G1/G2;
    'marginal' if the gaps are tiny; else 'PASS' (this rung stops the ladder)."""
    if not (gate["G1"] and gate["G2"] and gate["G3"]):
        return "FAIL"
    if not all(neighbours_pass):
        return "knife-edge"
    if gate.get("marginal"):
        return "marginal"
    return "PASS"


# ── ERCOT West spread proxy (rung 6 sub-signal) ─────────────────────────────

def _default_ercot_fn(start, end):
    from grid_resilience.data.grid_data import fetch_lmp
    return fetch_lmp("ERCOT", start, end, use_cache=True)


def _ercot_west_spread_z(start: str, end: str, *, window: int, min_periods: int,
                         winsor: float, ercot_fn=None) -> pd.Series:
    """Daily |LZ_WEST - LZ_NORTH| price spread as a congestion proxy (ERCOT LMP
    has no cached component breakdown), normalised and trailing-z-scored. Empty
    series when the two zones aren't both present."""
    ercot_fn = ercot_fn or _default_ercot_fn
    df = ercot_fn(start, end)
    if df is None or df.empty or "location" not in df.columns:
        return pd.Series(dtype=float)
    df = df[df["location"].isin(["LZ_WEST", "LZ_NORTH"])].copy()
    df["_date"] = pd.to_datetime(df["time"]).dt.normalize()
    piv = df.pivot_table(index="_date", columns="location", values="lmp", aggfunc="mean")
    if not {"LZ_WEST", "LZ_NORTH"} <= set(piv.columns):
        return pd.Series(dtype=float)
    denom = piv[["LZ_WEST", "LZ_NORTH"]].abs().mean(axis=1) + 1e-6
    spread_frac = (piv["LZ_WEST"] - piv["LZ_NORTH"]).abs() / denom
    spread_frac.index.name = "date"
    return _trailing_zscore(spread_frac, window, min_periods, winsor)


# ── the pre-registered ladder runner (spec s5, s9) ─────────────────────────

def _rung_composite(rung: dict, start: str, end: str, composite_fn, ercot_fn,
                    zscore_window: int, zscore_minp: int, winsor: float) -> pd.Series:
    pjm = composite_fn(start, end, zones=list(rung["zones"]),
                       w_cong=rung["w_cong"], w_reserve=rung["w_reserve"],
                       zone_weight=rung.get("zone_weight", "equal"))
    if not rung.get("ercot_west"):
        return pjm
    ercot = _ercot_west_spread_z(start, end, window=zscore_window,
                                 min_periods=zscore_minp, winsor=winsor, ercot_fn=ercot_fn)
    if ercot.empty:
        warnings.warn("grid_regime rung 6: no ERCOT LZ_WEST/LZ_NORTH data; "
                      "falling back to the PJM composite alone.", RuntimeWarning, stacklevel=2)
        return pjm
    return _combine_subsignals({"pjm": (pjm, 0.5), "ercot": (ercot, 0.5)})


def _rung_multiplier(rung: dict, composite: pd.Series,
                     month_lookback: int) -> pd.Series:
    return regime_multiplier(composite, mode=rung["mode"],
                             thresh=rung.get("thresh", config.REGIME_THRESH),
                             hi=config.REGIME_HI, lo=config.REGIME_LO,
                             k=rung.get("k", config.REGIME_K),
                             month_lookback=month_lookback)


def _mechanics(mult: pd.Series, ret: pd.Series) -> dict:
    m = mult.reindex(ret.index).astype(float).fillna(1.0)
    return {"avg_mult": round(float(m.mean()), 3),
            "min_mult": round(float(m.min()), 3),
            "max_mult": round(float(m.max()), 3),
            "pct_leaned_in": round(float((m > 1.0).mean()), 3),
            "pct_stepped_back": round(float((m < 1.0).mean()), 3),
            "flips": int((m.diff().abs() > 1e-9).sum())}


def regime_report(price_fn=None, composite_fn=None, ercot_fn=None,
                  signal_start: str = config.REGIME_SIGNAL_START,
                  primary: tuple[str, str] = config.REGIME_PRIMARY_WINDOW,
                  prior: tuple[str, str] = config.REGIME_PRIOR_WINDOW,
                  zscore_window: int = config.REGIME_ZSCORE_WINDOW,
                  zscore_minp: int = config.REGIME_ZSCORE_MINP,
                  winsor: float = config.REGIME_ZSCORE_WINSOR,
                  month_lookback: int = config.REGIME_MONTH_LOOKBACK) -> dict:
    """Run the frozen ladder (`config.REGIME_LADDER`) top-to-bottom over both
    windows; stop at the first rung whose verdict is 'PASS'. Returns a dict of
    baselines + per-rung metrics/gate/verdict/mechanics + `stopped_at`.

    The regime signal is z-scored over `signal_start`..`primary[1]` so the
    primary window sees a fully-warm signal; basket returns are evaluated only
    on `primary` and `prior`. `price_fn` / `composite_fn` / `ercot_fn` are
    injectable for tests; defaults read the cached parquet (zero network I/O).
    """
    price_fn = price_fn or overlay._default_price_fn
    composite_fn = composite_fn or regime_composite
    rf = config.RISK_FREE_RATE
    full_end = primary[1]

    ret, lvl = overlay._basket_series(prior[0], full_end, price_fn)
    windows = {"primary": tuple(primary), "prior": tuple(prior)}

    def _block(series: pd.Series) -> dict:
        out = {}
        for wk, (ws, we) in windows.items():
            seg = series.loc[ws:we].dropna()
            out[wk] = {"metrics": compute_metrics(seg, rf, _ANN),
                       "calendar": calendar_year_returns(seg)}
        return out

    bh = _block(ret)
    vt_only = _block(overlay.apply_overlay(ret, lvl, rf, ma_days=10 ** 9))
    l1 = _block(overlay.apply_overlay(ret, lvl, rf))

    def _eval(rung: dict) -> tuple[dict, pd.Series]:
        comp = _rung_composite(rung, signal_start, full_end, composite_fn, ercot_fn,
                               zscore_window, zscore_minp, winsor)
        mult = _rung_multiplier(rung, comp, month_lookback)
        return _block(overlay.apply_overlay_l2(ret, mult, rf)), mult

    rungs_out: list[dict] = []
    stopped_at = None
    for rung in config.REGIME_LADDER:
        if rung.get("deferred"):
            rungs_out.append({"name": rung["name"], "status": "deferred",
                              "reason": rung.get("reason", "")})
            continue

        block, mult = _eval(rung)
        gate = gate_check(block["primary"], block["prior"],
                          l1["primary"], l1["prior"], bh["primary"], bh["prior"])
        nb_pass = []
        for override in rung.get("neighbours", []):
            nblock, _ = _eval({**rung, **override})
            ng = gate_check(nblock["primary"], nblock["prior"],
                            l1["primary"], l1["prior"], bh["primary"], bh["prior"])
            nb_pass.append(bool(ng["G1"] and ng["G2"]))
        verdict = final_verdict(gate, nb_pass or [True])

        rungs_out.append({"name": rung["name"], "params": {k: rung[k] for k in rung
                                                           if k not in ("neighbours",)},
                          "block": block, "gate": gate, "neighbours_pass": nb_pass,
                          "verdict": verdict, "mechanics": _mechanics(mult, ret),
                          "multiplier": mult.reindex(ret.index).astype(float).fillna(1.0)})
        if verdict == "PASS":
            stopped_at = rung["name"]
            break

    return {"signal_start": signal_start, "windows": windows,
            "baselines": {"buy_and_hold": bh, "vol_target_only": vt_only, "layer1_only": l1},
            "rungs": rungs_out, "stopped_at": stopped_at}


# ── the shipped live overlay (ladder rung 1, adopted 2026-08-31) ────────────

def shipped_config() -> dict:
    """The frozen live layer-2 config = ladder rung 1: discrete 3-state, the
    four DC-heavy PJM zones (DOM/AEP/COMED/PPL), congestion component only.
    Adopted as the recommended overlay by PM decision despite the strict gate
    failing on primary-window Calmar -- see docs/grid-regime-layer2-results.md."""
    cfg = dict(config.REGIME_LADDER[0])
    cfg.pop("neighbours", None)
    return cfg


def live_multiplier(signal_start: str = config.REGIME_SIGNAL_START,
                    end: str | None = None, *, composite_fn=None,
                    zscore_window: int = config.REGIME_ZSCORE_WINDOW,
                    zscore_minp: int = config.REGIME_ZSCORE_MINP,
                    winsor: float = config.REGIME_ZSCORE_WINSOR,
                    month_lookback: int = config.REGIME_MONTH_LOOKBACK) -> pd.Series:
    """Daily exposure multiplier for the shipped layer-2 overlay (rung 1).

    Compose it with the layer-1 vol target via
    ``overlay.apply_overlay_l2(basket_returns, grid_regime.live_multiplier(...))``.
    The signal is z-scored from ``signal_start`` (long warm-up); ``end`` defaults
    to the primary-window end. ``composite_fn`` is injectable for tests.
    """
    end = end or config.REGIME_PRIMARY_WINDOW[1]
    cfg = shipped_config()
    comp = (composite_fn or regime_composite)(
        signal_start, end, zones=list(cfg["zones"]),
        w_cong=cfg["w_cong"], w_reserve=cfg["w_reserve"],
        zone_weight=cfg.get("zone_weight", "equal"),
        zscore_window=zscore_window, zscore_minp=zscore_minp, winsor=winsor)
    return _rung_multiplier(cfg, comp, month_lookback)


_REGIME_CAVEAT = (
    "Regime honesty: ~36 primary / ~24 active-prior monthly obs, one macro cycle. "
    "A trailing z-score flags the congestion *transition* then decays as the new "
    "level becomes the norm. Prior window is effectively 2021-2022 after warm-up. "
    "No rung passing => ship layer-1-only or a smaller un-overlaid basket."
)


def regime_table(report: dict) -> str:
    def _row(label: str, blk: dict) -> str:
        p, q = blk["primary"]["metrics"], blk["prior"]["metrics"]
        return (f"  {label:<34}"
                f"{_pct(p['cagr']):>8}{_f(p['sharpe']):>7}{_calmar_s(p):>7}{_pct(p['max_dd']):>8}   |"
                f"{_pct(q['cagr']):>8}{_f(q['sharpe']):>7}{_calmar_s(q):>7}{_pct(q['max_dd']):>8}")

    L = [f"LAYER-2 GRID-REGIME LADDER   signal z-scored from {report['signal_start']}",
         f"  primary {report['windows']['primary'][0]}..{report['windows']['primary'][1]}   "
         f"prior {report['windows']['prior'][0]}..{report['windows']['prior'][1]}",
         "",
         f"  {'':<34}{'CAGR':>8}{'Shrp':>7}{'Clmr':>7}{'MaxDD':>8}   |{'CAGR':>8}{'Shrp':>7}{'Clmr':>7}{'MaxDD':>8}"]
    for key, lab in [("buy_and_hold", "buy & hold"), ("vol_target_only", "vol-target only"),
                     ("layer1_only", "layer-1 only (trend gate + VT)")]:
        L.append(_row(lab, report["baselines"][key]))
    L.append("")
    for r in report["rungs"]:
        if r.get("status") == "deferred":
            L.append(f"  {r['name']:<34}DEFERRED -- {r['reason']}")
            continue
        L.append(_row(r["name"], r["block"]))
        g = r["gate"]
        me = r["mechanics"]
        L.append(f"  {'':<34}G1 {_b(g['G1'])}  G2 {_b(g['G2'])}  G3 {_b(g['G3'])}  "
                 f"plateau {sum(r['neighbours_pass'])}/{len(r['neighbours_pass'])}  "
                 f"-> {r['verdict']}   [avg x{me['avg_mult']}, in {me['pct_leaned_in']*100:.0f}%/"
                 f"back {me['pct_stepped_back']*100:.0f}%, {me['flips']} flips]")
    L.append("")
    L.append(f"  stopped at: {report['stopped_at'] or 'NONE -- no rung passed the gate'}")
    L.append("  " + _REGIME_CAVEAT)
    return "\n".join(L)


def _pct(x) -> str:
    return "n/a" if x != x else f"{x * 100:.1f}%"


def _f(x) -> str:
    return "n/a" if x != x else f"{x:.2f}"


def _calmar_s(m: dict) -> str:
    c = _calmar(m)
    return "n/a" if c != c else (">9" if c == float("inf") else f"{c:.2f}")


def _b(v: bool) -> str:
    return "PASS" if v else "fail"
