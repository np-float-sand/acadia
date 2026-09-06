"""VA/Dominion transmission-project probe -- checks (a) DC-$ fraction over time,
(b) lead/lag vs the buildout basket, (c) per-region and per-work-type granularity
for name weighting. Throwaway analysis for the go/no-go."""
from __future__ import annotations
import sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sandhyapersad/acadia")

df = pd.read_csv("va_transmission_projects.csv")
df["filed_year"] = pd.to_datetime(df["filed_date"]).dt.year
df["approved_year"] = pd.to_datetime(df["approved_date"]).dt.year
df["filed_q"] = pd.to_datetime(df["filed_date"]).dt.to_period("Q")
df["cost"] = df["cost_usd_m"].astype(float)
active = df[df["disposition"] != "Canceled"].copy()

print("=" * 78)
print(f"POPULATION: {len(df)} Dominion VA transmission CPCN cases, filed "
      f"{df.filed_year.min()}-{df.filed_year.max()}  ({len(active)} ex-canceled)")
print(f"Total project-$ (ex-canceled): ${active.cost.sum()/1000:.1f}B  "
      f"median ${active.cost.median():.0f}M")
print(f"driver mix (count): {active.driver.value_counts().to_dict()}")
print(f"driver mix ($M):    "
      + ", ".join(f"{k}=${v/1000:.2f}B" for k, v in active.groupby('driver').cost.sum().items()))

# ---------- CHECK (a): DC-$ fraction over time ----------
def year_table(frame, ycol):
    g = frame.groupby(ycol)
    out = pd.DataFrame({
        "n": g.size(),
        "tot_$M": g.cost.sum().round(0),
        "dc_strict_$M": g.apply(lambda x: x.loc[x.dc_strict == 1, "cost"].sum()).round(0),
        "dc_broad_$M": g.apply(lambda x: x.loc[x.dc_broad == 1, "cost"].sum()).round(0),
    })
    out["frac_strict"] = (out["dc_strict_$M"] / out["tot_$M"]).round(3)
    out["frac_broad"] = (out["dc_broad_$M"] / out["tot_$M"]).round(3)
    out["dc_strict_n"] = g.apply(lambda x: int((x.dc_strict == 1).sum()))
    return out

print("\n" + "=" * 78)
print("CHECK (a) -- DC-driven project-$ FRACTION over time")
print("\n--- by FILED year (application year; ex-canceled) ---")
fa = year_table(active, "filed_year")
print(fa.to_string())
print("\n--- by APPROVED year (final-order year; ex-canceled, approved only) ---")
ap = year_table(active[active.approved_year.notna()], "approved_year")
print(ap.to_string())

# trend test on filed-year fraction
for lbl, col in [("strict", "frac_strict"), ("broad", "frac_broad")]:
    s = fa[col].dropna()
    x = s.index.values.astype(float); y = s.values
    b = np.polyfit(x, y, 1)[0]
    r = np.corrcoef(x, y)[0, 1]
    print(f"  filed-year {lbl:6} fraction: 2019={s.iloc[0]:.2f} -> 2026={s.iloc[-1]:.2f}  "
          f"slope={b*100:+.1f} pp/yr  corr(t)={r:+.2f}")

# ---------- basket series ----------
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket import config
from grid_equipment_basket.basket import simulate_basket

px = fetch_prices(sorted(config.UNIVERSE), "2018-06-01", "2026-08-31")
cols = [t for t in config.UNIVERSE if t in px.columns]
br = simulate_basket(px[cols], "2018-06-01", "2026-08-31",
                     config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT, None)
bret = br.returns
cal = (1 + bret).groupby(bret.index.year).prod() - 1.0     # calendar-year total return
qret = (1 + bret).groupby(bret.index.to_period("Q")).prod() - 1.0

print("\n" + "=" * 78)
print("CHECK (b) -- lead/lag vs the 9-name buildout basket")
print("basket calendar-year total return:")
print("  " + "  ".join(f"{y}:{v*100:+.0f}%" for y, v in cal.items()))

# annual series: filed-year DC-$ (strict & broad), total-$, and fraction
ann = fa.reindex(range(2019, 2027))
series = {
    "dc_strict_$": ann["dc_strict_$M"].fillna(0),
    "dc_broad_$": ann["dc_broad_$M"].fillna(0),
    "total_$": ann["tot_$M"].fillna(0),
    "frac_broad": ann["frac_broad"],
}
print("\nlag corr(  filed-year metric ,  basket cal-year return shifted by k )   n≈7")
print(f"  {'metric':12} " + "".join(f"k={k:+d}  " for k in (-2, -1, 0, 1, 2)))
for name, s in series.items():
    row = []
    for k in (-2, -1, 0, 1, 2):
        a = s.copy()
        b = cal.reindex(s.index).shift(-k)      # k>0 => basket LEADS metric ; k<0 => metric LEADS basket
        m = a.notna() & b.notna()
        row.append(np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 2 else np.nan)
    print(f"  {name:12} " + "".join(f"{v:+.2f}  " for v in row))
print("  (k<0 = filings LEAD the basket; k>0 = basket LEADS filings)")

# growth-vs-return, YoY
print("\nYoY: filed-year DC-$ (broad) growth  vs  basket return that year / next year")
g = ann["dc_broad_$M"].fillna(0)
yoy = g.pct_change().replace([np.inf, -np.inf], np.nan)
comp = pd.DataFrame({"dc_broad_$M": g, "yoy_%": (yoy*100).round(0),
                     "basket_same_yr_%": (cal.reindex(g.index)*100).round(0),
                     "basket_next_yr_%": (cal.reindex(g.index).shift(-1)*100).round(0)})
print(comp.to_string())

# quarterly (filed quarter) -- more points
qg = active.groupby("filed_q").apply(lambda x: x.loc[x.dc_broad == 1, "cost"].sum())
qg = qg.reindex(pd.period_range("2019Q1", "2026Q2", freq="Q")).fillna(0)
qb = qret.reindex(qg.index)
print(f"\nquarterly (filed-Q DC-$ broad, n={qg.notna().sum()}):")
for k in (-4, -2, -1, 0, 1, 2, 4):
    a = qg; b = qb.shift(-k)
    m = a.notna() & b.notna() & (a > 0)
    c = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 3 else np.nan
    print(f"  lag k={k:+d}: corr={c:+.2f}  (n={int(m.sum())})")

# ---------- CHECK (c1): per-region ----------
print("\n" + "=" * 78)
print("CHECK (c1) -- per-SCC-region project-$ (all one TO = Dominion, ~one PJM zone DOM)")
reg = active.pivot_table(index="filed_year", columns="region", values="cost",
                         aggfunc="sum").fillna(0).round(0)
print(reg.to_string())
print("region share of total $ (all years):")
print((active.groupby("region").cost.sum() / active.cost.sum()).round(3).to_string())

# ---------- CHECK (c2): work-type mix -> supplier tilt ----------
print("\n" + "=" * 78)
print("CHECK (c2) -- work-type $ mix over time  ->  implied name tilt")
WT = ["new_line", "rebuild", "new_substation", "substation_upgrade", "transformer",
      "conversion", "reconductor", "underground", "corridor", "loop",
      "gen_interconnect"]
def wt_dollars(frame):
    rows = {}
    for _, r in frame.iterrows():
        tags = str(r["work_type"]).split("|")
        for t in tags:
            rows.setdefault(t, 0.0)
            rows[t] += r["cost"] / len(tags)      # split $ evenly across a project's tags
    return pd.Series(rows)
wt_all = wt_dollars(active).sort_values(ascending=False)
print("\ntotal $ by work-type (project $ split evenly across its tags), $M:")
print(wt_all.round(0).to_string())
wt_yr = active.groupby("filed_year").apply(wt_dollars).unstack().fillna(0)
share_yr = wt_yr.div(wt_yr.sum(axis=1), axis=0).round(3)
print("\nwork-type SHARE of $ by filed year:")
print(share_yr[["new_line", "rebuild", "new_substation", "transformer", "underground",
                "gen_interconnect"]].to_string())

# handoff work-type -> supplier map (9-name book only; ADRs/POWL omitted -- no prices here)
MAP = {
    "new_line":         {"HUBB": 1.0, "PWR": 1.0, "MYRG": 1.0, "PRIM": 1.0},
    "rebuild":          {"HUBB": 1.0, "PWR": 1.0, "MYRG": 1.0, "PRIM": 1.0},
    "reconductor":      {"HUBB": 1.0, "PWR": 1.0, "MYRG": 1.0},
    "corridor":         {"PWR": 1.0, "MYRG": 1.0, "PRIM": 1.0},
    "loop":             {"HUBB": 1.0, "PWR": 1.0, "MYRG": 1.0, "PRIM": 1.0},
    "new_substation":   {"ETN": 1.0, "HUBB": 1.0, "GEV": 0.5},
    "substation_upgrade": {"ETN": 1.0, "HUBB": 1.0},
    "transformer":      {"GEV": 1.0},
    "conversion":       {"ETN": 1.0, "HUBB": 1.0},
    "underground":      {"PRIM": 1.0, "MYRG": 1.0},
    "gen_interconnect": {"GEV": 1.0, "PWR": 1.0},
}
NAMES = sorted(config.UNIVERSE)
def tilt_from(wtser):
    w = pd.Series(0.0, index=NAMES)
    for t, d in wtser.items():
        m = MAP.get(t)
        if not m:
            continue
        tot = sum(m.values())
        for nm, wt in m.items():
            if nm in w.index:
                w[nm] += wtser[t] * wt / tot
    return (w / w.sum()).round(3)

print("\nimplied 9-name tilt from FULL-period work-type $ mix (vs equal 0.111):")
tw = tilt_from(wt_all)
print(tw.to_string())
print(f"  max overweight {tw.idxmax()} {tw.max():.3f} ; max underweight {tw.idxmin()} {tw.min():.3f}")
print(f"  active share vs equal-weight (sum |w-1/9|)/2 = {(tw - 1/9).abs().sum()/2:.3f}")

print("\nyear-by-year implied tilt (does the mix move weights enough to matter?):")
ty = wt_yr.apply(lambda r: tilt_from(r.replace(0, np.nan).dropna()), axis=1)
print(ty.round(3).to_string())
print("\nstdev of each name's yearly implied weight (higher = mix actually shifts it):")
print(ty.std().round(3).sort_values(ascending=False).to_string())
