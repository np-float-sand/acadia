"""THROWAWAY (2026-09-06) -- open item #15: regulatory-calendar DER short trigger.
Does shorting the resi-solar sleeve in a window around scheduled ADVERSE net-metering
decisions add value?  (a) event study, (b) conditional-short overlay on `screen +val+sleeve`.
n is tiny (CA NEM 3.0 dominates); this is a feasibility read, not a validated signal."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket.backtest import compute_metrics
from electrification_strategy import config, universe, backtest

START, END = "2017-06-01", "2026-08-31"; RF, AF = 0.04, 252
FAIR = ("2025-01-01", "2026-08-31")
DER = ["ENPH", "SEDG", "RUN", "NOVA"]           # resi-solar, NEM-exposed

cal = pd.read_csv("electrification_strategy/data/nem_calendar.csv", comment="#", parse_dates=["date"])
adverse = cal[(cal.direction == "cut") & (cal.date >= "2017-06-01")].sort_values("date")
print("Adverse NEM milestones in-window:")
for _, r in adverse.iterrows():
    print(f"  {r.date.date()}  {r.state:3s} {r.severity:5s}  {r.proceeding} [{r.milestone}]")

tk = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index) | {"SPY", "GLD", "IEF", "TAN"} | set(DER))
px = fetch_prices(tk, START, END)
spy = px["SPY"].pct_change()
cand = px[[c for c in tk if c in px.columns]]
base_r = backtest.basket_returns(cand, "screen", START, END); IDX = base_r.index
gld_ief = px[["GLD", "IEF"]].pct_change().mean(axis=1).reindex(IDX).fillna(0.0)
ship = 0.85 * backtest.compose(base_r, spy, use_valuation=True) + 0.15 * gld_ief
plain = backtest.compose(base_r, spy, use_valuation=False)
der = px[[c for c in DER if c in px.columns]].pct_change().mean(axis=1).reindex(IDX).fillna(0.0)
lb_m = (1 + plain).resample("ME").prod() - 1
tan_m = (1 + px["TAN"].pct_change().reindex(IDX)).resample("ME").prod() - 1

# ---- (a) event study: DER and DER-minus-longbook cum return around each adverse milestone ----
def cum(s, a, b):
    x = s.loc[a:b]; return float((1 + x).prod() - 1)
print("\n=== EVENT STUDY (adverse milestones) ===")
print(f"{'date':12s}{'sev':6s}{'DER -90..-1d':>13}{'DER 0..+90d':>13}{'DER -90..+90':>13}{'(DER-book) win':>15}")
rows = []
for _, r in adverse.iterrows():
    d = r.date
    pre  = cum(der, d - pd.Timedelta(days=90), d - pd.Timedelta(days=1))
    post = cum(der, d, d + pd.Timedelta(days=90))
    span = cum(der, d - pd.Timedelta(days=90), d + pd.Timedelta(days=90))
    book_span = cum(plain, d - pd.Timedelta(days=90), d + pd.Timedelta(days=90))
    rows.append((d, r.severity, pre, post, span, span - book_span))
    print(f"{str(d.date()):12s}{r.severity:6s}{pre*100:>12.1f}%{post*100:>12.1f}%{span*100:>12.1f}%{(span-book_span)*100:>14.1f}%")
maj = [x for x in rows if x[1] == "major"]
print(f"\nmean (all {len(rows)}):   DER span {np.mean([x[4] for x in rows])*100:+.1f}% rel-book  |  "
      f"major only ({len(maj)}): {np.mean([x[4] for x in maj])*100:+.1f}%")

# ---- (b) overlay: short DER during 'adverse regulatory windows' [milestone-120d, milestone+45d] ----
win = pd.Series(False, index=IDX)
for _, r in adverse.iterrows():
    lo, hi = r.date - pd.Timedelta(days=120), r.date + pd.Timedelta(days=45)
    win.loc[(IDX >= lo) & (IDX <= hi)] = True
print(f"\nadverse-window mask on {win.mean()*100:.0f}% of days "
      f"({win.groupby(IDX.year).any().sum()} distinct years: {sorted(set(IDX.year[win]))})")

def stat(label, rser):
    def M(s, a=None, b=None): return compute_metrics(s.loc[a:b].dropna() if a else s.dropna(), RF, AF)
    m, f = M(rser), M(rser, *FAIR)
    hm = (1 + rser).resample("ME").prod() - 1
    bad = ((lb_m < 0) & (tan_m > 0)).reindex(hm.index).fillna(False)
    shm = (1 + ship).resample("ME").prod() - 1
    fear = float((hm[bad] - shm.reindex(hm.index)[bad]).sum())
    print(f"{label:<40} full Sh {m['sharpe']:.2f}  CAGR {m['cagr']*100:5.1f}%  MaxDD {m['max_dd']*100:6.1f}%"
          f"  | 19-22 Sh {M(rser,'2019-01-01','2022-12-31')['sharpe']:+.2f}"
          f"  | fair Sh {f['sharpe']:.2f}  | feared vs ship {fear*100:+.0f}%")

print("\n=== OVERLAY on screen +val+sleeve ===")
stat("shipped (no NEM short)", ship)
for w in (0.15, 0.25, 0.40):
    stat(f"  - {w:.2f}x DER during adverse NEM windows", ship - w * (der * win.astype(float)))
# contribution of the short leg alone during windows
sleeve = (-(der * win.astype(float))).dropna()
print(f"\nshort-DER-in-window sleeve, standalone: CAGR {compute_metrics(sleeve,RF,AF)['cagr']*100:+.1f}%/yr  "
      f"(positive => the timed short made money)")
