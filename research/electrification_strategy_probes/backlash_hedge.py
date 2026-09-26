"""THROWAWAY (2026-09-06) -- can we hedge a *political/NIMBY* de-rating of the
data-center buildout (moratoriums, permit denials) with NO macro downturn?
No historical backlash episode exists in-sample, so this tests STRUCTURAL
properties + an idiosyncratic-DC-selloff proxy, not a validated trigger."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket.basket import simulate_basket
from grid_equipment_basket.backtest import compute_metrics
from grid_equipment_basket import hedges
START, END = "2017-06-01", "2026-08-31"; RF, AF = 0.04, 252

DC_POWER   = ["VRT", "GEV"]                                   # greenfield hyperscaler DC power
GRID_MAINT = ["ETN","HUBB","PWR","MYRG","PRIM","EMR","AME","RRX","POWL","ATKR","AEIS"]  # T&D / utility / EPC / broad electrification
MERCHANT   = ["VST","NRG","CEG","TLN"]                        # merchant power / IPP
THEMATIC   = ["ETN","HUBB","GEV","VRT","PWR","NVT","POWL","AEIS","GNRC","AME"]  # the shipped winner universe

px = fetch_prices(sorted(set(DC_POWER+GRID_MAINT+MERCHANT+THEMATIC+["SPY","DLR","EQIX","GLD","IEF","TAN"])), START, END)

def vt(r, t=0.20, lb=21, c=1.5):
    r=r.dropna(); rv=r.rolling(lb).std()*np.sqrt(AF); lev=(t/rv).clip(upper=c).shift(1)
    return (lev*r+(1-lev)*RF/AF).dropna()
def ew(tickers):
    cols=[t for t in tickers if t in px.columns]
    return simulate_basket(px[cols], START, END, 42, 1.0).returns
def M(r): m=compute_metrics(r,RF,AF); return f"CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}  MaxDD {m['max_dd']*100:6.1f}%"
def mcorr(a,b):
    x=(1+a).resample("ME").prod()-1; y=(1+b).resample("ME").prod()-1
    d=pd.concat([x,y],axis=1).dropna(); return d.iloc[:,0].corr(d.iloc[:,1])
def beta(a,b):
    d=pd.concat([a.rename('a'),b.rename('b')],axis=1).dropna(); return d['a'].cov(d['b'])/d['b'].var()

dcp, grd, mer = ew(DC_POWER), ew(GRID_MAINT), ew(MERCHANT)
spy = px["SPY"].pct_change(); dcreit = ew(["DLR","EQIX"]); tan = px["TAN"].pct_change()

print("=== 1. structural exposure to the DC-power narrative (monthly corr / beta) ===")
print(f"  DC-power sleeve   : {M(vt(dcp))}")
print(f"  grid-maint sleeve : {M(vt(grd))}")
print(f"  merchant sleeve   : {M(vt(mer))}")
print(f"  corr(grid-maint, VRT)      {mcorr(grd, px['VRT'].pct_change()):+.2f}   vs corr(DC-power, VRT) {mcorr(dcp, px['VRT'].pct_change()):+.2f}")
print(f"  corr(grid-maint, DC-REITs) {mcorr(grd, dcreit):+.2f}   vs corr(DC-power, DC-REITs) {mcorr(dcp, dcreit):+.2f}")
print(f"  beta(grid-maint -> DC-power sleeve) {beta(grd, dcp):.2f}   beta(merchant -> DC-power sleeve) {beta(mer, dcp):.2f}")

print("\n=== 2. idiosyncratic-DC-selloff proxy: worst 12 months for the DC-power sleeve where SPY was flat/up (>= -1%) ===")
dcp_m=(1+dcp).resample("ME").prod()-1; spy_m=(1+spy).resample("ME").prod()-1
grd_m=(1+grd).resample("ME").prod()-1; mer_m=(1+mer).resample("ME").prod()-1
idio = dcp_m[(spy_m>=-0.01)].nsmallest(12)
print("  month      DCpower  gridMaint  merchant   SPY")
for dt,v in idio.items():
    print(f"  {dt.date()}  {v*100:+6.1f}  {grd_m.get(dt,np.nan)*100:+7.1f}  {mer_m.get(dt,np.nan)*100:+7.1f}  {spy_m.get(dt,np.nan)*100:+5.1f}")
print(f"  MEAN        {idio.mean()*100:+6.1f}  {grd_m.reindex(idio.index).mean()*100:+7.1f}  {mer_m.reindex(idio.index).mean()*100:+7.1f}  {spy_m.reindex(idio.index).mean()*100:+5.1f}")

print("\n=== 3. winner book: thematic EW  vs  70/30 grid-maint/DC-power tilt (both vol-targeted + 15% GLD/IEF sleeve) ===")
sleeve = ew(["GLD","IEF"])
base_ew  = vt(ew(THEMATIC))
tilt_raw = 0.70*grd.reindex(base_ew.index).fillna(0) + 0.30*dcp.reindex(base_ew.index).fillna(0)
base_tilt= vt(tilt_raw)
for nm, b in [("thematic EW", base_ew), ("70/30 grid-tilt", base_tilt)]:
    book = 0.85*b + 0.15*sleeve.reindex(b.index).fillna(0)
    eps = {k: hedges.episode_drawdown(book, *hedges.find_drawdown_episode(book, pw, te))
           for k,(pw,te) in {"Rate22":(("2021-11-01","2022-01-31"),"2022-12-31"),
                             "DeepSeek":(("2024-11-01","2025-01-15"),"2025-05-31"),
                             "Selloff26":(("2026-05-01","2026-06-20"),"2026-08-31")}.items()}
    print(f"  {nm:<16} {M(book)}  | " + "  ".join(f"{k} {v*100:.0f}%" for k,v in eps.items()))

print("\n=== 4. conditional merchant-power short on the winner, triggered by DC-power UNDERPERFORMING grid-maint over 126d ===")
book = (0.85*base_ew + 0.15*sleeve.reindex(base_ew.index).fillna(0))
dcp_lvl=(1+dcp).cumprod(); grd_lvl=(1+grd).cumprod()
trig_d = (dcp_lvl.pct_change(126) < grd_lvl.pct_change(126))
# month-hold the trigger
per = book.index.to_period("M")
me = trig_d.reindex(book.index).ffill().fillna(False).groupby(per).last()
trig = pd.Series(me.shift(1).reindex(per).to_numpy(), index=book.index).fillna(False).astype(bool)
mer_al = mer.reindex(book.index).fillna(0.0)
for w in (0.0, 0.15, 0.25, 0.40):
    h = book - w*(mer_al*trig.astype(float))
    carry = compute_metrics((-mer_al*trig.astype(float)).dropna(),RF,AF)["cagr"]
    print(f"  short {w:.2f}x merchant when triggered ({trig.mean()*100:.0f}% of days): {M(h)}   sleeve carry {carry*100:+.1f}%/yr")
