"""Does combining the marginal (|t|>=1.5) signals -- multivariate or nonlinear --
produce something that works in 2023-26 (the regime that matters)?"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0,"/Users/sandhyapersad/acadia")
import yfinance as yf, urllib.request, json
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket.data.category_demand import fetch_category_demand, momentum
from grid_equipment_basket import config
KEY=open("/Users/sandhyapersad/acadia/.env").read().split("EIA_API_KEY=")[1].split()[0]
def mc(t,s="2009-06-01"):
    x=yf.Ticker(t).history(period="max")["Close"].dropna(); x.index=x.index.tz_localize(None)
    return x.resample("ME").last().loc[s:]
def eia(sid):
    u=f"https://api.eia.gov/v2/steo/data/?frequency=monthly&data[0]=value&facets[seriesId][]={sid}&length=5000&api_key={KEY}"
    d=json.loads(urllib.request.urlopen(u,timeout=40).read())["response"]["data"]
    s=pd.Series({pd.Period(r["period"],"M").to_timestamp("M"):float(r["value"]) for r in d}).sort_index()
    return s

ngf=mc("NG=F"); unl=mc("UNL"); smh=mc("SMH").pct_change(); d10y=mc("^TNX").diff()
NAMES=sorted(config.UNIVERSE)
PX=fetch_prices(NAMES,"2016-06-01","2026-08-31"); PX=PX[[c for c in NAMES if c in PX.columns]].ffill()
bret=(1+PX.pct_change()).resample("ME").prod().mean(1)-1.0

# category-demand composite score (basket-level: exposure-weighted cat momentum, avg across names)
CD=fetch_category_demand()
def zmom(c,m):
    s=momentum(CD,c,m); return (s-s.expanding(24).mean())/s.expanding(24).std()
catscore=(0.4*zmom("ppi_switchgear",6)+0.3*zmom("ppi_transformers",6)+0.3*zmom("ee_new_orders",6))
catscore.index=catscore.index.to_period("M").to_timestamp("M"); catscore=catscore.shift(2)

# STEO forward power (latest vintage -- has mild look-ahead, flagged)
pjw=eia("ELWHU_PJ"); erc=eia("ELWHU_TX")
X=pd.DataFrame(index=pd.date_range("2015-01-31","2026-08-31",freq="ME"))
X["ng_chg12"]=np.log(ngf).reindex(X.index).diff(12)
X["ng_chg3"] =np.log(ngf).reindex(X.index).diff(3)
X["unl_ema"] =np.log(unl).reindex(X.index).diff().ewm(span=6).mean()*6
X["catscore"]=catscore.reindex(X.index)
X["pjw_fchg6"]=np.log(pjw).reindex(X.index).diff(6)     # PJM West wholesale-price 6m change (fwd-tail-contaminated)
X["erc_fchg6"]=np.log(erc).reindex(X.index).diff(6)
X["smh"]=smh.reindex(X.index); X["d10y"]=d10y.reindex(X.index)
X["y"]=bret.reindex(X.index).shift(-1)
WINS={"2019-22":("2019-01","2022-12"),"2023-26":("2023-01","2026-08")}

def ols(cols, win, ctrl=()):
    a,b=WINS[win]; sub=X.loc[a:b,list(cols)+list(ctrl)+["y"]].dropna()
    if len(sub)<15: return None
    M=np.column_stack([np.ones(len(sub))]+[sub[c].values for c in list(cols)+list(ctrl)])
    bta,*_=np.linalg.lstsq(M,sub["y"].values,rcond=None)
    res=sub["y"].values-M@bta; s2=(res@res)/(len(sub)-M.shape[1])
    cov=s2*np.linalg.inv(M.T@M); se=np.sqrt(np.diag(cov))
    r2=1-(res@res)/((sub["y"].values-sub["y"].mean())**2).sum()
    return sub, [(c,bta[i+1],bta[i+1]/se[i+1]) for i,c in enumerate(list(cols)+list(ctrl))], r2, len(sub)

print("="*82)
print("COMBINING THE MARGINAL SIGNALS -- does anything survive in 2023-26?")
print("\n[A] each signal ALONE, per window, OLS y ~ x  (coef t):")
for c in ["ng_chg12","ng_chg3","unl_ema","catscore","pjw_fchg6","erc_fchg6"]:
    r=[]
    for w in WINS:
        o=ols([c],w)
        r.append(f"{w}: t={o[1][0][2]:+.2f}(n{o[3]})" if o else f"{w}: n<15")
    print(f"  {c:12} "+"   ".join(r))

print("\n[B] MULTIVARIATE  y ~ ng_chg12 + unl_ema + catscore + pjw_fchg6  (per window):")
for w in WINS:
    o=ols(["ng_chg12","unl_ema","catscore","pjw_fchg6"],w)
    if not o: print(f"  {w}: n<15"); continue
    print(f"  {w}  R2={o[2]:+.2f}  n={o[3]}")
    for c,coef,t in o[1]: print(f"     {c:12} coef {coef:+.4f}  t={t:+.2f}")

print("\n[C] MULTIVARIATE + CONTROLS  y ~ [signals] + smh + d10y  (2023-26 only):")
o=ols(["ng_chg12","unl_ema","catscore","pjw_fchg6","erc_fchg6"],"2023-26",ctrl=["smh","d10y"])
if o:
    print(f"  R2={o[2]:+.2f}  n={o[3]}")
    for c,coef,t in o[1]: print(f"     {c:12} coef {coef:+.4f}  t={t:+.2f}")

print("\n[D] NONLINEAR checks (2023-26):")
sub=X.loc["2023-01":"2026-08"].dropna(subset=["ng_chg12","unl_ema","catscore","pjw_fchg6","y"])
# sign-agreement composite
comp=np.sign(sub["ng_chg12"])+np.sign(sub["unl_ema"])+np.sign(sub["catscore"])+np.sign(sub["pjw_fchg6"])
ic=np.corrcoef(comp.rank(),sub["y"].rank())[0,1]; print(f"  sign-agreement composite rank-IC = {ic:+.2f}  t={ic*np.sqrt(len(sub)-1):+.2f}  n={len(sub)}")
# interaction: gas-momentum x category-score
inter=sub["ng_chg12"]*sub["catscore"]
ic2=np.corrcoef(inter.rank(),sub["y"].rank())[0,1]; print(f"  ng_chg12 x catscore interaction rank-IC = {ic2:+.2f}  t={ic2*np.sqrt(len(sub)-1):+.2f}")
# quadratic in the best 2019-22 signal
ic3=np.corrcoef((sub["ng_chg12"]**2).rank(),sub["y"].rank())[0,1]; print(f"  ng_chg12^2 rank-IC = {ic3:+.2f}  t={ic3*np.sqrt(len(sub)-1):+.2f}")
# regime dummy interaction: does 2019-22 signal work when it ALSO agrees with catscore?
agree=sub[np.sign(sub["ng_chg12"])==np.sign(sub["catscore"])]
if len(agree)>=8:
    ic4=np.corrcoef(agree["ng_chg12"].rank(),agree["y"].rank())[0,1]
    print(f"  ng_chg12 IC on months it agrees w/ catscore (n={len(agree)}) = {ic4:+.2f}  t={ic4*np.sqrt(len(agree)-1):+.2f}")
