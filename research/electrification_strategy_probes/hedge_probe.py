"""THROWAWAY (2026-09-06) -- short-leg-as-insurance probe for the electrification
long-only book.  v2: fixed-notional sizing (0.25/0.50/0.75x), always-on + conditional
variants, corrected gates.  Long book = 20%-vol-targeted grid_equipment_basket proxy."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket.basket import simulate_basket
from grid_equipment_basket.backtest import compute_metrics
from grid_equipment_basket import hedges

START, END = "2017-06-01", "2026-08-31"
RF, AF = 0.04, 252
CALM = {"pre-COVID 17-19": ("2017-06-01", "2019-12-31"), "AI-bull 23-24": ("2023-01-01", "2024-10-31")}
EPISODES = {"COVID 20": (("2020-02-01","2020-02-25"),"2020-04-30"),
            "Rate shock 22": (("2021-11-01","2022-01-31"),"2022-12-31"),
            "DeepSeek 25": (("2024-11-01","2025-01-15"),"2025-05-31"),
            "Tariff Apr-25": (("2025-03-15","2025-04-03"),"2025-04-30"),
            "Selloff 26": (("2026-05-01","2026-06-20"),"2026-08-31")}
LONGBOOK = ["ETN","HUBB","GEV","VRT","PWR","MYRG","NVT","FLNC","PRIM"]
SHORTS = {"firm_gen": ["VST","NRG","CEG","TLN"], "dc_reits": ["DLR","EQIX"],
          "der_sleeve": ["ENPH","SEDG","CHPT","RUN","BLNK","STEM"], "crowding": ["SMCI","DELL","ANET","GEV"]}
ALL = sorted(set(LONGBOOK + [t for v in SHORTS.values() for t in v] + ["SPY","TAN","GLD","IEF"]))
NOTIONALS = [0.25, 0.50, 0.75]

def vol_target(r, target=0.20, lb=21, cap=1.5, rf=RF):
    r = r.dropna(); rv = r.rolling(lb).std() * np.sqrt(AF)
    lev = (target / rv).clip(upper=cap).shift(1)
    return (lev * r + (1 - lev) * rf / AF).dropna()

def fred(sid):
    f = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2000-01-01",
                    parse_dates=[0], index_col=0)
    return pd.to_numeric(f.iloc[:, 0], errors="coerce").dropna()

def monthly_hold(daily_bool, idx):
    s = daily_bool.reindex(idx).ffill().fillna(False)
    me = s.groupby(idx.to_period("M")).last()
    return pd.Series(me.shift(1).fillna(False).reindex(idx.to_period("M")).to_numpy(), index=idx).astype(bool)

def beta(r, mkt):
    d = pd.concat([r.rename("r"), mkt.rename("m")], axis=1).dropna()
    return float(np.cov(d["r"], d["m"])[0, 1] / np.var(d["m"]))

def cagr(r): return compute_metrics(r, RF, AF)["cagr"]
def seg(r, a, b): s = r.loc[a:b]; return float((1 + s).prod() - 1)

px = fetch_prices(ALL, START, END)
lb = vol_target(simulate_basket(px[LONGBOOK], START, END, 42, 0.25).returns)
IDX = lb.index
spy = px["SPY"].pct_change().reindex(IDX)
gld_ief = simulate_basket(px[["GLD","IEF"]], START, END, 42, 1.0).returns.reindex(IDX).fillna(0.0)
tan_m = (1 + px["TAN"].pct_change().reindex(IDX)).resample("ME").prod() - 1
lb_m = (1 + lb).resample("ME").prod() - 1
SH = {n: simulate_basket(px[[t for t in tk if t in px.columns]], START, END, 42, 1.0).returns.reindex(IDX).fillna(0.0)
      for n, tk in SHORTS.items()}

dfii = fred("DFII10").reindex(pd.date_range(START, END)).ffill()
S1 = monthly_hold(((dfii - dfii.shift(126)) > 0).shift(5).fillna(False), IDX)   # handoff 1b: 6-mo real-yield rise
S2 = hedges.conditional_short_mask((1 + lb).cumprod(), 100, 20, 252).reindex(IDX).fillna(False)
ALWAYS = pd.Series(True, index=IDX)
print(f"S1 (real-yield rising) on {S1.mean()*100:.0f}% | S2 (price-vol) on {S2.mean()*100:.0f}% of days")

EP = {}
for nm, (pw, te) in EPISODES.items():
    try: EP[nm] = hedges.find_drawdown_episode(lb, pw, te)
    except ValueError: EP[nm] = None

rows = []
def evaluate(label, hedged, carry=np.nan):
    hedged = hedged.reindex(IDX).dropna()
    m = compute_metrics(hedged, RF, AF)
    row = {"label": label, "carry": carry, "CAGR": m["cagr"], "vol": m["ann_vol"],
           "Sharpe": m["sharpe"], "maxDD": m["max_dd"], "beta": beta(hedged, spy), "drag": cagr(lb) - m["cagr"]}
    for cn, (a, b) in CALM.items():
        row[f"drag[{cn}]"] = cagr(lb.loc[a:b]) - cagr(hedged.loc[a:b])
    dd_red = []
    for nm, pk in EP.items():
        if pk is None: row[f"dd[{nm}]"] = np.nan; row[f"pnl[{nm}]"] = np.nan; continue
        p, t = pk
        row[f"dd[{nm}]"] = hedges.episode_drawdown(hedged, p, t)
        row[f"pnl[{nm}]"] = seg(hedged, p, t) - seg(lb, p, t)
        dd_red.append(abs(hedges.episode_drawdown(lb, p, t)) - abs(hedges.episode_drawdown(hedged, p, t)))
    row["dd_red_avg3"] = np.mean(sorted(dd_red)[-3:]) if len(dd_red) >= 3 else np.nan
    hm = (1 + hedged).resample("ME").prod() - 1
    fear = ((lb_m < 0) & (tan_m > 0)).reindex(hm.index).fillna(False)
    row["fear_n"] = int(fear.sum())
    row["fear_pnl"] = float((hm[fear] - lb_m.reindex(hm.index)[fear]).sum())
    rows.append(row)

evaluate("unhedged (long-only)", lb)
evaluate("de-lever 0.85x", 0.85 * lb + 0.15 * RF / AF)
evaluate("+GLD/IEF 15% sleeve", 0.85 * lb + 0.15 * gld_ief)
for w in NOTIONALS:
    evaluate(f"-SPY static {w:.2f}x", lb - w * spy, carry=cagr(-spy))
for hn in SHORTS:
    evaluate(f"[{hn} standalone short carry]", -SH[hn], carry=cagr(-SH[hn]))
    for sn, M in [("always", ALWAYS), ("S1", S1), ("S2", S2)]:
        pair = (-SH[hn] * M.astype(float)).reindex(IDX).fillna(0.0)
        for w in NOTIONALS:
            evaluate(f"-{hn} {sn} {w:.2f}x", lb + w * pair, carry=cagr(pair))

df = pd.DataFrame(rows).set_index("label")
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 80)
core = ["carry","CAGR","vol","Sharpe","maxDD","beta","drag","dd_red_avg3","fear_n","fear_pnl"]
print("\n=== CORE (%, except Sharpe/beta/fear_n) ===\n", (df[core] * [100,100,100,1,100,1,100,100,1,100]).round(2))
print("\n=== EPISODE DRAWDOWNS (%) ===\n", (df[[c for c in df if c.startswith('dd[')]] * 100).round(1))
print("\n=== HEDGE P&L CONTRIB PER EPISODE (%) ===\n", (df[[c for c in df if c.startswith('pnl[')]] * 100).round(1))
print("\n=== CALM ANNUAL DRAG (%) ===\n", (df[[c for c in df if c.startswith('drag[')]] * 100).round(2))

base = df.loc["unhedged (long-only)"]
epc = [c for c in df if c.startswith("dd[")]
print("\n=== GATES + INSURANCE EFFICIENCY (decision @ 0.50x) ===")
print(f"{'hedge':<26}{'eff':>6}{'g1_noDeepen':>12}{'g2_fear':>9}{'g3_calmDrag':>13}  verdict")
for lab in df.index:
    if "standalone" in lab or lab == "unhedged (long-only)": continue
    r = df.loc[lab]
    eff = (r["dd_red_avg3"] * 100) / (r["drag"] * 100) if r["drag"] > 1e-4 else np.inf
    g1 = all(r[c] >= base[c] - 1e-6 for c in epc if r[c] == r[c])
    g2 = r["fear_pnl"] >= -0.02
    g3 = (r["drag[pre-COVID 17-19]"] <= 0.06 + 1e-9) and (r["drag"] <= 0.06 + 1e-9)
    print(f"{lab:<26}{eff:>6.2f}{str(g1):>12}{str(g2):>9}{str(g3):>13}  {'PASS' if (g1 and g2 and g3) else 'fail'}")
