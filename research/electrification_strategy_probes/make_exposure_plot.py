"""3-panel proposal figure: growth, rolling Sharpe, and how much the shipped
strategy (screen +val+sleeve) trims the supplier basket over time."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, base64, io
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from grid_equipment_basket.data.prices import fetch_prices
from electrification_strategy import config, universe, backtest, valuation_overlay

START, END = "2017-06-01", "2026-08-31"; RF, AF = 0.04, 252
C_SHIP, C_PLAIN, C_VOLT, C_INK, C_SOFT, C_LINE = "#a8560f", "#2456a6", "#8a94a0", "#1b1f24", "#5b626b", "#e6e0d6"
PLOT_FROM = "2019-01-01"                     # skip vol-target warm-up for the Sharpe panel

tk = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index) | {"SPY", "GLD", "IEF", "VOLT"})
px = fetch_prices(tk, START, END)
spy = px["SPY"].pct_change()
cand = px[[c for c in tk if c in px.columns]]
base_r = backtest.basket_returns(cand, "screen", START, END)
IDX = base_r.index
vscal = backtest._vol_scalar(base_r).reindex(IDX)
vmult = valuation_overlay.extension_multiplier((1 + base_r).cumprod(), spy.reindex(IDX)).reindex(IDX).fillna(1.0)
exposure = (vmult * vscal).clip(upper=config.VOL_MAX_LEVERAGE).fillna(1.0)
supplier_wt = ((1 - 0.15) * exposure).rename("w")
plain = backtest.compose(base_r, spy, use_valuation=False)
gld_ief = px[["GLD", "IEF"]].pct_change().mean(axis=1).reindex(IDX).fillna(0.0)
ship = 0.85 * backtest.compose(base_r, spy, use_valuation=True) + 0.15 * gld_ief
volt = px["VOLT"].pct_change().dropna()

def growth(r): return (1 + r.dropna()).cumprod()
def roll_sharpe(r, w=252):
    ex = (r - RF / AF).dropna()
    return (ex.rolling(w).mean() / ex.rolling(w).std() * np.sqrt(AF)).loc[PLOT_FROM:]

fig, ax = plt.subplots(3, 1, figsize=(9.2, 8.8), sharex=True,
                       gridspec_kw={"height_ratios": [2.1, 1.3, 1.6], "hspace": 0.13})
for a in ax:
    a.set_facecolor("white")
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(C_LINE)
    a.tick_params(colors=C_SOFT, labelsize=8.5, length=3)
    a.grid(axis="y", color=C_LINE, lw=0.8, alpha=0.7)
    a.set_xlim(pd.Timestamp("2017-04-01"), pd.Timestamp("2026-12-01"))

# panel 1 -- growth of $1 (log)
g_ship, g_plain = growth(ship), growth(plain)
g_volt = g_plain.reindex(volt.index, method="nearest").iloc[0] * growth(volt)
ax[0].plot(g_plain.index, g_plain.values, color=C_PLAIN, lw=1.5, label="screen  (no overlays)")
ax[0].plot(g_ship.index, g_ship.values, color=C_SHIP, lw=2.3, label="screen +val+sleeve   (shipped)")
ax[0].plot(g_volt.index, g_volt.values, color=C_VOLT, lw=1.5, ls=(0, (4, 2)), label="VOLT ETF  (from Dec-2024)")
ax[0].set_yscale("log"); ax[0].set_ylabel("growth of $1  (log)", color=C_SOFT, fontsize=9)
ax[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}x"))
ax[0].annotate(f"{g_ship.iloc[-1]:.1f}x", (g_ship.index[-1], g_ship.iloc[-1]), xytext=(7, 0),
               textcoords="offset points", color=C_SHIP, fontsize=9, fontweight="bold", va="center")
ax[0].annotate(f"{g_plain.iloc[-1]:.1f}x", (g_plain.index[-1], g_plain.iloc[-1]), xytext=(7, 5),
               textcoords="offset points", color=C_PLAIN, fontsize=9, fontweight="bold", va="center")
ax[0].legend(frameon=False, fontsize=8.7, loc="upper left", labelcolor=C_INK)
ax[0].set_title("Electrification Strategy  —  shipped variant  (screen +val+sleeve)",
                color=C_INK, fontsize=11.5, fontweight="bold", loc="left", pad=10)

# panel 2 -- rolling 1y Sharpe
rs_ship, rs_plain = roll_sharpe(ship), roll_sharpe(plain)
ax[1].axhline(0, color=C_SOFT, lw=0.9)
ax[1].plot(rs_plain.index, rs_plain.values, color=C_PLAIN, lw=1.4)
ax[1].plot(rs_ship.index, rs_ship.values, color=C_SHIP, lw=2.1)
ax[1].set_ylabel("rolling 1-yr Sharpe", color=C_SOFT, fontsize=9)
ax[1].set_ylim(-3.2, 3.7)
ax[1].annotate(f"{rs_ship.iloc[-1]:.2f}", (rs_ship.index[-1], rs_ship.iloc[-1]), xytext=(7, 0),
               textcoords="offset points", color=C_SHIP, fontsize=9, fontweight="bold", va="center")
full_sh = (ship - RF/AF).mean() / (ship - RF/AF).std() * np.sqrt(AF)
ax[1].text(0.012, 0.08, f"full-period Sharpe  {full_sh:.2f}", transform=ax[1].transAxes,
           color=C_SOFT, fontsize=8.2, va="bottom")

# panel 3 -- how much the supplier basket is trimmed
sw = supplier_wt.loc[PLOT_FROM:]
sw_s = sw.rolling(10).mean()
ax[2].axhline(0.85, color=C_SOFT, lw=1.0, ls=(0, (2, 2)))
ax[2].fill_between(sw.index, sw.values, 0.85, where=(sw.values < 0.85), color=C_SHIP, alpha=0.14, interpolate=True)
ax[2].fill_between(sw.index, sw.values, 0.85, where=(sw.values >= 0.85), color=C_PLAIN, alpha=0.09, interpolate=True)
ax[2].plot(sw.index, sw.values, color=C_SHIP, lw=0.5, alpha=0.35)
ax[2].plot(sw_s.index, sw_s.values, color=C_SHIP, lw=1.9, label="net weight in supplier basket (10-day avg)")
ax[2].plot(vmult.loc[PLOT_FROM:].index, vmult.loc[PLOT_FROM:].values * 0.85, color=C_INK, lw=1.1, alpha=0.5,
           label="valuation rule alone (x 0.85)")
ax[2].set_ylabel("weight in supplier basket", color=C_SOFT, fontsize=9)
ax[2].set_ylim(0, 1.4)
ax[2].text(pd.Timestamp("2019-02-01"), 0.90, "baseline 0.85  (15% held in GLD / IEF sleeve)",
           color=C_SOFT, fontsize=7.8, va="bottom")
tr = sw.idxmin()
ax[2].annotate(f"deepest trim {sw.min():.2f}   {tr:%b %Y}", (tr, sw.min()), xytext=(10, 26),
               textcoords="offset points", color=C_SHIP, fontsize=8,
               arrowprops=dict(arrowstyle="-", color=C_SHIP, lw=0.8))
ax[2].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=C_INK)
ax[2].text(0.012, -0.30, "Below 0.85 the vol-target and valuation rules are trimming; above, the book is levered "
           "toward the 20% vol target (1.5x cap). Trimmed ~73% of days; valuation rule active (mult < 1) ~42%.",
           transform=ax[2].transAxes, color=C_SOFT, fontsize=7.6, va="top")

fig.savefig("docs/electrification-strategy-exposure.png", dpi=150, facecolor="white", bbox_inches="tight")
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, facecolor="white", bbox_inches="tight")
open("scratchpad/exposure_datauri.txt", "w").write("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
print(f"PNG {len(buf.getvalue())//1024} KB   data-uri {len(base64.b64encode(buf.getvalue()))//1024} KB")
print(f"weight min {sw.min():.2f} max {sw.max():.2f} mean {sw.mean():.2f} | recent rolling Sharpe {rs_ship.iloc[-1]:.2f}")
