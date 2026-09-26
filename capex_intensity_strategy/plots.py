"""Charts comparing the strategy against the cheap alternatives.

The question these answer is not "did it go up" but **"could I have bought this
more cheaply?"** -- so the benchmarks are the things a person would actually buy
instead: a broad index, the sector, and the thematic ETFs that hold the same
kind of names for a published expense ratio.

Palette is validated (lightness band, chroma floor, CVD separation, normal-vision
floor, contrast) -- four categorical hues is the ceiling that passes, so the four
real contenders carry colour and broad-market context series stay neutral grey.
The orange carries a contrast warning, which is why every series is directly
labelled and a table accompanies the figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grid_equipment_basket.backtest import compute_metrics

from capex_intensity_strategy import config

# Two validated palettes. Dark is SELECTED, not flipped: its own steps, run
# through the six checks against the dark surface (all pass, no warnings).
# Light: #2166AC #E08214 #1B9E77 #A6519C  (orange carries a contrast WARN ->
#        every series is directly labelled, which is the required relief)
# Dark:  #4A90D9 #C87519 #2AAE7C #B268C4  (all checks pass)
PALETTES = {
    "light": {
        "accent": {"strategy": "#2166AC", "a": "#E08214",
                   "b": "#1B9E77", "c": "#A6519C"},
        "ink": "#1a1d21", "muted": "#6b7783", "grid": "#dfe4e9",
        "surface": "#fcfcfb", "neutral": ["#8b95a1", "#aab3bd"],
    },
    "dark": {
        "accent": {"strategy": "#4A90D9", "a": "#C87519",
                   "b": "#2AAE7C", "c": "#B268C4"},
        "ink": "#e6ebf0", "muted": "#95a1ad", "grid": "#2b343d",
        "surface": "#12171c", "neutral": ["#7d8894", "#5d6873"],
    },
}

ACCENT = PALETTES["light"]["accent"]
INK = PALETTES["light"]["ink"]
MUTED = PALETTES["light"]["muted"]
GRID_LINE = PALETTES["light"]["grid"]
SURFACE = PALETTES["light"]["surface"]
NEUTRAL = PALETTES["light"]["neutral"]


def set_mode(mode: str) -> None:
    """Switch the module palette. Charts embedded in a dark document must be
    drawn dark, or the page glares white wherever a figure sits."""
    global ACCENT, INK, MUTED, GRID_LINE, SURFACE, NEUTRAL
    pal = PALETTES[mode]
    ACCENT, INK, MUTED = pal["accent"], pal["ink"], pal["muted"]
    GRID_LINE, SURFACE, NEUTRAL = pal["grid"], pal["surface"], pal["neutral"]


# What a person would buy instead, and what it costs them.
ALTERNATIVES = {
    "SPY": ("S&P 500", 0.09),
    "XLI": ("Industrials sector", 0.08),
    "GRID": ("Smart-grid ETF", 0.57),
    "PAVE": ("US infrastructure ETF", 0.47),
    "XLU": ("Utilities sector", 0.08),
    "VOLT": ("Electrification ETF", 0.75),
}


def _style(ax, title: str, ylabel: str, subtitle: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_LINE, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_LINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    full = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full, color=INK, fontsize=12.5, fontweight="600",
                 loc="left", pad=14, linespacing=1.5)


def _end_label(ax, x, y, text, color) -> None:
    """Direct label in ink; the line arriving at it carries the identity."""
    ax.annotate(f" {text}", xy=(x, y), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK, fontweight="500",
                annotation_clip=False)
    ax.plot([x], [y], marker="o", markersize=4, color=color, zorder=5)


def _series_colors(names: list[str]) -> dict[str, str]:
    """Strategy gets the hero hue; the next three contenders get the other three."""
    out = {names[0]: ACCENT["strategy"]}
    hero_rest = [ACCENT["a"], ACCENT["b"], ACCENT["c"]]
    neutrals = list(NEUTRAL)
    for name in names[1:]:
        if hero_rest:
            out[name] = hero_rest.pop(0)
        else:
            out[name] = neutrals.pop(0) if neutrals else MUTED
    return out


def _spread_labels(values: dict[str, float], min_gap: float, log: bool = False
                   ) -> dict[str, float]:
    """Push overlapping end-labels apart so none collide.

    Works in log space when the axis is logarithmic, so the nudge is a constant
    visual distance rather than a constant dollar amount.
    """
    items = sorted(values.items(), key=lambda kv: kv[1])
    placed: list[tuple[str, float]] = []
    for name, v in items:
        pos = np.log10(v) if log else v
        if placed:
            prev = np.log10(placed[-1][1]) if log else placed[-1][1]
            if pos - prev < min_gap:
                pos = prev + min_gap
        placed.append((name, float(10 ** pos) if log else pos))
    return dict(placed)


def growth_chart(ax, curves: pd.DataFrame, colors: dict[str, str]) -> None:
    """Growth of $100, log scale -- equal vertical distance = equal percentage."""
    for name in curves.columns:
        s = curves[name].dropna()
        lw = 2.2 if name == curves.columns[0] else 1.5
        ax.plot(s.index, s.values, color=colors[name], linewidth=lw,
                zorder=4 if name == curves.columns[0] else 3)
    ax.set_yscale("log")
    ends = {n: float(curves[n].dropna().iloc[-1]) for n in curves.columns}
    span = np.log10(max(ends.values())) - np.log10(min(ends.values()))
    label_y = _spread_labels(ends, min_gap=max(span * 0.11, 0.028), log=True)
    x_end = curves.index[-1]
    for name, y in label_y.items():
        true_y = ends[name]
        ax.plot([x_end], [true_y], marker="o", markersize=4,
                color=colors[name], zorder=5)
        ax.annotate(f" {name} {true_y:,.0f}", xy=(x_end, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=INK, fontweight="500", annotation_clip=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _style(ax, "Growth of $100", "log scale",
           "Equal vertical distance is equal percentage change")


def underwater_chart(ax, curves: pd.DataFrame, colors: dict[str, str],
                     rival: str | None = None) -> None:
    """Strategy against its single closest rival -- two series, not six.

    Six overlapping underwater curves is unreadable noise; the magnitude
    comparison across all of them belongs in the bar chart instead.
    """
    strategy = curves.columns[0]
    show = [strategy] + ([rival] if rival and rival in curves.columns else [])
    for name in show:
        s = curves[name].dropna()
        dd = (s / s.cummax() - 1.0) * 100
        is_hero = name == strategy
        ax.fill_between(dd.index, dd.values, 0, color=colors[name],
                        alpha=0.22 if is_hero else 0.10, zorder=2)
        ax.plot(dd.index, dd.values, color=colors[name],
                linewidth=2.0 if is_hero else 1.4, zorder=3)
        ax.annotate(f" {name} {dd.min():.0f}%", xy=(dd.idxmin(), dd.min()),
                    xytext=(6, 2), textcoords="offset points", fontsize=8.5,
                    color=INK, fontweight="600" if is_hero else "400")
    ax.axhline(0, color=GRID_LINE, linewidth=1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    sub = f"Against {rival}, the closest rival" if rival else "Peak-to-trough"
    _style(ax, "Time under water", "peak-to-trough", sub)


def maxdd_chart(ax, stats: pd.DataFrame, colors: dict[str, str]) -> None:
    """Worst drawdown as bars -- a magnitude comparison, so bars, not lines.

    Plotted as depth (a positive magnitude) so the bars read left-to-right, with
    values labelled outside the bar end in ink rather than buried inside it.
    """
    ordered = stats.sort_values("max_dd", ascending=False)   # shallowest first
    ys = np.arange(len(ordered))
    depth = ordered["max_dd"].abs().values * 100
    hero = ordered.index.get_loc(stats.index[0])
    bars = ax.barh(ys, depth, color=[colors[n] for n in ordered.index],
                   height=0.6, zorder=3)
    for i, bar in enumerate(bars):
        bar.set_alpha(1.0 if i == hero else 0.78)
    for y, v in zip(ys, depth):
        ax.annotate(f"{v:.0f}%", xy=(v, y), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=9, color=INK,
                    fontweight="600" if y == hero else "400", zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels(ordered.index, fontsize=9, color=INK)
    ax.set_xticks([])
    ax.set_xlim(0, depth.max() * 1.22)      # room for the outside labels
    ax.invert_yaxis()
    _style(ax, "Worst drawdown", "", "Shallower is better")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def risk_return_chart(ax, stats: pd.DataFrame, colors: dict[str, str]) -> None:
    """Vol against return -- the chart that answers 'is the complexity worth it'."""
    for sharpe in (0.25, 0.5, 0.75, 1.0):
        xs = np.linspace(0.5, max(stats["vol"]) * 100 + 4, 50)
        ax.plot(xs, config.RF * 100 + sharpe * xs, color=GRID_LINE,
                linewidth=0.9, linestyle=(0, (4, 4)), zorder=1)
        ax.annotate(f"Sharpe {sharpe:g}", xy=(xs[-1], config.RF * 100 + sharpe * xs[-1]),
                    fontsize=7.5, color=MUTED, va="bottom", ha="right", zorder=1)
    pts = {n: (r["vol"] * 100, r["cagr"] * 100) for n, r in stats.iterrows()}
    xr = max(x for x, _ in pts.values()) - min(x for x, _ in pts.values())
    yr = max(y for _, y in pts.values()) - min(y for _, y in pts.values())
    for name, row in stats.iterrows():
        is_strategy = name == stats.index[0]
        x, y = pts[name]
        ax.scatter(x, y, s=150 if is_strategy else 85, color=colors[name],
                   zorder=5, edgecolor=SURFACE, linewidth=2)
        # If a neighbour sits just to the right, label to the left instead so the
        # text never lands on another marker.
        crowded = any(o != name and 0 < px - x < xr * 0.16 and abs(py - y) < yr * 0.07
                      for o, (px, py) in pts.items())
        ax.annotate(f"{name}  " if crowded else f"  {name}", xy=(x, y),
                    ha="right" if crowded else "left", fontsize=9, color=INK,
                    va="center", fontweight="600" if is_strategy else "400")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("annualised volatility", color=MUTED, fontsize=9)
    _style(ax, "Risk against return", "annualised return",
           f"Dashed lines are equal Sharpe at a {config.RF:.0%} cash rate — up and to the left is better")


def legs_chart(ax, legs: pd.DataFrame, start: str, end: str) -> None:
    """Combined book solid; the two legs dotted, since neither is held alone.

    The short leg is shown *as held* (i.e. already negative-signed), which is
    the only honest way to read it -- its standalone Sharpe is negative, and it
    earns its place through correlation, not return.
    """
    seg = legs.loc[start:end].dropna(how="all")
    spec = [("combined", ACCENT["strategy"], "-", 2.4, "Combined book (1.0 long − 0.5 short)"),
            ("long", ACCENT["b"], (0, (2, 2)), 1.7, "Long leg — asset-light suppliers"),
            ("short", ACCENT["a"], (0, (2, 2)), 1.7, "Short leg — as held (capital-heavy)")]
    for col, colour, style, lw, label in spec:
        if col not in seg.columns:
            continue
        s = (1 + seg[col].fillna(0)).cumprod() * 100
        m = compute_metrics(seg[col].dropna(), config.RF, config.ANN)
        ax.plot(s.index, s.values, color=colour, linewidth=lw, linestyle=style,
                zorder=4 if col == "combined" else 3,
                label=f"{label}   Sharpe {m['sharpe']:.2f}")
        ax.plot([s.index[-1]], [s.iloc[-1]], marker="o", markersize=4,
                color=colour, zorder=5)
        ax.annotate(f" {s.iloc[-1]:,.0f}", xy=(s.index[-1], s.iloc[-1]),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK, fontweight="500", annotation_clip=False)
    ax.axhline(100, color=GRID_LINE, linewidth=1, zorder=1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK,
                    handlelength=2.6, borderpad=0)
    for txt in leg.get_texts():
        txt.set_color(INK)
    _style(ax, "Where the return comes from", "growth of $100",
           "The short leg loses money alone — it earns its place by diversifying, not by returning")


def legs_figure(legs: pd.DataFrame, out_dir: Path, start: str = "2019-01-01",
                end: str = "2026-09-18") -> Path:
    fig, ax = plt.subplots(figsize=(11.0, 6.2), facecolor=SURFACE)
    legs_chart(ax, legs, start, end)
    fig.tight_layout(rect=(0, 0, 0.96, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "legs.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return p


def build_curves(net: pd.Series, prices: pd.DataFrame, start: str, end: str,
                 tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Indexed growth curves and a metric table over one common window."""
    rets = {"Capex-intensity": net}
    for t in tickers:
        if t in prices.columns:
            rets[t] = prices[t].pct_change()
    frame = pd.DataFrame(rets).loc[start:end]
    # A common window: every series must be present, so nothing is flattered by
    # a later, kinder start date.
    frame = frame.dropna(how="any")
    curves = (1 + frame).cumprod() * 100
    rows = {}
    for name in frame.columns:
        m = compute_metrics(frame[name].dropna(), config.RF, config.ANN)
        dd = m.get("max_dd", np.nan)
        rows[name] = {"cagr": m.get("cagr", np.nan), "vol": m.get("ann_vol", np.nan),
                      "sharpe": m.get("sharpe", np.nan), "max_dd": dd,
                      "calmar": m.get("cagr", np.nan) / abs(dd) if dd else np.nan}
    return curves, pd.DataFrame(rows).T


def make_figures(net: pd.Series, prices: pd.DataFrame, out_dir: Path,
                 start: str = "2019-01-01", end: str | None = None) -> dict[str, Path]:
    """Write the comparison figures. Returns {name: path}."""
    end = end or str(net.dropna().index[-1].date())
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # -- long-history comparison: the alternatives that existed the whole time --
    long_tickers = [t for t in ("GRID", "XLI", "XLU", "SPY", "PAVE")
                    if t in prices.columns]
    curves, stats = build_curves(net, prices, start, end, long_tickers)
    order = list(curves.columns)
    colors = _series_colors(order)

    # The closest rival on Sharpe is the one worth an underwater comparison.
    rivals = stats.drop(index=stats.index[0])
    rival = rivals["sharpe"].idxmax() if len(rivals) else None

    fig = plt.figure(figsize=(16.5, 7.0), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.8], wspace=0.38,
                          top=0.80, bottom=0.09, left=0.055, right=0.985)
    growth_chart(fig.add_subplot(gs[0]), curves, colors)
    underwater_chart(fig.add_subplot(gs[1]), curves, colors, rival=rival)
    maxdd_chart(fig.add_subplot(gs[2]), stats, colors)
    fig.suptitle("Capex-intensity strategy against the cheaper alternatives",
                 fontsize=15.5, fontweight="650", color=INK, x=0.055, ha="left", y=0.975)
    fig.text(0.055, 0.925,
             f"{curves.index[0]:%b %Y} – {curves.index[-1]:%b %Y} · net of 10bp costs · "
             f"common window, every series present throughout",
             fontsize=9.5, color=MUTED, ha="left")
    p = out_dir / "growth_and_drawdown.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE); plt.close(fig)
    written["growth_and_drawdown"] = p

    fig, ax = plt.subplots(figsize=(8.6, 6.4), facecolor=SURFACE)
    risk_return_chart(ax, stats, colors)
    fig.tight_layout()
    p = out_dir / "risk_return.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE); plt.close(fig)
    written["risk_return"] = p

    # -- VOLT era: VOLT listed 2024-12-05, so it gets its own honest window --
    if "VOLT" in prices.columns:
        volt_start = str(prices["VOLT"].dropna().index[0].date())
        v_tickers = [t for t in ("VOLT", "GRID", "XLU", "SPY") if t in prices.columns]
        v_curves, v_stats = build_curves(net, prices, volt_start, end, v_tickers)
        if len(v_curves) > 30:
            v_colors = _series_colors(list(v_curves.columns))
            fig, ax = plt.subplots(figsize=(10.5, 6.0), facecolor=SURFACE)
            for name in v_curves.columns:
                s = v_curves[name].dropna()
                ax.plot(s.index, s.values, color=v_colors[name],
                        linewidth=2.2 if name == v_curves.columns[0] else 1.5)
                _end_label(ax, s.index[-1], s.iloc[-1],
                           f"{name} {s.iloc[-1]:,.0f}", v_colors[name])
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            _style(ax, "Since VOLT listed", "growth of $100",
                   f"VOLT began trading {pd.Timestamp(volt_start):%d %b %Y} — "
                   f"only {len(v_curves)} trading days, far too short to judge")
            fig.tight_layout(rect=(0, 0, 0.94, 1))
            p = out_dir / "volt_era.png"
            fig.savefig(p, dpi=150, facecolor=SURFACE); plt.close(fig)
            written["volt_era"] = p
            v_stats.to_csv(out_dir / "stats_volt_era.csv")

    stats.to_csv(out_dir / "stats_long_history.csv")
    return written


def format_table(stats: pd.DataFrame, title: str) -> str:
    lines = [title, "-" * len(title),
             f"{'':22s} {'CAGR':>8} {'Vol':>7} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'fee':>6}"]
    for name, r in stats.iterrows():
        label, fee = ALTERNATIVES.get(name, ("", None))
        fee_s = f"{fee:.2f}%" if fee is not None else "  --"
        shown = f"{name} ({label})" if label else name
        lines.append(f"{shown[:22]:22s} {r['cagr'] * 100:7.1f}% {r['vol'] * 100:6.1f}% "
                     f"{r['sharpe']:7.2f} {r['max_dd'] * 100:7.1f}% {r['calmar']:7.2f} {fee_s:>6}")
    return "\n".join(lines)
