"""
Generate output/dc_signal_comparison.png

Cumulative return overlay: ICR regulated-path signal vs. DC load signal,
both hard-switch, both on today's (2026-08-14) complete/corrected data.
Also plots XLU buy-and-hold for reference.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# This script lives in output_dcqueue/ — its own directory holds the
# dc-queue backtest artifacts (pnl.csv etc); the icr comparison run's
# artifacts live in the sibling output_icr/ directory at repo root.
HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
OUT = HERE

icr = pd.read_csv(REPO_ROOT / "output_icr" / "pnl.csv", parse_dates=["Date"], index_col="Date")
dcq = pd.read_csv(HERE / "pnl.csv", parse_dates=["Date"], index_col="Date")

fig, ax = plt.subplots(figsize=(11, 5.5))

ax.plot(icr.index, icr["cum_strategy"] * 100, label="Hard switch + ICR", color="#c94040", linewidth=1.6)
ax.plot(dcq.index, dcq["cum_strategy"] * 100, label="Hard switch + DC load signal", color="#1a7f7a", linewidth=1.8)
ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")

ax.set_title("Cumulative Return — ICR vs. DC Load Signal (both corrected, same data)", fontsize=13, fontweight="bold")
ax.set_ylabel("Cumulative Return (%)")
ax.legend(loc="upper left", fontsize=10, frameon=False)
ax.grid(alpha=0.25)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
out_path = OUT / "dc_signal_comparison.png"
fig.savefig(out_path, dpi=150)
print(f"Written: {out_path}")
