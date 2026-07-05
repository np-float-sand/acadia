"""
Generate output/executive_summary_v2.html

Updated for the business-model-aware factor architecture (July 2026):
- Hard switch architecture: Sharpe 0.331 (up from 0.272 non-PJM baseline)
- Full 18-ticker universe including PJM
- IC KPI corrected to overall mean (no stressed/calm split)
- Merchant generator weakness added
- Phase 2 signal roadmap updated

Run: python generate_executive_summary_v2.py
"""

import base64, io, sys
from pathlib import Path

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

ROOT = Path(__file__).parent
OUT  = ROOT / "output"

# ── Numbers ────────────────────────────────────────────────────────────────────
STRATEGY = {
    "cumulative":   "+95%",
    "ann_return":   "9.38%",
    "sharpe":       "0.33",
    "max_dd":       "−19.5%",
    "mean_ic":      "0.14",
    "pct_ic_pos":   "~58%",
    "backtest":     "Apr 2018 – Dec 2025",
    "universe":     "18 stocks, 5 ISOs (ERCOT, PJM, MISO, CAISO, SPP)",
}

XLU = {
    "cumulative":   "+113%",
    "ann_return":   "12.0%",
    "sharpe":       "0.24",
    "max_dd":       "−36.1%",
}

# ── Chart embedding ────────────────────────────────────────────────────────────
chart_b64 = ""
chart_path = OUT / "backtest_performance.png"
if chart_path.exists():
    chart_b64 = base64.b64encode(chart_path.read_bytes()).decode()

chart_html = (
    f'<img src="data:image/png;base64,{chart_b64}" style="width:100%;border-radius:8px;">'
    if chart_b64 else
    '<p style="color:#888;text-align:center;padding:60px 0">Chart not found — run backtest first</p>'
)

# ── HTML ───────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grid Resilience Equity Strategy — Executive Summary v2</title>
<style>
  :root {{
    --navy:#0f2744; --teal:#1a7f7a; --amber:#e8a020; --red:#c94040;
    --green:#2e7d32; --light:#f5f7fa; --border:#dde3ea;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#fff; color:#1a2332; font-size:14px; }}
  .page {{ max-width:1100px; margin:0 auto; padding:32px 28px; }}

  /* Header */
  .header {{ background:var(--navy); color:#fff; padding:28px 32px; border-radius:10px; margin-bottom:24px; }}
  .header h1 {{ font-size:22px; font-weight:700; letter-spacing:.3px; }}
  .header .sub {{ font-size:13px; color:#9bb4d0; margin-top:6px; }}
  .header .badge {{ display:inline-block; background:var(--teal); color:#fff; font-size:11px;
                    padding:3px 10px; border-radius:12px; margin-top:10px; }}
  .header .v2badge {{ display:inline-block; background:var(--amber); color:#fff; font-size:11px;
                    padding:3px 10px; border-radius:12px; margin-top:10px; margin-left:8px; }}

  /* KPI strip */
  .kpis {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:24px; }}
  .kpi {{ background:var(--light); border:1px solid var(--border); border-radius:8px; padding:16px 14px; }}
  .kpi .label {{ font-size:11px; color:#667; text-transform:uppercase; letter-spacing:.5px; }}
  .kpi .value {{ font-size:24px; font-weight:700; color:var(--navy); margin:4px 0 2px; }}
  .kpi .bench {{ font-size:11px; color:#888; }}
  .kpi .bench span {{ color:var(--red); font-weight:600; }}

  /* Two-column sections */
  .cols2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }}
  .cols3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:24px; }}

  /* Cards */
  .card {{ background:var(--light); border:1px solid var(--border); border-radius:8px; padding:20px; }}
  .card h3 {{ font-size:13px; font-weight:700; color:var(--navy); margin-bottom:10px; text-transform:uppercase; letter-spacing:.4px; }}
  .card p {{ font-size:13px; line-height:1.6; color:#334; }}

  /* Architecture highlight */
  .arch-box {{ background:#eef6f5; border:1px solid var(--teal); border-radius:8px; padding:16px 18px; margin-bottom:24px; }}
  .arch-box h3 {{ color:var(--teal); font-size:13px; font-weight:700; text-transform:uppercase; margin-bottom:8px; }}
  .arch-box p {{ font-size:13px; line-height:1.6; }}

  /* Note box */
  .note-box {{ background:#fff8e8; border-left:4px solid var(--amber); padding:14px 16px; border-radius:0 6px 6px 0; margin:14px 0; font-size:13px; line-height:1.6; }}

  /* Signal table */
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th {{ background:var(--navy); color:#fff; padding:8px 10px; text-align:left; font-weight:600; }}
  td {{ padding:7px 10px; border-bottom:1px solid var(--border); vertical-align:top; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:nth-child(even) td {{ background:#f9fafb; }}

  /* Tags */
  .tag {{ display:inline-block; font-size:10px; font-weight:700; padding:2px 8px; border-radius:10px; }}
  .tag-warn {{ background:#fff0d0; color:#a06000; border:1px solid #e8c060; }}
  .tag-road {{ background:#e8f5e9; color:#2e7d32; border:1px solid #81c784; }}
  .tag-new  {{ background:#e3f2fd; color:#1565c0; border:1px solid #64b5f6; }}

  /* Chart */
  .chart-wrap {{ margin-bottom:24px; }}
  .chart-wrap h3 {{ font-size:13px; font-weight:700; color:var(--navy); margin-bottom:10px; text-transform:uppercase; }}

  /* Footer */
  .footer {{ border-top:1px solid var(--border); margin-top:28px; padding-top:16px; font-size:11px; color:#888; line-height:1.7; }}

  @media print {{
    body {{ font-size:12px; }}
    .page {{ padding:18px; max-width:100%; }}
    .kpis {{ gap:8px; }}
    .kpi .value {{ font-size:20px; }}
  }}
</style>
</head>
<body>
<div class="page">

<!-- Header -->
<div class="header">
  <h1>Grid Resilience Equity Strategy</h1>
  <div class="sub">A systematic long/short equity strategy for U.S. utility stocks driven by electricity grid stress signals</div>
  <div class="sub" style="margin-top:8px">{STRATEGY["backtest"]} &nbsp;·&nbsp; {STRATEGY["universe"]} &nbsp;·&nbsp; Backtest only</div>
  <span class="badge">Confidential — Not Investment Advice</span>
  <span class="v2badge">v2 · July 2026 · Business-Model-Aware Architecture</span>
</div>

<!-- KPI Strip -->
<div class="kpis">
  <div class="kpi">
    <div class="label">Cumulative Return</div>
    <div class="value">{STRATEGY["cumulative"]}</div>
    <div class="bench">XLU buy &amp; hold: <span>{XLU["cumulative"]}</span></div>
  </div>
  <div class="kpi">
    <div class="label">Ann. Return</div>
    <div class="value">{STRATEGY["ann_return"]}</div>
    <div class="bench">XLU: <span>{XLU["ann_return"]}</span></div>
  </div>
  <div class="kpi">
    <div class="label">Sharpe Ratio</div>
    <div class="value">{STRATEGY["sharpe"]}</div>
    <div class="bench">XLU: <span>{XLU["sharpe"]}</span></div>
  </div>
  <div class="kpi">
    <div class="label">Max Drawdown</div>
    <div class="value">{STRATEGY["max_dd"]}</div>
    <div class="bench">XLU: <span>{XLU["max_dd"]}</span></div>
  </div>
  <div class="kpi">
    <div class="label">Mean IC (21-day)</div>
    <div class="value">{STRATEGY["mean_ic"]}</div>
    <div class="bench">Positive {STRATEGY["pct_ic_pos"]} of months</div>
  </div>
</div>

<!-- Architecture highlight -->
<div class="arch-box">
  <h3>v2 Architecture — Business-Model-Aware Factor</h3>
  <p>The factor now distinguishes between <strong>merchant generators</strong> (VST, NRG — scored on grid stress beta) and
  <strong>rate-regulated T&amp;D utilities</strong> (AEP, PPL, FE, D, EXC, PEG and others — scored on Interest Coverage Ratio).
  A <em>pass_through</em> coefficient (0–1) tags each ticker's exposure to spot electricity prices.
  The <strong>hard-switch architecture</strong> (tickers with pass_through &lt; 0.5 route to the ICR signal) produced
  Sharpe <strong>0.331</strong> with the full 18-name universe including PJM — beating the prior 0.272 baseline that
  excluded PJM entirely. Revenue-mix and dual-track blending architectures were also tested and underperformed.</p>
</div>

<!-- Thesis + Signal table -->
<div class="cols2">
  <div class="card">
    <h3>Investment Thesis</h3>
    <p>U.S. utility stocks are systematically mispriced around grid stress events.
    A utility's physical exposure to electricity grid stress — the topology of its transmission network,
    the mix of generation it relies on, and its regulatory regime — determines how its earnings respond to
    LMP spikes, congestion events, and extreme weather. These structural exposures are slow to be priced in
    because they require combining grid physics with equity analysis.</p>
    <br>
    <p>For merchant generators, grid stress drives earnings directly through spot prices.
    For regulated T&amp;D utilities, the relevant risk is financial fragility — highly leveraged utilities
    face refinancing stress when rates rise alongside grid costs.</p>
    <div class="note-box">
      XLU buy-and-hold returned {XLU["cumulative"]} over the same period, versus the strategy's {STRATEGY["cumulative"]}.
      But XLU's max drawdown was {XLU["max_dd"]} (vs {STRATEGY["max_dd"]} here), with annualised volatility ~20% vs ~13%.
      This strategy is positioned as a risk-managed complement to long-only sector exposure.
    </div>
  </div>
  <div class="card">
    <h3>Signal Construction</h3>
    <table>
      <tr><th>Step</th><th>Description</th></tr>
      <tr><td>1. Grid Stress Index</td><td>Daily composite of LMP z-score (40%), congestion fraction (25%), reserve tightness (20%), event flag (15%) per ISO</td></tr>
      <tr><td>2. Stress events</td><td>Named events + algo-detected LMP spikes (97th pct, 3+ days) and congestion episodes (30% fraction, 10+ days)</td></tr>
      <tr><td>3. Stress beta</td><td>Rolling OLS of excess stock return on GSI during event windows — merchants only</td></tr>
      <tr><td>4. ICR signal</td><td>Interest coverage ratio (EBIT / interest expense) — regulated utilities only</td></tr>
      <tr><td>5. Factor score</td><td>Hard switch: stress beta (merchants) or ICR (regulated). Renewable quality addon 15% for all. Cross-sectional z-score → rank.</td></tr>
      <tr><td>6. Portfolio</td><td>3 long / 5 short, equal-weight, monthly rebalance. Individual shorts — no ETF hedge.</td></tr>
    </table>
  </div>
</div>

<!-- Chart -->
<div class="chart-wrap">
  <h3>Backtest Performance (Apr 2018 – Dec 2025)</h3>
  {chart_html}
</div>

<!-- Three Key Findings -->
<div class="cols3">
  <div class="card">
    <h3>Finding 1 — Business Model Routing Recovers PJM</h3>
    <p>Rate-regulated T&amp;D utilities (PJM) were excluded from the original strategy because LMP signals
    have no mechanistic link to regulated revenues. Routing them to an ICR signal instead raised Sharpe
    from 0.272 (no PJM) to <strong>0.331</strong> with all 18 names. Three blending architectures were
    tested; the binary hard-switch outperformed continuous blending.</p>
  </div>
  <div class="card">
    <h3>Finding 2 — Short Book Drives Alpha</h3>
    <p>All 81 configurations using an XLU hedge (replacing the short book with a sector ETF) produced
    <strong>negative Sharpe</strong>. The strategy's alpha comes from picking individual short candidates
    — names that are specifically vulnerable to grid stress or financial fragility — not from sector-relative
    positioning. The long book's top-scored names do not outperform XLU by enough to make a relative trade.</p>
  </div>
  <div class="card">
    <h3>Finding 3 — Concentrate the Long Book</h3>
    <p>3 long names outperforms 5 or 7 at the same stress configuration. The factor has enough
    cross-sectional dispersion that concentrating in the top 3 adds return without proportionate
    increase in drawdown. Duration filter (10 consecutive days of congestion) carries most of the
    noise-reduction burden; the 30% congestion threshold can be lower than originally designed.</p>
  </div>
</div>

<!-- Weaknesses + Future Work -->
<div class="cols2">
  <div class="card">
    <h3>Known Weaknesses</h3>
    <p style="margin-bottom:8px"><span class="tag tag-warn">Data depth</span>&nbsp; 7 years, 3 meaningful stress regimes. Limited out-of-sample evidence.</p>
    <p style="margin-bottom:8px"><span class="tag tag-warn">Merchant narratives</span>&nbsp; VST and NRG are competitive generators whose equity returns are driven by AI/data center demand narratives orthogonal to grid stress. IC degraded in 2024 when VST re-rated +262% while model was short.</p>
    <p style="margin-bottom:8px"><span class="tag tag-warn">ICR data depth</span>&nbsp; yfinance returns only 4–12 quarters of ICR history. Regulated-utility signal is weaker pre-2022. EDGAR XBRL sourcing planned.</p>
    <p style="margin-bottom:8px"><span class="tag tag-warn">Beta lag</span>&nbsp; Rolling OLS takes ~1–2 quarters to reflect structural changes. Missed both the 2024 VST long and the 2025 VST short on the first leg.</p>
    <p style="margin-bottom:8px"><span class="tag tag-warn">pass_through estimates</span>&nbsp; Mixed-ticker coefficients (ETR, EXC, DTE) are initial estimates. Unverified against 10-K revenue disclosures.</p>
    <p><span class="tag tag-warn">Transaction costs</span>&nbsp; Zero slippage assumed. Estimated Sharpe impact: 0.05–0.10.</p>
  </div>
  <div class="card">
    <h3>Roadmap — Phase 2</h3>
    <p style="margin-bottom:8px"><span class="tag tag-new">In progress</span>&nbsp; <strong>EDGAR ICR sourcing</strong> — replace yfinance ICR with EDGAR XBRL filings for 10+ year history on the regulated signal path.</p>
    <p style="margin-bottom:8px"><span class="tag tag-road">Near-term</span>&nbsp; <strong>Rate case calendar signal</strong> — utilities with pending FERC/state PUC dockets have earnings uncertainty; filed-and-approved cases have earnings momentum.</p>
    <p style="margin-bottom:8px"><span class="tag tag-road">Near-term</span>&nbsp; <strong>Data center load growth signal</strong> — PJM interconnection queue + EIA-861 load by class captures the AI demand wave differentially across service territories (Dominion/VA, PSEG/NJ, ComEd/IL are primary beneficiaries).</p>
    <p style="margin-bottom:8px"><span class="tag tag-road">Near-term</span>&nbsp; <strong>Merchant regime filter</strong> — rolling correlation of VST/NRG with XNG/XLE vs XLU. Exclude from universe when stock "leaves the utility regime."</p>
    <p><span class="tag tag-road">Medium-term</span>&nbsp; <strong>RT/DA LMP spread</strong> — real-time vs day-ahead spread as a 5th GSI sub-signal, separating generator and T&amp;D utility dynamics.</p>
  </div>
</div>

<!-- Footer -->
<div class="footer">
  <strong>Disclaimer:</strong> This document presents historical backtest results only. Backtest performance is not indicative of future results.
  No representation is made regarding live trading performance. This is not investment advice or a recommendation to buy or sell any security.
  All trading decisions rest solely with the Client. Strategy universe: {STRATEGY["universe"]}.
  Backtest period: {STRATEGY["backtest"]}. All figures use 4% risk-free rate. Transaction costs not modelled.
</div>

</div>
</body>
</html>"""

out_path = OUT / "executive_summary_v2.html"
out_path.write_text(html, encoding="utf-8")
print(f"Written: {out_path}")
print(f"Chart embedded: {'yes' if chart_b64 else 'NO — run backtest first to generate backtest_performance.png'}")
