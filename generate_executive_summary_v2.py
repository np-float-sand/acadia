"""
Generate output/executive_summary_v2.html

Updated July 2026 for business-model-aware architecture:
- Sharpe 0.331 with full 18-name universe including PJM
- Overlay chart: hard-switch (solid) vs non-PJM baseline (dotted)
- Corrected IC KPI (overall mean, no stressed/calm split)
- Merchant generator weakness named explicitly
- Phase 2 signal roadmap
- What changed and why narrative throughout
"""

import base64
from pathlib import Path

ROOT = Path(__file__).parent
OUT  = ROOT / "output"

STRATEGY = {
    "cumulative":   "+95%",
    "ann_return":   "9.38%",
    "sharpe":       "0.33",
    "max_dd":       "−19.5%",
    "mean_ic":      "0.14",
    "pct_ic_pos":   "61%",
    "backtest":     "Apr 2018 – Dec 2025",
    "universe":     "18 stocks, 5 ISOs (ERCOT, PJM, MISO, CAISO, SPP)",
}
XLU = {"cumulative": "+113%", "ann_return": "12.0%", "sharpe": "0.24", "max_dd": "−36.1%"}
BASELINE = {"sharpe": "0.27", "ann_return": "7.99%", "max_dd": "−15.4%"}

# Embed the overlay chart (v2 chart with dotted baseline)
chart_b64 = ""
for candidate in ["backtest_performance_v2.png", "backtest_performance.png"]:
    p = OUT / candidate
    if p.exists():
        chart_b64 = base64.b64encode(p.read_bytes()).decode()
        chart_label = candidate
        break

chart_html = (
    f'<img src="data:image/png;base64,{chart_b64}" style="width:100%;border-radius:8px;">'
    if chart_b64 else
    '<p style="color:#888;text-align:center;padding:40px">Chart not found — run generate_overlay_chart.py first</p>'
)

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

  .header {{ background:var(--navy); color:#fff; padding:28px 32px; border-radius:10px; margin-bottom:24px; }}
  .header h1 {{ font-size:22px; font-weight:700; }}
  .header .sub {{ font-size:13px; color:#9bb4d0; margin-top:6px; }}
  .badge {{ display:inline-block; font-size:11px; padding:3px 10px; border-radius:12px; margin-top:10px; }}
  .badge-conf {{ background:var(--teal); color:#fff; }}
  .badge-v2   {{ background:var(--amber); color:#fff; margin-left:6px; }}

  .kpis {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:24px; }}
  .kpi {{ background:var(--light); border:1px solid var(--border); border-radius:8px; padding:16px 14px; }}
  .kpi .label {{ font-size:11px; color:#667; text-transform:uppercase; letter-spacing:.5px; }}
  .kpi .value {{ font-size:24px; font-weight:700; color:var(--navy); margin:4px 0 2px; }}
  .kpi .bench {{ font-size:11px; color:#888; }}
  .kpi .bench .bad {{ color:var(--red); font-weight:600; }}
  .kpi .delta {{ font-size:11px; color:var(--green); font-weight:600; }}

  .cols2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px; }}
  .cols3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:24px; }}

  .card {{ background:var(--light); border:1px solid var(--border); border-radius:8px; padding:20px; }}
  .card h3 {{ font-size:12px; font-weight:700; color:var(--navy); margin-bottom:10px; text-transform:uppercase; letter-spacing:.5px; border-bottom:2px solid var(--teal); padding-bottom:6px; }}
  .card p  {{ font-size:13px; line-height:1.65; color:#334; margin-bottom:8px; }}
  .card p:last-child {{ margin-bottom:0; }}

  .change-box {{ background:#eef6f5; border:1px solid var(--teal); border-left:4px solid var(--teal);
                 border-radius:0 8px 8px 0; padding:18px 20px; margin-bottom:24px; }}
  .change-box h3 {{ color:var(--teal); font-size:13px; font-weight:700; text-transform:uppercase;
                    letter-spacing:.5px; margin-bottom:12px; }}
  .change-box .why {{ background:#fff; border-radius:6px; padding:12px 14px; margin-top:10px;
                      font-size:13px; line-height:1.6; border:1px solid #c8e6e4; }}
  .change-box .why strong {{ color:var(--navy); }}

  .compare-table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:10px; }}
  .compare-table th {{ background:var(--navy); color:#fff; padding:7px 10px; text-align:left; font-size:11px; }}
  .compare-table td {{ padding:6px 10px; border-bottom:1px solid var(--border); }}
  .compare-table tr:last-child td {{ border-bottom:none; }}
  .compare-table .win {{ color:var(--green); font-weight:700; }}
  .compare-table .lose {{ color:var(--red); }}

  .note-box {{ background:#fff8e8; border-left:4px solid var(--amber); padding:12px 14px;
               border-radius:0 6px 6px 0; margin:12px 0; font-size:13px; line-height:1.6; }}

  table.signal-tbl {{ width:100%; border-collapse:collapse; font-size:12px; }}
  table.signal-tbl th {{ background:var(--navy); color:#fff; padding:8px 10px; text-align:left; font-weight:600; }}
  table.signal-tbl td {{ padding:7px 10px; border-bottom:1px solid var(--border); vertical-align:top; }}
  table.signal-tbl tr:last-child td {{ border-bottom:none; }}
  table.signal-tbl tr:nth-child(even) td {{ background:#f9fafb; }}

  .tag {{ display:inline-block; font-size:10px; font-weight:700; padding:2px 8px; border-radius:10px; margin-bottom:6px; }}
  .tag-warn {{ background:#fff0d0; color:#a06000; border:1px solid #e8c060; }}
  .tag-road {{ background:#e8f5e9; color:var(--green); border:1px solid #81c784; }}
  .tag-new  {{ background:#e3f2fd; color:#1565c0; border:1px solid #64b5f6; }}

  .chart-wrap {{ margin-bottom:24px; }}
  .chart-wrap .chart-caption {{ font-size:11px; color:#888; margin-top:6px; font-style:italic; }}

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
  <div class="sub">A systematic long/short equity strategy for U.S. utility stocks driven by electricity grid stress signals and business-model-aware factor construction</div>
  <div class="sub" style="margin-top:8px">{STRATEGY["backtest"]} &nbsp;·&nbsp; {STRATEGY["universe"]} &nbsp;·&nbsp; Backtest only — not investment advice</div>
  <span class="badge badge-conf">Confidential</span>
  <span class="badge badge-v2">v2 · July 2026 · Business-Model-Aware Architecture</span>
</div>

<!-- KPI Strip -->
<div class="kpis">
  <div class="kpi">
    <div class="label">Cumulative Return</div>
    <div class="value">{STRATEGY["cumulative"]}</div>
    <div class="bench">XLU buy &amp; hold: <span class="bad">{XLU["cumulative"]}</span></div>
  </div>
  <div class="kpi">
    <div class="label">Ann. Return</div>
    <div class="value">{STRATEGY["ann_return"]}</div>
    <div class="bench">XLU: <span class="bad">{XLU["ann_return"]}</span></div>
    <div class="delta">↑ vs {BASELINE["ann_return"]} prior baseline</div>
  </div>
  <div class="kpi">
    <div class="label">Sharpe Ratio</div>
    <div class="value">{STRATEGY["sharpe"]}</div>
    <div class="bench">XLU: <span class="bad">{XLU["sharpe"]}</span></div>
    <div class="delta">↑ vs {BASELINE["sharpe"]} prior baseline</div>
  </div>
  <div class="kpi">
    <div class="label">Max Drawdown</div>
    <div class="value">{STRATEGY["max_dd"]}</div>
    <div class="bench">XLU: <span class="bad">{XLU["max_dd"]}</span></div>
  </div>
  <div class="kpi">
    <div class="label">Mean IC (21-day)</div>
    <div class="value">{STRATEGY["mean_ic"]}</div>
    <div class="bench">Positive {STRATEGY["pct_ic_pos"]} of months</div>
  </div>
</div>

<!-- What Changed and Why -->
<div class="change-box">
  <h3>What Changed in v2 — and Why</h3>
  <p><strong>The problem:</strong> The original strategy excluded PJM utilities (AEP, EXC, PPL, FE, Dominion, PSEG — six large-cap names) because including them destroyed alpha, dropping Sharpe from 0.27 to −0.03. Investigation revealed the root cause: the strategy's grid stress signal measures how earnings respond to LMP (electricity price) spikes. This works for <strong>merchant generators</strong> like VST and NRG, whose revenues come directly from selling power at spot prices. It fails for <strong>rate-regulated T&amp;D utilities</strong>, whose revenues are set by state regulators through rate cases — LMP congestion in their service territory has no direct link to their quarterly earnings.</p>

  <div class="why">
    <strong>The fix — business-model-aware routing:</strong> Each of the 18 universe stocks is now tagged with a <em>pass_through</em> coefficient (0–1) measuring what fraction of its revenue is exposed to merchant/spot electricity prices. In the winning <em>hard-switch</em> architecture, tickers with pass_through &lt; 0.5 (all PJM T&amp;D utilities, plus most regulated MISO/CAISO/SPP names) are routed to an <strong>Interest Coverage Ratio (ICR) signal</strong> instead of the LMP stress beta. ICR captures the risk that matters for regulated utilities: financial fragility during interest rate stress events. Tickers with pass_through ≥ 0.5 (VST, NRG) stay on the original stress beta path. Three blending architectures were tested — hard switch, revenue-mix, and dual-track; hard switch won because binary routing is cleaner than diluting a well-calibrated signal with noisy ICR data.
  </div>

  <table class="compare-table" style="margin-top:14px">
    <tr><th>Configuration</th><th>Sharpe</th><th>Ann Return</th><th>Max DD</th><th>IC@21d</th></tr>
    <tr><td><strong>Hard switch — full universe (v2)</strong></td><td class="win">0.331</td><td class="win">9.38%</td><td>−19.5%</td><td>0.143</td></tr>
    <tr><td>Non-PJM baseline (ERCOT/MISO/CAISO/SPP only)</td><td>0.272</td><td>7.99%</td><td>−15.4%</td><td>0.144</td></tr>
    <tr><td>Revenue mix — full universe</td><td>0.185</td><td>6.64%</td><td>−14.7%</td><td>0.134</td></tr>
    <tr><td>Dual-track — full universe</td><td class="lose">0.031</td><td class="lose">4.44%</td><td>−19.0%</td><td>0.130</td></tr>
    <tr><td>Original with PJM, broken signal</td><td class="lose">−0.027</td><td class="lose">3.69%</td><td>−15.7%</td><td>0.067</td></tr>
  </table>
</div>

<!-- Chart -->
<div class="chart-wrap">
  {chart_html}
  <div class="chart-caption">Solid line: hard-switch strategy (full 18-name universe including PJM). Dotted line: prior non-PJM baseline (ERCOT/MISO/CAISO/SPP only, 12 names). Backtest Apr 2018–Dec 2025, monthly rebalance, zero transaction costs.</div>
</div>

<!-- Thesis + Signal table -->
<div class="cols2">
  <div class="card">
    <h3>Investment Thesis</h3>
    <p>U.S. utility stocks are systematically mispriced around electricity grid stress events. A utility's physical exposure to grid stress — the topology of its transmission network, its generation mix, and its regulatory regime — determines how earnings respond to LMP spikes, congestion, and extreme weather. These structural exposures are slow to be arbitraged because they require combining grid physics with equity analysis.</p>
    <p>For <strong>merchant generators</strong> (VST, NRG), grid stress is an earnings event: high LMP = high revenue margin. For <strong>rate-regulated T&amp;D utilities</strong> (the PJM names and most MISO/CAISO/SPP companies), the signal that matters is financial fragility — utilities with thin interest coverage are most vulnerable when capital markets tighten during stress periods.</p>
    <div class="note-box">
      XLU buy-and-hold returned {XLU["cumulative"]} over the same period, versus the strategy's {STRATEGY["cumulative"]}. But XLU's max drawdown was {XLU["max_dd"]} versus {STRATEGY["max_dd"]} here, with annualised volatility ~20% vs ~13%. This strategy is designed as a risk-managed complement to long-only sector exposure — lower absolute return, substantially lower tail risk.
    </div>
  </div>
  <div class="card">
    <h3>Signal Construction</h3>
    <table class="signal-tbl">
      <tr><th>Step</th><th>Description</th></tr>
      <tr><td>1. Grid Stress Index</td><td>Daily composite: LMP z-score 40%, congestion fraction 25%, reserve tightness 20%, stress event flag 15%. Computed per ISO (ERCOT, PJM, MISO, CAISO, SPP).</td></tr>
      <tr><td>2. Stress events</td><td>Named events + algo-detected LMP spikes (97th pct, 3+ days) and congestion episodes (30% fraction, 10+ consecutive days).</td></tr>
      <tr><td>3. Business model routing</td><td><em>pass_through</em> coefficient tags each stock. Merchants (≥0.5) → stress beta path. Regulated utilities (&lt;0.5) → ICR signal path.</td></tr>
      <tr><td>4. Stress beta</td><td>Rolling OLS of excess stock return on GSI during event windows. Used for merchant/mixed tickers (VST, NRG).</td></tr>
      <tr><td>5. ICR signal</td><td>Interest coverage ratio (EBIT / interest expense). Used for regulated utilities — captures refinancing fragility. 45-day reporting lag applied.</td></tr>
      <tr><td>6. Factor score</td><td>Hard-switch routing + 15% renewable quality addon for all tickers. Cross-sectional z-score → rank.</td></tr>
      <tr><td>7. Portfolio</td><td>3 long / 5 short, equal-weight, monthly rebalance. Individual shorts — no ETF hedge.</td></tr>
    </table>
  </div>
</div>

<!-- Three Key Findings -->
<div class="cols3">
  <div class="card">
    <h3>Finding 1 — Business Model Routing Fixes PJM</h3>
    <p>Rate-regulated T&amp;D utilities were actively destroying alpha because LMP signals have no mechanistic link to regulated revenues. Routing them to an ICR signal raised Sharpe from 0.272 (no PJM) to <strong>0.331</strong> with all 18 names. The binary hard-switch outperformed continuous blending because mixing a well-calibrated stress beta with noisy quarterly ICR data reduces rather than improves signal quality.</p>
  </div>
  <div class="card">
    <h3>Finding 2 — Short Book Drives Alpha</h3>
    <p>All 81 configurations using an XLU sector-ETF hedge produced <strong>negative Sharpe</strong>. The strategy's alpha comes entirely from picking individual short candidates — names that are specifically fragile to grid stress or financial leverage — not from a sector-relative trade. The long book's top-scored names do not outperform XLU by enough to make a relative trade work. Use individual shorts.</p>
  </div>
  <div class="card">
    <h3>Finding 3 — Concentrate the Long Book</h3>
    <p>3 long names outperforms 5 or 7 at the same stress configuration. The duration filter (10+ consecutive days of congestion) carries most of the noise-reduction burden; a 30% congestion fraction threshold captures more valid signals than the original 50% design. The factor has enough cross-sectional dispersion that concentrating the long book adds return without proportionate drawdown increase.</p>
  </div>
</div>

<!-- Weaknesses + Future Work -->
<div class="cols2">
  <div class="card">
    <h3>Known Weaknesses</h3>
    <p><span class="tag tag-warn">Data depth</span><br>7 years, 3 meaningful stress regimes. Limited out-of-sample evidence.</p>
    <p><span class="tag tag-warn">Merchant narratives</span><br>VST and NRG are competitive generators whose returns can be driven by AI/data center demand narratives orthogonal to grid stress. IC degraded in 2024 when VST re-rated +262% on power contract news while the model held a short position.</p>
    <p><span class="tag tag-warn">ICR data depth</span><br>yfinance returns only 4–12 quarters of ICR history per ticker. The regulated-utility signal is weaker pre-2022. EDGAR XBRL sourcing is planned to extend history to 10+ years.</p>
    <p><span class="tag tag-warn">pass_through estimates</span><br>Mixed-ticker coefficients (ETR 0.35, EXC 0.10, DTE 0.15) are initial estimates pending verification against 10-K revenue disclosures.</p>
    <p><span class="tag tag-warn">Beta lag</span><br>Rolling OLS takes 1–2 quarters to reflect structural changes in how a stock responds to grid stress.</p>
    <p><span class="tag tag-warn">Transaction costs</span><br>Zero slippage assumed. Estimated Sharpe impact: 0.05–0.10.</p>
  </div>
  <div class="card">
    <h3>Roadmap — Phase 2 Signal Optimisation</h3>
    <p>The business-model-aware architecture is confirmed. Phase 2 tests alternative signals for the regulated-utility slot, in order of build effort:</p>
    <p><span class="tag tag-new">Next</span><br><strong>EDGAR ICR sourcing</strong> — replace yfinance ICR with EDGAR XBRL filings for 10+ year history, improving the regulated signal quality pre-2022.</p>
    <p><span class="tag tag-road">Near-term</span><br><strong>Data center load growth signal</strong> — PJM interconnection queue + EIA-861 load by customer class. Captures the AI demand wave: Dominion (Virginia), PSEG (New Jersey), and ComEd (Chicago) are primary beneficiaries. Most differentiated signal for the current macro theme.</p>
    <p><span class="tag tag-road">Near-term</span><br><strong>Rate case calendar signal</strong> — utilities with pending FERC/state PUC dockets carry earnings uncertainty; recently-approved cases have earnings momentum.</p>
    <p><span class="tag tag-road">Near-term</span><br><strong>Merchant regime filter</strong> �� rolling correlation of VST/NRG with XNG (energy ETF) vs XLU. Exclude from universe when the stock behaviorally "leaves the utility sector."</p>
    <p><span class="tag tag-road">Medium-term</span><br><strong>RT/DA LMP spread</strong> — real-time vs day-ahead price spread as a 5th GSI sub-signal, separating generator and T&amp;D utility dynamics within the same ISO.</p>
  </div>
</div>

<!-- Footer -->
<div class="footer">
  <strong>Disclaimer:</strong> This document presents historical backtest results only and does not constitute investment advice or a recommendation to buy or sell any security. Backtest performance is not indicative of future results. No representation is made regarding live trading performance. All trading decisions rest solely with the Client. Universe: {STRATEGY["universe"]}. Backtest period: {STRATEGY["backtest"]}. Risk-free rate: 4%. Transaction costs not modelled. Backtest only — not investment advice.
</div>

</div>
</body>
</html>"""

out_path = OUT / "executive_summary_v2.html"
out_path.write_text(html, encoding="utf-8")
print(f"Written: {out_path}")
print(f"Chart: {chart_label if chart_b64 else 'MISSING'}")
