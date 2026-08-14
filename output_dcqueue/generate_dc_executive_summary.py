"""
Generate output_dcqueue/dc_executive_summary.html

DC Load Signal (regulated-path Phase 2) — executive summary.
Written 2026-08-14 after a final-review fix pass corrected a Critical
zone-mapping bug in the first implementation. Numbers below are from the
corrected code, re-run on today's data. See docs/compact_2026-08-13-dc-load-signal-results.md
for the full account, including a separate, unrelated data-drift discovery
that means these numbers are not comparable to older CLAUDE.md/compact-doc
Sharpe figures.
"""

import base64
from pathlib import Path

# This script lives in output_dcqueue/ and both reads and writes there —
# the dc-queue backtest run's own artifacts (pnl.csv, dc_signal_comparison.png)
# sit alongside it in the same directory.
OUT = Path(__file__).parent

ICR = {
    "sharpe": "-0.276", "ann_return": "0.80%", "max_dd": "-21.93%",
    "ic21": "0.0615", "ic21_t": "1.26", "ic63": "0.1028", "ic63_t": "2.17",
}
DCQ = {
    "sharpe": "-0.213", "ann_return": "2.06%", "max_dd": "-19.63%",
    "ic21": "0.0290", "ic21_t": "0.77", "ic63": "0.0680", "ic63_t": "2.01",
}
XLU = {"sharpe": "0.162", "ann_return": "7.90%", "max_dd": "-36.47%"}
EW = {"sharpe": "0.216", "ann_return": "8.42%", "max_dd": "-38.78%"}

chart_b64 = ""
chart_path = OUT / "dc_signal_comparison.png"
if chart_path.exists():
    chart_b64 = base64.b64encode(chart_path.read_bytes()).decode()

chart_html = (
    f'<img src="data:image/png;base64,{chart_b64}" style="width:100%;border-radius:8px;">'
    if chart_b64 else
    '<p style="color:#888;text-align:center;padding:40px">Chart not found — run generate_dc_chart.py first</p>'
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DC Load Signal — Executive Summary</title>
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
  .badge-warn {{ background:var(--amber); color:#fff; margin-left:6px; }}

  .alert-box {{ background:#fdeeee; border:1px solid var(--red); border-left:4px solid var(--red);
                border-radius:0 8px 8px 0; padding:16px 20px; margin-bottom:24px; }}
  .alert-box h3 {{ color:var(--red); font-size:13px; font-weight:700; text-transform:uppercase;
                   letter-spacing:.5px; margin-bottom:8px; }}
  .alert-box p {{ font-size:13px; line-height:1.65; color:#334; margin-bottom:6px; }}
  .alert-box p:last-child {{ margin-bottom:0; }}

  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:10px; }}
  .kpi {{ background:var(--light); border:1px solid var(--border); border-radius:8px; padding:16px 14px; }}
  .kpi .label {{ font-size:11px; color:#667; text-transform:uppercase; letter-spacing:.5px; }}
  .kpi .value {{ font-size:22px; font-weight:700; color:var(--navy); margin:4px 0 2px; }}
  .kpi .bench {{ font-size:11px; color:#888; }}
  .kpis-caption {{ font-size:11px; color:#888; font-style:italic; margin-bottom:24px; }}

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

  .compare-table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:10px; }}
  .compare-table th {{ background:var(--navy); color:#fff; padding:7px 10px; text-align:left; font-size:11px; }}
  .compare-table td {{ padding:6px 10px; border-bottom:1px solid var(--border); }}
  .compare-table tr:last-child td {{ border-bottom:none; }}
  .compare-table .win {{ color:var(--green); font-weight:700; }}
  .compare-table .lose {{ color:var(--red); }}
  .compare-table .neutral {{ color:#a06000; font-weight:600; }}

  .note-box {{ background:#fff8e8; border-left:4px solid var(--amber); padding:12px 14px;
               border-radius:0 6px 6px 0; margin:12px 0; font-size:13px; line-height:1.6; }}

  table.rank-tbl {{ width:100%; border-collapse:collapse; font-size:12px; }}
  table.rank-tbl th {{ background:var(--navy); color:#fff; padding:6px 10px; text-align:left; font-weight:600; }}
  table.rank-tbl td {{ padding:5px 10px; border-bottom:1px solid var(--border); }}
  table.rank-tbl tr:nth-child(even) td {{ background:#f9fafb; }}
  table.rank-tbl .long {{ color:var(--green); font-weight:700; }}
  table.rank-tbl .short {{ color:var(--red); font-weight:700; }}

  .tag {{ display:inline-block; font-size:10px; font-weight:700; padding:2px 8px; border-radius:10px; margin-bottom:6px; }}
  .tag-warn {{ background:#fff0d0; color:#a06000; border:1px solid #e8c060; }}
  .tag-road {{ background:#e8f5e9; color:var(--green); border:1px solid #81c784; }}
  .tag-bug  {{ background:#fdeeee; color:var(--red); border:1px solid #e0a0a0; }}

  .chart-wrap {{ margin-bottom:24px; }}
  .chart-wrap .chart-caption {{ font-size:11px; color:#888; margin-top:6px; font-style:italic; }}

  .footer {{ border-top:1px solid var(--border); margin-top:28px; padding-top:16px; font-size:11px; color:#888; line-height:1.7; }}

  @media print {{
    body {{ font-size:12px; }}
    .page {{ padding:18px; max-width:100%; }}
  }}
</style>
</head>
<body>
<div class="page">

<!-- Header -->
<div class="header">
  <h1>DC Load Signal — Regulated-Path Phase 2</h1>
  <div class="sub">Replacing interest coverage ratio (ICR) with a data-center demand proxy for rate-regulated PJM utilities in the Grid Resilience Strategy's hard-switch factor architecture</div>
  <div class="sub" style="margin-top:8px">Backtest Apr 2018 – Dec 2025 &nbsp;·&nbsp; 18-stock universe, all 5 ISOs &nbsp;·&nbsp; Backtest only — not investment advice</div>
  <span class="badge badge-conf">Confidential</span>
  <span class="badge badge-warn">Data caveat — read before quoting any number</span>
</div>

<!-- Prominent data caveat -->
<div class="alert-box">
  <h3>⚠ These numbers are not comparable to prior sessions' documented results</h3>
  <p><strong>A pre-existing PJM data gap was found and fixed while building this feature:</strong> one month of PJM LMP history (2024-06) was silently missing from the cache until this session backfilled it. Re-running the unchanged ICR baseline on the now-complete cache alone moved Sharpe from the previously documented <strong>0.331</strong> to <strong>-0.276</strong>.</p>
  <p><strong>A second, broader drift was also found</strong> (likely yfinance equity-price data): excluding PJM entirely no longer reproduces the documented "without PJM" Sharpe of 0.272 either (now 0.165), even though that comparison is untouched by the PJM cache fix.</p>
  <p>Every number on this page is measured on today's data, with both configurations (ICR and DC load signal) run identically so the <em>relative</em> comparison between them is valid. The <em>absolute</em> levels should not be compared to CLAUDE.md or any compact doc dated before 2026-08-13. Re-validating the grid search and business-model-arch baselines on complete data is the top-priority follow-up — see Roadmap.</p>
</div>

<!-- KPI comparison -->
<div class="kpis">
  <div class="kpi">
    <div class="label">Sharpe — ICR</div>
    <div class="value">{ICR["sharpe"]}</div>
    <div class="bench">DC load: <strong>{DCQ["sharpe"]}</strong></div>
  </div>
  <div class="kpi">
    <div class="label">Ann. Return — ICR</div>
    <div class="value">{ICR["ann_return"]}</div>
    <div class="bench">DC load: <strong>{DCQ["ann_return"]}</strong></div>
  </div>
  <div class="kpi">
    <div class="label">Max Drawdown — ICR</div>
    <div class="value">{ICR["max_dd"]}</div>
    <div class="bench">DC load: <strong>{DCQ["max_dd"]}</strong></div>
  </div>
  <div class="kpi">
    <div class="label">IC@21d — ICR</div>
    <div class="value">{ICR["ic21"]}</div>
    <div class="bench">DC load: <strong>{DCQ["ic21"]}</strong> (weaker)</div>
  </div>
</div>
<div class="kpis-caption">XLU buy-and-hold over the same window: Sharpe {XLU["sharpe"]}, Ann Ret {XLU["ann_return"]}, Max DD {XLU["max_dd"]}. Equal-weight universe: Sharpe {EW["sharpe"]}, Ann Ret {EW["ann_return"]}, Max DD {EW["max_dd"]}.</div>

<!-- What Changed and Why -->
<div class="change-box">
  <h3>What This Signal Replaces, and Why</h3>
  <p><strong>The problem with ICR:</strong> the hard-switch architecture routes rate-regulated PJM utilities (AEP, D, EXC, FE, PPL, PEG) to an ICR (interest coverage ratio) signal instead of stress beta, because LMP has no mechanistic link to regulated revenues. ICR measures balance-sheet fragility under rate stress — it says nothing about whether a utility's <em>territory</em> is capturing the data-center demand wave, the dominant current theme in these names (D sits in Northern Virginia, the largest data-center market globally; PEG sits in the NJ corridor with Amazon/Google build-out).</p>
  <p><strong>The replacement:</strong> a two-component signal built from PJM's public interconnection generation queue (co-located generation build as a proxy for data-center demand — PJM has no direct load-interconnection queue) and PJM zone-level metered load as a utility-size denominator: <strong>Level (60%)</strong> = queued MW / zone size, <strong>Momentum (40%)</strong> = MW newly queued in the trailing 90 days / zone size.</p>

  <table class="compare-table" style="margin-top:14px">
    <tr><th>Configuration</th><th>Sharpe</th><th>Ann Return</th><th>Max DD</th><th>IC@21d</th><th>IC@63d</th></tr>
    <tr><td>Hard switch + ICR</td><td class="lose">{ICR["sharpe"]}</td><td>{ICR["ann_return"]}</td><td>{ICR["max_dd"]}</td><td class="win">{ICR["ic21"]} (t={ICR["ic21_t"]})</td><td class="win">{ICR["ic63"]} (t={ICR["ic63_t"]})</td></tr>
    <tr><td><strong>Hard switch + DC load signal</strong></td><td class="neutral">{DCQ["sharpe"]}</td><td class="neutral">{DCQ["ann_return"]}</td><td class="neutral">{DCQ["max_dd"]}</td><td>{DCQ["ic21"]} (t={DCQ["ic21_t"]})</td><td>{DCQ["ic63"]} (t={DCQ["ic63_t"]})</td></tr>
  </table>
  <p style="margin-top:10px"><strong>Mixed result — not a clean win.</strong> DC load signal improves Sharpe (+0.06), Ann Return (+1.3pp), and Max DD (+2.3pp) over ICR, but has weaker IC at both horizons. This does not clear the original design spec's success bar (IC improvement vs. ICR baseline).</p>
</div>

<!-- Chart -->
<div class="chart-wrap">
  {chart_html}
  <div class="chart-caption">Cumulative return, both configurations run identically on today's (corrected) data. Neither series should be compared to charts generated before 2026-08-13.</div>
</div>

<!-- The Critical Bug -->
<div class="cols2">
  <div class="card">
    <h3><span class="tag tag-bug">Bug — found and fixed this session</span><br>Zone-Code Mismatch</h3>
    <p>The first implementation silently mismatched PJM's zone-level load feed (short codes: <code>PS</code>, <code>PL</code>, <code>PE</code>, <code>BC</code>, <code>PEP</code>, <code>CE</code>, <code>JC</code>, <code>DAY</code>) against the ticker map's zone names (<code>PSEG</code>, <code>PPL</code>, <code>PECO</code>, <code>BGE</code>, <code>PEPCO</code>, <code>COMED</code>, <code>JCPL</code>, <code>DAYTON</code>). Only 3 of 6 intended PJM tickers were actually on the DC signal; EXC, PPL, and PEG silently fell through to ICR, and AEP/FE had numerator/denominator zone mismatches inflating their ratios.</p>
    <p>A first-draft result claimed "PEG flips to a long candidate, validating the hypothesis" — that was an artifact of PEG never touching the new signal. <strong>That claim is retracted.</strong> Fixed with a verified zone-alias map applied once at the fetch boundary, plus a loud warning on any future zone-mapping gap. All 6 PJM tickers now get real DC-signal values; numbers on this page are post-fix.</p>
  </div>
  <div class="card">
    <h3>Latest Rebalance — Where the 6 PJM Names Land (2025-12-31)</h3>
    <table class="rank-tbl">
      <tr><th>Ticker</th><th>Zones</th><th>Score</th><th>Rank of 18</th></tr>
      <tr><td>FE</td><td>ATSI, JCPL</td><td class="long">+1.842</td><td>1st (long)</td></tr>
      <tr><td>AEP</td><td>AEP, DAYTON</td><td>+0.432</td><td>5th</td></tr>
      <tr><td>D</td><td>DOM</td><td>+0.204</td><td>7th</td></tr>
      <tr><td>EXC</td><td>PECO, BGE, PEPCO, COMED</td><td>-0.183</td><td>9th</td></tr>
      <tr><td>PEG</td><td>PSEG</td><td class="short">-0.629</td><td>13th</td></tr>
      <tr><td>PPL</td><td>PPL</td><td class="short">-0.760</td><td>16th (short)</td></tr>
    </table>
  </div>
</div>

<!-- Findings -->
<div class="cols3">
  <div class="card">
    <h3>Finding 1 — PEG Does Not Validate</h3>
    <p>PEG (PSEG/NJ corridor), the clearest qualitative case for this signal, scores <strong>-0.629, 13th of 18</strong> on the real (post-fix) DC signal. The original "flips to long" claim was measuring ICR fallback data, not the DC signal. The hypothesis does not hold for PEG as implemented.</p>
  </div>
  <div class="card">
    <h3>Finding 2 — D Is Directionally Right, Weakly So</h3>
    <p>D (Dominion/Virginia — the largest data-center market in the universe) sign-flips from -1.44 pre-fix to <strong>+0.204</strong>, but stays mid-pack (7th of 18), well short of the long book. Direction is consistent with the thesis; magnitude does not clear the bar to call it validated.</p>
  </div>
  <div class="card">
    <h3>Finding 3 — FE Is a New, Unconfirmed Surprise</h3>
    <p>FE becomes the top-scoring name (+1.842), driven entirely by level (zero momentum — no new PJM filings in the trailing 90 days). 10,412 of its 13,932 MW numerator sits in one zone (JCPL) across 18 projects averaging 578 MW — plausibly large generation buildout, not confirmed as data-center-driven. Worth checking what those projects actually are before trusting this rank.</p>
  </div>
</div>

<!-- Weaknesses + Roadmap -->
<div class="cols2">
  <div class="card">
    <h3>Known Weaknesses</h3>
    <p><span class="tag tag-warn">Historical data drift</span><br>See the alert box above — every backtest number in this codebase predating 2026-08-13 should be treated as unverified until re-run.</p>
    <p><span class="tag tag-warn">MW Capacity not point-in-time</span><br>The interconnection queue's MW figure reflects PJM's current, possibly-amended project size, not what was known at each historical rebalance date — a real look-ahead bias, concentrated in the early (2018-2021) backtest years. No historical queue archive exists to fix this.</p>
    <p><span class="tag tag-warn">ICR coverage</span><br>The ICR baseline has almost no historical signal — usable data starts mid-2025 (last ~2 quarters of an 8-year backtest). The ICR-vs-DC-queue comparison above mostly measures 2025 behavior for the ICR row against full-history behavior for the DC row.</p>
    <p><span class="tag tag-warn">Unswept parameters</span><br>The 60/40 level/momentum weights and 100MW/90-day thresholds were asserted from reasoning, not grid-searched.</p>
    <p><span class="tag tag-warn">v1 is PJM-only</span><br>Non-PJM regulated tickers (WEC, CMS, AEE, PCG, EIX, XEL, EVRG) still fall back to ICR — this signal only differentiates within the 6 PJM names.</p>
  </div>
  <div class="card">
    <h3>Roadmap</h3>
    <p><span class="tag tag-road">Priority</span><br><strong>Re-validate historical baselines</strong> — re-run the grid search and hard-switch/revenue-mix/dual-track comparison on today's complete data before any further signal work. Nothing downstream of this should be trusted until it's done.</p>
    <p><span class="tag tag-road">Next</span><br><strong>Investigate FE's JCPL projects</strong> — pull project-level detail from the queue to confirm or rule out data-center-driven generation before treating FE as a genuine signal winner.</p>
    <p><span class="tag tag-road">Near-term</span><br><strong>Parameter sweep</strong> on level/momentum weights and MW/day thresholds, mirroring the existing grid-search process, once baselines are re-validated.</p>
    <p><span class="tag tag-road">Medium-term</span><br><strong>Extend beyond PJM</strong> via EIA-861 realized load-by-customer-class, and evaluate PJM's annual "Large Load Additions" workshop PDFs for a true (not generation-proxy) load-side signal.</p>
  </div>
</div>

<!-- Footer -->
<div class="footer">
  <strong>Disclaimer:</strong> This document presents historical backtest results only and does not constitute investment advice or a recommendation to buy or sell any security. Backtest performance is not indicative of future results. Numbers on this page reflect a data-quality discovery made 2026-08-13/14 and are not comparable to earlier documents in this repository. See docs/compact_2026-08-13-dc-load-signal-results.md for the full technical account. Universe: 18 stocks, 5 ISOs. Backtest period: Apr 2018 – Dec 2025. Transaction costs not modelled. Backtest only — not investment advice.
</div>

</div>
</body>
</html>"""

out_path = OUT / "dc_executive_summary.html"
out_path.write_text(html, encoding="utf-8")
print(f"Written: {out_path}")
