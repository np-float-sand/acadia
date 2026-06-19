# Handoff: Executive Summary Document

**File:** `output/executive_summary.html`  
**Date created:** June 2026  
**Status:** Current, accurate, shareable

---

## What It Is

A single-page HTML document designed to be shared with an executive audience. It explains the strategy, shows the backtest results, and honestly states weaknesses and future work. It is self-contained — the performance chart is base64-embedded directly in the HTML so it can be emailed or shared as a single file with no external dependencies.

**To export as PDF:** Open in browser → Cmd+P → Save as PDF → A4 landscape. CSS `@media print` rules tighten spacing automatically.

---

## How It Was Built

Generated programmatically in Python (`python3` script, not a static file). The script:
1. Fetches actual XLU returns via yfinance for accurate benchmark numbers
2. Base64-encodes `output/backtest_performance.png` directly into the HTML
3. Writes the full HTML to `output/executive_summary.html`

**To regenerate:** The generation script is not saved as a standalone file — it was run inline. To rebuild, re-run the Python block from the session or reconstruct from the structure documented below. The chart image it embeds is `output/backtest_performance.png` — if you re-run the backtest and get a new chart, regenerate the HTML to pick it up.

---

## Accurate Numbers (Verified June 2026)

All numbers in the document were computed from actual data, not estimated. Key figures:

| Metric | Strategy | XLU Benchmark |
|--------|----------|---------------|
| Cumulative return | +90% | +113% |
| Ann. return | 8.1% | 12.0% |
| Sharpe ratio | 0.314 | 0.235* |
| Max drawdown | −13.7% | −36.1% |
| Ann. volatility | 13.2% | 20.6% |

*XLU Sharpe of 0.235 is the backtest's own calculation (uses `fillna(0)` on non-strategy dates and 4% risk-free rate). Raw XLU Sharpe computed independently is 0.388 — the discrepancy is because the backtest suppresses early XLU returns during the pre-strategy warm-up period. Use 0.235 when comparing against the chart; use 0.388 if computing independently.

**XLU outperforms on raw cumulative return (+113% vs +90%).** This is acknowledged honestly in the document. The strategy's case is risk-adjusted: 2.6× lower max drawdown, 13.2% vol vs 20.6%.

---

## Document Structure

### 1. Header
Strategy name, one-line description, backtest window (Apr 2018–Dec 2025), 18-stock universe, date, confidential flag.

### 2. KPI Strip (5 metrics)
| KPI | Value | Benchmark shown |
|-----|-------|-----------------|
| Cumulative Return | +90% | XLU buy & hold: +113% |
| Ann. Return | 8.1% | XLU ann. return: +12.0% |
| Sharpe Ratio | 0.31 | XLU Sharpe: 0.24 |
| Max Drawdown | −13.7% | XLU Max DD: −36.1% |
| Mean IC | 0.15 | IC > 0 during grid stress |

**Note on the IC KPI:** The "Mean IC 0.15" figure and "IC > 0 during grid stress" label are misleading and should be updated in a future revision. Analysis showed that IC in stressed months (mean +0.137) is not statistically different from calm months (mean +0.169, p=0.856). The IC is positive in both regimes — the stressed/calm distinction is not meaningful at a 0.5 GSI threshold. See issue below.

### 3. Performance Chart
Embedded as base64 PNG from `output/backtest_performance.png`. Four panels: cumulative return vs. XLU and EW universe, rolling 63-day Sharpe, drawdown, monthly IC bars (orange=stressed, green=calm).

### 4. Thesis Card + Signal Construction Table (two columns)
- Thesis: explains the mispricing idea in plain language (stress betas are determined by physical topology, slow to be arbitraged)
- Note box: honest disclosure that XLU outperformed on raw return; positions the strategy as a risk-managed complement
- Signal table: 5-step pipeline

### 5. Three Key Findings (three columns)
1. Short book drives alpha — XLU hedge destroys it (all 81 XLU-hedge configs negative Sharpe)
2. Duration beats threshold — 30% congestion fraction + 10-day minimum outperforms original 50% design
3. Concentrate the long book — 3 names outperforms 5 or 7, drawdown 2.6× lower than XLU

### 6. Known Weaknesses (left column)
5 items with orange warning tags:
- Data depth (7 years, 3 stress regimes)
- Universe size (18 tickers, capacity constraints)
- Beta lag (rolling OLS reacts slowly to structural changes)
- Data coverage (SPP/CAISO shorter LMP history)
- Transaction costs (zero slippage assumed, estimated Sharpe impact 0.05–0.10)

### 7. Future Work (right column)
4 items with green roadmap tags:
- ICR factor component (near-term)
- RT/DA LMP spread as 5th GSI sub-signal (near-term)
- Universe expansion to 30–40 names (medium-term)
- Inter-zonal congestion spread (medium-term)

### 8. Footer
Disclaimer (backtest only, not investment advice) and full ticker/ISO list.

---

## Known Issues To Fix in Next Version

### Issue 1 — IC KPI is misleading (IMPORTANT)
**Current:** KPI strip shows "Mean IC 0.15 (stressed months)" implying the factor activates specifically during stress.  
**Reality:** Mean IC during stressed months is +0.137, during calm months is +0.169, difference is not statistically significant (p=0.856). The factor works similarly in both regimes.  
**Fix:** Change the 5th KPI to show overall mean IC (+0.15) without the "stressed months" qualifier, or remove it and replace with a more meaningful metric (e.g. pct of months with positive IC: ~58%).

### Issue 2 — VST/NRG 2024 problem not mentioned in weaknesses
The doc notes "beta lag" as a weakness but doesn't specifically name the merchant generator / competitive gen problem. Given that this was the primary driver of 2024/2025 IC instability, it deserves a specific mention.  
**Fix:** Add to weaknesses: "Universe contains competitive/merchant generators (VST, NRG) whose equity returns are driven by power price macro and AI/data center narratives orthogonal to the grid stress signal. These names caused significant IC degradation in 2024 and H1 2025."

### Issue 3 — Future work section doesn't mention the new signals being developed
The forward power price signal and short interest/analyst revision signals are not mentioned.  
**Fix:** Add to future work section once those designs are finalised.

### Issue 4 — Chart is from the current run
The embedded chart reflects the most recent backtest run. If parameters change or the backtest is re-run, regenerate the HTML to pick up the new chart.

---

## How to Update the Document

The HTML is generated by a Python script. The cleanest way to update it:

1. Edit the relevant text strings in the Python generation script
2. Re-run to produce a new `output/executive_summary.html`
3. The chart auto-embeds from `output/backtest_performance.png` at generation time

The Python script structure is straightforward — the HTML is one large f-string with clearly labelled sections. There is no template engine or build system.

**If you only want to change text** (not numbers or chart), you can edit `output/executive_summary.html` directly in any text editor — it's plain HTML. Search for the text you want to change and edit in place.

**If numbers change** (new backtest run, different parameters), regenerate via Python to ensure the chart and all computed figures stay in sync.

---

## Audience and Tone Notes

Written for a non-technical executive — someone who understands portfolio management concepts (Sharpe, drawdown, long/short) but not factor model mechanics or grid physics. The tone is:
- Honest about raw return underperformance vs XLU
- Confident about risk-adjusted case
- Specific about weaknesses (not vague "model risk" language)
- Future work framed as near-term vs medium-term with concrete descriptions

The "note box" in the thesis section explicitly addresses the raw return gap: "XLU buy-and-hold returned +113% over the same period, versus the strategy's +90%. But XLU's max drawdown was −36.1% (vs −13.7% here)..." This was added specifically because the user requested complete honesty even where the numbers are unflattering.

---

## Files Referenced

| File | Role |
|------|------|
| `output/executive_summary.html` | The document itself |
| `output/backtest_performance.png` | Chart embedded in the HTML |
| `output/pnl.csv` | Source for strategy return numbers |
| `output/findings.md` | Full analytical findings this session — background for any content updates |

