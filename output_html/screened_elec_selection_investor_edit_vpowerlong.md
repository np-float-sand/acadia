# Screened Electrification Equipment - v2
Date: 2026-10-12

> **In brief.** Electricity demand is growing faster than the grid can carry it. Regulated utilities
> pay for the new wires, transformers and substations.  Equipment makers and contractors that supply
> them are paid as the work is delivered. The strategy takes advantage of the opposite positioning on the balance sheet. It holds the best-scoring the suppliers and shorts the weakest of electrification companies, mostly the
> utilities and pipelines doing the paying. The selection is based on a consolidated score of
> fourteen measures (power market sensitivity, capital intensity, revenue growth/acceleration, earnings and revenue surprises, and
> price trends).
>
> The grid stress that took hold in 2023 looks durable.\*  Utilities have committed to multi-year capital
> plans, equipment lead times run to years, and real yields remain elevated. The suppliers' own
> filings show no slowdown: their median revenue growth is 12.0% a year after inflation and still
> accelerating, the fastest since early 2023. Over that period the long/short has delivered a Sharpe
> ratio of 2.28 after trading costs, with a beta of 0.47 to the S&P 500. 
> 
>This is a regime-dependent strategy that will profit during the grid-stress regime. The strategy writeup includes some gradual/partial mitigants and a full step-aside.

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

\*An inidicative public signal came on 29 November 2022, when PJM, the grid operator for 13 mid-Atlantic
and Midwestern states and DC, added a separate "Data Centers" adjustment to its load forecast for the
AEP, APS and Dominion utility zones. Its January 2023 forecast raised the data-center load it expects
by 2030 from 4.8 GW to 12.8 GW; the January 2025 forecast raised it to 32.8 GW.

</div>



---

## Performance

![Sharpe ratio of the strategy, GRID, PAVE, the S&P 500 and VOLT](img/tophalf_powerlong/inv_sharpe.png)

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

- **Bars:** Sharpe ratio (return per unit of risk, after trading costs) over the full period (light) and
  since the grid became tight in 2023 (dark).
- **\*VOLT** (an electrification ETF) launched in December 2024; its bar and line cover only its own
  21-month history.

</div>

<p><img src="img/tophalf_powerlong/inv_rolling.png" alt="Rolling 12-month Sharpe with the regime indicators" style="width:min(72rem,94vw);max-width:none;position:relative;left:50%;transform:translateX(-50%)"></p>

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

- **Lines:** Sharpe over the trailing 12 months. The shaded labels mark the market environment behind
  each stretch.
- **Solid blue:** the strategy, shown where its trailing year was fully invested.
- **Dashed blue:** the same strategy had the step-aside rule kept it invested, shown where the real strategy was
  partly or wholly in T-bills (a year mostly in cash has no meaningful Sharpe). It lagged GRID through
  2020–21 and turned negative in 2022 — what the step-aside rule stepped around — and joins the solid line once the
  strategy has been fully invested for a year.
- **Hatched grey:** time in T-bills (1 November 2019 – 29 April 2022, plus two brief spells in August 2019).
- **Amber dotted line (right axis):** median revenue growth, after inflation, of the suppliers that pass
  the long book's rules, from their filings as published — a proxy for the buildout. It measures
  demand, not valuations.
- **Red dotted line (right axis):** the 10-year real yield.
- **Zero lines:** the Sharpe, supplier-growth and real-yield axes share one zero line, so both dotted
  lines above it means the regime is green.
- **Regime strip:** green when both the real yield and the suppliers' real revenue growth are above
  zero — the only time the step-aside rule keeps the strategy invested — amber when one is, red when neither. Amber
  or red from late 2019 to spring 2022; green without a break since.

</div>

### Tight-Grid Period (January 2023 – August 2026)

| | **Long/short (the strategy)** | Without step-aside | GRID ETF | PAVE ETF | S&P 500 | VOLT ETF* |
|---|---|---|---|---|---|---|
| *Sharpe ratio* | **2.28** | **2.28** | 0.93 | 0.91 | 1.16 | 0.78 |
| *Calmar ratio* | **4.57** | **4.57** | 1.11 | 0.87 | 1.20 | 0.97 |
| *Max drawdown* | **−8.5%** | **−8.5%** | −20.8% | −26.2% | −18.8% | −23.4% |
| *Volatility* | **13.0%** | **13.0%** | 20.3% | 20.6% | 15.0% | 25.5% |
| *Beta to the S&P 500* | **0.47** | **0.47** | 1.10 | 1.09 | 1.00 | 1.00 |
| *Return a year* | **38.6%** | **38.6%** | 23.0% | 22.8% | 22.6% | 22.8% |
| *Max time under water* | 4.3 months | 4.3 months | 7.6 months | 7.9 months | **4.2 months** | 5.3 months |
| *Return / Sharpe after trading costs* | 38.6% / 2.28 | 38.6% / 2.28 | | | | |
| *Return / Sharpe before trading costs* | 39.2% / 2.31 | 39.2% / 2.31 | | | | |

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*\*VOLT: December 2024 – August 2026 only (21 months). Over that same window the strategy's Sharpe
was 1.82, its max drawdown −8.5% and its return 27.8% a year.*

</div>

### Earlier Period (January 2019 – December 2022)

| | **Long/short (the strategy)** | Without step-aside | GRID ETF | PAVE ETF | S&P 500 |
|---|---|---|---|---|---|
| *Sharpe ratio* | 0.39 | **0.77** | 0.77 | 0.60 | 0.48 |
| *Calmar ratio* | 0.61 | **0.88** | 0.58 | 0.43 | 0.39 |
| *Max drawdown* | **−12.1%** | −17.1% | −40.6% | −44.1% | −33.7% |
| *Volatility* | **9.0%** | 14.4% | 27.4% | 29.7% | 22.5% |
| *Beta to the S&P 500* | **0.11** | 0.32 | 1.06 | 1.13 | 1.00 |
| *Return a year* | 7.4% | 15.1% | **23.5%** | 19.1% | 13.1% |
| *Max time under water* | 35.2 months | 13.2+ months | 13.3+ months | 11.8+ months | 11.9+ months |
| *Return / Sharpe after trading costs* | 7.4% / 0.39 | 15.1% / 0.77 | | | |
| *Return / Sharpe before trading costs* | 7.8% / 0.44 | 15.8% / 0.82 | | | |

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*Sharpe uses a flat **4.0% risk-free rate**. Returns are total returns, dividends reinvested. The
strategy's figures are after 10bp on every trade — long and short name changes, resizing, sleeve and
short resets, and each step-aside switch — and 1% a year borrow on the short ([Costs](#running-costs));
the "before trading costs" row removes the 10bp. ETF returns are net of their own fees. Max time under water is the longest stretch below a previous peak
within the window; "+" means the drawdown was still unrecovered at the end of 2022.*

</div>

**This is a grid-tightness strategy.** It does not work well when the grid has ample spare capacity.  The strategy deals with this in 4 ways:

*Partial/Gradual*

1. Stretch de-lever:  When the book runs more than 15% above its 200-day average, it is cut to 80% and the rest earns T-bill interest. This is intended to slow down when the book is more likely to be near the peak.  
2. Volatility target:  When markets become choppy, it targets a 20% vol target for the book and moves the rest into T-bills.  This is intended to guard against a sharp change into an oversupplied grid regime.  
3. The strategy inherently rotates toward suppliers whose sales are still growing.

*Full switch-off*

4. Step-aside: the whole portfolio goes into T-bills whenever the either the real yield is below zero or suppliers' real revenue growth is below zero (in plot, regime strip isn't green).



**Since 2023 the strategy leads every benchmark on every measure but one** — time unnder water. (But before 2023, this was not the appropriate grid-tightness regime that would serve the strategy well)

<p><img src="img/tophalf_powerlong/inv_growth.png" alt="Growth of $1 with the regime strip" style="width:min(72rem,94vw);max-width:none;position:relative;left:50%;transform:translateX(-50%)"></p>

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

- **VOLT's line** starts on the strategy's line at VOLT's launch, so the two can be compared from that
  date.
- **Hatched grey:** time in T-bills under the step-aside rule. **Dashed blue:** the strategy without the step-aside rule.
- **Amber and red dotted lines (right axes):** the buildout proxy and the 10-year real yield, as
  above; their zero lines are aligned, so where both sit above the shared zero the regime is green.
- **Regime strip:** as above.
- Over the whole period GRID compounded to 4.95×, behind the strategy without the step-aside (5.77×) and ahead of it with the step-aside (4.37×); PAVE finished at 4.25×. Since 2023 the strategy compounded fastest.

</div>

---

## Portfolio Construction

Both long and short legs begin by restricting to reasonably liquid eqectrical equipment and power related names (by industry code and keyword).  The long leg restricts further to supplier (not owner), profitable, non-subsidy.  the short book chooses its candidates from teh rest.  For both we develop a score as in the table below. The long book holds the top portion (half) of its screened suppliers, the short the bottom portion (fifth) of its eligible set. The long book adds risk control, and one rule decides when to step aside. 

<table style="border-collapse:collapse; width:100%; font-size:0.75em;">
<thead>
<tr style="border-bottom:2px solid #999;">
  <th style="width:2.4em;"></th>
  <th style="text-align:left; padding:6px 8px;">Aspect</th>
  <th style="text-align:left; padding:6px 8px;">Long book</th>
  <th style="text-align:left; padding:6px 8px;">Short book</th>
  <th style="text-align:left; padding:6px 8px;">Source</th>
</tr>
</thead>
<tbody>

<!-- SCREEN -->
<tr style="border-top:3px solid #fff;">
  <td rowspan="6" style="background:#dbe8fb; color:#1f3a63; font-weight:700; letter-spacing:0.12em; text-align:center; vertical-align:middle; padding:8px 4px;"><div style="writing-mode:vertical-rl; transform:rotate(180deg); margin:auto;">SCREEN</div></td>
  <td style="padding:8px; vertical-align:top;"><b>Electrification population</b></td>
  <td colspan="2" style="padding:8px; vertical-align:top; background:rgba(127,127,127,0.06);">Companies in the SEC's electrical-equipment and power industry codes, plus nuclear fuel-cycle names found by keyword; at least $10m of stock traded on a typical day over the past year and a full year of price history (about 200 names)</td>
  <td style="padding:8px; vertical-align:top;"><ul style="margin:0; padding-left:1.1em;"><li>Industry codes: SEC EDGAR</li><li>Keywords: SEC EDGAR full-text search</li><li>Prices and volume: Yahoo Finance</li></ul></td>
</tr>
<tr style="border-top:1px solid #ddd;">
  <td style="padding:8px; vertical-align:top;"><b>On-theme</b></td>
  <td colspan="2" style="padding:8px; vertical-align:top; background:rgba(127,127,127,0.06);">The 10-K annual report used <i>data center, substation, switchgear, transmission and distribution, grid modernization</i> or <i>electric utilities</i> in the previous four years. <i>Data center</i> counts only where the business description uses it for customers or markets, not for the company's own IT systems. Gas pipeline and gas distribution companies don't qualify: their <i>transmission and distribution</i> is gas, not the power grid</td>
  <td style="padding:8px; vertical-align:top;">SEC EDGAR full-text search of 10-Ks</td>
</tr>
<tr style="border-top:1px solid #ddd;">
  <td style="padding:8px; vertical-align:top;"><b>Supplier, not owner</b></td>
  <td style="padding:8px; vertical-align:top;">Capital spending ÷ sales in the lightest 60% of the population, which separates suppliers from the utilities that own the wires</td>
  <td style="padding:8px; vertical-align:top;">--</td>
  <td style="padding:8px; vertical-align:top;">SEC EDGAR XBRL filings</td>
</tr>
<tr style="border-top:1px solid #ddd;">
  <td style="padding:8px; vertical-align:top;"><b>Profitable</b></td>
  <td style="padding:8px; vertical-align:top;">Trailing-12-month net income above zero, using only figures already filed at the time</td>
  <td style="padding:8px; vertical-align:top;">--</td>
  <td style="padding:8px; vertical-align:top;">SEC EDGAR XBRL filings</td>
</tr>
<tr style="border-top:1px solid #ddd;">
  <td style="padding:8px; vertical-align:top;"><b>Not subsidy-dependent</b></td>
  <td style="padding:8px; vertical-align:top;">No 10-K in the previous four years mentions the <i>investment tax credit, production tax credit, advanced manufacturing production credit</i> or <i>net energy metering</i></td>
  <td style="padding:8px; vertical-align:top;">--</td>
  <td style="padding:8px; vertical-align:top;">SEC EDGAR full-text search of 10-Ks</td>
</tr>
<tr style="border-top:1px solid #ddd;">
  <td style="padding:8px; vertical-align:top;"><b>Not held long</b></td>
  <td style="padding:8px; vertical-align:top;">--</td>
  <td style="padding:8px; vertical-align:top;">A company cannot be long and short at once: names the long book holds are removed before the short basket is picked</td>
  <td style="padding:8px; vertical-align:top;">--</td>
</tr>

<!-- RANK -->
<tr style="border-top:3px solid #fff;">
  <td rowspan="2" style="background:#d9f0e1; color:#1d4d2e; font-weight:700; letter-spacing:0.12em; text-align:center; vertical-align:middle; padding:8px 4px;"><div style="writing-mode:vertical-rl; transform:rotate(180deg); margin:auto;">RANK</div></td>
  <td style="padding:8px; vertical-align:top;"><b>One score</b></td>
  <td colspan="2" style="padding:8px; vertical-align:top; background:rgba(127,127,127,0.06);"><ul style="margin:0; padding-left:1.1em;">
    <li><b>From filings:</b> capital intensity, revenue acceleration, revenue growth, revenue and earnings surprises, growth in capital spending</li>
    <li><b>From prices:</b> 12-month momentum, 1-month reversal</li>
    <li><b>From power markets:</b> two signals for each measure below: the stock's 36-month sensitivity to its shocks (who benefits structurally) and that sensitivity × the latest 3-month shock (who is benefiting right now)<ul style="margin:0; padding-left:1.1em;">
      <li><i>Wholesale prices:</i> mean day-ahead LMP, PJM hubs and ERCOT hub; benefits from higher power prices</li>
      <li><i>Congestion:</i> mean absolute day-ahead congestion, PJM zones; benefits from a constrained grid</li>
      <li><i>Scarcity:</i> 99th-percentile hourly day-ahead price, ERCOT hub; benefits from price spikes</li></ul></li>
    <li>Average the fourteen standardised scores</li></ul></td>
  <td style="padding:8px; vertical-align:top;"><ul style="margin:0; padding-left:1.1em;">
    <li>Filings: SEC EDGAR XBRL filings</li>
    <li>Prices: Yahoo Finance</li>
    <li>Power: PJM and ERCOT day-ahead market prices</li></ul></td>
</tr>
<tr style="border-top:1px solid #ddd;">
  <td style="padding:8px; vertical-align:top;"><b>Selection</b></td>
  <td style="padding:8px; vertical-align:top;">Hold the top half each quarter; a held name stays while in the top 65%</td>
  <td style="padding:8px; vertical-align:top;">Short the bottom fifth, re-formed each quarter (13 names on average, never fewer than 10, 18 today)</td>
  <td style="padding:8px; vertical-align:top;">--</td>
</tr>

<!-- WEIGHT -->
<tr style="border-top:3px solid #fff;">
  <td style="background:#fbefd0; color:#5c4410; font-weight:700; letter-spacing:0.12em; text-align:center; vertical-align:middle; padding:8px 4px;"><div style="writing-mode:vertical-rl; transform:rotate(180deg); margin:auto;">WEIGHT</div></td>
  <td style="padding:8px; vertical-align:top;"><b>Equal weight</b></td>
  <td style="padding:8px; vertical-align:top;">Each name weighted equally</td>
  <td style="padding:8px; vertical-align:top;">Each name weighted equally; $0.25 short for every $1 of capital, reset each month</td>
  <td style="padding:8px; vertical-align:top;">--</td>
</tr>

<!-- RISK CONTROL -->
<tr style="border-top:3px solid #fff;">
  <td rowspan="3" style="background:#f6dcdc; color:#5e2323; font-weight:700; letter-spacing:0.12em; text-align:center; vertical-align:middle; padding:8px 4px;"><div style="writing-mode:vertical-rl; transform:rotate(180deg); margin:auto;">RISK CONTROL</div></td>
  <td style="padding:8px; vertical-align:top;"><b>Volatility target</b></td>
  <td style="padding:8px; vertical-align:top;">Size the book daily to target 20% annual volatility, never above 1.5×; resize only on moves of more than 20 points of exposure</td>
  <td rowspan="3" style="padding:8px; vertical-align:middle; color:var(--ink-soft);">None of its own: the short stays at $0.25 per $1 of capital while these controls shrink the long side, so the book moves toward market-neutral as it de-risks</td>
  <td style="padding:8px; vertical-align:top;">--</td>
</tr>
<tr style="border-top:1px solid #ddd;">
  <td style="padding:8px; vertical-align:top;"><b>Stretch de-lever</b></td>
  <td style="padding:8px; vertical-align:top;">Cut to 80% when the book is over 15% above its 200-day average or far ahead of the market; 60% at extremes</td>
  <td style="padding:8px; vertical-align:top;">--</td>
</tr>
<tr style="border-top:1px solid #ddd;">
  <td style="padding:8px; vertical-align:top;"><b>Gold / Treasury sleeve</b></td>
  <td style="padding:8px; vertical-align:top;">15% in GLD and IEF, reset monthly as a macro hedge against growth scares</td>
  <td style="padding:8px; vertical-align:top;">--</td>
</tr>

</tbody>
</table>

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*Shaded cells span both books: the rule is the same for each.*

*Every input is point in time: company figures are used only after they were filed, the 10-K keyword
checks read only filings made before each year's screen, CPI is lagged a month for its release date,
and the real yield and supplier growth enter the step-aside rule a day late.*

</div>

### Drawdown Reduction by Control

Each row adds one control on top of the rows above it. These figures are for the long book alone,
before the short side, stepping aside and trading costs.

![Reduction in drawdown from each incremental control, Jan 2019 – Aug 2026](img/tophalf_powerlong/overlay.png)

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*Chart: full period, Jan 2019 – Aug 2026.*

</div>

**2023 – Aug 2026**

<table style="border-collapse:collapse; font-size:0.75em;">
<thead><tr><th style="text-align:left; padding:4px 10px; border-bottom:2px solid #999;">cumulative controls</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">Sharpe</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">return a year</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">worst drawdown</th></tr></thead>
<tbody>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">Selected names, no risk control</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.60</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>62.7%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−33.5%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ volatility target</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.77</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">49.5%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−21.7%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ stretch de-lever</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.95</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">43.6%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−19.4%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ gold/Treasury sleeve </td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>2.03</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">39.2%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>−15.9%</b></td></tr>
</tbody>
</table>

**Full period, Jan 2019 – Aug 2026**

<table style="border-collapse:collapse; font-size:0.75em;">
<thead><tr><th style="text-align:left; padding:4px 10px; border-bottom:2px solid #999;">cumulative controls</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">Sharpe</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">return a year</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">worst drawdown</th></tr></thead>
<tbody>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">Selected names, no risk control</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.12</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>40.9%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−47.5%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ volatility target</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.32</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">35.1%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−24.3%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ stretch de-lever</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.35</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">31.9%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−24.3%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ gold/Treasury sleeve </td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>1.39</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">28.6%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>−20.2%</b></td></tr>
</tbody>
</table>

### Short Book and Step-Aside Contribution

Each row adds one piece on top of the rows above it. All figures are after trading costs and borrow,
which is why the first row differs slightly from the long-book figures above.

**2023 – Aug 2026**

<table style="border-collapse:collapse; font-size:0.75em;">
<thead><tr><th style="text-align:left; padding:4px 10px; border-bottom:2px solid #999;">cumulative</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">Sharpe</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">return a year</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">worst drawdown</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">beta to S&P 500</th></tr></thead>
<tbody>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">Long book, always invested</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.93</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>39.5%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−16.0%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">0.72</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ short side</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>2.28</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">38.6%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>−8.5%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>0.47</b></td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ stepping aside (the strategy)</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>2.28</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">38.6%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>−8.5%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>0.47</b></td></tr>
</tbody>
</table>

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*The strategy stayed invested throughout 2023 – Aug 2026, so stepping aside changes nothing in this window.*

</div>

**Full period, Jan 2019 – Aug 2026**

<table style="border-collapse:collapse; font-size:0.75em;">
<thead><tr><th style="text-align:left; padding:4px 10px; border-bottom:2px solid #999;">cumulative</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">Sharpe</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">return a year</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">worst drawdown</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">beta to S&P 500</th></tr></thead>
<tbody>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">Long book, always invested</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.29</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>27.6%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−22.9%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">0.62</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ short side</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>1.45</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">25.8%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−17.1%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">0.36</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ stepping aside (the strategy)</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.43</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">21.3%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>−12.1%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>0.21</b></td></tr>
</tbody>
</table>



---

## Performance History

### Annual Returns

![Calendar-year returns](img/tophalf_powerlong/inv_years.png)

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*A year spent in T-bills has no
meaningful Sharpe. The hatched blue bars show the return had the step-aside rule kept the strategy invested, for
each year it spent partly or wholly in T-bills (2019: the last two months; 2022: the first four).*

</div>

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
|---|---|---|---|---|---|---|---|---|
| **Long/short** | 27.4% | 0.4% *(T-bills)* | 0.1% *(T-bills)* | 4.0% | 45.5% | 44.7% | 40.9% | 10.8% |
| *Long/short had the step-aside rule kept it invested* | *39.1%* | *11.3%* | *20.9%* | *−6.2%* | *45.5%* | *44.7%* | *40.9%* | *10.8%* |
| GRID ETF | 42.2% | 48.8% | 27.7% | −13.9% | 21.6% | 15.2% | 29.7% | 17.1% |
| PAVE ETF | 32.6% | 19.7% | 36.4% | −7.2% | 31.0% | 17.9% | 19.4% | 14.7% |
| S&P 500 | 31.1% | 18.3% | 28.7% | −18.2% | 26.2% | 24.9% | 17.7% | 13.1% |
| Long/short Sharpe | 1.46 | *in T-bills* | *in T-bills* | 0.04 | 2.53 | 2.42 | 2.43 | 1.24 |
| Long/short worst drawdown | −12.1% | 0.0% | 0.0% | −9.3% | −8.0% | −6.7% | −8.5% | −4.4% |

The strategy beat GRID in every year from 2022 to 2025.  While it trails it so far in 2026, when the theme
itself has rallied hard, as a long/short strategy, it is also lower beta.  Should hedging out the market move not be of interest, trading the long arm as 'smart beta' strategy is also valuable.

Over its one episode the step-aside rule cost return — it gave up 12 points in late 2019, 11 in 2020 and
21 in 2021, and saved 10 in 2022 — in exchange for holding the strategy flat through the COVID crash and
the 2022 rate shock, which cut the worst drawdown from −17.1% to −12.1%.

### Drawdowns

![Drawdown](img/tophalf_powerlong/inv_drawdown.png)

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

- **Dashed blue:** the same strategy had the step-aside rule kept it invested. It differs from the solid
  line only from November 2019 to April 2022, where it fell as far as −17% while the strategy sat flat in T-bills.

</div>

<!--
---

## Regime Shifts

The step-aside rule exists because the strategy is highly regime-dependent. Two conditions mark that market, and the strategy stays invested only while both hold.

- Real yields above zero: When the 10-year real yield is negative, money is effectively free, and
  the market pays up for distant earnings, subsidy-dependent growth and story stocks — everything the
  rules exclude. When it is positive, capital has a cost, and suppliers that are paid now are worth
  more than companies spending heavily to be paid later.
- Suppliers' revenue still growing after inflation: When the median supplier's real sales are
  shrinking, the buildout is not reaching the companies the strategy owns, and the ranking has little
  to separate winners from losers.

| | Sharpe, full | Sharpe, 2019–22 | Sharpe, 2023 on | max drawdown, full | Calmar, full | CAGR | time invested |
|---|---|---|---|---|---|---|---|
| **Step-aside rule: both conditions (the strategy)** | 1.43 | 0.39 | 2.28 | **−12.1%** | **1.76** | 21.3% | 67% |
| Real yield alone | 1.40 | 0.36 | 2.28 | **−12.1%** | **1.76** | 21.2% | 70% |
| Always invested | **1.45** | **0.77** | 2.28 | −17.1% | 1.51 | **25.8%** | 100% |

- Step-aside costs nothing in the grid-tight period. Both conditions have held since spring 2022, so the
  step-aside rule has not been triggered since; every 2023-on figure is identical with or without it.
- It trades return for protection when the environment turns. Over the full period the worst fall
  drops from −17.1% to −12.1% and the Calmar ratio rises (1.51 to 1.76), but return is about 4.5 points
  a year lower and the Sharpe ratio slightly lower (1.43 against 1.45): the long book did well in
  2020–21, the years the rule spent in T-bills. The dashed line in the rolling-Sharpe chart shows what it
  stepped around.

- Rising real yields also benefit the short leg. Utilities trade like bonds: their monthly returns move
  against changes in the real yield (correlation −0.4 to −0.7 in every period since 2013), and they
  fell 2–3% on average in months when real yields jumped more than 0.2 points.
- The 10-year real yield was 2.44% at the end of August 2026, close to its
  post-2008 high of about 2.5%, and suppliers' real revenue growth is 12.0% and rising. Both conditions
  are well clear of triggering.
-->



---

## Return Attribution

#### Compared with Theme (GRID) or Market (SPY) 

To attribute the sources of return, we conduct some simple regression of the strategy's daily return on GRID or SPY.  Whatever is left (the regression intercept) after accounting
for exposure to the theme (or in case of SPY, the market) is alpha. A t-statistic above 2 means the result is unlikely to be by chance alone.



| 2023 – Aug 2026 | alpha per year | t-statistic |
|---|---|---|
| Long/short vs GRID | **24.5%** | **4.3** |
| Long/short vs S&P 500 | 23.5% | 3.9 |
| *Ranking alone: top half vs all qualifying names, equally weighted* | 9.3% | 3.5 |

- Strategy alpha survives the short. About 24.5% a year of return is not explained by moving with GRID
  (beta 0.40 to GRID). The short removes theme exposure; what remains comes from the sources below.
- The alpha is statistically strong:  a t-statistic of 4.3 against GRID and 3.9 against the S&P 500.
- To test whether the ranking adds value, we regress the long book's daily return on that of the same book holding every name that passes the screen in equal weight, with identical risk controls. The intercept, 9.3% a year since 2023 (t-stat 3.5), is the value added from picking the top half of the screened names.

#### Sources of the Alpha

We can decompose the strategy's daily return splits into the components below.


 Each day the strategy's return is

r = (1−w)·[h·R + (1−h)·rf] + w·S − s·B − s·borrow − costs

where R is the long arm's return, h the long exposure set by the risk controls, rf the daily cash
rate, w the macro sleeve weight (15%) and S its return, s the short size (0.25) and B the short basket's
return. Writing h as the average exposure h̄ (0.611) plus the day's deviation from it, h̄ + (h − h̄),
splits the long part in three:

<div markdown="1" style="font-size:0.8em">

```
r = (1−w)·[h·R + rf − h·rf] + w·S − s·B − s·borrow − costs

  = (1−w)·[(h̄ + (h − h̄))·R + rf − (h̄ + (h − h̄))·rf] + …

  = (1−w)·[h̄·R + (h − h̄)·R + rf − h̄·rf − (h − h̄)·rf] + …

  = (1−w)·[(h̄·R + rf − h̄·rf) + ((h − h̄)·R − (h − h̄)·rf)] + …

  = (1−w)·[h̄·R + (1 − h̄)·rf + (h − h̄)·(R − rf)] + …

r = (1−w)·h̄·R + (1−w)·(1−h̄)·rf + (1−w)·(h−h̄)·(R−rf) + w·S − s·B − s·borrow − costs
```

</div>

The seven terms, in order, are long selection, cash, timing, sleeve, short, borrow and costs:

| piece | daily formula |
|---|---|
| Long stock selection | (1−w) · h̄ · R |
| Cash interest | (1−w) · (1−h̄) · rf |
| Risk-control timing | (1−w) · (h − h̄) · (R − rf) |
| Gold/Treasury sleeve | w · S |
| Short | − s · B |
| Short borrow | − s · 1% ÷ 252 |
| Trading costs | − 10bp × amount traded |


Each piece's daily series is regressed on GRID's daily return; its intercept × 252 is the
alpha in the table. The pieces sum to the strategy's return every day, so their alphas sum to the
total.


| 2023 – Aug 2026 | alpha per year, ex-GRID | share  |
|---|---|---|
| *Long stock selection: the held names, at the book's average size (61% exposure)* | **11.7%** | 48% |
| *Short stock selection: the shorted names doing worse than their theme exposure implies* | 4.0% | 16% |
| *Risk-control timing: volatility target and stretch cut, versus holding the average size throughout* | 6.5% | 27% |
| *Gold/Treasury sleeve (15%)* | 1.6% | 7% |
| *Cash interest on the part the risk controls keep out of stocks* | 1.3% | 5% |
| *Short borrow cost* | −0.2% | −1% |
| *Trading costs* | −0.4% | −2% |
| **Total** | **24.5%** | 100% |

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

</div>

- Share is the component's alpha divided by total alpha of 24.5%.
- About two-thirds is stock selection (16 points, long and short together). 
  - Per dollar, the long names beat their GRID exposure by 22.4%.  Multiply by 0.85 outside macro sleeve and 0.611 (average exposure) and you get to 0.52. Then 0.52 x 22.4 ~ 11.6 (11.7 in the table).
  - The shorted names returned 15.8% a year less than their GRID exposure predicted, good for the strategy as we are short them. The strategy size the short at 0.25 per dollar.  This is about 4.0% (the short selection line). The short earns positive alpha (4.0%) even though it lost a little money (about −1% a year): by removing
  theme exposure it gives up GRID-driven return, and what it contributes beyond that is positive.
- Part of the alpha is choosing which corner of the theme to own (capital-light, profitable, on-theme
  suppliers); the ranking within that corner adds about 9 points a year (t ≈ 3.5) over holding every
  qualifying name equally. 
- About a quarter is timing from the risk controls (6.5 points): the book was larger when markets
  were calm and smaller when they were volatile or stretched.
- The gold/Treasury sleeve and cash add about 3 points: their returns are unrelated to GRID.


### Style Factors

The alpha above could, in principle, be a disguised bet on well-known investment styles. To check, the
same regression was re-run with style factors added alongside GRID.

| 2023 – Aug 2026, net of | alpha per year | t-statistic |
|---|---|---|
| ex-GRID *(as above)* | 24.5% | 4.3 |
| ex-academic factors | 20.8% | 4.0 |
| ex-academic factors and GRID | 20.7% | 4.2 |
| ex-ETF style factors and GRID | **25.4%** | **4.9** |

Academic factors: (daily, Kenneth French data library): the five Fama-French factors (market,
size, value, profitability and investment) plus Carhart momentum.

ETF style factors: (each measured as the gap between two ETFs):

| style | measured as |
|---|---|
| Momentum | MTUM (iShares MSCI USA Momentum) minus SPY |
| Growth vs value | IWF (Russell 1000 Growth) minus IWD (Russell 1000 Value) |
| Quality | QUAL (iShares MSCI USA Quality) minus SPY |
| Low volatility | USMV (iShares MSCI USA Min Vol) minus SPY |
| Size | IWM (Russell 2000) minus SPY |

Loadings on each style, with GRID in the regression (t-statistic in brackets; above about 2 in size
means the loading is unlikely to be zero):

| style | academic factors and GRID | factor's avg. annual return, 2023–26 | avg. annual contribution (loading × return) | ETF style factors and GRID |
|---|---|---|---|---|
| GRID | 0.24 (8.2) | -- | -- | 0.28 (11.7) |
| Market | 0.18 (4.8) | +17.0% | +3.1% | -- |
| Momentum | **0.23 (10.2)** | +5.4% | **+1.2%** | 0.07 (2.5) |
| Size (small minus large) | 0.27 (7.4) | −4.0% | −1.1% | 0.17 (5.2) |
| Profitability / quality | 0.25 (6.4) | −3.2% | −0.8% | 0.48 (5.1) |
| Value (academic) / growth minus value (ETF) | −0.08 (−2.5) | −0.8% | +0.1% | 0.15 (3.7) |
| Investment (conservative minus aggressive) | −0.13 (−3.0) | −6.8% | +0.9% | -- |
| Low volatility | -- | -- | -- | −0.13 (−2.1) |
| **All five styles (excluding market)** | | | **+0.3%** | |
| R² | 0.47 | | | 0.44 |

- Controlling for academic style factors leaves 20.7% of alpha, and for ETF styles leaves 25.4%.  The tilts largely cancel:  momentum added about 1.2 points a year while the small-cap and profitability tilts cost about 1.9. 
- The book does lean on some styles:
  - *Momentum:* clearly positive (a loading of 0.23 on the academic momentum factor or 0.07 on the ETF measure, meaning the book tends to move about a quarter as much as the momentum factor). This is the one that matters: if momentum
    stocks sell off sharply, the strategy will likely fall with them (see *Momentum Sell-Offs* below).  
  - *Small size and profitability/quality:* moderate tilts. That fits the rules, which pick
    profitable, capital-light mid-caps.
  - *Growth vs value:** a slight tilt toward growth on both measures (0.15 on ETF, -0.08 on academic value factor).  These are small enough so that we are reasonably comfortable that the strategy isn't a disguised growth bet.
  - *Low volatility:* it leans away from low-volatility stocks.
- *Most of the return is not style:* the factors explain less than half of the daily variation
  (R² 0.44–0.47).

#### Momentum Sell-Offs

The sharpest falls in the momentum factor since 2019, and how each book did over the same days.
Best return in each row in bold.

| period | momentum factor | strategy | strategy, always invested | GRID | PAVE | S&P 500 |
|---|---|---|---|---|---|---|
| 27 Aug – 16 Sep 2019 | −13.1% | 0.6% | 3.8% | 5.0% | **11.5%** | 4.6% |
| 9 Oct – 8 Nov 2019 | −8.5% | 6.7% *(T-bills from 1 Nov)* | 11.8% | 9.3% | **12.9%** | 6.1% |
| 16 Apr – 8 Jun 2020 | −33.6% | *in T-bills* | 2.3% | 29.7% | **32.6%** | 15.8% |
| 14 Oct – 24 Nov 2020 | −21.6% | *in T-bills* | 6.1% | 11.4% | **15.5%** | 4.4% |
| 12 Feb – 29 Mar 2021 | −18.8% | *in T-bills* | 6.2% | −2.8% | **9.1%** | 1.1% |
| 14 Jun – 10 Aug 2022 | −16.0% | 0.6% | 0.6% | **15.3%** | 14.7% | 12.8% |
| 27 Dec 2022 – 2 Feb 2023 | −22.6% | 7.0% | 7.0% | 10.1% | **11.0%** | 9.3% |
| 12 May – 16 Jun 2023 | −8.0% | **9.6%** | **9.6%** | 8.1% | 9.0% | 7.2% |
| 10 Jul – 5 Aug 2024 | −6.0% | −3.1% | −3.1% | −8.9% | **−1.0%** | −7.8% |
| 27 Jan 2025 (DeepSeek) | −3.3% | −5.9% | −5.9% | −5.6% | −3.8% | **−1.4%** |
| 7 Feb – 10 Mar 2025 | −10.5% | **−4.8%** | **−4.8%** | −5.5% | −9.1% | −6.7% |
| 22 Jun – 31 Aug 2026 *(ongoing)* | −17.6% | −1.8% | −1.8% | −9.3% | −7.5% | **3.0%** |

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*Episodes are peak-to-trough falls of at least 7% in the Fama-French momentum factor, plus July 2024
(when the MTUM ETF fell 13%) and the DeepSeek day (a shock to the AI-power theme). The strategy held
T-bills from November 2019 to April 2022 under the step-aside rule. "Always invested" is the same strategy with the step-aside rule switched off; it differs only where the strategy held T-bills.*

</div>

- **Momentum sell-offs come in two kinds.** In most of them (2019, 2020–21, mid-2022, early 2023)
  momentum fell because beaten-down stocks rebounded in a rising market. The strategy stayed positive
  every time it was invested but lagged GRID, PAVE and the S&P 500, because its low beta and the short
  side cap its upside. Had the step-aside rule kept it invested through 2020–21, it would
  still have made money in all three of that period's crashes (2.3% to 6.2%), though far behind GRID and PAVE.
- **In sell-offs in a falling market it generally lost less than GRID:** −3.1% against −8.9% in July
  2024, −4.8% against −5.5% in February–March 2025 and −1.8% against −9.3% in June–August 2026, but
  slightly more on the DeepSeek day (−5.9% against −5.6%).
- **Its largest loss in any episode is −5.9%**, on the DeepSeek day.

---

## Short Book Detail

### Short Basket Composition

![The short basket against the long book and PAVE](img/tophalf_powerlong/inv_short_leg.png)

The shorted stocks rose **4.0% a year** over the full period, against 20.8% for PAVE, with a worst fall
of −52%. They move with the theme, so the short gains when the theme falls and cushions the long book.
They have also risen far less than the theme, so in a rising market the short costs much less than the
long book earns.

#### Rationale for the Short

As is desirable of short legs, the short doesn't need its member names to fall always but only to do worse
than the long book. The proposed strategy accomplishes this. The long book has beaten the shorted stocks
in both rising and falling months (by 5.1% a month when GRID fell and 1.1% when it rose).

| 2023 – Aug 2026, average month | shorted stocks | long book | long book − 0.25 × shorted stocks | GRID |
|---|---|---|---|---|
| *GRID down (17 months)* | −6.0% | −0.9% | **0.6%** | −3.8% |
| *GRID up (27 months)* | 4.2% | 5.3% | 4.2% | **5.5%** |
| *All months (44)* | 0.2% | 2.9% | **2.8%** | 1.9% |

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*Long book: the long side with its risk controls, before the short. Best in each row in bold, between
the last two columns.*

</div>

The short costs about 1 point of return a year (39.5% to 38.6% since 2023), and in exchange cuts the
worst drawdown from −16.0% to −8.5% and beta to the S&P 500 from 0.72 to 0.47, raising the Sharpe
ratio from 1.93 to 2.28. In months when GRID fell, the strategy was slightly positive.

It is weakest when beaten-down stocks rebound sharply. The shorted stocks are largely recent laggards,
so when laggards rally, as in mid-2020 and mid-2022, they rose 22–29% in a few weeks and the short
cost money. In falling-market sell-offs it helped: in February–March 2025 the shorted stocks fell 12.3%
and the short cut the book's loss from −7.6% to −4.8%.

**The basket today (18 names, formed 30 June 2026):**

| group | names | why they rank low |
|---|---|---|
| Electric utilities (8) | BKH · CNP · EIX · ES · OGE · PCG · POR · TXNM | Own the wires: capital spending a quarter to half of sales |
| Power producers (2) | NRG · XIFR | Own generation |
| Nuclear developers (2) | NNE · SMR | Spending ahead of revenue |
| Industrials and contractors (6) | CARR · IESC · JCI · LII · RRX · XYL | Slowing sales, below-trend results, weak price momentum |

About 60% of the basket, on average since 2019, is utilities and power producers — the companies
paying for the buildout. The rest are electrification industrials that the buildout is passing by. Gas
pipelines and gas distributors are not shorted: their "transmission and distribution" is gas, not the
power grid.

### PAVE Alternative

**PAVE is the simple alternative.** A $0.25 short of the PAVE ETF (US infrastructure and electrical
companies — closest in make-up to the long book) gives a similar result under the same rules:
Sharpe 1.26 over the full period and 2.06 since 2023 (against 1.43 and 2.28), with a worst fall of −11.7% against −12.1%. It is one liquid line
($13.5bn fund, about $82m traded a day, listed options), normally cheap to borrow — no single-name
borrow to arrange. We prefer the single-name short because its edge is in stock selection rather than
index tracking; an investor who values simplicity gives up some return per unit of risk with PAVE.

---

## Current Holdings

![Names held long and short](img/tophalf_powerlong/inv_breadth.png)

**Long (15 names, equally weighted; rebalanced 11 August 2026):**

> AZZ · BWXT · CW · DY · ETN · KEYS · MIR · MKSI · MYRG · NVT · PLPC · POWL · VRT · WCC · WTS

**Short (18 names, equally weighted, $0.25 per $1 of the fund; formed 30 June 2026):** listed
[above](#short-basket-composition).

The thinnest long name, AZZ Inc., trades about $26m a day; the median long holding trades
$155m. The long book held as few as one name at a 2019 rebalance and has held 11 on average since 2023.

---

## Running Costs

| source of trading | 2019–2026, two-way | 2019–2026, one-way | 2019–2026, cost a year at 10bp | 2023–2026, two-way | 2023–2026, one-way | 2023–2026, cost a year at 10bp |
|---|---|---|---|---|---|---|
| Changing long names at each quarterly rebalance (with the hold buffer) | 141% | 70% | 0.14% | 98% | 49% | 0.10% |
| Resizing the long side (volatility target and stretch cut, 20-point band) | 172% | 86% | 0.17% | 140% | 70% | 0.14% |
| Resetting the gold/Treasury sleeve monthly | 15% | 8% | 0.02% | 15% | 8% | 0.02% |
| Re-forming the short basket quarterly ($0.25 per $1) | 128% | 64% | 0.13% | 124% | 62% | 0.12% |
| Resetting the short to $0.25 monthly | 15% | 7% | 0.01% | 15% | 7% | 0.01% |
| **Total a year while invested** | **~470%** | **~235%** | **0.47%** | **~390%** | **~195%** | **0.39%** |
| **Total a year over the whole window**, counting time in T-bills and the step-aside switches (invested 67% of 2019–2026 with six switches; 100% of 2023–2026 with none) | **~415%** | **~205%** | **0.41%** | **~390%** | **~195%** | **0.39%** |
| *For comparison, one-way as reported by each fund: GRID 26%, PAVE about 5–13%, SPY 3%* | | | | | | |

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*Cost a year = two-way turnover × 10bp. The 2023–2026 total of 0.39% is the same cost, on the same trades, that appears as the −0.4% trading-cost line in [Sources of the Alpha](#sources-of-the-alpha). Changing long names falls from
141% to 98% a year after 2023 because the long book has held more names since then (11 on average,
against as few as one in 2019), so each quarterly rebalance replaces a smaller share of it.*

*Turnover measures how much is traded in a year relative to the portfolio's capital. **Two-way**
counts every purchase and every sale: each day, the absolute change in every position's weight is
added up, and the daily sums are totalled and annualised. **One-way** — the convention in fund
prospectuses, including the ETF figures above — counts roughly one side, the lesser of purchases or
sales, so it is about half the two-way figure. On the one-way convention the strategy trades like an
active fundamental long/short fund or a systematic factor fund (typically 100–600% a year), not like
an index ETF.*

</div>

<table style="border-collapse:collapse; font-size:0.75em;">
<thead><tr><th style="text-align:left; padding:4px 10px; border-bottom:2px solid #999;"></th><th colspan="3" style="text-align:center; padding:4px 10px;">Full period, Jan 2019 – Aug 2026</th><th colspan="3" style="text-align:center; padding:4px 10px;background:rgba(42,120,214,0.08);">2023 – Aug 2026</th></tr>
<tr><th style="text-align:left; padding:4px 10px; border-bottom:2px solid #999;">trading cost per trade</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">return a year</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">Sharpe</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">max drawdown</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;background:rgba(42,120,214,0.08);">return a year</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;background:rgba(42,120,214,0.08);">Sharpe</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;background:rgba(42,120,214,0.08);">max drawdown</th></tr></thead>
<tbody>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">None <i>(before trading costs)</i></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">21.8%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.47</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−11.4%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">39.2%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">2.31</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">−8.4%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">2bp <i>(resizing done in index futures)</i></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">21.7%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.46</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−11.6%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">39.1%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">2.30</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">−8.4%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;"><b>10bp (the figures on this page)</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>21.3%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>1.43</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>−12.1%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);"><b>38.6%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);"><b>2.28</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);"><b>−8.5%</b></td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">20bp</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">20.8%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.40</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−12.7%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">38.1%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">2.25</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">−8.5%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">10bp <b>+ 1% management fee</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">20.5%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.37</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−12.3%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">37.2%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">2.20</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">−8.6%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">10bp <b>+ 2% management fee</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">19.7%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.31</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−12.5%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">35.9%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">2.12</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">−8.7%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">20bp <b>+ 2% management fee</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">19.2%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.28</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−13.1%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">35.3%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">2.09</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">−8.8%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">10bp <b>+  <i>2% management fee + 20% of gains above the previous high, charged yearly)</i></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">15.9%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">0.97</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−13.8%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">28.7%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">1.58</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);">−9.9%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;"><i>GRID ETF (net of its fee)</i></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><i>23.3%</i></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><i>0.82</i></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><i>−40.6%</i></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);"><i>23.0%</i></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);"><i>0.93</i></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;background:rgba(42,120,214,0.08);"><i>−20.8%</i></td></tr>
</tbody>
</table>

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*Shaded columns: 2023 – Aug 2026. Borrow on the short (1% a year) is charged in every row.*

</div>

Even at twice our
assumed trading cost, or with a 2% management fee, the strategy keeps a Sharpe well above every
benchmark's.

Even after a full "2 and 20" fee, the strategy beats GRID since 2023 on all three measures in the table:

- *Return:* 28.7% a year against 23.0%.
- *Sharpe:* 1.58 against 0.93.
- *Max drawdown:* −9.9% against −20.8%.

---

## Strategy Suitability, Risks and Mitigants

| risk | mitigant |
|---|---|
| **Thin long book.** The long book holds 15 names today and 11 on average since 2023, and fell as low as one name at a 2019 rebalance. The October 2026 keyword clean-up removed about two names on average. A few names drive a large share of the return. | The hold buffer (a held name stays while in the top 65%) keeps the book from shrinking at each rebalance. *Planned:* test a wider universe to raise the name count. |
| **The tight-grid environment may end.** The strategy is built for a grid short of capacity. Once new capacity catches up with demand, it will outlive its usefulness. | The strategy has three partial/gradual mechanisms to adjust for this eventuality: i) when book is likely near peak, partially switch to T-bills, ii) target 20% vol so that if there is a sharp change to an oversupplied grid regime the strategy starts to scale back, iii) long book requires growing revenue.  If that stops then we will shift endogenously to tbills.  There is also the option to have a step-aside so that when real revenue growth drops below zero or real yield is below zero, we fully switch into t-bills|

---

## Next Steps 

- Improve the strategy design
  - Perturbations to expand breadth:
    - Examine holding top 60% etc intead of top half for long book to expand breadth.
    - Examine non-US grid equipment makers (Siemens, Schnieder, ABB, etc)
    - Lower liquidity floor

  - Strenghten the short arm

  - Use interconnection queue data as a glut detector for a faster wind-down.

  - DOE's SPARK initiative launched (yesterday) $5.25billion initiative to fund 31 grid improvement projects. Will this impact strategy (Utilities in the US spend over $100bn on transmission and distribution so this is a small figure, but worth keeping an eye on with respect to grid-stress easing).

  - Test idea on ohter capacity crunches. (AI and data center specific buildouts, Gas turbines )

  - Test more accurate factor model cleaning of strategy

- Pipeline the strategy

- Hand over

- Paper trade

