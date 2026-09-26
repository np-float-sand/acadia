# Screened Electrification Equipment

> **In brief.** Electricity demand is growing faster than the grid can carry it. Regulated utilities
> pay for the new wires, transformers and substations; the equipment makers and contractors that supply
> them are paid as the work is delivered. The strategy ranks every electrification company on one score of
> fourteen measures (capital intensity, revenue growth and acceleration, earnings and revenue surprises,
> price trends and sensitivity to power markets), holds the best-scoring half of those suppliers and shorts the weakest fifth of electrification companies, today mostly the
> utilities and pipelines doing the paying.
>
> The grid stress that took hold in 2023 looks durable. Utilities have committed to multi-year capital
> plans, equipment lead times run to years, and real yields remain elevated. The suppliers' own
> filings show no slowdown: their median revenue growth is 10.5% a year after inflation and still
> accelerating, the fastest since early 2023. Over that period the long/short has delivered a Sharpe
> ratio of 2.05 after trading costs, with a beta of 0.54 to the S&P 500. It steps aside into T-bills
> whenever real yields turn negative or suppliers' sales stop growing after inflation — the one
> environment in which it has failed.

*Backtest window: January 2019 – August 2026 · All strategy figures are after trading costs (10bp on
every trade) and borrow · Management fee not assumed*

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
- **Hatched grey:** time in T-bills (1 November 2019 – 29 April 2022, plus three brief spells in 2019).
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

| | **Long/short (the strategy)** | GRID ETF | PAVE ETF | S&P 500 | VOLT ETF* |
|---|---|---|---|---|---|
| *Sharpe ratio* | **2.05** | 0.93 | 0.91 | 1.16 | 0.78 |
| *Calmar ratio* | **3.62** | 1.11 | 0.87 | 1.20 | 0.97 |
| *Max drawdown* | **−10.2%** | −20.8% | −26.2% | −18.8% | −23.4% |
| *Volatility* | **13.8%** | 20.3% | 20.6% | 15.0% | 25.5% |
| *Beta to the S&P 500* | **0.54** | 1.10 | 1.09 | 1.00 | 1.00 |
| *Return a year* | **36.8%** | 23.0% | 22.8% | 22.6% | 22.8% |
| *Max time under water* | 4.4 months | 7.6 months | 7.9 months | **4.2 months** | 5.3 months |
| *Return / Sharpe after trading costs* | 36.8% / 2.05 | | | | |
| *Return / Sharpe before trading costs* | 37.4% / 2.09 | | | | |

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*\*VOLT: December 2024 – August 2026 only (21 months). Over that same window the strategy's Sharpe
was 1.58, its max drawdown −10.2% and its return 27.2% a year.*

</div>

### Earlier Period (January 2019 – December 2022)

| | **Long/short (the strategy)** | GRID ETF | PAVE ETF | S&P 500 |
|---|---|---|---|---|
| *Sharpe ratio* | 0.44 | **0.77** | 0.60 | 0.48 |
| *Calmar ratio* | **0.71** | 0.58 | 0.43 | 0.39 |
| *Max drawdown* | **−11.1%** | −40.6% | −44.1% | −33.7% |
| *Volatility* | **9.2%** | 27.4% | 29.7% | 22.5% |
| *Beta to the S&P 500* | **0.12** | 1.06 | 1.13 | 1.00 |
| *Return a year* | 7.9% | **23.5%** | 19.1% | 13.1% |
| *Max time under water* | 35.0 months | 13.3+ months | 11.8+ months | 11.9+ months |
| *Return / Sharpe after trading costs* | 7.9% / 0.44 | | | |
| *Return / Sharpe before trading costs* | 8.3% / 0.48 | | | |

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*Sharpe uses a flat **4.0% risk-free rate**. Returns are total returns, dividends reinvested. The
strategy's figures are after 10bp on every trade — long and short name changes, resizing, sleeve and
short resets, and each step-aside switch — and 1% a year borrow on the short ([Costs](#running-costs));
the "before trading costs" row removes the 10bp. ETF returns are net of their own fees. Max time under water is the longest stretch below a previous peak
within the window; "+" means the drawdown was still unrecovered at the end of 2022.*

</div>

**This is a grid-tightness strategy.** It does not work well when the grid has ample spare capacity:
in 2020–21 the suppliers' median revenue growth fell to −8% a year after inflation, and the strategy
trailed the theme ETFs. There is no exit signal for the buildout itself, but the strategy de-risks on
its own. It steps aside into T-bills unless both the 10-year real yield and the suppliers' real revenue
growth are above zero (the step-aside rule — it held T-bills from November 2019 to April 2022). Within the book, position size falls automatically as
volatility rises or prices run far above trend (to a third of full size in April 2025), and the
ranking rotates toward suppliers whose sales are still growing.

**Since 2023 the strategy leads every benchmark on every measure but one** — the S&P 500 recovered
from the 2025 tariff selloff slightly faster (4.2 months under water against 4.4). **Before 2023 it was a capital-preservation
strategy:** a quarter of the drawdown of the theme ETFs, the highest Calmar ratio, and almost no market
exposure — but a lower Sharpe and return than GRID and PAVE, and a long spell under water (July
2019 – June 2022), most of it spent in T-bills while a clean-energy rally lifted exactly the stocks the
rules exclude.

<p><img src="img/tophalf_powerlong/inv_growth.png" alt="Growth of $1 with the regime strip" style="width:min(72rem,94vw);max-width:none;position:relative;left:50%;transform:translateX(-50%)"></p>

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

- **VOLT's line** starts on the strategy's line at VOLT's launch, so the two can be compared from that
  date.
- **Hatched grey:** time in T-bills under the step-aside rule. **Dashed blue:** the strategy's path had the step-aside rule
  kept it invested, starting from its level on the day it moved to T-bills. It fell through the COVID
  crash, then recovered and ended the spell about 25% above where the strategy stood in T-bills: the
  step-aside rule gave up return in 2020–21 in exchange for avoiding the swings.
- **Amber and red dotted lines (right axes):** the buildout proxy and the 10-year real yield, as
  above; their zero lines are aligned, so where both sit above the shared zero the regime is green.
- **Regime strip:** as above.
- Over the whole period GRID, carrying the full theme at 1.1× the market's beta, compounded faster than
  the strategy (4.95× against 4.23×), and PAVE finished level with it (4.25×); since 2023 the strategy
  compounded fastest.

</div>

---

## Portfolio Construction

The gap in capital intensity between the utilities and their suppliers is a fact in the accounts
rather than a judgment call, which is what lets the whole portfolio be built by rule. The long book and
the short side go through the same stages (screen, rank, weight) and are ranked on the same fourteen-signal
score: the long book holds the top half of its screened suppliers, the short the bottom fifth of the wider
universe. The long book adds risk control, and one rule decides when to step aside. The exact thresholds, timing and data behind every rule are in the
[rule book](screened_elec_strategy_log_vpowerlong.md).

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
  <td colspan="2" style="padding:8px; vertical-align:top; background:rgba(127,127,127,0.06);">The 10-K annual report used <i>data center, substation, switchgear, transmission and distribution, grid modernization</i> or <i>electric utilities</i> in the previous four years</td>
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
    <li><b>From filings (6):</b> capital intensity, revenue acceleration, revenue growth, revenue and earnings surprises, growth in capital spending</li>
    <li><b>From prices (2):</b> 12-month momentum, 1-month reversal</li>
    <li><b>From power markets (6):</b> two signals for each measure below: the stock's 36-month sensitivity to its shocks (who benefits structurally) and that sensitivity × the latest 3-month shock (who is benefiting right now)<ul style="margin:0; padding-left:1.1em;">
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
  <td style="padding:8px; vertical-align:top;">Short the bottom fifth, re-formed each quarter (16 names on average, never fewer than 12, 24 today)</td>
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
  <td style="padding:8px; vertical-align:top;">15% in GLD and IEF, reset monthly</td>
  <td style="padding:8px; vertical-align:top;">--</td>
</tr>

</tbody>
</table>

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*Shaded cells span both books: the rule is the same for each.*

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
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">Selected names, no risk control</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.59</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>57.9%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−30.9%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ volatility target</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.72</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">47.7%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−21.6%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ stretch de-lever</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.84</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">42.2%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−21.0%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ gold/Treasury sleeve (the long book)</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>1.91</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">38.1%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>−17.3%</b></td></tr>
</tbody>
</table>

**Full period, Jan 2019 – Aug 2026**

<table style="border-collapse:collapse; font-size:0.75em;">
<thead><tr><th style="text-align:left; padding:4px 10px; border-bottom:2px solid #999;">cumulative controls</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">Sharpe</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">return a year</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">worst drawdown</th></tr></thead>
<tbody>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">Selected names, no risk control</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.07</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>38.1%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−48.9%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ volatility target</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.27</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">33.7%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−24.8%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ stretch de-lever</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.27</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">30.4%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−24.8%</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ gold/Treasury sleeve (the long book)</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>1.31</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">27.3%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>−20.8%</b></td></tr>
</tbody>
</table>

### Short Book and Step-Aside Contribution

Each row adds one piece on top of the rows above it. All figures are after trading costs and borrow,
which is why the first row differs slightly from the long-book figures above.

**2023 – Aug 2026**

<table style="border-collapse:collapse; font-size:0.75em;">
<thead><tr><th style="text-align:left; padding:4px 10px; border-bottom:2px solid #999;">cumulative</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">Sharpe</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">return a year</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">worst drawdown</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">beta to S&P 500</th></tr></thead>
<tbody>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">Long book, always invested</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.87</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>39.5%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−17.2%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">0.76</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ short side</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>2.05</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">36.8%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>−10.2%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>0.54</b></td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ stepping aside (the strategy)</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>2.05</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">36.8%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>−10.2%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>0.54</b></td></tr>
</tbody>
</table>

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*The strategy stayed invested throughout 2023 – Aug 2026, so stepping aside changes nothing in this window.*

</div>

**Full period, Jan 2019 – Aug 2026**

<table style="border-collapse:collapse; font-size:0.75em;">
<thead><tr><th style="text-align:left; padding:4px 10px; border-bottom:2px solid #999;">cumulative</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">Sharpe</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">return a year</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">worst drawdown</th><th style="text-align:right; padding:4px 10px; border-bottom:2px solid #999;">beta to S&P 500</th></tr></thead>
<tbody>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">Long book, always invested</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.26</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>27.6%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−23.2%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">0.64</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ short side</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>1.38</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">25.1%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">−13.9%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">0.40</td></tr>
<tr><td style="text-align:left; padding:4px 10px; border-top:1px solid #ddd;">+ stepping aside (the strategy)</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">1.34</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;">20.8%</td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>−11.1%</b></td><td style="text-align:right; padding:4px 10px; border-top:1px solid #ddd;"><b>0.24</b></td></tr>
</tbody>
</table>

### Step-Aside Rule

**We hold the strategy only while the regime is green: the 10-year real yield is above zero and
the suppliers' revenue is still growing after inflation.** The first is the 10-year inflation-protected
Treasury (TIPS) yield; the second is the median trailing-twelve-month revenue growth, as filed and less
CPI inflation, of the names that pass the long-book screen. When either turns negative, the whole portfolio —
long, short and sleeve — moves into 3-month T-bills the next day, and returns when both are positive
again. See [Real Yields and the Buildout](#real-yields-and-the-buildout).


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
| **Long/short** | 28.9% | 0.4% *(T-bills)* | 0.1% *(T-bills)* | 4.5% | 45.1% | 38.2% | 40.6% | 11.0% |
| *Long/short had the step-aside rule kept it invested* | *40.3%* | *7.8%* | *22.3%* | *−4.2%* | *45.1%* | *38.2%* | *40.6%* | *11.0%* |
| GRID ETF | 42.2% | 48.8% | 27.7% | −13.9% | 21.6% | 15.2% | 29.7% | 17.1% |
| PAVE ETF | 32.6% | 19.7% | 36.4% | −7.2% | 31.0% | 17.9% | 19.4% | 14.7% |
| S&P 500 | 31.1% | 18.3% | 28.7% | −18.2% | 26.2% | 24.9% | 17.7% | 13.1% |
| Long/short Sharpe | 1.55 | *in T-bills* | *in T-bills* | 0.10 | 2.50 | 2.04 | 2.24 | 1.02 |
| Long/short worst drawdown | −11.1% | 0.0% | 0.0% | −7.9% | −8.6% | −5.5% | −10.2% | −7.4% |

The strategy beat GRID in every year from 2022 to 2025 and trails it so far in 2026, when the theme
itself has rallied hard. It held T-bills from November 2019 to April 2022, so it missed GRID's 49%
and 28% in 2020 and 2021 — a rally in the clean-energy and subsidy stocks the rules exclude — but also
the COVID crash, and was up 4.5% in 2022, when the S&P lost 18%.

Over its one episode the step-aside rule cost return — it gave up 11 points in late 2019, 7 in 2020 and
22 in 2021, and saved 9 in 2022 — in exchange for holding the strategy flat through the COVID crash and
the 2022 rate shock, which cut the worst drawdown from −13.9% to −11.1%.

### Drawdowns

![Drawdown](img/tophalf_powerlong/inv_drawdown.png)

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

- **Dashed blue:** the same strategy had the step-aside rule kept it invested. It differs from the solid
  line only from November 2019 to April 2022, where it fell as far as −14% while the strategy sat flat in T-bills.

</div>

---

## Real Yields and the Buildout

**The step-aside rule exists because the strategy has one clear weakness: a market that no longer rewards what
it holds.** Two conditions mark that market, and the strategy stays invested only while both hold.

- **Real yields above zero.** When the 10-year real yield is negative, money is effectively free, and
  the market pays up for distant earnings, subsidy-dependent growth and story stocks — everything the
  rules exclude. When it is positive, capital has a cost, and suppliers that are paid now are worth
  more than companies spending heavily to be paid later.
- **Suppliers' revenue still growing after inflation.** When the median supplier's real sales are
  shrinking, the buildout is not reaching the companies the strategy owns, and the ranking has little
  to separate winners from losers.

| | Sharpe, full | Sharpe, 2019–22 | Sharpe, 2023 on | max drawdown, full | Calmar, full | return a year | time invested |
|---|---|---|---|---|---|---|---|
| **Step-aside rule: both conditions (the strategy)** | 1.34 | 0.44 | 2.05 | **−11.1%** | **1.88** | 20.8% | 67% |
| Real yield alone | 1.30 | 0.39 | 2.05 | **−11.1%** | 1.87 | 20.6% | 70% |
| Always invested | **1.38** | **0.79** | 2.05 | −13.9% | 1.80 | **25.1%** | 100% |

- **It costs nothing in the grid-tight period.** Both conditions have held since spring 2022, so the
  step-aside rule has not been triggered since; every 2023-on figure is identical with or without it.
- **It trades return for protection when the environment turns.** Over the full period the worst fall
  drops from −13.9% to −11.1% and the Calmar ratio edges up (1.81 to 1.88), but return is about 4 points
  a year lower and the Sharpe ratio slightly lower (1.34 against 1.38): the long book did well in
  2020–21, the years the rule spent in T-bills. The dashed line in the rolling-Sharpe chart shows what it
  stepped around.
- **Requiring both conditions performs about the same as real yields alone** (Sharpe 1.34 against 1.31)
  and steps aside a few months
  earlier — in November 2019, as supplier sales began to shrink, rather than in March 2020. We use both
  because each describes a different half of the thesis: the cost of capital and the buildout itself.
- **Rising real yields also help the short.** Utilities trade like bonds: their monthly returns move
  against changes in the real yield (correlation −0.4 to −0.7 in every period since 2013), and they
  fell 2–3% on average in months when real yields jumped more than 0.2 points.
- **Where it stands now.** The 10-year real yield was **2.44% at the end of August 2026**, close to its
  post-2008 high of about 2.5%, and suppliers' real revenue growth is 10.5% and rising. Both conditions
  are well clear of triggering.

**The caveat.** The rule has been tested on essentially one episode — late 2019 to spring 2022 — and it
was designed after seeing that episode. We adopt it because the mechanism is clear and the weakness is
known, not because the backtest proves it; it will be judged on the live record. Its case rests on
limiting drawdowns, not on raising the Sharpe ratio: staying invested throughout would have earned more,
with a deeper worst fall. Moving to T-bills also lengthens recovery: the strategy's July 2019 peak was not
regained until June 2022.

---

## Return Attribution

A regression of each book's daily return on GRID's (or the S&P's): whatever is left after accounting
for exposure to the theme is alpha. A t-statistic above 2 means the result is unlikely to be luck.

| 2023 – Aug 2026 | alpha per year | t-statistic |
|---|---|---|
| Long/short vs GRID | **21.7%** | **3.9** |
| Long/short vs S&P 500 | 20.8% | 3.4 |
| *Ranking alone: top half vs all qualifying names, equally weighted* | 7.1% | 2.7 |

- **The alpha survives the short.** About 21.7% a year of return is not explained by moving with GRID
  (beta 0.47 to GRID). The short removes theme exposure; what remains comes from the sources below.
- **The alpha is statistically strong**, with a t-statistic of 3.9 against GRID and 3.4 against the S&P 500.

#### Sources of the Alpha

The strategy's daily return splits into the pieces below; each piece's alpha against GRID adds up
exactly to the total.

| 2023 – Aug 2026 | alpha per year, ex-GRID | share |
|---|---|---|
| *Long stock selection: the held names, at the book's average size (67% exposure)* | **11.8%** | 54% |
| *Short stock selection: the shorted names doing worse than their theme exposure implies* | 2.0% | 9% |
| *Risk-control timing: volatility target and stretch cut, versus holding the average size throughout* | 5.9% | 27% |
| *Gold/Treasury sleeve (15%)* | 1.6% | 7% |
| *Cash interest on the part the risk controls keep out of stocks* | 1.1% | 5% |
| *Short borrow cost* | −0.2% | −1% |
| *Trading costs* | −0.5% | −2% |
| **Total** | **21.7%** | 100% |

- **About two-thirds is stock selection** (14 points, long and short together). Per dollar, the long
  names beat their GRID exposure by 20.5% a year and the shorted names fell short of theirs by 7.4%.
  Part of this is choosing which corner of the theme to own (capital-light, profitable, on-theme
  suppliers); the ranking within that corner adds about 7 points a year (t ≈ 2.7) over holding every
  qualifying name equally. Part is style: controlling for momentum, size, quality and the other style
  factors leaves 18.6% of the 21.7% (see *Style Factors* below).
- **About a quarter is timing from the risk controls** (6 points): the book was larger when markets
  were calm and smaller when they were volatile or stretched.
- **The gold/Treasury sleeve and cash add about 3 points**: their returns are unrelated to GRID.
- **The short earns positive alpha (2.0%) even though it lost money** (about −2% a year): by removing
  theme exposure it gives up GRID-driven return, and what it contributes beyond that is positive.

### Style Factors

The alpha above could in principle be a disguised bet on well-known investment styles. To check, the
same regression was re-run with style factors added alongside GRID.

| 2023 – Aug 2026, net of | alpha per year | t-statistic |
|---|---|---|
| ex-GRID *(as above)* | 21.7% | 3.9 |
| ex-academic factors | 18.7% | 3.6 |
| ex-academic factors and GRID | 18.6% | 3.8 |
| ex-ETF style factors and GRID | **22.8%** | **4.6** |

**Academic factors** (daily, Kenneth French data library): the five Fama-French factors (market,
size, value, profitability and investment) plus Carhart momentum.

**ETF style factors** (each measured as the gap between two ETFs):

| style | measured as |
|---|---|
| Momentum | MTUM (iShares MSCI USA Momentum) minus SPY |
| Growth vs value | IWF (Russell 1000 Growth) minus IWD (Russell 1000 Value) |
| Quality | QUAL (iShares MSCI USA Quality) minus SPY |
| Low volatility | USMV (iShares MSCI USA Min Vol) minus SPY |
| Size | IWM (Russell 2000) minus SPY |

- **What the styles explain:** about 3 points of the 21.7%. Over the full 2019–2026 period the alpha
  is 12–15% a year, with t-statistics of 3.3–4.1.
- **The book does lean on some styles:**
  - **Momentum:** clearly positive (a loading of 0.23–0.32, meaning the book tends to move about a
    quarter to a third as much as the momentum factor). This is the one that matters: if momentum
    stocks sell off sharply, the strategy will likely fall with them (see *Momentum Sell-Offs* below).
  - **Small size and profitability/quality:** moderate tilts. That fits the rules, which pick
    profitable, capital-light mid-caps.
  - **Growth vs value:** a slight tilt toward growth on the ETF measure (0.13) and none on the academic
    value factor, so it isn't a disguised growth bet.
  - **Low volatility:** it leans away from low-volatility stocks.
- **Most of the return is not style:** the factors explain only about half of the daily variation
  (R² 0.49–0.54).

#### Momentum Sell-Offs

The sharpest falls in the momentum factor since 2019, and how each book did over the same days.
Best of the four books in bold.

| period | momentum factor | strategy | strategy, always invested | GRID | PAVE | S&P 500 |
|---|---|---|---|---|---|---|
| 27 Aug – 16 Sep 2019 | −13.1% | 0.2% | 3.3% | 5.0% | **11.5%** | 4.6% |
| 9 Oct – 8 Nov 2019 | −8.5% | 6.8% *(T-bills from 1 Nov)* | 11.8% | 9.3% | **12.9%** | 6.1% |
| 16 Apr – 8 Jun 2020 | −33.6% | *in T-bills* | 0.8% | 29.7% | **32.6%** | 15.8% |
| 14 Oct – 24 Nov 2020 | −21.6% | *in T-bills* | 6.1% | 11.4% | **15.5%** | 4.4% |
| 12 Feb – 29 Mar 2021 | −18.8% | *in T-bills* | 5.6% | −2.8% | **9.1%** | 1.1% |
| 14 Jun – 10 Aug 2022 | −16.0% | 1.9% | 1.9% | **15.3%** | 14.7% | 12.8% |
| 27 Dec 2022 – 2 Feb 2023 | −22.6% | 7.6% | 7.6% | 10.1% | **11.0%** | 9.3% |
| 12 May – 16 Jun 2023 | −8.0% | **9.1%** | **9.1%** | 8.1% | 9.0% | 7.2% |
| 10 Jul – 5 Aug 2024 | −6.0% | −1.4% | −1.4% | −8.9% | **−1.0%** | −7.8% |
| 27 Jan 2025 (DeepSeek) | −3.3% | −5.6% | −5.6% | −5.6% | −3.8% | **−1.4%** |
| 7 Feb – 10 Mar 2025 | −10.5% | −6.2% | −6.2% | **−5.5%** | −9.1% | −6.7% |
| 22 Jun – 31 Aug 2026 *(ongoing)* | −17.6% | −6.5% | −6.5% | −9.3% | −7.5% | **3.0%** |

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*Episodes are peak-to-trough falls of at least 7% in the Fama-French momentum factor, plus July 2024
(when the MTUM ETF fell 13%) and the DeepSeek day (a shock to the AI-power theme). The strategy held
T-bills from November 2019 to April 2022 under the step-aside rule. "Always invested" is the same strategy with the step-aside rule switched off; it differs only where the strategy held T-bills.*

</div>

- **Momentum sell-offs come in two kinds.** In most of them (2019, 2020–21, mid-2022, early 2023)
  momentum fell because beaten-down stocks rebounded in a rising market. The strategy stayed positive
  every time it was invested but lagged GRID, PAVE and the S&P 500, because its low beta and the short
  side cap its upside. Had the step-aside rule kept it invested through 2020–21, it would
  still have made money in all three of that period's crashes (0.8% to 6.1%), though far behind GRID and PAVE.
- **In sell-offs in a falling market it generally lost less than GRID:** −1.4% against −8.9% in July
  2024 and −6.5% against −9.3% in June–August 2026, level on the DeepSeek day (−5.6%), but slightly more
  in February–March 2025 (−6.2% against −5.5%).
- **Its largest loss in any episode is −6.5%**, in the one still under way at the end of the data.

---

## Short Book Detail

### Short Basket Composition

![The short basket against the long book and PAVE](img/tophalf_powerlong/inv_short_leg.png)

The shorted stocks rose **6.4% a year** over the full period, against 20.8% for PAVE, with a worst fall
of −47%. They move with the theme, so the short gains when the theme falls and cushions the long book.
They have also risen far less than the theme, so in a rising market the short costs much less than the
long book earns.

#### Rationale for the Short

As is desirable of short legs, the short doesn't need its member names to fall but only to do worse
than the long book. The proposed strategy accomplishes this. The long book has beaten the shorted stocks
in both rising and falling months (by 3.5% a month when GRID fell and 1.4% when it rose).

| 2023 – Aug 2026, average month | shorted stocks | long book | long book − 0.25 × shorted stocks | GRID |
|---|---|---|---|---|
| *GRID down (17 months)* | −4.5% | −1.1% | **0.1%** | −3.8% |
| *GRID up (27 months)* | 4.0% | 5.4% | 4.4% | **5.5%** |
| *All months (44)* | 0.7% | 2.9% | **2.7%** | 1.9% |

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*Long book: the long side with its risk controls, before the short. Best in each row in bold, between
the last two columns.*

</div>

The short costs about 3 points of return a year (39.5% to 36.8% since 2023), and in exchange cuts the
worst drawdown from −17.2% to −10.2% and beta to the S&P 500 from 0.76 to 0.54, raising the Sharpe
ratio from 1.87 to 2.05. In months when GRID fell, the strategy was close to flat.

It is weakest when beaten-down stocks rebound sharply. The shorted stocks are largely recent laggards,
so when laggards rally, as in mid-2020 and mid-2022, they rose 25–36% in a few weeks and the short
cost money. In falling-market sell-offs it helped: in February–March 2025 the shorted stocks fell 11.6%
and the short cut the book's loss from −8.9% to −6.2%.

**The basket today (24 names, formed 30 June 2026):**

| group | names | why they rank low |
|---|---|---|
| Electric utilities (6) | BKH · CNP · EIX · ES · PCG · TXNM | Own the wires: capital spending a quarter to half of sales |
| Gas utilities (5) | ATO · NJR · NWN · SR · SWX | Own the pipe networks |
| Pipelines and LNG (5) | EPD · KNTK · LNG · OKE · TRGP | Own pipelines and terminals |
| Power producer (1) | XIFR | Owns generation |
| Nuclear developers (2) | NNE · SMR | Spending ahead of revenue |
| Industrials and contractors (5) | CARR · ENR · IESC · RRX · XYL | Slowing sales, estimate misses, weak momentum |

About half the basket, on average since 2019, is utilities, power producers and pipelines — the
companies paying for the buildout. The rest are electrification industrials that the buildout is
passing by.

### PAVE Alternative

**PAVE is the simple alternative.** A $0.25 short of the PAVE ETF (US infrastructure and electrical
companies — closest in make-up to the long book) gives almost the same result under the same rules:
Sharpe 1.24 over the full period and 2.00 since 2023, with a worst fall of −11.7% against −11.1%. It is one liquid line
($13.5bn fund, about $82m traded a day, listed options), normally cheap to borrow — no single-name
borrow to arrange. We prefer the single-name short because its edge is in stock selection rather than
index tracking; an investor who values simplicity gives up very little with PAVE.

---

## Current Holdings

![Names held long and short](img/tophalf_powerlong/inv_breadth.png)

**Long (17 names, equally weighted; rebalanced 11 August 2026):**

> AZZ · BWXT · CW · DY · ETN · FIX · KEYS · MIR · MKSI · MOG-A · MYRG · NVT · POWL · VRT · WCC ·
> WTS · WWD

**Short (24 names, equally weighted, $0.25 per $1 of the fund; formed 30 June 2026):** listed
[above](#short-basket-composition).

The thinnest long name, AZZ Inc., trades about $24m a day; the median long holding trades
$154m. The long book held as few as one name at a 2019 rebalance and has held 14 on average since 2023.

---

## Running Costs

| source of trading | turnover a year while invested, two-way | one-way |
|---|---|---|
| Changing long names at each quarterly rebalance (with the hold buffer) | 145% | 72% |
| Resizing the long side (volatility target and stretch cut, 20-point band) | 207% | 103% |
| Resetting the gold/Treasury sleeve monthly | 15% | 8% |
| Re-forming the short basket quarterly ($0.25 per $1) | 126% | 63% |
| Resetting the short to $0.25 monthly | 12% | 6% |
| **Total, while invested** | **~500%** | **~250%** |
| **Total, averaged over 2019–2026** (67% of the time invested under the step-aside rule, including its six switches in and out) | **~435%** | **~220%** |
| *For comparison, one-way as reported by each fund: GRID 26%, PAVE about 5–13%, SPY 3%* | | |

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*Turnover measures how much is traded in a year relative to the portfolio's capital. **Two-way**
counts every purchase and every sale: each day, the absolute change in every position's weight is
added up, and the daily sums are totalled and annualised. **One-way** — the convention in fund
prospectuses, including the ETF figures above — counts roughly one side, the lesser of purchases or
sales, so it is about half the two-way figure. On the one-way convention the strategy trades like an
active fundamental long/short fund or a systematic factor fund (typically 100–600% a year), not like
an index ETF.*

</div>

| trading cost per trade | return a year | Sharpe | max drawdown |
|---|---|---|---|
| None *(before trading costs)* | 21.3% | 1.38 | −10.4% |
| 2bp *(resizing done in index futures)* | 21.2% | 1.37 | −10.5% |
| **10bp (the figures on this page)** | **20.8%** | **1.34** | **−11.1%** |
| 20bp | 20.3% | 1.30 | −11.7% |
| 10bp **+ 1% management fee** | 20.0% | 1.28 | −11.3% |
| 10bp **+ 2% management fee** | 19.2% | 1.23 | −11.5% |
| **2023 on, 10bp** | **36.8%** | **2.05** | −10.2% |
| *GRID ETF, 2023 on (net of its fee)* | *23.0%* | *0.93* | *−20.8%* |

<div markdown="1" style="font-size:0.7em;color:var(--ink-soft);line-height:1.45;margin-top:-0.4em">

*Full period unless stated. Borrow on the short (1% a year) is charged in every row.*

</div>

Even at twice our
assumed trading cost, or with a 2% management fee, the strategy keeps a Sharpe well above every
benchmark's.
