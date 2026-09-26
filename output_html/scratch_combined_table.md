# SCRATCH: combined long + short construction table

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
