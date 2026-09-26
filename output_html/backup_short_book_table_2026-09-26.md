<!-- Backup of the Short Book table from screened_elec_selection_investor_edit.md, 2026-09-26, before the power-markets bullet was expanded -->

### Short Book

<table style="border-collapse:collapse; width:100%; font-size:0.75em;">
<thead>
<tr style="border-bottom:2px solid #999;">
  <th style="width:2.6em;"></th>
  <th style="text-align:left; padding:6px 10px;">Aspect</th>
  <th style="text-align:left; padding:6px 10px;">Rules</th>
  <th style="text-align:left; padding:6px 10px;">Source</th>
</tr>
</thead>
<tbody>

<!-- SCREEN -->
<tr style="border-top:3px solid #fff;">
  <td rowspan="3" style="background:#dbe8fb; color:#1f3a63; font-weight:700; letter-spacing:0.12em; text-align:center; vertical-align:middle; padding:8px 4px;"><div style="writing-mode:vertical-rl; transform:rotate(180deg); margin:auto;">SCREEN</div></td>
  <td style="padding:8px 10px; vertical-align:top;"><b>Electrification companies</b></td>
  <td style="padding:8px 10px; vertical-align:top;"><ul style="margin:0; padding-left:1.1em;">
    <li>Companies in the SEC's electrical-equipment and power industry codes, plus nuclear fuel-cycle names found by keyword</li>
    <li>At least $10m of stock traded on a typical day (about 200 names)</li></ul></td>
  <td style="padding:8px 10px; vertical-align:top;"><ul style="margin:0; padding-left:1.1em;">
    <li>Industry codes: SEC EDGAR</li>
    <li>Keywords: SEC EDGAR full-text search</li>
    <li>Daily prices and volume (Yahoo Finance)</li></ul></td>
</tr>
<tr style="border-top:1px solid #ddd;">
  <td style="padding:8px 10px; vertical-align:top;"><b>On-theme</b></td>
  <td style="padding:8px 10px; vertical-align:top;"><ul style="margin:0; padding-left:1.1em;">
    <li>The 10-K used the same buildout wording as the long book in the previous four years</li>
    <li>This removes banks, insurers and industrials with no real grid business</li></ul></td>
  <td style="padding:8px 10px; vertical-align:top;"><ul style="margin:0; padding-left:1.1em;">
    <li>SEC EDGAR full-text search of 10-Ks</li></ul></td>
</tr>
<tr style="border-top:1px solid #ddd;">
  <td style="padding:8px 10px; vertical-align:top;"><b>Not held long</b></td>
  <td style="padding:8px 10px; vertical-align:top;"><ul style="margin:0; padding-left:1.1em;">
    <li>Any name in the long book is excluded</li></ul></td>
  <td style="padding:8px 10px; vertical-align:top;">--</td>
</tr>

<!-- RANK -->
<tr style="border-top:3px solid #fff;">
  <td style="background:#d9f0e1; color:#1d4d2e; font-weight:700; letter-spacing:0.12em; text-align:center; vertical-align:middle; padding:8px 4px;"><div style="writing-mode:vertical-rl; transform:rotate(180deg); margin:auto;">RANK</div></td>
  <td style="padding:8px 10px; vertical-align:top;"><b>Short the weakest fifth</b></td>
  <td style="padding:8px 10px; vertical-align:top;"><ul style="margin:0; padding-left:1.1em;">
    <li><b>From filings:</b> capital intensity, revenue acceleration, revenue growth, revenue and earnings surprises, growth in capital spending</li>
    <li><b>From prices:</b> 12-month momentum, 1-month reversal</li>
    <li><b>From power markets:</b> sensitivity to wholesale power prices, congestion and scarcity pricing</li>
    <li>Score all fourteen each month-end, average them with equal weight, keep the bottom fifth</li></ul></td>
  <td style="padding:8px 10px; vertical-align:top;"><ul style="margin:0; padding-left:1.1em;">
    <li>Filings: SEC EDGAR XBRL filings</li>
    <li>Prices: Yahoo Finance</li>
    <li>Power: PJM and ERCOT wholesale market prices</li></ul></td>
</tr>

<!-- WEIGHT -->
<tr style="border-top:3px solid #fff;">
  <td style="background:#fbefd0; color:#5c4410; font-weight:700; letter-spacing:0.12em; text-align:center; vertical-align:middle; padding:8px 4px;"><div style="writing-mode:vertical-rl; transform:rotate(180deg); margin:auto;">WEIGHT</div></td>
  <td style="padding:8px 10px; vertical-align:top;"><b>Equal weight, fixed size</b></td>
  <td style="padding:8px 10px; vertical-align:top;"><ul style="margin:0; padding-left:1.1em;">
    <li>Each name weighted equally; basket re-formed each quarter (16 names on average, never fewer than 12, 24 today)</li>
    <li>$0.25 short for every $1 in the long book, reset each month</li></ul></td>
  <td style="padding:8px 10px; vertical-align:top;">--</td>
</tr>

</tbody>
</table>
