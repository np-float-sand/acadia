# Transmission Rate-Base Compounders — Design Spec (2026-08-31)

## 1. Summary

A low-turnover, sector-neutral equity factor over US regulated electric utilities.
**Thesis:** utilities whose FERC-jurisdictional **transmission rate base is
compounding** — because they must build lines for new large loads (data centers,
electrification, interregional transfer) — are the under-owned, "boring"
expression of the power buildout. Rank the regulated universe by transmission-
rate-base growth; go long the fast compounders, short the flat / merchant-drag
names.

This is proposal **B** from `docs/handoff_2026-08-31-strategy-proposals.md`, the
sole survivor of the 2026-08-31 triage (`docs/triage_2026-08-31-proposals-bda.md`):
proposal D failed its pre-registered gate and proposal A failed the breadth floor.

The build is **gated**. It ships as a strategy module only if it clears a
pre-registered pass/fail bar (§11). If it fails, the outcome is a documented
negative result and no shipped factor — same discipline as `backlog_factor/`.

## 2. Why this might be additive (and the risk)

Transmission for most large IOUs is recovered under **FERC formula rates** tied
directly to booked transmission plant (accounts 350–359) net of depreciation,
plus CWIP, less ADIT. So booked net transmission plant tracks the FERC
transmission rate base closely — much more closely than booked plant tracks the
state-jurisdictional distribution/generation rate base. The dispersion in
transmission-rate-base growth across the ~40 regulated names is real (DOE's
National Transmission Needs Study names AEP, Eversource, NextEra, ITC/Fortis and
Berkshire as leaders; many peers are flat) and the leaders span a range of
dividend yields, so the signal is *plausibly* not just a yield/low-vol repackage.

**The risk, and the reason for the additivity gate (§10):** a regulated-utility
"rate-base growth" tilt can collapse into a bond-proxy / low-vol / dividend-yield
trade. The factor must show residual alpha after those controls or it does not
ship.

## 3. Universe

**Full US-listed regulated electric + multi-utility set that files FERC Form 1**,
target ~40–45 parents. Seed list (finalised in `data/utility_map.py`):

AEP, AEE, AES, CNP, CMS, D, DTE, DUK, ED, EIX, ES, ETR, EVRG, EXC, FE, IDA, LNT,
NEE, NWE, OGE, PCG, PEG, PNW, POR, PPL, SO, WEC, XEL,
plus smaller/adjacent: AVA, BKH, HE, MGEE, OTTR, TXNM (ex-PNM).
Names no longer independently listed (ALE/Allete — taken private 2025; PNM →
TXNM) are resolved at universe-freeze time: use the current listed successor
ticker, or drop the name if there is none. Gas-only names (NJR, OGS, SR, UGI,
ATO) are excluded.

Selection rules:
- Must file FERC Form 1 as (or through subsidiaries that are) an electric utility.
- Exclude pure gas LDCs / propane / midstream (NJR, OGS, SR, UGI, ATO — gas-only).
- Keep electric-heavy multi-utilities (DTE, CMS, WEC, XEL, D, PEG, SRE) — segment
  mix is handled by neutralising on unregulated- and non-electric-revenue share
  (§7), not by exclusion.
- Merchant-heavy names (VST, NRG, CEG, TLN) are **not** in the universe — they
  have little or no regulated transmission rate base; they may appear only as
  short-side "merchant-drag" context, not scored.
- Point-in-time universe: a name enters when it has ≥4 years of Form 1 filing
  history under its current corporate structure (so the first 3-yr growth figure
  is computable — see §6); spin-offs (e.g. post-2022 restructurings) start when
  the successor filer has 4 years.

Final universe list is frozen in `data/utility_map.py` and **never revised from
results** (base-spec discipline, matching `grid_equipment_basket`).

## 4. Data layer — `data/ferc_form1.py`

Source: **PUDL nightly parquet** (Catalyst Cooperative, CC-BY-4.0), fetched over
HTTPS from `https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/<table>.parquet`.
No auth, no cost. Confirmed reachable this session (largest table 298,611 rows,
1.8 s).

Tables pulled and cached locally as parquet under
`transmission_rate_base/data/cache/`:

| Table | Use |
|---|---|
| `core_ferc1__yearly_plant_in_service_sched204` | gross transmission plant in service (FERC accts 350–359), by `utility_id_ferc1` × `report_year` |
| `core_ferc1__yearly_depreciation_by_function_sched219` | accumulated depreciation, `plant_function == 'transmission'` → net transmission plant |
| `core_ferc1__yearly_utility_plant_summary_sched200` | total utility plant in service + accumulated depreciation → net total plant (denominator) |
| `core_pudl__assn_ferc1_pudl_utilities` | `utility_id_ferc1` → `utility_name_ferc1` → `utility_id_pudl` crosswalk |

Extraction rules:
- **Gross transmission plant** for a filer-year: prefer the labelled subtotal row
  (`ferc_account_label == 'transmission_plant'`, `row_type_xbrl == 'total'`); fall
  back to summing the leaf `*_transmission_plant` accounts. A unit test asserts
  the two agree within 0.5 % for a sample of filer-years; log and prefer the
  subtotal when they diverge.
- Exclude `*_regional_transmission_and_market_operation_plant` (FERC accts ~380s)
  — that is RTO/market software & equipment, not transmission wires rate base.
- Filter `utility_type == 'electric'`, `plant_status == 'in_service'`.
- **Net transmission plant** = gross transmission plant − accumulated
  depreciation where `plant_function == 'transmission'` (from sched219). If
  sched219 has no transmission row for a filer-year, fall back to gross ×
  (net total plant / gross total plant) from sched200 (pro-rata depreciation);
  flag these rows.
- Values are nominal USD; no inflation adjustment (growth rates and shares are
  ratio measures).

Caching: one parquet per source table, written on first fetch. `refresh=True`
re-pulls. Annual data — a yearly refresh is enough. The fetch module must degrade
gracefully offline: if the HTTPS pull fails and a cache file exists, use the
cache and warn; if neither, raise a clear error.

## 5. Utility map — `data/utility_map.py`

Static hand-built dict:

```python
PARENT_FILERS: dict[str, list[str]] = {
    "AEP": ["Appalachian Power Co", "Ohio Power Co", "Indiana Michigan Power Co",
            "AEP Texas Inc", "Public Service Co of Oklahoma",
            "Southwestern Electric Power Co", "Kentucky Power Co", ...],
    ...
}
```

Keys = tickers; values = `utility_name_ferc1` strings (resolved to
`utility_id_ferc1` at load time via the crosswalk; both DBF-era and XBRL-era ids
for the same `utility_id_pudl` are included).

Build procedure: for each parent, list its regulated electric subsidiaries from
the latest 10-K ("Item 1 — Regulated Operations" / subsidiary exhibit), match each
to `utility_name_ferc1`, verify the FERC id has non-trivial transmission plant.
~40 parents × 1–8 filers.

Tests: every mapped name resolves to exactly one `utility_id_pudl`; no
`utility_id_pudl` is mapped to two parents; every parent has ≥1 filer with ≥3
years of transmission-plant history.

Known hard cases documented inline: NextEra (NEET/FPL split — include FPL + NextEra
Energy Transmission filers, exclude NEER), Berkshire's utilities (not separately
listed — BRK is excluded from the tradable universe), post-2022 spin successors,
Fortis/ITC (ITC files Form 1 but ITC's parent Fortis is TSX/NYSE `FTS` — include
as `FTS` only if we accept a Canadian parent; default **exclude**, note in
limitations).

## 6. Signal — `signal.py`

Per parent, per `report_year` Y:

- `gross_tx[Y]`  = Σ over filers of gross transmission plant in service
- `net_tx[Y]`    = Σ over filers of net transmission plant
- `net_total[Y]` = Σ over filers of net total utility plant
- `tx_share[Y]`  = `net_tx[Y] / net_total[Y]`

**Primary signal** (rank-average of two components, no z-scoring — house rule):

1. `g3_net_tx`   = (`net_tx[Y] / net_tx[Y-3]`) ** (1/3) − 1   (3-yr net-transmission-plant CAGR)
2. `d3_tx_share` = `tx_share[Y] − tx_share[Y-3]`               (3-yr change in transmission share, ppts)

`raw_signal = rank(g3_net_tx) .add( rank(d3_tx_share) ) / 2`, `rank(pct=True)`
over the point-in-time universe. Guards: NaN unless ≥4 consecutive annual
observations `Y-3..Y` with positive `net_tx` and `net_total`; a `|log(net_tx[y]/
net_tx[y-1])| > log(2)` step NaNs that parent-year (suspected restatement /
structural break).

Rationale for the two components: the CAGR captures absolute build pace; the
share change captures *mix shift toward transmission* and is robust to a parent
that is simply growing its whole rate base. Rank-average (not sum of z-scores)
matches `grid_equipment_basket` and avoids one component's fat tail dominating.

Sensitivity variants computed and reported but **not** the headline: gross
instead of net; 5-yr instead of 3-yr window; each component alone.

## 7. Cross-sectional neutralisation — `signal.py`

`neutralize(raw_signal, log_rate_base, nonreg_rev_share) -> residual`:

1. Winsorise `raw_signal` at ±2.5 σ (`grid_resilience.factor.neutralize.winsorize`
   pattern).
2. Z-score cross-sectionally.
3. OLS regress on `[1, log_rate_base, nonreg_rev_share]`; take residuals.
4. Re-z-score residuals.

- `log_rate_base` = log of `net_total[Y]` summed across filers (size control — a
  big utility mechanically adds more transmission $).
- `nonreg_rev_share` = 1 − (regulated electric **operating** revenue / total
  **operating** revenue), from the parent 10-K segment note (hand-collected once
  into a small static table `data/segment_mix.csv`, ~40 rows, columns
  `ticker,fy,reg_elec_op_rev,total_op_rev`; updated annually). Controls for "this
  is really a regulated-purity trade".

Follows `backlog_factor.signal.neutralize_cross_section` in spirit (winsor →
z-score → regress out covariates → residual), without the industry dummies (all
utilities).

## 8. Portfolio — `portfolio.py`

- **Rebalance:** once per year, first trading day of **May** (Form 1 for year Y-1
  is filed by ~April 18 of year Y; May gives a clean margin). Signal at a May-Y
  rebalance uses only `report_year ≤ Y-1`.
- **Ranking:** neutralised residual signal, point-in-time universe (≥3 yr
  history).
- **Primary book — dollar-neutral quintile L/S:**
  - Long  = top quintile (~8 names), equal-weight, +1.0 gross.
  - Short = bottom quintile (~8 names), equal-weight, −1.0 gross.
  - Ex-ante beta neutralisation: scale the short leg by
    `beta_long / beta_short`, where each leg beta is the trailing-252-**trading-
    day** OLS beta of the equal-weight leg return on the EW utility-universe
    return, computed from the **daily** closes in the price cache (available even
    though P&L is accrued monthly), capped to [0.5, 2.0]. Net beta target ≈ 0.
- **Secondary book — long-only tilt** (reported alongside, for the can't-short
  mandate): start from equal weight over the point-in-time universe, multiply
  quintile-5 weights by 1.25 and quintile-1 by 0.75, renormalise to 1.0 gross,
  long only.
- Turnover expected low (annual, ~2–4 names per leg change per year).

## 9. Backtest — `backtest.py`

- **Windows:** primary **2011-01 → 2025-12** (post-FERC Order 1000, which opened
  competitive interregional transmission). Secondary **2003–2010** as a
  pre-thesis out-of-sample sign check.
- **Returns:** monthly total returns from the `grid_resilience` yfinance
  monthly-wide price cache, re-pointed via `data/prices.py`. ~40 utility tickers
  + `XLU`, `SPY`, EW-utility benchmark. ~20 are already cached; the rest fetch
  when yfinance is reachable. **The pipeline must run to a verdict from cache
  alone** (offline-capable) so implementation is not blocked by the current
  yfinance rate-limiting; a `--offline` flag skips all network fetches.
- **Metrics** (reuse `grid_resilience.portfolio.backtest`):
  ann return, ann vol, Sharpe, Sortino, max DD, hit rate, per-year returns.
  Rank-IC of the neutralised signal vs **forward 12-month relative return**
  (relative to EW utilities), computed at each May rebalance
  (`compute_ic`-style, Spearman).
- **Benchmarks:** EW utility universe, XLU. Report factor vs both.
- Outputs to `output_transmission_rate_base/` (gitignored): `signal_panel.csv`,
  `weights.csv`, `pnl.csv`, `annual_returns.csv`, `ic.csv`, `additivity.csv`,
  `verdict.txt`, and a performance PNG.

## 10. Additivity test — `additivity.py`

Regress the **primary L/S monthly factor return** on control factors built from
public data over 2011–2025:

| Control | Construction (all cross-sectional L/S within the utility universe unless noted) |
|---|---|
| `util_beta` | return on the EW utility universe (level, not L/S) |
| `xlu` | XLU excess return (level) |
| `dividend_yield` | top-minus-bottom quintile by trailing dividend yield |
| `low_vol` | bottom-minus-top quintile by trailing 12-m realised vol |
| `size` | small-minus-big by log market cap |
| `momentum` | winner-minus-loser by 12–1 month return |
| `capex_intensity` | high-minus-low by (capex / net PP&E) |
| `value` | high-minus-low by book/price |

OLS with Newey-West (12-lag) standard errors. Report full coefficient table,
R², and the annualised intercept ("residual alpha") with its t-stat.

Dividend/'yield'/size/value/capex inputs: yfinance fundamentals where available;
for pre-2015 gaps, a small hand-filled `data/style_inputs.csv` (annual) — scope
allows this to be partial, with the additivity window trimmed to the covered
years if necessary (documented).

## 11. Pre-registered pass/fail gate

Written now, evaluated **once**, on the primary 2011–2025 window. B ships as a
strategy module only if **all four** hold:

1. **Rank-IC** (neutralised signal → forward 12-m relative return):
   mean IC ≥ 0.03 **and** t-stat ≥ 2.0.
2. **Q5−Q1 primary book:** annualised Sharpe ≥ 0.40, quintile mean returns
   monotone in signal rank (≤ 1 adjacent inversion), and positive in ≥ 60 % of
   calendar years.
3. **Additivity:** residual alpha ≥ 3 %/yr annualised **and** t-stat ≥ 2.0
   **and** regression R² < 0.6.
4. **Pre-thesis window (2003–2010):** factor return not significantly negative
   (one-sided t > −2.0). Sign check, not a hard gate — a fail here is a caveat,
   not a stop, but is reported prominently.

If 1, 2 or 3 fails → **negative result**: write `docs/transmission-rate-base-results.md`
documenting what failed and why, keep the reusable data/signal code, ship **no**
strategy module, add no README strategy line. If all pass → proceed to a
portfolio/robustness pass and a results writeup recommending a configuration.

No parameter search over the signal definition, window, quintile count, or
neutralisation set before the gate is evaluated. The §6 sensitivity variants are
computed for the writeup only and cannot be promoted to headline without a fresh
pre-registered gate.

## 12. Module layout

```
transmission_rate_base/
  __init__.py
  config.py                 # universe seed, windows, quintile size, thresholds
  data/
    __init__.py
    ferc_form1.py            # PUDL parquet fetch + local cache + extraction
    utility_map.py           # PARENT_FILERS static dict + resolution helpers
    segment_mix.csv          # hand: ticker,fy,reg_elec_rev,total_rev
    style_inputs.csv         # hand: annual yield/mktcap/bookval/capex/ppe gap-fill
    prices.py                # re-point of grid_resilience monthly-wide price cache
  signal.py                 # net_tx / tx_share series, primary signal, neutralize
  portfolio.py              # quintile L/S (dollar+beta neutral), long-only tilt
  backtest.py               # P&L, rank-IC, benchmarks (wraps grid_resilience.portfolio.backtest)
  additivity.py             # control-factor build + Newey-West regression
  report.py                 # assemble verdict, write output_transmission_rate_base/*
  __main__.py               # CLI
tests/transmission_rate_base/
  test_ferc_form1.py  test_utility_map.py  test_signal.py
  test_portfolio.py   test_additivity.py   test_gate.py  test_pipeline.py
```

CLI:
```
python -m transmission_rate_base \
    --start 2011-01-01 --end 2025-12-31 \
    [--offline] [--refresh-ferc] [--pre-thesis] [--sensitivity]
```
Prints the §11 verdict.

## 13. Testing plan

Unit (synthetic FERC frames / synthetic prices, deterministic):
- `ferc_form1`: subtotal-vs-leaf-sum agreement; RTO-plant exclusion; net = gross −
  dep; sched200 pro-rata fallback; offline cache fallback.
- `utility_map`: bijective parent↔pudl-id; every parent has ≥3 yr history;
  unknown name raises.
- `signal`: 3-yr CAGR and share-change math on a hand frame; history/positivity
  guards; structural-break NaN; rank-average; neutralise removes a planted
  size/nonreg-share effect.
- `portfolio`: quintile assignment; dollar neutrality; net beta ≈ 0 on synthetic
  betas; long-only tilt sums to 1 and stays non-negative; annual rebalance dates.
- `additivity`: recovers a planted alpha and planted factor loadings; R² sane;
  Newey-West runs.
- `gate`: each of the 4 conditions passes/fails independently; all-pass ⇒ pass;
  any-of-1..3 fail ⇒ fail; NaN inputs ⇒ fail.

Integration (`test_pipeline.py`): full run on 3 hand-built parents + synthetic
prices, `--offline`, asserts a verdict dict is produced with all expected keys.

Target ~22–26 tests. Full repo suite must stay green.

## 14. Deliverables

- This spec (committed).
- Implementation plan via `writing-plans`.
- `transmission_rate_base/` module + tests.
- `docs/transmission-rate-base-results.md` — pass or fail writeup.
- If it passes: README sibling-module line; a memory update.
- `.gitignore`: add `output_transmission_rate_base/`.

## 15. Known limitations / open questions

- **Breadth.** ~40 parents × annual signal × 15 yrs ≈ 600 parent-years, but only
  ~120–160 effectively-independent bets after autocorrelation. Below the "few
  hundred" bar the handoff sets. This is disclosed in the results doc; the gate
  t-stat thresholds (≥2.0) are the guard against over-reading a thin sample.
- **Signal is public.** Transmission capex guidance is in every 10-K; part of the
  edge may already be priced. The additivity test partly addresses this (if it's
  priced as a known style, R² will be high / alpha low).
- **FERC Form 1 lag & restatements.** Annual, ~4-month lag, occasional
  reclassification of plant between function codes. The structural-break guard
  (§6) and the subtotal-vs-leaf check (§4) are the mitigations.
- **ITC / Fortis.** The cleanest listed pure-play transmission operator files
  Form 1 but its parent trades as Canadian `FTS`. Default: excluded. Open
  question for review: include `FTS` and accept one non-US parent?
- **Style-input history.** yfinance fundamentals are shallow pre-2015; the
  additivity window may be trimmed. Acceptable for a gate; flagged if it bites.
- **Segment-mix table is hand-collected** (~40 rows/yr). One-time cost; annual
  maintenance.

## 16. Out of scope

- 10-Q / rate-case forward-rate-base parsing (a possible later refinement to add
  intra-year signal refresh; not in this build).
- Options overlays, factor timing, position sizing beyond equal-weight legs.
- Non-US utilities (except the ITC/FTS open question).
- Client risk-model factor data (public-data controls only, per the SOW).
