# Backlog pricing-power language — length/size-control robustness check — results (2026-09-21)

**Verdict: FAIL. Gate not cleared. This closes the pricing-power-language thread entirely.**
Pre-registration: `docs/backlog-pricing-power-length-control-preregistration.md` (written and
locked before any residual was computed). Follow-up to `docs/backlog-pricing-power-results.md`
(RESEARCH-LOG item 12, primary metric FAILED/NOT ADOPTED).

## TL;DR

- Raw keyword count was the one non-primary diagnostic in the original test that technically
  cleared the full pre-registered gate (pooled `t=-2.73`, both sub-windows past `|t|>=1.5`). The
  results doc suspected this tracked filing length / company size rather than a real
  pricing-power effect, but never tested that suspicion directly. This does.
- **After controlling for filing word count and log(point-in-time market cap), the residual
  fails the gate.** 3-month pooled `t=-2.01` still barely clears the pooled `|t|>=2.0` bar, but
  **both** sub-windows drop below the required `|t|>=1.5` (2019-2022: `t=-1.45`; 2023-2026:
  `t=-1.38`). 1-month never clears the pooled bar at all (`t=-0.45`).
- A secondary, non-gating check regressing raw_count on word count *alone* (no market cap) fails
  even more decisively: pooled 3m `t=-1.94` doesn't clear the pooled bar by itself.
- **This confirms the size/complexity-confound diagnosis already written into the original
  results doc.** Most of raw_count's apparent predictive power was a proxy for how long/large
  the filer is, not genuine pricing-power-language content. What little survives residualization
  (pooled 3m marginally past 2.0) does not survive the sub-window split -- the same
  "wins pooled, dies in the regime split" failure mode this research log has hit repeatedly
  (category-demand, STEO, and the original density metric itself).
- **No further variant of this data or mechanism is worth chasing.** Close the thread.

## 1. Scope

No filing was re-fetched or re-scored. Reused unchanged: `research/backlog_pricing_power/
keywords.py` (locked list), `_cache/filing_scores.parquet` (986 filings, same as the original
test, including the already-present `n_words` column), `build_panel.py`'s point-in-time
convention, and `ic_test.py`'s `rank_ic_series` / `summarize` (imported, not reimplemented).
Reproduced the original raw-count result exactly with this pipeline before residualizing, as a
correctness check on the point-in-time panel construction: 3m pooled `t=-2.725`, 2019-2022
`t=-1.927`, 2023-2026 `t=-1.916` -- matches `docs/backlog-pricing-power-results.md` §6 to three
decimal places.

## 2. Controls used

1. **Filing word count (`n_words`)** -- already in `filing_scores.parquet`, the exact denominator
   of the original (failed) `density_per_1k_words` metric. No new fetch.
2. **log(point-in-time market cap)** -- new. Shares outstanding via SEC EDGAR XBRL
   `companyfacts` `dei:EntityCommonStockSharesOutstanding` (the cover-page figure, joined to each
   scored filing **by accession number** -- the exact same filing, not a nearby one), multiplied
   by that ticker's **raw, unadjusted** close price on the nearest trading day at-or-before
   `filingDate` (yfinance, `auto_adjust=False`, deliberately not the project's standard
   adjusted-close cache -- adjusted close is rescaled to current-share-count terms and would
   mismatch contemporaneous shares outstanding for the 5 of 27 tickers with a split since 2016:
   APG, IESC, JCI, POWL, TT). Coverage: shares outstanding + price both resolved for 966/986
   filings (98.0%); PWR (71.8%) and VRT (81.8%) are the two names with gaps, both from early
   filings predating consistent cover-page XBRL tagging -- every other ticker is 96-100%.

`research/backlog_pricing_power/fetch_controls.py` builds this; writes
`_cache/filing_controls.parquet` (966 rows with a resolved `log_market_cap`, joined by
`accessionNumber`) without touching `filing_scores.parquet`.

## 3. Method

Per month-end in the existing point-in-time panel (`known_date = filingDate + 1 day`,
forward-filled), for the cross-section of tickers with a non-null `total_count`, `n_words`, and
`log_market_cap` that month (`n >= 8` required, same `MIN_CROSS_SECTION` as the original test):
winsorize all three at (2.5th, 97.5th) percentile and z-score all three cross-sectionally, using
`grid_resilience.factor.neutralize.winsorize` / `cross_section_zscore` unmodified. OLS
(`numpy.linalg.lstsq`): `z(total_count) = a + b1*z(n_words) + b2*z(log_market_cap) + resid`. The
residual, per ticker per month, feeds into `ic_test.rank_ic_series` / `summarize` unchanged --
same horizons (1m, 3m), same three windows (pooled, 2019-2022, 2023-2026).

`research/backlog_pricing_power/residualize.py` builds the residual panel (99/99 months in the
build range clear the `n>=8` cross-section requirement for both the two-control and one-control
versions). `research/backlog_pricing_power/ic_test_residual.py` runs the IC test on it, importing
`rank_ic_series`/`summarize`/`WINDOWS`/`HORIZONS`/`get_monthly_prices` from `ic_test.py` directly.

## 4. Results

**Bold** = clears `|t|>=2.0` (pooled) or `|t|>=1.5` (sub-window). Gate requires ALL of: pooled
bold on >=1 horizon, AND both sub-windows bold with the same sign.

### Primary check (gated) -- residual of raw_count ~ n_words + log_market_cap

| Horizon | Window | n months | mean IC | t-stat |
|---|---|---|---|---|
| 1m | Pooled | 90 | -0.009 | -0.45 |
| 1m | 2019-2022 | 48 | +0.010 | +0.32 |
| 1m | 2023-2026 | 42 | -0.030 | -1.11 |
| 3m | Pooled | 89 | -0.038 | **-2.01** |
| 3m | 2019-2022 | 48 | -0.037 | -1.45 |
| 3m | 2023-2026 | 41 | -0.040 | -1.38 |

**Gate: FAIL.** 1m never clears the pooled bar. 3m clears the pooled bar (barely -- `-2.01` vs.
the `-2.0` threshold) but **both** sub-windows fall short of the required `|t|>=1.5` (`-1.45` and
`-1.38`). The gate requires both, not one; this is exactly the "wins pooled, dies in the regime
split" pattern this log has flagged repeatedly (category-demand, STEO, the original density
metric).

### Secondary, non-gating check -- residual of raw_count ~ n_words only

| Horizon | Window | n months | mean IC | t-stat |
|---|---|---|---|---|
| 1m | Pooled | 90 | -0.009 | -0.43 |
| 1m | 2019-2022 | 48 | +0.016 | +0.54 |
| 1m | 2023-2026 | 42 | -0.037 | -1.24 |
| 3m | Pooled | 89 | -0.042 | -1.94 |
| 3m | 2019-2022 | 48 | -0.037 | -1.31 |
| 3m | 2023-2026 | 41 | -0.048 | -1.42 |

Fails more decisively than the two-control version: 3m pooled (`-1.94`) doesn't even clear the
pooled bar on its own. Confirms word count alone is already most of the confound; market cap adds
a small further reduction in the pooled 3m t-stat but doesn't change the outcome.

### For reference -- the original (pre-residual) raw_count result being tested here

| Horizon | Window | n months | mean IC | t-stat |
|---|---|---|---|---|
| 3m | Pooled | 89 | -0.057 | -2.73 |
| 3m | 2019-2022 | 48 | -0.057 | -1.93 |
| 3m | 2023-2026 | 41 | -0.057 | -1.92 |

(Reproduced exactly with this pipeline before residualizing -- see §1.)

## 5. What the drop means

Comparing pooled and sub-window t-stats before/after residualizing (3-month horizon, the only
horizon that was ever non-trivial):

| Cut | Original raw_count `t` | Residual `t` (two-control) | Change |
|---|---|---|---|
| Pooled | -2.73 | -2.01 | shrinks, barely still clears 2.0 |
| 2019-2022 | -1.93 | -1.45 | drops below the 1.5 sub-window bar |
| 2023-2026 | -1.92 | -1.38 | drops below the 1.5 sub-window bar |

Both sub-window t-stats moved from comfortably clearing the gate to comfortably missing it once
word count and market cap are partialled out. The pooled stat survives only marginally, and the
one-control (word-count-only) version doesn't even survive pooled. This is the pattern the
original results doc predicted before this test ran: raw_count's cross-sectional information is
overwhelmingly explained by how long and how large the filer is, not by anything specific to
pricing-power language content. There is no genuine, unexplained residual signal left to promote.

## 6. Disposition

**FAIL. Confirms the size/complexity-confound diagnosis. Close the pricing-power-language thread
entirely** -- per the pre-registration's stated gate consequence (§4): this was the last
non-primary result in this thread that had cleared the original gate at all, and it does not
survive controlling for the two most obvious confounds named at the time. No further variant of
this data or mechanism (this keyword list, this universe, this EDGAR filing source, any
re-tuning of the phrase list) is worth chasing after this -- consistent with the original
results doc's own closing recommendation, and now backed by a direct test rather than a
suspicion. `research/backlog_pricing_power/` (including the new `fetch_controls.py`,
`residualize.py`, `ic_test_residual.py`, and `_cache/filing_controls.parquet`,
`_cache/raw_count_residual_panel*.parquet`) is kept as a throwaway reference build, same status
as the rest of that directory -- nothing wired into `electrification_strategy` or
`grid_equipment_basket`.
