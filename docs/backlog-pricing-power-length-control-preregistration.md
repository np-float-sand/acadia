# Backlog pricing-power language — length/size-control robustness check — pre-registration (2026-09-21)

**Written before any residual is computed or any IC is run on it.** Follow-up to
`docs/backlog-pricing-power-results.md` (RESEARCH-LOG item 12, FAILED/NOT ADOPTED on the
pre-registered primary metric). This test is narrower and does not reopen that verdict: the
**primary metric (density per 1,000 MD&A words) stays FAILED**, full stop. What's being tested
here is a specific loose end the results doc flagged but didn't resolve: two *non-primary*
diagnostics — raw keyword count and distinct-phrase breadth — technically cleared the full
pre-registered gate (pooled t=-2.73 and -2.47, both sub-windows past |t|>=1.5, stable
magnitude). They were not promoted because the results doc's own read was: "whatever
cross-sectional information exists here tracks filing/company size and complexity, not the
density of pricing-power language" -- i.e., a suspected confound with filing length / company
size, not a real pricing-power effect. That diagnosis was never actually tested; this does that.

**Sign note, stated up front:** the working hypothesis in the original pre-registration was "more
pricing-power language -> better forward returns." Every non-trivial result so far has been the
opposite sign. Per explicit instruction on this task, a wrong sign does not disqualify a result --
only lack of genuine predictive power does. The question this test answers is not "is the sign
right" but "does raw_count predict forward returns for a reason *other than* being a proxy for how
big/long the filing is." A passing residual would be a genuine, if unexpected and separately
mechanism-unclear, finding -- not something the original hypothesis predicted, but not disqualified
by that either.

## 1. Scope — reuse, no rescoring

No filing is re-fetched or re-scored. This reuses, unchanged:
- `research/backlog_pricing_power/keywords.py` -- the locked 12-phrase list (untouched).
- `research/backlog_pricing_power/_cache/filing_scores.parquet` -- 986 already-scored 10-K/10-Q
  filings for the same 27-name `screen` universe, 2017-01 -> 2026-08. Already contains `n_words`
  (word count of the extracted MD&A text, or whole-document word count when MD&A extraction
  fell back -- same field the original `density_per_1k_words` metric was normalized by), so the
  first, most direct control needs **no new fetch at all**, only a join already implicit in the
  cached file.
- `research/backlog_pricing_power/build_panel.py`'s point-in-time convention (`known_date =
  filingDate + 1 calendar day`, forward-filled to month-end) and `research/backlog_pricing_power/
  ic_test.py`'s exact rank-IC method (monthly cross-sectional Spearman, mean/t across the monthly
  series, same `WINDOWS`, same `MIN_CROSS_SECTION=8`, same `summarize()`) -- imported, not
  reimplemented, so the residual test is apples-to-apples with the original.

The only new fetch is the second control (market cap -- see §2.2), which is genuinely not present
anywhere in the existing cache.

## 2. Controls

### 2.1 Control 1 — filing word count (`n_words`)

The direct length confound the results doc named by name. Already computed per filing
(`score.py`'s `score_text()`, stored as `n_words` in `filing_scores.parquet`) -- this is literally
the denominator of the failed primary metric (`density_per_1k_words = total_count / n_words *
1000`). No new data needed.

### 2.2 Control 2 — log(point-in-time market cap)

Feasibility checked before writing this pre-registration (not assumed): SEC EDGAR XBRL
`companyfacts` (not `companyconcept` -- that endpoint returned an empty `units` object for 2 of 27
tickers, HUBB and JCI, for reasons that look like an EDGAR API quirk tied to those entities'
CIK/name history, even though the same tag has full data in the `companyfacts` JSON for the same
CIK; `companyfacts` is used for all 27 to avoid that inconsistency) `dei:
EntityCommonStockSharesOutstanding` tag -- the cover-page shares-outstanding figure every 10-K/10-Q
tags, indexed by the filing's own accession number. Confirmed present for all 27 tickers (10
facts for GEV, spun off April 2024, up to 71 for others -- no zero-coverage names).

Shares outstanding is joined onto each scored filing **by accession number** (the same filing,
not a nearby one) and multiplied by that ticker's **raw, unadjusted** close price on the nearest
trading day at-or-before `filingDate` (tolerance up to 5 trading days back, to cross holidays/gaps)
to get point-in-time market cap.

**Deliberately not** the project-standard adjusted-close price cache
(`grid_equipment_basket.data.prices.fetch_prices`, `auto_adjust=True`): adjusted close rescales
historical prices to current-share-count terms, which would mismatch a *contemporaneous*
point-in-time shares-outstanding figure for any ticker with a split in the window. Checked before
choosing this: 5 of 27 names (APG 3:2 Jul-2025, IESC 2:1 Aug-2026, JCI 0.955 Sep-2016, POWL 3:1
Apr-2026, TT 1.289 Mar-2020) had a split-type corporate action since 2016. Raw unadjusted close x
contemporaneous shares outstanding is internally consistent at every historical date regardless of
later splits; adjusted close x contemporaneous shares is not. A new, separate raw-price fetch
(`auto_adjust=False`) is used for this control only -- the return-basis IC test itself still uses
the existing adjusted-close price cache, unchanged.

`log_market_cap = log(shares_outstanding_at_filing * raw_close_price_at_filing)`.

## 3. Method

For each month-end `t` in the existing monthly panel (`build_panel.monthly_panel`, same
`known_date = filingDate + 1 day`, forward-filled, same `start="2018-06-01", end="2026-08-31"`
build range), build the cross-section of tickers with a non-null point-in-time `total_count`,
`n_words`, and `log_market_cap` that month. Require `n >= 8` (`MIN_CROSS_SECTION`, same as the
original test -- months with fewer are dropped, exactly as before).

Within that month's cross-section:
1. **Winsorize** `total_count`, `n_words`, `log_market_cap` at the (2.5th, 97.5th) percentile,
   using `grid_resilience.factor.neutralize.winsorize` (the project's existing helper, default
   limits, unmodified) -- not a new clipping rule invented for this test.
2. **Z-score** all three cross-sectionally with `grid_resilience.factor.neutralize.
   cross_section_zscore` (same helper).
3. **OLS**: regress z-scored `total_count` on an intercept + z-scored `n_words` + z-scored
   `log_market_cap` (`numpy.linalg.lstsq`, no new statistical dependency). The **residual** of
   that regression, per ticker, is "raw_count with the length-and-size-explainable part removed."
4. Assemble the monthly residuals into a month x ticker panel -- same shape/index as any other
   score panel `build_panel.monthly_panel` would produce, so it plugs directly into
   `ic_test.rank_ic_series` unchanged.

Then run **the exact same rank-IC test** as `ic_test.py` on this residual panel: `rank_ic_series`
+ `summarize`, imported (not reimplemented), for horizons `h in {1, 3}` months, on the same three
windows -- **pooled** (2019-01 -> 2026-06), **2019-2022** (pre-AI), **2023-2026** (AI regime) --
against the same `grid_equipment_basket.data.prices.fetch_prices` monthly adjusted-close forward
returns used throughout this log.

**Secondary, non-gating check reported alongside:** the same residual computed with `n_words` as
the sole control (no market cap), to see whether the two-control result is being carried entirely
by one control or is a joint effect. This does not change the gate or the verdict -- only the
two-control residual is gated.

## 4. Gate — identical to the original pre-registration, applied to the residual

**PASS** requires: pooled `|t| >= 2.0` on at least one horizon (1m or 3m) on the residual series,
**and** the same sign with `|t| >= 1.5` in **both** non-overlapping sub-windows. Anything short of
that is FAIL, reported with the actual numbers, no post-hoc reframing of the bar.

**Stated explicitly, before running anything:** passing this gate on the residual would be **the
first genuinely-promotable result in this entire thread** -- every prior test in
`docs/backlog-pricing-power-results.md` either failed outright (density) or passed only in a form
(raw/unnormalized) the results doc itself flagged as confounded. A pass here means raw_count
predicts forward returns for a reason other than "longer filing / bigger company," which would be
a real, if mechanistically unclear and unexpected, finding requiring its own follow-up before any
adoption -- not something to quietly wire in off the back of this check alone. **Failing this gate
confirms the size/complexity-confound diagnosis already written into the results doc** and closes
the pricing-power-language thread entirely: no further variant of this data or mechanism (this
keyword list, this universe, this filing source) would be worth chasing after this.

## 5. What's built, throwaway

New files under `research/backlog_pricing_power/` (existing files in that directory --
`cik_map.py`, `edgar_filings.py`, `keywords.py`, `score.py`, `build_panel.py`, `ic_test.py`,
`run_fetch_score.py`, and the `_cache/` contents -- are read-only inputs to this test, not
modified):
- `fetch_controls.py` -- EDGAR `companyfacts` shares-outstanding fetch (all 27 tickers, cached
  JSON) + raw unadjusted-close price fetch (yfinance, cached parquet), joined by accession number
  / nearest trading day onto `filing_scores.parquet` into a new derived file
  `_cache/filing_controls.parquet` (does not overwrite `filing_scores.parquet`).
- `residualize.py` -- the monthly winsorize -> z-score -> OLS -> residual panel construction in
  §3, reusing `grid_resilience.factor.neutralize` and `build_panel`'s point-in-time convention.
  Writes `_cache/raw_count_residual_panel.parquet`.
- `ic_test_residual.py` -- runs `ic_test.rank_ic_series` / `summarize` (imported) on the residual
  panel, same windows/horizons, prints the results table.

Results -> `docs/backlog-pricing-power-length-control-results.md` after this runs, verdict
reported plainly regardless of outcome.
