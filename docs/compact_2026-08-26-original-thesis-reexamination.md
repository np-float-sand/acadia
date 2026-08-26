# Grid Resilience Strategy — Original Thesis Re-Examination (2026-08-19 to 2026-08-26)

**Status:** Investigation complete. No configuration tested beats XLU. Recommendation: the
underlying signal is too weak in this sample, not a construction or engineering problem.

## Purpose

This session started from a direct question: examine the original pitch's intent (`pitch/GridResilienceStrategy.pdf`
— "reward utilities that are better prepared for grid stress, via physical grid data rather
than headline ESG metrics") against the strategy's actual current-data performance, and find
another approach to make it work. It ends with a systematic, closed-out audit of every
angle of the core thesis — *"companies better prepared for / that withstand shocks to energy
price or to congestion are more profitable"* — tried across construction, signal design, and
factor validation. None produced a strategy that beats XLU.

## Starting state

On corrected data (post the 2026-08-13 PJM cache-gap fix), the strategy trailed XLU outright:
Sharpe ≈ −0.21 to −0.28 vs. XLU's 0.162 (`output_dcqueue/dc_executive_summary.html`). The short
book was already suspected as the structural problem — no configuration tried before this
session had fixed it.

## Investigation threads

### 1. Peer-group (basket-vs-basket) portfolio construction — implemented, tested, did not help

**Diagnosis it was built to fix:** whole-universe long/short (top-3 long / bottom-5 short across
all 18 names) exposes both legs to sector beta — long book correlation to XLU +0.76 (and
underperforms buy-and-hold XLU on raw return); short book Sharpe −0.615, statistically no
better than a naive equal-weight short of the entire 18-name universe (−0.557).

**Built:** `build_grouped_weights()` in `grid_resilience/portfolio/construction.py`, splitting the
universe into merchant/mixed/regulated peer groups and building long+short baskets within each.
`PEER_GROUP_CONSTRUCTION` config flag (default `False`), `--peer-group` CLI flag, 12 unit tests.

**Result** (`docs/compact_2026-08-19-peer-group-construction-results.md`):

| | Whole-universe (baseline) | Peer-group |
|---|---|---|
| Sharpe | −0.276 | **−0.343** (worse) |
| Ann. Return | 1.48% | 1.69% |
| Ann. Vol | 9.13% | **6.73%** (−26%) |
| Max Drawdown | −19.39% | **−13.72%** (−29%) |
| Strategy corr. to XLU | +0.104 | +0.198 (worse) |

The mechanism worked exactly as designed at the sleeve level (regulated/mixed/merchant net
correlation to XLU dropped to +0.16 / +0.15 / +0.02, each far below that sleeve's long leg alone)
— but the whole-universe baseline's correlation to XLU was *already* low, apparently by chance,
so peer-group construction didn't beat it in aggregate. Sharpe got worse specifically because
raw excess return is negative in both configurations (annualized return < the 4% risk-free
rate), and dividing a negative excess return by a smaller volatility produces a more negative
Sharpe — the metric penalizes the very risk reduction the redesign achieved.

**Conclusion:** the binding constraint is the factor's raw excess-return generation, not
portfolio construction. Left off by default; kept available via `--peer-group` since the
vol/drawdown reduction could be useful if paired with a signal that does generate excess return.

### 2. Outage / reserve-margin availability signal — researched, designed, reviewed, spiked, killed

**Idea:** differentiate VST/NRG by which has more *available* (non-outaged) capacity to capture
a given stress event, rather than treating both as interchangeable in the stress-beta regression.

**Data research:** ERCOT's hourly resource-outage report
(`gridstatus.Ercot.get_hourly_resource_outage_capacity`) has no historical archive — rolling
window only. EIA's `operating-generator-capacity` dataset (Form 860/860M) does have 2008–2026
monthly coverage with a `status` facet (`OP`/`OS`/`SB`/`OA`).

**Bugs found and fixed along the way** in `grid_resilience/data/eia_data.py::fetch_plant_capacity()`
(useful independent of this idea's fate, covered by `tests/data/test_eia_data.py`):
- Dead API route (`electricity/operating-generator/generator` → `electricity/operating-generator-capacity`)
- Wrong frequency (`annual` → HTTP 400; dataset is monthly-only)
- No pagination (5000-row API cap; a full year for ERCOT alone is ~10,600 rows) — added
  offset-based pagination via `_EIA_PAGE_LENGTH`/retry logic

**Design spec written** (`docs/superpowers/specs/2026-08-20-outage-availability-signal-design.md`):
modulate merchant stress-beta by a 3-month trailing OP-ratio, 100-day reporting lag (measured
live: latest available period was 2026-05 as of check date 2026-08-20, ~90 days).

**Independent harsh review found blocking issues before any implementation:**
1. **Directional logic backwards for negative-beta names.** `beta_z * availability_ratio`
   dampens toward zero regardless of sign — correct for capping upside on a positive-beta long
   candidate, but wrong for a negative-beta short candidate. Winter Storm Uri's actual mechanism
   was NRG/VST losing money *because* fleets tripped offline while still owing power at fixed
   prices — low availability should *confirm* a short, not dampen it.
2. **`dual_track` architecture never touched by the fix**, verified against the actual code —
   its branch rebuilds beta straight from raw `stress_betas`, bypassing the patched variable.
3. Feature possibly inert under the CLAUDE.md-documented "winning" config, which never sets `--arch`.
4. **Entity curation checked only one month (Jan 2025)** for an 8-year backtest, despite Vistra
   acquiring Dynegy in April 2018 — essentially the backtest's start date — a real reason to
   distrust a single current-day snapshot.
5. Lag stacking: 100-day design lag + 3-month trailing average on top of an already up-to-a-year-lagged
   stress beta ≈ 6+ months of staleness modulating a signal meant to catch week-scale events.
6. No empirical validation plan — unit tests only, no backtest comparison or success criterion.

**Spike test (before fixing any of the above) disproved the premise entirely.** Live-pulled EIA
status history for VST/NRG's ERCOT entities for all of 2020–2021: **`OP` ratio = 1.0 in every
single month for both tickers, including February 2021 — the month of Winter Storm Uri**, one
of the most severe ERCOT generation-outage events on record. Sanity-checked this wasn't a fetch
bug: the dataset *can* show non-`OP` status generally (41 of 1330 ERCOT generators did in Feb
2021), just none of them are VST/NRG entities, and 3% is nowhere near Uri's actual scale.
**Root cause:** EIA-860's status codes are annual-scale (`OS` = "not expected to return to
service in next calendar year") — they track retirements/mothballing, not days-long forced
outages. A unit down for 3–5 days during Uri and back online before month-end never touches
this field, at any lag.

**Conclusion: SHELVED.** Both realistic data sources for this idea are now ruled out (see
`docs/handoff_2026-08-15-outage-reserve-margin-signal.md`, rewritten to record this).

### 3. RT/DA LMP spread signal — data-availability split, not pursued further

**ERCOT: fully practical.** `gridstatus.Ercot.get_dam_spp(year)` pulls a full calendar year of
hub-level day-ahead prices in one bulk call (129,121 rows for 2019, ~7 seconds). Real-time side
already proven in production (existing backtest already runs on ERCOT RT-ish LMP for the full
2018–2025 window).

**PJM: not practical as scoped.** For any date older than PJM's ~2-year archive window,
`gridstatus` downloads the *entire* PJM nodal network before filtering to the 4 hubs we need —
confirmed live (300,000+ rows for a single day at both 5-min and hourly resolution), and PJM's
API returned 429 rate-limit errors before even one day's fetch completed.

**Why this matters:** the thesis's "T&D hurt by unexpected RT spikes" half lives almost entirely
in PJM names (AEP, EXC, PPL, FE, D, PEG) — the side of the data that's blocked is the side with
most of the names to test the thesis on. Only ERCOT's VST/NRG/CNP would be practical today.
Not pursued further this session; no code changes made.

### 4. Universe expansion (merchant sleeve 2→4 names) — implemented, tested, no improvement

**Motivation:** a 2-name merchant sleeve (VST/NRG) was correctly flagged as "too thin to be a
real strategy" — no diversification, no real cross-sectional statistics, high concentration
risk. Client is an equity trader (won't trade power/commodities directly), so the fix had to
widen the *equity* universe, not switch instruments.

**Research:** the entire US-listed independent-power-producer universe is 9 companies
(stockanalysis.com). Only two are genuine, liquid fits: **CEG** (Constellation Energy, $98.65B,
Exelon's spun-off nuclear generation fleet, listed 2022-01-19) and **TLN** (Talen Energy,
$14.74B, PJM nuclear/gas, re-listed 2023-06-02 post-Chapter 11). Excluded: OKLO (pre-revenue SMR
developer), TAC (Canadian fleet), KEN (Israeli holding company), DGXX/HNRG (wrong business
model / microcap).

**Implemented:** added CEG, TLN to `TICKER_NODE_MAP` (both `iso="PJM"`, `business_model="merchant"`
— CEG reuses EXC's already-validated COMED/PECO/BGE zones since it's literally Exelon's spun-off
generation fleet; TLN reuses PPL's zone for Susquehanna). Active universe: 18 → 20 tickers
(merchant 2→4, mixed 5, regulated 11 unchanged). 4 new tests (`tests/data/test_universe.py`).

**Second bug found and fixed:** `yf.download()` silently drops a ticker from its batch result
under concurrent/rate-limited load without raising — surfaced when the new tickers' full
historical backfill needed ~96 months re-fetched at once. Added `_download_with_retry()`
(`grid_resilience/data/equity_prices.py`) — retries once on incomplete columns, capped so it
doesn't loop forever on a ticker that genuinely has no data yet (e.g. pre-listing). 3 new tests
(`tests/data/test_equity_prices.py`). Verified live: repeated backfills converged to
CEG 2022-02-02→2025-12-31, TLN 2023-07-05→2025-12-31, VST/NRG essentially full 2018–2025
(NRG has a small, stable ~3-week gap in Dec 2025, likely a Yahoo-side data gap, not a fetch bug).

**Backtest comparison** (18 vs. 20 tickers, same `--arch hard-switch --regulated-signal dc-queue`
config):

| | 18-ticker baseline | 20-ticker (+CEG/TLN) |
|---|---|---|
| Full Sharpe | −0.299 | −0.321 |
| Long Sharpe | +0.212 | +0.201 |
| Short Sharpe | −0.916 | −0.928 |

**Result: no improvement, slightly worse.** Global top-3-long/bottom-5-short ranking means
CEG/TLN just compete in the same pool as everyone else; given their large data gaps, they
likely rarely ranked into the book anyway.

### 5. Price-only vs. blended GSI (separating the "energy price" and "congestion" legs) — tested, blend wins

The thesis names energy-price shocks and congestion shocks as two distinct exposures, but the
existing Grid Stress Index blends `lmp_zscore` (40%), `congestion_frac` (25%),
`reserve_tightness` (20%), and `event_flag` (15%) into one composite that stress-beta regresses
against. Added a `weights` parameter to `build_gsi()`/`build_multi_iso_gsi()` and a
`GSI_WEIGHTS_PRICE_ONLY` config (100% `lmp_zscore`, 0% everything else), plus a
`--price-only-gsi` CLI flag, to test whether isolating price improves on the blend. 4 new tests
(`tests/signals/test_gsi_weights.py`).

**Result:**

| | Blended GSI | Price-only GSI |
|---|---|---|
| Full Sharpe | −0.321 | −0.585 |
| Long Sharpe | +0.201 | **+0.014** |
| IC@21d | −0.0003 | −0.0026 |
| IC@63d | −0.0351 | −0.0383 |

**Isolating price alone is worse, not better** — the long book's Sharpe nearly collapsed to zero.
Congestion/reserve/event evidently *contribute* useful information to the blend rather than
diluting a cleaner price signal, the opposite of the hypothesis that motivated this test. The
symmetric mirror test (100% congestion) was considered and skipped — reasoned to be diagnostic
only (attribution between ingredients), very unlikely to itself beat the existing blend, since
blends of several imperfect proxies routinely outperform any single ingredient alone.

### 6. FTR (Financial Transmission Rights) — real data found, but not the piece needed; incomplete

Explored using FTR clearing price vs. realized settled congestion as a forward-looking,
market-priced congestion signal (an alternative to the backward-looking `congestion_frac`
proxy), motivated by user's own idea partway through this session.

- `gridstatus` has **zero** FTR/CRR/ARR methods for any ISO.
- PJM's Data Miner 2 API (queried directly, bypassing `gridstatus`) does have a genuine
  "Financial Transmission Rights" feed category: `ftr_bids_annual`, `ftr_bids_long_term`,
  `ftr_bids_mnt` (2006+ history, 4-month posting delay), `ftr_cong_lmp` (2014+, only 7-year
  retention — tight against a 2018 backtest start), `mnt_ftr_zonal_lmps` (2010+).
- **But these are all bid-level data** (`quoted_price`, `quoted_mw` — what was bid, not what
  cleared). No feed exposes actual clearing/awarded price directly.
- The "settled" side of the equation (realized congestion from LMP spread) needs nothing new —
  already computable from existing day-ahead LMP data.

**Status: parked, incomplete.** Getting clearing price would need a bespoke scrape of PJM's
public auction-results report pages (same class of workaround as the interconnection queue's
`get_raw_interconnection_queue()`), not a clean API call. ERCOT's CRR equivalent was not checked.
No code changes made.

### 7. Orthogonality check — is the factor just a repackaged quality/beta factor? Clean, but underpowered

Tested the actual `factor_score` (averaged per ticker across the full backtest, from the
hard-switch/dc-queue run) against four standard factors via yfinance's `.info` fields, across
the 20-ticker universe:

| Factor | corr with factor_score | p-value |
|---|---|---|
| ROE | −0.141 | 0.553 |
| Debt/Equity | +0.058 | 0.807 |
| Dividend yield | −0.132 | 0.590 |
| Market beta (standard) | +0.287 | 0.219 |

**No significant correlation with any of them.** If the factor were secretly a repackaged
quality or low-vol factor, at least one of these would likely show |r| in the 0.5–0.7+ range —
nothing here comes close. The largest (market beta, +0.287) is structurally explainable rather
than alarming (merchant names carry higher standard beta and are routed differently by the
hard-switch architecture).

**Caveat:** n=20 is a small cross-section — this rules out a *strong* repackaging, not
necessarily a *moderate* one hiding under low statistical power. This was a one-off diagnostic
script, not added to the pipeline.

## Overall conclusion

Every angle of the original thesis has now been examined:

- **Signal formulations:** stress-beta (original), ICR, DC-load-signal, outage-availability
  (killed — no usable data), RT/DA spread (ERCOT-only practical, PJM blocked), price-only vs.
  blended GSI (blend wins).
- **Portfolio construction:** whole-universe long/short, peer-group baskets, XLU hedge (already
  known to be strictly worse from an earlier grid search), wider universe (2→4 merchant names).
- **Factor validation:** not a repackaged quality/leverage/dividend/beta factor.

**None of these beat XLU.** The short book has no edge over a naive short in every configuration
tested. The most consistent explanation across all of this evidence: the underlying grid-stress
signal is real but weak, and the 2018–2025 window contains too few genuine stress regimes (a
handful of named events, per earlier IC-significance testing) for a monthly-rebalanced
cross-sectional factor to reliably separate winners from losers — a sample-size and
signal-strength problem, not a construction, mislabeling, or fixable-data-availability problem.

## Code shipped this session (all off by default; full test suite 113/113 passing)

| Feature | Flag / entry point | Files |
|---|---|---|
| Peer-group construction | `--peer-group` | `portfolio/construction.py`, `config.py` |
| Universe expansion (+CEG, +TLN) | (always on — universe data) | `data/utility_node_map.py` |
| Price-only GSI | `--price-only-gsi` | `signals/grid_stress_index.py`, `main.py`, `config.py` |
| EIA fetch fix (route/frequency/pagination) | (bug fix, no flag) | `data/eia_data.py` |
| Equity price retry fix | (bug fix, no flag) | `data/equity_prices.py` |

## Open / unfinished threads

- **RT/DA spread, ERCOT-only:** data-availability confirmed practical, but the signal itself
  was never built.
- **FTR clearing price:** real PJM feed category found, but only bid-level data via the
  structured API — a bespoke report scrape (or checking ERCOT's CRR auctions instead) is the
  next step if this is worth pursuing.
- **Congestion-only GSI (the mirror of thread 5):** reasoned through but not run — expected to
  also underperform the blend, low expected value.
- **A fundamentally different pivot** (trading power-price/commodity exposure directly instead
  of the equity wrapper) was raised and explicitly ruled out this session — the client is an
  equity trader and won't trade power.
