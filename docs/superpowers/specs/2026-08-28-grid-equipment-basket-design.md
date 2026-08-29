# Design Spec — Grid Equipment Suppliers Thematic Basket (Backtested, Layered)

**Status:** Approved design, not yet built.
**Location:** new top-level directory `grid_equipment_basket/`, sibling to `grid_resilience/` and `dc_demand_basket/`.
**Supersedes:** `grid_equipment_basket/2026-08-28-grid-equipment-suppliers-design.md` (the original draft spec). This
document keeps that spec's thesis and universe but resolves two open decisions in it: (1) validation is a
short-window backtest over the AI-buildout regime, not forward-tracking-only; (2) construction is layered —
prove the theme first, add a backlog tilt only if the theme clears a stated bar.

## 1. Thesis

Grid / data-center electrical-infrastructure suppliers are a structurally different bet on the same multi-year
trend the `grid_resilience` and `dc_demand_basket` work targets (AI power demand, electrification, reshoring
driving a grid buildout), but with a data advantage: these companies have **direct revenue exposure to the
capex** rather than inferred demand exposure, and order **backlog / book-to-bill** is a standard,
quarterly-disclosed, quantitative figure in their own SEC filings — not something inferred from a third
party's interconnection queue.

The strategy is scoped as a **long-only thematic basket**, matching `dc_demand_basket`'s scope, not a
cross-sectional factor. It is built and validated in two layers:

- **Step 1 — the theme.** Does simply holding an equal-weight basket of the verified names beat broad
  industrials (XLI) and off-the-shelf thematic ETFs (GRID, PAVE) over 2023-present?
- **Step 2 — the tilt (contingent on Step 1 passing).** Does tilting basket weights by disclosed
  order-backlog growth improve on equal-weight?

If Step 1 fails its decision gate (§6), the project stops there with a written negative result and **no
backlog data is collected** — the expensive, error-prone manual work is gated behind evidence that the theme
is worth it.

## 2. Candidate universe

**Inclusion is decided on business description only — never on historical returns.** A name qualifies if a
plain reading of its most recent 10-K says it derives material revenue from selling grid / data-center
electrical equipment or from engineering/constructing electric-power infrastructure. This rule is the primary
defense against the hindsight bias described in §7 and must be applied literally, including for names that
have been poor performers.

US-listed candidates (research starting point — each verified in `candidate_research.md` before inclusion):

| Ticker | Company | Bucket | Verification notes |
|---|---|---|---|
| ETN | Eaton | Electrical equipment (transformers, switchgear, DC power) | Large diversified industrial; Electrical Americas + Electrical Global segments. Confirm they are a majority of revenue and that backlog is disclosed (segment-level preferred). |
| HUBB | Hubbell | Utility & electrical solutions (grid components, transformers, meters) | Utility Solutions segment. Confirm backlog disclosure. |
| GEV | GE Vernova | Grid equipment (HVDC, transformers, switchgear) + power generation | **Listed 2024-04-02** (spun off from GE). Short price history — enters the basket on its first trading day; the basket equal-weights only names with data on each date. Electrification segment backlog disclosed. |
| VRT | Vertiv | Data-center power & thermal management | Cleanest single-name expression of the theme. Reports backlog and book-to-bill every quarter. |
| PWR | Quanta Services | Electric-power infrastructure EPC | Reports remaining performance obligations (RPO) and 12-month / total backlog every quarter. Electric Power Infrastructure segment. |
| MYRG | MYR Group | Transmission & distribution EPC | Reports backlog. Small-cap (~$2-3B) — confirm average daily dollar volume is adequate. |
| NVT | nVent Electric | Electrical connection & protection, enclosures, liquid cooling | Data-center exposure via enclosures / liquid cooling. Confirm backlog disclosure and DC attribution. |
| FLNC | Fluence Energy | Grid-scale battery storage | Reports order intake / backlog explicitly. Chronic underperformer with going-concern-adjacent noise. **Include if it meets the business-description test** — deliberately keeping a known loser limits cherry-picking. |
| PRIM | Primoris Services | Utility / power-delivery infrastructure EPC | Not in the original spec. Evaluate: Utilities and Energy segments, power-delivery backlog. Include or exclude with reasoning. |

**Foreign-listed names** (ABB, Siemens Energy, Prysmian, Nexans): logged in `candidate_research.md` with the
thesis rationale, but **excluded from the backtestable basket** — their only US-accessible lines are OTC ADRs
(e.g. ABBNY, SMNEY) with poor adjusted-close quality and thin volume, not comparable to the primary listings.
Record this as an explicit exclusion, not an oversight.

**Expected verified count:** 6–9 names. Report the actual count honestly; a small basket is not a failed
deliverable if verification was done correctly.

## 3. Benchmarks

| Benchmark | Role |
|---|---|
| **XLI** (Industrials Select Sector SPDR) | Primary. Most of the universe is GICS Industrials, not Utilities — this is the relevant sector benchmark. |
| **SPY** (S&P 500) | Secondary — broad-market reference. |
| **XLU** (Utilities Select Sector SPDR) | Secondary — continuity with the `grid_resilience` / `dc_demand_basket` work, which benchmarks against XLU. |
| **GRID** (First Trust NASDAQ Clean Edge Smart Grid Infrastructure) | Honesty check — a rules-based, third-party thematic ETF covering nearly this exact theme. |
| **PAVE** (Global X U.S. Infrastructure Development) | Honesty check — broader US infrastructure-buildout thematic ETF. |

The GRID/PAVE comparison is the core methodological guard: if a hand-picked basket cannot beat a rules-based
thematic ETF that already exists, the "strategy" is just exposure to the theme, and the writeup will state
that plainly.

## 4. Construction — Step 1 (equal-weight)

- **Equal-weight** across all verified names that have price history as of each rebalance date. A name with no
  data before its listing date (e.g. GEV before 2024-04-02) is simply absent from the weight vector until it
  has data, at which point the next rebalance includes it.
- **Quarterly rebalance**, dated approximately six weeks after each calendar quarter-end
  (~mid-Feb, mid-May, mid-Aug, mid-Nov), so that Step 2's backlog figures would actually be disclosed and
  available at each rebalance. Step 1 and Step 2 therefore run on the same calendar.
- **Price drift between rebalances** — weights are set at the rebalance date and drift with relative price
  performance until the next one. No daily re-equalization.
- **Single-name cap: 25%.** Documented here rather than buried in code. With 6–9 names the equal weight is
  11–17%, so the cap only binds after a Step 2 tilt or if the universe verifies very small.
- **Long-only, fully invested, no leverage, no short leg.** The thesis has no natural short side; forcing one
  adds risk without economic rationale (same reasoning as `dc_demand_basket`).
- **Total-return basis** — yfinance auto-adjusted closes (dividends reinvested).
- **Transaction costs** — turnover at this cadence and breadth is negligible; costs are noted in the README
  but not modeled. Rebalance frequency is documented as a low-sensitivity parameter, not an optimized one.

## 5. Validation

Primary window: **2023-01-01 → the last complete calendar month** at implementation time.

Prior-regime panel: **2020-01-01 → 2022-12-31**, computed and reported alongside the primary window. The
original spec is explicit that pre-2023 grid-capex conditions were a different demand environment; the
response is to show both windows side by side, not to hide the earlier one or to pretend the primary window
is longer than it is.

Metrics, matching `grid_resilience/portfolio/backtest.py` conventions (risk-free rate 4% annual, 252
annualization factor, Sharpe = mean(excess) / std(excess) × √252, max drawdown from the cumulative-return /
running-max series):

- CAGR, annualized volatility, Sharpe, Sortino, maximum drawdown
- Per-calendar-year total return (basket and each benchmark)
- Correlation of daily returns to XLI
- Tracking error and information ratio vs XLI
- Excess CAGR vs each of the five benchmarks

**Mandatory caveat, stated in the README and any results doc:** the primary window is only ~3.5 years (~43
monthly observations) and GEV contributes less than half of it. The standard error on an annualized Sharpe
estimate from a sample this size is on the order of ±0.5 (so a ~±1.0 band at 95%). "Beats the benchmark" in
this document means a directional point-estimate difference, not a statistically significant one. This is
exploratory evidence about whether the theme is worth pursuing further, consistent with the original spec's
instruction not to manufacture a Sharpe/IC number to defend.

### Decision gate

Step 1 **passes** — and only then is Step 2 built — if **all** of the following hold over the primary window:

1. Basket CAGR > XLI CAGR, and
2. Basket Sharpe > XLI Sharpe, and
3. Basket CAGR is within 2 percentage points of the better of GRID / PAVE, or higher.

If Step 1 fails, the deliverable is `candidate_research.md`, the Step 1 code, and a short results writeup
recording the negative result. **No backlog data is collected.**

## 6. Construction — Step 2 (contingent on the §5 gate passing)

**Data collection — `backlog_data.py` + `data/backlog_quarterly.csv`:**

- **Point-in-time discipline.** Backlog figures are as-of-quarter-end but disclosed ~4–6 weeks later. Use the
  filing / earnings-call date as the availability date, never the quarter-end date.
- **Structured first.** For each name, try the SEC XBRL `companyconcept` API
  (`https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/RevenueRemainingPerformanceObligation.json`,
  plus the `...Current` / `...Noncurrent` variants). Free, no key, and every data point carries its filing
  date. Expected to cover PWR and VRT and possibly GEV; confirmed *not* available for ETN (checked during
  design — Eaton does not tag this concept).
- **Hand-collected fallback.** For names that disclose backlog only as a non-GAAP figure in the press release
  or MD&A, record the disclosed number, the filing / call date, and the exact source URL. Each row carries a
  `disclosure_type` field: `xbrl_rpo` | `nongaap_backlog_total` | `nongaap_backlog_segment` |
  `book_to_bill_only`. **Definitions are not coerced to a common metric** — the original spec is emphatic on
  this and it is preserved here. Capture what each company actually discloses and record the inconsistency.
- **Verification bar.** Every hand-collected row cites a primary source (filing or transcript), not trade
  press. This project has already found a previously-"confirmed" data point wrong by ~10x; re-verification
  discipline applies.

**Schema for `data/backlog_quarterly.csv`:**

`ticker, quarter_end, availability_date, metric_value, metric_unit, disclosure_type, segment_scope (total | segment_named), source_url, notes`

**Signal:** trailing-four-quarter backlog growth rate per name (YoY change in the disclosed figure). For a
name that discloses only book-to-bill, use the book-to-bill level (>1 = backlog growing) as the signal.
Rank the growth-rate names and the book-to-bill names separately, then combine on rank so the two disclosure
styles are comparable without inventing a common absolute metric.

**Tilt:** start from the Step 1 equal weights; multiply the top-half-ranked names by a fixed **1.25×** and the
bottom-half by **0.75×**; renormalize to sum to 1; re-apply the 25% cap. With an odd universe count the
median-ranked name is left untilted (1.0×). The tilt magnitude is a single, documented constant in
`config.py` — not a fitted parameter and not an opaque z-score.

**Re-validation:** re-run §5 for the tilted basket. Report tilted vs equal-weight vs all five benchmarks side
by side. If the tilt does not improve **both** Sharpe and CAGR relative to equal-weight, equal-weight remains
the recommended basket and the README says so explicitly.

**Descriptive backlog check (from the original spec, kept):** for each name, plot backlog growth rate against
subsequent one- and two-quarter forward return. Exploratory only — it informs whether the Phase 2 factor
option (§8) is worth a future session, not a number to defend now.

## 7. Known biases and design mitigations

**Survivorship / hindsight bias is the headline risk.** The universe is chosen in 2026 with full knowledge
that VRT, GEV, and PWR have been large winners over the backtest window. A backtest built on that universe is
biased upward. Mitigations, all stated in the README and results writeup:

1. Inclusion is decided on business description from the 10-K only, never on returns (§2).
2. Names that meet the business test but have been poor performers (FLNC) are deliberately kept in.
3. Equal-weighting removes the ability to cherry-pick weights toward the winners.
4. The GRID / PAVE benchmarks are rules-based third-party thematic baskets; if the hand-built basket merely
   matches them, the honest finding is "this is just the theme," and that is what will be written.
5. The results writeup claims only what a short, biased window supports — directional, exploratory evidence.

**Short sample.** ~43 monthly observations (§5), GEV covering less than half of them — the Sharpe standard-error
caveat is mandatory in all reporting.

**Look-ahead in Step 2.** Filing-date availability dating (§6) is the mitigation; the rebalance calendar is
set six weeks after quarter-end so the data would genuinely have been in hand.

## 8. Phase 2 — cross-sectional factor (explicitly NOT this spec's deliverable)

Backlog / book-to-bill is a recurring quarterly time series across (potentially) 6–9 names with several years
of history — thicker than `dc_demand_basket`'s event data. **If** verification lands on 8+ names with clean,
comparable, segment-level backlog and multiple years of quarterly history, a cross-sectional factor (rank by
backlog-growth surprise) becomes methodologically defensible in a way it was not for the DC-demand data.

This spec does **not** build that. It is recorded here and in the README so a future session does not
rediscover the option, and so nobody quietly reintroduces `build_factor()` / z-scoring / IC machinery into
what is scoped here as a basket. Building the factor requires its own spec.

## 9. Module layout

```
grid_equipment_basket/
  __init__.py
  config.py                     # universe list, benchmark tickers, windows, rf rate,
                                #   rebalance cadence, single-name cap, Step 2 tilt magnitude
  data/
    __init__.py
    prices.py                   # yfinance fetch + parquet cache + gap-fill + retry-on-incomplete
                                #   (mirrors grid_resilience/data/equity_prices.py patterns)
    cache/                      # *.parquet — gitignored
    backlog_quarterly.csv       # Step 2 only — schema in §6
  backlog_data.py               # Step 2 only — SEC XBRL companyconcept fetch + CSV loader
  basket.py                     # build_weights(): equal-weight + quarterly rebalance + cap
                                #   + optional backlog tilt; returns a (date x ticker) weight frame
  backtest.py                   # chain basket returns, compute §5 metrics vs all benchmarks,
                                #   render an equity-curve + drawdown chart
  __main__.py                   # minimal CLI: python -m grid_equipment_basket
                                #   [--start 2023-01-01] [--end YYYY-MM-DD] [--tilt backlog]
  candidate_research.md         # per-name verification log (metric disclosed, segment vs total,
                                #   grid/DC attribution, include/exclude + reasoning; negative
                                #   results recorded so future sessions don't re-research)
  README.md                     # current composition, weights, window, benchmark rationale,
                                #   Step 1 result, the §5 gate outcome, the §7 bias disclosures,
                                #   and the §8 Phase-2 note
```

Reuse: `prices.py` mirrors — and where practical imports from — `grid_resilience/data/equity_prices.py`
(specifically its `_download_with_retry` incomplete-columns handling). No changes to `grid_resilience/`.

## 10. Testing

Test-first (TDD). All tests are offline — no network — using small deterministic fixtures.

- `tests/grid_equipment_basket/test_basket.py`
  - equal weights sum to 1 and are equal across names present on a date
  - a name with no data before its listing date is excluded, then included at the first rebalance after data begins
  - 25% cap is enforced after a tilt and weights still sum to 1
  - quarterly rebalance dates land ~6 weeks after quarter-end
  - backlog-tilt math (1.25× / 0.75× → renormalize → re-cap) on a fixed rank fixture
- `tests/grid_equipment_basket/test_backtest.py`
  - basket return chaining reproduces a hand-computed two-asset, two-period example
  - Sharpe and max-drawdown match known closed-form values on a constructed series
  - benchmark alignment handles differing trading-day sets (date intersection, no forward-fill of prices)
- `tests/grid_equipment_basket/test_prices.py`
  - cache hit / miss / gap-fill with a mocked fetcher
  - retry-on-incomplete-columns behavior, capped so a genuinely dataless ticker doesn't loop

Live fetch is validated once manually during implementation and the result recorded in the README.

## 11. Deliverables

**Step 1 (always):**
- `grid_equipment_basket/candidate_research.md`
- `grid_equipment_basket/config.py`, `data/prices.py`, `basket.py`, `backtest.py`, `__main__.py`
- `grid_equipment_basket/README.md` with the Step 1 result and the §5 gate outcome
- Test suite per §10 (basket + backtest + prices)
- One-line pointer to the module added to the repo-root `README.md`
- A short results writeup (in `docs/`) recording Step 1 numbers and the gate decision

**Step 2 (only if the §5 gate passes):**
- `grid_equipment_basket/backlog_data.py`, `data/backlog_quarterly.csv`
- Backlog-tilt path in `basket.py` + tests
- Tilted-vs-equal-weight comparison added to the README and the results writeup
- The descriptive backlog-vs-forward-return check

## 12. Out of scope

Cross-sectional z-scoring, `build_factor()` / `hard_switch` integration, IC@21d or any factor-style
predictive-power metric, any claim of statistical significance, an 8-year historical backtest as primary
validation, any short leg, and any modification to `grid_resilience/`. The Phase 2 factor (§8) requires its
own spec.
