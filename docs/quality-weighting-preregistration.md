# Quality / balance-sheet weighting — pre-registration

**Written and locked before any backtest is run.** RESEARCH-LOG open item 13. Original scoping:
`docs/handoff_2026-09-01-grid-buildout-long-short.md` §8.4 (9-name `grid_equipment_basket`
universe). **This probe runs the same idea against the actually-shipped 27-name `screen`
construction** (`electrification_strategy.universe.members("screen", asof, prices)`), per the
task's explicit context update — that's the live-relevant version of the question.

## 1. Universe and baseline

`screen` construction, `electrification_strategy` package. Baseline = shipped **`screen
+val+sleeve`** stack: equal-weight, 25% single-name cap, 42-day post-quarter-end rebalance lag,
252-day listing gate, 20%-vol-target overlay (21d lookback, 1.5x cap), valuation/extension
de-risk multiplier, 15% GLD/IEF sleeve. Documented full-sample figures (`docs/RESEARCH-LOG.md`
§1): Sharpe 0.69, MaxDD −24%, SPY β 0.59. Full sample = `config.START_DEFAULT` (2017-06-01) →
today (2026-09-21). Sub-windows = `electrification_strategy.config.SUBWINDOWS` verbatim:
2019-01-01→2022-12-31 ("2019-22", pre-AI) and 2023-01-01→2026-08-31 ("2023-26", AI-buildout).

The quality-weighted variant changes **only the weighting step** inside `screen` — same
membership rule, same cap, same lag, same listing gate — then is composed through the
**identical** `+val+sleeve` overlay (same vol-target and valuation-overlay code, same sleeve),
so the comparison isolates the weighting scheme rather than mixing in an overlay difference.

## 2. The three quality inputs — data sourcing and point-in-time convention

All three pulled from SEC XBRL `companyconcept` facts, point-in-time dated by the fact's own
`filed` field (the actual EDGAR filing date returned by the API) — **not** an assumed lag like
the project's `_ICR_LAG_DAYS≈45` convention (`grid_resilience/factor/resilience_score.py`). Using
the real filing date is more precise than an assumed lag and is the same convention already used
in `grid_equipment_basket/margin_data.py` and `research/backlog_pricing_power/`. A value is
"known" as of date *t* only if its `availability_date` (= filed date) ≤ *t*.

CIK map: `research/backlog_pricing_power/cik_map.py`'s `CIK_BY_TICKER` (27 tickers, built this
session for the sibling backlog-pricing-power probe) — **not** the original 9-name
`grid_equipment_basket.backlog_data.CIK_BY_TICKER`, which only covers the marquee 9.

**(a) Leverage — Debt / Equity (balance-sheet instant facts), *not* Debt/EBITDA.**
Total debt = sum of whichever of `{LongTermDebtNoncurrent, LongTermDebtCurrent, DebtCurrent,
ShortTermBorrowings, LongTermDebt}` report a value at a given quarter-end (tags vary by filer;
summed across whatever's present, NaN only if *none* report that date). Equity =
`StockholdersEquity`, falling back to
`StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest`. **Tradeoff, stated
upfront:** Debt/EBITDA is the more standard leverage metric named in the original scoping, but
building a reliable quarterly EBITDA series across 27 heterogeneous industrial filers requires
reconciling YTD-cumulative cash-flow D&A tags into discrete quarters — real reconciliation risk
for a half-day-budgeted probe. Debt/Equity uses two balance-sheet *stocks* at the same instant, no
flow reconciliation, and is mechanically far more reliable to get right across a heterogeneous
filer set. This is the "simpler proxy, tradeoff documented" option the task text explicitly
allows, built on the project's existing point-in-time XBRL infra rather than switched to
yfinance (which lacks full 2017-2026 history for a point-in-time backtest).

**(b) Margin stability — reuses `grid_equipment_basket/margin_data.py` directly.** Its
`_union_tags`/`_flow_facts` quarterly revenue (`Revenues` union
`RevenueFromContractWithCustomerExcludingAssessedTax`) and gross-profit (`GrossProfit`, or
revenue-minus-cost fallback) fetch is called unchanged, just re-pointed at the 27-ticker CIK map.
Quarterly gross margin = gross_profit / revenue per disclosed quarter. Stability = **−stdev of
the trailing 8 disclosed quarterly margins** (minimum 6 required, else NaN) — sign-flipped so
higher = more stable = better, consistent with the other two inputs' "higher is better" polarity
before z-scoring.

**(c) FCF conversion — (CFO − Capex) / Revenue, annual (10-K) facts.**
CFO = `NetCashProvidedByUsedInOperatingActivities`. Capex = first-nonempty of
`{PaymentsToAcquirePropertyPlantAndEquipment, PaymentsForCapitalImprovements,
PaymentsToAcquireProductiveAssets}`. Revenue = same union as (b), annual span. All three pulled
with a 350–380-day start/end span filter (10-K figures only) — cash-flow-statement facts are
almost always disclosed as annual or YTD-cumulative, not discrete-quarter, in 10-Qs, so an
80–100-day span filter (as `margin_data.py` uses for revenue/gross-profit) would starve this
input; annual is both the mechanically reliable choice and the natural cadence for this metric.
**FCF/Revenue, not FCF/Net Income:** Net Income is small or near-zero for some name-years in this
universe, which would blow up a Net-Income-denominator ratio; FCF/Revenue is scale-stable and
still answers "how much of top-line converts to free cash."

Update cadence differs by input, by construction: leverage and margin stability can update every
quarter (both are quarterly-disclosed); FCF conversion updates once a year (10-K only). This is
disclosed, not smoothed away.

## 3. Composite formula (pre-registered, exact)

At each point in time, for the `screen`-eligible names on that date:

1. Pull each name's latest known value of the three raw inputs (`availability_date <= asof`).
2. Cross-sectionally **winsorize** each of the three raw series at the 5th/95th percentile
   (guards against a single extreme balance-sheet or FCF outlier dominating the z-score).
3. **Z-score** each winsorized series cross-sectionally. Leverage is sign-flipped (`z = −zscore
   (leverage)`, lower leverage gives higher z); margin stability's sign flip is already baked into
   its definition in §2(b); FCF conversion z as-is.
4. **Composite z = simple mean of the three z-scores** — but only for names with **all three**
   inputs known that date. A name missing even one input (thin XBRL history, e.g. GEV spun off
   April 2024) gets composite = NaN, which is treated as **neutral** (weight factor = 1.0, i.e.
   that name keeps its plain equal share) rather than dropped or imputed. This is the same
   "middle/unknown stays neutral" convention already used by `grid_equipment_basket.basket`'s
   `backlog_tilt_targets` / `coverage_tilt_targets`.
5. **Weight factor = 1 + 0.25 × clip(composite_z, −2, +2)** — bounded in [0.5, 1.5], centered on
   1.0 at composite_z = 0. The 0.25 tilt-strength coefficient is **not searched** — it's chosen to
   match this project's existing rank-tilt precedent (`backlog_tilt_targets`/
   `coverage_tilt_targets` use a ±25% top/bottom-half multiplier, i.e. 1.25x/0.75x); a
   composite_z of ±1 (a fairly typical one-sigma name) lands at the same 1.25x/0.75x the codebase
   already uses elsewhere, just applied continuously instead of as a binary top/bottom-half split.
6. Raw weight = (1/N) × factor for the N eligible names; renormalize to sum to 1; apply
   `grid_equipment_basket.basket.apply_cap` at the same 25% single-name cap the shipped book uses.

This formula is locked. It will not be re-tuned after seeing results.

## 4. Pre-registered gate

**Gate (a) — beats equal-weight on Sharpe in BOTH sub-windows.** Quality-weighted `screen
+val+sleeve` Sharpe > equal-weight `screen +val+sleeve` Sharpe in *both* "2019-22" and "2023-26"
(`config.SUBWINDOWS`). Matches the original spec's "both windows" bar exactly.

**Gate (b) — not a low-vol repackage.** Build the monthly excess-return series `quality_r minus
equal_weight_r` (both fully composed through `+val+sleeve`, compounded to monthly). Build a
monthly **low-vol factor** return series = SPLV monthly return minus SPY monthly return (Invesco
S&P 500 Low Volatility ETF minus the market). **Why SPLV-minus-SPY and not a within-universe
low-vol tilt:** a low-vol construction built from the same 27-name universe risks partial
circularity — low-leverage/stable-margin names mechanically tend to run lower-vol, so a
within-universe control could absorb some of the *same* cross-sectional information the quality
composite is built from, biasing the test toward a false PASS. An out-of-universe, market-wide
low-vol proxy is the harder, cleaner bar, and is a standard off-the-shelf choice (no construction
judgment calls). Regress `excess_r_t = alpha + beta * lowvol_factor_t + eps_t` (OLS, monthly).
**"Residual t >= 1.5" is interpreted as: the t-stat on the regression intercept alpha (the part
of the excess return *not* explained by low-vol-factor exposure) must satisfy |t(alpha)| >= 1.5.**
This is stated explicitly here because the original scoping text ("residual t") is otherwise
ambiguous between the intercept t-stat and something else — the intercept-after-controlling-for-
low-vol-exposure is the natural reading and is what "not just a low-vol repackage" is actually
asking.

**Disposition rule:** PASS requires (a) AND (b). Fails either -> FAIL / NOT ADOPTED, reported
with full honesty on where it broke (as this project's house style requires — see
`docs/backlog-pricing-power-results.md`, `docs/hyperscaler-capex-accelerator-results.md` for the
standard of reporting sub-window splits honestly rather than rounding a partial result up).

## 5. Prior

**Low**, per the original scoping — a within-universe weighting tweak inside an already-shipped
book, likely small and regime-dependent even if it works. Extending from 9 to 27 names raises
the cross-sectional sample the composite z-score is computed over (helps statistical stability
of the z-scores themselves) but doesn't change the *mechanism* being tested, so the prior is
carried forward unchanged from the original handoff.

## 6. Build plan

`research/quality_weighting/`:
- `fetch_fundamentals.py` — XBRL pulls for leverage (new instant-fact fetch) and FCF conversion
  (new annual-flow fetch), reusing `grid_equipment_basket.margin_data`'s quarterly revenue /
  gross-profit fetch unchanged for margin stability. Own disk cache, does not touch
  `grid_equipment_basket`'s or `electrification_strategy`'s production caches.
- `signals.py` — point-in-time visibility filter, winsorize/z-score, composite formula (§3).
- `run_backtest.py` — baseline sanity check (reproduce documented `screen +val+sleeve` ~0.69
  Sharpe), quality-weighted variant, sub-window Sharpes, low-vol-control regression, gate
  evaluation.

Throwaway probe. Not wired into `electrification_strategy/` or `grid_equipment_basket/`
regardless of result, per the task's explicit instruction.
