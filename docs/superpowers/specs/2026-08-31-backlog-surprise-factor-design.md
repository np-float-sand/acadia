# Design Spec — Backlog Growth Surprise: Event Study + Cross-Sectional Factor

**Status:** Approved design, not yet built.
**Location:** new top-level module `backlog_factor/`, sibling to `grid_resilience/`, `grid_equipment_basket/`, `dc_demand_basket/`.
**Origin:** the `grid_equipment_basket` work concluded there is no differentiated stock-selection edge inside the
grid/AI-power theme at 9 names over 3.5 years, and the theme itself is mainstream (a public Fidelity
learning-center piece recommends GE Vernova / Eaton / Quanta by name). This spec pivots to a **breadth-first
cross-sectional factor** on a much wider universe, trading a less-mined disclosed series. It also rehabilitates
`grid_equipment_basket` spec §8 (the "backlog-*surprise*, not raw growth" idea, explicitly deferred there
pending "8+ names with comparable segment-level history") — that condition is now met by widening the universe.

---

## 1. Thesis

Order backlog / remaining performance obligations (RPO) is disclosed every quarter by order-driven
industrials. The market prices the **level** and the **trend** of that series; it under-reacts to the
**residual** — the part of this quarter's backlog change that a naive extrapolation of the company's own
recent order trend would not have predicted. That residual, **backlog growth surprise**, is a
fundamental-momentum signal on a series far less mined than analyst EPS-estimate revisions, and
post-disclosure prices drift toward it for weeks.

The failed `grid_equipment_basket` Step 2 tilt ranked names on **raw** backlog growth — which is mostly the
trend the market already extrapolated, and it over-weighted a fast-grower off a tiny base (FLNC). Ranking on
the **surprise** (growth minus its own trailing trend) strips the extrapolated part out. Raw growth is a
control in this spec, not the signal.

**Breadth is the design constraint.** The strategy must make thousands of independent bets — a wide
cross-section rebalanced monthly over multiple years — not a handful of concentrated positions. A signal that
only works on 9 names over one window is a hindsight artifact; this spec is scoped so the evidence is
cross-sectional.

---

## 2. Universe

US-listed industrials / capital-goods businesses that disclose a forward order metric (backlog, book-to-bill,
or RPO). Sectors in scope: machinery, electrical equipment, engineering & construction / EPC, aerospace &
defense, building products, semiconductor capital equipment, and rail / agricultural / construction equipment.

**Inclusion is by business description only — never by returns.** A name qualifies if a plain reading of its
most recent 10-K shows it is an order-driven manufacturer or contractor that discloses a forward order metric.
Membership is refreshed annually from filings and applied point-in-time: a name enters the factor only once it
has the required signal history (§4), and a name that stops disclosing (or is acquired / delisted) leaves at
that date.

**Phase 1 — the XBRL-RPO sub-universe (~75–125 names).** Every name that tags
`us-gaap:RevenueRemainingPerformanceObligation` (or the `...Current` / `...Noncurrent` pair). Zero
hand-collection; every data point carries its SEC filing date. This is the universe for the §5 event study and
the first §6 factor.

**Phase 2 — full universe (~200 names), contingent on the §7 factor gate.** Hand-collect the non-GAAP
backlog / book-to-bill figures for the order-driven names that do not tag RPO (most pure-play defense and
machinery names disclose backlog only in the MD&A or press release). Each hand-collected row cites a primary
source (filing or transcript), carries its disclosure date, and records `disclosure_type`
(`nongaap_backlog_total` | `nongaap_backlog_segment` | `book_to_bill_only`). Definitions are not coerced to a
common metric — the surprise is computed on each name's own series, so level-definition differences wash out
(but a mid-sample definition change does not — see §8).

**Expected count:** report the actual Phase 1 count honestly. If it lands below ~60 the breadth premise is
weak and that is a finding, not a failure to paper over.

---

## 3. Benchmarks & neutralization targets

| Item | Role |
|---|---|
| **GICS industry-group ETF** (per name) | Abnormal-return benchmark for the event study; neutralization target for the factor. Where no clean group ETF exists, beta-adjusted **SPY**. |
| **SPY** | Broad-market reference; residual-beta hedge target for the factor. |
| **Analyst EPS-estimate-revision momentum factor** (same universe) | The load-bearing additivity control (§7). |
| **12-1 price momentum**, **profitability/quality** | Secondary additivity controls. |
| **Raw backlog-growth factor** (the failed tilt's signal, same construction) | Comparison — surprise must beat it. |

Metric conventions match `grid_resilience` / `grid_equipment_basket`: risk-free 4% annual, 252 annualization,
Sharpe = mean(excess)/std(excess)×√252, max drawdown from the cumulative simple-return / running-max series,
**simple returns** throughout.

---

## 4. Signal — backlog growth surprise

Computed per name on each SEC filing date (the **availability date** — never the period-end date):

- `g_t` = natural-log change in the disclosed backlog / RPO figure from the prior disclosed quarter (QoQ).
  A **YoY** variant (`g_t` vs the figure four quarters prior) is computed alongside for names with seasonal
  order patterns; the event study reports both and the factor uses whichever the event study shows is
  stronger, fixed thereafter.
- `expected_g_t` = trailing **four-quarter mean** of `g` (a naive random-walk-with-drift extrapolation — a
  fixed formula, **not** a fitted model).
- **`surprise_t = g_t − expected_g_t`.**

Then, on each monthly rebalance date, across all names with a currently-valid surprise:

- **winsorize** at ±3σ,
- **cross-sectionally standardize** (z-score),
- **neutralize** to GICS industry group and to size (log market cap) — regress the z-score on
  industry-group dummies + log-mktcap, take the residual,
- **staleness decay**: multiply by a linear factor that is 1.0 on the disclosure date and declines to 0.0
  over `DRIFT_HORIZON_DAYS` (set from the §5 event study; provisionally 63 trading days). A surprise older
  than `DRIFT_HORIZON_DAYS` contributes 0.

**Guards** (mirroring `grid_equipment_basket/backlog_data.py`): NaN unless the name has ≥ **6** disclosed
quarters of history; NaN if the `iloc[-1]..iloc[-5]` quarter-end span is outside 300–430 days (gappy series);
NaN if the latest disclosed quarter is > **200** days before the rebalance date.

---

## 5. Event study (Phase 1 — runs first)

**Purpose:** measure whether backlog growth surprise predicts post-disclosure abnormal returns, how large the
effect is, and how long the drift lasts (which sizes the factor's holding period). No factor is built until
this clears its gate.

**Method.** For every (name, disclosure-date) pair with a valid surprise:

- **Cumulative abnormal return (CAR)** over `[0, +5]`, `[0, +21]`, `[0, +42]`, `[0, +63]` trading days,
  where day 0 is the first full trading day after the SEC filing date. Abnormal = raw name return minus the
  GICS-industry-group ETF return (beta-adjusted SPY where no clean group ETF exists). Both raw and abnormal
  are reported.
- **Quintile sort** of `surprise` (cross-sectional, per calendar month). Report mean CAR per quintile per
  window, and the **Q5 − Q1 spread** with a **month-clustered t-statistic** (clustering by month absorbs the
  cross-sectional return correlation that would otherwise inflate significance).
- **Monotonicity** of mean CAR across Q1 → Q5 (a real signal ramps; a fluke is spiky).
- **Positive vs negative surprises reported separately** — is the drift symmetric, or downside/upside-skewed
  the way post-earnings-announcement drift is.
- **Sub-period stability** — split the sample in half by date; the effect must have the same sign and a
  clustered t > 1 in each half.
- **By industry group** — the effect must appear with the right sign in ≥ 3 of {machinery, electrical,
  aerospace & defense, E&C, building products}, not be carried by one group.

### Pre-registered event-study gate

**Passes iff all hold:**

1. Q5 − Q1 abnormal CAR at its peak window is **positive** and **monotone** across quintiles, with a
   month-clustered **t > 2**;
2. the effect holds (same sign, clustered t > 1) in **both** sub-period halves;
3. the effect is **not concentrated** — right sign in ≥ 3 of the 5 major industry groups.

**Fail →** the deliverable is `candidate_research.md`, the Phase-1 event-study code, and a short written
negative result. **No Phase 2 hand-collection.** `DRIFT_HORIZON_DAYS` for a passing result is set to the
window at which the Q5 − Q1 CAR stops growing (rounded to 21 / 42 / 63).

---

## 6. Factor construction (contingent on the §5 gate passing)

- **Monthly rebalance.** On each rebalance date, each name's signal = its most recent valid `surprise` per
  §4 (winsorized, standardized, industry- and size-neutralized, staleness-decayed over
  `DRIFT_HORIZON_DAYS`).
- **Dollar-neutral, sector-neutral long/short:** long the top quintile, short the bottom quintile,
  **equal-weight within each leg**; both legs constructed so net GICS-industry-group and net size exposure ≈
  0. Residual market beta hedged to ≈ 0 with SPY.
- **Costs** modeled at **10 bps per name per side** (documented, with a sensitivity in the results doc);
  turnover reported (the signal moves only on each name's quarterly disclosure plus the monthly staleness
  decay).
- No leverage beyond the 1× long / 1× short gross. Quintile cut, winsor level, and `DRIFT_HORIZON_DAYS` are
  **documented constants**, and the §7 holdout is reserved so none of them is chosen on out-of-sample data.

---

## 7. Validation

### 7.1 Windows

Structured window: **~2019-01-01 → the last complete calendar month** at implementation (the XBRL RPO tag is
common only post-ASC-606). ~28 quarters × ~75–125 names. The breadth is cross-sectional; the calendar span is
one macro cycle and that limitation is stated in every results doc (§8). Phase 2's hand-collected series
extend the calendar back for the older industrials and that panel is reported separately.

### 7.2 Method

- **Walk-forward, expanding window:** at each rebalance, the signal is formed using only data available on
  that date — no full-sample standardization, no full-sample industry/size betas. The equity curve is a true
  pseudo-live series.
- **Reserved holdout:** the last **18 months** of the structured window. No parameter
  (`DRIFT_HORIZON_DAYS`, quintile cut, winsor level, QoQ-vs-YoY choice) is selected on it.

### 7.3 Metrics

- **IC** (Spearman rank correlation of `surprise` to forward 21-day and 63-day return), monthly: mean IC,
  IR of IC (mean/std × √12), hit rate, IC decay curve.
- **Quintile-spread portfolio:** CAGR, annualized vol, Sharpe, Sortino, max drawdown, turnover,
  ADV-based capacity estimate, per-calendar-year return.
- **Additivity (the load-bearing test):** regress the factor's monthly returns on (a) an analyst
  EPS-estimate-revision momentum factor built on the same universe, (b) a 12-1 price-momentum factor,
  (c) a profitability/quality factor. Backlog-surprise must retain a **positive alpha with t > 2** after
  these controls, and its return correlation to the revision-momentum factor is reported — if that
  correlation is above ~0.7 the "less-mined series" claim is weak even with residual alpha, and the results
  doc says so.
- **Control-factor construction:** 12-1 price momentum is built from `data/prices.py`; revision momentum
  from `data/estimates.py` (3-month change in consensus FY1 EPS divided by the absolute prior estimate,
  same winsor + neutralization as the signal); profitability/quality from a lightweight fundamentals pull
  (trailing ROA and gross-margin stability). If the quality data proves impractical to assemble
  point-in-time, the additivity test drops to the two momentum controls and the results doc says so.
- **vs raw growth:** the identical factor built on `g_t` (raw backlog growth, the failed tilt's signal)
  instead of `surprise_t`. Surprise must beat raw growth on IR and on walk-forward Sharpe.

### 7.4 Pre-registered factor gate

**Passes iff all hold:**

1. Walk-forward quintile-spread **Sharpe > 0.5** over the full structured window (net of the §6 costs);
2. quintile-spread return is **positive in the reserved 18-month holdout**;
3. the §7.3 additivity **alpha is positive with t > 2**;
4. **surprise beats raw growth** on IR and walk-forward Sharpe.

**Passes →** Phase 2 (hand-collect to ~200 names), re-run §5 and §6–§7 on the wider universe, and the
deliverable states whether the wider universe improves the factor or not.
**Fails →** written negative result; the event-study finding (if it passed §5) is still reported as
"a real drift that does not survive as a tradeable monthly factor after costs and controls."

---

## 8. Known biases and caveats (stated in the results doc)

- **Short calendar span.** ~7 years of structured data, one macro cycle (2019 late-cycle → 2020 COVID →
  2021–22 boom → 2022–23 rate shock → 2023–26 AI capex). Breadth lives in the cross-section; a regime that
  broadly breaks fundamental momentum (2022-style) breaks this factor too, and no amount of cross-sectional
  breadth fixes a time-series that is one cycle long.
- **RPO / backlog definition drift.** A firm restating what is included in RPO mid-sample (e.g. adding or
  removing long-term service agreements) creates a spurious surprise for that name. Screen every series for
  level discontinuities; NaN the surprise across a detected break and for the following four quarters.
- **Survivorship / look-back universe.** The universe is assembled in 2026; E&C firms that blew up and
  defense names that were acquired are missing. Build the point-in-time membership list from historical
  filings where feasible and document the residual upward bias.
- **Look-ahead in parameter choice.** `expected_g` is a trailing mean with no fitting, but the quintile
  breakpoints, `DRIFT_HORIZON_DAYS`, and the QoQ-vs-YoY choice are selected on the event-study sample — the
  reserved 18-month holdout (§7.2) is the mitigation.
- **Factor crowding.** Fundamental momentum / estimate revisions is a well-known factor family. The §7.3
  additivity test is the guard; a high return correlation to revision momentum is disclosed even when the
  residual alpha is significant.
- **Capacity.** Many names are mid-caps; a dollar-neutral quintile long/short at size moves them. Capacity
  is estimated from trailing ADV and reported honestly, not assumed away.
- **Hand-collection error (Phase 2).** This project has already found a previously-"confirmed" figure wrong
  by ~10×. Every hand-collected row cites a primary source; a re-verification pass is part of Phase 2.

---

## 9. Approach — phased, gated data assembly

Roughly half the target universe discloses backlog only as a non-GAAP figure, and hand-collecting ~100 names'
multi-year quarterly series is slow and error-prone. Rather than spend that effort before knowing the effect
exists:

- **Phase 1** uses only the XBRL-RPO sub-universe (~75–125 names, zero hand-collection) for the §5 event
  study and the first §6 factor + §7 validation.
- **Phase 2** — hand-collect the non-GAAP names to ~200 — is **unblocked only if the §7 factor gate
  passes**. This mirrors how `grid_equipment_basket` gated its hand-collected backlog data behind a Step 1
  decision gate.

If Phase 1's XBRL-RPO count lands below ~60 names, the event study still runs but the breadth premise is
flagged as weak in the writeup.

---

## 10. Module layout

```
backlog_factor/
  __init__.py
  config.py                  # Phase-1 universe (CIKs + tickers); GICS industry-group map (hand-curated:
                             #   seeded from yfinance sector/industry, then reviewed against 10-K segment
                             #   descriptions -- yfinance industry labels are coarse);
                             #   structured window, holdout start (18 months back),
                             #   DRIFT_HORIZON_DAYS (set from the event study; default 63),
                             #   quintile cut (0.2), winsor sigma (3.0), cost bps (10),
                             #   history-quarters minimum (6), staleness / span guards
  data/
    __init__.py
    rpo.py                   # SEC XBRL companyconcept fetch for RevenueRemainingPerformanceObligation
                             #   (+ Current/Noncurrent), filing-date dating, per-CIK parquet cache.
                             #   Mirrors grid_equipment_basket/margin_data.py + backlog_data.py.
    prices.py                # yfinance adjusted closes for the universe + industry-group ETFs + SPY
    estimates.py             # analyst EPS consensus-estimate history for the revision-momentum control
    cache/                   # *.parquet — gitignored
    backlog_manual.csv       # Phase 2 only — hand-collected non-GAAP backlog; schema in §2
  signal.py                  # g_t, expected_g_t, surprise_t; winsorize; cross-sectional z-score;
                             #   industry + size neutralization; staleness decay. Explicit, tested steps —
                             #   no opaque build_factor() blob.
  event_study.py             # CAR windows; quintile CARs; month-clustered t-stats; monotonicity;
                             #   positive/negative split; sub-period halves; by-industry-group; plots
  factor.py                  # monthly cross-sectional L/S: quintile legs, industry + size neutralization,
                             #   residual-beta hedge, turnover
  backtest.py                # IC series + decay; quintile-spread portfolio metrics; capacity;
                             #   additivity regression (revision-mom / price-mom / quality);
                             #   walk-forward + holdout; raw-growth comparison
  __main__.py                # CLI: python -m backlog_factor [--phase 1|2] [--event-study] [--factor]
                             #   [--start YYYY-MM-DD] [--end YYYY-MM-DD]
  candidate_research.md       # per-name inclusion log (business test, RPO tag Y/N, definition notes,
                             #   include/exclude + reasoning; negative results recorded)
  README.md                  # universe, signal definition, event-study result + gate outcome,
                             #   factor result + gate outcome, §8 caveats
docs/
  superpowers/specs/2026-08-31-backlog-surprise-factor-design.md   # this spec
  backlog-surprise-factor-results.md                               # event study + factor results, both gates
```

Reuse: `data/rpo.py` follows `grid_equipment_basket/margin_data.py` (SEC XBRL `companyconcept`, User-Agent
`acadia-research sand.gh1902@gmail.com`, monthly/per-CIK parquet cache, hand-collected-fallback discipline).
Metric helpers match `grid_resilience` / `grid_equipment_basket` conventions. **Cross-sectional z-scoring is
appropriate in this module** — it is a genuine cross-sectional factor, unlike the `grid_equipment_basket`
basket module where spec §12 forbids it. The standardize / neutralize / decay steps are explicit and
separately unit-tested.

---

## 11. Testing

Test-first (TDD). All offline, deterministic fixtures, no network.

- `tests/backlog_factor/test_signal.py`
  - `g_t` = log QoQ change on a constructed backlog series; `expected_g_t` = trailing-4Q mean; `surprise_t`
    = residual — hand-computed
  - winsorize clips at ±3σ; cross-sectional z-score has mean ≈ 0 / std ≈ 1 on a fixture
  - industry + size neutralization: the residual is orthogonal to the industry dummies and to log-mktcap
  - staleness decay: 1.0 at disclosure, 0.0 at `DRIFT_HORIZON_DAYS`, linear between; > horizon → 0
  - guards: < 6 quarters → NaN; gappy `iloc[-1]..iloc[-5]` span → NaN; latest quarter > 200 days stale → NaN
  - RPO definition-break screen: a level discontinuity NaNs the surprise across the break and the next 4Q
- `tests/backlog_factor/test_event_study.py`
  - CAR over a window on a constructed return series matches a hand-computed value; day 0 = first trading
    day after the filing date
  - quintile assignment is cross-sectional per month
  - month-clustered t-stat matches a closed-form value on a constructed panel (and differs from the naive
    OLS t-stat)
  - monotonicity flag on a monotone vs a spiky quintile-CAR vector
- `tests/backlog_factor/test_factor.py`
  - long/short legs are the top/bottom quintile; each leg equal-weight; net industry-group and net size
    exposure ≈ 0; dollar-neutral
  - turnover computed correctly on a two-rebalance fixture
  - walk-forward has no look-ahead: a fixture where expanding-window vs full-sample standardization give
    different signs, and the walk-forward path uses only past data
- `tests/backlog_factor/test_backtest.py`
  - IC = Spearman(signal, forward return) on a fixture
  - additivity regression recovers a known alpha/beta on constructed factor + control series
  - raw-growth vs surprise: the harness runs both and reports both

Live fetch is validated once manually during implementation and the Phase-1 universe count recorded in the
README.

---

## 12. Deliverables

**Phase 1 (always):**
- `backlog_factor/candidate_research.md` (Phase-1 XBRL-RPO universe, verified)
- `backlog_factor/config.py`, `data/rpo.py`, `data/prices.py`, `data/estimates.py`, `signal.py`,
  `event_study.py`, `__main__.py`
- Event-study code + the §5 pre-registered gate evaluation
- If the gate passes: `factor.py`, `backtest.py`, the §7 walk-forward + holdout + additivity + raw-growth
  comparison, and the §7 factor-gate evaluation
- `backlog_factor/README.md` with both results and both gate outcomes and the §8 caveats
- Test suite per §11
- `docs/backlog-surprise-factor-results.md`
- One-line pointer added to the repo-root `README.md`

**Phase 2 (only if the §7 factor gate passes):**
- `backlog_factor/data/backlog_manual.csv` + the hand-collection log in `candidate_research.md`
- Re-run of §5–§7 on the ~200-name universe, wider-universe-vs-Phase-1 comparison added to the results doc
- A `CLAUDE.md` "Future Work" note if any Phase-3 idea (segment-level surprise, options-implied) surfaces

---

## 13. Out of scope

Trading options or power / commodities directly (the client trades equities); intraday execution
(the factor is monthly, the event study is a measurement not a trade); a single-name concentrated book
(this spec exists specifically to have breadth); any claim of statistical significance beyond the
pre-registered gates; the other three strategy directions raised alongside this one (transmission rate-base
compounders; zone-matched congestion pair; grid-data macro nowcast) — each gets its own spec.
