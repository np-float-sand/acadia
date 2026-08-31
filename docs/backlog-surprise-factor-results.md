# Backlog Growth Surprise — Event Study Results (Phase 1)

Spec: `docs/superpowers/specs/2026-08-31-backlog-surprise-factor-design.md`
Plan: `docs/superpowers/plans/2026-08-31-backlog-surprise-factor-event-study.md`
Run date: 2026-08-31. RPO: SEC XBRL `RevenueRemainingPerformanceObligation` (`companyconcept`).
Prices: yfinance adjusted close. Window: 2019-01-01 → 2026-07-31.

## Result: the pre-registered gate FAILS. Stop here — no factor, no Phase 2.

## Phase-1 universe

**109 names**, curated from the **207** in-scope XBRL-RPO filers discovered via the SEC `frames` API
(construction / machinery / electrical / transportation-equipment SIC prefixes), dropping medical & lab
names whose RPO is service-contract deferred revenue, pre-revenue / SPAC hype names, autos, and
software-heavy "industrial tech". Every keep/drop decision is in `backlog_factor/candidate_research.md`.
By industry group: machinery 46 · aerospace_defense 21 · engineering_construction 20 ·
electrical_equipment 11 · semiconductor_equipment 8 · building_products 3.

**Breadth achieved.** 103 of 109 names have an RPO series; after the ≥6-quarter + clean-span guards,
**91 names contribute 1,847 surprise events → 1,778 scored (ticker, filing-date) pairs**. Events per
year: 75 (2019) → ~290 (2023–25) as names accumulate the required history. This is the cross-sectional
breadth the design set out to get; the negative result below is not a small-sample artifact.

## Method

Abnormal CAR = daily simple return minus the name's GICS-industry-group ETF (XLI / PAVE / ITA / XHB /
SOXX), cumulative-summed over `[0,+5/+21/+42/+63]` trading days measured from the close of day 0
(day 0 = first trading day strictly after the SEC filing date). `surprise = g − trailing-4Q-mean(g)`,
`g = log(RPO_t / RPO_{t-1})`. Surprise sorted into cross-sectional quintiles within each calendar month,
then pooled. Q5−Q1 significance = month-clustered t.

## Quintile abnormal CAR (%)

| surprise quintile | 5d | 21d | 42d | 63d |
|---|---|---|---|---|
| Q1 (most negative) | −0.47 | +0.87 | +2.14 | +3.33 |
| Q2 | +0.68 | +1.44 | +2.79 | +3.32 |
| Q3 | +0.45 | +0.59 | +0.52 | +2.24 |
| Q4 | +0.17 | +0.60 | +1.73 | +1.80 |
| Q5 (most positive) | +0.80 | +1.31 | +1.86 | +2.29 |
| **Q5 − Q1** | **+1.27** | **+0.44** | **−0.27** | **−1.03** |

(All five quintiles have *positive* abnormal CAR at every horizon — these industrials broadly beat their
sector ETFs over 2019–2026. Only the Q5−Q1 *spread* is informative, and it is small and reverses.)

## Pre-registered gate (spec §5)

Peak window = 5 days (largest |Q5−Q1|).

| Check | Value | Pass? |
|---|---|---|
| Q5−Q1 abnormal CAR at peak window > 0 **and monotone** across quintiles | +1.27% ; **monotone = FALSE** (Q2 > Q3 > Q4; Q1 the only clearly-negative bucket) | **FAIL** |
| month-clustered t > 2 | 2.12 | pass (marginal) |
| both sub-period halves same sign, clustered t > 1 | h1 +/2.20 ; h2 **+/1.01** | pass (h2 marginal) |
| right sign in ≥ 3 of 5 major industry groups | 4/5 (electrical_equipment −0.94%, semiconductor_equipment −0.72% wrong-signed; machinery +2.34% t=2.82 carries it) | pass |

**GATE: FAIL** — fails the monotonicity requirement outright; the clustered-t and sub-period-2 checks
only marginally clear.

## What the drift actually looks like

- **Short-lived and reversing.** The Q5−Q1 spread is +1.27% at one week, decays to +0.44% at one month,
  and turns **negative** at two and three months (−0.27%, −1.03%). The thesis was a *multi-week drift
  toward the surprise*; the data show a ~1-week bump that unwinds. This is more consistent with a brief
  attention / liquidity effect on the lowest-surprise names than with information being slowly
  incorporated.
- **Not a rank signal.** At 5 days the quintile means are [−0.47, +0.68, +0.45, +0.17, +0.80]%. It is
  "Q1 underperforms for a week," not "higher surprise → higher return." Q2 outperforms Q5 at the 21d and
  42d horizons.
- **Direction carries no information.** Names with a *negative* surprise drift up *more* over 63 days
  (+3.04%) than names with a *positive* surprise (+2.15%).
- **One-sector effect.** Machinery (+2.34% at 5d, t 2.82) is the only group with a clean positive Q5−Q1;
  electrical equipment and semi-cap equipment are negative.

## Caveats (spec §8)

- ~7.5-year structured span (2019–2026), one macro cycle. Breadth is cross-sectional (1,778 events); the
  time-series is short and the negative result is for this cycle.
- 6 non-positive RPO rows (GEOS 1, RAIL 2, VMI 3) were NaN'd before the log (data-hygiene guard added
  during the run); they do not affect the outcome.
- RPO definition drift: the `|g| > log(3)` break screen NaN'd a name's surprise across each detected
  level jump and the following 4 quarters.
- Survivorship: the 109-name universe was assembled in 2026 from *current* RPO filers; E&C firms that
  delisted and defense names that were acquired are absent.
- Look-ahead: `expected_g` is a fixed trailing mean; the peak window was chosen here (5d), which would
  need a reserved holdout — moot, since the gate fails before that matters.

## Decision (spec §5)

**Stop.** The pre-registered event-study gate is not met. Per spec §5 the deliverable is
`backlog_factor/candidate_research.md`, the Phase-1 event-study code (`backlog_factor/`), and this
written negative result. **No factor is built and no Phase-2 hand-collection is done.**

The narrow finding worth keeping: across a genuinely broad cross-section of order-driven industrials,
a quarter's backlog/RPO growth *surprise* does **not** produce a monotone, multi-week drift in
industry-adjusted returns. There is a weak, machinery-concentrated, ~1-week positive reaction that
reverses over the following two months — not tradeable as a monthly factor after costs.
