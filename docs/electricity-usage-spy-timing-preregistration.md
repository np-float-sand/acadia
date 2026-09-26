# Industrial electricity usage growth -> SPY forward-return timing (pre-registration)

**Date:** 2026-09-21. **Status:** PRE-REGISTERED, NOT YET RUN.
Research-log open item 23. Picks up `docs/handoff-2026-09-16-industrial-electricity-market-timing.md`
(SCOPED, NOT RUN). Replicates Da, Huang & Yun (2017, *JFQA* 52(1)) on a US-only,
2001-2026 sample before any application to this project's own book, per that handoff's
"own-horizon replication first" instruction.

This is a **broad-market (SPY) timing signal**, not a cross-sectional stock-picking factor.
Locked in writing below, before the regression is run, per this project's standing
discipline (see `docs/steo-spark-spread-results.md`, `docs/interconnection-queue-signal-results.md`
for house style/tone).

## 1. Data series

- **Source:** EIA API v2, route `electricity/retail-sales`, endpoint
  `/v2/electricity/retail-sales/data/`.
- **Facets:** `sectorid=IND` (industrial sector), `stateid=US` (U.S. Total, a native national
  rollup -- no state-summing needed, confirmed live against the route's own metadata endpoint).
- **Field:** `data[0]=sales` (million kilowatt-hours sold to ultimate customers), monthly
  frequency. This is the sales/volume field of the same `electricity/retail-sales` endpoint
  this project already uses for *price* (`research/electricity_earnings_exposure/probe_b.py::eia_state_industrial_price`)
  -- same access pattern, `EIA_API_KEY` from `.env`, sibling field on the same route.
- **History:** confirmed live: `startPeriod=2001-01`, `endPeriod` tracking ~2-3 months behind
  the current date (306 months as of the 2026-09-21 scoping check, through 2026-06). This is
  **25 years of US monthly data**, not the paper's 1956-2010 US/Japan/UK panel -- flagged
  up front as a real power reduction vs. the original study, not hidden until the results doc.
- **US-only**, not the paper's 3-country panel: this project trades only US equities, and every
  prior EIA fetcher in this repo (probe_b, the STEO/PJM work) has stuck to US-only for the same
  reason. Cheaper, and the only actionable leg for this book.

## 2. Point-in-time handling (the flagged open question from the handoff)

Checked live and confirmed **before** designing around it, not assumed:

- **No vintage/revision facet exists on this API route.** The route's own metadata endpoint
  (`/v2/electricity/retail-sales/`) lists exactly `stateid` and `sectorid` as facets and no
  revision/release-version dimension -- the API serves only the current (latest-revised) value
  for each period, same limitation flagged as unconfirmed in the handoff.
- **Attempted the STEO precedent** (`docs/steo-spark-spread-results.md` sec 1: Wayback-crawled
  vintages of a fixed, overwritten-every-month EIA URL) against EIA's Electric Power Monthly
  fixed-URL table (`epm_table_grapher.php?t=epmt_5_1`, `xls/table_5_01.xlsx`) as a possible
  point-in-time source. **Blocked by Wayback CDX rate-limiting (HTTP 429) on this pass** --
  not pursued further. Per this project's "don't spend the expensive step on something that
  hasn't cleared the cheap step" convention, a full vintage-reconstruction effort is only
  worth investing in *after* the base (revised-data) regression clears its own gate below,
  not before.
- **Resulting treatment -- two separate mechanisms, applied together, each addressing a
  different half of the lookahead problem:**
  1. **Values are latest-revised, not point-in-time.** This is a genuine, unresolved
     limitation -- the true real-time revision size for this specific series is unmeasured in
     this session. It is very likely the original paper (1956-2010, published 2017) also used
     final/revised data, standard for an academic study of that era, per the handoff's own
     note -- so this is not a new problem this replication introduces, but it is not fixed
     either. **Reported as a caveat in the results doc, not hidden.**
  2. **Timing is lag-adjusted to prevent temporal (not revision) lookahead.** EIA's Electric
     Power Monthly is released roughly 65-90 days after month-end (confirmed indirectly: the
     live API's `endPeriod` on 2026-09-21 was 2026-06, a ~2.5-3 month lag from query date).
     Each sales month `t`'s figure is treated as **first knowable at the month-end of `t+2`**
     (i.e., usable for a position entered at the start of month `t+3`) -- a conservative
     2-full-month-after-month-end lag, mirroring the `_month_hold` "decided at month-end,
     applied next month" convention already used by `grid_equipment_basket/overlay.py` and
     the ~2-month 10-Q lag convention in `research/electricity_earnings_exposure/probe_b.py`.
  This eliminates the "trading on a number before it existed" failure mode. It does **not**
  eliminate revision lookahead (using a since-revised, more-accurate figure). Both are stated
  plainly in the results doc; neither is allowed to be waved away as solved.

## 3. Transform

- **Primary regressor** `g_t`: year-over-year log growth of a trailing-3-month average,
  `g_t = ln(mean(sales_t-2..t)) - ln(mean(sales_t-14..t-12))`. Smooths single-month survey
  noise (the raw series has a strong seasonal swing, e.g. Jun-2020 78.4k vs Apr-2020 72.8k
  million kWh) and nets out seasonality via the YoY comparison without a seasonal-adjustment
  model, in the spirit of the paper's "usage growth" framing.
- **Secondary/robustness regressor**: unsmoothed raw YoY log growth,
  `ln(sales_t) - ln(sales_t-12)`. Reported alongside, not separately gated.
- All regressors indexed to their **lag-adjusted effective month-end** (sec 2), not their raw
  sales-period month.

## 4. Target

- **SPY** (`auto_adjust=True` monthly close, same convention as
  `research/electricity_earnings_exposure/probe_b.py::monthly_returns`) as the tradeable
  US broad-market proxy.
- **Primary horizon:** forward 12-month SPY return from the lag-adjusted effective month-end
  -- matches the paper's own headline horizon, the only horizon this replication is trying to
  reproduce before trusting the mechanism at all.
- **Secondary horizon:** forward 1-month SPY return -- the horizon this project's own overlay
  stack (`electrification_strategy/backtest.py::compose`, monthly-effective exposure scalars)
  can actually consume. Reported, not gated as the primary pass/fail (see sec 6).

## 5. Regression spec

OLS, `fwd_return_t = a + b * g_t + e_t`, Newey-West (HAC) standard errors --
`maxlags=11` for the fwd-12m regression (absorbs the 11-month observation overlap from
monthly sampling of an annual-horizon target), `maxlags=1` for the fwd-1m regression.
Report `b`, HAC t-stat, R^2, n, for both the smoothed and raw-YoY regressor.

## 6. Fit/holdout split and gate

25 years of monthly history (2001-2026) supports a genuine split -- used, not full-sample:

- **Fit window:** 2001-01 through 2014-12 (effective-month basis). ~14 years.
- **Holdout window:** 2015-01 through the most recent month with a computable fwd-12m target
  (bounded by today, 2026-09-21, minus 12 months minus the publication lag -> roughly
  2025-mid). ~10-11 years. This split also happens to separate the low-rate/shale-boom era
  from the 2015+/AI-capex-buildout era the handoff flagged as a regime-stability concern.

**Primary gate (checked and reported BEFORE any holdout interpretation, per this project's
default):** on the **fit-window, fwd-12m, smoothed-regressor** spec --
**|HAC t-stat on `b`| >= 2.0 AND correct sign** (negative -- higher usage growth predicting
*lower* forward market return, the paper's countercyclical-risk-premium direction).

- **If this fails: FAIL. Report and stop.** No holdout interpretation is substituted for a
  passing verdict, no portfolio-overlay build is attempted (per the task's "don't spend the
  expensive step on something that already failed cheap" instruction, see
  `docs/interconnection-queue-signal-results.md` Thread 7/8 for the precedent).
- **If this passes:** report, descriptively (not as a second hard gate, since the task only
  specifies one primary-coefficient gate): holdout-window sign-consistency and out-of-sample
  correlation between fit-estimated predictions and holdout realized returns; the fwd-1m
  fit-window regression (own gate check, |t|>=2.0, since this is the horizon an actual
  monthly overlay would need); the raw-YoY-regressor robustness check; and a purely
  descriptive tercile split (top vs. bottom third of `g_t` months) as a light non-linearity
  check -- explicitly not a second bite at a linear-spec failure, and not itself gated, given
  this project's repeated experience (Henry Hub GBM, narrative-mispricing Approach C) that
  non-linear specs overfit short windows.
- **Only if the primary gate passes** does step 4 of the task (portfolio-overlay test on
  `screen +val+sleeve`, stacked with L1 trend-gate / L2 regime) get built at all.

## 7. What "pass" would mean, and what it would not

A pass here says the *broad-market* mechanism replicates on a US-only, 2001-2026 sample --
it says nothing yet about whether it helps *this project's specific book* when layered as a
third overlay; that is an entirely separate, conditional test (see portfolio role in the
handoff doc), not assumed to follow automatically from a clean regression.

## References

- Da, Z., Huang, D., & Yun, H. (2017). "Industrial Electricity Usage and Stock Returns."
  *JFQA* 52(1), 37-69.
- `docs/handoff-2026-09-16-industrial-electricity-market-timing.md` -- scoping doc this
  pre-registration operationalizes.
- `docs/steo-spark-spread-results.md` sec 1 -- the Wayback vintage-reconstruction precedent
  attempted (and blocked) for point-in-time handling here.
- `research/electricity_earnings_exposure/probe_b.py` -- EIA `retail-sales` fetch pattern and
  ~2-month reporting-lag convention this design reuses.
- `grid_equipment_basket/overlay.py` -- month-end-decided/next-month-held overlay convention
  this design's lag treatment mirrors.
