# Industrial electricity usage growth -> SPY forward-return timing (results)

**Date:** 2026-09-21. **Status: FAIL on the pre-registered primary gate. NOT ADOPTED.**
Stopped after the own-horizon replication step, per pre-registration -- **the
portfolio-overlay test (step 4 of the task, layering this on `screen +val+sleeve`) was
NOT built**, per this project's "don't spend the expensive step on something that already
failed cheap" convention (see `docs/interconnection-queue-signal-results.md` Thread 7/8).

RESEARCH-LOG open item 23. Pre-registration: `docs/electricity-usage-spy-timing-preregistration.md`
(written and locked before this regression was run). Scoping doc:
`docs/handoff-2026-09-16-industrial-electricity-market-timing.md`.
Probe: `research/electricity_usage_spy_timing/probe.py`. Data cache/outputs:
`research/electricity_usage_spy_timing/data/` (`eia_industrial_sales_us.parquet`,
`spy_monthly.parquet`, `panel.csv`, `results.json`).

---

## TL;DR

- Replicated Da, Huang & Yun (2017, *JFQA*) on a US-only, 2001-2026 sample: EIA national
  industrial electricity sales (`electricity/retail-sales`, `sectorid=IND`, `stateid=US`)
  YoY growth of a trailing-3-month average, lagged 2 months past month-end for a
  publication-timing buffer, regressed on forward 12-month SPY return.
- **Pre-registered primary gate -- fit-window (2001-2014) coefficient, |t| >= 2.0, sign
  negative -- FAILS**: `b=-1.04, t=-1.64, R^2=0.061, n=152`. Sign is correct (matches the
  paper's countercyclical direction) and the effect size is in the right ballpark
  (paper: ~-0.92 per unit; here: -1.04), but statistical power in the 14-year fit window
  falls short of the bar.
- **Full-sample (2001-2026, descriptive only, not the gate) does cross the bar**:
  `b=-1.49, t=-2.61, R^2=0.117, n=281`. This is flagged explicitly as *not* a substitute
  pass -- using the full sample as the decision rule instead of a genuine fit/holdout
  split is exactly the shortcut the pre-registration was designed to prevent, and the
  paper's own headline R^2 (~9%, 1956-2010) is comparable in magnitude to what shows up
  here only once the whole sample -- including what would have been the holdout -- is
  used to generate the t-stat.
- Per the pre-registered stopping rule, no holdout interpretation was computed as a
  decision input and no portfolio-overlay test was built. Point-in-time handling (lag
  applied, revision-vintage limitation documented) was moot for the verdict since the
  base regression didn't clear its own gate.

---

## 1. What was tested (full spec: `docs/electricity-usage-spy-timing-preregistration.md`)

- **Series:** EIA `electricity/retail-sales`, `sectorid=IND`, `stateid=US` (national
  rollup), `data=sales` (million kWh), monthly, 2001-01 through 2026-06 (306 months as
  fetched).
- **Regressor `g_smooth`:** YoY log growth of a trailing-3-month average of sales,
  `ln(mean(sales_t-2..t)) - ln(mean(sales_t-14..t-12))`.
- **Lag:** each sales month `t` treated as first knowable at month-end `t+2`; the
  regressor is indexed to that effective month-end, and forward returns are measured
  starting the following month -- mirrors `grid_equipment_basket/overlay.py`'s
  month-end-decided/next-month-held convention. This blocks temporal lookahead but not
  revision lookahead (see Caveats).
- **Target:** SPY (`auto_adjust=True` monthly close) forward 12-month return (primary,
  matches the paper's own horizon) and forward 1-month return (secondary).
- **Regression:** OLS with Newey-West/HAC standard errors, `maxlags=11` for fwd-12m
  (absorbing the 11-month overlap from monthly sampling of an annual target).
- **Split:** fit 2001-01..2014-12 (n=152 after the rolling/lag warm-up), holdout
  2015-01..2025-09 (n=129 window, not regressed since the gate failed).
- **Gate:** `|t| >= 2.0` AND `b < 0` on the fit-window fwd-12m `g_smooth` coefficient.

## 2. Results

| Spec | window | n | b | HAC t | R^2 |
|---|---|--:|--:|--:|--:|
| **fwd12 ~ g_smooth (PRIMARY GATE)** | fit 2001-14 | 152 | -1.039 | **-1.64** | 0.061 |
| fwd12 ~ g_smooth (descriptive) | full 2001-26 | 281 | -1.486 | -2.61 | 0.117 |
| fwd1 ~ g_smooth (descriptive) | full 2001-26 | 292 | -0.097 | -1.25 | 0.006 |

(Full per-run output and cached regression dict: `research/electricity_usage_spy_timing/data/results.json`.)

Sign is negative (correct direction) in every cut tried, including the fit window that
fails the magnitude bar -- this is not a sign-flip failure, it is a power failure. The
economic magnitude of `b` (-1.0 to -1.5 across cuts) is in the same order as the paper's
own reported relationship (a 1pp usage-growth rise -> ~0.92pp forward-return decline,
i.e. `b~-0.92` in the paper's own units), which is a genuine partial validation of the
mechanism even though the pre-registered statistical bar isn't cleared on the window that
was supposed to decide the verdict.

Because the gate failed, the secondary checks specified conditionally in the
pre-registration (fwd-1m fit-window gate, raw-YoY robustness variant, holdout
sign-consistency refit, out-of-sample R^2/correlation, tercile split) were **not run** --
they were explicitly gated on a pass, per §6 of the pre-registration, and running them
anyway after a fail would just be a second, unprincipled bite at the same question.

## 3. Why full-sample passes but the pre-registered fit-window doesn't

This is worth being explicit about rather than picking whichever number is more
convenient. Two honest readings coexist:

- **The full sample has more statistical power** (n=281 vs n=152) simply because it's
  longer, and a longer window that happens to include 2015-2026 (the era originally
  earmarked as holdout) crosses the bar. This is expected -- more data usually raises a
  t-stat when the underlying relationship is real but modest, exactly as here.
- **But using the full sample as the decision rule is the shortcut the fit/holdout split
  exists to prevent.** If the paper's own R^2 (~9%, 1956-2010, ~55 years, US/Japan/UK) is
  the honest benchmark, replicating with a 14-year US-only fit window was always going to
  be underpowered relative to the original study -- and the pre-registration's whole
  point was to commit to that harder, honest test *before* seeing that the easier
  full-sample cut would have passed. Reporting the full-sample number as context (not as
  a substitute pass) is the correct way to surface this tension without either hiding it
  or quietly using it to flip the verdict.

## 4. Mechanism / caveats

- **Point-in-time data — moot for this verdict, but still a real limitation for any
  future revival.** The EIA `electricity/retail-sales` v2 API exposes no
  revision/vintage facet (confirmed live against the route's own metadata). A
  Wayback-based vintage-reconstruction attempt (the method that worked for the STEO
  spark-spread panel, `docs/steo-spark-spread-results.md`) was tried against EIA's
  Electric Power Monthly fixed-URL table and blocked by Wayback CDX rate-limiting
  (HTTP 429) -- not pursued further given the gate failure makes it moot for now. The
  regression above uses latest-revised values with a 2-month publication-timing lag
  (blocks temporal lookahead, not revision lookahead) -- see pre-registration §2 for the
  full treatment. If this signal is ever revisited, real vintage data is the first thing
  to chase before trusting a passing number.
- **Sample-size mismatch vs. the original paper.** 25 years of US-only monthly data
  (154-month fit window) is a fraction of the paper's 55-year, 3-country panel. The
  underlying mechanism (usage growth as a countercyclical risk-premium proxy) may well be
  real -- the sign and rough magnitude match across every cut here -- but this project's
  US-only, post-2001 sample may simply not carry enough independent annual observations
  (roughly 12-13 non-overlapping 12-month periods in the fit window) to clear a strict
  |t|>=2.0 bar even if the true relationship matches the paper's.
- **2008-09 GFC sits inside the fit window** and is a large, well-known outlier for any
  regression against forward market returns; no attempt was made to exclude or
  special-case it (that would be exactly the kind of post-hoc slicing this project's
  discipline forbids), but it is worth naming as a plausible reason the fit-window
  estimate is noisier than the full-sample one.
- **This was never proposed as a standalone strategy.** Per the handoff, the only
  application under consideration was a third overlay layer on the shipped
  `electrification_strategy` book. That test was never built, because the pre-registered
  discipline is to not spend the expensive combination-test step on a signal that failed
  its own cheap replication check first.

## 5. Disposition

**FAIL on the pre-registered primary gate. NOT ADOPTED.** Own-horizon replication does
not clear this project's default |t|>=2.0 bar on the pre-registered fit/holdout split,
despite a correctly-signed, right-order-of-magnitude coefficient and a full-sample cut
that would have passed had full-sample been used as the (weaker) decision rule. No
portfolio-overlay test was built. If revisited, the two prerequisites are (a) a genuine
point-in-time vintage panel (STEO-style Wayback reconstruction, retried past the
rate-limit) and (b) more history or a panel extension (e.g. the paper's Japan/UK legs, or
EIA's pre-2001 annual SEDS data at annual frequency) to give the fit window real power
before a second attempt is worth the effort.

## References

- Da, Z., Huang, D., & Yun, H. (2017). "Industrial Electricity Usage and Stock Returns."
  *JFQA* 52(1), 37-69.
- `docs/electricity-usage-spy-timing-preregistration.md` -- full locked spec.
- `docs/handoff-2026-09-16-industrial-electricity-market-timing.md` -- original scoping.
- `docs/steo-spark-spread-results.md` -- Wayback vintage-reconstruction precedent
  (attempted here, blocked by rate-limiting).
- `docs/interconnection-queue-signal-results.md` Thread 7/8 -- the "cheap test before
  expensive test" precedent this writeup follows in stopping before the overlay build.
