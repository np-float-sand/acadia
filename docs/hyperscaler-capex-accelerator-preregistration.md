# Pre-registration: hyperscaler capex-reacceleration as an entry accelerator

**Date:** 2026-09-21. Written and committed to disk BEFORE the probe script is run.
Open item #16 in `docs/RESEARCH-LOG.md` §4.

## 1. Context

`grid_equipment_basket/capex_signal.py` already built and cached (`grid_equipment_basket/data/cache/bigfour_capex.parquet`)
a point-in-time quarterly series of aggregate Big-4 hyperscaler (MSFT/GOOGL/AMZN/META) capex,
with `yoy` (trailing-4-quarter growth) and `decel2` (2-quarter change in `yoy`) columns, each
row stamped with a `known_date` (quarter end + 50 days, i.e. the date the last of the four
10-Qs makes the print public). That module used the data **one-directionally as an exit
de-risk trigger**: cut exposure to 0.5x when `yoy < 0.15` and `decel2 < 0`. Per
`docs/RESEARCH-LOG.md` line 137, that use **FAILED** -- "fires ~a year late; lagging
confirmation," consistent with every other demand-data class tried as a *timing* signal in
this research line (they all lag a 2-4-year hyperscaler capex-guidance-to-spend cycle too
much to call a turn).

This probe tests the **opposite framing**: the same growth data as a long-only **entry
accelerator** -- size UP when capex growth is reaccelerating, on the theory that lag is far
more tolerable when you're confirming a trend you're already in (adding exposure to a book
that's already long the theme) than when you're trying to call a top or bottom.

## 2. Re-acceleration rule (stated before running, no post-hoc tuning)

Reuses the exact same `yoy` field already computed by `capex_signal.fetch_bigfour_capex()` --
no new capex computation, no new free parameter search over window lengths or thresholds.

**Rule:** at each quarter, `yoy` is "reaccelerating" iff it exceeds its own trailing
**4-quarter average of the prior 4 quarters** (i.e. `yoy > yoy.shift(1).rolling(4).mean()`,
excluding the current quarter from its own baseline to avoid self-reference). This is a
plain moving-average crossover -- the same style of rule already used elsewhere in this
codebase for regime/trend gating (`overlay.trend_gate`'s price-vs-trailing-MA), applied here
to the growth rate itself rather than to price. The 4-quarter window matches the window
`capex_signal.py` already uses to build `yoy` itself (trailing-4-quarter), so no new window
parameter is introduced either.

This operationalizes both re-acceleration framings named in the RESEARCH-LOG open item
("accelerating off a trough" and "crossing above its own trailing average") as a single rule:
a trough-to-recovery move mechanically crosses back above a depressed trailing average once
growth turns, and a rule based purely on the trailing average requires no separate trough
detector.

**Point-in-time discipline:** identical to `capex_signal.py` -- the multiplier for a given
quarter only becomes live on that quarter's `known_date` (quarter end + 50 days) and holds
(ffill) until the next known print. No change to the existing lag convention.

## 3. Multiplier construction

Long-only, one-way-up (mirror image of the existing exit trigger's one-way-down):

```
capex_accel_mult = REGIME_HI (1.25)   if yoy > trailing_avg_yoy   (known, point-in-time)
                  = 1.0                otherwise, and during warm-up (pre-2019Q1, no 4q history)
```

`1.25` reuses `grid_equipment_basket.config.REGIME_HI`, the existing discrete-regime "up"
multiplier already calibrated and used elsewhere in this codebase (the grid-congestion Layer-2
overlay) -- not a new number chosen to fit this test. It sits inside the
`electrification_strategy.config.VOL_MAX_LEVERAGE` ceiling (1.5x) with headroom for the
vol-target scalar to also contribute.

## 4. Combination with the existing overlay stack

Exactly the pattern in `docs/steo-spark-spread-results.md` §10 (the direct template for this
test), applied to the actual shipped `screen +val+sleeve` book, core equity leg only --
15% GLD/IEF sleeve and the 27-name screen selection are both untouched:

```
exposure = clip(capex_accel_mult x valuation_mult x vol_scalar, upper=VOL_MAX_LEVERAGE)
cell_r = (1 - HEDGE_SLEEVE_WEIGHT) x (exposure x core_r + (1-exposure) x rf/ANN)
       + HEDGE_SLEEVE_WEIGHT x sleeve_r
```

where `valuation_mult` and `vol_scalar` are the unmodified functions already used by
`electrification_strategy.backtest.compose()`.

## 5. Sample and gate

**Sample:** full 2017-06-01 (`config.START_DEFAULT`) -> today (2026-09-21), matching every
other full-sample test in this log. Capex data starts 2018Q1 (known 2018-05-20); before the
first known print and before the 4-quarter warm-up completes (~2019-05-20, when the first
`yoy` trailing average exists), `capex_accel_mult` is neutral at 1.0 -- same fallback
convention as the existing `capex_derisk_multiplier`.

**Sanity check required before trusting results:** reproduce the unmodified `screen
+val+sleeve` baseline (Sharpe 0.69 full-sample per RESEARCH-LOG §1) with this harness before
reporting the variant.

**Pre-registered two-part gate** (same AND-gate used throughout this log for every overlay
test):
- **(a)** Full-sample Sharpe of the capex-accelerated cell > the unmodified `screen
  +val+sleeve` cell.
- **(b)** Full-sample max drawdown does not get worse (variant MaxDD >= baseline MaxDD, i.e.
  no deeper drawdown).

Both must hold for **PASS**. Either failing is **FAIL / NOT ADOPTED**, regardless of the
other. No threshold or window re-picking after seeing the result.

## 6. Mechanism caveat, stated before running

- The Big-4 print is the demand side of a book that is priced on equipment-*supplier*
  earnings, not hyperscaler capex directly -- the transmission mechanism already ran through
  once in the exit-trigger test (lagged confirmation, not leading). The entry-accelerator
  framing changes what lag *costs* (missed early upside vs. false-late exit) but does not
  change the lag itself; if the accelerator only turns on ~a year after a reacceleration has
  already re-rated the names (the same "fires ~a year late" pattern), it may still add little
  because most of the acceleration-driven re-rating has already happened by the time it fires
  -- the entry-accelerator framing predicts this cost should be smaller than the exit-trigger's
  cost, not zero.
- `yoy > trailing_avg` is true 58% of quarters in the cached data (2018Q1-2026Q2,
  post-warmup) -- a real, non-degenerate split, not a rare-event trigger, so this is closer to
  "mostly levered up with occasional pullbacks to neutral" than a sparse conviction signal.
  That raises the bar: since capex growth has structurally been positive for most of the
  sample (the AI buildout is the dominant trend for 2020-2026), a coarse yoy-vs-own-average
  crossover may end up close to "usually 1.25x," which is a leverage/turnover question as much
  as a timing one.
- Only 34 quarterly observations total (2018Q1-2026Q2) back this signal; any full-sample
  result is driven by a handful of regime transitions (2019-20 recovery, 2021-23 deceleration,
  2023-26 AI supercycle), not a statistically rich sample.

## 7. What will NOT be done

- No touching `electrification_strategy/` or `grid_equipment_basket/` production files.
- No editing `docs/RESEARCH-LOG.md` (a separate consolidation pass handles that).
- No threshold grid search after seeing the Sharpe/MaxDD numbers -- the rule above (§2-3) is
  final before the probe script runs.
