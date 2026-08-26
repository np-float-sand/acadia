# Peer-Group (Basket-vs-Basket) Portfolio Construction — Design

**Status:** Implemented (`--peer-group` CLI flag, `PEER_GROUP_CONSTRUCTION` config, off by
default). Validated on backtest — cuts volatility/drawdown materially (~26%/~29%) but does not
improve Sharpe or beat XLU, and the predicted sector-beta-cancellation effect did not hold at
the whole-strategy level despite working correctly within each peer group. Full results:
`docs/compact_2026-08-19-peer-group-construction-results.md`.

## Background

Full diagnosis: `docs/handoff_2026-08-15-pairs-basket-construction.md`. Summary: the current
portfolio construction (`grid_resilience/portfolio/construction.py::build_weights()`) ranks the
whole 18-name universe and takes top-3 long / bottom-5 short. Backtest analysis showed:

- The long book is highly correlated with XLU (+0.76) and underperforms buy-and-hold XLU — it's
  riding sector beta, not adding stock-selection alpha.
- The short book (Sharpe -0.615) performs no better than a naive equal-weight short of the entire
  18-name universe (Sharpe -0.557) — most of its loss is uncompensated short-beta drag, not bad
  picks.

Diagnosis: the problem is architectural (portfolio construction exposes the whole book to sector
beta), not a signal-quality problem. Splitting the universe into peer groups with similar
business-model/beta exposure, and building long+short baskets *within* each group, cancels most of
the sector-beta exposure while preserving diversification within each leg.

Business-model groups (`grid_resilience/data/utility_node_map.py`, `business_model` field):

| Group | Tickers | Count |
|---|---|---|
| merchant | VST, NRG | 2 |
| mixed | CNP, DTE, ETR, EXC, PEG | 5 |
| regulated | AEE, AEP, CMS, D, EIX, EVRG, FE, PCG, PPL, WEC, XEL | 11 |

## Decisions

1. **Merchant group (VST, NRG) is left untouched.** It's too small (2 names) for a diversified
   basket on either side regardless of grouping. Stress-beta already works reasonably for
   generators specifically — that's why hard-switch already routes them to a different signal path
   than regulated names. The redesign applies only to regulated+mixed (16 names), where the
   diagnosed sector-beta problem was actually found.
2. **Book sizes are proportional to group size**, computed from the 16-name regulated+mixed
   universe and preserving the current global 3-long/5-short ratio (~8-name total book,
   long:short ≈ 3:5):

   | Group | Names | n_long | n_short |
   |---|---|---|---|
   | regulated | 11 | 2 | 3 |
   | mixed | 5 | 1 | 2 |

3. **Dollar-neutrality is enforced independently per sleeve** (merchant, regulated, mixed), not
   just globally. Each sleeve's longs sum to +its capital share, shorts to -its capital share. This
   maximizes beta-cancellation within each group, which is the point of the redesign, and keeps
   each sleeve's P&L independently interpretable.
4. **Capital is split across the three sleeves proportional to name count** across all 18 tickers,
   so the total book stays at the current ~100% gross (0.5 long / -0.5 short):

   | Sleeve | Names | Long capital | Short capital |
   |---|---|---|---|
   | merchant | 2 | 0.5 × 2/18 ≈ 0.0556 | -0.0556 |
   | mixed | 5 | 0.5 × 5/18 ≈ 0.1389 | -0.1389 |
   | regulated | 11 | 0.5 × 11/18 ≈ 0.3056 | -0.3056 |

5. **Rebalance cadence is unchanged.** `build_rolling_weights()` still rebalances at month-end;
   the redesign changes how weights are computed at each rebalance date (per-group basket
   construction instead of universe-wide ranking), not how often rebalancing happens.

## Architecture

Introduce `build_grouped_weights()` in `grid_resilience/portfolio/construction.py` as a new
orchestration entry point, sitting alongside (not replacing) the existing `build_weights()`:

- `build_weights()` stays as the single-group primitive — it already takes `n_long`/`n_short` and
  produces a dollar-neutral vector scaled to ±0.5. It is reused unmodified inside each sleeve.
- `build_grouped_weights(factor_scores)`:
  1. Splits `factor_scores` into three sub-series by `business_model` (merchant / mixed /
     regulated), using `grid_resilience.data.utility_node_map`.
  2. Calls `build_weights()` per sleeve with that sleeve's `n_long`/`n_short` from the table above.
  3. Rescales each sleeve's ±0.5-normalized output to that sleeve's actual capital share (e.g.
     regulated's ±0.5 output × 0.3056/0.5).
  4. Concatenates the three rescaled weight vectors into one `pd.Series` covering all 18 tickers
     (0.0 for any ticker not selected in its sleeve).
- `build_rolling_weights()` gains a `grouped: bool = False` parameter (default off, matching how
  `BUSINESS_MODEL_ARCH` defaults to `None`/original behavior). When `True`, it calls
  `build_grouped_weights()` instead of `build_weights()` at each rebalance date. `n_long`/`n_short`
  parameters are ignored in grouped mode (sizes come from the per-group table); `xlu_hedge` applies
  only to the merchant sleeve's short side, since XLU-hedging a 2-name generator sleeve is the one
  case where it was never invalidated by the grid-search finding (that finding concerned XLU
  hedging the *whole* short book, not this narrow sleeve) — default `xlu_hedge=False` preserves
  current merchant behavior exactly.

## Config changes

`grid_resilience/config.py` additions:

- `PEER_GROUP_CONSTRUCTION: bool = False` — feature flag; when `False`, `build_rolling_weights()`
  behaves exactly as today. Mirrors the existing `BUSINESS_MODEL_ARCH` pattern of off-by-default
  until validated in backtesting.
- `PEER_GROUP_BOOK_SIZES = {"regulated": (2, 3), "mixed": (1, 2)}` — `(n_long, n_short)` per group,
  from the table above. Merchant is intentionally absent from this table: it's called through
  `build_weights()` with the existing global `PORTFOLIO_LONG_N`/`PORTFOLIO_SHORT_N` defaults, but
  since the merchant sub-series only has 2 tickers and `3 + 5 > 2`, `build_weights()`'s existing
  `len(scores) < n_long + n_short` shrink fallback (`construction.py:42-44`) always kicks in and
  reduces it to 1 long / 1 short — both VST and NRG are always included, one on each side. This is
  the same "effectively a pair regardless" behavior the merchant group already has today, so no new
  constant is needed for it.
- A CLI flag `--peer-group` / `--no-peer-group` (mirroring the existing `--arch`, `--icr` pattern
  documented in README) to toggle `PEER_GROUP_CONSTRUCTION` per backtest run.

## Data flow

```
factor_scores (pd.Series, 18 tickers)
        │
        ▼
build_grouped_weights()
        │
        ├─ merchant sub-series (2 tickers)  → build_weights() [shrinks to 1L/1S] → scale to ±0.0556
        ├─ mixed sub-series (5 tickers)     → build_weights(n_long=1, n_short=2) → scale to ±0.1389
        └─ regulated sub-series (11 tickers)→ build_weights(n_long=2, n_short=3) → scale to ±0.3056
        │
        ▼
pd.concat → single weight Series (18 tickers, sums to 0.5 / -0.5)
```

`build_rolling_weights(grouped=True)` calls this once per rebalance date, same as today's loop
structure — no change to `weights_to_matrix()` or downstream backtest P&L code, since both produce
the same `[date, ticker, weight]` shape.

## Error handling / edge cases

- A group with zero valid (non-NaN) scores on a given rebalance date: skip that sleeve for that
  date (contributes 0 weight to all its tickers), same fallback behavior `build_weights()` already
  has for empty input.
- A group where available scored names are fewer than its configured `n_long + n_short` (e.g. only
  3 of 5 mixed tickers have data that day): reuse `build_weights()`'s existing shrink-to-half
  fallback logic unmodified — it already handles `len(scores) < n_long + n_short`.
- Merchant sleeve's `xlu_hedge` and `n_long`/`n_short` behavior is completely unchanged from today
  — no new edge cases introduced there.

## Testing

- Unit tests for `build_grouped_weights()`: verify (a) per-sleeve dollar-neutrality, (b) total book
  sums to exactly 0.5/-0.5, (c) correct tickers selected per group given known scores, (d) capital
  scaling matches the table above.
- Unit test for `build_rolling_weights(grouped=True)`: verify it produces the same `[date, ticker,
  weight]` shape as the ungrouped path and that `weights_to_matrix()` consumes it without changes.
- Backtest comparison: run the full backtest with `PEER_GROUP_CONSTRUCTION=True` vs `False`,
  compare Sharpe/return/correlation-to-XLU for the combined book and for regulated+mixed alone,
  against the diagnosis numbers in the handoff (long book correlation to XLU, short book Sharpe vs
  naive short).
- Regression: existing tests for `build_weights()` and `build_rolling_weights()` (default
  `grouped=False`) must pass unchanged, confirming the feature flag truly no-ops when off.

## Out of scope

- RT/DA LMP spread signal and outage-reserve-margin signal (separate handoffs,
  `docs/handoff_2026-08-15-rt-da-spread-signal.md` and
  `docs/handoff_2026-08-15-outage-reserve-margin-signal.md`) — independently scoped, may plug into
  this construction mechanism later but are not part of this design.
- ICR Phase 2 / GSI congestion Phase 2 work tracked in `CLAUDE.md` — unrelated to this redesign.
- Re-validating XLU hedge on the *whole* short book — grid search already found this destroys
  alpha (see `CLAUDE.md`); this design does not revisit that finding, only the narrower merchant-
  sleeve XLU-hedge question noted above.
