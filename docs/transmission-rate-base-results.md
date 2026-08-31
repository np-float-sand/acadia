# Transmission Rate-Base Compounders — Results (NEGATIVE RESULT)

Implements `docs/superpowers/specs/2026-08-31-transmission-rate-base-design.md`
via `docs/superpowers/plans/2026-08-31-transmission-rate-base.md`. Branch
`transmission-rate-base`.

## TL;DR

**Pre-registered gate FAILED on all three hard conditions.** Ranking ~34 US
regulated electric utilities by FERC transmission-rate-base growth does **not**
sort forward returns: over 2011–2025 the five signal quintiles all earned
essentially the same ~9–11 %/yr (the utility-sector return), the top-minus-bottom
long/short has a **negative** Sharpe (−0.61), rank-IC is −0.018 (t −0.35), and
there is no residual alpha (−0.6 %/yr, t −0.32). Per spec §11 this is a
documented negative result: the reusable data/signal code stays, **no strategy
module is promoted**, no README strategy line.

This is the **fourth** cross-sectional stock-selection attempt in/around the
AI-power theme to fail its pre-registered gate (after the backlog-growth tilt,
the value-chain maker-vs-contractor tilt, and the backlog-*surprise* factor).

## 1. What was built (Tasks 1–13, all unit-tested — 48 tests)

`transmission_rate_base/`:

| file | role |
|---|---|
| `data/ferc_form1.py` | PUDL FERC Form 1 fetch/cache + schedule extraction (gross tx plant accts 350–359 from sched204; accumulated depreciation for the transmission function from sched219; net/gross total plant from sched200; pro-rata fallback) |
| `data/utility_map.py` | `PARENT_FILERS` — 34 tickers → **118 FERC filer ids**, `validate()` passes (no id double-mapped) |
| `signal.py` | parent panel (filer sums + `n_filers`), primary signal (3-yr net-tx CAGR + 3-yr Δ tx-share, rank-avg), guards, cross-sectional neutralisation |
| `portfolio.py` | annual (May) rebalance, dollar+beta-neutral quintile L/S, long-only ±25 % tilt |
| `backtest.py` | monthly P&L + metrics, relative-return rank-IC |
| `additivity.py` | numpy Newey-West HAC OLS vs style controls |
| `gate.py` | the pre-registered pass/fail rule |
| `report.py` / `__main__.py` | orchestrator + CLI (`python -m transmission_rate_base [--offline] [--pre-thesis]`) |

Reusable regardless of the verdict: the FERC Form 1 transmission-plant
extraction, the filer→parent map, and the annual-rebalance sector-neutral
backtest harness.

## 2. Universe & signal (frozen, per spec)

34 US regulated electric / electric-heavy multi-utilities → 118 FERC filers.
Signal per parent-year, from FERC Form 1 via PUDL (free parquet, 1994–2025):
`net_tx` = Σ filers gross transmission plant − transmission accumulated
depreciation; `tx_share` = `net_tx` / Σ net total utility plant; raw signal =
mean within-year percentile rank of (3-yr `net_tx` CAGR, 3-yr Δ `tx_share`);
guards = ≥4 consecutive years, positive plant, |Δlog net_tx| ≤ log 2, no
filer-set change over the window; then winsor→z→regress out log(rate base)→
residual. Coverage after guards: ~19–26 names/yr (of 34), 2010–2025.

Real-data sanity (this part works): AEP net transmission plant $2.9 B (2013) →
$6.9 B (2018) → $12.1 B (2023), tx-share 10 %→18 %→23 %; CNP / PPL / PEG / AEE
rank high in their build-out years; OGE / EIX / IDA / OTTR rank consistently low.
The signal measures what it should — it just doesn't pay.

## 3. Live run (primary window 2011-05 → 2025, annual rebalance)

Prices: yfinance, full 180/180 months for all 36 symbols (single-threaded
per-ticker backfill; the threaded batch path thread-crashes at this scale).

### Gate conditions

| # | condition | result | pass? |
|---|---|---|---|
| 1 | rank-IC mean ≥ 0.03 **and** t ≥ 2.0 | mean **−0.0178**, t **−0.35** (n=15) | **FAIL** |
| 2 | Q5−Q1 ann Sharpe ≥ 0.40, monotone (≤1 inv.), ≥60 % yrs + | Sharpe **−0.61**, **2** inversions, **47 %** yrs + | **FAIL** |
| 3 | additivity α ≥ 3 %/yr, t ≥ 2.0, R² < 0.60 | α **−0.6 %/yr**, t **−0.32**, R² 0.02 | **FAIL** |
| 4 | 2003–2010 pre-thesis: not sig. negative | one-sided t **+0.58** | ok (caveat) |

### The tell — quintile forward-12-month returns

| quintile (1 = slowest tx growth … 5 = fastest) | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| mean fwd 12m return | **10.9 %** | 9.9 % | 9.2 % | 9.5 % | **10.6 %** |

All five quintiles earned ~the utility-sector return. The signal carries no
cross-sectional information; the L/S just harvests noise (ann −0.8 %, vol 8 %,
max DD −30 %).

### Long-only ±25 % tilt

Sharpe 0.45, ann 10.5 %, 80 % positive years — but this is **sector beta, not
skill**: it tracks the EW utility universe, and the L/S that isolates the tilt's
*active* bet is negative. No evidence the tilt adds value over equal weight.

## 4. Why it fails (interpretation)

- **The signal is public.** Transmission capex guidance is in every utility
  10-K and covered by every sell-side analyst; a growth measure built from it
  has little unpriced information left.
- **Heavy transmission capex is not a free lunch.** Fast rate-base growers carry
  more regulatory-lag, rate-case-outcome, financing and equity-dilution risk,
  which appears to roughly offset the rate-base growth in realised returns.
- **The additivity controls barely explain the L/S (R² 0.02):** the factor
  return is close to pure idiosyncratic noise — not a repackaged known style, and
  not an edge.

## 5. Caveats (do not rescue the result, but note)

- `data/segment_mix.csv` and `data/style_inputs.csv` are header-only in this run,
  so neutralisation is size-only and additivity uses price-based controls
  (util-beta, XLU, low-vol, momentum) only. Adding the non-regulated-revenue
  neutraliser and the fundamental controls (yield, size, value, capex intensity)
  can only push α further from significance — the raw signal already produces a
  negative IC and a negative Q5−Q1 Sharpe *before* any of that.
- Filer-set guard drops ~1/3 of name-years; a looser guard changes details, not
  the sign (spot-checked with the 5-yr window / gross-vs-net variants).
- One macro cycle, ~15 annual cross-sections, ~120–160 effectively independent
  bets. "No evidence of an edge," not "proof of none" — the same standard the
  prior three negative results were held to.
- ITC / Fortis (`FTS`), the cleanest listed pure-play transmission operator, is
  excluded (non-US parent).

## 6. Recommendation

Do not promote a strategy module. With four independent cross-sectional
stock-selection attempts in this theme now failed on pre-registered gates, the
honest options are: (a) accept "own the theme, equal-weight, risk-managed" as
the answer and build the trend-gate + vol-target overlay as a real module, or
(b) take the search to a less-picked-over area. See
`docs/handoff_2026-08-31-strategy-proposals.md` and
`docs/triage_2026-08-31-proposals-bda.md`.
