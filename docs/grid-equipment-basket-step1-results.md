# Grid Equipment Suppliers Basket — Step 1 Results

Spec: docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md
Plan: docs/superpowers/plans/2026-08-28-grid-equipment-basket.md
Run date: 2026-08-29. Prices: yfinance auto-adjusted (total-return) daily closes, monthly-parquet cache.

## Basket

Universe (9 names, all verified on the latest-10-K business description only — never on
returns; see `grid_equipment_basket/candidate_research.md`):

ETN (Eaton), HUBB (Hubbell), GEV (GE Vernova), VRT (Vertiv), PWR (Quanta Services),
MYRG (MYR Group), NVT (nVent Electric), FLNC (Fluence Energy), PRIM (Primoris Services).

Construction: equal-weight across the names that have price history on each rebalance date;
quarterly rebalance ~6 weeks (42 calendar days) after each calendar quarter-end
(~mid-Feb / mid-May / mid-Aug / mid-Nov); 25% single-name cap (never binds at equal weight
with 8-9 names); long-only, fully invested, no leverage; total-return basis; **simple-return
convention** (`pct_change`, `(1+r).prod()`), unlike `grid_resilience` which uses log returns.
Weights are set on the rebalance date and drift with relative price performance until the next
one. GEV has vendor price history only from 2024-03-27 and therefore joins the basket at the
2024-05-13 rebalance — the equal weight steps from 1/8 to 1/9 there. That is the designed
behaviour (spec §4), not a gap.

Metrics conventions (match `grid_resilience/portfolio/backtest.py`): risk-free 4% annual,
252 annualization, Sharpe = mean(excess)/std(excess) x sqrt(252), max drawdown from the
cumulative simple-return / running-max series.

## Primary window 2023-01-01 -> 2026-07-31

896 trading days (~3.56 years; 43 monthly observations). Basket holds 9 names at the end.

```
Window 2023-01-01 -> 2026-07-31   (basket names at end: 9)
               CAGR      Vol   Sharpe  Sortino    MaxDD
BASKET        59.8%    36.5%     1.36     1.77   -41.3%
XLI           20.2%    16.5%     0.95     1.42   -18.5%
SPY           22.4%    15.1%     1.15     1.57   -18.8%
XLU            9.9%    16.5%     0.41     0.59   -20.2%
GRID          23.8%    20.3%     0.96     1.37   -20.8%
PAVE          24.5%    20.7%     0.97     1.46   -26.2%

vs           exCAGR     corr       TE       IR
XLI           39.6%     0.70    27.5%     1.23
SPY           37.4%     0.68    28.5%     1.13
XLU           49.9%     0.26    35.9%     1.19
GRID          35.9%     0.82    23.1%     1.31
PAVE          35.3%     0.77    24.6%     1.20
```

Exact metric values (`output_grid_equipment/metrics.csv`):

| name | n_obs | cagr | ann_vol | sharpe | sortino | max_dd | hit_rate |
|---|---|---|---|---|---|---|---|
| BASKET | 896 | 0.5977 | 0.3648 | 1.360 | 1.774 | -0.4135 | 0.557 |
| XLI | 896 | 0.2020 | 0.1653 | 0.954 | 1.415 | -0.1849 | 0.542 |
| SPY | 896 | 0.2240 | 0.1511 | 1.148 | 1.567 | -0.1876 | 0.568 |
| XLU | 896 | 0.0989 | 0.1646 | 0.412 | 0.595 | -0.2020 | 0.535 |
| GRID | 896 | 0.2384 | 0.2033 | 0.957 | 1.371 | -0.2077 | 0.549 |
| PAVE | 896 | 0.2447 | 0.2068 | 0.969 | 1.460 | -0.2623 | 0.544 |

Relative to XLI (primary benchmark): daily-return correlation 0.70, tracking error 27.5%,
information ratio 1.23, excess CAGR +39.6 pp.

### Calendar-year total return (%)

| Year | BASKET | XLI | SPY | XLU | GRID | PAVE |
|---|---|---|---|---|---|---|
| 2023 | 79.2 | 17.9 | 26.7 | -7.2 | 21.6 | 31.0 |
| 2024 | 52.2 | 17.3 | 24.9 | 23.3 | 15.2 | 17.9 |
| 2025 | 51.6 | 19.4 | 17.7 | 16.0 | 29.6 | 19.4 |
| 2026 (Jan-Jul, partial) | 27.9 | 16.6 | 10.1 | 5.3 | 17.8 | 18.1 |

The basket beat every benchmark in every full calendar year of the window. 2026 is a
partial year (7 months, through the 2026-07-31 cutoff) and is not annualized.

## Prior-regime panel 2020-01-01 -> 2022-12-31

Context only — **not** a second decision gate. 755 trading days (~3.0 years; ~36 monthly
observations). Reported per spec §5, which is explicit that pre-2023 grid-capex conditions
were a different demand environment.

```
Window 2020-01-01 -> 2022-12-31   (basket names at end: 8)
               CAGR      Vol   Sharpe  Sortino    MaxDD
BASKET        25.4%    37.2%     0.69     0.90   -45.4%
XLI            7.6%    27.6%     0.26     0.33   -42.3%
SPY            7.3%    25.0%     0.25     0.31   -33.7%
XLU            6.7%    26.9%     0.23     0.29   -36.1%
GRID          17.2%    30.1%     0.55     0.70   -40.6%
PAVE          15.0%    32.7%     0.47     0.60   -44.1%

vs           exCAGR     corr       TE       IR
XLI           17.8%     0.87    18.8%     0.99
SPY           18.1%     0.81    22.4%     0.87
XLU           18.7%     0.58    30.8%     0.64
GRID           8.3%     0.87    18.7%     0.49
PAVE          10.5%     0.90    16.0%     0.64
```

Exact metric values (`output_grid_equipment_prior/metrics.csv`):

| name | n_obs | cagr | ann_vol | sharpe | sortino | max_dd | hit_rate |
|---|---|---|---|---|---|---|---|
| BASKET | 755 | 0.2542 | 0.3717 | 0.690 | 0.897 | -0.4536 | 0.525 |
| XLI | 755 | 0.0758 | 0.2763 | 0.258 | 0.330 | -0.4233 | 0.532 |
| SPY | 755 | 0.0730 | 0.2500 | 0.247 | 0.307 | -0.3372 | 0.530 |
| XLU | 755 | 0.0673 | 0.2686 | 0.228 | 0.294 | -0.3607 | 0.528 |
| GRID | 755 | 0.1716 | 0.3012 | 0.545 | 0.697 | -0.4056 | 0.532 |
| PAVE | 755 | 0.1496 | 0.3270 | 0.469 | 0.598 | -0.4408 | 0.531 |

Calendar-year total return (%):

| Year | BASKET | XLI | SPY | XLU | GRID | PAVE |
|---|---|---|---|---|---|---|
| 2020 | 40.1 | 8.8 | 17.2 | 1.8 | 46.2 | 19.9 |
| 2021 | 43.4 | 21.1 | 28.7 | 17.7 | 27.6 | 36.4 |
| 2022 | -1.9 | -5.6 | -18.2 | 1.4 | -13.9 | -7.2 |

Panel-specific data limitations (in addition to the general caveats below):

- **GEV is absent for the entire panel** — no vendor price history before 2024. The basket
  runs on 8 names throughout.
- **FLNC enters only at the 2021-11-11 rebalance** (IPO 2021-10-28) — it is in the basket for
  roughly the last 14 months of the 3-year panel, not the whole window.
- **VRT's pre-February-2020 quotes are the predecessor SPAC** (GS Acquisition Holdings,
  near its ~$10 trust value); early-panel VRT returns are near-flat placeholder data rather
  than a true operating-company equity. The vendor feed splices them onto VRT without a break.
- In 2020 the basket (+40.1%) actually trailed GRID (+46.2%); its full-panel lead over
  GRID/PAVE comes from 2021-2022.

The `run` orchestration prints one benign yfinance line for the GEV fetch during this panel
(`$GEV: possibly delisted; no price data found ... 1 Failed download: ['GEV']`). GEV has no
data in 2020-2022; the basket simply equal-weights the 8 names that do. This is expected, not
an error.

## Decision gate (Spec §5)

Evaluated on the **primary window only**, from `output_grid_equipment/metrics.csv`.
Gate passes iff **all three** hold.

| # | Check | Values | Pass? |
|---|---|---|---|
| 1 | basket CAGR > XLI CAGR | 59.77% vs 20.20%  (+39.57 pp) | **PASS** |
| 2 | basket Sharpe > XLI Sharpe | 1.360 vs 0.954  (+0.41) | **PASS** |
| 3 | basket CAGR >= max(GRID CAGR, PAVE CAGR) - 2.00 pp | 59.77% vs 24.47% - 2.00 pp = 22.47%  (+35.30 pp over PAVE, the better of the two) | **PASS** |

**Gate result: PASS.**

All three conditions clear, and by wide margins. Per the plan, Step 2 (the disclosed-backlog
weight tilt) is now unblocked: Tasks 6-8 — SEC XBRL / hand-collected backlog data, the tilt
path in `basket.py`, and the tilted-vs-equal-weight re-validation — may proceed.

## Mandatory caveats (Spec §5, §7)

### Sample size and significance (Spec §5)

- The primary window is ~3.56 years — 896 trading days, ~43 monthly observations. Short.
- The standard error on an annualized Sharpe estimate from a sample this size is on the order
  of +/-0.5 (a roughly +/-1.0 band at 95%). Basket Sharpe 1.36 vs XLI 0.95 is a **directional
  point-estimate difference, not a statistically significant one**. The same applies to the
  CAGR gaps. This is exploratory evidence about whether the theme is worth pursuing further,
  not a Sharpe/CAGR number to defend.
- **GEV coverage.** GEV has vendor price history only from 2024-03-27 and joins the basket at
  the 2024-05-13 rebalance. It contributes to 27 of the 43 monthly observations (~63%) and is
  absent from the first ~16 months (~37%) of the window — i.e. it affects only the back ~60%.
  (Spec §5's pre-implementation estimate was "less than half"; with the realized 2026-07-31
  cutoff GEV in fact covers slightly more than half. The direction of the concern is
  unchanged — a large winner is missing from a material early stretch of an already-short
  sample, biasing the basket's point estimates upward — but it is milder than the spec
  anticipated.)

### Survivorship / hindsight bias (Spec §7) — the headline risk

The universe was chosen in 2026 with full knowledge that VRT, GEV, and PWR have been among
the largest winners of the backtest window. A backtest built on that universe is biased
upward, and the 59.8% CAGR and the +35 pp gap over GRID/PAVE must be read in that light.
Mitigations actually applied:

1. **Inclusion on business description only.** Every name qualifies on a plain reading of its
   most recent 10-K, never on stock returns (`grid_equipment_basket/candidate_research.md`).
2. **A known loser is deliberately kept in.** FLNC (Fluence) is a chronic underperformer that
   passes the business test; it stays in the basket and its drag is in the numbers above.
3. **Equal-weight.** No ability to cherry-pick weights toward the winners.
4. **GRID / PAVE comparison.** These are rules-based third-party thematic ETFs covering
   approximately this theme. The basket beats both by a wide margin (CAGR 59.8% vs 23.8% /
   24.5%; Sharpe 1.36 vs 0.96 / 0.97), so the result is not merely "this is the theme" —
   *but* a hand-picked 2026-vintage universe can out-run a rules-based one precisely because
   the winners were already known, so this check does not neutralise the hindsight bias, only
   rules out the weakest version of it.
5. The writeup claims only what a short, biased window supports — directional, exploratory
   evidence.

### Risk profile

The edge here is high-beta, not smooth. The basket ran at **36.5% annualized volatility**
(XLI 16.5%, GRID 20.3%) with a **-41.3% max drawdown** (XLI -18.5%, GRID -20.8%). Its Sharpe
advantage comes from very high returns on roughly 2.2x the benchmark's volatility, not from a
better risk-adjusted ride at comparable volatility. Correlation to XLI is 0.70.

## Next step

Gate PASSED -> proceed to Task 6 (backlog data collection) and the Step 2 weight tilt, then
re-run the §5 metrics for the tilted basket and report tilted vs equal-weight vs all five
benchmarks side by side (spec §6). If the tilt does not improve **both** Sharpe and CAGR
versus equal-weight, equal-weight remains the recommended basket.

(Had the gate FAILED: stop here. The negative result plus `candidate_research.md` and the
Step 1 code would have been the deliverable, and no backlog data would be collected.)

---

# Step 2 — backlog tilt

Run date: 2026-08-29. Same price data, same **2023-01-01 -> 2026-07-31** window, same
rebalance calendar as Step 1. Tilt path: `grid_equipment_basket/basket.py::backlog_tilt_targets`,
invoked via `python -m grid_equipment_basket --start 2023-01-01 --tilt backlog`
(`output_grid_equipment_tilt/`).

## Construction

Start from the Step 1 equal weights on each rebalance date. Compute
`backlog_growth_signal(backlog_df, asof)` — trailing YoY growth in each name's own disclosed
backlog / RPO figure, point-in-time on the filing / earnings date; book-to-bill-only names
would be scored on (level - 1), though none are in the final data. Rank ascending; among the
names with a non-NaN rank on that date, the top half are multiplied by **1.25x**, the bottom
half by **0.75x**, and (on an odd count) the median name plus every name with no rank stay at
**1.0x**. Renormalize to sum 1; re-apply the 25% single-name cap. The 1.25 / 0.75 constants
live in `config.py` (`BACKLOG_TILT_TOP` / `BACKLOG_TILT_BOTTOM`) — fixed, documented, not
fitted.

The tilt is small and self-financing: with 9 names the equal weight is 11.11% and a tilted
name moves to 13.89% (up) or 8.33% (down) — a +/-2.78 pp step. The top and bottom halves are
equal in size, so the steps net to zero and the 25% cap never binds anywhere in the run.

## VRT gappy-signal guard (carried from the Task 6 review)

`backlog_growth_signal` picks the year-ago value positionally (`g.iloc[-5]`). That is a true
year ago only on a clean consecutive-quarter series. VRT discloses backlog irregularly
(9 rows, 6 quarters missing across 2023-26), so `iloc[-5]` can sit ~18 months back and
inflate VRT's "YoY" growth — over-ranking a known 2023-26 winner into the +1.25x bucket for
the wrong reason, a bias the §6 re-validation gate would not catch. A lookback-span guard now
sets a name's signal to NaN whenever its `iloc[-5] -> iloc[-1]` quarter-end span falls
outside **300-430 days**.

Effect on this run: VRT is left **untilted** (no rank) at the 2025-05-12, 2025-08-11 and
2025-11-11 rebalances (span 547-548 days each). It is tilted **up** (x1.25, top half) at
2026-02-11 and 2026-05-12, where its span is a legitimate 365 days and the ~108% YoY jump is
real (Q4-25 backlog 9.5bn -> 15.0bn). Without the guard VRT would have been over-ranked into
the top bucket across 2025 on an 18-month lookback.

## Which names the tilt actually moved

| Name | Disclosure | Tilt behaviour over the window |
|---|---|---|
| ETN | xbrl_rpo, then non-GAAP total | tilted (both directions) from 2024-05-13 on |
| FLNC | xbrl_rpo | tilted **up** 2024-08 -> 2025-02 (fastest disclosed-backlog growth in the set), down thereafter |
| GEV | xbrl_rpo | no signal until 2025-05-12 (needs 5 quarters of post-spin history); tilted after |
| MYRG | xbrl_rpo | tilted **down** at every active rebalance — slowest backlog growth in the set |
| PRIM | xbrl_rpo | tilted (both directions) |
| PWR | xbrl_rpo | tilted (mostly up) from 2024-05-13 on |
| VRT | non-GAAP total (gappy) | untilted across 2024-25 (span guard); tilted up at the last two rebalances |
| HUBB | non-GAAP total, **annual only** | **never tilted** — NaN signal at every rebalance |
| NVT | non-GAAP total, **annual only** | **never tilted** — NaN signal at every rebalance |

No tilt is applied at any 2023 rebalance or at 2024-02-12 (fewer than two names have a
non-NaN rank). The first active tilt is **2024-05-13**; from there the basket carries a tilt
at every rebalance to the end of the window.

## Tilted vs equal-weight vs the five benchmarks

Primary window 2023-01-01 -> 2026-07-31, 896 trading days (~3.56 y, 43 monthly obs). Exact
values: `output_grid_equipment_tilt/metrics.csv` (tilt), `output_grid_equipment/metrics.csv`
(equal-weight, from Task 5).

| Series | CAGR | Vol | Sharpe | Sortino | MaxDD |
|---|---|---|---|---|---|
| **BASKET — equal-weight** | 59.77% | 36.48% | 1.360 | 1.774 | -41.35% |
| **BASKET — backlog tilt** | 59.16% | 35.98% | 1.363 | 1.784 | -43.01% |
| XLI | 20.20% | 16.53% | 0.954 | 1.415 | -18.49% |
| SPY | 22.40% | 15.11% | 1.148 | 1.567 | -18.76% |
| XLU | 9.89% | 16.46% | 0.412 | 0.595 | -20.20% |
| GRID | 23.84% | 20.33% | 0.957 | 1.371 | -20.77% |
| PAVE | 24.47% | 20.68% | 0.969 | 1.460 | -26.23% |

Basket-vs-benchmark, **tilt** (equal-weight in parentheses):

| vs | excess CAGR | corr | TE | IR |
|---|---|---|---|---|
| XLI | +38.96% (+39.57%) | 0.71 (0.70) | 26.9% (27.5%) | 1.24 (1.23) |
| SPY | +36.76% (+37.37%) | 0.68 (0.68) | 28.0% (28.5%) | 1.13 (1.13) |
| XLU | +49.27% (+49.88%) | 0.26 (0.26) | 35.4% (35.9%) | 1.19 (1.19) |
| GRID | +35.32% (+35.93%) | 0.82 (0.82) | 22.4% (23.1%) | 1.32 (1.31) |
| PAVE | +34.69% (+35.30%) | 0.77 (0.77) | 24.0% (24.6%) | 1.21 (1.20) |

Calendar-year total return (%), tilt vs equal-weight:

| Year | tilt | equal-weight | XLI | SPY | XLU | GRID | PAVE |
|---|---|---|---|---|---|---|---|
| 2023 | 79.2 | 79.2 | 17.9 | 26.7 | -7.2 | 21.6 | 31.0 |
| 2024 | 52.0 | 52.2 | 17.3 | 24.9 | 23.3 | 15.2 | 17.9 |
| 2025 | 44.7 | 51.6 | 19.4 | 17.7 | 16.0 | 29.6 | 19.4 |
| 2026 (Jan-Jul, partial) | 32.5 | 27.9 | 16.6 | 10.1 | 5.3 | 17.8 | 18.1 |

2023 is identical (no tilt before 2024-05-13). The tilt cost ~7 pp in 2025 — it consistently
underweighted MYRG and, at points, PWR / GEV, all of which were strong that year — and
recovered ~5 pp in 2026. Over the full window the tilt is a small **net negative** on total
return.

## §6 keep-or-adopt decision

Spec §6: *"If the tilt does not improve **both** Sharpe and CAGR relative to equal-weight,
equal-weight remains the recommended basket and the README says so explicitly."*

| Gate metric | equal-weight | tilt | tilt better? |
|---|---|---|---|
| Sharpe | 1.360 | 1.363 | yes, by +0.003 |
| CAGR | 59.77% | 59.16% | **no, -0.61 pp** |

**Decision: keep equal-weight.** The tilt fails the §6 bar — CAGR is lower. The nominal
+0.003 Sharpe uplift comes entirely from slightly lower volatility (35.98% vs 36.48%) and is
meaningless against the ±0.5 Sharpe standard error on a 43-observation sample (§5); the tilt
also deepened the max drawdown by 1.7 pp. Mechanically the tilt did the wrong thing where it
mattered: it over-weighted **FLNC** — the basket's designated chronic underperformer —
through late 2024 / early 2025 because FLNC posted the fastest disclosed-backlog growth in
the set, and it under-weighted **MYRG** throughout even though MYRG was a steady contributor.
Backlog-growth rank did not line up with forward return over this window.

This does **not** resolve whether a backlog tilt could work with a longer history, with clean
segment-level disclosure across all nine names (HUBB and NVT contribute nothing today), or
with a different signal definition (level vs growth, surprise vs raw). It says only that the
fixed 1.25x / 0.75x growth-rank tilt does not beat equal-weight on this ~43-month,
survivorship-biased sample.

## Caveats unchanged (Spec §5, §7)

The Step 2 comparison inherits every §5 and §7 caveat from Step 1 without exception: the same
~3.56-year / 43-observation window, the same Sharpe standard error on the order of ±0.5 (so
the tilt-vs-equal gaps are far inside the noise), the same GEV coverage gap (GEV in the
basket only for the back ~60% of the window), and the same headline survivorship / hindsight
bias from choosing the universe in 2026 knowing which names won. The tilt does not lengthen
the sample or reduce the bias — it adds one more fitted-looking degree of freedom (the tilt
direction) on top of the universe choice, which is a further reason to prefer the simpler
equal-weight construction here.
