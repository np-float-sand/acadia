# Grid Equipment Basket — Value-Chain Reframe: Results

Spec: docs/superpowers/specs/2026-08-29-value-chain-reframe-design.md
Plan: docs/superpowers/plans/2026-08-29-value-chain-reframe.md
Run date: 2026-08-29 (refreshed after the final-review fix wave). Prices: yfinance auto-adjusted
daily closes (warm monthly-parquet cache). Fundamentals: SEC XBRL `companyconcept`, filing-dated —
`GrossProfit` (single tag), revenue **unioned** across `Revenues` +
`RevenueFromContractWithCustomerExcludingAssessedTax` (deduped per quarter, earliest filing kept),
`CostOfRevenue` / `CostOfGoodsAndServicesSold` fallback for the derived-gross path.

Both constructions are produced by one function (`backtest.value_chain_report`) and one table
(`backtest.value_chain_table`); `--construction value-chain-tilt` and `--construction pair`
print byte-identical output. The gate lines are the operative difference in intent, not in code.

## Buckets (frozen, from 10-K business descriptions — spec §3)

Makers (pricing power): **ETN, HUBB, GEV, VRT, NVT**
Contractors / assemblers (price-takers): **PWR, MYRG, PRIM, FLNC**  (FLNC borderline — assembles
third-party cells, bids competitively, chronic negative gross margin: no pricing power.)

Assigned once from business-model language, frozen in `config.py` before the first backtest,
never revised from results.

## Signal coverage (live SEC XBRL fetch, 2026-08-29 — post revenue-tag-union fix)

The final review found a **code bug** in `margin_data`: the revenue concept was picked by
first-nonempty over `("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")`. For
MYRG / PRIM / PWR / VRT the legacy `Revenues` tag returns a handful of stale pre-2019 quarterly
facts, so first-nonempty stopped there and the modern ASC-606 tag (25–38 recent facts) was never
read. That silently killed the backlog-coverage signal for four of nine names. The fix unions the
two tags (dedupe per `quarter_end`, keep the earliest filing). Effect below.

Per-name quarterly-fundamentals rows now returned by `margin_data.fetch_fundamentals`, and how
many of the 14 primary-window rebalances each signal is live:

| Name | Bucket | XBRL rows | Span | Gross-margin §4.1 | Backlog-coverage §4.2 |
|---|---|---:|---|---|---|
| ETN  | maker      | 27 | 2017-09 → 2026-06 | 0/14 — never tags a discrete December quarter (months 3/6/9 only); trailing-5-quarter span ≈ 456d trips the 300–430d guard | **9/14**, from the 2024-05 rebalance |
| HUBB | maker      |  0 | — | 0/14 — **genuinely factless**: `Revenues` → 404, `RevenueFromContractWithCustomerExcludingAssessedTax` → 0 facts, `GrossProfit` → 0 facts, `CostOfGoodsAndServicesSold` → 0 facts, `CostOfRevenue` → 404. HUBB does not file these concepts as discrete-quarter XBRL facts under any of the tags this module reads | 0/14 — no TTM-revenue denominator |
| GEV  | maker      | 11 | 2023-03 → 2026-06 | 0/14 — months 3/6/9 only (no discrete Dec quarter) | **4/14**, from 2025-08 (needs 8 post-spin quarters of revenue for the TTM-YoY) |
| VRT  | maker      | 25 | 2019-03 → 2026-06 | 0/14 — tagged discrete Dec quarters through ~2021 then stopped; recent trailing-5 span ≈ 456d > 430d | **9/14**, from 2024-05  *(fixed by the tag union — was 3 rows / never live)* |
| NVT  | maker      | 33 | 2017-03 → 2026-06 | 0/14 — stopped tagging discrete Dec quarters ~2022; recent span > 430d | **7/14**, from 2024-05 — annual-only backlog; the §4.2 400-day carry-forward (see below) lifts it from ~2/yr to ~3/yr |
| PWR  | contractor | 32 | 2017-03 → 2026-06 | 0/14 — stopped tagging discrete Dec quarters ~2022; recent span > 430d | **9/14**, from 2024-05  *(fixed by the tag union — was 6 rows / never live)* |
| MYRG | contractor | 33 | 2017-03 → 2026-06 | 0/14 — stopped tagging discrete Dec quarters ~2022; recent span > 430d | **9/14**, from 2024-05  *(fixed by the tag union — was 3 rows / never live)* |
| PRIM | contractor | 38 | 2017-03 → 2026-06 | **14/14**, from the first rebalance — PRIM tags all four discrete quarters cleanly, so the trailing-5 span is ≈ 365d and passes the guard | **9/14**, from 2024-05  *(fixed by the tag union — was 6 rows / never live)* |
| FLNC | contractor | 18 | 2020-12 → 2026-06 | 0/14 — tags Dec but **skips the September quarter** every year; trailing-5 span ≈ 456d > 430d | **10/14**, from 2024-02 |

**What changed vs the pre-fix write-up.** The old doc claimed the coverage signal was live for only
four names (ETN/FLNC/GEV/NVT) and the gross-margin signal "never fired anywhere". Both statements
were consequences of the revenue-tag bug plus an over-broad reading of the span guard. Corrected:

- **Backlog-coverage (§4.2) is now live for 8 of 9 names** (all but HUBB), from the 2024-05
  rebalance onward (FLNC from 2024-02, GEV from 2025-08). Only HUBB — which files none of the
  revenue/cost concepts as quarterly XBRL — gets the base bucket multiplier at every rebalance.
- **Gross-margin (§4.1) is live for exactly one name in the primary window: PRIM (14/14).** The
  other eight are NaN'd by the trailing-5-quarter span guard, for four distinct reasons: never
  tags a discrete Q4 (ETN, GEV), *stopped* tagging discrete Q4 around 2022 (NVT, VRT, MYRG, PWR),
  skips Q3 (FLNC), no facts at all (HUBB). This is the same "derive Q4 = FY − 9-month YTD"
  follow-up logged in spec §9 / root `CLAUDE.md` — not fixed in this wave.
- In the **2020–2022 prior-regime panel** the gross-margin signal *does* fire — for up to four of
  MYRG / NVT / PRIM / PWR / VRT at most rebalances — because those filers still tagged discrete
  December quarters back then. (The old doc's "inert here too" was wrong.)

**What actually tilts the book.** Construction 1's within-bucket move is now driven by a real
§4.2 coverage-change rank on 8 names from mid-2024 plus PRIM's §4.1 margin rank throughout — but
it is still small relative to the static maker ×1.25 / contractor ×0.75 base split. The parameter
plateau below shows the tilt with the within-bucket signal switched off (×1.00/×1.00) at Sharpe
1.434 / CAGR 63.18% vs the shipped 1.441 / 64.13% — inside rounding. The Gate-1 pass is the
bucket split, not the signal.

## Construction 1 — value-chain-tilted long-only, primary window 2023-01-01 → 2026-07-31

896 trading days (~43 monthly obs).

| Series | CAGR | Vol | Sharpe | Sortino | MaxDD |
|---|---:|---:|---:|---:|---:|
| equal-weight (Step 1 basket) | 59.77% | 36.48% | 1.360 | 1.774 | -41.35% |
| **value-chain tilt** | **64.13%** | **36.26%** | **1.441** | 1.857 | -41.76% |
| XLI  | 20.20% | 16.53% | 0.954 | 1.415 | -18.49% |
| SPY  | 22.40% | 15.11% | 1.148 | 1.567 | -18.76% |
| XLU  |  9.89% | 16.46% | 0.412 | 0.595 | -20.20% |
| GRID | 23.84% | 20.33% | 0.957 | 1.371 | -20.77% |
| PAVE | 24.47% | 20.68% | 0.969 | 1.460 | -26.23% |

### GATE 1 (spec §7.4)

tilt Sharpe **1.441** vs equal-weight **1.360** (+0.081)  |  tilt CAGR **64.13%** vs **59.77%**
(+4.36 pp)  →  **PASS**

Both legs clear, so per the pre-registered rule the tilt clears its gate and is available as
`--construction value-chain-tilt`. **The pass is the static bucket split, not the signal** — the
plateau row with the within-bucket signal off is 1.434 / 63.18%, inside rounding of the shipped
1.441 / 64.13%. Both gaps are well inside the ±0.5 Sharpe standard error (§8). The revenue-tag
fix moved the tilt CAGR from 63.66% to 64.13% (the coverage signal now shapes 8 names' ranks
instead of 4) — it did **not** move the Gate-1 verdict.

## Construction 2 — market-neutral pair, primary window 2023-01-01 → 2026-07-31

| Series | CAGR | Vol | Sharpe | Sortino | MaxDD | episodeDD | carry |
|---|---:|---:|---:|---:|---:|---:|---:|
| pair (standalone, 100/100 gross) | 15.74% | 32.97% | 0.485 | 0.718 | -33.93% | — | 15.74% |
| EW + pair 30% overlay | 69.36% | 36.89% | 1.508 | — | -43.08% | -43.08% | — |
| EW + conditional QQQ short | 59.73% | 35.23% | 1.395 | — | -37.34% | -37.34% | -0.53% |
| EW + pair risk-matched overlay | n/a | n/a | n/a | — | n/a | n/a | k = **NaN** |

Episode (fixed span, spec §7.3): peak **2024-11-22**, trough **2025-04-04**; equal-weight episode
drawdown **-41.35%**.

**The risk-matched row is `NaN`, not `k = 0`.** `hedges.risk_match_weight` previously assumed
`vol(k)` is monotone in `k` and bisected; the final review flagged that as a bug (a hedging pair
makes `vol(k)` U-shaped and bisection can converge to a levered upper root). The fixed routine
grid-scans for the achievable vol minimum and returns `NaN` when the target vol is unreachable.
Here the conditional short *lowers* portfolio vol (35.23%) below the un-hedged basket (36.48%),
and the current pair is vol-*additive* over `k ≥ 0`, so **no non-negative pair weight can match
the conditional short's vol** — `NaN` is the correct answer (the old `k = 0` was a silent clamp).
This is a reporting-row change only; Gate 2 uses the fixed 30% overlay, not the risk-matched one.

### GATE 2 (spec §7.4)

pair episodeDD **-43.08%** vs conditional-short episodeDD **-37.34%**  |  pair carry **15.74%** vs
conditional-short carry **-0.53%**  →  **FAIL**

Gate 2 requires **both** conditions.

- **Carry leg — passes.** Pair standalone carry 15.74% is *better*, not worse, than the
  conditional short's -0.53%. (Ruling 4 note: the two carry numbers are rates on different gross
  bases — the pair is 100/100, the conditional short is an intermittent 30% QQQ leg — so read
  the comparison as "is the pair's carry no worse", per the spec §7.4 wording, not as a
  like-for-like P&L.)
- **Protection leg — fails.** The pair 30% overlay made the DeepSeek episode *deeper*
  (-43.08% vs -41.35% un-hedged), because long-makers / short-contractors is not a hedge for a
  chain-wide de-rate: both legs fell together and the makers leg fell further. The conditional
  QQQ short cut the same episode to -37.34% at negligible carry.

Borrow cost that would erase the pair's carry edge over the conditional short:
**16.27 pp/yr** on the 100%-gross short leg (15.74% − (−0.53%)). Concentrated in **MYRG**
(thin, ~$20–80M/day traded) and **FLNC** (hard/expensive to borrow — already heavily shorted,
going-concern noise). Academic here: the pair fails Gate 2 on protection regardless of carry.

## Prior-regime panel 2020–2022 (context only — spec §7.1, not a gate)

2020-01-01 → 2022-12-31, 755 trading days. **Now produced through the real
`backtest.value_chain_report` / CLI path** — the final review fixed the `--prior-regime` crash
(the Gate-2 drawdown episode is pinned to a hard-coded 2024-H2 window; a 2020–2022 run has no
data there, so Gate 2 is now reported as *not applicable* instead of raising `ValueError`).

| Series | CAGR | Vol | Sharpe | Sortino | MaxDD |
|---|---:|---:|---:|---:|---:|
| equal-weight | 25.42% | 37.17% | 0.690 | 0.897 | -45.36% |
| value-chain tilt | 24.51% | 36.16% | 0.678 | 0.877 | -45.32% |
| XLI  |  7.58% | 27.63% | 0.258 | 0.330 | -42.33% |
| SPY  |  7.30% | 25.00% | 0.247 | 0.307 | -33.72% |
| XLU  |  6.73% | 26.86% | 0.228 | 0.294 | -36.07% |
| GRID | 17.16% | 30.12% | 0.545 | 0.697 | -40.56% |
| PAVE | 14.96% | 32.70% | 0.469 | 0.598 | -44.08% |
| pair (standalone) | -14.18% | 32.13% | -0.437 | -0.543 | -54.32% |

Re-applying the Gate-1 test on this panel: tilt Sharpe 0.678 **<** equal-weight 0.690 **and**
tilt CAGR 24.51% **<** 25.42% → the tilt **loses on both legs**. Overweighting makers and
underweighting contractors was mildly harmful in 2020–2022. **GATE 2: n/a** (no 2024-H2 episode).
Panel limitations carried from Step 1: **GEV absent** the whole panel (no vendor price history
before 2024-03), **FLNC** only from its 2021-10 IPO, **VRT** early quotes are the predecessor
SPAC. Here the §4.1 gross-margin signal *does* fire (for up to four of MYRG / NVT / PRIM / PWR /
VRT — those filers still tagged discrete December quarters pre-2022) and §4.2 coverage never
fires (no in-window backlog rows before 2022-12); the tilt is still ≈ the static ×1.25/×0.75
split plus a small pre-2022 margin nudge.

## Drop-winners robustness — drop VRT, GEV from makers (spec §7.1), primary window

makers = **ETN, HUBB, NVT**; contractors unchanged.

| Series | CAGR | Vol | Sharpe | Sortino | MaxDD |
|---|---:|---:|---:|---:|---:|
| equal-weight | 42.26% | 35.22% | 1.065 | 1.413 | -43.04% |
| value-chain tilt | 41.53% | 33.54% | 1.086 | 1.421 | -42.00% |
| pair (standalone) | -13.86% | 32.00% | -0.432 | -0.599 | -62.03% |

Re-applying Gate 1 on this reduced universe: tilt Sharpe 1.086 > equal-weight 1.065, but
tilt CAGR 41.53% **<** equal-weight 42.26% → **Gate 1 would FAIL**. "Makers beat contractors on
return" does **not** survive removing the two largest winners; only a thin Sharpe edge survives,
and it comes entirely from the tilt running ~1.7 pp lower vol (concentrating away from the four
contractors), not from stock selection. The pair standalone over the same reduced universe is
CAGR -13.86% / Sharpe -0.43 / max DD -62.0% (risk-matched overlay `k` ≈ 0.10 here — a real small
root, since this pair *is* mildly vol-reducing). Gate 2 on the reduced universe also FAILs
(episodeDD -42.04% vs -38.91%; carry -13.86% vs -0.42%).

## Parameter plateau (spec §5, §8 — reported, not optimized)

`backtest.value_chain_report` re-run with the four `config` constants monkeypatched in an inline
script (no search added to the module). Primary window; Gate-1 column is the tilt-vs-equal-weight
pass/fail.

| Config | CAGR | Vol | Sharpe | MaxDD | Gate 1 |
|---|---:|---:|---:|---:|:--:|
| equal-weight (reference) | 59.77% | 36.48% | 1.360 | -41.35% | — |
| base 1.20/0.80, within 1.10/0.90 | 63.44% | 36.32% | 1.427 | -41.78% | PASS |
| **base 1.25/0.75, within 1.10/0.90 (shipped)** | **64.13%** | **36.26%** | **1.441** | **-41.76%** | **PASS** |
| base 1.30/0.70, within 1.10/0.90 | 64.80% | 36.21% | 1.454 | -41.74% | PASS |
| base 1.25/0.75, within 1.05/0.95 | 63.65% | 36.10% | 1.437 | -41.46% | PASS |
| base 1.25/0.75, within 1.15/0.85 | 64.60% | 36.41% | 1.444 | -42.07% | PASS |
| base 1.25/0.75, within 1.00/1.00 (signal off) | 63.18% | 35.95% | 1.434 | -41.15% | PASS |
| base 1.00/1.00, within 1.00/1.00 (= equal-weight, sanity) | 59.77% | 36.48% | 1.360 | -41.35% | FAIL |

Flat and monotone in the base split (Sharpe 1.427 → 1.454 as the split widens 1.20/0.80 →
1.30/0.70) and almost independent of the within-bucket multiplier (1.434 signal-off vs 1.441
shipped). Nothing was tuned; the result is not sensitive to the exact constants, and it is not
coming from the fundamental signal.

## Caveats (spec §8 — carried + new)

- **~3.56-year / ~43-monthly-obs window, one macro regime.** Standard error on an annualized
  Sharpe from this sample is ~±0.5 (≈ ±1.0 at 95%). Every gate comparison and every gap above is
  a directional point estimate, **not** a significant result. The Gate 1 Sharpe gap (+0.081) and
  CAGR gap (+4.36 pp) are both well inside that noise.
- **GEV** has vendor price history only from 2024-03-27 — it covers only the back ~60% of the
  primary window and none of the prior-regime panel.
- **Headline survivorship / hindsight bias:** the 9-name universe was chosen in 2026 knowing VRT,
  GEV, PWR won.
- **The 5/4 bucket split overlaps the handoff's hindsight-flagged hand-split.** Mitigations:
  business-model split rule frozen pre-backtest; prior-regime panel (tilt loses on both legs);
  drop-winners pass (Gate 1 fails on CAGR); PWR — a large winner over the window — sits in the
  *short* leg, so shorting it cost money. The drop-winners failure means much of Gate 1's pass
  rides on VRT and GEV being in the overweight bucket.
- **The §4.1 gross-margin signal is live for only PRIM in the primary window** (14/14); the other
  eight names are span-guard-rejected because they never tag / stopped tagging / skip a discrete
  quarter. A rebuild that derives Q4 = FY − 9-month YTD (spec §9, root `CLAUDE.md`) is the open
  follow-up. The within-bucket signal is a small perturbation on the static bucket split either
  way (plateau: signal-off 1.434 vs shipped 1.441).
- **Backlog-coverage (§4.2) is live for 8 of 9 names** (all but HUBB) from the 2024-05 rebalance
  (FLNC from 2024-02, GEV from 2025-08). HUBB takes the base bucket multiplier at every rebalance
  because it files none of the revenue/cost concepts as quarterly XBRL facts.
- **Short leg is not costless.** Pair results are gross; borrow-cost sensitivity stated above
  (16.27 pp/yr to erase the carry edge; MYRG thin, FLNC hard to borrow).
- **The pair adds fitted-looking degrees of freedom** (base split, within-bucket multipliers, three
  conditional-short thresholds). The plateau is reported; nothing was optimized.

## Decision

**Gate 1 — long-only tilt: PASS (as written), adopt with a health warning.** Over the primary
window the value-chain tilt beats the Step 1 equal-weight basket on both Sharpe (1.441 vs 1.360)
and CAGR (64.13% vs 59.77%), so by the pre-registered rule it clears the gate and becomes an
available long construction (`--construction value-chain-tilt`). But the evidence that this is an
*edge* rather than a relabeled bet on the 2026-vintage winners is weak: the gaps are inside ±0.5
Sharpe noise; the pass is reproduced with the fundamental signal switched off (it is the static
maker/contractor split); it does **not** survive dropping VRT and GEV (tilt CAGR 41.53% <
equal-weight 42.26%); and the prior-regime panel has the tilt losing on both legs (Sharpe 0.678
vs 0.690, CAGR 24.51% vs 25.42%). Recommended framing: keep **equal-weight** as the headline
Step 1 basket; offer the tilt as a documented alternative, not as a demonstrated skill.

**Gate 2 — the pair as a hedge: FAIL → use the conditional QQQ short.** The pair 30% overlay made
the spec §7.3 episode *deeper* (-43.08% vs -41.35% un-hedged); the conditional QQQ short cut it
(-37.34%) at a carry cost of -0.53%/yr. Per spec §7.4 the recommendation on Gate-2 failure is
"use the conditional QQQ short" (or "size down, no hedge") — both acceptable. Adopt the
**conditional QQQ short** as the intra-theme drawdown hedge. Note separately that the pair has
positive standalone carry (15.74%, up from 11.45% pre-fix as the coverage signal now shapes both
legs) and, as a 30% overlay, *raised* return and Sharpe (69.36% / 1.508) over this window — it is
interesting as a return sleeve, but it is not a hedge, and Gate 2 tests it as a hedge.
