# Grid Equipment Basket — Value-Chain Reframe: Results

Spec: docs/superpowers/specs/2026-08-29-value-chain-reframe-design.md
Plan: docs/superpowers/plans/2026-08-29-value-chain-reframe.md
Run date: 2026-08-29. Prices: yfinance auto-adjusted daily closes (warm monthly-parquet cache).
Fundamentals: SEC XBRL `companyconcept` (`GrossProfit` / `Revenues` / `CostOfRevenue`), filing-dated.

Both constructions are produced by one function (`backtest.value_chain_report`) and one table
(`backtest.value_chain_table`); `--construction value-chain-tilt` and `--construction pair`
print byte-identical output. The gate lines are the operative difference in intent, not in code.

## Buckets (frozen, from 10-K business descriptions — spec §3)

Makers (pricing power): **ETN, HUBB, GEV, VRT, NVT**
Contractors / assemblers (price-takers): **PWR, MYRG, PRIM, FLNC**  (FLNC borderline — assembles
third-party cells, bids competitively, chronic negative gross margin: no pricing power.)

Assigned once from business-model language, frozen in `config.py` before the first backtest,
never revised from results.

## Signal coverage (Task 12 Step 1 — live SEC XBRL fetch)

Per-name quarterly-fundamentals rows returned by `margin_data.fetch_fundamentals`:

| Name | Bucket | XBRL rows | Span | Gross-margin signal (§4.1) | Backlog-coverage signal (§4.2) |
|---|---|---:|---|---|---|
| ETN  | maker      | 27 | 2017-09 → 2026-06 | **never live** — every Dec quarter untagged as a discrete 3-month period, so the last-5-quarter span is ~547d and trips the 300–430d span guard | live from the 2024-05 rebalance onward |
| HUBB | maker      |  0 | — | never live — no `GrossProfit`/`Revenues` discrete-quarter facts returned | never live — no TTM-revenue denominator |
| GEV  | maker      | 11 | 2023-03 → 2026-06 | never live — Dec quarters untagged, last-5 span ~547d > 430d | live from the 2025-08 rebalance onward |
| VRT  | maker      |  3 | 2019-03 → 2019-09 | never live — 3 stale rows, < 8-quarter minimum | never live — no fresh TTM revenue |
| NVT  | maker      | 33 | 2017-03 → 2026-06 | never live — Dec quarters untagged, last-5 span ~547d > 430d | live intermittently (2024-05, 2025-05, 2026-05 — annual-only backlog, 4 rows) |
| PWR  | contractor |  6 | 2023-03 → 2024-09 | never live — < 8-quarter minimum | never live — < 4 visible quarters at the year-prior comparison date |
| MYRG | contractor |  3 | 2018-03 → 2018-09 | never live — 3 stale rows | never live — no fresh TTM revenue |
| PRIM | contractor |  6 | 2017-03 → 2018-09 | never live — stale, pre-window | never live — no fresh TTM revenue |
| FLNC | contractor | 18 | 2020-12 → 2026-06 | never live — a missing quarter pushes the last-5 span to ~455d > 430d | live from the 2024-02 rebalance onward |

**Material finding — the gross-margin signal (§4.1) is inert for the entire primary window.**
`_flow_facts` keeps only XBRL facts whose period is 80–100 days (true discrete quarters). US
filers overwhelmingly tag the December quarter only inside the annual (~365-day) figure, so every
name's Q4 is absent from the frame. The `VC_SPAN_MIN_DAYS`/`VC_SPAN_MAX_DAYS` guard (300–430d
across the trailing five quarter-ends) — added to reject gappy series — therefore rejects **every**
name at **every** rebalance. No name receives a §4.1 margin adjustment anywhere in
2023-01-01 → 2026-07-31 (or in the 2020–2022 panel).

**What actually tilts the book:** only the backlog-coverage change (§4.2), and only for
**ETN, FLNC, GEV, NVT** (the four names with recent XBRL revenue for the TTM denominator), and
only from 2024-02 onward. HUBB, VRT, PWR, MYRG, PRIM get the **base bucket multiplier only** at
every rebalance. So the tilt is, in practice, ~entirely the static maker ×1.25 / contractor ×0.75
split with a small within-maker nudge (ETN vs GEV/NVT) in 2024–2026.

## Construction 1 — value-chain-tilted long-only, primary window 2023-01-01 → 2026-07-31

896 trading days (~43 monthly obs).

| Series | CAGR | Vol | Sharpe | Sortino | MaxDD |
|---|---:|---:|---:|---:|---:|
| equal-weight (Step 1 basket) | 59.77% | 36.48% | 1.360 | 1.774 | -41.35% |
| **value-chain tilt** | **63.66%** | 35.97% | **1.441** | 1.860 | -41.15% |
| XLI  | 20.20% | 16.53% | 0.954 | 1.415 | -18.49% |
| SPY  | 22.40% | 15.11% | 1.148 | 1.567 | -18.76% |
| XLU  |  9.89% | 16.46% | 0.412 | 0.595 | -20.20% |
| GRID | 23.84% | 20.33% | 0.957 | 1.371 | -20.77% |
| PAVE | 24.47% | 20.68% | 0.969 | 1.460 | -26.23% |

### GATE 1 (spec §7.4)

tilt Sharpe **1.441** vs equal-weight **1.360**  |  tilt CAGR **63.66%** vs **59.77%**  →  **PASS**

Both legs clear (tilt > equal-weight on Sharpe and on CAGR), so per the pre-registered rule the
tilt clears its gate and is available as `--construction value-chain-tilt`.

**Mechanically, the pass is the static bucket split, not the signal.** At the last rebalance the
tilt holds makers at 0.122–0.149 and every contractor at 0.081 (vs 0.111 equal-weight); the only
within-bucket move is ETN pushed to the ×0.90 slot and GEV to ×1.10 among makers (from their
backlog-coverage ranks), with HUBB/VRT/NVT and all four contractors left at the bucket base. The
parameter plateau below shows within-bucket multipliers of ×1.00/×1.00 (i.e. signal switched
off) give Sharpe 1.434 / CAGR 63.18% — inside rounding of the shipped 1.441 / 63.66%. The tilt is
"overweight the five makers, underweight the four contractors, hold it roughly fixed."

## Construction 2 — market-neutral pair, primary window 2023-01-01 → 2026-07-31

| Series | CAGR | Vol | Sharpe | Sortino | MaxDD | episodeDD | carry |
|---|---:|---:|---:|---:|---:|---:|---:|
| pair (standalone, 100/100 gross) | 11.45% | 33.93% | 0.371 | 0.524 | -45.85% | — | 11.45% |
| EW + pair 30% overlay | 68.15% | 36.06% | 1.514 | 1.936 | -41.04% | -41.04% | — |
| EW + conditional QQQ short | 59.73% | 35.23% | 1.395 | 1.817 | -37.34% | -37.34% | -0.53% |
| EW + pair risk-matched overlay | 59.77% | 36.48% | 1.360 | 1.774 | -41.35% | -41.35% | k = 0.00 |

Episode (fixed span, spec §7.3): peak **2024-11-22**, trough **2025-04-04**; equal-weight episode
drawdown **-41.35%**.

The risk-matched row is degenerate: the conditional short *lowers* portfolio vol (35.23%) below
the naked basket (36.48%), so matching the pair overlay to that vol drives its weight to zero and
the row reproduces the equal-weight basket.

### GATE 2 (spec §7.4)

pair episodeDD **-41.04%** vs conditional-short episodeDD **-37.34%**  |  pair carry **11.45%** vs
conditional-short carry **-0.53%**  →  **FAIL**

Gate 2 requires **both** conditions. The carry condition holds (pair carry 11.45% is *better*, not
worse, than the conditional short's -0.53%). The protection condition fails: the pair 30% overlay
barely touched the DeepSeek episode (-41.04% vs -41.35% naked — 0.3pp), because long-makers /
short-contractors is not a hedge for a chain-wide de-rate: both legs fell together and the makers
leg fell further (pair standalone -45.85% max DD). The conditional QQQ short cut the same episode
to -37.34% (~4pp) at negligible carry.

Borrow cost that erases the pair's carry edge over the conditional short: **11.98 pp/yr** on the
short leg. Concentrated in **MYRG** (thin, ~$20–80M/day traded) and **FLNC** (hard/expensive to
borrow — already heavily shorted, going-concern noise). This number is academic here: the pair
fails Gate 2 on protection regardless of carry.

## Prior-regime panel 2020–2022 (context only — spec §7.1, not a gate)

2020-01-01 → 2022-12-31, 755 trading days.

| Series | CAGR | Vol | Sharpe | Sortino | MaxDD |
|---|---:|---:|---:|---:|---:|
| equal-weight | 25.42% | 37.17% | 0.690 | 0.897 | -45.36% |
| value-chain tilt | 23.71% | 36.29% | 0.659 | 0.851 | -45.46% |
| XLI | 7.58% | 27.63% | 0.258 | — | -42.33% |
| GRID | 17.16% | 30.12% | 0.545 | — | -40.56% |
| PAVE | 14.96% | 32.70% | 0.469 | — | -44.08% |

In the prior regime the tilt **loses** to equal-weight on every metric (CAGR -1.71pp, Sharpe
-0.031) — overweighting makers and underweighting contractors was mildly harmful in 2020–2022.
Panel limitations carried from Step 1: **GEV absent** the whole panel (no vendor price history
before 2024-03), **FLNC** only from its 2021-10 IPO, **VRT** early quotes are the predecessor
SPAC. The §4.1 margin signal is inert here too, and §4.2 coverage never fires (no in-window
backlog rows before 2022-12), so the "tilt" here is the pure static ×1.25/×0.75 split.

CLI note: `python -m grid_equipment_basket --construction value-chain-tilt --prior-regime`
**raises** `ValueError: no data in peak window ('2024-07-01', '2024-12-31')` — `value_chain_report`
unconditionally builds the Gate-2 drawdown episode from the hard-coded 2024-H2 `DRAWDOWN_PEAK_WINDOW`,
which has no data in a 2020–2022 run. The panel numbers above were produced by calling
`simulate_basket` directly for the two long books (no code change). Logged as a next-step fix in
§9 / CLAUDE.md.

## Drop-winners robustness — drop VRT, GEV from makers (spec §7.1), primary window

makers = **ETN, HUBB, NVT**; contractors unchanged.

| Series | CAGR | Vol | Sharpe | Sortino | MaxDD |
|---|---:|---:|---:|---:|---:|
| equal-weight | 42.26% | 35.22% | 1.065 | 1.413 | -43.04% |
| value-chain tilt | 41.40% | 33.42% | 1.086 | 1.420 | -41.40% |

Re-applying the Gate 1 test on this reduced universe: tilt Sharpe 1.086 > equal-weight 1.065, but
tilt CAGR 41.40% **<** equal-weight 42.26% → **Gate 1 would FAIL**. "Makers beat contractors on
return" does **not** survive removing the two largest winners; only a thin Sharpe edge survives,
and it comes entirely from the tilt running ~1.8pp lower vol (concentrating away from the four
contractors), not from stock selection. The pair standalone over the same reduced universe is
CAGR -14.07% / Sharpe -0.41 / max DD -61.3%.

## Parameter plateau (spec §5, §8 — reported, not optimized)

`backtest.value_chain_report` re-run with the four `config` constants monkeypatched in an inline
script (no search added to the module). Primary window; Gate-1 column is the tilt-vs-equal-weight
pass/fail.

| Config | CAGR | Vol | Sharpe | MaxDD | Gate 1 |
|---|---:|---:|---:|---:|:--:|
| equal-weight (reference) | 59.77% | 36.48% | 1.360 | -41.35% | — |
| base 1.20/0.80, within 1.10/0.90 | 62.98% | 36.03% | 1.428 | -41.17% | PASS |
| **base 1.25/0.75, within 1.10/0.90 (shipped)** | **63.66%** | **35.97%** | **1.441** | **-41.15%** | **PASS** |
| base 1.30/0.70, within 1.10/0.90 | 64.34% | 35.92% | 1.454 | -41.13% | PASS |
| base 1.25/0.75, within 1.05/0.95 | 63.42% | 35.96% | 1.438 | -41.15% | PASS |
| base 1.25/0.75, within 1.15/0.85 | 63.91% | 35.98% | 1.445 | -41.15% | PASS |
| base 1.25/0.75, within 1.00/1.00 (signal off) | 63.18% | 35.95% | 1.434 | -41.15% | PASS |
| base 1.00/1.00 (= equal-weight, sanity) | 59.77% | 36.48% | 1.360 | -41.35% | FAIL |

The surface is flat and monotone in the base split (Sharpe 1.428 → 1.454 as the split widens
1.20/0.80 → 1.30/0.70) and almost independent of the within-bucket multiplier (1.434 with the
signal off vs 1.441 shipped). Nothing was tuned; the result is not sensitive to the exact
constants, and it is not coming from the fundamental signal.

## Caveats (spec §8 — carried + new)

- **~3.56-year / ~43-monthly-obs window, one macro regime.** Standard error on an annualized
  Sharpe from this sample is ~±0.5 (≈ ±1.0 at 95%). Every gate comparison and every gap above is
  a directional point estimate, **not** a significant result. The Gate 1 Sharpe gap (+0.081) and
  CAGR gap (+3.89pp) are both well inside that noise.
- **GEV** has vendor price history only from 2024-03-27 — it covers only the back ~60% of the
  primary window and none of the prior-regime panel.
- **Headline survivorship / hindsight bias:** the 9-name universe was chosen in 2026 knowing VRT,
  GEV, PWR won.
- **The 5/4 bucket split overlaps the handoff's hindsight-flagged hand-split.** Mitigations:
  business-model split rule frozen pre-backtest; prior-regime panel (tilt loses); drop-winners
  pass (Gate 1 fails); PWR — a large winner over the window — sits in the *short* leg, so shorting
  it cost money. The drop-winners failure means this mitigation is only partly reassuring: much of
  Gate 1's pass rides on VRT and GEV being in the overweight bucket.
- **The §4.1 gross-margin signal never fired** (every rebalance, both windows) — XBRL discrete-Q4
  facts are not tagged, and the span guard rejects the resulting 5-quarter-gap series. The shipped
  tilt is effectively the static bucket split plus a §4.2 backlog-coverage nudge on ETN/FLNC/GEV/NVT
  only. Gross-margin dilution for ETN/GEV/PRIM from non-grid segments is therefore moot for now;
  the operating-margin variants (§9) are logged for a later spec.
- **Backlog-coverage (§4.2) is live for 4 of 9 names and only from 2024-02** — HUBB, VRT, PWR,
  MYRG, PRIM take the base bucket multiplier at every rebalance.
- **Short leg is not costless.** Pair results are gross; borrow-cost sensitivity stated above
  (11.98 pp/yr to erase the carry edge; MYRG thin, FLNC hard to borrow).
- **The pair adds fitted-looking degrees of freedom** (base split, within-bucket multipliers, three
  conditional-short thresholds). The plateau is reported; nothing was optimized.

## Decision

**Gate 1 — long-only tilt: PASS (as written), adopt with a health warning.** Over the primary
window the value-chain tilt beats the Step 1 equal-weight basket on both Sharpe (1.441 vs 1.360)
and CAGR (63.66% vs 59.77%), so by the pre-registered rule it clears the gate and becomes an
available long construction (`--construction value-chain-tilt`). But the evidence that this is an
*edge* rather than a relabeled bet on the 2026-vintage winners is weak: the gaps are inside ±0.5
Sharpe noise; the pass is reproduced with the fundamental signal switched off (it is the static
maker/contractor split); it does **not** survive dropping VRT and GEV; and the prior-regime panel
has the tilt losing. Recommended framing: keep **equal-weight** as the headline Step 1 basket;
offer the tilt as a documented alternative, not as a demonstrated skill.

**Gate 2 — the pair as a hedge: FAIL → use the conditional QQQ short.** The pair 30% overlay did
not protect the spec §7.3 episode (-41.04% vs -41.35% naked); the conditional QQQ short did
(-37.34%) at a carry cost of -0.53%/yr. Per spec §7.4 the recommendation on Gate-2 failure is
"use the conditional QQQ short" (or "size down, no hedge") — both acceptable. Adopt the
**conditional QQQ short** as the intra-theme drawdown hedge. Note separately that the pair has
positive standalone carry (11.45%) and, as a 30% overlay, *raised* return and Sharpe
(68.15% / 1.514) over this window — it is interesting as a return sleeve, but it is not a hedge,
and Gate 2 tests it as a hedge.
