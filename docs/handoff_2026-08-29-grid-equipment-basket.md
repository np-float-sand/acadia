# Handoff — Grid Equipment Suppliers Basket (2026-08-29)

## TL;DR

We have a **solid long book** and a validated construction process. What it does **not** yet have
is (1) a differentiated edge beyond "own the theme" and (2) a working hedge / short leg. The theme
is already well-documented in the market and the marquee names are partly crowded, so the next
phase of work is **innovation + hedge**, not more polishing of the long book.

Branch: `grid-equipment-basket` (13 commits off `main`, final whole-branch review clean with minor
parked items — **not merged**, integration path not yet chosen).
Spec: `docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md`
Plan: `docs/superpowers/plans/2026-08-28-grid-equipment-basket.md`
Results: `docs/grid-equipment-basket-step1-results.md`
Module: `grid_equipment_basket/` (config, prices, basket, backtest, backlog_data, CLI, README, tests 41/41)
SDD ledger (all rulings): `.superpowers/sdd/2026-08-28-grid-equipment-basket/progress.md`

## What we have — the long book

**Universe (9, fixed, business-description test only):** ETN, HUBB, GEV, VRT, PWR, MYRG, NVT, FLNC, PRIM
— grid / data-center electrical equipment and power-infrastructure EPC. Verified against latest 10-Ks
(`grid_equipment_basket/candidate_research.md`). GEV enters at its 2024-04 listing; the rest from 2023-01.

**Core strategy — equal-weight, quarterly rebalance, 25% name cap, long-only, total return.**
Cleared the pre-registered decision gate on the primary window (2023-01-01 → 2026-07-31):

| | Basket (equal-weight) | XLI | GRID | PAVE | SPY |
|---|---|---|---|---|---|
| CAGR | **59.8%** | 20.2% | 23.8% | 24.5% | 22.4% |
| Sharpe (rf 4%) | **1.36** | 0.95 | 0.96 | 0.97 | 1.15 |
| Max drawdown | −41.3% | −18.5% | −20.8% | −26.2% | −18.8% |

Beat broad industrials **and** the off-the-shelf thematic ETFs (GRID/PAVE) on both CAGR and Sharpe.
Prior-regime panel (2020–2022) also beat XLI (25% vs 8% CAGR) but with a −45% drawdown.

### Recommended live overlay (all long-only — de-levering, not shorting)

Two rules on top of the basket, then an optional third:

1. **Trend gate** — hold the basket (or each name) only while price is above its ~100-day moving
   average; otherwise that slice sits in cash / T-bills. Re-checked monthly.
2. **Volatility target ~20%** — scale exposure inversely to trailing 20-day realized vol, capped at
   ~1.5x leverage, slack in T-bills. Update the scalar monthly (daily over-trades it).
3. **(Last resort) ~10–15% gold sleeve** — the only *additive* diversifier that helped without
   costing carry over this sample.

Backtested effect (in-sample, primary window):

| Config | CAGR | Sharpe | Max DD |
|---|---|---|---|
| Basket only | 59.8% | 1.36 | −41.3% |
| + trend gate | 63.3% | 1.46 | −28.3% |
| + trend gate + monthly vol-target | ~54% | **1.85** | **−18.5%** |
| + 15% gold on top | ~50% | **1.95** | **−15.4%** |

Mechanics: ~1.5 name in/out changes per month, avg ~5% cash from the gate, vol-target averages
~0.70x exposure (min 0.24x in the worst spike). Parameters sit on a **plateau** (MA 50–150, vol
target 15–25% all give Sharpe 1.7–2.0), not a knife-edge — reassuring on overfit.

**Important honesty note on the overlay:** it *underperforms* buy-and-hold in every calendar year
except 2025. Its entire edge is turning the one −42% event (Nov 2024 → Apr 2025, the DeepSeek
efficiency scare) into ~−9%. In a 3.5-year window without a −40% event it would read as pure drag.
Trend + vol rules generalize better than a fitted stock-selection signal, but do not expect Sharpe
1.9 live. Every added layer is a tuned choice — the robust core is only "hold less when volatility
is high and the trend is down."

## Why this is not yet a finished product

- **The theme is already recognized.** "Power / the grid is the bottleneck for AI" has been a
  mainstream sell-side and financial-press narrative since 2023–24. This is not an undiscovered idea.
- **Partly crowded.** Marquee names (VRT, GEV, ETN, PWR) have re-rated hard and sit in most
  thematic / AI-infrastructure funds. Median basket-name daily dollar volume is up **~10x** since
  2022 (Vertiv 34x, the GRID ETF 32x) vs SMH ~4x — trading interest has surged faster here than in
  semis. Less crowded than NVDA/semis, but no longer contrarian.
- **The only "active" decision is the universe**, and it was made in 2026 knowing which names won.
  Survivorship / hindsight bias is the headline risk (documented at full strength in the results doc).
- **Single regime.** ~3.5 years, one macro environment (AI bull with sharp corrections).
- **No edge inside the theme.** The disclosed-backlog-growth tilt we built and tested **failed** —
  it over-weighted the weak performer (FLNC) and lost to equal-weight; a descriptive backlog-vs-
  forward-return check was mixed-to-negative (4 of 6 names negatively correlated). So the "data
  advantage" thesis (order backlog is quarterly-disclosed) did not convert to a usable signal.

## Open need 1 — INNOVATION (a differentiated edge)

The long book currently earns thematic beta. To justify active fees / a real allocation it needs an
edge that isn't just "we picked the theme early." Leads, roughly in order of promise:

- **Scarcity / lead-time signal, not growth.** We tested backlog *growth* rank (failed). Untested:
  backlog *coverage* (years of revenue in backlog) rising, sustained book-to-bill > 1, or explicit
  lead-time / price-escalation language in filings — a "pricing power" signal distinct from growth.
  Transformer/switchgear/HVDC lead times are the real bottleneck; commodity electrical is not.
- **Within-theme sub-structure.** "Constrained equipment" (ETN/GEV/HUBB/NVT/VRT — long-lead-time
  gear) did Sharpe 1.55 stand-alone vs the EPC/labor names (PWR/MYRG/PRIM) at 1.20. But that's 5
  names, a *worse* max drawdown, and probably hindsight. Worth a rigorous look with an ex-ante rule
  (e.g. segment-revenue mix from 10-Ks) rather than a hand-split.
- **Cross-sectional factor (spec §8, Phase 2 — not built).** Once there are 8+ names with several
  years of comparable segment-level backlog, rank by backlog-*surprise* (actual vs a simple
  expectation) instead of raw growth. Needs its own spec. Do NOT reintroduce z-scoring / build_factor
  machinery into the basket module.
- **Earnings-estimate revisions / analyst breadth** — never tested here; a standard momentum-of-
  fundamentals signal that historically adds on trending themes.
- **Quality / balance-sheet screen.** We deliberately kept FLNC (going-concern-adjacent) for
  anti-cherry-picking. A live book could weight by leverage / margin stability instead.
- **Widen the universe** for diversification and to reduce single-name crowding: HVDC cable
  (Prysmian/Nexans — foreign, need ADR/liquidity check), backup & distributed power (GNRC, CMI),
  transformer pure-plays, nuclear / SMR enablers, electrical distribution (Rexel/Sonepar-listed peers).
  Only helps if adds aren't ~0.9 correlated to the existing names.
- **Per-name momentum** (monthly top-5 by 6-month return) pushed CAGR to 73% but with no drawdown
  control — pair it with the vol overlay and check whether the selection adds anything over
  equal-weight after costs.

## Open need 2 — HEDGE / short leg

**We could not find a structural short.** Record of what was tried so the next person doesn't repeat it:

| Candidate short | Result | Why it fails |
|---|---|---|
| Short SMH (semis) | Sharpe → −0.02 (dollar-neutral) | +0.74 corr, semis *outperformed* the basket (61% CAGR) — shorting it bleeds carry |
| Short QQQ / SPY (broad beta) | Sharpe 1.36 → 1.15 (0.5x) | +0.68 corr, rose over the window — same carry problem |
| Short XLU | Sharpe → 1.08 | low corr (0.26), positive drift — weak hedge, negative carry |
| Short solar (TAN) | Sharpe 1.69 → ~2.0, DD −24% → −21% | **only "works" via solar's unrelated 2023–25 bear market**; +0.52 corr so it does partially hedge, but it's a tactical "solar stays weak" bet that reverses if solar recovers |
| Short energy-intensive industrials (AA, OLN, steel, chem) | mild DD help, lower Sharpe | all +0.3–0.45 corr, fall *with* the basket — no anti-correlation despite the power-cost-competition thesis |
| Short crypto miners (MARA/RIOT/CLSK) | Sharpe → 0.42 | they pivoted to AI hosting and ripped +40–72% — now longs, not losers |
| Short homebuilders (XHB) | Sharpe → 1.43–1.62 | +0.48 corr, still rose (+17%) — transformer-crowd-out thesis real but equities didn't cooperate |
| Systematic QQQ put-spreads | Sharpe 1.85 → 1.53 | ~8%/yr premium drag overwhelmed payoffs in a bull-with-dips regime |
| Long Treasuries / managed futures on top | modest, regime-dependent | bonds hurt in the Q4-2023 rate-spike drawdown; DBMF weak in this trend cycle |

**Structural finding:** every candidate that hedges the drawdown is positively correlated equity
beta that *also rose* over the window, so shorting it costs carry and cuts Sharpe. There is no stock
that structurally *wins* when the data-center trade *loses* — the "theme losers" (solar squeezed out
of interconnection queues, manufacturers facing power-cost inflation, homebuilders crowded out of
the transformer supply chain) are directionally real stories but produce correlated risk assets, not
anti-correlated ones.

**When the trade does badly:** an expected slowdown in hyperscaler AI capex — from weak AI
monetization, an efficiency breakthrough that cuts compute/power intensity (the Jan-2025 case), a
rate/financial-conditions shock, or a recession — de-rates the whole picks-and-shovels chain at once.
The only things reliably up across *all* those triggers were long-vol, anti-beta (BTAL), inverse
equity, and cash; bonds/gold/defensives only help in the growth-scare / rate-cut version.

**Where to take the hedge work next:**
- Is there a genuine long/short here at all, or is the honest answer "run it smaller / de-levered"?
  The overlay (gate + vol-target) is de-levering, not hedging — cheap and robust, but it forgoes
  upside and isn't market-neutral.
- A **conditional** beta short (put on a QQQ/SMH short only when the trend gate is "off" and vol is
  elevated) instead of a static one — may keep more upside than always-on. Not yet tested.
- **Options** structured for cheapness: put *spreads* financed by a call overwrite on the
  strongest-momentum name, or a wide collar — accepting a capped upside in exchange for a hard floor.
  Needs an options data source to backtest properly.
- A dedicated **"unprepared utilities" short** (the original `grid_resilience` thesis) as the demand-
  side pair — different short universe than XLU. Note `grid_resilience` already found the utility
  short has no standalone edge.

## Suggested next steps

1. Decide the branch's fate (merge / PR / shelve) — it's review-clean and self-contained.
2. Pick ONE innovation lead (scarcity/lead-time signal or the sub-structure split look most promising)
   and spec it — same gated, pre-registered, honest-caveats discipline as this project.
3. In parallel, test the **conditional** beta hedge and, if an options feed is available, a
   put-spread-plus-overwrite structure. Treat "no viable hedge, size down instead" as an acceptable
   conclusion if the evidence says so.
4. Keep the survivorship caveat front-and-centre in any PM-facing material — this is a hindsight-
   selected, single-regime, partly-crowded thematic long, strong as that is.
