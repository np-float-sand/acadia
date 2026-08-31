# Results — Grid-Congestion Regime Signal (Layer 2)

**Status:** BUILT. Pre-registered gate **not passed** (every rung beats layer-1-only on Sharpe but
not on primary-window Calmar, 2.16 vs 2.32). **ADOPTED as the recommended overlay by PM decision
(2026-08-31)** — see §1a. Shipped config = ladder **rung 1** (`grid_regime.shipped_config()`),
`config.REGIME_ENABLED = True`.
**Date:** 2026-08-31
**Spec:** `docs/superpowers/specs/2026-08-31-grid-regime-layer2-design.md`
**Module:** `grid_equipment_basket/grid_regime.py` (+ `overlay.py` seam, `--overlay-l2`).
**Run:** `python -m grid_equipment_basket --overlay-l2` — zero network I/O, all data from the
`grid_resilience` parquet cache (PJM zone LMP + zonal load 2018-01→2025-12; ERCOT settlement-point
LMP for rung 6).

## 1a. Adoption decision (post-hoc, deliberate)

The pre-registered gate (beat layer-1-only on Sharpe **and** Calmar on **both** windows, on a
plateau) was **not met** — see §2/§3. The basket owner reviewed the result and adopted rung 1 as
the recommended overlay anyway. Rationale, recorded honestly as a judgment call rather than a gate
pass:

- **Generalisation.** Rung 1 is the only rule across this entire line of work (layer 1, five
  cross-sectional attempts, this ladder) that is Sharpe-positive on *both* the 2023–25 and 2020–22
  regimes (0.61 prior vs layer-1's 0.20). It fixes layer 1's documented failure mode — the price
  gate overfits the one DeepSeek drawdown and hurts on the prior window.
- **Differentiator.** The input is measured PJM transmission congestion in the data-center zones —
  physical, non-price, non-consensus — which is the part of the thesis that is not just crowded
  picks-and-shovels beta.
- **Cost accepted.** ~5 percentage points deeper primary-window drawdown (−23% vs −18%) and a
  slightly worse Calmar. In a future AI-capex scare where power demand has not yet rolled over,
  layer 2 will ride it down further than layer 1 would.

This does not change the evidentiary record below: the strict gate did not pass, the sample is one
macro cycle, and the prior-window "win" predates real data-center congestion (§5).

---

## 1. One-paragraph verdict

The grid-congestion regime signal (zonal transmission-congestion $ + reserve-tightness in the
DC-heavy PJM zones, trailing-3y z-scored, → a monthly basket-exposure multiplier) does **not**
strictly beat the layer-1 price trend gate under the pre-registered gate: **no rung, and no rung's
parameter neighbours, cleared G1** (Sharpe beats layer-1-only, Calmar does not). The reason is
specific and informative: the congestion signal and the price gate are **complementary, not
substitutes**. The congestion signal recovers most of the prior-window (2020–2022) Sharpe that the
price gate destroys (0.61 vs 0.20), but during the one clean equity-led drawdown in the primary
window — the Nov-2024→Apr-2025 "DeepSeek" scare — physical congestion was genuinely *high*, so the
regime signal stayed levered in (multiplier 1.0–1.25 every month Jan–May 2025) and took a −23%
drawdown where the price MA-cross cut it to −18%. Better Sharpe on both windows, worse
primary-window Calmar. Per §1a the basket owner **adopted rung 1 as the recommended overlay**
(`config.REGIME_ENABLED = True`), on the generalisation + differentiator argument, accepting the
deeper drawdown; layer 1 stays available via `--overlay`.

---

## 2. The ladder (live)

Signal z-scored from 2018-01-01 (so the primary window sees a fully-warm signal). Basket returns
evaluated on **primary 2023-01-01 → 2025-12-31** and **prior 2020-01-01 → 2022-12-31** (the same
prior panel layer 1 uses). Gate: **G1** beat layer-1-only on Sharpe AND Calmar, primary; **G2**
same, prior; **G3** no bigger calendar-year drag than layer 1, both windows; **G4** every parameter
neighbour also clears G1 & G2.

|  | primary CAGR | Shrp | Clmr | MaxDD | | prior CAGR | Shrp | Clmr | MaxDD | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| buy & hold | 60.4% | 1.44 | 1.46 | −41.3% | | 25.4% | 0.69 | 0.56 | −45.4% | — |
| vol-target only | 49.0% | 1.66 | 2.19 | −22.3% | | 12.4% | 0.42 | 0.30 | −41.5% | — |
| **layer-1 only** (trend gate + VT) | 42.7% | **1.58** | **2.32** | **−18.4%** | | 6.1% | **0.20** | **0.15** | −41.5% | *baseline* |
| 1: discrete, PJM-4, congestion-only | 50.3% | 1.64 | 2.16 | −23.3% | | 17.5% | 0.61 | 0.54 | −32.4% | **FAIL** (G1: Calmar) |
| 2: + reserve-tightness | 51.6% | 1.72 | 2.22 | −23.3% | | 12.3% | 0.42 | 0.30 | −41.5% | **FAIL** (G1: Calmar) |
| 3: continuous multiplier | 49.5% | 1.64 | 2.11 | −23.5% | | 14.1% | 0.47 | 0.36 | −38.6% | **FAIL** (G1: Calmar) |
| 4: widen zones to 6 | 49.7% | 1.65 | 2.18 | −22.8% | | 14.2% | 0.47 | 0.37 | −38.0% | **FAIL** (G1: Calmar) |
| 5: load-weighted zones | 49.6% | 1.66 | 2.14 | −23.1% | | 13.9% | 0.46 | 0.36 | −38.1% | **FAIL** (G1: Calmar) |
| 6: + ERCOT West spread proxy | 49.6% | 1.64 | 2.10 | −23.7% | | 8.3% | 0.29 | 0.17 | −50.2% | **FAIL** (G1: Calmar; also *marginal*) |
| 7: + RT/DA spread sub-signal | — | | | | | | | | | DEFERRED |

Every active rung: **G1 fail** (Sharpe beats layer-1-only, Calmar does not), **G2 pass**,
**G3 pass**, **plateau 0/N** (not knife-edge — the neighbours don't clear either).
`stopped_at: NONE`.

Rung 7 (RT/DA real-time-vs-day-ahead spread) is registered in `config.REGIME_LADDER` but not run:
it needs a `fetch_lmp_rt()` in `grid_resilience/data/grid_data.py` and a live `gridstatus`
availability probe over 2018–2025 (spec §7). The `_combine_subsignals` seam is in place for it.

---

## 3. Why it fails — the mechanism

**Primary window (2023–2025).** Every rung earns a *higher Sharpe* than layer-1-only (1.64–1.72 vs
1.58) and a *higher CAGR* (~50% vs 42.7%) — it keeps more of the basket's upside because the
congestion signal spends most months neutral-to-leaning-in (avg multiplier ~1.02–1.05, "stepped
back" only 4–47% of days depending on rung). But it takes a *deeper drawdown* (−23% vs −18.4%),
so **Calmar is worse** (2.1–2.2 vs 2.32). The decisive episode: the DeepSeek drawdown. Rung-1
monthly multiplier over that window —

```
2024-10  1.00   2024-11  1.00   2024-12  1.00   2025-01  1.00
2025-02  1.25   2025-03  1.00   2025-04  1.00   2025-05  1.25
```

— never stepped back, and *leaned in* in Feb and May 2025. Physical PJM congestion in the DC-heavy
zones was genuinely elevated in early 2025 (COMED/PPL/DOM all high — see the descriptive pass in the
brainstorm), so the signal correctly read "grid bottleneck intact" at the exact moment the equities
sold off on an AI-efficiency scare. The price MA-cross, which knows nothing about the grid, caught
that drawdown; the grid signal, working as designed, did not. **The stocks and the physical grid
disagreed, and for that one episode the price was the better exposure signal.**

**Prior window (2020–2022).** Here the regime signal *wins* — rung 1 lifts Sharpe from layer-1's
0.20 to 0.61 (near buy-and-hold's 0.69) and cuts MaxDD from −45.4% to −32.4%. This is exactly the
window where the handoff documented that layer-1's price gate *hurts* (it overfit the DeepSeek
episode). The congestion signal de-risked usefully through the 2022 energy-price/rate stress
(multiplier floor 0.60). **So G2 passes for every rung — the physical signal generalises where the
price gate doesn't.**

**Net:** the gate asks one signal to beat the other on both windows. Neither does. The price gate
owns the clean equity-led drawdown; the congestion signal owns the messier macro regime.

---

## 4. Rung-by-rung notes

- **Rung 1** (discrete, PJM-4, congestion-only) is the **best generaliser**: positive contribution
  on *both* windows (prior Sharpe 0.61, primary Sharpe 1.64), unlike layer 1 (−0.49 Sharpe swing
  between windows) or buy-and-hold. It just can't beat layer 1 on primary-window Calmar.
- **Rung 2** (+reserve-tightness at 0.3 weight) *neutralises* the signal — the blended composite
  rarely crosses ±0.5, so on the prior window it collapses back to vol-target-only (Sharpe 0.42,
  MaxDD −41.5%, avg multiplier 1.02, "stepped back" 4% of days). Reserve-tightness (peak load /
  trailing-p99 peak load, per zone) is too slow-moving to add regime information here.
- **Rungs 3–5** (continuous, wider zones, load-weighted) are near-identical to each other
  (primary Sharpe ~1.65, prior ~0.47) and all fail the same way. Continuous mode trades more
  (71 flips vs 30) for no gain. Load-weighting the zones vs equal-weighting them changes nothing
  material.
- **Rung 6** (+ERCOT LZ_WEST−LZ_NORTH spread proxy, equal-weight with the PJM composite) is the
  *worst* on the prior window (Sharpe 0.29, MaxDD −50.2%) and is the only rung flagged **marginal**.
  Mixing a true congestion-component signal (PJM) with a raw inter-zonal spread proxy (ERCOT, no
  component data cached) adds noise, as the spec's apples/oranges caveat anticipated.

---

## 5. Honesty caveats (carried from the spec)

- **Sample:** one macro cycle; a monthly decision ⇒ ~36 primary and ~24 *active* prior monthly
  observations (post 3y z-score warm-up). The plateau + two-window rules are the only defense
  against fitting, and here they simply confirm the negative.
- **Warm-up:** the 756-day z-score means the prior-window test is effectively **2021–2022**, not
  2020–2022.
- **The prior window is not an AI-buildout test.** Congestion in 2021–2022 was weather /
  wind-transmission / gas-price driven, not data-center driven. Rung 1 clearing G2 means the signal
  captures a generic "physical grid stress ⇒ de-risk grid-equipment equities" relationship — useful,
  but *not* evidence that a data-center-congestion regime read works, because there was no
  data-center congestion regime yet in the prior window.
- **Proposal D** already found monthly ΔGSI barely co-moves with equity aggregates. This result is
  consistent: the congestion signal doesn't *hurt* (G3 passes everywhere) but it doesn't beat a
  price rule that's tuned to the one drawdown in the evaluation window.
- Layer 2 is one exposure decision per month — low breadth by construction. The gate failing means
  the "physical-data reason to set the weight is worth paying for" argument is not supported on this
  sample.

---

## 6. What we keep / what's next

**Shipped live (merged, `config.REGIME_ENABLED = True`):**
- **Rung 1** as the recommended overlay: `grid_regime.shipped_config()` (discrete, PJM-4 DC-heavy
  zones, congestion only), applied as
  `overlay.apply_overlay_l2(basket_returns, grid_regime.live_multiplier(...))`.
  `live_multiplier` on real data 2018–2025: avg exposure ×1.03, leaned-in 19% of days,
  stepped-back 4%; by year — 2020 ×0.95, 2021 ×1.08, 2022 ×1.10, 2023–24 ×0.99, 2025 ×1.13.
- Layer 1 stays available and unchanged via `--overlay`.

**Also merged:**
- `grid_regime.py` — signal, gate (`gate_check` / `final_verdict`), ladder runner
  (`regime_report` / `regime_table`), ERCOT spread proxy, RT/DA `_combine_subsignals` hook.
- `overlay.py` layer-2 seam (`regime_exposure`, `apply_overlay_l2`, `exposure_series_l2`) — layer-1
  untouched.
- `--overlay-l2` CLI (runs the full evidence ladder) + `config.REGIME_*` frozen constants.
- 40 new tests (7 overlay seam + 17 signal incl. shipped-config + 11 gate + 4 ladder runner + 1 CLI).

**Do NOT:**
- re-tune rung 1 or re-run the ladder hoping for a strict pass — it's pre-registered and closed;
  adoption was a documented judgment call, not a moved goalpost;
- read the 2020–2022 G2 pass as evidence a *data-center* congestion regime works — there was no
  such regime then (§5);
- add rung 7 (RT/DA) as a rescue without its own spec.

**Possible follow-ups (each its own spec):**
1. **Combine, don't substitute.** Pre-register: layer-1 trend gate AND a congestion *veto in one
   direction only* ("if physical congestion is easing hard, step back even if price is still above
   its MA") — the spec's option (c), untried because scope said "replace the gate." This is the
   natural next step given the complementary finding.
2. **A longer / out-of-sample window.** The Calmar miss is decided by one episode (DeepSeek).
   Re-running once 2026+ data is cached adds the first genuinely out-of-sample observations and is
   the cleanest way to confirm or retire the adoption.

---

## 7. Option A — DC-zone congestion RELATIVE to the rest of PJM (2026-08-31) — FAILED

**Motivation.** The absolute rung-1 signal's "step back" days cluster in three episodes — Mar–Apr
2020, Mar 2023, Mar 2024 — all broad risk-off / shoulder-season windows, none data-center-driven.
The absolute signal can't tell "the DC bottleneck relaxed" from "total electricity demand fell." A
**relative** signal — z-score of `mean|congestion $|(DOM,AEP,COMED,PPL) − mean|congestion $|(rest of
PJM, 16 zones)` — nets out any system-wide demand swing and isolates congestion *concentrating* in
the data-center zones. `grid_regime.relative_regime_composite` / `relative_signal_report`
(`_rest_of_pjm_congestion`, `REGIME_DC_ZONES`), 5 tests.

**Pre-registered bar:** beat layer-1-only on Sharpe AND Calmar on both windows, on a threshold
plateau, AND shave the prior-window drawdown vs the absolute signal (the COVID artifact it targets).

**Result — FAIL, and it settles the earlier open question.**

| rung-1 variant | primary Sharpe / Calmar / MaxDD | prior Sharpe / Calmar / MaxDD |
|---|---|---|
| layer-1 only (baseline) | 1.58 / 2.32 / −18.4% | 0.20 / 0.15 / −41.5% |
| **absolute** (shipped) | 1.64 / 2.16 / −23.3% | **0.61 / 0.54 / −32.4%** |
| **relative** (option A) | 1.68 / 2.16 / −24.2% | **0.21 / 0.15 / −41.5%** |

- G1 still fails (Sharpe up, Calmar 2.16 < 2.32). Plateau 0/2. `fixes_prior_dd = False`
  (−41.5% vs the absolute's −32.4% — it made the prior drawdown *worse*).
- **The prior-window edge was the common-mode component.** Removing it drops prior Sharpe 0.61 → 0.21
  (≈ layer-1-only / vol-target-only). So the absolute signal's 2020–22 "win" **was** the COVID
  demand-collapse coincidence, exactly as suspected. The data-center-specific signal has ~no timing
  edge in this sample: there was no DC-congestion regime in 2020–22 to detect, and in 2023–25 it
  still trades drawdown for upside.

**Takeaway.** The relative signal is the *honest* construction and it shows the timing edge is not
there once the macro/seasonal artifact is stripped. It does not replace the shipped absolute rung 1
(which is retained as a disclosed judgment call, §1a), but it caps how much weight that adoption can
bear. `relative_regime_composite` is kept for the option-B pair test and for a future
seasonally-adjusted variant.

---

## 8. Option B — long basket / short rest-of-PJM utilities, hedge tilted by the signal (2026-08-31) — FAILED

**Construction.** Long the 9-name equipment basket; short a rest-of-PJM utilities proxy (XLU, and a
6-name D/AEP/EXC/PPL/PEG/FE basket); hedge ratio `k` set by the DC-minus-rest congestion signal —
`k=1.0` (full short) when DC congestion is concentrating, `k=0` when it's easing, `k=0.5` between.
`grid_regime.pair_signal_report` / `pair_signal_table` (`_hedge_ratio`, `_PJM_UTILITIES`), 2 tests.

**Pre-registered gate:** the signal-tilted XLU pair must (1) beat the static `−1.0·XLU` pair on
Sharpe on both windows, (2) be *spread-informative* — the daily `basket − XLU` spread must be
larger in "signal-high" months than "signal-low" months and positive when high, and (3) beat
layer-1-only on full-span Sharpe.

**Result — FAIL on the check that matters.**

| | primary Sharpe | prior Sharpe | full Sharpe / MaxDD |
|---|---|---|---|
| shipped overlay (absolute) | 1.64 | 0.61 | 1.11 / −32.4% |
| static pair (−1.0·XLU) | 1.15 | 0.51 | 0.84 / −38.4% |
| tilted pair (XLU) | 1.33 | 0.79 | 1.07 / −38.7% |
| tilted pair (util basket) | 1.26 | 0.86 | 1.07 / −42.6% |

- **beats-static: True.** **beats-L1-full: True.** **spread-informative: FALSE — and it's backwards.**
  `basket − XLU` averages **+7.3 bp/day when the signal says DC congestion is concentrating** vs
  **+18.2 bp/day when it says easing.** The signal does not predict when equipment beats utilities;
  if anything it's mildly anti-predictive on that spread.
- The tilted pair's edge over the static pair comes from carrying a *lower average short* (avg
  `k` 0.53, so less carry drag from a utility sector that rose) plus timing noise — not from the
  congestion signal carrying cross-sectional information. The gate's spread-informative leg exists
  precisely to catch that, and it did.
- The tilted pair also doesn't beat what we already ship (full Sharpe 1.07 vs 1.11; primary 1.33 vs
  1.64) — shorting utilities costs carry, the same finding as the original hedge search.

**Combined takeaway for §7–§8.** Once the COVID/seasonal artifact is removed, there is **no
demonstrable data-center-congestion edge in this sample — neither timing (A) nor cross-sectional
(B)**. That is now 6 ladder rungs + 2 relative-signal variants, all short of their pre-registered
bars. The shipped absolute rung 1 stays adopted on the disclosed §1a judgment call, with these two
negatives as the ceiling on that call. Realistic next moves are unchanged: option (c) combine (not
substitute), a seasonally-adjusted relative signal, or wait for out-of-sample 2026+ data.
