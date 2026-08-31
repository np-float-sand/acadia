# Results — Grid-Congestion Regime Signal (Layer 2)

**Status:** BUILT + gate **FAILED** (pre-registered negative result).
**Date:** 2026-08-31
**Spec:** `docs/superpowers/specs/2026-08-31-grid-regime-layer2-design.md`
**Module:** `grid_equipment_basket/grid_regime.py` (+ `overlay.py` seam, `--overlay-l2`).
**Run:** `python -m grid_equipment_basket --overlay-l2` — zero network I/O, all data from the
`grid_resilience` parquet cache (PJM zone LMP + zonal load 2018-01→2025-12; ERCOT settlement-point
LMP for rung 6).

---

## 1. One-paragraph verdict

The grid-congestion regime signal (zonal transmission-congestion $ + reserve-tightness in the
DC-heavy PJM zones, trailing-3y z-scored, → a monthly basket-exposure multiplier) does **not** beat
the layer-1 price trend gate at setting basket exposure, under the pre-registered gate. It is not a
knife-edge miss — **no rung, and no rung's parameter neighbours, cleared G1**. The reason is
specific and informative: the congestion signal and the price gate are **complementary, not
substitutes**. The congestion signal recovers most of the prior-window (2020–2022) Sharpe that the
price gate destroys (0.61 vs 0.20), but during the one clean equity-led drawdown in the primary
window — the Nov-2024→Apr-2025 "DeepSeek" scare — physical congestion was genuinely *high*, so the
regime signal stayed levered in (multiplier 1.0–1.25 every month Jan–May 2025) and took a −23%
drawdown where the price MA-cross cut it to −18%. Better Sharpe, worse Calmar, so it fails the
"beat on Sharpe **and** Calmar, on **both** windows" bar. **Shipped recommendation: keep
layer-1-only** (trend gate + vol target), or run a smaller un-overlaid basket. `REGIME_*` config
stays present but unused; no `REGIME_ENABLED` flag is added.

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

**Keep (merged):**
- `grid_equipment_basket/grid_regime.py` — the signal, the gate (`gate_check` / `final_verdict`),
  the ladder runner (`regime_report` / `regime_table`), the ERCOT spread proxy, the RT/DA
  `_combine_subsignals` hook.
- `overlay.py` layer-2 seam (`regime_exposure`, `apply_overlay_l2`, `exposure_series_l2`) — layer-1
  untouched.
- `--overlay-l2` CLI + `config.REGIME_*` (present, unused, documented as a negative result).
- 37 new tests (7 overlay seam + 14 signal + 11 gate + 4 ladder runner + 1 CLI).

**Do NOT:**
- enable the regime multiplier live (no rung passed);
- re-run the ladder hoping for a pass — it's pre-registered and closed;
- add rung 7 (RT/DA) as a rescue attempt without its own spec — the failure mode here (physical
  signal disagrees with price on the decisive episode) is not obviously fixed by a *different*
  physical signal.

**Possible follow-ups (each its own spec, not a rerun of this one):**
1. **Combine, don't substitute.** The finding is that price and congestion are complementary. A
   test worth pre-registering: layer-1 trend gate AND a congestion *veto in one direction only*
   ("if physical congestion is easing hard, step back even if price is still above its MA") — i.e.
   the spec's option (c), which this build did not try because the scope answer was "replace the
   gate."
2. **Rung 1 as a standalone robustness overlay.** It's the only rule in this whole line of work
   that is Sharpe-positive on both windows. Not gate-passing, but a candidate for "run the basket
   at rung-1 exposure" as a lower-variance alternative to layer 1, judged on blended/robustness
   metrics rather than beat-layer-1.
3. **A longer / out-of-sample window.** The gate is decided by one episode (DeepSeek). Re-running
   once 2026+ data is cached would add the first genuinely out-of-sample observations.
