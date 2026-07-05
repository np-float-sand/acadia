# Design Spec — Business-Model-Aware Factor Architecture

**Date:** 2026-06-29  
**Branch:** res2  
**Status:** Approved for implementation

---

## Problem

The current factor ranks all utility stocks using a GSI stress beta — the stock's sensitivity to grid stress events measured by LMP spikes and congestion. This works for merchant generators (VST, NRG) because LMP flows directly to their earnings. It fails for rate-regulated T&D utilities (AEP, PPL, FE, D, PEG, EXC) because revenues are set by state rate cases, not LMP. Adding PJM (all regulated T&D) drops Sharpe from ~0.272 to ~-0.009 regardless of whether hub or zone LMPs are used.

**Root cause:** One signal architecture for two fundamentally different business models.

---

## Goal

Make the factor model business-model-aware so that:
- Merchant generators continue to use the existing GSI stress beta
- Regulated T&D utilities are scored on a regulated-utility signal (starting with ICR)
- Mixed utilities are blended proportionally

Test three blending architectures sequentially and pick the winner. Then (Phase 2, separate spec) optimise the regulated-utility signal itself (ICR → rate case calendar → load growth).

**Success threshold:** At least one architecture achieves Sharpe > 0.20 with PJM included, vs current ~-0.009.

---

## Phase 1 Scope (This Spec)

Three architectures, evaluated in order, using ICR as the regulated signal (already built).  
Phase 2 (signal optimisation) is deferred until the best architecture is identified.

---

## Shared Foundation — Business Model Tagging

Every ticker in `utility_node_map.py` gets two new fields:

- `business_model`: `"merchant"` | `"regulated"` | `"mixed"`
- `pass_through`: float 0.0–1.0 — fraction of revenue exposed to merchant/spot prices

### Ticker Assignments

| Ticker | ISO | business_model | pass_through | Rationale |
|---|---|---|---|---|
| VST | ERCOT | merchant | 1.00 | 100% competitive generation |
| NRG | ERCOT | merchant | 0.85 | Mostly competitive; some retail |
| ETR | MISO | mixed | 0.35 | Merchant nuclear in competitive markets |
| PEG | PJM | mixed | 0.25 | PSEG Nuclear competes in PJM capacity market |
| DTE | MISO | mixed | 0.15 | Some merchant midstream + gen |
| CNP | ERCOT | mixed | 0.15 | Mostly regulated T&D |
| EXC | PJM | mixed | 0.10 | Post-Constellation spinoff (2022), mostly regulated |
| D | PJM | regulated | 0.10 | Some generation but VA SCC dominates revenue |
| AEP | PJM | regulated | 0.05 | Predominantly regulated |
| PPL | PJM | regulated | 0.05 | Pure regulated |
| FE | PJM | regulated | 0.05 | Pure regulated |
| WEC | MISO | regulated | 0.05 | Pure regulated |
| CMS | MISO | regulated | 0.05 | Pure regulated |
| AEE | MISO | regulated | 0.05 | Pure regulated |
| PCG | CAISO | regulated | 0.05 | Pure regulated |
| EIX | CAISO | regulated | 0.05 | Pure regulated |
| XEL | SPP | regulated | 0.05 | Pure regulated |
| EVRG | SPP | regulated | 0.05 | Pure regulated |

**Note:** These assignments are initial estimates to be verified against 10-K revenue disclosures. `pass_through` values for mixed tickers in particular carry uncertainty and will be refined in Phase 2.

---

## Config Addition

`config.py`:
```python
from typing import Literal
BUSINESS_MODEL_ARCH: Literal["hard_switch", "revenue_mix", "dual_track"] = "hard_switch"
```

CLI flag in `main.py`:
```
--arch [hard-switch|revenue-mix|dual-track]   Architecture for blending merchant/regulated signals (default: hard-switch)
```

---

## Architecture 1 — Hard Switch

### Logic

Each ticker is routed to exactly one signal path based on `pass_through`:

- `pass_through >= 0.5` → **merchant path**: existing GSI stress beta (unchanged)
- `pass_through < 0.5` → **regulated path**: ICR score (promoted from 15% addon to primary signal)

With current assignments: VST, NRG → merchant. All others → regulated.

### Factor score construction

Both paths produce a raw score per ticker per period. The two populations are z-scored together cross-sectionally before ranking. No existing merchant-path logic changes.

### Trade-offs

- Simplest to implement and debug
- Binary: ETR (0.35) and DTE (0.15) are routed to the regulated path even though they have meaningful generation exposure
- EXC at 0.10 correctly goes regulated post-Constellation spinoff

---

## Architecture 2 — Revenue Mix Blend

### Logic

Every ticker uses both signals. Scores are blended by `pass_through`:

```
score_i = z(stress_beta_i) × pass_through_i + z(icr_score_i) × (1 - pass_through_i)
```

Both inputs are z-scored cross-sectionally across all tickers before blending.

### Trade-offs

- Continuous: no binary cutoff, mixed tickers handled proportionally
- NRG (0.85) picks up a small ICR component; PPL (0.05) picks up a tiny stress beta component — both appropriate
- ICR data quality is weaker pre-2022 (yfinance limitation); this dilutes the regulated-side signal in the backtest's earlier window

---

## Architecture 3 — Dual-Track with Independent Normalisation

### Logic

Same blend formula as Architecture 2, but each signal is z-scored within its natural peer group rather than across all 18 tickers together:

1. Compute `stress_beta` for all tickers; z-score it **within the merchant+mixed group** (`pass_through > 0`)
2. Compute `icr_score` for all tickers; z-score it **within the regulated+mixed group** (`pass_through < 1`)
3. Blend: `score_i = z_merchant(stress_beta_i) × pass_through_i + z_regulated(icr_score_i) × (1 - pass_through_i)`

The difference from Architecture 2: in arch 2, VST's stress beta is z-scored against all 18 tickers — including the 14 regulated utilities with near-zero stress betas — which inflates VST's relative rank. In arch 3, VST's beta is ranked only against other generators. Each signal is more meaningful within its own population. Exact subgroup boundary (the pass_through cutoff) is an implementation decision.

### Trade-offs

- Cross-sectional ranks are more meaningful within peer groups
- Most statistically defensible for cross-asset blending
- Adds one normalisation step; slightly harder to debug
- Mixed tickers appear in both normalisation groups, which is correct

---

## Files Changed

| File | Change |
|---|---|
| `grid_resilience/data/utility_node_map.py` | Add `business_model` and `pass_through` fields to all 18 tickers |
| `grid_resilience/config.py` | Add `BUSINESS_MODEL_ARCH` Literal enum, default `"hard_switch"` |
| `grid_resilience/signals/resilience_score.py` | Implement three architecture paths in `build_factor()`; ICR promoted from 15% addon to primary regulated-path signal |
| `grid_resilience/main.py` | Add `--arch` CLI flag; pass value through to `build_factor()` |
| `tests/signals/test_business_model_arch.py` | New: one test per architecture verifying routing logic and blend math |

No existing tests should break — the default config is `"hard_switch"` and the merchant signal path is the existing logic unchanged.

---

## Implementation Sequence

1. **Tag tickers** — add `business_model` + `pass_through` to `utility_node_map.py` (foundation; no logic changes)
2. **Config + CLI** — add `BUSINESS_MODEL_ARCH` and `--arch` flag
3. **Hard switch** — implement in `build_factor()`, write tests, run backtest, record results
4. **Revenue mix** — implement blend in `build_factor()`, run backtest, record results
5. **Dual-track** — implement independent normalisation, run backtest, record results
6. **Compare** — pick winner; commit; open Phase 2 spec for signal optimisation (ICR → rate case → load growth)

---

## Evaluation

For each architecture run:
```bash
python -m grid_resilience.main --no-plot --arch <arch>
```

Record into a comparison table:

| Architecture | Sharpe | Ann Ret | Max DD | IC@21d | IC t-stat |
|---|---|---|---|---|---|
| Baseline (no PJM) | ~0.272 | ~8.0% | ~-15.4% | 0.144 | 1.918 |
| Hard switch | **0.331** | **9.38%** | -19.53% | 0.1434 | 1.865 |
| Revenue mix | 0.185 | 6.64% | **-14.70%** | 0.1340 | 1.702 |
| Dual-track | 0.031 | 4.44% | -19.02% | 0.1295 | 1.652 |

**Success threshold:** Sharpe > 0.20 with PJM included. If no architecture clears this, the fallback is to remove PJM from the universe (Option A from the handoff doc) and proceed with the clean non-PJM baseline.

---

## Phase 2 Preview (Not In Scope Here)

Once the best architecture is identified, the regulated-utility signal slot will be tested with:
- **ICR** (already built — yfinance, ~4–12 quarters history)
- **Rate case calendar** (FERC/state PUC dockets — new data source, medium complexity)
- **Load growth / data center queue** (EIA-861 + PJM interconnection queue — new data source, captures the data center demand wave)

Each will be evaluated on IC@21d improvement for the PJM + regulated MISO/CAISO/SPP tickers.
