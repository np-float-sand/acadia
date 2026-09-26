# Narrative mispricing — Approach C results (broadened: all-features GBM/RF)

**Verdict: FAILED.** Same overfit shape this project already hit once (Henry Hub
forward-gas Round 4/5): a recent-window result that looks real vanishes on the
full sample.

## Context

`docs/untried-tasks-log.md` item 9, Approach C (`docs/handoff-narrative-mispricing-approach-c.md`).
Approach A (narrative composite alone, linear rank-IC) was NOT ADOPTED — not
significant at any horizon. The handoff for C asked, at minimum, to test the
narrative composite jointly with one fundamentals measure (ICR) in a small
RF/GBM. At the user's explicit request this session, the feature set was
broadened further: **every reasonably alignable feature already built anywhere
in `docs/RESEARCH-LOG.md`** for this 25-name universe, not just ICR, "even if
it was only t-stat ~1 on its own."

## Universe and data

The `grid_resilience` 25-name universe (the same one the narrative composite is
built on): NRG, VST, CNP, AEP, EXC, PPL, FE, CEG, TLN, ETR, WEC, DTE, CMS, PCG,
EIX, XEL, ES, ED, D, PEG, AEE, EVRG, DUK, SO, NEE. 88 monthly cross-sections,
2018-06 → 2025-09 (bounded by the narrative composite's own coverage).

**Per-ticker features** (vary by name and month):
- `narrative_z` — the refined EDGAR+GDELT composite (`composite_monthly_refined.parquet`)
- `icr` — interest coverage ratio, EDGAR XBRL (`icr_edgar_panel.parquet`), quarterly ffilled
- `stress_beta` — rolling merchant stress-beta (`rolling_betas_baseline.parquet`)
- `gsi_factor` — the production grid-stress-index factor score, from a **fresh
  `grid_resilience.main.run()`** with the current (post grid-search) config —
  not a stale cached file
- `own_mom6m` — each ticker's own trailing 6-month price momentum

**Market-wide features** (same value broadcast to every ticker each month —
each of these was originally built/tested on a *different* universe or as a
basket-level overlay, so a per-name mapping doesn't exist; included here as
macro conditioning variables per the user's "in tandem" framing):
- `capex_guidance_z` — the 15-utility capex-guidance revision composite
- `category_demand_z` — Census M3 electrical-equipment category-demand momentum
- `ancillary_pjm_z` — PJM ancillary/scarcity clearing-price z-score (2018–2021 only, NaN after)
- `real_yield_chg6m` — 6-month change in the 10y TIPS real yield (FRED DFII10)
- `ng_mom6m` — Henry Hub front-month natural gas 6-month momentum

Target: forward 1-month and forward 3-month return per ticker.

Reproducible: `research/narrative_mispricing/approach_c/{build_panel.py,model.py,panel.parquet,results.csv}`.

## Methodology (pre-registered before any result was looked at)

- **Two windows**: FULL (2018-06 → 2025-09, 88 months) and RECENT (2023-01 →
  2025-09, ~24 months) — the exact discipline that already caught one overfit
  in this project's Henry Hub work.
- **Walk-forward only** — `sklearn.TimeSeriesSplit(5)` on the month axis, no
  random k-fold, given ~90 autocorrelated monthly cross-sections.
- **OOF rank-IC** — mean cross-sectional Spearman(pred, fwd_ret) across the 25
  tickers in each held-out month, HAC/Newey-West (3-lag) t-stat across months.
- **Two feature sets** per cell: `full` (all 10 features above) and `lean`
  (narrative_z, icr, gsi_factor, capex_guidance_z) — the prior Henry Hub
  rounds found a smaller feature set is the more honest test on this little
  data, not a weaker one.
- **Permutation null**: 60 draws shuffling the target *within* each month
  (preserves cross-sectional structure, breaks the feature→target link),
  `perm_p` = fraction of \|null IC\| ≥ \|observed IC\|.
- **Pre-registered PASS bar**: \|t\| ≥ 2 on OOF rank-IC in **both** windows,
  same sign, **and** `perm_p` < 0.05 in **both** windows. A recent-window
  flicker alone does not pass — that is exactly the failure mode this bar
  exists to catch.

## Results

| window | model | feats | target | mean IC | t | perm p |
|---|---|---|---|---:|---:|---:|
| FULL 2018–2025 | GBM | full | 1mo | −0.010 | −0.30 | 0.700 |
| FULL 2018–2025 | RF  | full | 1mo | −0.007 | −0.21 | 0.800 |
| FULL 2018–2025 | GBM | full | 3mo | −0.004 | −0.11 | 0.917 |
| FULL 2018–2025 | RF  | full | 3mo | −0.028 | −0.65 | 0.250 |
| FULL 2018–2025 | GBM | lean | 1mo | −0.019 | −0.78 | 0.383 |
| FULL 2018–2025 | RF  | lean | 1mo | −0.025 | −0.93 | 0.200 |
| FULL 2018–2025 | GBM | lean | 3mo | +0.016 | +0.41 | 0.500 |
| FULL 2018–2025 | RF  | lean | 3mo | +0.006 | +0.16 | 0.833 |
| RECENT 2023–2025 | GBM | full | 1mo | +0.082 | +2.24 | 0.083 |
| RECENT 2023–2025 | RF  | full | 1mo | +0.114 | +2.85 | 0.017 |
| RECENT 2023–2025 | GBM | full | 3mo | +0.131 | +2.79 | 0.000 |
| RECENT 2023–2025 | RF  | full | 3mo | +0.136 | +2.49 | 0.000 |
| RECENT 2023–2025 | GBM | lean | 1mo | +0.060 | +1.64 | 0.183 |
| RECENT 2023–2025 | RF  | lean | 1mo | +0.139 | +3.55 | 0.000 |
| RECENT 2023–2025 | GBM | lean | 3mo | +0.168 | +3.73 | 0.000 |
| RECENT 2023–2025 | RF  | lean | 3mo | +0.120 | +2.56 | 0.000 |

**Every FULL-window cell is flat-to-negative** (\|t\| < 1, perm p 0.20–0.92 —
indistinguishable from noise). **Every RECENT-window cell is positive**, half
clearing \|t\| ≥ 2 with perm p ≤ 0.017 — a result that, seen alone, would look
like a real, statistically clean edge. Because the pre-registered bar requires
both windows, **this fails outright**: the "2023–25 signal" does not survive
being tested on the fuller history, the same shape as the Henry Hub
forward-gas Round 4 result (2023-26 rank-IC +0.18/+0.29 that fell to +0.02/+0.00
on 2018-26). With ~24 independent monthly cross-sections in the recent window
and 10 correlated features, a tree model has ample room to fit noise that
looks like signal until more history is added.

**Feature importances (full model, averaged across folds) never put
`narrative_z` in the top 3**, in any of the four full-model cells (FULL 1mo,
FULL 3mo, RECENT 1mo, RECENT 3mo): its share ranges 0.02–0.10, versus 0.19–0.44
for `own_mom6m`, `category_demand_z`, `gsi_factor`, and `stress_beta` at
various horizons. This directly answers the "does narrative interact with
other data" question the handoff raised: **no** — even inside the inflated
recent-window fits where the model found *something*, the narrative composite
is not what it found. The lift comes from price momentum and the existing
production signals (GSI factor, stress-beta), not narrative content.

## Disposition

**Approach C: FAILED, closed.** Nothing wired live; `research/narrative_mispricing/approach_c/`
kept as a reusable, documented negative result and a working walk-forward
GBM/RF + permutation-null harness for anyone testing another feature
combination on this universe later. Approach B (event study) remains
un-started — its own handoff (`docs/handoff-narrative-mispricing-approach-b.md`)
is unaffected by this result and still the next open thread in this line if
narrative-mispricing work continues.
