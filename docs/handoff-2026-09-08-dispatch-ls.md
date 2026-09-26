# Handoff — "Dispatch": a merchant-power cross-sectional L/S worth broadening (2026-09-08)

**Status: LEAD / NOT ADOPTED.** Extends the two-fold walk-forward finding
(`docs/ml-two-fold-walkforward-results.md`). That doc's "do not wire anything live"
still stands — this handoff is the build plan to make it standalone-worthy, not a
green light.

---

## The idea, named

**Dispatch** — a dollar-neutral cross-sectional long/short that ranks the listed US
power complex (regulated utilities + merchant/IPP generators) the way an economic
dispatch stack ranks generators: cheapest/strongest long, weakest short. GBM ranks
each name monthly on 7 individually-weak signals — merchant **stress-beta** (the
load-bearing one), interest-coverage, data-centre / large-load disclosure counts
(EDGAR), 6-mo momentum, an XLU-vs-gas spark-spread proxy, PJM ancillary pricing.
A point-in-time profitability screen keeps loss-makers out of the long book.

## Evidence so far (this session, walk-forward from `ml_panel_2022_2026.parquet`)

GBM classifier, expanding window, monthly rebalance, long-top / short-bottom tercile
(~3–4 names/side), 2023-03 → 2026-08:

| Variant | Ann return | Sharpe | Max DD |
|---|---|---|---|
| Raw (no screen) | +2.1%/yr | 0.22 | −16.2% |
| **+ bottom-tercile trailing-net-margin screen on long leg** | **+9.5%/yr** | **0.74** | **−13.2%** |
| Benchmarks: XLI | +21.3%/yr | 1.33 | −10.5% |
| PAVE | +23.2%/yr | 1.17 | −16.4% |
| VOLT (from Dec-24) | +29.5%/yr | 1.19 | −7.4% |

Screened, per year: 2024 +12.6% (S0.70) · 2025 +3.4% (S0.32) · 2026† +7.4% (S1.53).
Near-zero market beta throughout (dollar-neutral). †2026 ≈ 7 months.

## Why it's only a lead, not a book

- **Depth:** only 10–13 of the 23 universe names have all 7 features non-NaN in any
  month — a feature-coverage artifact, not a real universe. 3–4 names/side is too thin.
- **Two broken inputs poison the ranks:** `icr_z` for TLN is frozen at a 2016 value
  and forward-filled; `gdelt_z` is frozen at ~2025-09 (the 2026 pull never completed).
- **Momentum-dominated in practice:** `mom_z` (imp ≈0.27) + `xlu_xng_spread` (≈0.22)
  outweigh the stress-beta/ICR/EDGAR trio the parent doc credited. It chased the
  2024–25 winners into 2026 — the whole 2026 drag is the long leg, mostly TLN
  (−4.3%/mo in 2026, longed 9 of 12 months).
- Long-only benchmarks (XLI/PAVE/VOLT) still beat the screened version on raw return.

## Workstreams to broaden depth + boost return

1. **Restore the universe to ~23.** Fix `icr_z` staleness/ffill, replace or properly
   lag `gdelt_z`, confirm every feature has a real value per name per month. Target:
   ≥20 names with complete features every month 2019→now.
2. **Point-in-time profitability screen (the proven lever).** EDGAR TTM net margin via
   `research/electricity_earnings_exposure/edgar.py::quarterly_flow`, filing-date lag,
   exclude bottom tercile from the long leg. Also test operating margin, FCF margin,
   ROE; test as a continuous weight, not a binary gate; add a "≥N quarters of
   post-event financials" guard (but note the crude `min_history` cut *backfired* —
   it benched good longs).
3. **De-bias away from momentum.** Test dropping `mom_z` (the doc's ablation said that
   *raised* AUC), cap single-name weight, cap ISO / regulated-vs-merchant
   concentration, add a turnover penalty.
4. **Horizon.** The doc's headline signal was 3-month (`fwd3`, AUC 0.758). Test a
   3-month hold / quarterly rebalance with overlap-adjusted stats.
5. **Risk overlays (bring vol down, raise Sharpe):**
   - vol-target the L/S to ~8–10% annualised;
   - hold unused capital in T-bills; add an explicit gold / long-Treasury sleeve sized
     to a portfolio-vol target (cf. `docs/electrification-short-leg-insurance-probe-results.md`
     — a GLD/short-duration sleeve did most of the drawdown work there);
   - trend/vol regime gate to scale gross — reuse the repo's REGIME layer
     (`grid_equipment_overlay` memory / `docs/grid-regime-layer2-results.md`).
6. **Costs.** Add borrow + turnover + spread. Short book is regulated utilities
   (cheap, deep borrow) — verify, and check the long-leg turnover the momentum tilt
   creates.
7. **Portfolio fit.** Marginal Sharpe / correlation vs the existing
   `grid_equipment_basket` + overlay — Dispatch only earns a slot if it diversifies.

## Pre-registered gates for the next build

- **Depth:** ≥20 names with complete features every month, 2019→present.
- **Return/risk:** net-of-cost Sharpe ≥ 0.6 full-sample AND ≥ 0.4 in each of ≥3
  non-overlapping sub-windows.
- **Bar:** beats a 60/40 XLI / T-bill blend on Sharpe, and correlation to XLI ≤ 0.3.
- **Vol-targeted variant:** realised annual vol within ±20% of target, max DD > −15%.
- **No lookahead:** every fundamental input point-in-time (EDGAR filing-date lag).

## Reproduce from

- `research/narrative_mispricing/ml_panel_2022_2026.parquet` — 7 features + `fwd1`/`fwd3`, 23 names, 2022-01→2026-08.
- `research/electricity_earnings_exposure/edgar.py` — point-in-time quarterly fundamentals.
- Parent: `docs/ml-two-fold-walkforward-results.md`. Diagnosis of the 2026 drag and the
  screen counterfactual are in this session's transcript (scratchpad scripts
  `bt_twofold.py`, `diag_2026.py`, `bt_screen.py`).

---

## Note (2026-09-08) — relationship to the narrative-mispricing thread; where breadth comes from

**Dispatch fully subsumes the narrative-mispricing Fold-B result.** Same panel
(`ml_panel_2022_2026.parquet`), same 23-name universe, same 7 features (the narrative
signals `edgar_z` + `gdelt_z` are 2 of them), same GBM. Dispatch only adds portfolio
construction (dollar-neutral L/S tercile book) and the profitability screen. So "combine
Dispatch + mispricing" is not stacking two independent edges — it re-labels one.

**Combining more narrative data does not add breadth of edge.** The ablation says dropping
`gdelt_z` *raises* AUC; `gdelt_z` is stale in-window; `edgar_z` contributes only modestly.
The load-bearing signal is merchant stress-beta.

**Breadth comes from the opposite move — pruning, not merging:**
- Drop the two coverage-killing features (`gdelt_z`, stale/ffilled `icr_z`). They are what
  caps the feature-complete universe at 10–13 of 23; without them it should reach ~20+, and
  the panel can run **2018→2026** (merchant stress-beta's true start) instead of 2022→ —
  roughly 2× names and 2× months. (= workstream 1.)
- **Horizon ensemble:** average the fwd-1m tercile rank (Dispatch) with the fwd-3m rank
  (Fold B's headline) and trade the blend — more cross-sectional obs, two horizons, no new
  signal invented.
- The profitability screen (workstream 2) is the one genuine *new* input, and it is the
  lever that flipped 2026.

**The one reusable mispricing artifact:** `research/narrative_mispricing/spike_c_2026_holdout.py`
— a continuous-rank-IC framing over a **25-name** universe, **9 features**, **2018-start**,
GBM & RF × {full, lean} × {1m, 3m}, with a proper train≤2025-12 / test-2026 holdout and HAC
t-stats. Fold it in as an ensemble member / independent breadth cross-check of the Dispatch
walk-forward, not as a separate strategy line.

### spike_c breadth cross-check — run 2026-09-08, FAILED

Ran `research/narrative_mispricing/spike_c_2026_holdout.py` (25 names, 9 features,
2018-start, continuous rank-IC, GBM & RF × {full,lean} × {1m,3m}, train≤2025-12 /
test-2026 holdout, HAC t):

- **2023-25 re-derivation:** only the lean models get \|t\|>1 (best lean fwd-3m GBM
  IC +0.124, t +2.25); full models all \|t\|<1.
- **+2026 (window B):** every cell decays — best is lean fwd-1m/3m GBM at t≈1.4, nothing ≥2.
- **2026 holdout (window C):** flat-to-**negative in every cell**. RF 3-month IC
  **−0.167 (t −3.77)** and **−0.186 (t −2.12)** — inverted out-of-sample. GBM cells ≈0.
- `narrative_z` importance rank **7–8 of 9** in the full model (≈0.01–0.04); `own_mom6m`
  dominates (0.28–0.34) — the same momentum-loading failure mode as the 7-feature panel,
  on a wider universe and a different (regression rank-IC, not tercile-classifier) framing.
- Data caveats shared with the main panel: `narrative_z` 0% coverage in 2026 (median-filled),
  `stress_beta` ffilled from 2025-12, thin GSI/capex history pre-2026.

**Takeaway:** the mispricing thread's broadest framing does **not** add breadth of edge —
it reproduces the "worked 2023-25, inverts 2026" pattern. The `spike_c` framing is *not*
worth folding in as an ensemble member as-is. Breadth work stays: prune dead features →
widen universe + extend to 2018, add the profitability screen, de-momentum.

---

## v2 — pruned features + widened universe (run 2026-09-08)

**Pruned OUT** (dead as cross-sectional rankers and/or stale in the eval window):
`gdelt_z` (frozen ~2025-09), `xlu_xng_spread` (one value for all names each month → zero
cross-sectional info + stale), `anc_composite` (2 levels → near-degenerate).
**Rebuilt:** `icr_z` point-in-time from EDGAR XBRL (TTM operating income / |interest expense|,
full history, 50-day filing lag) — fixes the frozen-2016-for-TLN bug, now **100% coverage**.
**Widened:** `stress_beta_z` = union of `rolling_betas_2018_2026` + `merchant_beta_orthogonalization/
rolling_betas_baseline` (adds NRG/VST/CNP/EIX/PCG with 2018 starts).
**Kept:** `edgar_z`, `mom_z`. Final feature set = `[edgar_z, icr_z, stress_beta_z, mom_z]`.

**Breadth: achieved.** Usable names/month 10–13 → **16–17 (2024), 17–19 (2025–26)**, i.e. 5–6
per side vs 3–4. Test window kept at 2023+; 2018 history used only to build features.

**Loose univariate check (feature inclusion, not a gate):** `stress_beta_z` IC +0.14 (t +3.1),
`mom_z` +0.15 (t +2.7) carry it; `edgar_z` −0.05 (t −1.4) and `icr_z` −0.01 (t −0.2) are weak —
kept for the model to combine.

**Performance (walk-forward L/S, per year — return / Sharpe / MaxDD):**

| model | 2023 | 2024 | 2025 | 2026* | FULL |
|---|---|---|---|---|---|
| GBM clf raw | −1.7 / −0.11 | **+61.7 / 2.55** | +3.8 / 0.29 | −6.5 / −0.75 | +12.9%/y / 0.77 |
| GBM clf + margin-screen | −4.1 / −0.42 | +42.3 / 2.06 | +7.8 / 0.46 | +2.2 / 0.49 | +12.1%/y / 0.79 |
| GBM reg raw | −8.2 / −0.91 | **+86.0 / 3.26** | +1.6 / 0.20 | +18.9 / 2.80 | +22.4%/y / 1.08 |
| RF reg + margin-screen | −12.5 / −1.36 | +24.1 / 1.51 | +10.8 / 0.53 | +7.5 / 0.96 | +7.4%/y / 0.50 |

**The headline Sharpe (0.77–1.08) is a 2024 artifact, not a cross-sectional edge.** Every
variant makes **+24 % to +86 %** in 2024 and is negative-to-flat in 2023, 2025, and (classifier)
2026. The merchant-minus-regulated return spread was **+68 % (2023) / +127 % (2024) / +35 % (2025)
/ −27 % (2026)** — Dispatch is a momentum-timed **merchant-vs-regulated sector bet**, and it was
only fully loaded (via `stress_beta_z` + `mom_z`) in time to catch 2024 and then rode the 2026
reversal down. Widening the universe added regulated shorts; it did **not** diversify the bet —
there are only 4 merchant names.

**Verdict: pruning did its job (breadth 13→19); the broadened result confirms rather than
overturns the earlier read — the edge is one-regime.** The one pre-registrable next test:
rank **within** the merchant/regulated buckets (or residualize returns on the merchant-spread
factor first) and check whether any cross-sectional signal survives across 2023 + 2025 + 2026.
If it is all 2024, there is no strategy here. Script: scratchpad `dispatch_pruned.py`.

---

## v3 — sector-neutral test (run 2026-09-09): the edge is ~100% the 2024 regime

Question: does anything survive once the merchant-vs-regulated axis is removed — i.e. across
2023 + 2025 + 2026, not just the 2024 merchant-IPP melt-up? Script: scratchpad
`dispatch_sector_neutral.py`.

| variant | 2024 | **ex-2024 (2023+2025+2026)** | FULL |
|---|---|---|---|
| full GBM clf raw | +61.7% / S2.55 | **−1.3%/y / S−0.05** | +12.9%/y / S0.77 |
| full GBM reg raw | +78.9% / S2.53 | **−0.2%/y / S+0.07** | +17.4%/y / S0.87 |
| REG-ONLY GBM clf raw | +1.0% / S0.20 | +1.0%/y / S+0.19 | +1.3%/y / S0.19 |
| REG-ONLY GBM clf **+margin-screen** | +3.8% / S0.64 | **+2.6%/y / S+0.36** | +3.7%/y / S0.40 |
| REG-ONLY GBM reg (± screen) | +1.6% / S0.32 | −4.0 to −4.4%/y / S≈−0.75 | negative |
| SECTOR-NEUT GBM clf raw | +24.5% / S1.95 | +0.5%/y / S+0.12 | +6.8%/y / S0.47 |
| SECTOR-NEUT GBM reg raw | +21.8% / S1.30 | +4.5%/y / S0.38 (2025 −27%, 2026 +45% — noise) | +10.4%/y / S0.58 |

**Findings:**
1. **~100% of Dispatch's Sharpe is the 2024 merchant-vs-regulated melt-up.** Full GBM classifier
   ex-2024 Sharpe **−0.05** (from +0.77); full regressor ex-2024 **+0.07** (from +0.87).
2. **Among regulated utilities alone there is a real but economically trivial signal** — GBM
   classifier ex-2024 Sharpe +0.19 raw, **+0.36 with the margin screen** — directionally
   consistent (not 2024-dependent) but **2023 is still negative**, so it fails a "positive in
   each of 2023 / 2025 / 2026" bar, and full-period Sharpe ~0.4 is not fundable standalone.
3. Regressor and sector-neutral-combined variants are noise outside 2024.

**Verdict: FAILED the regime split. Dispatch is not a standalone cross-sectional strategy** —
it's a momentum-timed merchant-vs-regulated sector bet whose one good regime (2024) has passed
and which whipsawed −27% into 2026. Consistent with RESEARCH-LOG §2 (~28 attempts, no
cross-sectional edge survives an honest regime split).

**What to keep:**
- The **point-in-time trailing-net-margin screen** as a *quality overlay* on any long book — it
  improved the worst year of every variant tested (this handoff and the shipped long-only book).
- The **pruned, widened, point-in-time panel** (`[edgar_z, icr_z(EDGAR), stress_beta_z(union),
  mom_z]`, 16–19 names/mo, `dispatch_pruned.py` / `dispatch_sector_neutral.py`) as reusable
  infrastructure if a future signal wants this universe.
- Fold B's AUC watch-item (RESEARCH-LOG §4 item 18) is unaffected — that was tercile
  *classification* AUC on 2026; this is the *returns* expression, and it does not clear the bar.
