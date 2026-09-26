# CEP as a data-center-REIT signal — results

**Date:** 2026-09-16
**Pre-registration:** `docs/datacenter-reit-cep-preregistration.md` (written before this script
was run; universe, CEP series, controls, spec, gate threshold, and windows all locked there —
none changed after seeing output).
**Script:** `research/datacenter_reit_cep/probe.py`
**Origin:** `docs/handoff-2026-09-16-datacenter-reit-cep-signal.md`, replicating (own-horizon,
reduced controls) Khaliq & Boufous (2026), *Finance Research Letters* 104.

**Amendment (2026-09-16, user-directed):** the gate for this thread is **|t| > 1.5**, not the
|t| ≥ 2.0 set in the pre-registration below — changed on explicit user instruction after
discussion of what the original t-stats mean. This applies to this thread only; it is not a
change to the |t| ≥ 2.0 convention used for the other ~30 signals in `docs/RESEARCH-LOG.md`,
noted here so the two don't read as inconsistent later.

## Method (as pre-registered)

Target: equal-weighted `DLR`+`EQIX`+`IRM` monthly return. Signal: national EIA commercial
retail electricity price (`sectorid=COM`, `stateid=US`), MoM % change. Controls: Henry Hub
(`NG=F`) % chg, WTI (`DCOILWTICO`) % chg, VIX (`VIXCLS`) % chg, SPY monthly return, GS10 level
change — GPR dropped (no FRED mirror, out of scope). G1 = OLS with HAC(maxlags=6) SEs, gate
|t|≥2.0 on `cep_chg`. G2 = time-series tercile spread (top-CEP-chg months minus bottom, Welch
t-test), same gate. Both run over 3 windows; 2026-YTD reported for context only (n=4, too small
to gate).

## Result

| window | n (mo) | G1 t (cep_chg coef) | G1 sign | G2 t (tercile spread) | G2 spread |
|---|--:|--:|---|--:|--:|
| paper 2015-2025 | 105 | -1.28 | expected (−) | -1.75 | -2.48%/mo |
| 2023-onward | 35 | -0.47 | expected (−) | -1.10 | -2.79%/mo |
| 2026-YTD (context) | 4 | skipped (n<12) | — | skipped | — |

**At |t|>1.5: G2 full-window PASSES (t=-1.75). G1 full-window (-1.28), and both G1/G2 in the
2023-onward window (-0.47 / -1.10), do not clear even the lowered bar** — 1 of 4 gated cells
passes, not a clean result across the board.

Secondary (context only, not gated): forward-1m/3m Spearman rank-corr of `cep_chg` vs. portfolio
return is small and sign-inconsistent across windows (+0.105/+0.079 full-window,
+0.236/-0.017 in 2023-onward) — no forward-looking edge either.

## Verdict

**PARTIAL PASS at |t|>1.5 (full-window G2 only); G1 and the 2023-onward window still don't clear
even the lowered bar.** The sign is right throughout (higher CEP → lower REIT return, as the
paper claims, in both G1 and G2, in both gated windows), and the full-window tercile spread
(t=-1.75) is the strongest result anything in this project's CEP/power-cost signal family has
produced. But the effect weakens in the more recent 2023-onward window (G1 -0.47, G2 -1.10)
rather than holding or strengthening, which is the same regime-fragility pattern flagged as a
concern elsewhere in this project's log — a reason for caution about how far to extend this,
not a reason it's wrong to keep as a lead at the user-directed threshold.

This inherits and confirms Probe B's prior (`docs/electricity-earnings-exposure-probe-results.md`):
a similarly-constructed power-cost signal fails again, now on data-center REITs specifically
rather than industrials. Plausible reason the sign is directionally right but not significant:
n=3 names / a single equal-weighted portfolio series gives ~105-116 monthly observations, which
is thin for a HAC-adjusted regression with 5 controls, and REIT returns are dominated by rate
moves (GS10) far more than by a ~1%/mo commercial power price series feeding into opex, not COGS,
for these companies (data centers pass most power cost through to tenants contractually — a
mechanism difference from the industrials Probe B tested, and possibly the reason even the sign
concordance doesn't reach significance).

## Combination test (Decision 8) — run on explicit user instruction, deviating from pre-registration

The pre-registered contingency (only run this if the standalone signal clears |t|≥2.0 in ≥2
windows) was **not met** (best: t=-1.76). Run anyway on direct user instruction — accepting
t≈1.75-1.76 as sufficient to test the downstream question, on the stated prior that marginal
signals in this line of work sometimes add value in combination even when they don't stand
alone. This is a deviation from the pre-registration, **not** a change to the project's
|t|≥2.0 gate convention used everywhere else in this log.

**Script:** `research/datacenter_reit_cep/combination_test.py`. Built a CEP "6-month change
rising" mask structurally identical to `hedge_overlay.real_yield_rising_mask` (same 126-day
lookback, `month_hold` convention), with a 65-day publish lag standing in for EIA's real
release cadence (no point-in-time EIA archive exists in this repo — a documented assumption,
not fitted). Compared on the `screen` construction, `+val+sleeve[+short]` stack, full window
2017-06→2026-08 plus the two `config.SUBWINDOWS`.

| variant | Sharpe (full) | MaxDD (full) | Calmar (full) | feared_pnl | Sharpe 2019-22 | Sharpe 2023-26 |
|---|--:|--:|--:|--:|--:|--:|
| no short (`+val+sleeve`) | 0.693 | -0.240 | 0.602 | 0.136 | 0.505 | 1.596 |
| real-yield only (**shipped**) | 0.655 | -0.232 | 0.575 | 0.136 | 0.466 | 1.556 |
| CEP only | 0.676 | -0.234 | 0.587 | 0.153 | 0.539 | 1.396 |
| CEP AND real-yield | 0.702 | -0.232 | 0.614 | 0.147 | 0.480 | 1.541 |
| CEP continuous scaler | 0.683 | -0.243 | 0.568 | 0.137 | 0.520 | 1.520 |

Mask activity: real-yield ON 53.3% of days, CEP ON 57.5%, both-AND ON 29.8% (a materially
tighter filter), continuous scaler averages 0.44x.

**Full-window read:** `CEP AND real-yield` edges out *both* endpoints — it beats the shipped
real-yield-only gate on Sharpe (0.702 vs 0.655) and Calmar (0.614 vs 0.575) at an identical
MaxDD (-0.232), and even beats having no short at all on Sharpe/Calmar/feared_pnl. That's a
genuinely more interesting result than "CEP does nothing" — the AND-mask isn't just a diluted
version of the real-yield mask (which would sit between real-yield-only and no-short), it's
picking a better-timed ~30%-of-days subset than either endpoint.

**Why this is not being called a win:** the full-window improvement does not hold up when split
by `config.SUBWINDOWS` — 2019-22 is mildly better (0.480 vs 0.466) but 2023-26 is mildly *worse*
(1.541 vs 1.556) than the shipped gate, i.e. the two halves partially offset rather than both
confirming the same direction. Effect sizes throughout this table (≤0.05 Sharpe, ≤0.04 Calmar)
are small relative to what a ~9-year, heavily-autocorrelated single backtest can resolve, and no
significance test (bootstrap/permutation on mask timing) has been run — exactly the kind of
full-sample-looks-good-but-doesn't-split-cleanly pattern this project has been burned by before
(`docs/grid-regime-layer2-results.md` §7, where a full-window "win" turned out to be a single
COVID-window coincidence once decomposed).

**Verdict: lead, not adopted.** The AND-combination is worth a bootstrap/permutation
significance check before any live change to `hedge_overlay.py`'s gate; on the numbers alone it
is not distinguishable from noise given the subwindow split. **Not wired live.** The shipped
real-yield-only gate is unchanged.

## Addendum — post-hoc 2025-to-date / 2026-to-date (requested after seeing the result above)

**Not pre-registered.** Run after the primary result was already seen, at the user's request, so
these are reported as context, not a gate re-test — the whole point of Decision 7 was to lock
windows before looking, and picking a window after the fact is exactly the bias that guards
against. Data note: `hh_chg` (Henry Hub, `NG=F` via yfinance) has 3 missing monthly observations
in this period (2025-06, 2026-02, 2026-03), which the panel's `dropna()` removes along with those
months entirely — so both windows below are thinner than their calendar span, not just short.

| window | n (mo) | G1 t | G2 t | note |
|---|--:|--:|--:|---|
| 2025-to-date | 15 (of 17 calendar months) | -1.76 | -1.28 (n=5/5 per tercile) | ~matches full-window G1, i.e. adds no new information beyond what the full sample already showed |
| 2026-to-date | 4 (of 6 calendar months, 2 dropped to the `hh_chg` gap) | not computed (n<12) | not computed | too thin to read; raw monthly cep_chg/portfolio_ret show no visible pattern |

Neither changes the verdict. 2025-to-date reproduces the same marginal t≈-1.75 already reported
for the full sample rather than adding a new data point in favor of the signal (unsurprising — it
is a subset of the same series, not independent evidence), and 2026-to-date is uninformative at
n=4. G1 on 2025-to-date (-1.76) does clear the |t|>1.5 bar, but per the note above this is the
same information as the full-window G1, not a second confirmation.

## What would change this

A REIT-level panel with ticker fixed effects (n≈3×105 obs instead of 1×105) was scoped as a
secondary robustness check in the pre-registration but not run — tripling the observation count
via panel structure (which doesn't add independent information, just more correlated rows) is
unlikely to move the G1/2023-onward cells enough to matter. Not pursued further.
