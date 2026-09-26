# Electricity-price volatility as a crypto-miner / merchant-generator signal — results

**Run:** 2026-09-16
**Pre-registration:** `docs/electricity-price-volatility-miner-preregistration.md` (Decisions 1-7)
**Origin:** `docs/handoff-2026-09-16-crypto-miner-power-volatility.md` (paywalled-paper summary,
"Do electricity prices predict the returns and volatility of cryptocurrency mining stocks?")
**Prior FAIL this probe must not reproduce:** `docs/electricity-earnings-exposure-probe-results.md`
Probe B — a crypto-miners-vs-other-sectors bloc move masquerading as an electricity-price
relationship. BTC control is mandatory here for exactly that reason.

**Verdict: FAIL, both primary arms.** Primary miners (CIFR, WULF) and primary generators
(VST, NRG, CEG, TLN) all miss the pre-registered gate (`|t| >= 2.0`, correct sign, test window,
selected hub). VST clears no hub at all in the train window. Three widened-robustness miners
(MARA, IREN, BTDR) hit `|t| >= 2.0` in the test window, but with the **wrong sign** — a
generator-like ("more hub variance -> more equity variance") result, not curtailment-hedging —
so per Decision 6/4 this is reported as an unexplained anomaly, not a pass, and cannot rescue
the primary-name verdict regardless. Combined-portfolio backtest **not run** (constraint: only
run it if a primary name passes standalone; none did).

Code: `research/electricity_price_volatility/` (`universe.py`, `hub_loader.py`, `data.py`,
`hub_selection.py`, `regressions.py`, `run_probe.py`). Run `python3 -m
research.electricity_price_volatility.run_probe` to reproduce every number below (raw output
also dumped to `research/electricity_price_volatility/_output/results.json`).

---

## 1. Pre-registered gate (restated)

`|t| >= 2.0` on the **test-window (2025-01-01 -> present) regression coefficient**, correct
pre-registered sign, for the hub **selected on the train window only** (2023-01-01 ->
2024-12-31, never re-selected on test data). 2026-YTD is a fragility check, not gate-bearing.
Sign predictions, locked before any output was seen:

- **Miners** (CIFR, WULF primary; MARA/RIOT/IREN/BTDR widened/robustness-only): lagged hub
  price-return variance -> **lower** subsequent equity variance (curtailment-hedging).
- **Generators** (VST, NRG, CEG, TLN): lagged hub price-return variance -> **higher**
  subsequent equity variance (more volatile realised wholesale revenue).
- CLSK excluded (no defensible ERCOT/PJM proxy for its GA/MS/TN/WY fleet). NYISO/CAISO/Mid-C
  excluded (no fetcher in this repo); WULF is therefore tested only against the weaker
  ERCOT/PJM proxies, not its real NYISO settlement market — flagged up front, not after the fact.

## 2. Method

- **Hub universe:** all 19 cached hubs (ERCOT 7 + PJM 12), scanned for every name regardless
  of its disclosed ISO — geographic plausibility is judged *after* selection, never used to
  restrict the scan (Decision 2).
- **Daily hub series:** hourly DA LMP resampled to a **daily mean price** per hub, then daily
  log return, then **realized variance proxy = squared daily log return** (`r_t^2`) — the
  standard one-day realized-variance proxy; no rolling smoothing was layered on top (stated
  choice per the pre-registration's "state your choice" requirement — see
  `research/electricity_price_volatility/data.py` docstring).
- **Alignment / lag:** hub markets settle every calendar day (incl. weekends); equities trade
  only business days. For each equity trading date `t`, "lagged hub variance" is the hub's
  realized variance as of the most recent hub calendar date at or before the equity trading day
  **one trading day before `t`** (`merge_asof`, backward) — i.e. information already public
  strictly before the day being predicted.
- **Linear spec A (GARCH-style, gate-bearing):**
  `eq_var_t = a + b*hub_var_lag1 + c*control_var_t + d*control_ret_t + e`, OLS, `b`'s t-stat is
  the headline number. `control_{var,ret}` = **contemporaneous BTC** (miner arm) or **SPY**
  (generator arm) — included specifically to net out the bloc-move confound before crediting
  the hub term with anything.
- **Linear spec B (secondary, not gate-bearing, no pre-registered sign):**
  `eq_ret_t = a + b*hub_ret_lag1 + c*control_ret_t + d*control_var_t + e`.
- **Non-linear spec C:** GradientBoostingRegressor (100 trees, depth 2) on
  `[hub_var_lag1, hub_ret_lag1, control_var, control_ret]` -> `eq_var_t`, **trained on the train
  window only**, scored out-of-sample on the test window (and, same fitted model, on the 2026-YTD
  subset) — the train/test split doubles as the OOS check per Decision 5.
- **Non-linear spec D (tail cut):** top-decile `|hub_ret_lag1|` days vs. the rest, within each
  window, Welch t-test on the difference in mean `eq_var_t`.
- **Windows:** train 2023-01-01 -> 2024-12-31 (selection only); test 2025-01-01 -> present
  (gate-bearing); 2026-01-01 -> present (fragility, not gating). Actual data coverage: ERCOT
  hub cache through 2026-09-06, PJM hub cache through 2026-08-31, equities/BTC through
  2026-09-15 (fetched fresh for this run).

## 3. A blocker hit and worked around — documented per the task's own instruction

`grid_resilience.data.equity_prices.fetch_prices`'s **threaded per-month cache warmer silently
dropped ticker columns** for CIFR/WULF/MARA/RIOT/IREN/BTDR/BTC-USD/VST/NRG/CEG/TLN/SPY when
~9-44 months were fetched concurrently (4 workers) — the exact failure mode the function's own
docstring already warns about ("yfinance's batch download can silently drop a ticker under
concurrent/rate-limited load without raising"). The first attempt returned a price panel with
40-70% NaN coverage, including a fully-NaN tail. **Fix:** called the module's own
`_download_with_retry` + `save_monthly_chunks_wide` primitives directly for a single
non-threaded full-range request (2023-01-01 -> 2026-09-16, all 12 tickers at once), which
populated the monthly cache cleanly (2 residual NaNs per miner ticker = legitimate BTC-only
trading days; TLN's 152-day gap = pre-emergence, it began trading 2023-06-02). Subsequent
`fetch_prices` calls read cleanly from the now-complete cache. No new fetcher was written —
this reuses `fetch_prices`'s own internals, just not through its buggy concurrent path — and is
recorded here as the honest blocker rather than silently substituted away.

## 4. Hub-selection results (train window, 2023-01-01 -> 2024-12-31)

| name | arm | hubs w/ correct sign | selected hub | train `t` | plausibility |
|---|---|---|---|---|---|
| **CIFR** | miner (primary) | 17/19 | ERCOT:HB_PAN | −1.73 | **Plausible** — 100% ERCOT/TX fleet. |
| **WULF** | miner (primary) | 15/19 | ERCOT:HB_PAN | −1.51 | **RED FLAG.** WULF's disclosed fleet is 100% NY(NYISO)/PA(PJM) — zero ERCOT exposure. This is *worse* than the pre-registered "known-weak PJM proxy" warning anticipated (which expected a PJM pick via the 20% PA share); the model instead picked an ISO with no connection to WULF's fleet at all. |
| MARA | miner (widened) | 11/19 | PJM:DOMINION HUB | −1.79 | **RED FLAG.** MARA disclosed TX/ND/OH/NE — none in Dominion's VA/NC PJM footprint. |
| RIOT | miner (widened) | 14/19 | PJM:EASTERN HUB | −1.19 | Weakly plausible — RIOT's ~15% KY exposure sits in LG&E/KU territory, which joined PJM in 2022; "Eastern Hub" is a generic PJM reference point, not KY-specific. |
| IREN | miner (widened) | 15/19 | PJM:WEST INT HUB | −1.08 | **RED FLAG — the textbook example.** IREN is disclosed as 100% TX/ERCOT; a PJM hub is exactly the implausible case the pre-registration named up front (`docs/electricity-price-volatility-miner-preregistration.md` Decision 2's own CIFR-via-PJM example). |
| BTDR | miner (widened) | 16/19 | ERCOT:HB_NORTH | −0.63 | Plausible for the ~60% TX majority; no hub covers the ~40% WA (WECC) share. |
| **VST** | generator (primary) | **0/19** | **none selected** | — | N/A — no hub cleared the sign requirement at all, an explicit nameable result per Decision 2/7, not folded into an average. |
| **NRG** | generator (primary) | 14/19 | ERCOT:HB_WEST | +2.22 | Plausible — NRG is ~81% ERCOT. |
| **CEG** | generator (primary) | 5/19 | PJM:CHICAGO GEN HUB | +0.33 | Plausible on reflection — Constellation's disclosed nuclear fleet includes a large Illinois/ComEd (PJM-Chicago-zone) component (Braidwood, Byron, Dresden, LaSalle, Quad Cities), even though the Probe A fleet table (built for a different purpose) modeled CEG's nuclear generically as "PJM." Weak signal either way: barely positive at t=+0.33, only 5/19 hubs even had the right sign. |
| **TLN** | generator (primary) | 12/19 | PJM:CHICAGO HUB | **+5.81** | **Mild flag** — TLN's disclosed fleet (Susquehanna PA nuclear, MD/PJM gas, ERCOT gas) has no Illinois exposure; the pick is in-ISO but the wrong sub-region. The **t=+5.81 in-sample is also the single largest number anywhere in this scan** — see §6 for what happened to it out-of-sample. |

Widened miners are robustness-only (Decision 3) and cannot rescue the CIFR/WULF primary verdict
regardless of their own numbers. Full 19-hub rankings for every name are in
`research/electricity_price_volatility/_output/results.json`.

## 5. Test-window (2025-01-01 -> present) and 2026-YTD results

### Primary miners — gate-bearing

| name | hub | spec | test coef | test `t` | test n | 2026-YTD coef | 2026-YTD `t` | 2026-YTD n |
|---|---|---|---|---|---|---|---|---|
| CIFR | ERCOT:HB_PAN | linear-var (A) | +6.31e-05 | **+0.47** | 557 | −5.64e-05 | −0.22 | 230 |
| CIFR | ERCOT:HB_PAN | linear-ret (B) | −0.0029 | −1.06 | 557 | — | — | — |
| WULF | ERCOT:HB_PAN | linear-var (A) | −3.97e-05 | **−0.19** | 557 | −7.63e-05 | −0.47 | 230 |
| WULF | ERCOT:HB_PAN | linear-ret (B) | −0.0039 | −1.49 | 557 | — | — | — |

**Non-linear (GBM, test window):** CIFR OOS R² −0.11, OOS Spearman ρ +0.24 (p<1e-7); WULF OOS
R² −0.10, OOS Spearman ρ +0.21 (p<1e-6). Positive rank-correlation with negative R² means the
model ranks high/low-variance days somewhat correctly but its predicted *magnitudes* are worse
than just guessing the mean — a weak, uncalibrated fit, not a usable variance forecast.

**Tail cut (test window):** CIFR top-decile |hub-return| days show mean eq-var 0.00436 vs.
0.00344 rest (t=+0.92, p=0.36 — wrong direction for the hedging story and not significant).
WULF: 0.00306 vs. 0.00314 (t=−0.10, p=0.92 — negligible).

**Gate verdict: FAIL for both CIFR and WULF** — neither the linear variance regression nor any
other spec clears `|t| >= 2.0` in the test window, in either sign.

### Primary generators — gate-bearing

| name | hub | spec | test coef | test `t` | test n | 2026-YTD coef | 2026-YTD `t` | 2026-YTD n |
|---|---|---|---|---|---|---|---|---|
| VST | — (no hub selected) | — | — | — | — | — | — | — |
| NRG | ERCOT:HB_WEST | linear-var (A) | −2.07e-05 | **−0.24** | 611 | −7.00e-05 | −0.76 | 248 |
| CEG | PJM:CHICAGO GEN HUB | linear-var (A) | −3.71e-05 | **−0.16** | 619 | +8.73e-05 | +0.32 | 258 |
| TLN | PJM:CHICAGO HUB | linear-var (A) | −6.48e-05 | **−0.23** | 619 | +9.75e-05 | +0.28 | 258 |

**Gate verdict: FAIL for NRG, CEG, TLN** (`|t|<2` in all cases); **FAIL for VST by construction**
(no hub cleared the train-window sign filter, so there is nothing to test).

**Non-linear (GBM, test window):** NRG OOS R² +0.01 (the only non-negative R² anywhere in this
probe), Spearman ρ +0.70; CEG OOS R² −0.10, ρ +0.65; TLN OOS R² −0.05, ρ +0.66 — all similarly
high rank-correlation / poor-magnitude pattern as the miners (see §6 on what's actually driving
this).

**Tail cut (test window):** no name shows a significant difference (all |t| < 0.6, all p > 0.55).

### Widened miners (robustness only — cannot rescue the primary verdict)

| name | hub | test coef | test `t` | sign | 2026-YTD coef | 2026-YTD `t` |
|---|---|---|---|---|---|---|
| MARA | PJM:DOMINION HUB | +0.00346 | **+2.81** | **wrong (generator-like)** | +0.00479 | **+2.93** |
| RIOT | PJM:EASTERN HUB | +0.00120 | +1.28 | wrong | +0.00167 | +1.38 |
| IREN | PJM:WEST INT HUB | +0.00641 | **+3.22** | **wrong (generator-like)** | +0.00544 | **+2.14** |
| BTDR | ERCOT:HB_NORTH | +0.00371 | **+3.15** | **wrong (generator-like)** | −0.00051 | −0.32 |

**MARA, IREN, and BTDR all clear `|t| >= 2.0` in the test window — with the wrong sign.**
Per the pre-registration's own logic for the generator arm (Decision 4: "a generator result of
the miner's predicted sign is not a bonus pass — it's a different, unexplained mechanism"), the
symmetric case applies here: **this is not a pass.** Higher lagged hub-price variance predicting
*higher*, not lower, subsequent equity variance is the opposite of the curtailment-hedging
mechanism and is reported as an unexplained anomaly, not evidence for anything. It also cannot
rescue CIFR/WULF regardless, per Decision 6.

**Why the GBM's high Spearman numbers should not be over-read:** feature-importance checks on
MARA/IREN/BTDR's GBM fits show `control_var`+`control_ret` (BTC) carrying 55-80% of total
importance vs. 15-35% for the two hub features (`hub_var_lag`, `hub_ret_lag`) — e.g. MARA:
hub 31% / BTC 69%; IREN: hub 18% / BTC 82%; BTDR: hub 44% / BTC 56%. Most of the GBM's OOS
rank-correlation is BTC-driven, not hub-driven — the same bloc-move pattern Probe B already
caught, now visible even inside a spec that explicitly controls for it. The *linear* spec's
wrong-sign significant coefficients already net out BTC linearly, so they are not simply this
same artifact — but the magnitude of the GBM's apparent skill is not to be trusted as "the hub
variable is doing this."

## 6. TLN's train-to-test collapse (regime fragility, illustrative)

TLN's train-window t-stat (PJM:CHICAGO HUB, +5.81) is the single largest number anywhere in
this 19-hub x 10-name scan — and it collapses to t=−0.23 in the test window (even flipping
sign), then to +0.28 in 2026-YTD. This is exactly the train/test decay pattern this project has
seen repeatedly (Henry Hub GBM, narrative-mispricing Approach C, the merchant-beta-
orthogonalization writeup) and is the reason Decision 2's split-sample discipline exists in the
first place — a full-sample fit would have reported TLN's PJM:CHICAGO HUB relationship as the
headline finding of this entire probe.

## 7. Headline verdict (Decision 6)

| arm | names | verdict |
|---|---|---|
| **Primary miners** | CIFR, WULF | **FAIL** — no spec clears `|t|>=2.0`; WULF's selected hub is also geographically implausible (red flag, not just a null result). |
| **Widened miners** (robustness only) | MARA, RIOT, IREN, BTDR | **FAIL against the pre-registered sign** — 3/4 hit significance with the *wrong* (generator-like) sign; IREN's and MARA's selected hubs are also geographically implausible. Cannot rescue the primary verdict per Decision 6. |
| **Primary generators** | VST, NRG, CEG, TLN | **FAIL** — VST clears no hub at all; NRG/CEG/TLN all miss `|t|>=2.0` in the test window; TLN's striking train-window number is a regime-fragility artifact (§6), not a real effect. |

**No hub, in any arm, for any primary name, clears the pre-registered gate on the held-out test
window.** This is a clean, symmetric null result across both the miner mechanism (curtailment
hedging) and the parallel generator mechanism (revenue-volatility pass-through) — the paper's
claimed relationship, at least as proxied by ERCOT/PJM daily hub variance with a BTC/SPY
control, does not replicate in this data.

## 8. Portfolio-combination note

Per the source handoff's "Portfolio role" section and this task's constraints: the
combined-portfolio backtest (blending this sleeve with `screen +val+sleeve`) is run **only if a
primary name passes standalone**. None did (§7), so **the combination step has not been run**
and is not scoped further here — consistent with this project's convention of not building on
top of a negative result. If this thread is revisited, the two live candidates are (a) resolving
the NYISO gap for WULF (its actual settlement market, not the ERCOT/PJM proxies tested here) or
(b) re-testing with an intraday/higher-frequency variance proxy instead of the single-day
squared-return proxy used here, since a daily proxy may simply be too coarse to catch a
short-horizon curtailment-hedging effect even if the mechanism is real.

## 9. Honest limitations

- The realized-variance proxy is the single-day squared log return, the simplest available
  choice given no intraday equity data — a noisier proxy than a GARCH-fitted conditional
  variance or a multi-day realized-variance estimator. This works against finding a true effect
  as much as it works against a false one; it is not obviously biased in either direction.
- NYISO (WULF's actual primary market) and CAISO/Mid-C are not fetchable in this repo — the
  miner arm's WULF leg and any WA-exposure share (BTDR) are untested against their real
  settlement markets.
- The train window (2023-2024) is the same two years used to select the hub for every name;
  with 19 candidates and a two-year window, some in-sample significance is expected by chance
  even under the split-sample discipline — TLN's t=+5.81 (§6) is the clearest illustration of
  why the test-window number, not the train number, is the one that gates.
