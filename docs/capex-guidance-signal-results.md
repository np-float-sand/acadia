# Utility Capex-Guidance Revision Signal ("Deliverable D") — results (2026-09-05)

**Status: FAILED — not adopted.** The aggregate utility capex-guidance revision signal
does not clear its pre-registered timing bar (spec §5) on any of the six series ×
three horizons, and the one exposure-overlay leg that the data actually exercises (the
two-sided scaler, spec §6.2) does not clear the de-risk/scaler gate. The one-directional
de-risk leg (§6.1) turned out to be **untestable in this window** — see §5.1. Logged as
**attempt #15** against the grid-equipment-basket theme's search for a signal-based edge
(spec §1's count; the running ledger is fuzzy — a couple of memory notes number a parallel
probe "~16" — but the conclusion is the same: no prior attempt has been adopted).

Spec: `docs/superpowers/specs/2026-09-04-capex-guidance-signal-design.md`.
Plan: `docs/superpowers/plans/2026-09-04-capex-guidance-signal.md`.
Handoff this follows: `docs/handoff_2026-09-03-transmission-project-filings.md` §6
(Deliverable D — "do first, cheapest, transcript-only, no scraping").

---

## 1. TL;DR

- **First test of the *commitment*-data class.** Everything tried before it
  (`docs/pjm-large-load-vintages-2026-09-03.md` Table B-9 MW forecasts,
  `docs/va-transmission-filings-probe-results.md` CPCN project-$, big-four capex-decel,
  grid-congestion regime, FTR bids) was *forecast* or *physical-flow* data that failed
  as a timing signal with quarterly corr ≈ 0. A raised 5-year capex-guidance number is
  a firm forward obligation the utility has already made to regulators — the hypothesis
  was that Δ(aggregate stated guidance) leads the equipment basket 1–3 quarters. **It
  does not, at monthly resolution over 2023-01 → 2026-08.**
- **Timing bar (spec §5): FAIL on every cell.** The two whole-panel revision-flow
  series (`all_usd`, `all_pct`) are the only ones with enough history to test. Their
  primary-window rank-IC vs forward basket return is **negative at every horizon**
  (−0.07 to −0.27) — the *opposite* sign to the thesis — and no rank-IC t clears
  |t| ≥ 2 (best −1.54). **3 of the 12 testable without-hyperscaler control-t cells**
  (2 series × 3 horizons × 2 books = 12; the other 24 cells are the n/a `dc_*` series)
  cross |t| ≥ 2, all with the wrong sign and none matched by a significant rank-IC —
  exactly the multiple-comparisons pattern spec §11 says to treat as noise, not signal.
- **The one holdout cell that clears |t| ≥ 2 is a 5-point sample that flips sign.**
  `all_pct` h=1m has a holdout rank-IC of +0.60 (t = +2.47) on **n = 5** months —
  *positive*, against a primary-window IC of −0.24 for the same cell. It is non-gating
  (spec §5.4 gates on the primary-window rank-IC + control-t), and the sign flip between
  a window and its own trailing carve-out is itself the signature of noise. Reported, not
  promoted.
- **DC-attributed sub-signal (spec §4.2): could not be tested at all.** Every stated
  data-center dollar figure in the panel is a 2026 vintage (DTE 2026-02-17, WEC
  2026-03-02, FE 2026-02-19, ETR 2026-04-29), so the composite's trailing z-score
  (252-day warm-up) never becomes non-NaN inside the test window. `dc_stated_*` and
  `dc_filled_*` return n/a for every horizon. This is a data-history limitation, not a
  code failure — reported as n/a, never fabricated.
- **De-risk leg (spec §6.1): NOT EXERCISED — no verdict on merit.** The `all_usd`
  composite's z-score **never drops below +0.212** anywhere in the span (the TTM
  revision-flow series trends up over 2024 → 2026 with no sustained pullback), so there is no
  capex-deceleration episode in this panel. `guidance_derisk_multiplier` is therefore a
  **constant 1.0 on every day of both windows across all 9 grid combos**
  (`active_days_primary = active_days_prior = 0` for every row), and its reported
  metrics are numerically identical to the **vol-target-only** baseline. The gate
  machinery prints FAIL; the honest reading is **not applicable** (§5.1).
- **Scaler leg (spec §6.2): tested and FAILED.** The two-sided scaler *is* genuinely
  exercised — `clip(1 + k·z, lo, hi)` moves off 1.0 for any non-zero z, and it is active
  on 685 of the primary window's days. It misses **G1** (−0.056 Sharpe / −0.550 Calmar
  vs layer-1) across its whole 6-combo plateau. Caveat on what exactly failed: because
  the composite z never goes negative, the realised multiplier range is **[1.000, 1.500]**
  — it **only ever leaned in, never out**. What failed is "lever up when aggregate
  guidance revisions accelerate", not the two-sided rule in full.
- **Feasibility (spec §3.3): PASS, 14/15 utilities usable** — well above the
  `DC_GUIDANCE_MIN_UTILITIES = 8` floor, so the full-panel signal ran (no fallback to
  the 3-name DC-heaviest set). NEE is the sole exclusion, for substantive reasons (§3).
- **Run data:** FULL. All 17 tickers (9 basket + 6-name DER short sleeve + SMH + ^TNX)
  downloaded live and the big-four hyperscaler capex series loaded from cache, so every
  leg — basket timing, long/short-spread timing, both control specs, both gates — ran on
  complete data. Nothing was degraded for missing prices.

---

## 2. Data quality — the seed panel

`grid_resilience/data/seed/utility_capex_guidance.csv` — 44 guidance-vintage rows across
the 15 utilities frozen in `config.DC_GUIDANCE_UNIVERSE` (D, AEP, NEE, SO, ETR, XEL, DUK,
PCG, EIX, PPL, FE, AEE, WEC, CMS, DTE), 2022–2026, 1–5 vintages each. Hand/web-assembled
from primary sources (earnings releases, 8-Ks, 10-Ks, investor-day press releases) where
possible, trade-press summary where not. **Every row carries an H/M confidence tag; no
row is L** (Task 3 either found a defensible primary/multiply-corroborated figure or
downgraded to M and stated in `source_detail` exactly what is weaker). Full per-utility
research summary: `.superpowers/sdd/2026-09-04-capex-guidance-signal/task-3-report.md`.

| confidence | rows | share |
|---|---|---|
| H (direct primary-source figure) | 37 | 84% |
| M (secondary reporting / 403'd primary / two-source derived figure) | 7 | 16% |
| L | 0 | 0% |

**The 7 M rows:** ETR 2026-02-12 (intermediate $43B step, corroborated by two secondary
sources, no primary doc with an aggregate figure), ETR 2026-07-29 (single secondary recap;
also a 4yr→5yr horizon change), PCG 2025-09-29 (primary DCD write-up 403'd on fetch),
EIX 2026-02-18 (10-K figure is fine but the 4yr→5yr horizon change makes the derived
revision partly an artifact), FE 2026-02-19 (the one `derived` DC-$ figure: 4.1 GW
contracted × $250M/GW, the two inputs from separate secondary sources), AEE 2025-02-14
(WebSearch summary only), WEC 2025-10-30 (WebSearch summary for the $ figure; only the
date independently confirmed).

Per-utility vintage counts and in-window (2023-01 → 2026-08) revisions:

| utility | vintages | in-window revisions | usable |
|---|---|---|---|
| D | 3 | 2 | yes |
| AEP | 4 | 3 | yes |
| NEE | 1 | 0 | **no** |
| SO | 4 | 4 | yes |
| ETR | 5 | 5 | yes |
| XEL | 3 | 2 | yes |
| DUK | 4 | 4 | yes |
| PCG | 3 | 2 | yes |
| EIX | 2 | 1 | yes |
| PPL | 2 | 1 | yes |
| FE | 2 | 1 | yes |
| AEE | 3 | 3 | yes |
| WEC | 3 | 3 | yes |
| CMS | 3 | 2 | yes |
| DTE | 2 | 2 | yes |
| **total** | **44** | — | **14 / 15** |

**Six rows carry a stated revision but no prior plan level.** AEE 2024-02-23, DTE
2025-02-13, DUK 2024-02-08, ETR 2025-02-18, SO 2024-02-15 and WEC 2024-10-31 are each a
utility's first vintage in the panel, so `prior_capex_plan_usd_m` is NaN even though the
company stated a revision figure. `aggregate_revision_series` drops these rows from
**both** sides of the size-weighted percent ratio. (Until the 2026-09-05 fix it kept them
in the numerator with a zero denominator contribution, which inflated `all_pct` by a
time-varying factor — ≈2.4× in 2024-11 decaying to 1.0× by 2026-02 — and manufactured a
spurious downtrend across the primary window. The `all_pct` cells in §4 are the corrected
ones; `all_usd` has no denominator and was never affected.)

**Data-center-attributed dollar coverage is thin and late.** Only 4 rows carry a
`stated`/`derived` DC-$ figure (ETR $15.0B, DTE $2.0B, FE $1.03B derived, WEC $1.0B) and
**all four are 2026 vintages**. The point-in-time EM-style imputation (spec §4.2) fills 7
more `qualitative`/`none` rows, but because its `ratio(t)` can only calibrate off
already-disclosed rows, every imputed value is also 2026-dated. Consequence: the
DC-attributed cut of the signal has **zero testable history inside the primary window**
(§4). Several utilities that the handoff assumed would give the cleanest DC attribution
turned out to disclaim it on record — **FPL's president (Jan 2026): the 2025 rate case
"was not driven by data centers"; SCE's CEO (Nov 2025): SCE "hasn't seen the same level
of demand from data centers as other utilities."** Both recorded as `dc_basis=none`.

---

## 3. Feasibility kill (spec §3.3) — PASS on the full panel

`feasibility_summary(panel, window=("2023-01-01","2026-08-31"))` → **n_usable = 14 / 15**
(a utility is "usable" with ≥1 non-null plan level *and* ≥1 non-null revision dated inside
the window). 14 ≥ `DC_GUIDANCE_MIN_UTILITIES = 8`, so `universe_used = "full"` — the
report ran on all 14, no fallback to `DC_GUIDANCE_FALLBACK = [D, AEP, NEE]`.

**NEE is the one exclusion, and it is a substantive negative finding, not a research gap.**
Task 3 spent the most search effort of any utility on NEE and found exactly one usable
vintage: FPL's "$90–100B through 2032" rate-agreement figure (H confidence, triple-
corroborated). NEE's holding-company structure — a regulated FPL rate-base plan versus an
unregulated NextEra Energy Resources renewables/storage backlog reported in GW, not a
comparable consolidated $ total — makes a second same-basis vintage unobtainable, so it
has a plan level but **0 in-window revisions**. And FPL's own president stated on record
(Jan 2026) that the plan "was not driven by data centers" — undercutting the handoff's
prior that NEE would be one of the cleanest DC-attribution names.

---

## 4. Test 1 — timing bar (spec §5)

Monthly observations, primary window **2023-01 → 2026-08**. At each month-end: the
composite's value vs the basket's forward 1 / 3 / 6-month return (NAV-resampled,
monthly-hold convention). **Rank-IC t** = HAC/Newey-West t on the slope of
`rank(fwd_return) ~ rank(signal)`, lag = h (spec §5.1). **Control-t** = the signal
coefficient's HAC t in `fwd_return ~ signal_z + Δ10y + SMH_return` (spec §5.2, the
gating "without-hyperscaler" spec) and, as non-gating context, the same regression
**+ big-four `decel2`**. **Combined verdict (spec §5.4): a cell passes only if BOTH the
primary-window rank-IC t AND the without-hyperscaler control-t clear |t| ≥ 2.**

The big-four hyperscaler control is aligned to its **`known_date`** (fiscal-quarter end
+ 50 days, when the prints are actually public), not to the quarter end — the same
point-in-time stamp `capex_signal.capex_derisk_multiplier` uses. Indexing on the quarter
end would have regressed forward returns on a control that was not yet observable at the
observation date.

### 4.1 vs the basket

The **holdout** column is the same rank-IC re-computed on
`DC_GUIDANCE_HOLDOUT_WINDOW = 2026-03-01 → 2026-08-31`, the trailing 6 months carved out
as the out-of-sample check spec §7 pre-registered in place of a usable prior window. At
h = 3m and h = 6m the holdout has too few forward-return observations to fit (n = 3 / 0).

| series | h | rank-IC (t) [n] | holdout IC (t) [n] | ctrl-t w/o hyperscaler | ctrl-t w/ hyperscaler | PASS? |
|---|---|---|---|---|---|---|
| `all_usd` | 1m | −0.157 (−0.75) [22] | +0.10 (+0.25) [5] | −0.71 | +0.12 | fail |
| `all_usd` | 3m | −0.266 (−1.39) [20] | n/a [3] | **−2.08** | −0.56 | fail |
| `all_usd` | 6m | −0.191 (−1.10) [17] | n/a [0] | −1.62 | −1.53 | fail |
| `all_pct` | 1m | −0.236 (−0.77) [13] | +0.60 (+2.47) [5] | −1.77 | −4.08 | fail |
| `all_pct` | 3m | −0.236 (−1.54) [11] | n/a [3] | −0.11 | −0.19 | fail |
| `all_pct` | 6m | −0.071 (−0.39) [8] | n/a [0] | **−3.06** | −2.69 | fail |
| `dc_stated_usd` | 1/3/6m | n/a [0] | n/a [0] | n/a | n/a | fail |
| `dc_stated_pct` | 1/3/6m | n/a [0] | n/a [0] | n/a | n/a | fail |
| `dc_filled_usd` | 1/3/6m | n/a [0] | n/a [0] | n/a | n/a | fail |
| `dc_filled_pct` | 1/3/6m | n/a [0] | n/a [0] | n/a | n/a | fail |

- **Every primary-window rank-IC point estimate is negative.** More aggregate
  capex-guidance-raising is associated with *lower* forward basket returns — the opposite
  sign to the "guidance leads spend leads equipment orders" thesis. None is significant
  (best `all_pct` h=3m, t = −1.54).
- **The two control-t cells here that cross |t| ≥ 2** (`all_usd` h=3m −2.08, `all_pct`
  h=6m −3.06) are both the wrong sign, neither is matched by a significant rank-IC, and
  the h=6m cells sit on n=8 with 3–4 regressors plus a constant. Per spec §5.4 neither is
  a pass; per spec §11 they are multiple-comparisons noise (3 of 12 testable cells across
  §4.1 + §4.2, scattered sign/horizon, no rank-IC support).
- **The `all_pct` h=1m holdout cell (+0.60, t = +2.47, n = 5) is the only |t| ≥ 2 rank-IC
  anywhere in the report**, and it is *positive* while the same cell's primary-window IC
  is −0.24. Five monthly observations, sign opposite the in-window estimate: this is what
  noise looks like, not an out-of-sample confirmation. It is also non-gating by
  construction (spec §5.4 gates on the primary-window rank-IC).
- **The DC-attributed rows are structurally untestable in this window** (§2): all stated
  and all imputed DC-$ values are 2026-dated, so the composite's 252-day z-score warm-up
  never completes before the span ends and every monthly value is NaN → `_rank_ic`
  returns n/a (n < 5). Not a bug — the graceful path; `guidance_composite`'s empty-Series
  guard was never hit because each of the six aggregate series had ≥1 event.

### 4.2 vs the long/short spread (basket − equal-weight DER short sleeve ENPH/SEDG/CHPT/RUN/BLNK/STEM)

| series | h | rank-IC (t) [n] | ctrl-t w/o hyperscaler | PASS? |
|---|---|---|---|---|
| `all_usd` | 1m | +0.098 (+0.50) [22] | +0.35 | fail |
| `all_usd` | 3m | +0.293 (+1.16) [20] | +1.21 | fail |
| `all_usd` | 6m | −0.152 (−0.74) [17] | +0.45 | fail |
| `all_pct` | 1m | 0.000 (−0.00) [13] | −0.74 | fail |
| `all_pct` | 3m | +0.118 (+0.57) [11] | −0.94 | fail |
| `all_pct` | 6m | −0.024 (−0.17) [8] | **−3.34** | fail |
| `dc_*` (4 series) | 1/3/6m | n/a [0] | n/a | fail |

Signs flip between horizons; nothing approaches the bar. The lone |t| ≥ 2 control cell
(`all_pct` h=6m, −3.34 on n=8) is the same small-sample degeneracy as in 4.1.

### 4.3 Continuity — lead-lag correlation (spec §5.3)

Plain Pearson corr of the monthly composite vs the 1-month-forward basket return,
shifted k = −6…+6 (negative k = signal leads):

| k | −6 | −5 | −4 | −3 | −2 | −1 | 0 | +1 | +2 | +3 | +4 | +5 | +6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `all_usd` | +0.30 | +0.35 | −0.05 | −0.23 | +0.06 | +0.05 | −0.20 | −0.09 | +0.06 | −0.22 | +0.07 | +0.05 | +0.31 |
| `all_pct` | −0.50 | +0.16 | −0.29 | −0.11 | +0.36 | +0.11 | −0.34 | +0.27 | +0.15 | −0.26 | −0.21 | −0.17 | +0.31 |

No coherent lead structure — the series oscillates around zero, with the largest
magnitudes at the extreme lags (k = ±6), the classic "no real relationship" shape.
This is *weaker* than the two prior negatives: the VA probe and PJM Table B-9 at least
produced a spurious +0.8 at k = −1 on annual (n = 7) data; at monthly resolution here
even that artifact is absent.

### 4.4 Sub-threshold t-stat inventory (|t| ≥ 1.5)

The pass bar (spec §5.4) is |t| ≥ 2 on **both** the primary-window rank-IC and the
gating without-hyperscaler control-t; §4.1/§4.2 show nothing clears it. This subsection
reports the fuller picture immediately below that bar — every cell in the §4.1/§4.2
tables with |t| ≥ 1.5 — on the reasoning that a weak effect could still matter combined
with another signal or used nonlinearly. Ten cells qualify. "n" is the monthly-observation
count reported for that series × horizon × window in §4.1/§4.2.

| series | horizon | book | statistic | t | n | sign vs thesis |
|---|---|---|---|---|---|---|
| `all_usd` | 3m | basket | control-t, w/o hyperscaler (**gating**) | −2.08 | 20 | wrong (inverse) |
| `all_usd` | 6m | basket | control-t, w/o hyperscaler (gating) | −1.62 | 17 | wrong (inverse) |
| `all_usd` | 6m | basket | control-t, w/ hyperscaler (non-gating) | −1.53 | 17 | wrong (inverse) |
| `all_pct` | 1m | basket | control-t, w/o hyperscaler (gating) | −1.77 | 13 | wrong (inverse) |
| `all_pct` | 1m | basket | control-t, w/ hyperscaler (non-gating) | −4.08 | 13 | wrong (inverse) |
| `all_pct` | 1m | basket | holdout rank-IC t | +2.47 | 5 | "right", but sign-flipped vs its own −0.24 primary IC |
| `all_pct` | 3m | basket | primary rank-IC t | −1.54 | 11 | wrong (inverse) |
| `all_pct` | 6m | basket | control-t, w/o hyperscaler (gating) | −3.06 | 8 | wrong (inverse) |
| `all_pct` | 6m | basket | control-t, w/ hyperscaler (non-gating) | −2.69 | 8 | wrong (inverse) |
| `all_pct` | 6m | spread | control-t, w/o hyperscaler (gating) | −3.34 | 8 | wrong (inverse) |

Of the ten cells at |t| ≥ 1.5, nine are negative. That is not scatter around zero — it is
a weak but directionally consistent *inverse* association: heavier aggregate
capex-guidance-raising precedes *lower* forward basket (and long/short-spread) returns.
That is the opposite sign to this probe's pre-registered thesis ("guidance leads spend
leads equipment orders") and the correct sign for a crowding / already-priced-in read —
the buildout is loudest in the guidance once it is already in the equities, so a fresh
raise coincides with, or slightly lags, a local top rather than leading a rally. It does
not change the verdict: per spec §5.4 a cell passes only when the primary-window rank-IC t
**and** the primary without-hyperscaler control-t both clear |t| ≥ 2, and no cell does;
the gating control-t cells that cross |t| ≥ 2 at all (`all_usd` h=3m −2.08 and `all_pct`
h=6m −3.06 vs the basket, `all_pct` h=6m −3.34 vs the spread) are every one the wrong sign
and none is matched by a significant rank-IC. Several reasons not to over-read the inverse
sign: the strongest cells (|t| 2.7–4.1) are all `all_pct` on n = 8–13 with 3–4 regressors
plus a constant — near-singular small samples; the §4.3 lead-lag table shows no coherent
structure, with the largest magnitudes at the extreme lags k = ±6; and this is ~36
statistical results in total, so a same-sign cluster at |t| 1.5–2 is within noise for a
weak real effect. The one thing worth carrying forward: the sign consistency says that if
this signal is ever revisited it should be tested *inverted* — a fade-the-guidance-raise
overlay — and/or as one input in a combined or nonlinear model, not in the thesis
direction as a standalone.

---

## 5. Test 2 — de-risk / scaler gate (spec §6)

Both variants feed `overlay.apply_overlay_l2(basket_ret, multiplier, rf)` and are scored
with the same `grid_regime.gate_check` / `final_verdict` machinery the layer-2 congestion
ladder and the FTR-bid signal use. The composite is the headline `all_usd` series.

**Read the exercised-day count first.** `derisk_scaler_report` records, per grid combo,
`active_days_primary` / `active_days_prior` — the number of days in that window where the
multiplier differs from 1.0, counted on the composite's own daily-calendar index (the
primary window is 1 339 calendar days, the prior window 730). A combo with 0 active days
in a window was **not exercised** there: its returns are mechanically the
**vol-target-only** baseline, and its gate flags say nothing about this signal. That is
why the baseline table below carries a vol-target-only row, and why `guidance_table`
prints `active_days(primary)=…` on the same line as each gate verdict.

Baselines (primary = 2023-01 → 2026-08, n = 918 trading days; prior = 2021-01 → 2022-12,
n = 502):

| book | window | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|---|
| buy & hold | primary | 54.2% | 36.5% | 1.264 | −41.3% | 1.31 |
| vol-target only (trend gate off) | primary | 44.0% | 22.8% | 1.540 | −22.3% | 1.97 |
| layer-1 (trend gate + vol target) | primary | 40.1% | 21.2% | **1.507** | −18.4% | **2.18** |
| buy & hold | prior | 19.9% | 30.3% | 0.620 | −32.8% | 0.61 |
| vol-target only | prior | 15.7% | 21.9% | 0.591 | −26.7% | 0.59 |
| layer-1 | prior | 6.0% | 17.1% | **0.192** | −24.8% | **0.24** |

### 5.1 One-directional de-risk (`guidance_derisk_multiplier`, spec §6.1) — NOT EXERCISED

Central config `floor_z = −0.5, lo_mult = 0.6`; plateau `floor_z ∈ {−0.25, −0.5, −0.75}`
× `lo_mult ∈ {0.5, 0.6, 0.7}` (9 combos).

**This leg never fired, in either window, at any grid setting.** The `all_usd` composite
is non-NaN from **2024-10-16** (the 252-day z-score warm-up after the panel's first
qualifying revision) and from that day forward its value **never falls below +0.212** —
the TTM revision-flow series trends up over 2024 → 2026 with no sustained pullback, so **the panel contains
no capex-deceleration episode at all**. Every `floor_z` in the grid is negative, so
`guidance_derisk_multiplier` returns a **constant 1.0 for every day of both windows in all
9 combos**: `active_days_primary = 0` and `active_days_prior = 0` on all 9 grid rows.

The block below is therefore **not a result about this signal.** It is
`apply_overlay_l2` with a flat 1.0 multiplier, which is exactly the **vol-target-only**
baseline — compare row-for-row with §5's baseline table; the numbers are identical.

| window | CAGR | Sharpe | MaxDD | Calmar | Sharpe gap vs L1 | Calmar gap vs L1 |
|---|---|---|---|---|---|---|
| primary | 44.0% | 1.540 | −22.3% | 1.97 | +0.033 | −0.21 |
| prior | 15.7% | 0.591 | −26.7% | 0.59 | +0.399 | +0.345 |

| gate | value |
|---|---|
| **G1** (beats L1 on Sharpe AND Calmar, primary) | False — but the +0.033 Sharpe / −0.21 Calmar gap is **vol-target-only vs trend-gate+vol-target**, i.e. "trend gate off vs on". It contains no information about capex guidance. |
| **G2** (prior) | True — same caveat, same flat multiplier |
| **G3** (annual return drag vs BH ≤ L1's, both windows) | True — same caveat |
| marginal | False |
| **neighbour plateau** | all 8 non-central combos are the *identical* constant-1.0 series, so their matching G1=False / G2=True / G3=True is a tautology, not a plateau probe |
| **verdict** | machinery prints **FAIL**; the honest reading is **not applicable — leg not exercised** |

**De-risk gate: n/a, not on merit.** A one-directional de-risk rule cannot be graded on a
window that contains nothing to de-risk into. Re-testable only once the panel covers a
period of decelerating or reversing aggregate guidance — which, on this evidence, has not
happened since the series became measurable.

### 5.2 Two-sided scaler (`guidance_scaler`, spec §6.2) — tested, FAILED

Central config `k = 0.35, lo = 0.5, hi = 1.5`; plateau `k ∈ {0.25, 0.35, 0.45}` ×
`hi ∈ {1.25, 1.5}` (6 combos).

**This leg is genuinely exercised.** `clip(1 + k·z, lo, hi)` moves off 1.0 for any
non-zero z, so the same all-positive composite drives it on **685 of the primary window's
1 339 days** (`active_days_primary = 685` — every day the composite is non-NaN; all 6
combos identical, since `k`/`hi` change the size of the tilt, not when it is on).
`active_days_prior = 0`: the prior window is entirely pre-warm-up, so its block is the
same inert vol-target-only book as §5.1's.

**One-sided in practice:** because the composite z never goes negative, the realised
multiplier range over both windows is **[1.000, 1.500]** — the scaler **only ever leaned
in, never out**. So what is shown to fail is specifically *"lever up when aggregate
guidance revisions accelerate"*; the de-lever half of the two-sided rule was never
reached, for the same reason §5.1's leg was not.

| window | CAGR | Sharpe | MaxDD | Calmar | Sharpe gap vs L1 | Calmar gap vs L1 |
|---|---|---|---|---|---|---|
| primary | 53.2% | 1.451 | −32.6% | 1.63 | **−0.056** | **−0.550** |
| prior | 15.7% | 0.591 | −26.7% | 0.59 | +0.399 | +0.345 |

| gate | value |
|---|---|
| **G1** (primary) | **False** — Sharpe −0.056, Calmar −0.550 |
| **G2** (prior) | True — but the prior-window block is the not-exercised flat-1.0 book (§5.3) |
| **G3** | True |
| marginal | False |
| **neighbour plateau** | all 5 non-central combos also fail G1 (primary Sharpe 1.451–1.499, Calmar 1.63–1.77, every one below layer-1's Sharpe 1.507 AND its Calmar 2.18) — a genuine plateau of *failing*, since these 6 combos really are 6 different multiplier series |
| **verdict** | **FAIL** (G1 false) |

Levering the basket up on an accelerating guidance flow adds return (CAGR 53.2% vs
layer-1's 40.1%) but adds drawdown faster (−32.6% vs −18.4%) — it is a beta-up trade
dressed as a signal, which is precisely what the Calmar half of G1 exists to catch.

### 5.3 Why G2 "passes" and why it does not rescue either leg

The composite is **entirely NaN before 2024-10-16** — the panel's first qualifying
revision is DUK 2024-02-08 and the trailing z-score needs a 252-day window before it
returns a number. Throughout the prior window (2021–2022) both the de-risk multiplier and
the scaler are therefore an inert constant 1.0, which is why their prior-window blocks are
byte-identical to each other *and* to the vol-target-only baseline. The "+0.40 Sharpe /
+0.34 Calmar" G2 pass is **not the signal working**: it is `apply_overlay_l2` with a flat
1.0 multiplier happening to beat the layer-1 trend-gate/vol-target book (Sharpe 0.19) in
the choppy 2021–2022 tape. Spec §7 pre-registered the prior window as structurally
uninformative for this signal and said a result there is not equivalent evidence — that
cuts both ways: the G2 "pass" carries no weight.

The gate that matters is **G1, the primary window** — and there, only the scaler leg has
anything to grade (§5.1 vs §5.2).

---

## 6. How this compares to the two prior forecast/flow negatives

| | PJM Table B-9 MW revisions | VA/Dominion CPCN project-$ | **this — utility capex-guidance revisions** |
|---|---|---|---|
| data class | forecast (planner estimate) | application (not yet a spend) | **commitment (stated forward obligation)** |
| doc | `docs/pjm-large-load-vintages-2026-09-03.md` | `docs/va-transmission-filings-probe-results.md` | this |
| feasibility | 6 vintages retrieved | 72 cases, easy | **14/15 utilities, 44 vintages** |
| timing result | annual corr +0.42 (n=5), quarterly ≈ 0; de-risked into a +30% rally | quarterly corr ≈ 0 at every lead; annual +0.8 at k=−1 is one shared 2022–24 inflection | **monthly rank-IC negative, no \|t\|≥2; no coherent lead-lag** |
| verdict | FAILED | logged, not extended | **FAILED** |

Deliverable D was the first probe of the untried *commitment* class and the one with the
best prior in the handoff (§6: "moderate — forward-looking and filing-based; the risk is
it's already in the equipment stocks' price given how covered this theme is"). That risk
is what the result looks like: the negative rank-IC sign is consistent with the equipment
names having *already* priced the capex cycle by the time the guidance number is printed,
so a fresh guidance raise coincides with or slightly lags a local top rather than leading
a rally.

---

## 7. Honesty caveats (verbatim from spec §11)

- The seed panel is web/knowledge-assembled, not a systematic transcript pull —
  confidence-tagged per row, and the results doc states the H/M/L mix plainly before any
  backtest number.
- The DC-attributed **imputed** series is a modeling construct, not disclosed fact —
  always shown next to the stated-only series, never presented alone.
- The prior window (2021-2022) is expected to be structurally uninformative for this
  signal (§7); a FAIL there is not treated as equivalent evidence to a FAIL in the
  primary window.
- This is one aggregate time series tested at monthly resolution over ~44 months
  (primary) — same small-n caution as every other signal in this theme's search.
- Six series (§4.1/§4.2) × three horizons × two control specs is a wide
  multiple-comparisons net for a single |t|≥2 bar; the results doc reports **all** cells,
  not just the best one, and treats a lone significant cell among many nulls as
  multiple-comparisons noise rather than a discovery.

Additional caveats specific to this run:

- **The de-risk leg was never exercised, so "FAIL" would overstate the evidence.** The
  composite has no sub-`floor_z` day anywhere in the span; §5.1 reports the leg as *not
  applicable*, and the metrics printed under it are the vol-target-only baseline's. The
  scaler leg's failure is real but is a failure of the *lean-in* half only — the realised
  multiplier range is [1.000, 1.500] and the de-lever half was never reached.
- **The DC-attributed cut of the signal (`dc_stated_*`, `dc_filled_*` — 4 of the 6
  series) was not testable at all.** Every stated and every imputed DC-$ value is a 2026
  vintage, so the trailing z-score never warms inside the window. These four series are
  reported n/a, not failed-on-merit. If the theme is revisited in ~2027 with 2027–2028
  vintages added, this cut becomes testable for the first time.
- **The scaler test is effectively a ~22-month test**, not the full 44-month primary
  window: the composite is NaN until 2024-10-16, so only 685 of the primary window's
  1 339 days carry a live multiplier.
- **Decision-timing asymmetry (favours the signal, and it still failed).**
  `guidance_derisk_multiplier` / `guidance_scaler` apply the composite's **same-day**
  value, whereas every other multiplier graded through `gate_check` in this package
  (`overlay.trend_gate`, `grid_regime.regime_multiplier`) is passed through
  `overlay._month_hold` first — decided at the prior month-end and held. So the guidance
  overlay trades on a slightly fresher signal than the layer-1 baseline it is scored
  against. This is spec-conformant ("the most-recently-known composite", §6.1) and the
  bias runs *in the signal's favour*, so the FAIL conclusion is safe; it is left in place
  rather than "fixed" so the published numbers match the spec as pre-registered. Anyone
  re-testing this on a passing signal should month-hold the multiplier first.
- Several M-confidence rows involve a 4yr→5yr horizon redefinition (EIX, ETR 2026-07-29),
  so the loader's derived revision for those vintages is partly a horizon-length artifact.

---

## 8. Verdict and bookkeeping

**FAILED — not adopted.** No series/horizon cell passed the combined rank-IC +
control-regression bar (spec §5); of the two exposure-overlay legs (spec §6), the scaler
missed G1 across its entire 6-combo plateau and the de-risk leg could not be graded at all
because the panel contains no deceleration episode to grade it on. The signal's
primary-window rank-IC sign is *negative*, consistent with the handoff's stated risk that a
heavily-covered theme is already priced into the equipment names before the guidance number
is public.

This is **attempt #15** against the grid-equipment-basket theme's search for a
signal-based stock-selection or timing edge (spec §1's count; ~14 prior, none adopted).
It is the **first test of the *commitment*-data class** the handoff §6 identified as
untried after PJM Table B-9, the VA CPCN probe, big-four capex-decel, grid congestion,
and FTR bids — and it lands the same way: no signal times this trade.

Of the handoff's Part-2 commitment-data deliverables, **D is now done (FAIL)**;
**C (state-PUC data-center tariff / ESA contracted-MW) and E (county DC permit /
zoning / abatement filings) remain** — 2 deliverables before the handoff's ~17-attempt
stop rule. Per handoff §6: if C and E also come back with quarterly rank-IC ≈ 0, that is
~17 attempts and the instruction is to **stop looking for a data-center demand signal and
ship the grid/DER trade as a discretionary thematic position**
(`docs/handoff_2026-09-01-grid-buildout-long-short.md` §7), keeping the E data capture
running for a 2028 re-test.

**No code is wired live.** `capex_guidance_signal.py` and the `--capex-guidance` CLI flag
stay in the repo as a runnable probe; nothing in `config.py` gates the basket on this
signal (contrast `REGIME_ENABLED`, which the layer-2 rung-1 overlay does use).

### Artifacts from the run

- `output_grid_equipment/capex_guidance_timing.csv` — per-series/horizon rank-IC,
  rank-IC t, without-hyperscaler control-t, pass flag (gitignored).
- `output_grid_equipment/capex_guidance_gates.json` — `universe_used`, full feasibility
  breakdown, and both gate verdicts with G1/G2/G3 (gitignored).
- Seed panel: `grid_resilience/data/seed/utility_capex_guidance.csv` (44 rows, kept).
- Task-3 research detail: `.superpowers/sdd/2026-09-04-capex-guidance-signal/task-3-report.md`.
- Final-review fix pass (2026-09-05): de-risk not-exercised framing + `active_days_*`
  instrumentation, `all_pct` NaN-denominator correction, empty-Series DatetimeIndex
  guards, big-four control re-aligned to `known_date`. `all_usd` and both gate verdicts
  are unchanged by all four.
