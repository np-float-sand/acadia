# Design Spec — Utility Capex-Guidance Revision Signal ("Deliverable D")

**Status:** design approved, not yet built.
**Date:** 2026-09-04
**Follows:** `docs/handoff_2026-09-03-transmission-project-filings.md` §6 (Deliverable D — "do
first, cheapest, transcript-only, no scraping"), which itself follows the VA transmission-filings
probe (`docs/va-transmission-filings-probe-results.md`) and PJM Table B-9 MW-revision signal
(`docs/pjm-large-load-vintages-2026-09-03.md`) — both **forecast/flow** data that failed as
timing signals (quarterly corr ≈ 0). D is the first test of the untried class: **commitment**
data (a raised capex-guidance number is a firm forward obligation, not a planner's estimate).
**Prior context (memory):** `capex-cycle-pair-classifier.md`, `va-transmission-filings-probe-negative.md`,
`pjm-mw-revision-signal-negative.md`. This is attempt #15 against the grid-equipment-basket theme's
search for a signal-based edge (14 prior, all negative or not-adopted).

---

## 1. Thesis and what this is / is not

**Thesis:** aggregate US electric utilities keep a rolling forward 5-year capital program and
revise it at each earnings call / 10-K / analyst day. Since ~2023 those revisions have been
increasingly driven by data-center / large-load interconnection requests, and utilities
increasingly say so explicitly. A raised guidance number is a *stated commitment* the utility
has already made to regulators and investors — unlike a PJM load forecast (a planning estimate,
revised down as easily as up) or a CPCN filing (an application, not yet a spend commitment), it
leads spend by construction. Grid-equipment makers' order books are downstream of exactly this
capital program, so the hypothesis is that Δ(aggregate stated capex guidance) leads the
equipment-basket's price 1–3 quarters, because the equity market prices customer capex
*guidance* ahead of the capex itself.

**What this is:** one aggregate time-series signal (not cross-sectional — it does not rank or
re-weight the 9 basket names) tested two ways: (1) as a **timing/rank-IC signal** against forward
basket and long/short-spread returns, and (2) as an **exposure scaler**, the same seam layer-1
(price trend gate + vol target) and layer 2 (grid-congestion regime, `REGIME_ENABLED=False`) and
the FTR-bid signal (`ftr_signal.py`, FAILED) already occupy.

**What this is not:** it is not a cross-sectional name-selection factor (no utility here is a
basket member — these are the basket's *customers*), and it is not built on any new data
infrastructure — the seed panel is a hand/web-assembled table, same posture as
`va_transmission_projects.csv` and `pjm_large_load_b9_vintages.csv`.

**Relationship to `capex_signal.py` (the big-four hyperscaler capex-deceleration signal):** that
signal tracks MSFT/GOOGL/AMZN/META capex from XBRL — the *hyperscalers'* own capex, one layer
upstream of the grid. This signal tracks the *utilities'* capex guidance — one layer downstream,
closer to the equipment basket's actual revenue. The handoff explicitly requires showing this
signal adds information **beyond** the big-four series, so `capex_signal.fetch_bigfour_capex` is
wired in as a control (§5.3), not left as an unrelated prior result.

---

## 2. Scope

**In scope (build now):** Deliverable D only — the aggregate utility capex-guidance revision
signal, both as a timing test and as an exposure scaler, pre-registered per handoff §6's shared
gate.

**Out of scope:** Deliverables C (DC tariff/ESA filings) and E (county DC permits) — handoff's
order-of-attack runs D first and independently; C and E are separate future specs. Any change to
basket composition or name weights. Any change to layer 1's or layer 2's existing code paths
(this is a new, independent rung, evaluated against them, not merged into the `REGIME_LADDER`).

---

## 3. Data

### 3.1 Universe

The 15 utilities named in the handoff: **D, AEP, NEE, SO, ETR, XEL, DUK, PCG, EIX, PPL, FE, AEE,
WEC, CMS, DTE.** Frozen before any figure is looked up — not revised based on which utilities
turn out to have the cleanest data (that selection bias is exactly what the feasibility kill in
§3.3 is designed to surface honestly, not hide).

### 3.2 Seed panel — `grid_resilience/data/seed/utility_capex_guidance.csv`

One row per **guidance vintage**: a distinct dated point at which a utility stated or revised its
forward multi-year (usually 5-year, sometimes 4- or 6-year as companies redefine their program)
capital plan.

| column | type | notes |
|---|---|---|
| `utility` | str | ticker, one of §3.1 |
| `report_date` | date | date the number became public (earnings call / 10-K filing / analyst day / rate-case testimony). **This is the point-in-time knowable date** — unlike `capex_signal.py`'s 50-day XBRL-lag, the guidance figure *is* the public event, no extra lag applied. |
| `plan_start_year`, `plan_end_year` | int | the stated horizon (plans roll forward — cross-vintage *levels* are not directly comparable across different horizons, which is why the signal is built on revisions, §4.1) |
| `capex_plan_usd_m` | float | total $ over the stated horizon |
| `revision_vs_prior_usd_m` | float, nullable | company-stated Δ vs. its own immediately-prior plan ("raised the five-year plan by $X billion"); when not stated outright, computed as `capex_plan_usd_m − previous row's capex_plan_usd_m` for the same utility, annotated `revision_quality='derived'` (see below) |
| `revision_quality` | enum | `stated` (company gave the delta) / `derived` (loader computed it from consecutive levels) |
| `dc_attributed_usd_m` | float, nullable | portion the utility attributes to data centers / large load, only when a $ figure is given (mostly NaN pre-2024) |
| `dc_basis` | enum | `stated` (explicit $) / `derived` (back-computed from disclosed incremental large-load MW × a stated or typical $/MW) / `qualitative` (mentioned, no $) / `none` |
| `source_type` | enum | `earnings_call` / `10-K` / `analyst_day` / `rate_case` / `press_release` |
| `source_detail` | str | quote fragment or URL |
| `confidence` | enum | `H` (direct primary-source figure) / `M` (secondary reporting of the figure, e.g. a reliable trade-press summary) / `L` (my own reconstruction/estimate) |

Populated by web research, DC-heaviest names first (D, AEP, NEE, SO, DUK, ETR, XEL, PCG), then
the rest. **Every row carries its confidence tag**; the results doc leads with a data-quality
table exactly as `va-transmission-filings-probe-results.md` did for its own hand-built panel.
No figure is adjusted after seeing the backtest result — the panel is frozen once feasibility
(§3.3) is scored, same discipline as `config.UNIVERSE`'s "verified on 10-K business description
only, never on historical returns."

### 3.3 Feasibility kill (runs first, before any backtest)

`grid_resilience.data.utility_capex_guidance.feasibility_summary(df)` counts, per utility,
whether it has ≥1 usable plan figure and ≥1 in-window revision (2023-01 → 2026-08).

- **≥ 8 of 15 usable → build the full-panel signal.**
- **< 8 → fall back to `config.DC_GUIDANCE_FALLBACK = ["D", "AEP", "NEE"]`** (the DC-heaviest,
  most likely to have clean disclosure) and note reduced breadth in the results doc.
- **Even the fallback not assemblable → log "not yet testable," same verdict shape as C's
  expected outcome, and stop** — no partial signal gets backtested on a panel too thin to trust.

This check is printed first in `guidance_signal_report()`'s output and gates whether the rest of
the report runs at all (mirrors how `ftr_signal.py`'s composite returns empty rather than a
degenerate report when a data source is unusable).

---

## 4. Signal construction — `grid_equipment_basket/capex_guidance_signal.py`

### 4.1 Two units for the headline revision-flow signal

Levels aren't comparable across utilities of very different size, or across one utility's own
plans of different horizons, so the signal is built on **revisions**, in two forms (both built,
both tested — the $ form is primary per the handoff's framing, the % form is the documented
fallback if $ doesn't clear the gate):

- **`agg_revision_ttm_usd`** — trailing-4-quarter sum of `revision_vs_prior_usd_m` across every
  utility with a known revision in that trailing window, indexed by `report_date`. Directly
  parallel to the VA probe's Δproject-$ and Table B-9's Δrev, so a comparison to those two
  negative results is apples-to-apples.
- **`agg_revision_ttm_pct`** — **size-weighted** trailing-4-quarter percent version:
  `sum(revision_vs_prior_usd_m over the window) / sum(capex_plan_usd_m as most-recently-known
  before each revision)`. Size-weighted (not an equal-weighted average of each utility's own %)
  so one small utility's noisy 40% swing doesn't dominate the panel the way it would in a naive
  mean-of-percents.

### 4.2 DC-attributed sub-signal, built unconditionally, with a point-in-time EM-style fill

Built regardless of how many utilities give a stated data-center dollar figure (per your
instruction — not coverage-gated). Two layers:

1. **Stated/derived only** (`dc_revision_ttm_usd_stated`, `_pct`) — NaN wherever `dc_basis` is
   `qualitative`/`none`. Sparse, especially pre-2024, but never fabricated.
2. **Stated+imputed** (`dc_revision_ttm_usd_filled`, `_pct`, plus a `dc_imputed` boolean column
   on the panel) — for rows where `dc_basis` is `qualitative`/`none`, impute
   `dc_attributed_usd_m` as `revision_vs_prior_usd_m × ratio(t)`, where `ratio(t)` is the
   cross-sectional mean of `dc_attributed_usd_m / revision_vs_prior_usd_m` over every
   `stated`/`derived` row with `report_date ≤ t` — **only already-disclosed data as of that
   date, never a later quarter's disclosure** (point-in-time safety; unit-tested directly, §10).
   Refined **EM-style**: after the first fill, `ratio(t)` is recomputed including the newly
   filled rows (down-weighted 0.5× vs. genuinely observed rows, so imputed values can inform the
   ratio without dominating it), and the fill is redone; repeated up to 5 iterations or until the
   aggregate TTM series changes by < 0.5% between iterations, whichever comes first. Rows with no
   `stated`/`derived` observation anywhere in the panel *before* their own date keep
   `dc_attributed_usd_m_filled = NaN` — there is nothing yet to calibrate the ratio from, and the
   loader does not invent one.
   `grid_resilience/data/utility_capex_guidance.py:impute_dc_attributed(df)` implements this and
   is unit tested for convergence, point-in-time safety, and the "nothing to impute from yet"
   edge case (§10).

The results doc reports the stated-only and stated+imputed cuts **side by side** — the imputed
series is there to test "does a percent change in the covered DC piece matter," not presented as
if it were disclosed fact.

### 4.3 Composite: z-scoring and daily broadcast

Each of the six series above (`usd`/`pct` × `all-revision`/`dc-stated`/`dc-filled`) goes through
the same seam every other signal in this package uses:

```python
def guidance_composite(events: pd.Series, start: str, end: str, *,
                       zscore_window: int = config.DC_GUIDANCE_ZSCORE_WINDOW,
                       zscore_minp: int = config.DC_GUIDANCE_ZSCORE_MINP,
                       winsor: float = config.DC_GUIDANCE_ZSCORE_WINSOR) -> pd.Series:
    """`events` indexed by report_date (already point-in-time, no further lag).
    Forward-fill to a daily calendar index, then `grid_regime._trailing_zscore`.
    Matches the contract `grid_regime.regime_composite` / `ftr_signal.ftr_composite`
    already share, so it drops into `regime_multiplier`-style consumers unchanged."""
```

No additional publication lag is applied (§3.2) — the only "point-in-time" work already happened
in the seed panel's `report_date` and in §4.2's ratio calculation.

---

## 5. Test 1 — timing bar (pre-registered, handoff §6)

Monthly observations over the common window **2023-01 → 2026-08** (`config.DC_GUIDANCE_PRIMARY_WINDOW`).
At each month-end *m*: the composite's value as of *m*, vs. the basket's forward return over the
next **1 / 3 / 6 months**, and vs. the forward **long/short spread** (basket − equal-weight DER
short sleeve `config.DER_SHORT_SLEEVE = ["ENPH","SEDG","CHPT","RUN","BLNK","STEM"]`, the sleeve
already used in `docs/handoff_2026-09-01-grid-buildout-long-short.md` §2 /
`docs/pjm-large-load-vintages-2026-09-03.md`). Forward returns are computed off month-end NAV
levels (`(1+daily_ret).cumprod()` resampled to month-end), matching the monthly-hold convention
`overlay.py`'s `_month_hold` already uses elsewhere in this package.

### 5.1 Rank-IC

Spearman correlation between the signal level at *m* and the forward return, computed across all
months in the window, per horizon and per one of the six series in §4.1/§4.2. **Bar: |t| ≥ 2.**
Because the h=3 and h=6 forward windows overlap month to month, the t-stat is **not** the naive
Spearman significance test (which assumes independent observations) — it is the HAC/Newey-West
t-statistic on the slope of an OLS regression of `rank(forward_return)` on `rank(signal)`
(`statsmodels`, `cov_type="HAC"`, `maxlags=h`), which is numerically the Spearman correlation's
slope with a serial-correlation-robust standard error. One `_hac_ols(y, X, lag)` helper is shared
between this and §5.2.

### 5.2 Control regression — run twice

`fwd_return ~ signal_z + Δ10y_yield + SMH_return` (HAC SEs, same `_hac_ols` helper), and
separately `fwd_return ~ signal_z + Δ10y_yield + SMH_return + bigfour_decel2` (adding
`capex_signal.fetch_bigfour_capex()["decel2"]`, forward-filled to the same monthly index) — **run
with and without the hyperscaler control, both reported**, so it is visible whether the
hyperscaler series is what absorbs the signal's information or whether the signal survives
regardless. Δ10y from `^TNX` (level, then differenced), SMH from `fetch_prices`, both monthly.
**Bar: the signal coefficient's |t| ≥ 2 in the without-hyperscaler regression** (handoff's
literal requirement is "survives a Δ10y + SMH control"); the with-hyperscaler regression is
reported as the stricter, non-gating stretch goal per the handoff's separate additivity ask.

### 5.3 Continuity table

A plain Pearson lead-lag correlation table at k = −6…+6 months, same shape as the VA probe's
§3 table and the Table B-9 doc's revision-vs-forward-return table, printed alongside the formal
test so a reader can see at a glance whether this result looks different from the two prior
negatives or is the same "coincident, not leading" shape.

### 5.4 Combined verdict

The handoff states the timing bar as one requirement ("rank-IC ... |t| ≥ 2 ...; survives a
control regression"), not two independent options — **Test 1 passes for a given series/horizon
only if both §5.1's rank-IC bar and §5.2's without-hyperscaler control-regression bar clear
|t| ≥ 2.** Either alone is reported but does not count as a pass on its own; §5.3's table and
the with-hyperscaler regression are context, not additional pass conditions.

---

## 6. Test 2 — de-risk / scaler bar

### 6.1 One-directional de-risk (built and tried first)

```python
def guidance_derisk_multiplier(index, composite, *, floor_z=-0.5, lo_mult=0.6) -> pd.Series:
    """{lo_mult, 1.0} — steps to lo_mult once the most-recently-known composite
    z-score drops below floor_z (revision-flow decelerating/reversing); back to
    1.0 once it clears. Never exceeds 1.0. Same shape as capex_signal.capex_derisk_multiplier."""
```

Parameter plateau: `floor_z ∈ {-0.25, -0.5, -0.75}` × `lo_mult ∈ {0.5, 0.6, 0.7}` (9 combos).

### 6.2 Two-sided scaler (built and tried second, per your instruction — "it can also hold more")

```python
def guidance_scaler(index, composite, *, k=0.35, lo=0.5, hi=1.5) -> pd.Series:
    """clip(1 + k*z, lo, hi) -- can lean in when revision-flow accelerates, not
    only de-risk. Same shape as grid_regime's continuous-mode multiplier."""
```

Plateau: `k ∈ {0.25, 0.35, 0.45}` × `hi ∈ {1.25, 1.5}` (lo fixed at 0.5) — 6 combos.

### 6.3 The gate

Both variants feed `overlay.apply_overlay_l2(ret, mult, rf)` — the identical seam
`ftr_signal.py` and `grid_regime.py`'s ladder already use — and are scored with the **same
gate machinery**, imported directly rather than re-implemented:
`grid_regime.gate_check(rung_primary, rung_prior, l1_primary, l1_prior, bh_primary, bh_prior)` +
`grid_regime.final_verdict(gate, neighbours_pass)`. This is exactly "beats the static book
(layer-1: price trend gate + vol target) on Sharpe AND Calmar, both windows, non-marginal,
survives the parameter-neighbour plateau probe" — the handoff's literal de-risk/scaler bar.
`neighbours_pass` = every other combo in §6.1/§6.2's grid also passing G1 & G2.

---

## 7. Windows, and the thin-prior-window honesty note

Per handoff §6's shared pre-registration, **verbatim, not `REGIME_PRIMARY_WINDOW`/`REGIME_PRIOR_WINDOW`
(those differ — 2020 start, 2025-12-31 end — because they're layer-2's own frozen pair, not this
signal's)**:

```python
DC_GUIDANCE_PRIMARY_WINDOW: tuple[str, str] = ("2023-01-01", "2026-08-31")
DC_GUIDANCE_PRIOR_WINDOW: tuple[str, str] = ("2021-01-01", "2022-12-31")
```

**Known going in:** the DC-driven guidance-raising cycle is structurally a 2023+ phenomenon —
2021–2022 utilities were mostly holding or trimming plans (rate pressure, supply-chain costs, not
data centers). The prior-window revision series is expected to be near-flat or driven by
unrelated causes (fuel cost pass-throughs, storm hardening), which the de-risk/scaler gate's "both
windows" requirement will likely fail structurally, not because the signal is wrong in the
window it's actually built for. Per the handoff's own instruction for Deliverable C's equally
thin history: **score the timing test (§5) on primary + a forward holdout**
(`DC_GUIDANCE_HOLDOUT_WINDOW = ("2026-03-01", "2026-08-31")`, the trailing 6 months carved out
and re-tested as an out-of-sample check), and say plainly in the results doc that the prior-window
leg of the de-risk gate is expected to be structurally uninformative rather than silently reporting
a FAIL as if it were a fair test.

---

## 8. Public API

`grid_resilience/data/utility_capex_guidance.py`:

| function | purpose |
|---|---|
| `load_capex_guidance(seed=SEED_CSV) -> pd.DataFrame` | parse the seed CSV, compute `derived` revisions where blank |
| `feasibility_summary(df) -> dict` | per-utility usable-plan / revision counts; §3.3 |
| `panel_asof(df, asof) -> pd.DataFrame` | most-recent row per utility with `report_date ≤ asof` |
| `impute_dc_attributed(df) -> pd.DataFrame` | §4.2's point-in-time EM-style fill |
| `aggregate_revision_series(df, *, value_col, use_pct, ttm_quarters=4) -> pd.Series` | §4.1/§4.2's six series, one function parameterised by column + unit |

`grid_equipment_basket/capex_guidance_signal.py`:

| function | purpose |
|---|---|
| `guidance_composite(events, start, end, **zscore_kwargs) -> pd.Series` | §4.3 |
| `guidance_derisk_multiplier(index, composite, *, floor_z, lo_mult) -> pd.Series` | §6.1 |
| `guidance_scaler(index, composite, *, k, lo, hi) -> pd.Series` | §6.2 |
| `guidance_signal_report(price_fn=None, panel_df=None, ...) -> dict` | orchestrates §3.3 feasibility, §5 timing bar (all six series × three horizons × two control specs), §6 de-risk/scaler gate (both variants, full plateau) |
| `guidance_table(rep) -> str` | printable report, same shape as `capex_signal.capex_table` / `grid_regime.regime_table` (`ftr_signal.py` has no table function of its own — its report dict is consumed directly by the results doc, not printed via a CLI table) |

`grid_equipment_basket/config.py` additions: `DC_GUIDANCE_UNIVERSE`, `DC_GUIDANCE_FALLBACK`,
`DC_GUIDANCE_MIN_UTILITIES=8`, `DC_GUIDANCE_ZSCORE_WINDOW/MINP/WINSOR` (reuse `REGIME_*` values —
756/252/3.0 — not re-tuned), `DC_GUIDANCE_PRIMARY_WINDOW`, `DC_GUIDANCE_PRIOR_WINDOW`,
`DC_GUIDANCE_HOLDOUT_WINDOW`, `DER_SHORT_SLEEVE`, the §6.1/§6.2 plateau grids.

---

## 9. Reporting & CLI

`grid_equipment_basket/__main__.py`: new `--capex-guidance` flag (alongside `--overlay-l2`),
prints `guidance_table(rep)` and writes `capex_guidance_metrics.csv` +
`capex_guidance_timeline.csv` to the output dir, same shape as `_write_regime_outputs`.

Results doc: `docs/capex-guidance-signal-results.md` (via Bash, per CLAUDE.md), leading with the
data-quality table (§3.2), the feasibility-kill outcome (§3.3), then §5's timing-bar table (six
series × three horizons × with/without-hyperscaler control) and §6's gate table, closing with the
verdict and the "≥8 utilities ⇒ ~2 more Part-2 attempts (C, E) before the 17-attempt stop rule
per the handoff" bookkeeping line.

---

## 10. Testing (TDD)

- `tests/data/test_utility_capex_guidance.py`: loader schema + dtypes; `derived` revision
  computed correctly when `revision_vs_prior_usd_m` blank; `feasibility_summary` counts;
  `panel_asof` never returns a row with `report_date > asof`; `impute_dc_attributed` —
  point-in-time safety (a later `stated` row must not change an earlier row's imputed value),
  convergence within 5 iterations, and the "nothing to impute from yet" case stays NaN.
- `tests/grid_equipment_basket/test_capex_guidance_signal.py`: `guidance_composite` matches the
  `_trailing_zscore` contract (no future leak — mirrors `test_ftr_composite_applies_lag_and_zscores`'s
  no-leak assertion); `guidance_derisk_multiplier` / `guidance_scaler` mechanics on synthetic
  composites; `guidance_signal_report` end-to-end with injected `price_fn` and a synthetic panel
  (deterministic, no network I/O, same pattern as `test_ftr_signal_report_runs_baselines_and_gate_end_to_end`);
  feasibility-kill fallback path (panel with <8 usable utilities routes to `DC_GUIDANCE_FALLBACK`).

---

## 11. Honesty caveats (carried verbatim into the results doc)

- The seed panel is web/knowledge-assembled, not a systematic transcript pull — confidence-tagged
  per row, and the results doc states the H/M/L mix plainly before any backtest number.
- The DC-attributed **imputed** series is a modeling construct, not disclosed fact — always shown
  next to the stated-only series, never presented alone.
- The prior window (2021-2022) is expected to be structurally uninformative for this signal (§7);
  a FAIL there is not treated as equivalent evidence to a FAIL in the primary window.
- This is one aggregate time series tested at monthly resolution over ~44 months (primary) — same
  small-n caution as every other signal in this theme's search.
- Six series (§4.1/§4.2) × three horizons × two control specs is a wide multiple-comparisons net
  for a single |t|≥2 bar; the results doc reports **all** cells, not just the best one, and treats
  a lone significant cell among many nulls as multiple-comparisons noise rather than a discovery.

---

## 12. Deliverables

1. `grid_resilience/data/seed/utility_capex_guidance.csv` — the hand/web-assembled panel.
2. `grid_resilience/data/utility_capex_guidance.py` — loader, feasibility, imputation, aggregation.
3. `grid_equipment_basket/capex_guidance_signal.py` — composite, multiplier, scaler, gate report.
4. `grid_equipment_basket/config.py` — new constants (§8).
5. `grid_equipment_basket/__main__.py` — `--capex-guidance` flag.
6. `grid_equipment_basket/README.md` — updated per CLAUDE.md.
7. `tests/data/test_utility_capex_guidance.py`, `tests/grid_equipment_basket/test_capex_guidance_signal.py`.
8. `docs/capex-guidance-signal-results.md` — write-up, verdict, honesty caveats.
9. Memory index update pointing at this spec + the eventual results doc.
