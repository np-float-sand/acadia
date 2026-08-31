# Design Spec — Grid-Congestion Regime Signal (Layer 2) for the Grid-Equipment Basket

**Status:** Approved design, not yet built.
**Date:** 2026-08-31
**Location:** new module `grid_equipment_basket/grid_regime.py` + edits to `overlay.py`, `config.py`, `__main__.py`, `README.md`.
**Follows:** `docs/handoff_2026-08-31-two-layer-overlay.md`, `docs/handoff_2026-08-29-grid-equipment-basket.md`.
**Prior context (memory):** `grid-equipment-overlay.md`, `transmission-rate-base-negative.md` — five pre-registered
cross-sectional stock-selection attempts in this theme have all failed; layer 1 (price trend gate + vol target)
is the one robust positive but does not generalise to 2018–2022.

---

## 1. Thesis and what this is / is not

The grid-equipment basket's bull case is "the electricity grid is the bottleneck for the AI buildout."
Whether that bottleneck is **tightening or easing** is measurable in the physical wholesale-power market —
zonal transmission congestion and reserve-margin tightness in the ISO zones where data-center load is
concentrating — *before* it shows up in equipment-maker margins or order books, and it is not in prices,
price momentum, or analyst models. Layer 2 turns that read into a single **monthly exposure multiplier** on
the equal-weight basket.

**This is a timing signal on one exposure knob, not a cross-sectional factor.** It sets how much of the
basket to hold each month; it never re-weights names. Proposal D (`docs/triage_2026-08-31-proposals-bda.md`)
already showed that monthly change in the grid stress index barely co-moves with equity aggregates — that
was a *cross-sectional sensitivity* factor and it failed. Layer 2 is a different bet: a regime read that
substitutes for layer 1's price trend gate.

**Relationship to layer 1.** Layer 1 is `exposure = trend_gate(price vs MA-100) × vol_target_scalar`. Layer
1's own finding: the price `trend_gate` overfits the single 2024-11→2025-04 "DeepSeek" drawdown and *hurts*
on the 2018–2022 panel (Sharpe 0.69→0.20); the `vol_target_scalar` is the robust part. Layer 2 **replaces
the trend gate, keeps the vol target**:

```
exposure_l2 = regime_multiplier(grid_regime)  ×  vol_target_scalar(basket_returns)
```

**Success criterion (why we are allowed to build this):** layer 2 is worth shipping only if a
physically-grounded regime read beats the price trend gate at this job on **both** the primary (2023–2025)
and prior (2018–2022) windows. If it does not, the honest deliverable is "ship layer-1-only, or a smaller
un-overlaid basket" — an acceptable outcome, matching the five prior negatives.

---

## 2. Scope

**In scope (build now):** input #2 from the handoff — a zonal congestion / scarcity / reserve-tightness
regime signal → one basket-exposure multiplier, pre-registered on both windows.

**Designed-for but not built now:** input #3 — the RT/DA (real-time vs day-ahead) spread as an additional
sub-signal. The signal's internal composition is a `{subsignal_name: (series, weight)}` map reduced by one
helper (`_combine_subsignals`), so adding an RT/DA sub-signal later is: add one key + a `fetch_lmp_rt()` in
`grid_resilience/data/grid_data.py`, with no rework of composition, mapping, or gate. It is ladder rung 7 and
runs only if a cheap live data-availability probe passes (§7).

**Out of scope:** input #1 (interconnection-queue velocity) — not backtestable, only 11 monthly ERCOT
snapshots exist. Any change to name weights. Any change to layer 1's existing functions.

---

## 3. Data

### 3.1 Sources (all already cached on disk — the gated run does zero network I/O)

| Dataset | Cache prefix | Coverage on disk | Use |
|---|---|---|---|
| PJM zone-level LMP with energy/congestion/loss components | `grid_resilience/data/cache/pjm_lmp_ZONE_*.parquet` | 2018-01 → 2025-12, hourly | primary congestion signal for DC-heavy zones |
| PJM per-zone metered load | `pjm_load_zonal_*.parquet` (via `fetch_zonal_load`) | 2017-01 → present, hourly | per-zone reserve-tightness |
| ERCOT settlement-point LMP (no component breakdown) | `ercot_lmp_settlement point_*.parquet` | 2018-01 → 2025-12, hourly | ladder rung 6 only — LZ_WEST−LZ_NORTH inter-zonal *spread* proxy |

Access is through **`grid_resilience.data.grid_data` public functions only** (`fetch_lmp`,
`fetch_zonal_load`, `daily_lmp_summary`, `daily_spread_summary`), the same containment rule
`grid_equipment_basket/data/prices.py` already follows for the cache helpers. The heavy
`grid_resilience.signals.grid_stress_index.build_multi_iso_gsi` pipeline is **not** imported — it drags in
equity betas, a load-capacity proxy calibrated for a different purpose, and an always-on event flag we do
not want.

### 3.2 DC-heavy zones

PJM zone codes present in the LMP cache: `DOM, AEP, COMED, PPL, PSEG, ATSI` (+ AECO, APS, BGE, DAY, DEOK,
DPL, DUQ, EKPC, JCPL, METED, PECO, PENELEC, PEPCO, RECO).

- **Core set (rungs 1–3, 5):** `DOM, AEP, COMED, PPL` — DOM (Dominion / Northern Virginia) is the dominant
  US data-center load zone; AEP (Ohio), COMED (Chicago), PPL (eastern PA) are the named secondary clusters
  in `docs/triage_2026-08-31-proposals-bda.md` §A.
- **Wide set (rung 4+):** add `PSEG, ATSI`.

Load-zone codes in `fetch_zonal_load` differ (`CE`=COMED, `PL`=PPL, `PS`=PSEG …) and are normalised by
`grid_data._PJM_LOAD_ZONE_ALIASES` at the fetch boundary — `grid_regime.py` consumes the normalised names.

### 3.3 Backtest windows (frozen — trimmed to cached coverage)

- **Primary:** `2023-01-01 → 2025-12-31`. Deviates from the basket's own primary end of 2026-07-31 because
  the LMP cache ends 2025-12; documented as a deviation in the results doc.
- **Prior:** `2018-01-01 → 2022-12-31` (`config.PRIOR_REGIME_START` / `PRIOR_REGIME_END`).

**No live refresh in the gated run.** Extending past 2025-12 needs a rate-limited `PJM_API_KEY` pull that
would inject a few months of differently-timed data at the most decision-relevant end of the sample — worse
than a clean frozen window. A `--refresh-regime` CLI flag (off by default) is the only code path that hits
the network; it is for later live use.

---

## 4. The signal — `grid_equipment_basket/grid_regime.py`

### 4.1 Daily construction, per zone

1. **Congestion level.** From hourly zone LMP rows, per day: `daily_cong = mean( |congestion| )` in absolute
   `$/MWh`. **Absolute dollars, not the `|congestion| / |LMP|` ratio** — a descriptive pass over 2018–2025
   showed the ratio blowing up to 388% at COMED from divide-by-near-zero LMP hours. Absolute congestion $ is
   the stable form (DC-heavy-zone mean rose ~$1.6/MWh in 2019 → ~$6–9/MWh by 2022 and 2025).
2. **Congestion z-score.** `cong_z = winsorise( (daily_cong − roll_mean) / roll_std , ±3 )` where
   `roll_mean/std` use a **trailing** 756-trading-day (~3y) window, `min_periods = 252`. Trailing-only ⇒
   point-in-time.
3. **Reserve-tightness.** Per zone, per day: `tight = daily_peak_load / trailing_756d_p99(daily_peak_load)`;
   `reserve_z` = winsorised z-score of `tight` with the same window. Mirrors the GSI's reserve-tightness
   proxy but computed per zone from `fetch_zonal_load`.
4. **Per-zone composite.** `zone_c = w_cong · cong_z + w_reserve · reserve_z`.
5. **Cross-zone composite.** `daily_composite = Σ_z  zoneweight_z · zone_c_z`, `Σ zoneweight = 1`.

`w_cong`, `w_reserve`, the zone set, and `zoneweight` are ladder parameters (§5).

### 4.2 Monthly reduction and multiplier

6. **Month-end value.** `monthly_composite[M] = mean( daily_composite over the trailing 20 trading days
   ending on the last trading day of month M )`.
7. **Regime multiplier.**
   - *Discrete (rungs 1, 2, 4, 5, 6):* `m = HI` if `monthly_composite ≥ +T`; `m = LO` if
     `≤ −T`; `m = 1.0` otherwise. Frozen: `T = 0.5`, `HI = 1.25`, `LO = 0.6`.
   - *Continuous (rung 3):* `m = clip(1.0 + k · monthly_composite, 0.5, 1.5)`, frozen `k = 0.35`.
8. **Hold / lag.** The value for month `M` is computed from rows dated ≤ last trading day of `M` and applied
   to **every trading day of month `M+1`**, via the same `_month_hold` helper `overlay.py` already uses.
9. **Warm-up.** Before the first month with a valid z-score (≥ 252 trailing obs), `m = 1.0` (neutral, fully
   invested). On the 2018–2022 window this means the signal is inactive until ~2021; disclosed in the
   results doc — the prior-window test is effectively 2021–2022.
10. **Missing data.** A month with no usable LMP rows → `m = 1.0` and a logged warning.

### 4.3 Public API

```python
def regime_composite(start: str, end: str, *, zones: list[str], w_cong: float,
                     w_reserve: float, zone_weight: str = "equal",
                     fetch_fn=None) -> pd.Series:
    """Daily cross-zone composite z (step 5), indexed by date. `fetch_fn` injectable for tests."""

def regime_multiplier(composite: pd.Series, *, mode: str = "discrete",
                      thresh: float = 0.5, hi: float = 1.25, lo: float = 0.6,
                      k: float = 0.35) -> pd.Series:
    """Daily multiplier series (steps 6-8), _month_hold-aligned, warm-up -> 1.0."""

def regime_report(start: str, end: str) -> dict          # runs the whole ladder for one window
def regime_table(report: dict) -> str                    # ladder x metrics, both windows, gate verdicts
```

`_combine_subsignals(subsignals: dict[str, tuple[pd.Series, float]]) -> pd.Series` is the reduction point
for step 4/5 and the RT/DA hook (§2).

---

## 5. The pre-registered ladder

Frozen here before any run. Run **top to bottom**; log every rung's full metrics for both windows; **stop at
the first rung that clears the gate (§6) and sits on a plateau.** Each rung changes exactly one thing from
the rung above.

| # | Variant | Params vs rung above | Order rationale |
|---|---|---|---|
| 0 | **Baselines** (not gated, always printed) | buy-and-hold; layer-1-only (`trend_gate × vol_target`); vol-target-only | reference rows |
| 1 | Discrete, PJM 4-zone, congestion-only | `zones={DOM,AEP,COMED,PPL}`, `w_cong=1.0`, `w_reserve=0.0`, equal zone weight, `mode=discrete`, `T=0.5`, `HI=1.25`, `LO=0.6` | simplest; cleanest cached data; matches repo idiom (GSI discrete stress events) |
| 2 | + reserve-tightness | `w_cong=0.7`, `w_reserve=0.3` | adds the demand-level leg at a small documented weight |
| 3 | Continuous multiplier | `mode=continuous`, `k=0.35`; keeps rung 2's `w_cong=0.7 / w_reserve=0.3` (changes only `mode`) | tests whether smoothing beats the 3-state step |
| 4 | Widen zones | `zones += {PSEG, ATSI}` (6-zone equal weight) | more of PJM's load; ATSI/PSEG carry real congestion |
| 5 | Load-weighted zones | `zone_weight="load"` (weight ∝ trailing zone peak load) | proxy for "where the data centers are" without point-in-time DC-MW reconstruction |
| 6 | + ERCOT West | add sub-signal: `LZ_WEST − LZ_NORTH` daily spread frac via `daily_spread_summary`, z-scored, equal sub-weight with the PJM composite | second ISO; spread proxy since ERCOT LMP has no congestion component cached |
| 7 | RT/DA hook | add an RT/DA sub-signal for PJM+ERCOT — **only if** the §7 probe confirms `gridstatus` returns RT LMP at usable granularity over 2018–2025 | deferred; cheapest to add given the `_combine_subsignals` seam |

**Plateau definition (per rung):** the rung's neighbours must **also** pass G1 and G2 (§6):
- discrete rungs: `T ∈ {0.25, 0.75}` (with the rung's own `HI/LO`), and zone set ±1 name
- continuous rung: `k ∈ {0.25, 0.45}`
- weight rungs: `w_cong ∈ {rung ±0.15}`

A rung that passes alone but fails on its neighbours is logged **knife-edge — not a pass**, and the ladder
continues. Same rule `overlay.overlay_report`'s 5×3 plateau grid enforces for layer 1.

---

## 6. The gate (frozen before the run)

`Calmar ≡ CAGR / |MaxDD|`. **"layer-1-only"** ≡ the series `overlay_report` already labels
`trend_gate_voltarget`, i.e. `apply_overlay(ret, prices, rf)` with the frozen `OVERLAY_*` defaults (MA 100,
vol lookback 20, target vol 0.20, max leverage 1.5) — trend gate **and** vol target both on. **"vol-target-only"**
≡ `apply_overlay` with the trend gate disabled (`ma_days` large enough to never gate), vol target on.

A ladder rung **passes** iff **all four** hold:

- **G1 — beats layer-1-only, primary window (2023-01-01 → 2025-12-31):**
  `Sharpe(rung) > Sharpe(L1)` **and** `Calmar(rung) > Calmar(L1)`.
- **G2 — beats layer-1-only, prior window (2018-01-01 → 2022-12-31):** same two inequalities on the prior panel.
- **G3 — no worse than layer 1 vs buy-and-hold:** mean over calendar years of `(BH_yr − rung_yr)` ≤ mean over
  calendar years of `(BH_yr − L1_yr)`, on **both** windows. (Layer 1's honest weakness is being pure drag in
  years without a −40% event; layer 2 must not be a *bigger* drag.)
- **G4 — plateau:** every neighbour (§5) satisfies G1 and G2.

**Marginal flag (does not stop the ladder):** a passing rung where any G1/G2 gap is `< 0.05` Sharpe or
`< 0.05` Calmar is reported **marginal**; the ladder continues past it. Only a **non-marginal plateau-pass**
stops the ladder and becomes the shipped layer-2 config.

**If no rung passes:** negative result. `docs/grid-regime-layer2-results.md` states plainly that a
grid-congestion regime signal does not beat a price trend gate at setting basket exposure on this sample;
the shipped recommendation stays **layer-1-only, or a smaller un-overlaid basket**. `config.py` gets no
`REGIME_ENABLED = True`. This is an acceptable outcome.

---

## 7. RT/DA data-availability probe (gate for rung 7 only)

Before rung 7, run a throwaway check (not committed as a test): does
`gridstatus` (0.34.0, already pinned) return **real-time** LMP for PJM and ERCOT at hourly-or-finer
granularity across a sample of months spanning 2018–2025? Record the finding in the results doc. If it does
not cleanly return RT LMP over the full window, rung 7 is **skipped** and logged as "not feasible with
current data" — not a failure.

---

## 8. Composition seam — edits to `overlay.py`

Layer 1 functions (`_month_hold`, `trend_gate`, `vol_target_scalar`, `apply_overlay`, `exposure_series`,
`overlay_report`, `overlay_table`, plateau constants) are **untouched**. Add, mirroring them exactly:

```python
def regime_exposure(regime_mult: pd.Series, basket_returns: pd.Series,
                    vol_lookback=config.OVERLAY_VOL_LOOKBACK,
                    target_vol=config.OVERLAY_TARGET_VOL,
                    max_leverage=config.OVERLAY_MAX_LEVERAGE) -> pd.Series:
    """regime_mult * vol_target_scalar, reindexed to basket_returns, warm-up -> 1.0,
    product clamped to <= config.OVERLAY_MAX_LEVERAGE (1.5)."""

def apply_overlay_l2(basket_returns, regime_mult, rf_annual=config.RISK_FREE_RATE, ...) -> pd.Series:
    """exposure * basket_ret + (1 - exposure) * rf_daily, exposure = regime_exposure(...)."""

def exposure_series_l2(basket_returns, regime_mult, ...) -> pd.Series:   # for mechanics stats
```

- `vol_target_scalar` is imported and used unchanged.
- **Frozen decision:** the combined exposure cap is `config.OVERLAY_MAX_LEVERAGE` (1.5×), i.e. the product
  `regime_mult × vol_scalar` is clamped at 1.5, not 1.25 × 1.5 = 1.875.
- A flat `regime_mult ≡ 1.0` must make `apply_overlay_l2` reproduce vol-target-only exactly (test).

---

## 9. Reporting & CLI

**`grid_regime.regime_report(start, end)` / `regime_table(report)`** — same shape as
`overlay_report` / `overlay_table`:

- rows: buy-and-hold · vol-target-only · layer-1-only · each ladder rung run so far — columns
  CAGR / Sharpe / Calmar / MaxDD, **primary and prior windows side by side**
- the four gate booleans G1–G4 per rung + verdict `PASS` / `marginal` / `knife-edge` / `FAIL`
- **mechanics** per rung: avg multiplier, % months in each regime state, multiplier flips, avg combined exposure
- **regime timeline**: monthly composite z and resulting multiplier, so the reader can eyeball *when* it
  leaned in / out relative to the 2024-11→2025-04 drawdown
- an honesty-caveat string (like `overlay._CAVEAT`)

**CLI:** `python -m grid_equipment_basket --overlay-l2` prints the layer-1 report (unchanged) then the
layer-2 ladder report for both windows, and writes `regime_metrics.csv` + `regime_timeline.csv` to the
output dir. `--refresh-regime` (default off) is the only network path.

`config.py` additions: `REGIME_ZONES_CORE`, `REGIME_ZONES_WIDE`, `REGIME_ZSCORE_WINDOW=756`,
`REGIME_ZSCORE_MINP=252`, `REGIME_MONTH_LOOKBACK=20`, `REGIME_THRESH=0.5`, `REGIME_HI=1.25`,
`REGIME_LO=0.6`, `REGIME_K=0.35`, `REGIME_W_CONG`/`REGIME_W_RESERVE` per rung, and `REGIME_LADDER` (the
frozen list of §5 as a list of dicts). All documented "frozen, not fitted" like the `OVERLAY_*` block.

---

## 10. Testing

**Unit (fast, deterministic — synthetic frames or tiny cached slices, `fetch_fn` injected):**

- `test_grid_regime.py`: absolute-`|congestion $|` daily aggregation; z-score uses a **trailing** window
  (feed a step change, assert no pre-step leakage); winsorisation at ±3; `_combine_subsignals` weight math
  (Σw=1, single vs multi); discrete mapping at/across `±T`; continuous mapping + `clip(0.5, 1.5)`;
  `_month_hold` lag — month-`M` rows only affect month-`M+1` exposure; warm-up (<252 obs) → 1.0;
  empty / all-NaN month → 1.0 + warning; load-weighted vs equal zone weight.
- `test_overlay_l2.py`: `regime_exposure == regime_mult × vol_scalar` up to reindex; product clamps at 1.5;
  `apply_overlay_l2` return identity `exposure·ret + (1−exposure)·rf`; flat `regime_mult=1.0` ⇒
  vol-target-only exactly; NaN regime days → treated as 1.0.
- gate logic: G1–G4 booleans over fixture metric dicts; plateau rule (neighbour fails ⇒ knife-edge);
  marginal-flag boundary at the 0.05 gap.

**Not a test — a live script:** the ladder run itself, invoked via `--overlay-l2`, with output committed to
`docs/grid-regime-layer2-results.md`.

**Look-ahead audit test:** assert the month-`M` multiplier is a pure function of rows with `date ≤` last
trading day of month `M`, then `_month_hold`-shifted to `M+1` — the property layer 1's tests check for
`trend_gate`.

---

## 11. Honesty caveats (carried verbatim into the results doc)

- One macro regime; a monthly decision ⇒ ~36 independent obs on the primary window, ~24 *active* obs on the
  prior window (post warm-up). Low breadth; the two-window + plateau rules are the only defense against
  fitting.
- The 756-day z-score warm-up makes the prior-window test effectively 2021–2022, not 2018–2022.
- Proposal D already showed monthly ΔGSI barely co-moves with equity aggregates. If the ladder fails, that is
  the expected reason, not a surprise.
- Rung 6 mixes a true congestion component (PJM) with a spread proxy (ERCOT) — a known apples/oranges seam.
- A pass on 2018–2022 is **not** an AI-buildout-era test: congestion then was weather / wind-transmission
  driven, not data-center driven. A signal that clears both windows is either robust to that difference or is
  picking up a generic "congestion ⇒ risk for grid gear" relationship — still a finding, just a different
  one; the results doc must say which.
- Layer 2 is still one exposure decision per month — low breadth by construction. The defence (from the
  handoff) is "the client owns the theme regardless; layer 2 only sets the weight, and a physical-data
  reason to set it is worth paying for." If the gate fails, that defence does not save it.

---

## 12. Deliverables

1. `grid_equipment_basket/grid_regime.py` + `overlay.py` seam + `config.py` constants + `--overlay-l2`.
2. `tests/grid_equipment_basket/test_grid_regime.py`, `test_overlay_l2.py` (+ gate-logic tests).
3. `docs/grid-regime-layer2-results.md` — full ladder table, gate verdicts, regime timeline, the §11
   caveats, and the shipped recommendation (a passing config, or "layer-1-only / smaller basket").
4. `README.md` updated: signal description, data provenance + staleness, the ladder, the gate outcome.
5. If a rung passes: `config.REGIME_* ` frozen to that rung + a one-line `REGIME_ENABLED` note. If none:
   explicit negative-result note, no enable flag.
