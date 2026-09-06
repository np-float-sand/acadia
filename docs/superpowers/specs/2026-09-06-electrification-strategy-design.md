# Electrification Strategy — v1 design spec

**Date:** 2026-09-06
**Status:** BUILT 2026-09-06 — results: `docs/electrification-strategy-v1-results.md`
(winner: `thematic +val+sleeve`, Sharpe 0.85 / MaxDD −24%; 8/12 cells pass the pre-registered
rule). Two deviations logged in the results doc: daily vol-target (not the month-held
`overlay.vol_target_scalar`) and valuation extension measured on the plain basket index.
**Package:** `electrification_strategy/` (new top-level, parallel to `grid_equipment_basket/`)
**Research provenance (read for context):**
`docs/electrification-ls-strategy-note.md` (long-only core, short-leg search),
`docs/electrification-short-leg-insurance-probe-results.md` (hedge + crowding probes,
2026-09-06 — the source of every frozen parameter here),
`docs/handoff_2026-09-05-electrification-strategy.md`,
memory `electrification-short-leg-insurance-probe`, `capex-cycle-pair-classifier`.

---

## 1. What this is

A rules-based, fully-mechanical **long-only electrification / grid-equipment basket** with a
**valuation-extension de-risk overlay** and an optional **hedge overlay** (GLD/short-duration
sleeve + a conditional DLR+EQIX short). It is honest **enhanced thematic beta**, not alpha —
close to the VOLT ETF on return, with disciplined risk control the ETF lacks.

The module is built as a **comparison harness**: two universe constructions and the hedge
legs on/off are all selectable, the backtest prints the full variant grid, and the pitch
shows the best variant. Serves three purposes at once (per brainstorming): a paper-trading
track (with a documented pre-live gate), an IC reference implementation, and a research base.

### Non-goals for v1 (§7)

- True point-in-time ETF-holdings / GICS-vintage universe. **v1 uses 2026 holdings snapshots
  back-cast**, with first-listing-date gating. Real point-in-time membership is a
  **pre-live requirement**, documented as a gate, not built now.
- Transaction-cost / market-impact modelling (repo norm: negligible at quarterly cadence).
- Options-structured hedges (blocked on options data — see prior handoffs).
- Forward-looking fundamental valuation z-score (needs an estimates feed; v1 uses the
  price-extension proxy).
- Updating `docs/electrification-strategy-proposal.html` (separate task after a variant is chosen).
- Live/paper-trading execution wiring — v1 is the backtest + the frozen live rule.

---

## 2. Architecture

### 2.1 Package layout

```
electrification_strategy/
  __init__.py
  config.py            # all params, frozen + documented + plateau notes
  universe.py          # U1 frozen-screen + U2 thematic-proxy + marquee reference
  valuation_overlay.py # graduated exposure multiplier (1.0 / 0.8 / 0.6)
  hedge_overlay.py     # GLD/IEF sleeve + conditional DLR+EQIX short
  fred.py              # FRED series fetch + parquet cache + committed CSV fallback
  backtest.py          # run_comparison(): full variant grid + plateau + reporting
  __main__.py          # CLI
  README.md
  data/
    universe_seed.csv          # committed: ticker, sub_industry, notes
    etf_holdings/
      VOLT_2026.csv  ELFY_2026.csv  ZAP_2026.csv  GRID_2026.csv  PAVE_2026.csv   # committed snapshots
    fred_DFII10.csv            # committed offline fallback
    cache/                     # runtime parquet (gitignored, like the other packages)
tests/electrification_strategy/
  test_universe.py  test_valuation_overlay.py  test_hedge_overlay.py
  test_fred.py  test_backtest.py  test_cli.py
docs/electrification-strategy-v1-results.md   # results doc (written after the first run)
```

### 2.2 Reuse (import, do not fork)

From `grid_equipment_basket`: `basket.simulate_basket`, `basket.apply_cap`,
`backtest.compute_metrics`, `backtest.calendar_year_returns`, `data.prices.fetch_prices`,
`overlay.vol_target_scalar`, `hedges.find_drawdown_episode`, `hedges.episode_drawdown`.
Reimplement the ~4-line month-hold helper locally (`overlay._month_hold` is private).
Precedent for cross-package helper reuse: `grid_equipment_basket/data/prices.py` imports
`grid_resilience.data.cache_utils`.

### 2.3 Layer composition (daily simple returns)

```
base_r      = simulate_basket(universe_prices, EW, 25% cap, quarterly, 42d lag).returns
exposure_t  = valuation_mult_t * vol_target_scalar_t          # both month-held; product clipped at 1.5
core_r      = exposure_t * base_r + (1 - exposure_t) * rf_daily
# hedge overlay, each leg independently toggleable:
hedged_r    = (1 - w_sleeve) * core_r
            + w_sleeve * sleeve_r                              # GLD/IEF EW daily
            - w_short  * (dcr_r * realyield_rising_t)          # DLR/EQIX EW daily, masked
```
Mirrors `overlay.py`'s `exposure = gate * vol_scalar` idiom; the hedge layer is additive.

---

## 3. Universe (`universe.py`)

Three constructions, all reconstituted quarterly on the same calendar as the basket
(42-day reporting lag), all with **first-listing-date gating** (a name enters only once it
has ≥ 252 trading days of history as of the reconstitution date), all equal-weight with a
25% single-name cap.

| construction | inclusion defined by | screen |
|---|---|---|
| `marquee` (reference) | `grid_equipment_basket.config.UNIVERSE` verbatim | none — the existing 9 |
| `frozen` (U1) | our quality judgment | ticker in `universe_seed.csv` pool **and** sub-industry ∈ map **and** in ≥ 1 of {VOLT,ELFY,ZAP,GRID,PAVE} 2026 snapshot **and** positive trailing-4Q net income **and** positive trailing-4Q free cash flow (as of asof − 42d) |
| `thematic` (U2) | thematic-ETF consensus | ticker in ≥ 2 of {VOLT,ELFY,ZAP,GRID} 2026 snapshots **and** US-listed **and** sub-industry ∈ map. **No profitability screen** — the ETFs decide. |

### 3.1 Seed candidate pool (`universe_seed.csv`, illustrative — finalised in the plan)

~30–35 tickers with a `sub_industry` tag from a documented manual map (same discipline as
`grid_equipment_basket/candidate_research.md`). Map values:
`electrical_equipment | heavy_electrical | electrical_e&c | dc_thermal | grid_scale_storage`;
anything else → excluded.

Starting pool: ETN, HUBB, NVT, EMR, AME, RRX, POWL, ATKR, AEIS, GNRC, AYI, ENS, THR
(electrical_equipment); GEV, VRT (heavy_electrical / dc_thermal); PWR, MYRG, PRIM, EME,
FIX, MTZ, IESC, APG (electrical_e&c); TT, CARR, JCI, SPXC, MOD (dc_thermal); FLNC, STEM
(grid_scale_storage). Profitability screen is expected to drop FLNC, STEM, and any
chronic lossmaker; `thematic` may pull in marquee-heavy names the pool under-weights —
the contrast is the point.

### 3.2 Data for the screen

- Fundamentals (net income, FCF): a lightweight per-ticker pull following
  `grid_equipment_basket/margin_data.py` conventions (yfinance quarterly financials /
  cash-flow), cached. Trailing-4Q sums, evaluated as of asof − 42d.
  **FCF := trailing-4Q operating cash flow − trailing-4Q capex.**
  yfinance quarterly financials are known-flaky; if the automated pull is unreliable for a
  name, the plan falls back to a hand-curated `profitable_2026` boolean column in
  `universe_seed.csv` (documented, not fitted — a name is `False` only if it has GAAP
  losses or negative FCF in most of the last 8 quarters). The automated pull is preferred
  where it works; the flag is the backstop so the screen is never silently wrong.
- ETF snapshots: committed CSVs sourced from the issuers' published holdings files
  (First Trust for GRID; PAVE = Global X; VOLT/ELFY/ZAP from their issuers). One snapshot
  date (2026, recorded in each file header). If a snapshot for a newer ETF (VOLT/ELFY/ZAP)
  cannot be sourced cleanly, document which and fall back to the `≥ 1 of the available`
  rule for U1 / `≥ 2 of the available, min 3 available` for U2, noted in the results doc.

### 3.3 Honesty caveat (carried into code + README)

> Universe membership is point-in-time only on the listing-date dimension. Candidate
> membership and the ETF snapshots are 2026-vintage, back-cast — pre-2025 results are
> survivorship-caveated. True point-in-time holdings / GICS vintages are a prerequisite
> before any live capital.

---

## 4. Overlays

### 4.1 Valuation / extension de-risk (`valuation_overlay.py`)

Monthly, decided at each month-end, applied to every day of the following month
(no lookahead). Inputs: the **plain basket** price index `P` (cumprod of `base_r`, before
vol-target and before this overlay) and SPY. (The probe measured these on the vol-targeted
series; the difference in `ext` / `rs12` is second-order — the plain index is cleaner and
carries no circularity with `exposure_t`.)

```
ext  = P / P.rolling(200).mean() - 1
rs12 = P.pct_change(252) - SPY.pct_change(252)
m = 1.0   if ext <= VAL_EXT_LO  and rs12 <= VAL_RS_LO
m = 0.6   if ext >  VAL_EXT_HI  or  rs12 >  VAL_RS_HI
m = 0.8   otherwise
```
Warm-up (MA200 / 252-day return not yet available) → `m = 1.0`. Freed capital at rf.

Frozen config: `VAL_EXT_LO=0.15, VAL_EXT_HI=0.35, VAL_RS_LO=0.25, VAL_RS_HI=0.50,
VAL_MULT_MID=0.8, VAL_MULT_LOW=0.6`. Plateau grid: all four thresholds × {0.8, 1.0, 1.2}
(shown stable in the probe — MaxDD improvement ~constant, drag 1.3–3.1%/yr).

Inline caveat: *a graduated sell-when-euphoric trim, not a crash shield — ~2.7pp MaxDD
improvement on the concentrated book, ~0 on the broadened book but +Sharpe. Better-behaved
than the binary layer-1 trend gate in `grid_equipment_basket/overlay.py`.*

### 4.2 Hedge overlay (`hedge_overlay.py`) — two independent toggles

**(a) Diversifier sleeve.** `w_sleeve = 0.15` of NAV in an equal-weight GLD + IEF basket
(daily returns via `fetch_prices`), funded proportionally from the core:
`(1 - w_sleeve)*core_r + w_sleeve*sleeve_r`. This is the clean winner in the probe
(raises Sharpe 0.84→0.87, +13% in the feared "solar squeeze" scenario, 1.9%/yr drag).

**(b) Conditional short.** `- w_short * (dcr_r * mask_t)`, where `w_short = 0.25`,
`dcr_r` = equal-weight DLR + EQIX daily return, and `mask_t` = "6-month change in DFII10
> 0" evaluated monthly, month-held, with a 5-business-day lag on the FRED series.

Frozen config: `HEDGE_SLEEVE_TICKERS=("GLD","IEF"), HEDGE_SLEEVE_WEIGHT=0.15,
HEDGE_SHORT_TICKERS=("DLR","EQIX"), HEDGE_SHORT_WEIGHT=0.25, HEDGE_SHORT_SIGNAL="DFII10",
HEDGE_SHORT_LOOKBACK_DAYS=126, HEDGE_SHORT_LAG_DAYS=5`.

Inline caveat: *the sleeve is the clean winner; the DLR+EQIX short is a levered
rising-real-yield bet — 2-name concentration, ~2.5%/yr dividend paid while short, all its
value in two rate episodes (2022 rate shock, DeepSeek), ~15%/yr drag inside a rate-rising
equity bull like 2023–24. Full stack: MaxDD −31→−22, SPY β 0.76→0.57, Sharpe held, ~4%/yr
CAGR cost.*

---

## 5. FRED data (`fred.py`)

`fetch_series(series_id: str, start: str, end: str) -> pd.Series`

1. Read `data/cache/fred_{series_id}.parquet` if present; find missing months.
2. On miss: GET `https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}`,
   parse (first col = date, second = value; `"."` → NaN → dropna), append to cache.
3. On network failure: fall back to committed `data/fred_{series_id}.csv` and emit a
   `RuntimeWarning`.

Only `DFII10` (10-year TIPS yield) is used in v1. DFII10 publishes next business day with
negligible revisions; the 5-day signal lag covers it. README documents the manual monthly
refresh of the committed CSV as part of the update path.

---

## 6. Backtest, CLI, outputs

### 6.1 `backtest.run_comparison(start, end, ...) -> dict`

Grid: **{marquee, frozen, thematic} × {plain, +val, +val+sleeve, +val+sleeve+short}** (12 cells).
Per cell: daily return series, `compute_metrics` (CAGR/vol/Sharpe/Sortino/MaxDD), SPY beta,
corr vs VOLT (common window, post 2024-12), effective #names (last weight row), the five
frozen episode drawdowns, feared-scenario P&L, calm-period CAGR drag (2017-06..2019-12 and
2023-01..2024-10).

**Feared-scenario P&L** for a cell = cumulative (cell monthly return − that universe's
plain-book monthly return) over the months where *that universe's plain book*
(vol-targeted, no overlays) is < 0 **and** TAN is > 0 — i.e. the "solar squeezes while
AI-power de-rates" divergence. The reference month set is fixed per universe, not per
overlay stack.

Frozen episodes (config `EPISODES`): COVID (2020-02→04), Rate shock 22 (2021-11→2022-12),
DeepSeek (2024-11→2025-05), Tariff Apr-25, Selloff 26 — same anchors as the probe, located
per-universe via `hedges.find_drawdown_episode`.

Plateau grids: valuation thresholds × {0.8,1.0,1.2}; vol-target × MA (reuse `overlay.py` shape).
Benchmarks: VOLT, PAVE, GRID, SPY, XLI.

### 6.2 CLI (`python -m electrification_strategy`)

`--start` (default 2017-06-01) · `--end` (default last month-end) ·
`--universe {marquee,frozen,thematic,all}` (default all) ·
`--layers {core,val,sleeve,short,all}` (default all) ·
`--output ./output_electrification` · `--no-plot`

Writes: `metrics.csv` (the grid), `episode_drawdowns.csv`, `plateau.json`,
`returns_<variant>.csv`, `performance.png` (best variant vs marquee-plain vs VOLT,
cumulative + drawdown). Prints a formatted comparison table.

### 6.3 Decision rule (pre-registered — what "the winner" means for the pitch)

Among the 12 cells, the pitched variant is the one with the **highest Sharpe on the full
window subject to**: (i) MaxDD ≤ −27%, (ii) no sub-window (2019-22, 2023-26) Sharpe below
0.4, (iii) feared-scenario P&L ≥ −2%, (iv) full-period CAGR drag vs the same universe's
plain book ≤ 6%/yr. Report all 12 regardless; state the winner and the runner-up.
The **pre-live gate** (point-in-time universe) is stated as blocking live capital, not the pitch.

---

## 7. Tests (`tests/electrification_strategy/`, synthetic fixtures, no network)

- `test_universe.py` — listing-date gate; profitability gate (positive trailing-4Q NI & FCF);
  ETF-membership gate (≥1 for frozen, ≥2 for thematic); sub-industry filter; U1 vs U2
  produce the expected member sets on a small fixture.
- `test_valuation_overlay.py` — the three multiplier buckets; month-hold + shift (no
  lookahead); warm-up → 1.0.
- `test_hedge_overlay.py` — sleeve funding math; short mask from a synthetic rising /
  falling DFII10 series; 5-day lag applied; subtraction sign.
- `test_fred.py` — CSV parse (`"."` → NaN); cache write/read; offline fallback to the
  committed CSV when HTTP is mocked to fail.
- `test_backtest.py` — `run_comparison` grid shape (3 × 4), metric keys present,
  episode-drawdown wiring on a fixture.
- `test_cli.py` — smoke: `main()` on a short window, tmp output dir, monkeypatched
  `fetch_prices` / `fetch_series`; asserts the expected files are written.

---

## 8. Deliverables

1. `electrification_strategy/` package (§2.1).
2. `tests/electrification_strategy/` (§7), green.
3. `electrification_strategy/README.md` — thesis, the three universes, the overlays, the
   frozen live rule, the honesty caveats, the pre-live gate, the update path.
4. `docs/electrification-strategy-v1-results.md` — the 12-cell grid, the winner + runner-up
   per the §6.3 rule, the plateau evidence, all caveats.
5. Spec status updated to BUILT with the results-doc link.
