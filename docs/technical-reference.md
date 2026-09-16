# Technical Reference — Screened Electrification Pipeline

One row per module: what it does, its key inputs/outputs, and its role in the pipeline.

## electrification_strategy/

| Module | Role | Key inputs | Key outputs |
|---|---|---|---|
| `config.py` | Frozen parameters (spec 6) | — | constants consumed by every other module |
| `universe.py` | Universe constructions (marquee/frozen/thematic/screen/screen_reit) | `data/universe_seed.csv`, `data/etf_membership_2026.csv`, prices | `target_fn(construction, prices)` |
| `fred.py` | DFII10 (10yr real yield) fetch + cache | FRED API | `fetch_series` |
| `_util.py` | Small shared helpers | — | `month_hold` (hold a monthly value into daily observations, lagged one month) |
| `short_interest.py` | Short-interest crowding/cost diagnostic (Task 1); flags implausible provider rows (0%/missing short-%-of-float with a near-zero share count) via `data_quality_flag` rather than reporting them as a clean zero | yfinance `Ticker.info` | `fetch_short_interest`, `crowding_diagnostic` |
| `risk_exposures.py` | Client risk-factor adapter (Task 2) | `data/risk_exposures.csv` (placeholder schema — see below) | `load_risk_exposures`, `align_to_universe`, `portfolio_exposure_report` |
| `valuation_overlay.py` | Graduated de-risk multiplier | basket level, SPY returns | `extension_multiplier` |
| `hedge_overlay.py` | 15% GLD/IEF sleeve + conditional DLR/EQIX short | prices, DFII10 | `sleeve_returns`, `short_returns`, `apply_hedge` |
| `backtest.py` | Signal -> weights -> P&L -> metrics; also writes `risk_exposure_report.csv` / `short_interest_diagnostic.csv` when that data is present | prices, DFII10 | `run_comparison`, `write_outputs` |
| `__main__.py` | CLI entry point; also calls `fetch_short_interest` and `risk_exposures.load_risk_exposures` so those diagnostics land in every run's output bundle. The short-interest call is wrapped in try/except — a provider failure warns and the run completes, since it must never discard an already-computed backtest | argv | `output_electrification*/` file bundle |

Note: `stress_beta_signal.py` does not exist — Task 6a was skipped because the stress-beta gate
(below) FAILED, so there was no passing signal to wire into the live pipeline.

## electrification_stress_beta/ (Task 4-5)

| Module | Role | Key inputs | Key outputs |
|---|---|---|---|
| `national_gsi.py` | Blends grid_resilience's per-ISO GSI into one national series | `grid_resilience.data.grid_data` | `build_national_gsi`, `fetch_and_build_national_gsi`, `uniform_ticker_iso_map` |
| `gate_test.py` | Pre-registered stress-beta gate (run once); stamps the exact `universe_tickers`/`n_tickers` it tested into its result, since the seed pool grows over time | prices, national GSI | `gate_result.json` |

`gate_test.py`'s pre-registered result was **FAIL** (t=0.144 vs. the |t|>=2.0 bar) — see
`docs/electrification-stress-beta-gate-results.md`. See
`docs/electrification-stress-sensitivity-response.md` for the actual delivered Phase 1 item 4
output: three built-in protections formalized in place of a stress-beta signal, since the gate
found no live relationship to build on.

## Reused, unmodified from grid_resilience/

The full transitive-import closure reused by `electrification_stress_beta/` (and, via
`grid_equipment_basket/data/prices.py`, by `electrification_strategy/` too), traced to a fixed
point past the module-level imports — nine modules, matching `docs/deliverable-manifest.md`.
An earlier draft of both documents listed only the first four; `ercot_auth.py`/`ercot_bulk.py`
are missed by an import-time trace because they are deferred imports inside
`grid_data.fetch_lmp()`, needed at call time.

| Module | Role |
|---|---|
| `signals/grid_stress_index.py` | GSI construction (LMP z-score / congestion / reserve tightness / event flag) |
| `signals/conditional_beta.py` | Rolling OLS stress-beta estimation |
| `signals/stress_events.py` | Named + algo-detected stress event calendar |
| `data/grid_data.py` | LMP/load fetch + daily summaries, per ISO |
| `config.py` | Shared config constants backing all of the above (supported ISOs, cache paths, thresholds) |
| `data/cache_utils.py` | Monthly-chunk parquet cache helpers (also used by `grid_equipment_basket/data/prices.py`) |
| `data/utility_node_map.py` | Ticker / ISO / pricing-node reference data (pulled in at import time via `stress_events.py`'s `STRESS_EVENTS` table) |
| `data/ercot_auth.py` | ERCOT API token handling (deferred import inside `grid_data.fetch_lmp()`) |
| `data/ercot_bulk.py` | ERCOT historical bulk LMP backfill (deferred import inside `grid_data.fetch_lmp()`) |

## Reused, unmodified from grid_equipment_basket/

The traced transitive-import closure reused by `electrification_strategy/` (`backtest.py`,
`universe.py`, `__main__.py`) and `electrification_stress_beta/gate_test.py` — eight modules,
matching `docs/deliverable-manifest.md`. This is *not* the whole package; the rest is that
package's own thematic-basket research code and stays internal.

| Module | Role |
|---|---|
| `data/prices.py` | Adjusted-close price fetcher (yfinance + the `grid_resilience` wide monthly parquet cache) — the pipeline's `fetch_prices` |
| `basket.py` | Long-only basket construction and hold-shares/periodic-rebalance simulation; supplies `apply_cap` (25% single-name cap) and the rebalance engine |
| `backtest.py` | Metrics, benchmark-relative stats and run orchestration (simple-return convention) reused by `electrification_strategy/backtest.py` |
| `hedges.py` | Hedge-overlay primitives and drawdown-episode scoring, reused by the GLD/IEF sleeve + conditional short overlay |
| `config.py` | That package's config constants, including `UNIVERSE` (the `marquee` construction's verbatim source) |
| `value_chain.py` | Maker/contractor buckets + gross-margin/backlog signal; enters the closure at import time via `hedges.py` |
| `margin_data.py` | Point-in-time quarterly revenue/gross-profit facts from SEC XBRL; imported by `value_chain.py` |
| `backlog_data.py` | Backlog / book-to-bill collection (SEC XBRL + hand-collected CSV); imported by `margin_data.py` for `CIK_BY_TICKER`, plus deferred imports in `basket.py`/`backtest.py` |

## Schema contract: risk_exposures.csv (Client-provided, Task 2)

Long format, one row per date x ticker x factor: `date, ticker, factor, exposure`. Place at
`electrification_strategy/data/risk_exposures.csv` (not committed — Client-provided). See
`electrification_strategy/data/risk_exposures_sample.csv` for a synthetic example of the exact
shape expected.
