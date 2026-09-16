# Technical Reference — Screened Electrification Pipeline

One row per module: what it does, its key inputs/outputs, and its role in the pipeline.

## electrification_strategy/

| Module | Role | Key inputs | Key outputs |
|---|---|---|---|
| `config.py` | Frozen parameters (spec 6) | — | constants consumed by every other module |
| `universe.py` | Universe constructions (marquee/frozen/thematic/screen/screen_reit) | `data/universe_seed.csv`, `data/etf_membership_2026.csv`, prices | `target_fn(construction, prices)` |
| `fred.py` | DFII10 (10yr real yield) fetch + cache | FRED API | `fetch_series` |
| `_util.py` | Small shared helpers | — | `month_hold` (hold a monthly value into daily observations, lagged one month) |
| `short_interest.py` | Short-interest crowding/cost diagnostic (Task 1) | yfinance `Ticker.info` | `fetch_short_interest`, `crowding_diagnostic` |
| `risk_exposures.py` | Client risk-factor adapter (Task 2) | `data/risk_exposures.csv` (placeholder schema — see below) | `load_risk_exposures`, `align_to_universe`, `portfolio_exposure_report` |
| `valuation_overlay.py` | Graduated de-risk multiplier | basket level, SPY returns | `extension_multiplier` |
| `hedge_overlay.py` | 15% GLD/IEF sleeve + conditional DLR/EQIX short | prices, DFII10 | `sleeve_returns`, `short_returns`, `apply_hedge` |
| `backtest.py` | Signal -> weights -> P&L -> metrics; also writes `risk_exposure_report.csv` / `short_interest_diagnostic.csv` when that data is present | prices, DFII10 | `run_comparison`, `write_outputs` |
| `__main__.py` | CLI entry point; also calls `fetch_short_interest` and `risk_exposures.load_risk_exposures` so those diagnostics land in every run's output bundle | argv | `output_electrification*/` file bundle |

Note: `stress_beta_signal.py` does not exist — Task 6a was skipped because the stress-beta gate
(below) FAILED, so there was no passing signal to wire into the live pipeline.

## electrification_stress_beta/ (Task 4-5)

| Module | Role | Key inputs | Key outputs |
|---|---|---|---|
| `national_gsi.py` | Blends grid_resilience's per-ISO GSI into one national series | `grid_resilience.data.grid_data` | `build_national_gsi`, `fetch_and_build_national_gsi`, `uniform_ticker_iso_map` |
| `gate_test.py` | Pre-registered stress-beta gate (run once) | prices, national GSI | `gate_result.json` |
| `decile_gate_test.py` | Decile (top-3/bottom-3) follow-up to `gate_test.py`'s tercile cut, checking for a tail/threshold effect the tercile test could have averaged away | prices, national GSI | `decile_gate_result.json` |

`gate_test.py`'s pre-registered result was **FAIL** (t=0.144 vs. the |t|>=2.0 bar) — see
`docs/electrification-stress-beta-gate-results.md`. The `decile_gate_test.py` follow-up also
**FAILED** (t=0.318 primary window, same order of magnitude short of the gate) — see
`docs/electrification-stress-beta-decile-gate-results.md`. See
`docs/electrification-stress-sensitivity-response.md` for the actual delivered Phase 1 item 4
output: three built-in protections formalized in place of a stress-beta signal, since neither
cut of this gate found a live relationship to build on.

## Reused, unmodified from grid_resilience/

| Module | Role |
|---|---|
| `signals/grid_stress_index.py` | GSI construction (LMP z-score / congestion / reserve tightness / event flag) |
| `signals/conditional_beta.py` | Rolling OLS stress-beta estimation |
| `signals/stress_events.py` | Named + algo-detected stress event calendar |
| `data/grid_data.py` | LMP/load fetch + daily summaries, per ISO |

## Schema contract: risk_exposures.csv (Client-provided, Task 2)

Long format, one row per date x ticker x factor: `date, ticker, factor, exposure`. Place at
`electrification_strategy/data/risk_exposures.csv` (not committed — Client-provided). See
`electrification_strategy/data/risk_exposures_sample.csv` for a synthetic example of the exact
shape expected.
