# Operational Runbook — Screened Electrification Pipeline

## Setup

```bash
poetry install
```

Environment variables needed (see `grid_resilience/config.py` for the full list; only
required if re-running the stress-beta gate / GSI, not for the day-to-day strategy):
`PJM_API_KEY` (free, pjm.com/api). No key is required for MISO, CAISO, SPP, NYISO, ISO-NE, or
for the core `electrification_strategy` pipeline (equity prices via yfinance, DFII10 via FRED,
short interest via yfinance).

## Refreshing data

Fetches are cached to parquet across three directories, by data type:

- **Equity prices** (used by `fetch_prices()`) cache under `grid_equipment_basket/data/cache/`
  — one **wide** `prices_YYYY-MM.parquet` chunk per calendar month, with every ticker as a
  column in that single file (`save_monthly_chunks_wide` in `grid_resilience/data/cache_utils.py`;
  e.g. `prices_2020-01.parquet` is one file shaped 23 rows x 249 ticker columns, not a file per
  ticker). Fundamentals cache separately as one file per ticker (`fundamentals_{TICKER}.parquet`,
  no month split) in the same directory.
- **FRED data** (DFII10) caches under `electrification_strategy/data/cache/` — this directory
  holds only `fred_DFII10.parquet`, nothing else.
- **Grid LMP/load data** caches under `grid_resilience/data/cache/` — only touched when
  re-running the stress-beta gate (`electrification_stress_beta`), not the core strategy.

To force a full refresh, delete the relevant cache directory before running — see Task 9 of the
implementation plan for the exact clear/rebuild/restore sequence used to verify reproducibility.

## Running the pipeline

```bash
poetry run python -m electrification_strategy --start 2017-06-01 --output output_electrification
```

Flags (verified against `electrification_strategy/__main__.py`'s `argparse` definition):
- `--universe {marquee,frozen,thematic,screen,all}` — restrict to one construction (default `all`)
- `--no-plot` — skip the performance.png chart (faster, useful for CI/smoke runs)
- `--output PATH` — output directory (default `./output_electrification`)
- `--start` / `--end` — date range (defaults: `2017-06-01` / last full calendar month)

Note: `universe.CONSTRUCTIONS` also includes `screen_reit` (added in Task 7), but it is not
currently one of the CLI's `--universe` choices — it is only reachable today via
`--universe all`, which iterates `universe.CONSTRUCTIONS` dynamically. This is a known, minor,
pre-existing gap in the CLI, not a bug to fix as part of routine operation.

To re-run the stress-beta gate (only needed if re-testing the stress-beta/GSI signal, not part
of routine operation — requires `PJM_API_KEY`):

```bash
poetry run python -m electrification_stress_beta.gate_test
```

## Rebalance cadence

Monthly, with a 42-day reporting lag (`config.REBALANCE_LAG_DAYS`) — a holding entering the
universe today is reflected in the portfolio 42 calendar days later, matching realistic
data-availability lag for the fundamental screen inputs. Vol-target and valuation-overlay
exposure are recalculated daily; the underlying holdings list changes roughly monthly.

## Interpreting output

| File | Contents |
|---|---|
| `metrics.csv` | CAGR / vol / Sharpe / MaxDD / beta / drag-vs-plain / feared-scenario P&L / corr-to-VOLT, one row per construction x overlay-stack cell, plus benchmark rows |
| `episode_drawdowns.csv` | Peak-to-trough drawdown per named market-shock episode, per cell |
| `plateau.json` | The pre-registered winner (per `config.WINNER_*` criteria) and the valuation-overlay sensitivity plateau |
| `returns.csv` | Daily return series per cell |
| `performance.png` | Growth-of-$1 + drawdown chart for the winning cell vs. `marquee plain` |
| `risk_exposure_report.csv` | Portfolio-level factor exposure (only written once a real `risk_exposures.csv` is in place — empty/absent otherwise) |
| `short_interest_diagnostic.csv` | Latest short-interest/crowding snapshot for the hedge short leg (DLR, EQIX) |

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `yfinance` returns all-NaN columns for some tickers | Rate limiting / partial batch download | Re-run; `fetch_prices` retries with backoff, see `grid_equipment_basket/data/prices.py` |
| `THR` returns 100% NaN from yfinance specifically | Permanent, known gap — Thermon Group Holdings was acquired by CECO Environmental and delisted 2026-06-01; Yahoo Finance no longer serves any historical quote for it (confirmed during Tasks 5/9) | Not a bug to chase — expected for this ticker going forward |
| `RuntimeWarning: FRED fetch ... failed` | FRED API unreachable | Falls back automatically to the committed `electrification_strategy/data/fred_DFII10.csv` — update that file periodically if running live for extended periods |
| `PJM_API_KEY` errors when re-running the stress-beta gate | Missing/expired key | Get a free key at pjm.com/api; only needed for `electrification_stress_beta` re-runs, not the core strategy |
| Stale cache producing outdated results | Cache never expires automatically | For **prices**, there is no per-ticker file to delete — each `prices_YYYY-MM.parquet` is wide (all tickers as columns), so a targeted refresh means deleting/re-fetching that month's file for every ticker in it, or the whole cache directory for a full refresh (see Task 9). For **fundamentals** (`fundamentals_{TICKER}.parquet`), a single ticker's file can be deleted independently. |
| `universe_seed.csv` raises `ValueError: unknown sub_industry values` | A typo or a new sub-industry added without updating `config.SUB_INDUSTRIES` | Fix the CSV or extend `config.SUB_INDUSTRIES` |
