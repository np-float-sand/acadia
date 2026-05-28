# Smart Data Cache Design

**Date:** 2026-05-28
**Scope:** `grid_resilience/config.py`, `grid_resilience/data/cache_utils.py`, `grid_resilience/data/grid_data.py`, `grid_resilience/data/equity_prices.py`

## Problem

The pipeline downloads data from external APIs (gridstatus ISOs, Yahoo Finance). Two caching problems exist:

1. **Grid data** caches by hashing the exact `(iso, dataset, start, end)` string. Any change to start or end — even one day — is a 100% cache miss, causing a full re-download.
2. **Equity prices** caches into a single flat parquet file but re-downloads all tickers and all dates if any ticker or date is missing.

Neither system fetches only the missing (uncached) date ranges.

## Goals

- Never re-download data that is already cached.
- When the requested date range extends beyond what is cached, fetch only the gap.
- Fix the backtest end date to `2026-05-27` (was `date.today()`, which caused a cache miss every day).
- Add `force_refresh=False` parameter to all fetch functions so users can opt into a full re-download when needed.

## Design

### 1. Config change

`grid_resilience/config.py` line 16:

```python
# Before
BACKTEST_END = date.today().strftime("%Y-%m-%d")

# After
BACKTEST_END = "2026-05-27"
```

### 2. New shared utility: `grid_resilience/data/cache_utils.py`

Three pure functions, no API calls:

**`read_cached(path: Path) -> pd.DataFrame`**
Reads a parquet file. Returns an empty DataFrame if the file does not exist.

**`compute_date_gaps(cached_df, time_col, requested_start, requested_end) -> list[tuple[str, str]]`**
Inspects the min/max date in `cached_df[time_col]` and returns a list of `(start, end)` string tuples representing date ranges not covered by the cache. Returns up to two gaps:
- Head gap: if `requested_start` < cached min date
- Tail gap: if `requested_end` > cached max date

If `cached_df` is empty, returns the full `[(requested_start, requested_end)]`.

**`merge_and_save(cached_df, new_df, time_col, path: Path) -> pd.DataFrame`**
Concatenates `cached_df` and `new_df`, drops duplicate rows by `time_col` (keeping the newer fetch), sorts by `time_col`, and writes to `path` as parquet. Returns the merged DataFrame.

### 3. Grid data — `grid_resilience/data/grid_data.py`

**Cache key change:**
`_cache_path(iso, dataset, start, end)` becomes `_cache_path(iso, dataset)` — removes start/end from the hash. One file per `(iso, dataset)` pair, e.g. `ercot_lmp_hub.parquet`.

**Fetch logic change** (applies to `fetch_lmp`, `fetch_load`, `fetch_fuel_mix`):

Each function gains `force_refresh: bool = False`.

With `force_refresh=False`:
1. Call `read_cached(cache_file)` to get what is already on disk.
2. Call `compute_date_gaps(cached, time_col, start, end)` to find missing ranges.
3. If no gaps: return cached data filtered to `[start:end]` and `locations` if specified.
4. For each gap `(gap_start, gap_end)`: call the gridstatus API for that range only.
5. Call `merge_and_save(cached, new_data, time_col, cache_file)`.
6. Return merged data filtered to `[start:end]`.

With `force_refresh=True`:
- Skip steps 1–3, fetch the full `(start, end)` range, overwrite the cache.

### 4. Equity prices — `grid_resilience/data/equity_prices.py`

`fetch_prices` already has `force_refresh=False`. The cache logic is replaced:

**New logic with `force_refresh=False`:**
1. Read `_PRICE_CACHE` via `read_cached`.
2. Determine **missing tickers**: `set(tickers) - set(cached.columns)`.
3. Determine **date gaps**: use `compute_date_gaps` on the cached index.
4. If missing tickers exist: download those tickers for the full `(start, end)` range; merge as new columns.
5. If date gaps exist: download all tickers for each gap range only; merge as new rows.
6. Call `merge_and_save` and return the result filtered to `tickers` and `[start:end]`.

Steps 4 and 5 are independent and can both happen in the same call.

**With `force_refresh=True`:** download all tickers for full range, overwrite cache.

## File changes summary

| File | Change |
|------|--------|
| `grid_resilience/config.py` | Fix `BACKTEST_END = "2026-05-27"` |
| `grid_resilience/data/cache_utils.py` | **New file** — `read_cached`, `compute_date_gaps`, `merge_and_save` |
| `grid_resilience/data/grid_data.py` | New cache key (no start/end), gap-filling fetch logic, `force_refresh` param |
| `grid_resilience/data/equity_prices.py` | Gap-filling for new tickers + new date ranges |

## Out of scope

- Cache TTL / expiry (once cached, data is kept indefinitely unless `force_refresh=True`)
- Compression or partitioning of large parquet files
- Backfilling historical gaps in the middle of a cached range (only head/tail gaps are handled)
