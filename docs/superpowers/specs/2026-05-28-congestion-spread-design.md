# Congestion Signal via Inter-Zonal Price Spreads

**Date:** 2026-05-28  
**Status:** Approved  
**Scope:** ERCOT and PJM only (first phase)

---

## Problem

The Grid Stress Index (GSI) allocates 25% weight to `congestion_frac` — the share of LMP attributable to transmission constraint stress. The GSI code already handles this signal (see `grid_stress_index.py:72-79`), but the `congestion_frac` column arrives as NaN for all ISOs because:

- `daily_lmp_summary()` only populates `congestion_frac` when a `"congestion"` column is present in the raw LMP DataFrame.
- The gridstatus library does not consistently expose the LMP congestion component under a reliably mapped column name.

When the column is absent, `build_gsi()` falls back to `lmp_z.copy()`, silently giving the LMP signal an effective 65% weight (40% + 25%) instead of the intended 40%.

---

## Solution: Inter-Zonal Price Spreads (Option B)

Use the market-implied congestion signal: the price spread between load zones within an ISO. When a transmission constraint binds between zones, LMPs diverge — the zone behind the constraint sees higher prices, the zone in front sees lower prices. The normalised spread (`spread / |mean_lmp|`) has the same [0, ∞) scale as the existing component-based fraction and is stable across time.

**ERCOT:** Four load zones — LZ_NORTH, LZ_SOUTH, LZ_WEST, LZ_HOUSTON. The West Texas corridor (LZ_WEST vs LZ_NORTH/HOUSTON) is historically the dominant congestion axis. A separate zone-level fetch is added; hub-level data is untouched.

**PJM:** ~17 load zones already fetched at `location_type="zone"`. Spread computed from existing data — no additional fetch required.

**Other ISOs (MISO, CAISO, SPP):** No change. `congestion_frac` remains NaN; GSI falls back to `lmp_z` proxy as before.

---

## Architecture

### Config (`config.py`)

Two new constants:

```python
# ISOs for which inter-zonal spread populates congestion_frac
CONGESTION_SPREAD_ISOS = {"ERCOT", "PJM"}

# Location type for zone-level LMP fetch (spread computation only)
# PJM already uses "zone" in ISO_LOCATION_TYPE — no separate entry needed
ISO_ZONE_LOCATION_TYPE = {
    "ERCOT": "zone",   # → LZ_NORTH, LZ_SOUTH, LZ_WEST, LZ_HOUSTON
}
```

### Data Layer (`grid_data.py`)

**`daily_spread_summary(zone_lmp_df) -> pd.DataFrame`**

- Groups hourly LMP by timestamp across all locations.
- Computes per-hour: `spread = max_lmp − min_lmp`, `frac = spread / (|mean_lmp| + 1e-6)`.
- Returns daily mean of `frac` as a DataFrame indexed by date with column `congestion_frac`.
- Returns empty DataFrame (with the column) if input is empty.

**`fill_congestion_from_spread(daily_lmp, spread_df) -> pd.DataFrame`**

- Fills NaN rows in `daily_lmp["congestion_frac"]` with values from `spread_df["congestion_frac"]`.
- Component-based values (when present) take priority — spread only fills gaps.
- Returns the modified `daily_lmp` DataFrame.

**Cache files:**
- ERCOT zone LMPs: `ercot_lmp_zone.parquet` (new, separate from `ercot_lmp_hub.parquet`)
- PJM zone LMPs: reuses existing `pjm_lmp_zone.parquet` — no new file

### Pipeline (`main.py`)

The grid data loop (step 2) adds spread computation after `daily_lmp_summary`:

```python
if iso in CONGESTION_SPREAD_ISOS:
    zone_raw = fetch_lmp(iso, start, end, location_type=ISO_ZONE_LOCATION_TYPE[iso]) \
               if iso in ISO_ZONE_LOCATION_TYPE else lmp_raw
    if not zone_raw.empty:
        spread_df = daily_spread_summary(zone_raw)
        daily_lmp  = fill_congestion_from_spread(daily_lmp, spread_df)
```

Everything downstream (GSI, betas, factor, backtest) is unchanged.

---

## Data Flow

```
ERCOT hub LMPs  →  daily_lmp_summary()  →  daily_lmp {lmp_max, lmp_mean, spike_hours, congestion_frac=NaN}
ERCOT zone LMPs →  daily_spread_summary() →  spread {congestion_frac}
                                              ↓
                    fill_congestion_from_spread()  →  daily_lmp {congestion_frac filled}
                                              ↓
                              build_gsi()  [unchanged]

PJM zone LMPs   →  daily_lmp_summary()  →  daily_lmp {lmp_max, lmp_mean, spike_hours, congestion_frac=NaN}
                →  daily_spread_summary() →  spread {congestion_frac}
                                              ↓
                    fill_congestion_from_spread()  →  daily_lmp {congestion_frac filled}
```

---

## What Does Not Change

- `build_gsi()` — interface and weights are identical.
- GSI weights in `config.py` — congestion_frac remains at 25%.
- ERCOT hub LMP fetch and cache — hub data is preserved.
- All other ISO pipelines (MISO, CAISO, SPP).
- Factor, portfolio, and backtest modules.

---

## Future Work (Phase 2)

Add LMP component extraction (energy / congestion / loss decomposition) AND use inter-zonal spread as a separate 5th GSI sub-signal with rebalanced weights. This was deferred because it changes the GSI's public interface and is best addressed once the spread signal has been validated in backtesting. See CLAUDE.md for the standing note.

