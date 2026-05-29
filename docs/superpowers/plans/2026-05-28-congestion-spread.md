# Congestion Signal via Inter-Zonal Price Spreads — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate the GSI's `congestion_frac` sub-signal for ERCOT and PJM using inter-zonal LMP price spreads, replacing the silent fallback to `lmp_z` that currently makes LMP carry 65% GSI weight instead of 40%.

**Architecture:** Add two helper functions to `grid_data.py` (`daily_spread_summary` and `fill_congestion_from_spread`), two config constants, and a small loop addition in `main.py`. Nothing downstream of `daily_lmp_summary` changes — the spread just populates the `congestion_frac` column that `build_gsi()` already expects.

**Tech Stack:** Python 3.11, pandas, numpy, pytest, gridstatus (for live fetches — not needed for unit tests)

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `grid_resilience/config.py` | Modify | Add `CONGESTION_SPREAD_ISOS` and `ISO_ZONE_LOCATION_TYPE` |
| `grid_resilience/data/grid_data.py` | Modify | Add `daily_spread_summary()` and `fill_congestion_from_spread()` |
| `grid_resilience/main.py` | Modify | Fetch zone LMPs for ERCOT; call spread helpers in grid data loop |
| `tests/__init__.py` | Create | Empty — marks `tests/` as a package |
| `tests/data/__init__.py` | Create | Empty — marks `tests/data/` as a package |
| `tests/data/test_grid_data_spread.py` | Create | Unit tests for `daily_spread_summary` and `fill_congestion_from_spread` |
| `CLAUDE.md` | Modify | Add Phase 2 future work note |
| `README.md` | Modify | Add `ercot_lmp_zone.parquet` row to cache table |

---

### Task 1: Add config constants

**Files:**
- Modify: `grid_resilience/config.py`

- [ ] **Step 1: Add the two constants to config.py**

Open `grid_resilience/config.py` and append after the `ISO_LOCATION_TYPE` block (after line 42):

```python
# ISOs for which inter-zonal spread fills congestion_frac when component data absent
CONGESTION_SPREAD_ISOS = {"ERCOT", "PJM"}

# Location type used to fetch zone-level LMPs for spread computation.
# PJM already uses "zone" in ISO_LOCATION_TYPE so needs no entry here.
ISO_ZONE_LOCATION_TYPE = {
    "ERCOT": "zone",   # LZ_NORTH, LZ_SOUTH, LZ_WEST, LZ_HOUSTON
}
```

- [ ] **Step 2: Verify the constants import cleanly**

```bash
.venv/bin/python -c "
from grid_resilience.config import CONGESTION_SPREAD_ISOS, ISO_ZONE_LOCATION_TYPE
print(CONGESTION_SPREAD_ISOS)
print(ISO_ZONE_LOCATION_TYPE)
"
```

Expected output:
```
{'ERCOT', 'PJM'}
{'ERCOT': 'zone'}
```

- [ ] **Step 3: Commit**

```bash
git add grid_resilience/config.py
git commit -m "feat: add CONGESTION_SPREAD_ISOS and ISO_ZONE_LOCATION_TYPE config constants"
```

---

### Task 2: Write and pass tests for `daily_spread_summary` and `fill_congestion_from_spread`

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_grid_data_spread.py`
- Modify: `grid_resilience/data/grid_data.py`

- [ ] **Step 1: Create the test package structure**

```bash
mkdir -p tests/data
touch tests/__init__.py
touch tests/data/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/data/test_grid_data_spread.py` with this exact content:

```python
import numpy as np
import pandas as pd
import pytest

from grid_resilience.data.grid_data import daily_spread_summary, fill_congestion_from_spread


# ── daily_spread_summary ──────────────────────────────────────────────────────

def test_spread_empty_input_returns_empty_with_column():
    result = daily_spread_summary(pd.DataFrame())
    assert result.empty
    assert "congestion_frac" in result.columns


def test_spread_single_location_gives_zero_spread():
    """One location → max == min → spread == 0 → frac == 0."""
    times = pd.date_range("2024-01-01", periods=24, freq="h")
    df = pd.DataFrame({
        "time": times,
        "location": ["HUB_A"] * 24,
        "lmp": [50.0] * 24,
    })
    result = daily_spread_summary(df)
    assert not result.empty
    assert result["congestion_frac"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_spread_two_zones_constant_difference():
    """ZONE_A at 100, ZONE_B at 50 for 24 hours.
    Hourly: spread=50, mean=75, frac=50/75.
    Daily mean of 24 identical frac values = 50/75."""
    times = pd.date_range("2024-01-01", periods=24, freq="h")
    df = pd.DataFrame({
        "time": list(times) * 2,
        "location": ["ZONE_A"] * 24 + ["ZONE_B"] * 24,
        "lmp": [100.0] * 24 + [50.0] * 24,
    })
    result = daily_spread_summary(df)
    expected_frac = 50.0 / 75.0
    assert result["congestion_frac"].iloc[0] == pytest.approx(expected_frac, rel=1e-4)


def test_spread_multi_day_aggregation():
    """Two days of data produce two rows, one per day."""
    times_d1 = pd.date_range("2024-01-01", periods=24, freq="h")
    times_d2 = pd.date_range("2024-01-02", periods=24, freq="h")
    all_times = list(times_d1) * 2 + list(times_d2) * 2

    df = pd.DataFrame({
        "time": all_times,
        "location": (["ZONE_A"] * 24 + ["ZONE_B"] * 24) * 2,
        "lmp": [100.0] * 24 + [50.0] * 24 + [80.0] * 24 + [60.0] * 24,
    })
    result = daily_spread_summary(df)
    assert len(result) == 2
    assert result.index.name == "date"
    # Day 1: spread=50, mean=75, frac=50/75
    assert result["congestion_frac"].iloc[0] == pytest.approx(50.0 / 75.0, rel=1e-4)
    # Day 2: spread=20, mean=70, frac=20/70
    assert result["congestion_frac"].iloc[1] == pytest.approx(20.0 / 70.0, rel=1e-4)


# ── fill_congestion_from_spread ───────────────────────────────────────────────

def test_fill_replaces_nan_rows():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    daily_lmp = pd.DataFrame({"congestion_frac": [np.nan, np.nan]}, index=dates)
    daily_lmp.index.name = "date"
    spread_df = pd.DataFrame({"congestion_frac": [0.1, 0.2]}, index=dates)
    spread_df.index.name = "date"

    result = fill_congestion_from_spread(daily_lmp, spread_df)
    assert result["congestion_frac"].iloc[0] == pytest.approx(0.1)
    assert result["congestion_frac"].iloc[1] == pytest.approx(0.2)


def test_fill_preserves_existing_non_nan_values():
    """Component-based value (0.3) must not be overwritten by spread (0.9)."""
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    daily_lmp = pd.DataFrame({"congestion_frac": [np.nan, 0.3]}, index=dates)
    daily_lmp.index.name = "date"
    spread_df = pd.DataFrame({"congestion_frac": [0.1, 0.9]}, index=dates)
    spread_df.index.name = "date"

    result = fill_congestion_from_spread(daily_lmp, spread_df)
    assert result["congestion_frac"].iloc[0] == pytest.approx(0.1)   # filled
    assert result["congestion_frac"].iloc[1] == pytest.approx(0.3)   # preserved


def test_fill_empty_spread_leaves_nans_untouched():
    dates = pd.to_datetime(["2024-01-01"])
    daily_lmp = pd.DataFrame({"congestion_frac": [np.nan]}, index=dates)
    result = fill_congestion_from_spread(daily_lmp, pd.DataFrame(columns=["congestion_frac"]))
    assert pd.isna(result["congestion_frac"].iloc[0])


def test_fill_returns_dataframe():
    dates = pd.to_datetime(["2024-01-01"])
    daily_lmp = pd.DataFrame({"congestion_frac": [np.nan]}, index=dates)
    spread_df = pd.DataFrame({"congestion_frac": [0.5]}, index=dates)
    result = fill_congestion_from_spread(daily_lmp, spread_df)
    assert isinstance(result, pd.DataFrame)
```

- [ ] **Step 3: Run tests — expect ImportError (functions not yet defined)**

```bash
.venv/bin/pytest tests/data/test_grid_data_spread.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'daily_spread_summary' from 'grid_resilience.data.grid_data'`

- [ ] **Step 4: Implement `daily_spread_summary` and `fill_congestion_from_spread` in grid_data.py**

Add the following two functions in `grid_resilience/data/grid_data.py`, after the `load_lmp_from_csv` function and before the `_normalise_lmp_columns` function:

```python
def daily_spread_summary(zone_lmp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily inter-zonal price spread as a congestion_frac proxy.

    Hourly spread = max_lmp - min_lmp across all locations.
    Normalised as frac = spread / (|mean_lmp| + 1e-6).
    Daily value = mean of hourly fracs.

    Returns DataFrame indexed by date with column: congestion_frac
    """
    if zone_lmp_df.empty:
        return pd.DataFrame(columns=["congestion_frac"])

    time_col = _time_col(zone_lmp_df)
    df = zone_lmp_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)

    hourly = df.groupby(level=0)["lmp"].agg(["max", "min", "mean"])
    hourly["spread"] = hourly["max"] - hourly["min"]
    hourly["frac"] = hourly["spread"] / (hourly["mean"].abs() + 1e-6)

    daily_frac = hourly["frac"].resample("D").mean().rename("congestion_frac")
    daily_frac.index = pd.to_datetime(daily_frac.index)
    daily_frac.index.name = "date"
    return daily_frac.to_frame()


def fill_congestion_from_spread(
    daily_lmp: pd.DataFrame,
    spread_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill NaN congestion_frac rows in daily_lmp with spread-based values.
    Component-based values (non-NaN) are preserved.
    Returns the modified daily_lmp DataFrame.
    """
    if spread_df.empty or "congestion_frac" not in spread_df.columns:
        return daily_lmp
    mask = daily_lmp["congestion_frac"].isna()
    daily_lmp.loc[mask, "congestion_frac"] = (
        spread_df["congestion_frac"].reindex(daily_lmp.index[mask])
    )
    return daily_lmp
```

- [ ] **Step 5: Run tests — expect all 8 pass**

```bash
.venv/bin/pytest tests/data/test_grid_data_spread.py -v
```

Expected:
```
tests/data/test_grid_data_spread.py::test_spread_empty_input_returns_empty_with_column PASSED
tests/data/test_grid_data_spread.py::test_spread_single_location_gives_zero_spread PASSED
tests/data/test_grid_data_spread.py::test_spread_two_zones_constant_difference PASSED
tests/data/test_grid_data_spread.py::test_spread_multi_day_aggregation PASSED
tests/data/test_grid_data_spread.py::test_fill_replaces_nan_rows PASSED
tests/data/test_grid_data_spread.py::test_fill_preserves_existing_non_nan_values PASSED
tests/data/test_grid_data_spread.py::test_fill_empty_spread_leaves_nans_untouched PASSED
tests/data/test_grid_data_spread.py::test_fill_returns_dataframe PASSED
8 passed
```

- [ ] **Step 6: Commit**

```bash
git add tests/__init__.py tests/data/__init__.py tests/data/test_grid_data_spread.py grid_resilience/data/grid_data.py
git commit -m "feat: add daily_spread_summary and fill_congestion_from_spread with tests"
```

---

### Task 3: Wire spread computation into the main pipeline

**Files:**
- Modify: `grid_resilience/main.py`

- [ ] **Step 1: Update the config import block**

The current config import block (lines 31–36) reads:
```python
from grid_resilience.config import (
    BACKTEST_START, BACKTEST_END,
    SUPPORTED_ISOS, REBALANCE_FREQ,
    PORTFOLIO_LONG_N, PORTFOLIO_SHORT_N,
)
```

Replace with:
```python
from grid_resilience.config import (
    BACKTEST_START, BACKTEST_END,
    SUPPORTED_ISOS, REBALANCE_FREQ,
    PORTFOLIO_LONG_N, PORTFOLIO_SHORT_N,
    CONGESTION_SPREAD_ISOS, ISO_ZONE_LOCATION_TYPE,
)
```

- [ ] **Step 2: Update the grid_data import block**

The current grid_data import block (lines 43–46) reads:
```python
from grid_resilience.data.grid_data import (
    fetch_lmp, fetch_load, daily_lmp_summary, daily_load_summary,
)
```

Replace with:
```python
from grid_resilience.data.grid_data import (
    fetch_lmp, fetch_load, daily_lmp_summary, daily_load_summary,
    daily_spread_summary, fill_congestion_from_spread,
)
```

- [ ] **Step 3: Replace the grid data loop body**

The current loop body inside the `run()` function (lines 96–107) reads:
```python
    for iso in isos:
        lmp_raw  = fetch_lmp(iso, start, end)
        load_raw = fetch_load(iso, start, end)

        if not lmp_raw.empty:
            daily_lmp_by_iso[iso]  = daily_lmp_summary(lmp_raw, iso)
        else:
            print(f"  [warn] No LMP data for {iso} — skipping.")

        if not load_raw.empty:
            daily_load_by_iso[iso] = daily_load_summary(load_raw)
```

Replace with:
```python
    for iso in isos:
        lmp_raw  = fetch_lmp(iso, start, end)
        load_raw = fetch_load(iso, start, end)

        if not lmp_raw.empty:
            daily_lmp = daily_lmp_summary(lmp_raw, iso)

            if iso in CONGESTION_SPREAD_ISOS:
                zone_loc = ISO_ZONE_LOCATION_TYPE.get(iso)
                zone_raw = fetch_lmp(iso, start, end, location_type=zone_loc) if zone_loc else lmp_raw
                if not zone_raw.empty:
                    spread_df = daily_spread_summary(zone_raw)
                    daily_lmp = fill_congestion_from_spread(daily_lmp, spread_df)

            daily_lmp_by_iso[iso] = daily_lmp
        else:
            print(f"  [warn] No LMP data for {iso} — skipping.")

        if not load_raw.empty:
            daily_load_by_iso[iso] = daily_load_summary(load_raw)
```

- [ ] **Step 4: Verify the module imports cleanly**

```bash
.venv/bin/python -c "from grid_resilience import main; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run the full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all 8 tests pass.

- [ ] **Step 6: Commit**

```bash
git add grid_resilience/main.py
git commit -m "feat: wire inter-zonal spread into GSI congestion_frac for ERCOT and PJM"
```

---

### Task 4: Update CLAUDE.md with Phase 2 future work note

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Append the future work section using Bash**

```bash
cat >> CLAUDE.md << 'EOF'

## Future Work (Tracked)

- **GSI congestion signal — Phase 2:** Add LMP component-based extraction (energy/congestion/loss decomposition from gridstatus) AND promote inter-zonal spread to a separate 5th GSI sub-signal with rebalanced weights. Deferred until the spread signal is validated in backtesting. See `docs/superpowers/specs/2026-05-28-congestion-spread-design.md`.
EOF
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note Phase 2 congestion signal work in CLAUDE.md"
```

---

### Task 5: Update README cache table

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add new cache file row using Bash**

```bash
sed -i '' 's/| `{iso}_fuel_mix.parquet` | gridstatus | one file per ISO |/| `{iso}_fuel_mix.parquet` | gridstatus | one file per ISO |\n| `ercot_lmp_zone.parquet` | gridstatus | ERCOT load-zone LMPs for inter-zonal spread (congestion signal) |/' README.md
```

- [ ] **Step 2: Verify the table**

```bash
grep -A 7 "Cache file" README.md | head -10
```

Expected: the cache table now contains a row for `ercot_lmp_zone.parquet`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add ercot_lmp_zone.parquet to README cache table"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `CONGESTION_SPREAD_ISOS = {"ERCOT", "PJM"}` | Task 1 |
| `ISO_ZONE_LOCATION_TYPE = {"ERCOT": "zone"}` | Task 1 |
| `daily_spread_summary()` — hourly spread/mean, daily avg | Task 2 |
| `fill_congestion_from_spread()` — NaN fill, preserve non-NaN | Task 2 |
| ERCOT: separate zone fetch cached as `ercot_lmp_zone.parquet` | Task 3 (`fetch_lmp` with `location_type="zone"` auto-caches) |
| PJM: uses existing zone data (lmp_raw already zones) | Task 3 (`zone_loc = None → zone_raw = lmp_raw`) |
| main.py loop wired up | Task 3 |
| `build_gsi()` and GSI weights unchanged | Verified — no tasks touch those files |
| CLAUDE.md Phase 2 note | Task 4 |
| README cache table updated | Task 5 |

No gaps found.
