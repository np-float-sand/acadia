# DC Load Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ICR with a two-component (level + momentum) data-center load-growth signal in the regulated-utility path of the business-model-aware factor, sourced from PJM's public interconnection queue and zone-level metered load.

**Architecture:** A new `grid_resilience/data/dc_load_data.py` module fetches and cleans the PJM interconnection queue (via direct DataMiner API calls, matching the existing `_fetch_pjm_load_direct` pattern — NOT via `gridstatus.get_interconnection_queue()`, which is broken in the installed `gridstatus==0.34.0`), reconstructs point-in-time queue membership from each project's dated fields, and combines it with a new zone-level load fetch (`grid_data.fetch_zonal_load()`) to produce a `dates x tickers` DataFrame in the same shape `equity_prices.fetch_icr()` already produces. `main.py` gets a `--regulated-signal icr|dc-queue` flag that swaps which DataFrame flows into `build_rolling_factor()`'s existing `icr_history` parameter — no change to the blending math in `resilience_score.py` beyond generalizing the ICR-specific reporting lag into a parameter.

**Tech Stack:** Python, pandas, `requests` (direct PJM DataMiner 2 API calls — the codebase already bypasses `gridstatus` for PJM LMP/load due to gridstatus's client-side filtering and version bugs), `pytest` with `unittest.mock.patch`.

## Global Constraints

- Follow existing code style: `from __future__ import annotations` at top of every new module, type hints on all function signatures, no comments explaining *what* code does (only *why*, and only when non-obvious).
- All new config constants go in `grid_resilience/config.py`, never hardcoded in the module that uses them.
- All new fetchers must cache to `CACHE_DIR` (`grid_resilience/config.py: CACHE_DIR`) — never re-hit a live API inside a test.
- Reuse `grid_resilience/data/cache_utils.py`'s long-format monthly cache helpers (`find_missing_months`, `read_monthly_cache`, `save_monthly_chunks`, `chunk_path`, `month_bounds`) for the new zonal load cache — do not invent a new caching scheme.
- No existing test may break. `tests/factor/test_business_model_arch.py` calls `build_factor(..., icr=icr, ...)` by keyword in 10 places — that parameter name and position must not change.
- `PJM_API_KEY` is already set in this environment (verified live during design) — new PJM fetches can be run and manually verified against the real API during implementation, not just mocked.

---

### Task 1: `fetch_zonal_load()` — PJM zone-level metered load

**Files:**
- Modify: `grid_resilience/data/grid_data.py`
- Test: `tests/data/test_zonal_load.py` (new)

**Interfaces:**
- Produces: `fetch_zonal_load(start: str, end: str, use_cache: bool = True) -> pd.DataFrame` — columns `[time, zone, load_mw]`, long format, one row per zone per hour.

The existing `_fetch_pjm_load_direct()` (grid_resilience/data/grid_data.py:171-208) already calls PJM's `hrl_load_metered` endpoint and aggregates away the `zone` field via `df.groupby("time")["mw"].sum()`. The raw JSON items already contain a `"zone"` key (verified live during design — e.g. `{"zone": "AEP", "mw": 3160.712, ...}`). Add a sibling function that keeps `zone` instead of summing it away, and a caching wrapper matching the `fetch_load()` pattern (grid_resilience/data/grid_data.py:431-479) exactly, but with dataset key `"load_zonal"`.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_zonal_load.py`:

```python
import pandas as pd
from unittest.mock import patch
from grid_resilience.data.grid_data import _fetch_pjm_load_by_zone_direct


def _mock_pjm_get_response(items):
    return {"items": items, "totalRows": len(items)}


def test_fetch_pjm_load_by_zone_direct_keeps_zone_column():
    items = [
        {"datetime_beginning_utc": "2024-06-01T04:00:00", "zone": "AEP", "mw": 3160.712},
        {"datetime_beginning_utc": "2024-06-01T04:00:00", "zone": "DOM", "mw": 10942.767},
        {"datetime_beginning_utc": "2024-06-01T05:00:00", "zone": "AEP", "mw": 3100.0},
    ]
    with patch(
        "grid_resilience.data.grid_data._pjm_get",
        return_value=_mock_pjm_get_response(items),
    ):
        df = _fetch_pjm_load_by_zone_direct("key", "2024-06-01", "2024-06-02")

    assert set(df.columns) == {"time", "zone", "load_mw"}
    assert len(df) == 3
    aep_rows = df[df["zone"] == "AEP"]
    assert len(aep_rows) == 2
    assert aep_rows["load_mw"].tolist() == [3160.712, 3100.0]


def test_fetch_pjm_load_by_zone_direct_empty_response():
    with patch(
        "grid_resilience.data.grid_data._pjm_get",
        return_value=_mock_pjm_get_response([]),
    ):
        df = _fetch_pjm_load_by_zone_direct("key", "2024-06-01", "2024-06-02")
    assert df.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_zonal_load.py -v`
Expected: FAIL with `ImportError: cannot import name '_fetch_pjm_load_by_zone_direct'`

- [ ] **Step 3: Implement `_fetch_pjm_load_by_zone_direct`**

In `grid_resilience/data/grid_data.py`, add immediately after `_fetch_pjm_load_direct` (after line 208):

```python
def _fetch_pjm_load_by_zone_direct(api_key: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch PJM metered hourly load directly from DataMiner 2, keeping the
    per-zone breakdown (unlike _fetch_pjm_load_direct, which sums to system total).
    Used for the DC load signal's zone-size denominator.
    """
    start_ept = pd.Timestamp(start).strftime("%m/%d/%Y %H:%M")
    end_ept   = pd.Timestamp(end).strftime("%m/%d/%Y %H:%M")
    url       = "https://api.pjm.com/api/v1/hrl_load_metered"
    row_count = 50000
    start_row = 1
    all_items: list = []

    while True:
        params = {
            "startRow":  start_row,
            "rowCount":  row_count,
            "datetime_beginning_ept": f"{start_ept}to{end_ept}",
        }
        data  = _pjm_get(url, params, api_key)
        items = data.get("items", [])
        all_items.extend(items)
        total = data.get("totalRows", 0)
        if start_row + row_count - 1 >= total or not items:
            break
        start_row += row_count
        time.sleep(1)

    if not all_items:
        return pd.DataFrame(columns=["time", "zone", "load_mw"])

    df = pd.DataFrame(all_items)
    df["time"] = pd.to_datetime(df["datetime_beginning_utc"])
    by_zone = df.groupby(["time", "zone"])["mw"].sum().rename("load_mw").reset_index()
    return by_zone
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_zonal_load.py -v`
Expected: 2 passed

- [ ] **Step 5: Add the caching wrapper `fetch_zonal_load()`**

Add to `tests/data/test_zonal_load.py`:

```python
def test_fetch_zonal_load_caches_to_disk(tmp_path, monkeypatch):
    from grid_resilience.data import grid_data
    monkeypatch.setattr(grid_data, "CACHE_DIR", tmp_path)

    items = [{"datetime_beginning_utc": "2024-06-01T04:00:00", "zone": "AEP", "mw": 100.0}]
    with patch(
        "grid_resilience.data.grid_data._pjm_get",
        return_value=_mock_pjm_get_response(items),
    ), patch.dict("os.environ", {"PJM_API_KEY": "test-key"}):
        df1 = grid_data.fetch_zonal_load("2024-06-01", "2024-06-30")
    assert not df1.empty

    cache_file = grid_data._cache_path("PJM", "load_zonal")
    chunk = grid_data.chunk_path(cache_file, 2024, 6)
    assert chunk.exists()

    with patch("grid_resilience.data.grid_data._pjm_get") as mock_get:
        df2 = grid_data.fetch_zonal_load("2024-06-01", "2024-06-30")
    mock_get.assert_not_called()
    assert len(df2) == len(df1)
```

Add to `grid_resilience/data/grid_data.py`, immediately after `fetch_load()` (after line 479):

```python
def fetch_zonal_load(
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch PJM hourly load broken out by zone. PJM-only (other ISOs return empty) —
    used for the DC load signal's zone-size denominator.
    Returns DataFrame with columns: time, zone, load_mw
    """
    cache_base = _cache_path("PJM", "load_zonal")
    api_key = os.environ.get("PJM_API_KEY", "")

    if use_cache:
        missing = find_missing_months(cache_base, start, end)
        if missing:
            def _fetch_month(ym: tuple[int, int]) -> None:
                time.sleep(12)
                ms, me = month_bounds(*ym)
                df = _fetch_pjm_load_by_zone_direct(api_key, ms, me)
                if not df.empty:
                    save_monthly_chunks(df, cache_base, "time")

            for ym in missing:
                _fetch_month(ym)

        merged = read_monthly_cache(cache_base, start, end)
    else:
        merged = _fetch_pjm_load_by_zone_direct(api_key, start, end)

    return _filter_time(merged, start, end)
```

Note: fetches are serialized (no `ThreadPoolExecutor`, unlike `fetch_load`'s non-PJM path) because PJM's `hrl_load_metered` endpoint has a strict rate limit — `fetch_load()` already serializes PJM for the same reason (grid_resilience/data/grid_data.py:468-469).

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_zonal_load.py -v`
Expected: 3 passed

- [ ] **Step 7: Manually verify against the live API and check historical depth**

Run:
```bash
source ~/.zshrc
.venv/bin/python -c "
from grid_resilience.data.grid_data import fetch_zonal_load
df = fetch_zonal_load('2017-01-01', '2017-02-01')
print(df.shape)
print(sorted(df['zone'].unique()))
print(df['time'].min(), df['time'].max())
"
```
Expected: non-empty result with `DOM`, `AEP`, `PSEG`, `PPL`, `ATSI`, `JCPL` all present in the zone list, confirming `hrl_load_metered` has coverage back to early 2017 (needed for a trailing-12-month zone-size denominator at the 2018-01-01 backtest start). If this returns empty or errors, record the actual earliest available date — Task 5's `zone_size()` trailing window will need to shrink near the start of the backtest instead of assuming a full 365 days is always available.

- [ ] **Step 8: Commit**

```bash
git add grid_resilience/data/grid_data.py tests/data/test_zonal_load.py
git commit -m "feat: add fetch_zonal_load for PJM zone-level metered load"
```

---

### Task 2: PJM interconnection queue fetch + cleaning

**Files:**
- Create: `grid_resilience/data/dc_load_data.py`
- Modify: `grid_resilience/config.py`
- Test: `tests/data/test_dc_load_data.py` (new)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `normalize_transmission_owner(to: str) -> str`
  - `fetch_interconnection_queue(use_cache: bool = True) -> pd.DataFrame` — cleaned queue with columns `[zone, mw_capacity, project_type, submitted_date, withdrawal_date, actual_in_service_date]`

Add to `grid_resilience/config.py`, after the `USE_ICR` line (find via `grep -n USE_ICR grid_resilience/config.py`):

```python
from typing import Literal

REGULATED_SIGNAL: Literal["icr", "dc_queue"] = "dc_queue"

DC_QUEUE_MW_MIN        = 100.0   # ignore sub-100MW queue entries as noise
DC_QUEUE_PROJECT_TYPES = ["Generation Interconnection"]
DC_MOMENTUM_WINDOW_DAYS = 90
DC_LEVEL_WEIGHT    = 0.6
DC_MOMENTUM_WEIGHT = 0.4
```

(If `config.py` already has `from typing import Literal` near the top, don't duplicate the import — add the import at the top of the file instead and just append the constants.)

- [ ] **Step 1: Write the failing test for TO normalization**

Create `tests/data/test_dc_load_data.py`:

```python
import pandas as pd
from grid_resilience.data.dc_load_data import normalize_transmission_owner


def test_normalize_transmission_owner_direct_match():
    assert normalize_transmission_owner("AEP") == "AEP"
    assert normalize_transmission_owner("PSEG") == "PSEG"
    assert normalize_transmission_owner("PECO") == "PECO"


def test_normalize_transmission_owner_case_insensitive():
    assert normalize_transmission_owner("Dayton") == "DAYTON"
    assert normalize_transmission_owner("ComEd") == "COMED"


def test_normalize_transmission_owner_dominion_override():
    assert normalize_transmission_owner("Dominion") == "DOM"


def test_normalize_transmission_owner_strips_semicolon_duplicates():
    assert normalize_transmission_owner("PSEG; PSEG") == "PSEG"


def test_normalize_transmission_owner_handles_nan():
    assert normalize_transmission_owner(float("nan")) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grid_resilience.data.dc_load_data'`

- [ ] **Step 3: Create `dc_load_data.py` with `normalize_transmission_owner`**

Create `grid_resilience/data/dc_load_data.py`:

```python
from __future__ import annotations

"""
Data-center load growth signal for the regulated-utility factor path.

Proxies data-center demand via co-located generation interconnection activity
in a utility's PJM zones (PJM has no separate load-interconnection queue —
data centers connect at the distribution level through the local utility).
See docs/superpowers/specs/2026-07-06-dc-load-signal-design.md.
"""

import pandas as pd

from grid_resilience.config import (
    CACHE_DIR,
    DC_QUEUE_MW_MIN,
    DC_QUEUE_PROJECT_TYPES,
    DC_MOMENTUM_WINDOW_DAYS,
    DC_LEVEL_WEIGHT,
    DC_MOMENTUM_WEIGHT,
)

# PJM interconnection queue "Transmission Owner" strings that don't uppercase
# to their matching TICKER_NODE_MAP zone name.
_TO_ZONE_OVERRIDES = {
    "DOMINION": "DOM",
}

_QUEUE_CACHE_FILE = CACHE_DIR / "pjm_interconnection_queue.parquet"


def normalize_transmission_owner(to) -> str | None:
    """Map a raw PJM 'Transmission Owner' string to a TICKER_NODE_MAP zone name."""
    if not isinstance(to, str):
        return None
    first = to.split(";")[0].strip()
    upper = first.upper()
    return _TO_ZONE_OVERRIDES.get(upper, upper)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v`
Expected: 5 passed

- [ ] **Step 5: Write the failing test for queue cleaning**

Add to `tests/data/test_dc_load_data.py`:

```python
def _raw_queue_row(**overrides):
    row = {
        "Transmission Owner": "PSEG",
        "MW Capacity": 500.0,
        "Project Type": "Generation Interconnection",
        "Status": "Active",
        "Submitted Date": "1/1/2019",
        "Withdrawal Date": None,
        "Actual In Service Date": None,
    }
    row.update(overrides)
    return row


def test_clean_queue_keeps_valid_generation_rows():
    from grid_resilience.data.dc_load_data import _clean_queue
    raw = pd.DataFrame([_raw_queue_row()])
    cleaned = _clean_queue(raw)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["zone"] == "PSEG"
    assert cleaned.iloc[0]["mw_capacity"] == 500.0
    assert pd.notna(cleaned.iloc[0]["submitted_date"])


def test_clean_queue_drops_sub_threshold_mw():
    from grid_resilience.data.dc_load_data import _clean_queue
    raw = pd.DataFrame([_raw_queue_row(**{"MW Capacity": 50.0})])
    cleaned = _clean_queue(raw)
    assert cleaned.empty


def test_clean_queue_drops_non_generation_project_types():
    from grid_resilience.data.dc_load_data import _clean_queue
    raw = pd.DataFrame([_raw_queue_row(**{"Project Type": "Merchant Transmission"})])
    cleaned = _clean_queue(raw)
    assert cleaned.empty


def test_clean_queue_drops_unmappable_transmission_owner():
    from grid_resilience.data.dc_load_data import _clean_queue
    raw = pd.DataFrame([_raw_queue_row(**{"Transmission Owner": None})])
    cleaned = _clean_queue(raw)
    assert cleaned.empty


def test_clean_queue_does_not_filter_by_status():
    """Status is a current-snapshot field with no date attached — filtering by
    it would corrupt the point-in-time reconstruction (see design spec).
    A withdrawn project must still appear in the cleaned queue; date-based
    filtering happens later in queued_mw()/in_queue()."""
    from grid_resilience.data.dc_load_data import _clean_queue
    raw = pd.DataFrame([_raw_queue_row(**{
        "Status": "Withdrawn",
        "Withdrawal Date": "6/1/2020",
    })])
    cleaned = _clean_queue(raw)
    assert len(cleaned) == 1
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v`
Expected: FAIL — `_clean_queue` doesn't exist

- [ ] **Step 7: Implement `_clean_queue`**

Add to `grid_resilience/data/dc_load_data.py`:

```python
def _clean_queue(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw PJM interconnection queue rows into the columns used by
    queued_mw()/in_queue(). Filters on static (non-date) fields only —
    Status is deliberately NOT filtered here (see design spec: it has no
    associated date, so filtering on it would corrupt point-in-time
    reconstruction for historical dates).
    """
    df = raw.copy()
    df["zone"] = df["Transmission Owner"].apply(normalize_transmission_owner)
    df["mw_capacity"] = pd.to_numeric(df["MW Capacity"], errors="coerce")
    df["submitted_date"] = pd.to_datetime(df["Submitted Date"], errors="coerce")
    df["withdrawal_date"] = pd.to_datetime(df["Withdrawal Date"], errors="coerce")
    df["actual_in_service_date"] = pd.to_datetime(df["Actual In Service Date"], errors="coerce")

    mask = (
        df["zone"].notna()
        & df["mw_capacity"].notna()
        & (df["mw_capacity"] >= DC_QUEUE_MW_MIN)
        & df["Project Type"].isin(DC_QUEUE_PROJECT_TYPES)
        & df["submitted_date"].notna()
    )
    cols = ["zone", "mw_capacity", "submitted_date", "withdrawal_date", "actual_in_service_date"]
    return df.loc[mask, cols].reset_index(drop=True)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v`
Expected: 10 passed

- [ ] **Step 9: Implement `fetch_interconnection_queue` with caching**

Add to `grid_resilience/data/dc_load_data.py`:

```python
def _load_raw_queue() -> pd.DataFrame:
    """Pull PJM's raw interconnection queue Excel export via gridstatus.
    Uses get_raw_interconnection_queue(), NOT get_interconnection_queue() —
    the latter raises AssertionError on gridstatus==0.34.0 (PJM removed a
    'Revised In Service Date' column this gridstatus version still expects)."""
    import os
    import gridstatus

    pjm = gridstatus.PJM(api_key=os.environ.get("PJM_API_KEY"))
    raw_bytes = pjm.get_raw_interconnection_queue()
    return pd.read_excel(raw_bytes)


def fetch_interconnection_queue(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch and clean the PJM interconnection queue.
    Cached for 1 day (queue updates weekly; daily freshness is more than enough).
    """
    today = pd.Timestamp.now().normalize()
    if use_cache and _QUEUE_CACHE_FILE.exists():
        age_days = (today - pd.Timestamp(_QUEUE_CACHE_FILE.stat().st_mtime, unit="s").normalize()).days
        if age_days < 1:
            return pd.read_parquet(_QUEUE_CACHE_FILE)

    cleaned = _clean_queue(_load_raw_queue())
    if use_cache:
        cleaned.to_parquet(_QUEUE_CACHE_FILE)
    return cleaned
```

- [ ] **Step 10: Manually verify against the live API**

Run:
```bash
source ~/.zshrc
.venv/bin/python -c "
from grid_resilience.data.dc_load_data import fetch_interconnection_queue
df = fetch_interconnection_queue(use_cache=False)
print(df.shape)
print(df['zone'].value_counts().head(10))
print(df['submitted_date'].min(), df['submitted_date'].max())
"
```
Expected: non-empty, with `DOM`, `PSEG`, `AEP`, `PPL`, `ATSI`, `COMED`/`PECO`/`BGE`/`PEPCO` zones present.

- [ ] **Step 11: Commit**

```bash
git add grid_resilience/data/dc_load_data.py grid_resilience/config.py tests/data/test_dc_load_data.py
git commit -m "feat: add PJM interconnection queue fetch and cleaning for DC load signal"
```

---

### Task 3: Point-in-time queue membership and level/momentum MW

**Files:**
- Modify: `grid_resilience/data/dc_load_data.py`
- Test: `tests/data/test_dc_load_data.py`

**Interfaces:**
- Consumes: `_clean_queue()` output shape from Task 2 (`[zone, mw_capacity, submitted_date, withdrawal_date, actual_in_service_date]`)
- Produces:
  - `in_queue(queue: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series` (boolean mask)
  - `queued_mw(queue: pd.DataFrame, zone: str, as_of: pd.Timestamp) -> float`
  - `new_mw_since(queue: pd.DataFrame, zone: str, as_of: pd.Timestamp, window_days: int) -> float`

This is the most safety-critical piece — it's what makes the backtest historically valid rather than look-ahead-biased. Test the exact boundary cases the design review flagged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/data/test_dc_load_data.py`:

```python
def _queue_row(zone="PSEG", mw=500.0, submitted="1/1/2019", withdrawn=None, in_service=None):
    return {
        "zone": zone,
        "mw_capacity": mw,
        "submitted_date": pd.Timestamp(submitted),
        "withdrawal_date": pd.Timestamp(withdrawn) if withdrawn else pd.NaT,
        "actual_in_service_date": pd.Timestamp(in_service) if in_service else pd.NaT,
    }


def test_in_queue_true_before_submission_is_false():
    from grid_resilience.data.dc_load_data import in_queue
    queue = pd.DataFrame([_queue_row(submitted="6/1/2020")])
    mask = in_queue(queue, pd.Timestamp("2020-01-01"))
    assert mask.tolist() == [False]


def test_in_queue_true_after_submission_before_exit():
    from grid_resilience.data.dc_load_data import in_queue
    queue = pd.DataFrame([_queue_row(submitted="1/1/2019")])
    mask = in_queue(queue, pd.Timestamp("2020-01-01"))
    assert mask.tolist() == [True]


def test_in_queue_false_after_withdrawal():
    from grid_resilience.data.dc_load_data import in_queue
    queue = pd.DataFrame([_queue_row(submitted="1/1/2019", withdrawn="6/1/2020")])
    assert in_queue(queue, pd.Timestamp("2020-01-01")).tolist() == [True]
    assert in_queue(queue, pd.Timestamp("2020-07-01")).tolist() == [False]


def test_in_queue_false_after_actual_in_service():
    from grid_resilience.data.dc_load_data import in_queue
    queue = pd.DataFrame([_queue_row(submitted="1/1/2019", in_service="6/1/2020")])
    assert in_queue(queue, pd.Timestamp("2020-01-01")).tolist() == [True]
    assert in_queue(queue, pd.Timestamp("2020-07-01")).tolist() == [False]


def test_queued_mw_sums_only_matching_zone_and_active_projects():
    from grid_resilience.data.dc_load_data import queued_mw
    queue = pd.DataFrame([
        _queue_row(zone="PSEG", mw=500.0, submitted="1/1/2019"),
        _queue_row(zone="PSEG", mw=300.0, submitted="1/1/2019", withdrawn="6/1/2019"),
        _queue_row(zone="DOM", mw=1000.0, submitted="1/1/2019"),
    ])
    assert queued_mw(queue, "PSEG", pd.Timestamp("2020-01-01")) == 500.0


def test_new_mw_since_only_counts_recent_submissions():
    from grid_resilience.data.dc_load_data import new_mw_since
    queue = pd.DataFrame([
        _queue_row(zone="PSEG", mw=200.0, submitted="2020-01-15"),  # within window
        _queue_row(zone="PSEG", mw=800.0, submitted="2019-01-01"),  # outside window
    ])
    as_of = pd.Timestamp("2020-02-01")
    result = new_mw_since(queue, "PSEG", as_of, window_days=90)
    assert result == 200.0


def test_new_mw_since_excludes_projects_already_exited():
    from grid_resilience.data.dc_load_data import new_mw_since
    queue = pd.DataFrame([
        _queue_row(zone="PSEG", mw=200.0, submitted="2020-01-15", withdrawn="2020-01-20"),
    ])
    as_of = pd.Timestamp("2020-02-01")
    assert new_mw_since(queue, "PSEG", as_of, window_days=90) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v -k "in_queue or queued_mw or new_mw_since"`
Expected: FAIL — functions don't exist

- [ ] **Step 3: Implement `in_queue`, `queued_mw`, `new_mw_since`**

Add to `grid_resilience/data/dc_load_data.py`:

```python
def in_queue(queue: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """
    Boolean mask: which queue rows represent a project actually in the queue
    as of `as_of`, reconstructed from each project's own dated fields
    (see design spec's point-in-time reconstruction section).
    """
    submitted = queue["submitted_date"] <= as_of
    not_withdrawn = queue["withdrawal_date"].isna() | (queue["withdrawal_date"] > as_of)
    not_in_service = queue["actual_in_service_date"].isna() | (queue["actual_in_service_date"] > as_of)
    return submitted & not_withdrawn & not_in_service


def queued_mw(queue: pd.DataFrame, zone: str, as_of: pd.Timestamp) -> float:
    """Total MW capacity in `zone`'s queue as of `as_of`."""
    mask = in_queue(queue, as_of) & (queue["zone"] == zone)
    return float(queue.loc[mask, "mw_capacity"].sum())


def new_mw_since(queue: pd.DataFrame, zone: str, as_of: pd.Timestamp, window_days: int) -> float:
    """MW capacity in `zone`'s queue as of `as_of` that was submitted within
    the trailing `window_days` — the momentum component's numerator."""
    cutoff = as_of - pd.Timedelta(days=window_days)
    mask = in_queue(queue, as_of) & (queue["zone"] == zone) & (queue["submitted_date"] > cutoff)
    return float(queue.loc[mask, "mw_capacity"].sum())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add grid_resilience/data/dc_load_data.py tests/data/test_dc_load_data.py
git commit -m "feat: add point-in-time queue membership and level/momentum MW calculations"
```

---

### Task 4: Zone size denominator and multi-zone ticker aggregation

**Files:**
- Modify: `grid_resilience/data/dc_load_data.py`
- Test: `tests/data/test_dc_load_data.py`

**Interfaces:**
- Consumes: `fetch_zonal_load()` output shape from Task 1 (`[time, zone, load_mw]`)
- Produces: `zone_size(zonal_load: pd.DataFrame, zones: list[str], as_of: pd.Timestamp) -> float`

- [ ] **Step 1: Write the failing tests**

Add to `tests/data/test_dc_load_data.py`:

```python
def _zonal_load_row(zone, time, mw):
    return {"time": pd.Timestamp(time), "zone": zone, "load_mw": mw}


def test_zone_size_averages_trailing_12_months_single_zone():
    from grid_resilience.data.dc_load_data import zone_size
    rows = [_zonal_load_row("PSEG", f"2019-{m:02d}-15", 1000.0 + m) for m in range(1, 13)]
    zonal_load = pd.DataFrame(rows)
    result = zone_size(zonal_load, ["PSEG"], pd.Timestamp("2020-01-01"))
    expected = sum(1000.0 + m for m in range(1, 13)) / 12
    assert abs(result - expected) < 1e-6


def test_zone_size_sums_multiple_zones():
    from grid_resilience.data.dc_load_data import zone_size
    zonal_load = pd.DataFrame([
        _zonal_load_row("PECO", "2019-06-15", 1000.0),
        _zonal_load_row("BGE", "2019-06-15", 500.0),
    ])
    result = zone_size(zonal_load, ["PECO", "BGE"], pd.Timestamp("2019-12-01"))
    assert result == 1500.0


def test_zone_size_excludes_data_after_as_of():
    from grid_resilience.data.dc_load_data import zone_size
    zonal_load = pd.DataFrame([
        _zonal_load_row("PSEG", "2019-06-15", 1000.0),
        _zonal_load_row("PSEG", "2020-06-15", 9000.0),  # future — must not leak in
    ])
    result = zone_size(zonal_load, ["PSEG"], pd.Timestamp("2019-12-01"))
    assert result == 1000.0


def test_zone_size_returns_nan_when_no_data():
    from grid_resilience.data.dc_load_data import zone_size
    zonal_load = pd.DataFrame(columns=["time", "zone", "load_mw"])
    result = zone_size(zonal_load, ["PSEG"], pd.Timestamp("2019-12-01"))
    assert pd.isna(result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v -k zone_size`
Expected: FAIL — `zone_size` doesn't exist

- [ ] **Step 3: Implement `zone_size`**

Add to `grid_resilience/data/dc_load_data.py`:

```python
def zone_size(zonal_load: pd.DataFrame, zones: list[str], as_of: pd.Timestamp) -> float:
    """
    Trailing-12-month average hourly load (MW), summed across `zones`.
    Used as the utility-size denominator for the level/momentum ratios.
    Returns NaN if no data is available in the window (e.g. before PJM's
    metered-load history begins — see Task 1 Step 7's coverage check).
    """
    if zonal_load.empty:
        return float("nan")
    window_start = as_of - pd.Timedelta(days=365)
    in_window = zonal_load[
        (zonal_load["zone"].isin(zones))
        & (zonal_load["time"] > window_start)
        & (zonal_load["time"] <= as_of)
    ]
    if in_window.empty:
        return float("nan")
    per_zone_avg = in_window.groupby("zone")["load_mw"].mean()
    return float(per_zone_avg.sum())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v`
Expected: 21 passed

- [ ] **Step 5: Commit**

```bash
git add grid_resilience/data/dc_load_data.py tests/data/test_dc_load_data.py
git commit -m "feat: add zone_size trailing-12-month load denominator"
```

---

### Task 5: `compute_dc_load_signal()` — combine level + momentum per ticker per date

**Files:**
- Modify: `grid_resilience/data/dc_load_data.py`
- Test: `tests/data/test_dc_load_data.py`

**Interfaces:**
- Consumes: `queued_mw`, `new_mw_since`, `zone_size` from Tasks 3-4; `TICKER_NODE_MAP` shape (`{"load_zones": list[str], ...}`) from `grid_resilience/data/utility_node_map.py`
- Produces: `compute_dc_load_signal(tickers: list[str], node_map: dict, queue: pd.DataFrame, zonal_load: pd.DataFrame, as_of_dates: list[pd.Timestamp]) -> pd.DataFrame` — index = `as_of_dates`, columns = tickers, values = raw (pre-z-score) DC score. Only tickers with `iso == "PJM"` in `node_map` get real values; others are all-NaN columns.

- [ ] **Step 1: Write the failing tests**

Add to `tests/data/test_dc_load_data.py`:

```python
def _node_map(**tickers):
    return tickers


def test_compute_dc_load_signal_single_pjm_ticker():
    from grid_resilience.data.dc_load_data import compute_dc_load_signal
    queue = pd.DataFrame([_queue_row(zone="PSEG", mw=1000.0, submitted="2019-01-01")])
    zonal_load = pd.DataFrame([_zonal_load_row("PSEG", "2019-06-15", 5000.0)])
    node_map = _node_map(PEG={"iso": "PJM", "load_zones": ["PSEG"]})

    result = compute_dc_load_signal(
        tickers=["PEG"], node_map=node_map, queue=queue, zonal_load=zonal_load,
        as_of_dates=[pd.Timestamp("2020-01-01")],
    )
    assert list(result.columns) == ["PEG"]
    # level = 1000/5000 = 0.2, momentum = 0 (submitted well outside 90d window)
    assert abs(result.loc[pd.Timestamp("2020-01-01"), "PEG"] - (0.6 * 0.2 + 0.4 * 0.0)) < 1e-6


def test_compute_dc_load_signal_non_pjm_ticker_is_nan():
    from grid_resilience.data.dc_load_data import compute_dc_load_signal
    queue = pd.DataFrame(columns=["zone", "mw_capacity", "submitted_date", "withdrawal_date", "actual_in_service_date"])
    zonal_load = pd.DataFrame(columns=["time", "zone", "load_mw"])
    node_map = _node_map(VST={"iso": "ERCOT", "load_zones": ["NORTH"]})

    result = compute_dc_load_signal(
        tickers=["VST"], node_map=node_map, queue=queue, zonal_load=zonal_load,
        as_of_dates=[pd.Timestamp("2020-01-01")],
    )
    assert pd.isna(result.loc[pd.Timestamp("2020-01-01"), "VST"])


def test_compute_dc_load_signal_multi_zone_ticker_sums_zones():
    from grid_resilience.data.dc_load_data import compute_dc_load_signal
    queue = pd.DataFrame([
        _queue_row(zone="PECO", mw=400.0, submitted="2019-01-01"),
        _queue_row(zone="BGE", mw=100.0, submitted="2019-01-01"),
    ])
    zonal_load = pd.DataFrame([
        _zonal_load_row("PECO", "2019-06-15", 2000.0),
        _zonal_load_row("BGE", "2019-06-15", 500.0),
    ])
    node_map = _node_map(EXC={"iso": "PJM", "load_zones": ["PECO", "BGE"]})

    result = compute_dc_load_signal(
        tickers=["EXC"], node_map=node_map, queue=queue, zonal_load=zonal_load,
        as_of_dates=[pd.Timestamp("2020-01-01")],
    )
    # level = (400+100)/(2000+500) = 0.2
    assert abs(result.loc[pd.Timestamp("2020-01-01"), "EXC"] - 0.6 * 0.2) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v -k compute_dc_load_signal`
Expected: FAIL — function doesn't exist

- [ ] **Step 3: Implement `compute_dc_load_signal`**

Add to `grid_resilience/data/dc_load_data.py`:

```python
def compute_dc_load_signal(
    tickers: list[str],
    node_map: dict,
    queue: pd.DataFrame,
    zonal_load: pd.DataFrame,
    as_of_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    Raw (pre-z-score) DC load signal per ticker per date:
        0.6 * level + 0.4 * momentum
    where level = queued_mw / zone_size, momentum = new_mw_since / zone_size.
    Only PJM tickers get real values (see design spec's v1 PJM-only limitation);
    all others are NaN, to be filled by fill_with_icr() downstream.
    """
    rows = {}
    for date in as_of_dates:
        row = {}
        for ticker in tickers:
            info = node_map.get(ticker, {})
            if info.get("iso") != "PJM":
                row[ticker] = float("nan")
                continue
            zones = info.get("load_zones", [])
            size = zone_size(zonal_load, zones, date)
            if not size or pd.isna(size) or size <= 0:
                row[ticker] = float("nan")
                continue
            level = sum(queued_mw(queue, z, date) for z in zones) / size
            momentum = sum(
                new_mw_since(queue, z, date, DC_MOMENTUM_WINDOW_DAYS) for z in zones
            ) / size
            row[ticker] = DC_LEVEL_WEIGHT * level + DC_MOMENTUM_WEIGHT * momentum
        rows[date] = row
    return pd.DataFrame.from_dict(rows, orient="index")[tickers]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v`
Expected: 24 passed

- [ ] **Step 5: Commit**

```bash
git add grid_resilience/data/dc_load_data.py tests/data/test_dc_load_data.py
git commit -m "feat: add compute_dc_load_signal combining level and momentum per ticker"
```

---

### Task 6: `fill_with_icr()` — non-PJM ticker fallback

**Files:**
- Modify: `grid_resilience/data/dc_load_data.py`
- Test: `tests/data/test_dc_load_data.py`

**Interfaces:**
- Consumes: `compute_dc_load_signal()` output shape (dates x tickers) from Task 5; `equity_prices.fetch_icr()` output shape (quarter-end dates x tickers)
- Produces: `fill_with_icr(dc_history: pd.DataFrame, icr_history: pd.DataFrame | None, lag_days: int = 45) -> pd.DataFrame`

This is the fix for the review finding that a flat cross-sectional mean fallback would collapse 8 of 14 regulated tickers to an identical value.

- [ ] **Step 1: Write the failing tests**

Add to `tests/data/test_dc_load_data.py`:

```python
def test_fill_with_icr_fills_nan_columns_from_most_recent_icr():
    from grid_resilience.data.dc_load_data import fill_with_icr
    dc_history = pd.DataFrame(
        {"PEG": [0.3], "WEC": [float("nan")]},
        index=[pd.Timestamp("2020-06-01")],
    )
    icr_history = pd.DataFrame(
        {"WEC": [4.0]},
        index=[pd.Timestamp("2020-03-01")],  # 92 days before as_of, past a 45d lag
    )
    result = fill_with_icr(dc_history, icr_history, lag_days=45)
    assert result.loc[pd.Timestamp("2020-06-01"), "WEC"] == 4.0
    assert result.loc[pd.Timestamp("2020-06-01"), "PEG"] == 0.3  # untouched — already had a value


def test_fill_with_icr_respects_reporting_lag():
    from grid_resilience.data.dc_load_data import fill_with_icr
    dc_history = pd.DataFrame({"WEC": [float("nan")]}, index=[pd.Timestamp("2020-06-01")])
    icr_history = pd.DataFrame({"WEC": [4.0]}, index=[pd.Timestamp("2020-05-01")])  # only 31 days back
    result = fill_with_icr(dc_history, icr_history, lag_days=45)
    assert pd.isna(result.loc[pd.Timestamp("2020-06-01"), "WEC"])


def test_fill_with_icr_handles_missing_icr_history():
    from grid_resilience.data.dc_load_data import fill_with_icr
    dc_history = pd.DataFrame({"WEC": [float("nan")]}, index=[pd.Timestamp("2020-06-01")])
    result = fill_with_icr(dc_history, None, lag_days=45)
    assert pd.isna(result.loc[pd.Timestamp("2020-06-01"), "WEC"])


def test_fill_with_icr_ticker_absent_from_icr_stays_nan():
    from grid_resilience.data.dc_load_data import fill_with_icr
    dc_history = pd.DataFrame({"EVRG": [float("nan")]}, index=[pd.Timestamp("2020-06-01")])
    icr_history = pd.DataFrame({"WEC": [4.0]}, index=[pd.Timestamp("2020-03-01")])
    result = fill_with_icr(dc_history, icr_history, lag_days=45)
    assert pd.isna(result.loc[pd.Timestamp("2020-06-01"), "EVRG"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v -k fill_with_icr`
Expected: FAIL — function doesn't exist

- [ ] **Step 3: Implement `fill_with_icr`**

Add to `grid_resilience/data/dc_load_data.py`:

```python
def fill_with_icr(
    dc_history: pd.DataFrame,
    icr_history: pd.DataFrame | None,
    lag_days: int = 45,
) -> pd.DataFrame:
    """
    Fill NaN cells in dc_history (non-PJM tickers, or PJM tickers with no
    zone-size data yet) using each ticker's own most-recent ICR reading as
    of (date - lag_days), instead of build_factor()'s flat cross-sectional
    mean fallback. Preserves real per-ticker differentiation for the 8 of 14
    regulated tickers the DC signal doesn't cover in v1 (see design spec).
    """
    result = dc_history.copy()
    if icr_history is None or icr_history.empty:
        return result

    icr_sorted = icr_history.sort_index()
    for date in result.index:
        cutoff = date - pd.Timedelta(days=lag_days)
        available = icr_sorted[icr_sorted.index <= cutoff]
        if available.empty:
            continue
        icr_row = available.iloc[-1]
        row = result.loc[date]
        for ticker in row[row.isna()].index:
            if ticker in icr_row.index and pd.notna(icr_row[ticker]):
                result.loc[date, ticker] = icr_row[ticker]
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v`
Expected: 28 passed

- [ ] **Step 5: Commit**

```bash
git add grid_resilience/data/dc_load_data.py tests/data/test_dc_load_data.py
git commit -m "feat: add fill_with_icr fallback for non-PJM regulated tickers"
```

---

### Task 7: Generalize the reporting lag in `resilience_score.py`

**Files:**
- Modify: `grid_resilience/factor/resilience_score.py`
- Test: `tests/factor/test_resilience_score.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `build_rolling_factor(..., regulated_signal_lag_days: int = _ICR_LAG_DAYS)` — all existing callers (including all 10 calls in `tests/factor/test_business_model_arch.py`) keep working unchanged since it's a new optional parameter with the old default.

- [ ] **Step 1: Write the failing test**

Add to `tests/factor/test_resilience_score.py` (check the file first with `grep -n "^def test\|^import\|^from" tests/factor/test_resilience_score.py` to match its existing fixture style before writing):

```python
def test_build_rolling_factor_respects_custom_regulated_signal_lag_days():
    import pandas as pd
    from grid_resilience.factor.resilience_score import build_rolling_factor

    dates = pd.MultiIndex.from_product(
        [[pd.Timestamp("2020-06-01")], ["PPL", "FE"]], names=["date", "ticker"]
    )
    rolling_betas = pd.DataFrame({"stress_beta": [-0.5, -0.5]}, index=dates)
    pt = pd.Series({"PPL": 0.05, "FE": 0.05})

    # icr_history dated 40 days before the rebalance date: visible with a 0-day
    # lag, but NOT visible with the default 45-day ICR lag.
    icr_history = pd.DataFrame({"PPL": [8.0], "FE": [1.0]}, index=[pd.Timestamp("2020-04-22")])

    with_zero_lag = build_rolling_factor(
        rolling_betas, icr_history=icr_history, arch="hard_switch",
        pass_through=pt, regulated_signal_lag_days=0,
    )
    with_default_lag = build_rolling_factor(
        rolling_betas, icr_history=icr_history, arch="hard_switch", pass_through=pt,
    )

    ppl_zero_lag = with_zero_lag[with_zero_lag["ticker"] == "PPL"]["factor_score"].iloc[0]
    fe_zero_lag  = with_zero_lag[with_zero_lag["ticker"] == "FE"]["factor_score"].iloc[0]
    assert ppl_zero_lag > fe_zero_lag  # ICR visible, PPL's higher ICR wins

    ppl_default = with_default_lag[with_default_lag["ticker"] == "PPL"]["factor_score"].iloc[0]
    fe_default  = with_default_lag[with_default_lag["ticker"] == "FE"]["factor_score"].iloc[0]
    assert ppl_default == fe_default  # ICR not yet visible under 45d lag -> both fall back to beta (tied)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/factor/test_resilience_score.py -v -k regulated_signal_lag_days`
Expected: FAIL with `TypeError: build_rolling_factor() got an unexpected keyword argument 'regulated_signal_lag_days'`

- [ ] **Step 3: Thread the parameter through**

In `grid_resilience/factor/resilience_score.py`, modify `build_rolling_factor`'s signature (currently at line ~175) and its call to `_icr_at_date`:

```python
def build_rolling_factor(
    rolling_betas: pd.DataFrame,
    renewable_share_by_period: pd.DataFrame | None = None,
    icr_history: pd.DataFrame | None = None,
    arch: str = "hard_switch",
    pass_through: pd.Series | None = None,
    regulated_signal_lag_days: int = _ICR_LAG_DAYS,
) -> pd.DataFrame:
```

Add one line to the docstring's Parameters section, after the `icr_history` entry:
```
    regulated_signal_lag_days  : Reporting lag applied before an icr_history row is
                                 considered observable. Default is the 45-day ICR
                                 filing lag; pass 0 for signals with no lag (e.g.
                                 the DC load signal, which is a live public dataset).
```

Change the call site inside the loop (currently `icr = _icr_at_date(icr_history, date)`) to:

```python
        icr = _icr_at_date(icr_history, date, lag_days=regulated_signal_lag_days)
```

And update `_icr_at_date`'s signature:

```python
def _icr_at_date(
    icr_history: pd.DataFrame | None,
    date: pd.Timestamp,
    lag_days: int = _ICR_LAG_DAYS,
) -> pd.Series | None:
    """Return the most recently available icr_history row as of `date` (with reporting lag)."""
    if icr_history is None or icr_history.empty:
        return None
    cutoff = date - pd.Timedelta(days=lag_days)
    available = icr_history[icr_history.index <= cutoff]
    if available.empty:
        return None
    return available.iloc[-1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/factor/test_resilience_score.py -v -k regulated_signal_lag_days`
Expected: PASS

- [ ] **Step 5: Run the full existing test suite to confirm nothing broke**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all passing (52 before this task's new test + 1 new test = 53; exact count depends on tests added in Tasks 1-6 too — should be strictly more than the 51 baseline, zero failures)

- [ ] **Step 6: Commit**

```bash
git add grid_resilience/factor/resilience_score.py tests/factor/test_resilience_score.py
git commit -m "feat: generalize ICR reporting lag into regulated_signal_lag_days param"
```

---

### Task 8: Wire `--regulated-signal` into `main.py`

**Files:**
- Modify: `grid_resilience/main.py`
- Test: manual verification (this task wires CLI/orchestration — the underlying logic is already unit-tested in Tasks 1-7)

**Interfaces:**
- Consumes: `dc_load_data.fetch_interconnection_queue`, `dc_load_data.compute_dc_load_signal`, `dc_load_data.fill_with_icr` (Tasks 2, 5, 6); `grid_data.fetch_zonal_load` (Task 1); `build_rolling_factor(..., regulated_signal_lag_days=...)` (Task 7)
- Produces: `run(..., regulated_signal: str = REGULATED_SIGNAL)`; CLI flag `--regulated-signal icr|dc-queue`

- [ ] **Step 1: Add the import and config constant**

In `grid_resilience/main.py`, modify the config import (currently line 89-93):

```python
from grid_resilience.config import (
    BACKTEST_START, BACKTEST_END,
    SUPPORTED_ISOS, REBALANCE_FREQ,
    PORTFOLIO_LONG_N, PORTFOLIO_SHORT_N,
    CONGESTION_SPREAD_ISOS, ISO_ZONE_LOCATION_TYPE,
    XLU_HEDGE, USE_ICR, BUSINESS_MODEL_ARCH, REGULATED_SIGNAL,
)
```

Add a new import after the existing `grid_data` import (currently lines 99-102):

```python
from grid_resilience.data.dc_load_data import (
    fetch_interconnection_queue, compute_dc_load_signal, fill_with_icr,
)
```

Modify the `grid_data` import line to add `fetch_zonal_load`:

```python
from grid_resilience.data.grid_data import (
    fetch_lmp, fetch_load, fetch_zonal_load, daily_lmp_summary, daily_load_summary,
    daily_spread_summary, fill_congestion_from_spread,
)
```

- [ ] **Step 2: Add the `regulated_signal` parameter to `run()`**

Modify `run()`'s signature (currently around line 121-131):

```python
def run(
    isos:              list[str] = SUPPORTED_ISOS,
    start:             str       = BACKTEST_START,
    end:               str       = BACKTEST_END,
    n_long:            int       = PORTFOLIO_LONG_N,
    n_short:           int       = PORTFOLIO_SHORT_N,
    xlu_hedge:         bool      = XLU_HEDGE,
    use_icr:           bool      = USE_ICR,
    arch:              str | None = BUSINESS_MODEL_ARCH,
    regulated_signal:  str       = REGULATED_SIGNAL,
    zone_gsi:          bool      = True,
    plot:              bool      = True,
    save_dir:          str       = "output",
) -> dict:
```

- [ ] **Step 3: Replace the ICR-fetch block with the branching fetch**

Find the current block (grid_resilience/main.py:303-317):

```python
    icr_history = None
    if use_icr:
        print("  Fetching interest coverage ratios…")
        icr_history = fetch_icr(universe_tickers)
        if not icr_history.empty:
            print(f"  ICR data: {len(icr_history)} quarters, {icr_history.notna().any().sum()} tickers")
        else:
            print("  [warn] ICR data unavailable — factor will use beta + renewables only")

    rolling_factors = build_rolling_factor(
        rolling_betas,
        icr_history=icr_history,
        arch=arch or "hard_switch",
        pass_through=pass_through,
    )
```

Replace it with:

```python
    icr_history = None
    regulated_signal_lag_days = 45
    if use_icr:
        print(f"  Fetching regulated-path signal ({regulated_signal})…")
        icr_history = fetch_icr(universe_tickers)
        if not icr_history.empty:
            print(f"  ICR data: {len(icr_history)} quarters, {icr_history.notna().any().sum()} tickers")
        else:
            print("  [warn] ICR data unavailable — factor will use beta + renewables only")

        if regulated_signal == "dc_queue":
            queue = fetch_interconnection_queue()
            zonal_load = fetch_zonal_load(start, end)
            dc_history = compute_dc_load_signal(
                tickers=universe_tickers,
                node_map=TICKER_NODE_MAP,
                queue=queue,
                zonal_load=zonal_load,
                as_of_dates=list(rebalance_dates),
            )
            icr_history = fill_with_icr(dc_history, icr_history, lag_days=45)
            regulated_signal_lag_days = 0
            print(f"  DC load signal: {dc_history.notna().any().sum()} PJM tickers with real data, "
                  f"{(icr_history.notna().any() & dc_history.isna().all()).sum()} filled from ICR")

    rolling_factors = build_rolling_factor(
        rolling_betas,
        icr_history=icr_history,
        arch=arch or "hard_switch",
        pass_through=pass_through,
        regulated_signal_lag_days=regulated_signal_lag_days,
    )
```

- [ ] **Step 4: Add the CLI flag**

In `_parse_args()`, immediately after the existing `--arch` argument (grid_resilience/main.py:400-406):

```python
    p.add_argument(
        "--regulated-signal",
        dest="regulated_signal",
        default=REGULATED_SIGNAL,
        choices=["icr", "dc-queue"],
        help="Signal used for the regulated path of --arch hard-switch "
             f"(default: {REGULATED_SIGNAL.replace('_', '-')})",
    )
```

And thread it through the `run()` call at the bottom of the file (currently lines 415-424):

```python
    run(
        isos             = args.iso,
        start            = args.start,
        end              = args.end,
        n_long           = args.long,
        n_short          = args.short,
        xlu_hedge        = args.xlu_hedge,
        use_icr          = args.use_icr,
        arch             = args.arch.replace("-", "_") if args.arch else None,
        regulated_signal = args.regulated_signal.replace("-", "_"),
        zone_gsi         = args.zone_gsi,
        plot             = not args.no_plot,
        save_dir         = args.output,
    )
```

- [ ] **Step 5: Verify `TICKER_NODE_MAP` is already imported**

Run: `grep -n "from grid_resilience.data.utility_node_map import" grid_resilience/main.py`
Expected: already present (it's used for `pass_through` construction) — no new import needed.

- [ ] **Step 6: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all passing, zero failures — this task only adds new code paths gated behind `--regulated-signal dc-queue`, doesn't touch existing default behavior (`REGULATED_SIGNAL = "dc_queue"` in config.py means this IS now the default — see Step 7 for why that's still safe).

- [ ] **Step 7: Manually verify both signal paths run end-to-end**

Run:
```bash
source ~/.zshrc
.venv/bin/python -m grid_resilience.main --no-plot --no-zone-gsi --arch hard-switch --regulated-signal icr
.venv/bin/python -m grid_resilience.main --no-plot --no-zone-gsi --arch hard-switch --regulated-signal dc-queue
```
Both must complete without errors and print an IC table. Compare the `dc-queue` run's printed IC@21d/IC@63d against the `icr` baseline (IC@21d 0.1434, IC@63d 0.2027) and against the design spec's success measure. Also manually inspect `output/factor_scores.csv` for the `dc-queue` run to check whether D and PEG score as long candidates (the qualitative motivation for this whole spec) rather than being dragged down by leverage-based ICR scoring.

- [ ] **Step 8: Commit**

```bash
git add grid_resilience/main.py
git commit -m "feat: add --regulated-signal CLI flag, wire DC load signal into main pipeline"
```

---

### Task 9: Record results

**Files:**
- Create: `docs/compact_2026-07-06-dc-load-results.md` (or append to the existing compact handoff doc — check `docs/` for the most recent compact doc filename first and follow its pattern)

**Interfaces:** none — this is a documentation task.

- [ ] **Step 1: Run the comparison and fill in the Evaluation table**

Using the results from Task 8 Step 7, update the "Evaluation" table in `docs/superpowers/specs/2026-07-06-dc-load-signal-design.md` (replace the `TBD` row) with actual Sharpe / Ann Ret / Max DD / IC@21d / IC@63d for the `dc-queue` run.

- [ ] **Step 2: Write the session summary**

Follow the existing compact-doc pattern (`docs/compact_2026-07-05.md`'s structure: Performance Scoreboard, What Was Done, Files Changed, Branch State, Open Questions). Explicitly call out:
- Whether DC load beat ICR on IC, and whether D/PEG flipped to long candidates
- The accepted look-ahead limitation on `MW Capacity` (Task 2) and whether it appears to matter in practice (e.g. does removing the earliest 12-18 months of the backtest change the conclusion?)
- Whether the v1 60/40 weights and 100MW/90-day thresholds should go through a grid search before being treated as final (per the design spec's explicit caveat)

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-07-06-dc-load-signal-design.md docs/compact_*.md
git commit -m "docs: record DC load signal backtest results"
```

