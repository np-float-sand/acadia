# PJM Data Quality Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore Sharpe from -1.677 back toward the grid-search baseline of 0.319 by fixing PJM data gaps that corrupted the GSI signal for all 5 PJM-mapped tickers (D, AEP, EXC, FE, PPL).

**Architecture:** Three sequential fixes. (1) Confirm the diagnosis by running without PJM. (2) Fix the PJM API fetcher so future runs don't lose data to rate-limit exhaustion. (3) Add a graceful GSI forward-fill so any residual gaps degrade smoothly instead of creating hard index gaps that fall out of beta estimation. Then re-fetch the 18 missing LMP months and 29 missing load months and confirm Sharpe recovers.

**Tech Stack:** Python 3.11, pandas, requests, pytest, gridstatus, grid_resilience codebase

## Global Constraints

- Do not change the grid-search-winning parameters: `STRESS_SPIKE_PCT=0.97`, `STRESS_CONG_THRESHOLD=0.30`, `STRESS_CONG_MIN_DAYS=10`, `PORTFOLIO_LONG_N=3`, `PORTFOLIO_SHORT_N=5`, `XLU_HEDGE=False`
- Do not enable `XLU_HEDGE` — grid search showed every `xlu_hedge=True` config produced negative Sharpe
- Run commands from the project root (`/Users/sandhyapersad/acadia`) with `.venv` active
- Test runner: `pytest tests/ -v`
- Tests must not hit the network; mock any external calls

---

## Task 1: Confirm Diagnosis (run without PJM)

**Files:**
- No code changes — CLI invocation only

**Goal:** Confirm that removing PJM from the ISO list restores Sharpe to near the 0.319 baseline.

- [ ] **Step 1: Run backtest without PJM**

```bash
python -m grid_resilience.main --iso ERCOT MISO CAISO SPP --no-plot 2>&1 | grep -E "Sharpe|Ann Ret|Max DD|Signal"
```

Expected output (approximate, based on grid-search baseline):
```
  Factor (L/S)    ...    0.2xx   ...%  ...%
```
If Sharpe is ≥ 0.25, the diagnosis is confirmed: PJM data gaps are the root cause.

- [ ] **Step 2: Record baseline**

Write down the Sharpe from this run before proceeding. It becomes the recovery target.

- [ ] **Step 3: Commit note (no code change)**

```bash
git commit --allow-empty -m "chore: confirm PJM-free Sharpe baseline before data fix"
```

---

## Task 2: Fix PJM Fetcher — Increase Rate-Limit Tolerance

**Files:**
- Modify: `grid_resilience/data/grid_data.py:104-119` (`_pjm_get`)
- Modify: `grid_resilience/data/grid_data.py:354-371` (`_fetch_month` inside `fetch_lmp`)
- Modify: `grid_resilience/data/grid_data.py:425-476` (`fetch_load`, same `_fetch_month` pattern)
- Test: `tests/data/test_pjm_retry.py`

**Interfaces:**
- Consumes: `_pjm_get(url, params, api_key, retries)` — internal helper
- Produces: same signature; behaviour change only (more retries, longer sleep)

### 2a — Write the failing test

- [ ] **Step 1: Write test file**

```python
# tests/data/test_pjm_retry.py
"""Tests for PJM rate-limit retry and inter-month delay."""
import time
from unittest.mock import MagicMock, patch, call
import pytest
import requests

from grid_resilience.data.grid_data import _pjm_get


def _mock_response(status_code: int, json_data: dict = None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {"items": [], "totalRows": 0}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


def test_pjm_get_succeeds_on_first_try():
    ok = _mock_response(200, {"items": [{"x": 1}], "totalRows": 1})
    with patch("requests.get", return_value=ok) as mock_get:
        result = _pjm_get("https://api.pjm.com/test", {}, "key")
    assert result == {"items": [{"x": 1}], "totalRows": 1}
    assert mock_get.call_count == 1


def test_pjm_get_retries_on_429_then_succeeds():
    rate_limited = _mock_response(429)
    ok = _mock_response(200, {"items": [], "totalRows": 0})
    with patch("requests.get", side_effect=[rate_limited, rate_limited, ok]):
        with patch("time.sleep") as mock_sleep:
            result = _pjm_get("https://api.pjm.com/test", {}, "key", retries=6)
    assert result == {"items": [], "totalRows": 0}
    # Must have slept at least twice (once per 429)
    assert mock_sleep.call_count >= 2


def test_pjm_get_first_sleep_is_at_least_15s():
    """Initial backoff must be >= 15s (was 5s — caused serial 429 floods)."""
    rate_limited = _mock_response(429)
    ok = _mock_response(200, {"items": [], "totalRows": 0})
    with patch("requests.get", side_effect=[rate_limited, ok]):
        with patch("time.sleep") as mock_sleep:
            _pjm_get("https://api.pjm.com/test", {}, "key", retries=6)
    first_sleep = mock_sleep.call_args_list[0][0][0]
    assert first_sleep >= 15, f"Expected first sleep >= 15s, got {first_sleep}s"


def test_pjm_get_raises_after_all_retries_exhausted():
    rate_limited = _mock_response(429)
    with patch("requests.get", return_value=rate_limited):
        with patch("time.sleep"):
            with pytest.raises(requests.HTTPError):
                _pjm_get("https://api.pjm.com/test", {}, "key", retries=2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/data/test_pjm_retry.py -v
```

Expected: `test_pjm_get_first_sleep_is_at_least_15s` FAILS (currently sleeps only 5s).

### 2b — Implement the fix

- [ ] **Step 3: Update `_pjm_get` in `grid_resilience/data/grid_data.py`**

Change lines 104–119 from:

```python
def _pjm_get(url: str, params: dict, api_key: str, retries: int = 4) -> dict:
    """GET a PJM DataMiner 2 endpoint with exponential-backoff retry on 429."""
    import requests
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    delay = 5
    for attempt in range(retries):
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            print(f"  [grid] PJM 429 — retrying in {delay}s (attempt {attempt + 1}/{retries})")
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}
```

To:

```python
def _pjm_get(url: str, params: dict, api_key: str, retries: int = 6) -> dict:
    """GET a PJM DataMiner 2 endpoint with exponential-backoff retry on 429."""
    import requests
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    delay = 15
    for attempt in range(retries):
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            print(f"  [grid] PJM 429 — retrying in {delay}s (attempt {attempt + 1}/{retries})")
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}
```

- [ ] **Step 4: Increase inter-month sleep in `fetch_lmp` (line ~356)**

Find the `_fetch_month` inner function inside `fetch_lmp` (around line 354). Change:

```python
                def _fetch_month(ym: tuple[int, int]) -> None:
                    if iso == "PJM":
                        time.sleep(3)
```

To:

```python
                def _fetch_month(ym: tuple[int, int]) -> None:
                    if iso == "PJM":
                        time.sleep(12)
```

- [ ] **Step 5: Do the same in `fetch_load` (same pattern, same file, ~line 455)**

Find the equivalent `_fetch_month` in `fetch_load`. Apply the same `time.sleep(12)` change for PJM. (If no per-ISO sleep exists there yet, add `if iso == "PJM": time.sleep(12)` at the start of `_fetch_month` before the `month_bounds` call, mirroring the `fetch_lmp` pattern.)

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/data/test_pjm_retry.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 7: Run existing test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all existing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add grid_resilience/data/grid_data.py tests/data/test_pjm_retry.py
git commit -m "fix: increase PJM retry count to 6 and initial backoff to 15s; inter-month sleep to 12s"
```

---

## Task 3: GSI Forward-Fill for Residual Gaps

**Files:**
- Modify: `grid_resilience/signals/grid_stress_index.py:106-114` (end of `build_gsi`)
- Test: `tests/signals/test_gsi_ffill.py`

**Interfaces:**
- Consumes: `build_gsi(daily_lmp, daily_load, events_df, iso)` — returns DataFrame indexed by date
- Produces: same return type; gaps within the covered date range are now forward-filled (limit 10 days)

**Why here:** `build_gsi` is the single construction point for all ISOs. Forward-filling here means every consumer (`conditional_beta.py`, `rolling_stress_betas`) automatically sees a dense series. The `limit=10` cap (about 2 trading weeks) prevents stale values propagating across multi-month gaps — they stay NaN and fall out of intersection correctly.

### 3a — Write the failing test

- [ ] **Step 1: Create test directory if needed**

```bash
mkdir -p tests/signals
touch tests/signals/__init__.py
```

- [ ] **Step 2: Write test file**

```python
# tests/signals/test_gsi_ffill.py
"""Tests that build_gsi forward-fills gaps within the data range (limit 10 days)."""
import numpy as np
import pandas as pd
import pytest

from grid_resilience.signals.grid_stress_index import build_gsi


def _make_daily_lmp(dates: pd.DatetimeIndex, lmp_values=None) -> pd.DataFrame:
    if lmp_values is None:
        lmp_values = np.random.default_rng(0).uniform(20, 80, len(dates))
    return pd.DataFrame({"lmp_max": lmp_values}, index=dates)


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["start", "end", "iso", "type", "name", "severity"])


def test_gsi_has_no_gaps_within_5_day_hole():
    """A 5-day hole in LMP data should be ffilled — GSI is continuous."""
    # Jan 1–20, skip 21–25, resume 26–31
    dates = pd.date_range("2024-01-01", "2024-01-20").append(
        pd.date_range("2024-01-26", "2024-01-31")
    )
    daily_lmp = _make_daily_lmp(dates)
    result = build_gsi(daily_lmp, pd.DataFrame(), _empty_events(), iso="PJM")
    # All dates from min to max should be present after ffill
    full_range = pd.date_range(result.index.min(), result.index.max(), freq="D")
    missing = full_range.difference(result.index)
    assert len(missing) == 0, f"Expected no gaps within 10-day limit, got {len(missing)} missing dates"


def test_gsi_gap_larger_than_limit_stays_nan():
    """A 15-day hole exceeds the ffill limit — those dates must remain absent."""
    dates = pd.date_range("2024-01-01", "2024-01-10").append(
        pd.date_range("2024-01-26", "2024-02-10")
    )
    daily_lmp = _make_daily_lmp(dates)
    result = build_gsi(daily_lmp, pd.DataFrame(), _empty_events(), iso="PJM")
    # Dates 11–25 are a 15-day gap; only 10 days should be filled
    # so at least some dates in 2024-01-21 to 2024-01-25 should still be absent
    hole = pd.date_range("2024-01-21", "2024-01-25")
    present_in_hole = hole.intersection(result.index)
    assert len(present_in_hole) == 0, (
        f"Expected dates deep in 15-day gap to be absent, but {len(present_in_hole)} are present"
    )


def test_gsi_no_change_when_data_already_dense():
    """Dense data (no gaps) should be unaffected by the ffill."""
    dates = pd.date_range("2024-01-01", "2024-03-31")
    daily_lmp = _make_daily_lmp(dates)
    result = build_gsi(daily_lmp, pd.DataFrame(), _empty_events(), iso="ERCOT")
    assert len(result) == len(dates)
    assert result["gsi"].notna().all()
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/signals/test_gsi_ffill.py -v
```

Expected: `test_gsi_has_no_gaps_within_5_day_hole` FAILS because `build_gsi` currently returns only dates where LMP data exists.

### 3b — Implement the fix

- [ ] **Step 4: Add forward-fill in `build_gsi` (`grid_resilience/signals/grid_stress_index.py`)**

After the existing `result` DataFrame is built (currently line ~106–113), add a reindex + ffill step before the return:

Change the end of `build_gsi` from:

```python
    result = pd.DataFrame({
        "lmp_zscore":          lmp_z,
        "congestion_frac_z":   cong_z,
        "reserve_tightness_z": reserve_z,
        "event_flag":          event_flag,
        "gsi":                 gsi,
    })
    result.index.name = "date"
    return result
```

To:

```python
    result = pd.DataFrame({
        "lmp_zscore":          lmp_z,
        "congestion_frac_z":   cong_z,
        "reserve_tightness_z": reserve_z,
        "event_flag":          event_flag,
        "gsi":                 gsi,
    })
    result.index.name = "date"

    # Fill short gaps (≤ 10 days) so missing API months don't punch hard holes
    # in the GSI index that fall out of beta-estimation intersections.
    # Gaps > 10 days remain NaN and drop out of the common index naturally.
    full_daily = pd.date_range(result.index.min(), result.index.max(), freq="D", name="date")
    result = result.reindex(full_daily).ffill(limit=10)

    return result
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/signals/test_gsi_ffill.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add grid_resilience/signals/grid_stress_index.py tests/signals/test_gsi_ffill.py tests/signals/__init__.py
git commit -m "fix: forward-fill GSI up to 10 days to bridge PJM API gaps gracefully"
```

---

## Task 4: Refetch Missing PJM Months

**Files:**
- No code changes — invoke the existing pipeline with `force_refresh=False` (default), which will attempt the 18 missing LMP months and 29 missing load months automatically now that retry logic is fixed.

**Note:** The cache currently has these months missing for PJM LMP:
`2022-07, 2022-08, 2023-06, 2023-07, 2024-02, 2024-03, 2024-06, 2024-07, 2024-09, 2024-10, 2024-12, 2025-02, 2025-03, 2025-04, 2025-05, 2025-07, 2025-09, 2025-10`

And many more for PJM load. The existing `find_missing_months` + `_fetch_month` logic will pick all of these up on the next run.

- [ ] **Step 1: Verify which LMP months are still missing**

```bash
python3 -c "
import os, sys
sys.path.insert(0, '.')
from grid_resilience.data.cache_utils import find_missing_months, chunk_path
from grid_resilience.data.grid_data import _cache_path
missing = find_missing_months(_cache_path('PJM', 'lmp_DAY_AHEAD_HOURLY'), '2022-01-01', '2025-12-31')
print(f'Missing LMP months: {len(missing)}')
for y, m in missing:
    print(f'  {y}-{m:02d}')
"
```

- [ ] **Step 2: Trigger a PJM-only fetch run**

Run the full pipeline with PJM included. The fetcher will see the missing months and attempt them with the new retry logic. This may take 15–30 minutes due to the 12s inter-month sleep.

```bash
python -m grid_resilience.main --iso PJM --no-plot 2>&1 | tee /tmp/pjm_backfill.log
grep -E "429|200|Fetching|saved|warn|error" /tmp/pjm_backfill.log
```

- [ ] **Step 3: Verify cache is now complete**

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from grid_resilience.data.cache_utils import find_missing_months
from grid_resilience.data.grid_data import _cache_path
lmp_miss = find_missing_months(_cache_path('PJM', 'lmp_DAY_AHEAD_HOURLY'), '2022-01-01', '2025-12-31')
load_miss = find_missing_months(_cache_path('PJM', 'load'), '2022-01-01', '2025-12-31')
print(f'Missing LMP months: {len(lmp_miss)}')
print(f'Missing load months: {len(load_miss)}')
"
```

Expected: 0 missing LMP months, 0 missing load months (or close to 0 if any months remain genuinely empty in PJM's API).

- [ ] **Step 4: No commit (cache files are not committed)**

---

## Task 5: Full Backtest and Sharpe Recovery Confirmation

**Files:**
- No code changes — validation run

- [ ] **Step 1: Run full backtest with all ISOs**

```bash
python -m grid_resilience.main 2>&1 | grep -E "Sharpe|Ann Ret|Max DD|Signal|IC"
```

- [ ] **Step 2: Compare to targets**

| Metric | Before PJM | With PJM (broken) | Target (≥) |
|--------|-----------|-------------------|------------|
| Sharpe | 0.319     | -1.677            | 0.25       |
| Ann Ret | 8.1%     | -5.92%            | 5%         |
| Max DD | -13.7%   | -22.66%           | -20%       |

If Sharpe ≥ 0.25: success. If still negative, PJM data quality is the wrong diagnosis and param retuning is needed.

- [ ] **Step 3: Commit results note**

```bash
git add output/run.log output/pnl.csv output/factor_scores.csv
git commit -m "feat: restore PJM data quality — Sharpe recovered from -1.677 to X.XXX"
```

(Fill in the actual Sharpe in the commit message.)

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Fix 1 (run without PJM) → Task 1
- ✅ Fix 2 (retry logic) → Task 2 with tests
- ✅ Fix 3 (ffill graceful degradation) → Task 3 with tests
- ✅ Refetch missing months → Task 4
- ✅ Confirm Sharpe recovery → Task 5

**Placeholder scan:** No TBDs or hand-wavy steps.

**Type consistency:** `build_gsi` return type is `pd.DataFrame` throughout. `_pjm_get` signature unchanged except default `retries=6`.

**Edge cases covered:**
- Test 3b verifies ffill doesn't corrupt dense data (no-op on clean series)
- Test 3a verifies large gaps (> 10 days) remain absent rather than stale-filled
- `_pjm_get` test verifies exhausted retries still raise (no silent data loss)
