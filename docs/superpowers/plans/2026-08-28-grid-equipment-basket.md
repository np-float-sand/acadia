# Grid Equipment Suppliers Basket — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a long-only thematic basket of US-listed grid / data-center electrical-infrastructure suppliers, backtest the equal-weight version over the 2023-present AI-buildout regime against XLI + GRID/PAVE, and — only if it clears a stated CAGR+Sharpe bar — add a disclosed-order-backlog-growth weight tilt.

**Architecture:** New top-level package `grid_equipment_basket/`, sibling to `grid_resilience/`. Prices come from yfinance through the same monthly-parquet wide cache `grid_resilience` uses (public helpers imported from `grid_resilience.data.cache_utils`; no changes to `grid_resilience/`). `basket.py` simulates a hold-shares / periodic-rebalance portfolio and emits a daily simple-return series plus a rebalance-date weight frame. `backtest.py` computes annualized metrics (rf 4%, 252-day) and benchmark-relative stats, and orchestrates a full run. Step 2 (backlog data + tilt) is gated behind Step 1's decision gate.

**Tech Stack:** Python 3.11+, pandas, numpy, yfinance, matplotlib, requests (all already in `pyproject.toml` — no dependency changes). SEC EDGAR / XBRL `companyconcept` REST API for Step 2 backlog data (free, no key).

**Spec:** `docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md`

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the spec.

- **Long-only, no short leg, no leverage, fully invested.**
- **No modifications to `grid_resilience/`.** Importing its *public* `cache_utils` helpers (`find_missing_months_wide`, `read_monthly_cache_wide`, `save_monthly_chunks_wide`, `month_bounds`, `chunk_path`) is allowed and expected; do not import underscore-prefixed names.
- **No cross-sectional z-scoring, no `build_factor()` / `hard_switch` integration, no IC@21d or factor-style predictive metric, no claim of statistical significance, no 8-year backtest as primary validation.** (Spec §12.)
- **Simple-return convention throughout this package** (`pct_change`, `(1+r).prod()`), which differs from `grid_resilience`'s log-return convention. State this in the README so a log-based recompute that doesn't tie out exactly is not mistaken for a bug.
- **Risk-free rate 4% annual, annualization factor 252** (matches `grid_resilience/portfolio/backtest.py`).
- **Primary window:** `2023-01-01` → the last complete calendar month at implementation time. **Prior-regime panel:** `2020-01-01` .. `2022-12-31`, reported alongside, never instead.
- **Single-name cap 25%.** **Quarterly rebalance ~6 weeks (42 days) after each calendar quarter-end.**
- **Backlog tilt (Step 2):** fixed `1.25x` top-half-ranked / `0.75x` bottom-half-ranked; median name untilted (`1.0x`) on an odd universe count; NaN-signal names untilted.
- **Universe inclusion is decided on business description from the latest 10-K only — never on historical returns.** Names that meet the business test but have been poor performers (FLNC) are deliberately kept in. Foreign-listed names (ABB, Siemens Energy, Prysmian, Nexans) are logged in `candidate_research.md` and excluded from the backtestable basket.
- **All markdown files are written with `cat > file.md << 'EOF' ... EOF`, not an editor tool** (repo CLAUDE.md).
- **After any code change, check whether it invalidates the repo-root `README.md`** and update it in the same task if so (repo CLAUDE.md).
- **Decision gate (Spec §5), evaluated in Task 7 — all three must hold over the primary window:** (1) basket CAGR > XLI CAGR; (2) basket Sharpe > XLI Sharpe; (3) basket CAGR within 2 percentage points of the higher of GRID / PAVE CAGR, or above it.
- **Tasks 8, 9, 10 execute ONLY if Task 7 records the gate as PASS.** If FAIL, the negative-result writeup is the final deliverable and no backlog data is collected.

---

## File Structure

| File | Responsibility |
|---|---|
| `grid_equipment_basket/__init__.py` | empty package marker |
| `grid_equipment_basket/config.py` | universe list, benchmark tickers, windows, rf rate, ann factor, rebalance lag, single-name cap, Step 2 tilt magnitudes, `CACHE_DIR` |
| `grid_equipment_basket/data/__init__.py` | empty package marker |
| `grid_equipment_basket/data/prices.py` | yfinance adjusted-close fetch + monthly-parquet wide cache + retry-on-dropped-column |
| `grid_equipment_basket/data/cache/` | `*.parquet` chunks — already covered by the repo `.gitignore` `*.parquet` rule |
| `grid_equipment_basket/basket.py` | `apply_cap`, `equal_weight_targets`, `rebalance_dates`, `simulate_basket` → `BasketResult`; (Step 2) `backlog_tilt_targets` |
| `grid_equipment_basket/backtest.py` | `compute_metrics`, `calendar_year_returns`, `relative_metrics`, `run`, `results_table`, `plot` |
| `grid_equipment_basket/backlog_data.py` | (Step 2) SEC XBRL RPO fetch, `backlog_quarterly.csv` loader, growth signal, ranks |
| `grid_equipment_basket/backlog_forward_check.py` | (Step 2) descriptive backlog-growth vs forward-return check |
| `grid_equipment_basket/data/backlog_quarterly.csv` | (Step 2) collected backlog / book-to-bill time series |
| `grid_equipment_basket/__main__.py` | CLI: `python -m grid_equipment_basket [--start ...] [--end ...] [--prior-regime] [--tilt none|backlog] [--output ...] [--no-plot]` |
| `grid_equipment_basket/candidate_research.md` | per-candidate verification log |
| `grid_equipment_basket/README.md` | composition, weights, window, benchmark rationale, Step 1 result, gate outcome, Phase-2 note, return-convention note |
| `docs/grid-equipment-basket-step1-results.md` | Step 1 (then Step 1+2) results writeup |
| `tests/grid_equipment_basket/__init__.py` | empty |
| `tests/grid_equipment_basket/test_prices.py` | cache hit/miss/gap-fill (mocked), retry wrapper |
| `tests/grid_equipment_basket/test_basket.py` | cap, equal weight, rebalance dates, simulation drift, pre-listing exclusion |
| `tests/grid_equipment_basket/test_backtest.py` | metrics closed-form, calendar-year split, relative metrics, `run` smoke |
| `tests/grid_equipment_basket/test_backlog_data.py` | (Step 2) CSV load/validate, availability-date gate, YoY signal, ranks |
| `tests/grid_equipment_basket/test_backlog_tilt.py` | (Step 2) tilt overweight/underweight, odd-count median, NaN handling, cap |

---

## Task 1: Package scaffold, config, price fetcher

**Files:**
- Create: `grid_equipment_basket/__init__.py` (empty)
- Create: `grid_equipment_basket/config.py`
- Create: `grid_equipment_basket/data/__init__.py` (empty)
- Create: `grid_equipment_basket/data/prices.py`
- Create: `tests/grid_equipment_basket/__init__.py` (empty)
- Test: `tests/grid_equipment_basket/test_prices.py`

**Interfaces:**
- Consumes: `grid_resilience.data.cache_utils` public helpers.
- Produces:
  - `grid_equipment_basket.config`: `CACHE_DIR: Path`, `UNIVERSE: list[str]`, `BENCHMARKS: list[str]`, `PRIMARY_BENCHMARK: str = "XLI"`, `HONESTY_BENCHMARKS: list[str] = ["GRID", "PAVE"]`, `PRIMARY_START: str = "2023-01-01"`, `PRIOR_REGIME_START: str = "2020-01-01"`, `PRIOR_REGIME_END: str = "2022-12-31"`, `RISK_FREE_RATE: float = 0.04`, `ANN_FACTOR: int = 252`, `REBALANCE_LAG_DAYS: int = 42`, `MAX_SINGLE_NAME_WEIGHT: float = 0.25`, `BACKLOG_TILT_TOP: float = 1.25`, `BACKLOG_TILT_BOTTOM: float = 0.75`
  - `grid_equipment_basket.data.prices.fetch_prices(tickers: list[str], start: str, end: str, use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame` — DatetimeIndex ascending, columns = the subset of `sorted(set(tickers))` that had data, values = adjusted close (float), intraday gaps ffilled up to 5 days.
  - `grid_equipment_basket.data.prices._download_with_retry(tickers, start, end, max_attempts=2) -> pd.DataFrame` (module-level, monkeypatched in tests).
  - `grid_equipment_basket.data.prices._PRICE_CACHE_BASE: Path` (module global, monkeypatched in tests).

- [ ] **Step 1: Create the three empty markers and `config.py`**

```bash
mkdir -p grid_equipment_basket/data tests/grid_equipment_basket
touch grid_equipment_basket/__init__.py grid_equipment_basket/data/__init__.py tests/grid_equipment_basket/__init__.py
```

```bash
cat > grid_equipment_basket/config.py << 'EOF'
"""Central configuration for the grid-equipment-suppliers thematic basket.

Spec: docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Provisional list — Task 6 (candidate_research.md) replaces this with the
# verified include-set. Inclusion is decided on 10-K business description only.
UNIVERSE: list[str] = ["ETN", "HUBB", "GEV", "VRT", "PWR", "MYRG", "NVT", "FLNC", "PRIM"]

BENCHMARKS: list[str] = ["XLI", "SPY", "XLU", "GRID", "PAVE"]
PRIMARY_BENCHMARK: str = "XLI"
HONESTY_BENCHMARKS: list[str] = ["GRID", "PAVE"]

PRIMARY_START: str = "2023-01-01"
PRIOR_REGIME_START: str = "2020-01-01"
PRIOR_REGIME_END: str = "2022-12-31"

RISK_FREE_RATE: float = 0.04
ANN_FACTOR: int = 252

REBALANCE_LAG_DAYS: int = 42          # ~6 weeks after each calendar quarter-end
MAX_SINGLE_NAME_WEIGHT: float = 0.25

# Step 2 (gated) — fixed, documented, not fitted.
BACKLOG_TILT_TOP: float = 1.25
BACKLOG_TILT_BOTTOM: float = 0.75
EOF
```

- [ ] **Step 2: Write the failing tests**

```bash
cat > tests/grid_equipment_basket/test_prices.py << 'EOF'
import pandas as pd
import pytest

from grid_equipment_basket.data import prices as px


def _fake_frame(tickers, start, end):
    idx = pd.bdate_range(start, end)
    return pd.DataFrame(
        {t: [float(i + 1) for i in range(len(idx))] for t in tickers},
        index=idx,
    )


def test_fetch_prices_returns_sorted_requested_tickers(monkeypatch, tmp_path):
    monkeypatch.setattr(px, "_PRICE_CACHE_BASE", tmp_path / "prices.parquet")
    monkeypatch.setattr(
        px, "_download_with_retry",
        lambda t, s, e, **k: _fake_frame(t, s, e),
    )
    out = px.fetch_prices(["VRT", "PWR"], "2023-01-02", "2023-03-31")
    assert list(out.columns) == ["PWR", "VRT"]
    assert out.index.is_monotonic_increasing
    assert out.notna().all().all()


def test_fetch_prices_uses_cache_on_second_call(monkeypatch, tmp_path):
    monkeypatch.setattr(px, "_PRICE_CACHE_BASE", tmp_path / "prices.parquet")
    calls = []

    def _spy(t, s, e, **k):
        calls.append((tuple(sorted(t)), s, e))
        return _fake_frame(t, s, e)

    monkeypatch.setattr(px, "_download_with_retry", _spy)
    px.fetch_prices(["VRT"], "2023-01-02", "2023-02-28")
    n_first = len(calls)
    assert n_first > 0
    px.fetch_prices(["VRT"], "2023-01-02", "2023-02-28")
    assert len(calls) == n_first          # nothing re-downloaded


def test_fetch_prices_gap_fills_only_new_months(monkeypatch, tmp_path):
    monkeypatch.setattr(px, "_PRICE_CACHE_BASE", tmp_path / "prices.parquet")
    months = []

    def _spy(t, s, e, **k):
        months.append(s[:7])
        return _fake_frame(t, s, e)

    monkeypatch.setattr(px, "_download_with_retry", _spy)
    px.fetch_prices(["VRT"], "2023-01-02", "2023-02-28")
    months.clear()
    px.fetch_prices(["VRT"], "2023-01-02", "2023-04-28")
    assert set(months) == {"2023-03", "2023-04"}


def test_download_with_retry_retries_on_missing_column(monkeypatch):
    monkeypatch.setattr(px, "_RETRY_DELAY_SECONDS", 0)
    frames = [
        pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2023-01-02"])),
        pd.DataFrame({"A": [1.0], "B": [2.0]}, index=pd.to_datetime(["2023-01-02"])),
    ]
    seen = []

    def _fake_download(t, s, e):
        seen.append(1)
        return frames[len(seen) - 1]

    monkeypatch.setattr(px, "_download", _fake_download)
    out = px._download_with_retry(["A", "B"], "2023-01-02", "2023-01-02", max_attempts=2)
    assert list(out.columns) == ["A", "B"]
    assert len(seen) == 2
EOF
```

- [ ] **Step 3: Run the tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_prices.py -q`
Expected: FAIL / ERROR — `grid_equipment_basket.data.prices` does not exist.

- [ ] **Step 4: Write `data/prices.py`**

```bash
cat > grid_equipment_basket/data/prices.py << 'EOF'
from __future__ import annotations

"""Adjusted-close price fetcher for the grid-equipment basket.

Thin wrapper over yfinance reusing grid_resilience's monthly-parquet WIDE cache
helpers (public functions only). Returns adjusted-close prices; every downstream
maths in this package works in SIMPLE returns, not log returns — see README.
"""

import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

from grid_equipment_basket.config import CACHE_DIR
from grid_resilience.data.cache_utils import (
    chunk_path,
    find_missing_months_wide,
    month_bounds,
    read_monthly_cache_wide,
    save_monthly_chunks_wide,
)

warnings.filterwarnings("ignore", category=FutureWarning)

_PRICE_CACHE_BASE = CACHE_DIR / "prices.parquet"
_RETRY_DELAY_SECONDS = 2


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    # yfinance `end` is exclusive — advance one day so `end` is included.
    end_exc = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        tickers, start=start, end=end_exc, auto_adjust=True,
        group_by="column", progress=False, threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw.xs("Close", axis=1, level=0)
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
    return prices.ffill(limit=5).dropna(how="all")


def _download_with_retry(
    tickers: list[str], start: str, end: str, max_attempts: int = 2
) -> pd.DataFrame:
    """Retry once when yfinance silently drops a ticker column under load.

    A column can also be legitimately absent (ticker not listed in the window),
    which looks identical here, so retries are capped, not looped.
    """
    frame = _download(tickers, start, end)
    for _ in range(max_attempts - 1):
        if all(t in frame.columns for t in tickers):
            break
        time.sleep(_RETRY_DELAY_SECONDS)
        frame = _download(tickers, start, end)
    return frame


def fetch_prices(
    tickers: list[str], start: str, end: str,
    use_cache: bool = True, force_refresh: bool = False,
) -> pd.DataFrame:
    tickers = sorted(set(tickers))

    if use_cache and not force_refresh:
        missing = find_missing_months_wide(_PRICE_CACHE_BASE, start, end, tickers)
        if missing:
            def _fetch_month(ym: tuple[int, int]) -> None:
                ms, me = month_bounds(*ym)
                frame = _download_with_retry(tickers, ms, me)
                if not frame.empty:
                    save_monthly_chunks_wide(frame, _PRICE_CACHE_BASE)
                else:
                    p = chunk_path(_PRICE_CACHE_BASE, *ym)
                    if not p.exists():
                        pd.DataFrame().to_parquet(p, index=False)

            with ThreadPoolExecutor(max_workers=min(len(missing), 4)) as ex:
                list(ex.map(_fetch_month, missing))

        cached = read_monthly_cache_wide(_PRICE_CACHE_BASE, start, end)
        if cached.empty:
            return pd.DataFrame()
        available = [t for t in tickers if t in cached.columns]
        return cached[available].loc[start:end].sort_index()

    prices = _download(tickers, start, end)
    if use_cache and not prices.empty:
        save_monthly_chunks_wide(prices, _PRICE_CACHE_BASE)
    available = [t for t in tickers if t in prices.columns]
    return prices[available].sort_index() if available else pd.DataFrame()
EOF
```

- [ ] **Step 5: Run the tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_prices.py -q`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add grid_equipment_basket/__init__.py grid_equipment_basket/config.py \
        grid_equipment_basket/data/__init__.py grid_equipment_basket/data/prices.py \
        tests/grid_equipment_basket/__init__.py tests/grid_equipment_basket/test_prices.py
git commit -m "feat(grid-equipment): package scaffold, config, cached price fetcher"
```

---

## Task 2: Basket weight helpers and portfolio simulation

**Files:**
- Create: `grid_equipment_basket/basket.py`
- Test: `tests/grid_equipment_basket/test_basket.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure pandas).
- Produces:
  - `apply_cap(weights: pd.Series, cap: float) -> pd.Series` — iteratively caps each weight at `cap`, redistributing excess pro-rata among uncapped names; result sums to 1.0. If `len(weights) * cap < 1`, emits `RuntimeWarning` and returns equal weights `1/n`.
  - `equal_weight_targets(available: list[str], cap: float = 0.25) -> pd.Series` — index `sorted(available)`, equal weights passed through `apply_cap`. Empty input → empty `pd.Series(dtype=float)`.
  - `rebalance_dates(start: str, end: str, lag_days: int = 42) -> list[pd.Timestamp]` — sorted; every calendar quarter-end (Mar/Jun/Sep/Dec month-end) whose date `+ lag_days` lands in `[start, end]`, mapped to that lagged date.
  - `@dataclass BasketResult`: `returns: pd.Series` (daily simple returns, first day dropped), `weights: pd.DataFrame` (rebalance-date index, ticker columns, target weights used for the period that begins on that date), `rebalances: list[pd.Timestamp]` (formation dates after the initial one).
  - `simulate_basket(prices: pd.DataFrame, start: str, end: str, lag_days: int = 42, cap: float = 0.25, target_fn=None) -> BasketResult` — `target_fn(available: list[str], asof: pd.Timestamp) -> pd.Series | None`; default builds `equal_weight_targets(available, cap)`. Holds fixed share counts between formation dates; re-forms on the first trading day `>= start` and on each `rebalance_dates` entry snapped forward to the next available trading day, using only tickers with a positive price that day. Portfolio value starts at 1.0.

- [ ] **Step 1: Write the failing tests**

```bash
cat > tests/grid_equipment_basket/test_basket.py << 'EOF'
import warnings

import pandas as pd
import pytest

from grid_equipment_basket import basket as bk


# ── apply_cap / equal_weight_targets ────────────────────────────────────────
def test_equal_weight_sums_to_one_and_is_equal():
    w = bk.equal_weight_targets(["C", "A", "B"])
    assert list(w.index) == ["A", "B", "C"]
    assert w.sum() == pytest.approx(1.0)
    assert w.nunique() == 1


def test_equal_weight_cap_not_binding_for_seven_names():
    w = bk.equal_weight_targets([f"T{i}" for i in range(7)], cap=0.25)
    assert w.max() == pytest.approx(1 / 7)


def test_apply_cap_binding_redistributes_pro_rata():
    w = bk.apply_cap(pd.Series({"A": 0.60, "B": 0.20, "C": 0.20}), cap=0.40)
    assert w["A"] == pytest.approx(0.40)
    assert w["B"] == pytest.approx(0.30)
    assert w["C"] == pytest.approx(0.30)
    assert w.sum() == pytest.approx(1.0)


def test_apply_cap_infeasible_returns_equal_and_warns():
    with pytest.warns(RuntimeWarning):
        w = bk.apply_cap(pd.Series({"A": 0.9, "B": 0.05, "C": 0.05}), cap=0.25)
    assert w.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_equal_weight_empty_input():
    assert bk.equal_weight_targets([]).empty


# ── rebalance_dates ─────────────────────────────────────────────────────────
def test_rebalance_dates_are_lagged_quarter_ends():
    dates = bk.rebalance_dates("2023-01-01", "2023-12-31", lag_days=42)
    expected = [
        pd.Timestamp("2022-12-31") + pd.Timedelta(days=42),
        pd.Timestamp("2023-03-31") + pd.Timedelta(days=42),
        pd.Timestamp("2023-06-30") + pd.Timedelta(days=42),
        pd.Timestamp("2023-09-30") + pd.Timedelta(days=42),
    ]
    assert dates == expected


# ── simulate_basket ────────────────────────────────────────────────────────
def test_simulate_two_asset_drift_hand_computed():
    idx = pd.bdate_range("2023-01-02", periods=3)
    prices = pd.DataFrame({"A": [10.0, 11.0, 11.0], "B": [10.0, 10.0, 12.0]}, index=idx)
    res = bk.simulate_basket(prices, "2023-01-02", "2023-01-04", lag_days=42)
    # shares: 0.5/10 each. day1 value 1.05 -> r=0.05 ; day2 value 1.15 -> r=1.15/1.05-1
    assert res.returns.iloc[0] == pytest.approx(0.05)
    assert res.returns.iloc[1] == pytest.approx(1.15 / 1.05 - 1.0)


def test_simulate_weight_rows_sum_to_one():
    idx = pd.bdate_range("2023-01-02", periods=200)
    prices = pd.DataFrame(
        {t: [100.0 + i for i in range(len(idx))] for t in ["A", "B", "C"]}, index=idx
    )
    res = bk.simulate_basket(prices, "2023-01-02", str(idx[-1].date()))
    assert (res.weights.sum(axis=1) - 1.0).abs().max() < 1e-9
    assert len(res.rebalances) >= 1


def test_simulate_excludes_ticker_until_it_has_a_price():
    idx = pd.bdate_range("2023-01-02", periods=200)
    a = [100.0 + i for i in range(len(idx))]
    b = [float("nan")] * 120 + [50.0 + i for i in range(len(idx) - 120)]
    prices = pd.DataFrame({"A": a, "B": b}, index=idx)
    res = bk.simulate_basket(prices, "2023-01-02", str(idx[-1].date()))
    first_row = res.weights.iloc[0]
    last_row = res.weights.iloc[-1]
    assert first_row.get("B", 0.0) == 0.0
    assert last_row["B"] > 0.0
EOF
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_basket.py -q`
Expected: FAIL / ERROR — `grid_equipment_basket.basket` does not exist.

- [ ] **Step 3: Write `basket.py`**

```bash
cat > grid_equipment_basket/basket.py << 'EOF'
from __future__ import annotations

"""Long-only thematic-basket construction and hold-shares/periodic-rebalance
simulation. Simple-return convention throughout.
"""

import warnings
from dataclasses import dataclass

import pandas as pd


def apply_cap(weights: pd.Series, cap: float) -> pd.Series:
    w = weights.astype(float).copy()
    n = len(w)
    if n == 0:
        return w
    if n * cap < 1.0 - 1e-9:
        warnings.warn(
            f"single-name cap {cap} infeasible for {n} names; using equal weight",
            RuntimeWarning, stacklevel=2,
        )
        return pd.Series(1.0 / n, index=w.index)
    w = w / w.sum()
    for _ in range(n):
        over = w[w > cap + 1e-12]
        if over.empty:
            break
        excess = float((over - cap).sum())
        w.loc[over.index] = cap
        under = w[w < cap - 1e-12]
        if under.empty:
            break
        w.loc[under.index] += excess * under / under.sum()
    return w


def equal_weight_targets(available, cap: float = 0.25) -> pd.Series:
    names = sorted(available)
    if not names:
        return pd.Series(dtype=float)
    raw = pd.Series(1.0 / len(names), index=names)
    return apply_cap(raw, cap)


def rebalance_dates(start: str, end: str, lag_days: int = 42) -> list:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for year in range(s.year - 1, e.year + 2):
        for m in (3, 6, 9, 12):
            qend = pd.Timestamp(year=year, month=m, day=1) + pd.offsets.MonthEnd(0)
            rd = qend + pd.Timedelta(days=lag_days)
            if s <= rd <= e:
                out.append(rd)
    return sorted(out)


@dataclass
class BasketResult:
    returns: pd.Series
    weights: pd.DataFrame
    rebalances: list


def simulate_basket(
    prices: pd.DataFrame, start: str, end: str,
    lag_days: int = 42, cap: float = 0.25, target_fn=None,
) -> BasketResult:
    px = prices.loc[start:end].sort_index().ffill()
    if px.empty:
        return BasketResult(pd.Series(dtype=float), pd.DataFrame(), [])

    if target_fn is None:
        def target_fn(available, asof):
            return equal_weight_targets(available, cap)

    raw_dates = [d for d in rebalance_dates(start, end, lag_days) if d <= px.index[-1]]
    snapped = []
    for d in raw_dates:
        pos = px.index.get_indexer([d], method="bfill")[0]
        if pos != -1:
            snapped.append(px.index[pos])
    form_dates = sorted(set([px.index[0], *snapped]))

    value = pd.Series(index=px.index, dtype=float)
    weights_log: dict = {}
    shares = pd.Series(dtype=float)
    cur_val = 1.0

    for i, day in enumerate(px.index):
        if day in form_dates:
            row = px.loc[day].dropna()
            available = [t for t in row.index if row[t] > 0]
            tgt = target_fn(available, day)
            if tgt is not None and not tgt.empty:
                tgt = tgt / tgt.sum()
                shares = (cur_val * tgt) / row.reindex(tgt.index)
                weights_log[day] = tgt
        if not shares.empty:
            held = px.loc[day, shares.index].fillna(0.0)
            cur_val = float((shares * held).sum())
        value.iloc[i] = cur_val

    ret = value.pct_change().dropna()
    wdf = pd.DataFrame(weights_log).T.sort_index()
    wdf = wdf.reindex(columns=sorted(wdf.columns)).fillna(0.0)
    return BasketResult(ret, wdf, [d for d in form_dates if d != px.index[0]])
EOF
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_basket.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add grid_equipment_basket/basket.py tests/grid_equipment_basket/test_basket.py
git commit -m "feat(grid-equipment): equal-weight construction + hold-shares rebalance simulation"
```

---

## Task 3: Metrics, benchmark-relative stats, run orchestration, CLI

**Files:**
- Create: `grid_equipment_basket/backtest.py`
- Create: `grid_equipment_basket/__main__.py`
- Test: `tests/grid_equipment_basket/test_backtest.py`

**Interfaces:**
- Consumes: `grid_equipment_basket.basket.simulate_basket` / `BasketResult`; `grid_equipment_basket.data.prices.fetch_prices`; `grid_equipment_basket.config`.
- Produces:
  - `compute_metrics(returns: pd.Series, rf_annual: float = 0.04, ann_factor: int = 252) -> dict` — keys `n_obs, cagr, ann_vol, sharpe, sortino, max_dd, hit_rate`. `< 2` obs or zero variance → metric values `nan`, `n_obs` still set. `cagr` = `(1+r).prod() ** (ann_factor/n) - 1`; `sharpe` = `mean(r - rf/ann) / std(r - rf/ann) * sqrt(ann)`; `max_dd` from `(1+r).cumprod()`.
  - `calendar_year_returns(returns: pd.Series) -> pd.Series` — int-year index, compounded simple return per year.
  - `relative_metrics(basket: pd.Series, bench: pd.Series, rf_annual: float = 0.04, ann_factor: int = 252) -> dict` — keys `corr, tracking_error, information_ratio, excess_cagr`, computed on the date intersection.
  - `run(start: str, end: str, universe: list[str] | None = None, benchmarks: list[str] | None = None, target_fn=None, price_fn=None) -> dict` — `price_fn(tickers, start, end) -> pd.DataFrame` defaults to `fetch_prices`. Returns:
    ```
    {"start","end","universe","n_names_end",
     "basket": <compute_metrics dict>,
     "basket_calendar": pd.Series,
     "basket_returns": pd.Series,
     "benchmark_returns": pd.DataFrame,
     "benchmarks": {sym: {"metrics": dict, "calendar": pd.Series, "relative": dict}}}
    ```
  - `results_table(results: dict) -> str` — plain-text table; contains the literal headers `CAGR` and `Sharpe` and one row per benchmark symbol.
  - `plot(results: dict, save_path: str) -> None` — 2 stacked panels (log-y equity curve, drawdown), basket + each benchmark, saved to `save_path`.
- `__main__.py` flags: `--start` (default `config.PRIMARY_START`), `--end` (default = last day of the previous calendar month), `--prior-regime` (overrides start/end with `config.PRIOR_REGIME_START/END`), `--tilt {none,backlog}` (default `none`; `backlog` calls `_backlog_target_fn` — see Task 9 — and exits with a clear message if `backlog_data` is absent), `--output` (default `./output_grid_equipment`), `--no-plot`. Writes `<output>/metrics.csv`, `<output>/basket_returns.csv`, `<output>/performance.png`; prints `results_table`.

- [ ] **Step 1: Write the failing tests**

```bash
cat > tests/grid_equipment_basket/test_backtest.py << 'EOF'
import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import backtest as bt


def test_compute_metrics_guard_on_zero_variance():
    out = bt.compute_metrics(pd.Series([0.1, 0.1, 0.1, 0.1]))
    assert out["n_obs"] == 4
    assert np.isnan(out["cagr"])
    assert np.isnan(out["sharpe"])


def test_compute_metrics_max_drawdown_closed_form():
    out = bt.compute_metrics(pd.Series([0.10, -0.50, 0.10]))
    # curve = [1.1, 0.55, 0.605]; peak 1.1 -> min dd = 0.55/1.1 - 1 = -0.5
    assert out["max_dd"] == pytest.approx(-0.5)


def test_compute_metrics_matches_independent_numpy():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.012, 252))
    out = bt.compute_metrics(r, rf_annual=0.04, ann_factor=252)
    growth = float((1 + r).prod())
    exp_cagr = growth ** (252 / len(r)) - 1
    ex = r - 0.04 / 252
    exp_sharpe = ex.mean() / ex.std() * np.sqrt(252)
    assert out["cagr"] == pytest.approx(round(exp_cagr, 4))
    assert out["sharpe"] == pytest.approx(round(float(exp_sharpe), 3))


def test_calendar_year_returns_split_on_year_boundary():
    idx = pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03"])
    r = pd.Series([0.01, 0.02, -0.01, 0.03], index=idx)
    cy = bt.calendar_year_returns(r)
    assert set(cy.index) == {2023, 2024}
    assert cy.loc[2023] == pytest.approx((1.01 * 1.02) - 1, abs=1e-4)


def test_relative_metrics_identical_series():
    r = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01],
                  index=pd.bdate_range("2023-01-02", periods=5))
    rel = bt.relative_metrics(r, r.copy())
    assert rel["corr"] == pytest.approx(1.0)
    assert rel["tracking_error"] == pytest.approx(0.0)
    assert rel["excess_cagr"] == pytest.approx(0.0)


def test_relative_metrics_uses_date_intersection():
    idx = pd.bdate_range("2023-01-02", periods=6)
    b = pd.Series([0.01] * 5, index=idx[:5])
    m = pd.Series([0.005] * 6, index=idx)
    rel = bt.relative_metrics(b, m)
    assert not np.isnan(rel["information_ratio"])


def test_run_with_synthetic_price_fn():
    idx = pd.bdate_range("2023-01-02", periods=260)
    rng = np.random.default_rng(1)

    def _prices(tickers, start, end):
        data = {}
        for k, t in enumerate(sorted(tickers)):
            steps = rng.normal(0.0005 + 0.0001 * k, 0.01, len(idx))
            data[t] = 100 * np.exp(np.cumsum(steps))
        return pd.DataFrame(data, index=idx)

    res = bt.run(
        "2023-01-02", str(idx[-1].date()),
        universe=["AAA", "BBB", "CCC"], benchmarks=["XLI", "GRID"],
        price_fn=_prices,
    )
    assert set(res["benchmarks"]) == {"XLI", "GRID"}
    assert res["basket"]["n_obs"] > 200
    assert np.isfinite(res["basket"]["sharpe"])
    tbl = bt.results_table(res)
    assert "CAGR" in tbl and "XLI" in tbl
EOF
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_backtest.py -q`
Expected: FAIL / ERROR — `grid_equipment_basket.backtest` does not exist.

- [ ] **Step 3: Write `backtest.py`**

```bash
cat > grid_equipment_basket/backtest.py << 'EOF'
from __future__ import annotations

"""Metrics, benchmark-relative stats, and full-run orchestration for the
grid-equipment basket. Simple-return convention (see README)."""

import numpy as np
import pandas as pd

from grid_equipment_basket import config
from grid_equipment_basket.basket import simulate_basket


def compute_metrics(returns: pd.Series, rf_annual: float = 0.04, ann_factor: int = 252) -> dict:
    r = returns.dropna()
    out: dict = {"n_obs": int(len(r))}
    if len(r) < 2 or r.std() == 0:
        out.update(cagr=np.nan, ann_vol=np.nan, sharpe=np.nan,
                   sortino=np.nan, max_dd=np.nan, hit_rate=np.nan)
        return out
    growth = float((1.0 + r).prod())
    years = len(r) / ann_factor
    cagr = growth ** (1.0 / years) - 1.0
    ann_vol = float(r.std() * np.sqrt(ann_factor))
    excess = r - rf_annual / ann_factor
    sharpe = float(excess.mean() / excess.std() * np.sqrt(ann_factor))
    downside = float(r[r < 0].std() * np.sqrt(ann_factor))
    sortino = float((r.mean() * ann_factor - rf_annual) / downside) if downside > 0 else np.nan
    curve = (1.0 + r).cumprod()
    max_dd = float((curve / curve.cummax() - 1.0).min())
    out.update(
        cagr=round(cagr, 4), ann_vol=round(ann_vol, 4), sharpe=round(sharpe, 3),
        sortino=round(sortino, 3) if sortino == sortino else np.nan,
        max_dd=round(max_dd, 4), hit_rate=round(float((r > 0).mean()), 3),
    )
    return out


def calendar_year_returns(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    by_year = (1.0 + r).groupby(r.index.year).prod() - 1.0
    by_year.index = by_year.index.astype(int)
    return by_year.round(4)


def relative_metrics(basket: pd.Series, bench: pd.Series,
                     rf_annual: float = 0.04, ann_factor: int = 252) -> dict:
    df = pd.concat([basket.rename("b"), bench.rename("m")], axis=1).dropna()
    if len(df) < 2:
        return {"corr": np.nan, "tracking_error": np.nan,
                "information_ratio": np.nan, "excess_cagr": np.nan}
    active = df["b"] - df["m"]
    te = float(active.std() * np.sqrt(ann_factor))
    ir = float(active.mean() / active.std() * np.sqrt(ann_factor)) if active.std() > 0 else np.nan
    b_cagr = compute_metrics(df["b"], rf_annual, ann_factor)["cagr"]
    m_cagr = compute_metrics(df["m"], rf_annual, ann_factor)["cagr"]
    excess_cagr = (b_cagr - m_cagr) if (b_cagr == b_cagr and m_cagr == m_cagr) else np.nan
    return {
        "corr": round(float(df["b"].corr(df["m"])), 3),
        "tracking_error": round(te, 4),
        "information_ratio": round(ir, 3) if ir == ir else np.nan,
        "excess_cagr": round(float(excess_cagr), 4) if excess_cagr == excess_cagr else np.nan,
    }


def _default_price_fn(tickers, start, end):
    from grid_equipment_basket.data.prices import fetch_prices
    return fetch_prices(tickers, start, end)


def run(start: str, end: str, universe=None, benchmarks=None,
        target_fn=None, price_fn=None) -> dict:
    universe = list(universe or config.UNIVERSE)
    benchmarks = list(benchmarks or config.BENCHMARKS)
    price_fn = price_fn or _default_price_fn

    prices = price_fn(sorted(set(universe + benchmarks)), start, end)
    uni_cols = [t for t in universe if t in prices.columns]
    br = simulate_basket(
        prices[uni_cols], start, end,
        config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT, target_fn,
    )
    bench_cols = [t for t in benchmarks if t in prices.columns]
    bench_ret = prices[bench_cols].loc[start:end].pct_change().dropna(how="all")

    res: dict = {
        "start": start, "end": end, "universe": universe,
        "n_names_end": int(br.weights.iloc[-1].gt(0).sum()) if not br.weights.empty else 0,
        "basket": compute_metrics(br.returns, config.RISK_FREE_RATE, config.ANN_FACTOR),
        "basket_calendar": calendar_year_returns(br.returns),
        "basket_returns": br.returns,
        "benchmark_returns": bench_ret,
        "benchmarks": {},
    }
    for b in bench_ret.columns:
        col = bench_ret[b].dropna()
        res["benchmarks"][b] = {
            "metrics": compute_metrics(col, config.RISK_FREE_RATE, config.ANN_FACTOR),
            "calendar": calendar_year_returns(col),
            "relative": relative_metrics(br.returns, col, config.RISK_FREE_RATE, config.ANN_FACTOR),
        }
    return res


def results_table(results: dict) -> str:
    rows = [("BASKET", results["basket"])]
    rows += [(b, results["benchmarks"][b]["metrics"]) for b in results["benchmarks"]]
    lines = [
        f"Window {results['start']} -> {results['end']}   "
        f"(basket names at end: {results['n_names_end']})",
        f"{'':<10}{'CAGR':>9}{'Vol':>9}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}",
    ]
    for name, m in rows:
        lines.append(
            f"{name:<10}{_p(m['cagr']):>9}{_p(m['ann_vol']):>9}"
            f"{_f(m['sharpe']):>9}{_f(m['sortino']):>9}{_p(m['max_dd']):>9}"
        )
    lines.append("")
    lines.append(f"{'vs':<10}{'exCAGR':>9}{'corr':>9}{'TE':>9}{'IR':>9}")
    for b in results["benchmarks"]:
        rel = results["benchmarks"][b]["relative"]
        lines.append(
            f"{b:<10}{_p(rel['excess_cagr']):>9}{_f(rel['corr']):>9}"
            f"{_p(rel['tracking_error']):>9}{_f(rel['information_ratio']):>9}"
        )
    return "\n".join(lines)


def _p(x) -> str:
    return "n/a" if x != x else f"{x * 100:.1f}%"


def _f(x) -> str:
    return "n/a" if x != x else f"{x:.2f}"


def plot(results: dict, save_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    basket = results["basket_returns"]
    bench = results["benchmark_returns"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    bcurve = (1 + basket).cumprod()
    ax1.plot(bcurve.index, bcurve.values, lw=2.0, color="#1a3a5c", label="Basket")
    for c in bench.columns:
        cc = (1 + bench[c].dropna()).cumprod()
        ax1.plot(cc.index, cc.values, lw=1.0, alpha=0.8, label=c)
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of 1 (log)")
    ax1.legend(fontsize=8, ncol=3)
    ax1.set_title("Grid Equipment Suppliers Basket vs Benchmarks")
    dd = bcurve / bcurve.cummax() - 1.0
    ax2.fill_between(dd.index, dd.values, 0, color="#8c2d04", alpha=0.5)
    ax2.set_ylabel("Basket drawdown")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
EOF
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_backtest.py -q`
Expected: 7 passed.

- [ ] **Step 5: Write `__main__.py`**

```bash
cat > grid_equipment_basket/__main__.py << 'EOF'
from __future__ import annotations

"""CLI: python -m grid_equipment_basket [options]"""

import argparse
from pathlib import Path

import pandas as pd

from grid_equipment_basket import backtest, config


def _default_end() -> str:
    today = pd.Timestamp.today().normalize()
    last_month_end = today.replace(day=1) - pd.Timedelta(days=1)
    return last_month_end.strftime("%Y-%m-%d")


def _backlog_target_fn():
    try:
        from grid_equipment_basket.backlog_data import load_backlog_csv
        from grid_equipment_basket.basket import backlog_tilt_targets
    except ImportError as exc:  # Step 2 not built yet
        raise SystemExit(
            "--tilt backlog requires the Step 2 backlog module, which is only "
            "built if the Step 1 decision gate passed. See "
            "docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md."
        ) from exc
    bdf = load_backlog_csv()

    def _fn(available, asof):
        return backlog_tilt_targets(
            available, asof, bdf,
            cap=config.MAX_SINGLE_NAME_WEIGHT,
            top=config.BACKLOG_TILT_TOP, bottom=config.BACKLOG_TILT_BOTTOM,
        )

    return _fn


def main() -> None:
    ap = argparse.ArgumentParser(prog="grid_equipment_basket")
    ap.add_argument("--start", default=config.PRIMARY_START)
    ap.add_argument("--end", default=_default_end())
    ap.add_argument("--prior-regime", action="store_true",
                    help="run the 2020-2022 panel instead of the primary window")
    ap.add_argument("--tilt", choices=["none", "backlog"], default="none")
    ap.add_argument("--output", default="./output_grid_equipment")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    start, end = args.start, args.end
    if args.prior_regime:
        start, end = config.PRIOR_REGIME_START, config.PRIOR_REGIME_END

    target_fn = _backlog_target_fn() if args.tilt == "backlog" else None

    res = backtest.run(start, end, target_fn=target_fn)
    print(backtest.results_table(res))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"name": "BASKET", **res["basket"]}]
    for b, blk in res["benchmarks"].items():
        rows.append({"name": b, **blk["metrics"]})
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    res["basket_returns"].rename("basket_return").to_csv(out / "basket_returns.csv")
    if not args.no_plot:
        backtest.plot(res, str(out / "performance.png"))
        print(f"\nwrote {out}/metrics.csv, basket_returns.csv, performance.png")


if __name__ == "__main__":
    main()
EOF
```

- [ ] **Step 6: Smoke-test the CLI wiring (no network — synthetic import check)**

Run: `.venv/bin/python -c "import grid_equipment_basket.__main__ as m; print(m._default_end()); print('ok')"`
Expected: prints a `YYYY-MM-DD` date (last day of previous month) and `ok`.

- [ ] **Step 7: Commit**

```bash
git add grid_equipment_basket/backtest.py grid_equipment_basket/__main__.py \
        tests/grid_equipment_basket/test_backtest.py
git commit -m "feat(grid-equipment): metrics, benchmark-relative stats, run orchestration, CLI"
```

---

## Task 4: Candidate universe verification

**Files:**
- Create: `grid_equipment_basket/candidate_research.md`
- Modify: `grid_equipment_basket/config.py` (replace the provisional `UNIVERSE`)

This task has no unit test — its deliverable is a verification log a fresh reader can check against the cited filings, plus a finalized `UNIVERSE`.

**Candidates to verify (from spec §2):** `ETN, HUBB, GEV, VRT, PWR, MYRG, NVT, FLNC, PRIM`.
**Log-but-exclude (foreign):** `ABB, SIEGY (Siemens Energy), PRYMY (Prysmian), NEXNF (Nexans)`.

- [ ] **Step 1: For each candidate, gather the three facts**

For each ticker:
1. Resolve CIK: `curl -s -A "acadia-research sand.gh1902@gmail.com" https://www.sec.gov/files/company_tickers.json` and grep the ticker.
2. Pull the latest annual filing index:
   `curl -s -A "acadia-research sand.gh1902@gmail.com" "https://data.sec.gov/submissions/CIK##########.json"` (10-digit zero-padded CIK) → find the most recent `form` in `{10-K, 20-F}` and its `primaryDocument`.
3. Open that document's "Business" and "Backlog"/"Remaining performance obligations" sections (URL pattern `https://www.sec.gov/Archives/edgar/data/<cik>/<accession-no-dashes>/<primaryDocument>`).
4. Record:
   - **business_verdict**: does a plain reading say material revenue from selling grid / data-center electrical equipment, or engineering/constructing electric-power infrastructure? `include` / `exclude`, with a one-sentence quote.
   - **backlog_disclosure**: one of `xbrl_rpo` / `nongaap_backlog_total` / `nongaap_backlog_segment` / `book_to_bill_only` / `none` (note segment name if segment-level).
   - **listing**: US exchange + ticker + first trading date with clean daily history (for GEV expect ~2024-04-02).
   - **decision** + one-line reason.

- [ ] **Step 2: Write `candidate_research.md`**

```bash
cat > grid_equipment_basket/candidate_research.md << 'EOF'
# Grid Equipment Suppliers — Candidate Verification Log

Spec: docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md (§2).
Inclusion rule: material grid / data-center electrical-infrastructure revenue per the
latest 10-K business description — NEVER historical returns. FLNC is deliberately
retained if it meets the business test. Foreign-listed names are logged and excluded
from the backtestable basket.

## Verified candidates

### ETN — Eaton Corp
- CIK: <cik>
- Latest annual filing: <form> <date> <url>
- business_verdict: <include|exclude> — "<quote>"
- backlog_disclosure: <type> (<segment name if segment-level>)
- listing: NYSE:ETN, clean daily history since <date>
- decision: <include|exclude> — <reason>

### HUBB — Hubbell Inc
<same structure>

### GEV — GE Vernova
<same structure — note 2024-04-02 listing>

### VRT — Vertiv Holdings
<same structure>

### PWR — Quanta Services
<same structure>

### MYRG — MYR Group
<same structure — include avg daily $ volume>

### NVT — nVent Electric
<same structure>

### FLNC — Fluence Energy
<same structure — record the business verdict honestly regardless of price history>

### PRIM — Primoris Services
<same structure>

## Logged but excluded (foreign-listed)

| Ticker | Company | Reason for exclusion |
|---|---|---|
| ABB | ABB Ltd | Foreign-listed; only US access is thin OTC ADR (ABBNY) — not comparable daily history |
| SIEGY | Siemens Energy | Foreign-listed; OTC ADR only |
| PRYMY | Prysmian | Foreign-listed; OTC ADR only |
| NEXNF | Nexans | Foreign-listed; OTC grey-market only |

## Final basket universe

<comma-separated verified include-list>

Count: <n>. (Report honestly — a small basket is not a failed deliverable.)
EOF
```

- [ ] **Step 3: Update `config.UNIVERSE`**

Replace the provisional list in `grid_equipment_basket/config.py` with the verified include-list from Step 2, keeping the explanatory comment.

- [ ] **Step 4: Verify prices actually fetch for the final universe**

Run:
```bash
.venv/bin/python -c "
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket import config
p = fetch_prices(config.UNIVERSE + config.BENCHMARKS, '2023-01-01', '2023-06-30')
print(p.tail(2)); print('missing:', [t for t in config.UNIVERSE+config.BENCHMARKS if t not in p.columns])
"
```
Expected: a price frame; `missing` is empty (or only names whose first listing is after 2023-06-30, e.g. GEV — acceptable, note it).

- [ ] **Step 5: Commit**

```bash
git add grid_equipment_basket/candidate_research.md grid_equipment_basket/config.py
git commit -m "feat(grid-equipment): verified candidate universe"
```

---

## Task 5: Execute Step 1 backtest, write results, evaluate the decision gate

**Files:**
- Create: `grid_equipment_basket/README.md`
- Create: `docs/grid-equipment-basket-step1-results.md`
- Modify: repo-root `README.md` (one-line pointer)

No unit test — the deliverable is a reproducible run plus a writeup that states the gate outcome with numbers.

- [ ] **Step 1: Run the primary window**

```bash
.venv/bin/python -m grid_equipment_basket --start 2023-01-01 --output output_grid_equipment | tee output_grid_equipment/step1_primary.txt
```

- [ ] **Step 2: Run the prior-regime panel**

```bash
.venv/bin/python -m grid_equipment_basket --prior-regime --output output_grid_equipment_prior | tee output_grid_equipment_prior/step1_prior.txt
```

- [ ] **Step 3: Evaluate the decision gate (Spec §5)**

From `output_grid_equipment/metrics.csv`, record for the primary window:
- basket CAGR, basket Sharpe
- XLI CAGR, XLI Sharpe
- GRID CAGR, PAVE CAGR

Gate PASSES iff **all**: `basket_CAGR > XLI_CAGR` **and** `basket_Sharpe > XLI_Sharpe` **and** `basket_CAGR >= max(GRID_CAGR, PAVE_CAGR) - 0.02`.

- [ ] **Step 4: Write the results doc**

```bash
cat > docs/grid-equipment-basket-step1-results.md << 'EOF'
# Grid Equipment Suppliers Basket — Step 1 Results

Spec: docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md
Plan: docs/superpowers/plans/2026-08-28-grid-equipment-basket.md

## Basket
Universe (verified, business-description test only): <list>
Construction: equal-weight, quarterly rebalance ~6 weeks after quarter-end,
25% single-name cap, long-only, total-return, simple-return convention.

## Primary window 2023-01-01 -> <end>
<paste output_grid_equipment/step1_primary.txt table>

Calendar-year returns (basket vs benchmarks): <table from metrics/calendars>

## Prior-regime panel 2020-01-01 -> 2022-12-31
<paste output_grid_equipment_prior/step1_prior.txt table>

## Decision gate (Spec §5)
| Check | Value | Pass? |
|---|---|---|
| basket CAGR > XLI CAGR | <b> vs <x> | <y/n> |
| basket Sharpe > XLI Sharpe | <b> vs <x> | <y/n> |
| basket CAGR within 2pp of max(GRID,PAVE) | <b> vs <g/p> | <y/n> |

**Gate result: PASS / FAIL**

## Mandatory caveats (Spec §5, §7)
- Primary window is ~3.5 years (~43 monthly observations); GEV covers less than half of it.
  Standard error on an annualized Sharpe estimate this size is ~+/-0.5. "Beats the benchmark"
  here is a directional point estimate, not a significant result.
- Survivorship / hindsight bias: the universe was chosen in 2026 knowing which names won.
  Mitigations: inclusion on business description only; FLNC (a known underperformer) kept in;
  equal-weight removes weight cherry-picking; GRID/PAVE comparison catches "this is just the theme".

## Next step
PASS -> proceed to Task 6 (backlog data) and the Step 2 tilt.
FAIL -> stop. This negative result is the deliverable. No backlog data collected.
EOF
```
Fill every `<...>` placeholder with real numbers from the run before committing.

- [ ] **Step 5: Write the module README**

```bash
cat > grid_equipment_basket/README.md << 'EOF'
# Grid Equipment Suppliers Thematic Basket

Long-only basket of US-listed grid / data-center electrical-infrastructure suppliers —
a direct-revenue bet on the power buildout, as opposed to the utility-demand bets in
`grid_resilience/` and `dc_demand_basket/`.

Spec: ../docs/superpowers/specs/2026-08-28-grid-equipment-basket-design.md

## Composition (as of <date>)
| Ticker | Company | Bucket |
|---|---|---|
<verified names + one-line bucket each>

Weighting: equal-weight, quarterly rebalance ~6 weeks after each calendar quarter-end,
25% single-name cap. Long-only, fully invested, no leverage. Total-return.

## Benchmarks
XLI (primary — universe is GICS Industrials), SPY and XLU (secondary / continuity with
the rest of the repo), GRID and PAVE (honesty check — rules-based thematic ETFs; if this
basket only matches them, the edge is "the theme", not the construction).

## Step 1 result
Window 2023-01-01 -> <end>. <one-line headline with basket vs XLI CAGR/Sharpe>.
Decision gate: **PASS / FAIL**. Full numbers: ../docs/grid-equipment-basket-step1-results.md

## Return convention
All maths here uses SIMPLE returns (`pct_change`, `(1+r).prod()`), unlike `grid_resilience`
which uses log returns. A log-based recompute will not tie out exactly — that is expected.

## Run
```
python -m grid_equipment_basket --start 2023-01-01      # primary window
python -m grid_equipment_basket --prior-regime          # 2020-2022 panel
python -m grid_equipment_basket --tilt backlog          # Step 2 (only if gate PASSED)
```

## Phase 2 — cross-sectional factor (NOT built here)
Only if verification ever lands 8+ names with clean, comparable, segment-level backlog and
multiple years of quarterly history does a cross-sectional backlog-growth-surprise factor
become defensible. That needs its own spec — do not reintroduce `build_factor()` / z-scoring
/ IC machinery into this basket.
EOF
```
Fill placeholders with real values.

- [ ] **Step 6: Add the repo-root README pointer**

In the repo-root `README.md`, directly under the top `# Grid Resilience Strategy` intro block, add:

```markdown
> **Sibling strategy modules:** `grid_equipment_basket/` — a long-only thematic basket of grid / data-center equipment suppliers (see its own README and `docs/grid-equipment-basket-step1-results.md`).
```

- [ ] **Step 7: Commit**

```bash
git add grid_equipment_basket/README.md docs/grid-equipment-basket-step1-results.md README.md
git commit -m "docs(grid-equipment): Step 1 backtest results and decision-gate outcome"
```

- [ ] **Step 8: GATE CHECKPOINT**

If the gate is **FAIL**, stop here — the plan is complete as a negative result. Do **not** start Task 6.
If the gate is **PASS**, continue.

---

## Task 6: Backlog data module (GATED — only if Task 5 gate PASSED)

**Files:**
- Create: `grid_equipment_basket/backlog_data.py`
- Create: `grid_equipment_basket/data/backlog_quarterly.csv`
- Test: `tests/grid_equipment_basket/test_backlog_data.py`

**Interfaces:**
- Produces:
  - `CIK_BY_TICKER: dict[str, str]` — 10-digit zero-padded CIKs for the verified universe, resolved once from `https://www.sec.gov/files/company_tickers.json` and hard-coded.
  - `fetch_rpo(ticker: str, use_cache: bool = True) -> pd.DataFrame` — SEC XBRL `companyconcept` for `us-gaap:RevenueRemainingPerformanceObligation` (falls back to summing `...Current` + `...Noncurrent`). Columns `["quarter_end", "availability_date", "metric_value"]` (`availability_date` = XBRL `filed`). Empty frame if the concept is untagged for that filer. Cached one parquet per ticker under `config.CACHE_DIR`.
  - `load_backlog_csv(path: str | None = None) -> pd.DataFrame` — reads `data/backlog_quarterly.csv`, parses `quarter_end` / `availability_date` as datetimes, validates `disclosure_type` against `{"xbrl_rpo","nongaap_backlog_total","nongaap_backlog_segment","book_to_bill_only"}` (raises `ValueError` on anything else), returns the tidy frame.
  - `backlog_growth_signal(df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series` — per ticker, using only rows with `availability_date <= asof`: trailing-4-quarter YoY growth of `metric_value`; for `disclosure_type == "book_to_bill_only"` rows use the latest available book-to-bill level minus 1.0 instead. `NaN` where a name lacks ≥ 5 quarters (or any B2B) as-of.
  - `backlog_ranks(signal: pd.Series) -> pd.Series` — dense rank ascending over non-NaN entries (higher signal → higher rank, `1..k`); NaN stays NaN.

**CSV schema (header, exact column order):**
`ticker,quarter_end,availability_date,metric_value,metric_unit,disclosure_type,segment_scope,source_url,notes`

- [ ] **Step 1: Write the failing tests**

```bash
cat > tests/grid_equipment_basket/test_backlog_data.py << 'EOF'
import pandas as pd
import pytest

from grid_equipment_basket import backlog_data as bd

_HEADER = ("ticker,quarter_end,availability_date,metric_value,metric_unit,"
           "disclosure_type,segment_scope,source_url,notes\n")


def _write(tmp_path, body: str):
    p = tmp_path / "backlog_quarterly.csv"
    p.write_text(_HEADER + body)
    return str(p)


def test_load_backlog_csv_parses_dates(tmp_path):
    path = _write(tmp_path,
        "PWR,2024-03-31,2024-05-02,30000,USD_million,xbrl_rpo,total,http://x,\n")
    df = bd.load_backlog_csv(path)
    assert pd.api.types.is_datetime64_any_dtype(df["quarter_end"])
    assert pd.api.types.is_datetime64_any_dtype(df["availability_date"])


def test_load_backlog_csv_rejects_bad_disclosure_type(tmp_path):
    path = _write(tmp_path,
        "PWR,2024-03-31,2024-05-02,30000,USD_million,guesstimate,total,http://x,\n")
    with pytest.raises(ValueError):
        bd.load_backlog_csv(path)


def test_backlog_growth_signal_yoy(tmp_path):
    body = "".join(
        f"AAA,{qe},{av},{val},USD_million,xbrl_rpo,total,http://x,\n"
        for qe, av, val in [
            ("2023-03-31", "2023-05-01", 100),
            ("2023-06-30", "2023-08-01", 105),
            ("2023-09-30", "2023-11-01", 110),
            ("2023-12-31", "2024-02-01", 120),
            ("2024-03-31", "2024-05-01", 130),
        ]
    )
    df = bd.load_backlog_csv(_write(tmp_path, body))
    sig = bd.backlog_growth_signal(df, pd.Timestamp("2024-06-01"))
    assert sig["AAA"] == pytest.approx(0.30)


def test_backlog_growth_signal_respects_availability_date(tmp_path):
    body = "".join(
        f"AAA,{qe},{av},{val},USD_million,xbrl_rpo,total,http://x,\n"
        for qe, av, val in [
            ("2023-03-31", "2023-05-01", 100),
            ("2024-03-31", "2024-05-01", 130),
        ]
    )
    df = bd.load_backlog_csv(_write(tmp_path, body))
    sig = bd.backlog_growth_signal(df, pd.Timestamp("2024-04-15"))  # before the 2024-05 filing
    assert "AAA" not in sig.dropna().index


def test_backlog_growth_signal_book_to_bill_level(tmp_path):
    body = ("BBB,2024-03-31,2024-05-01,1.20,ratio,book_to_bill_only,total,http://x,\n")
    df = bd.load_backlog_csv(_write(tmp_path, body))
    sig = bd.backlog_growth_signal(df, pd.Timestamp("2024-06-01"))
    assert sig["BBB"] == pytest.approx(0.20)


def test_backlog_ranks_ascending_higher_signal_higher_rank():
    sig = pd.Series({"A": -0.1, "B": 0.4, "C": 0.05, "D": float("nan")})
    ranks = bd.backlog_ranks(sig)
    assert ranks["B"] == 3
    assert ranks["A"] == 1
    assert pd.isna(ranks["D"])
EOF
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_backlog_data.py -q`
Expected: FAIL / ERROR — module does not exist.

- [ ] **Step 3: Write `backlog_data.py`**

```bash
cat > grid_equipment_basket/backlog_data.py << 'EOF'
from __future__ import annotations

"""Backlog / book-to-bill collection for the Step 2 weight tilt.

Structured-first: SEC XBRL companyconcept RevenueRemainingPerformanceObligation.
Everything else is hand-collected into data/backlog_quarterly.csv with an explicit
disclosure_type per row — definitions are NOT coerced to a common metric (spec §6).
"""

import io
import time

import pandas as pd
import requests

from grid_equipment_basket.config import CACHE_DIR

_SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}
_VALID_DISCLOSURE = {
    "xbrl_rpo", "nongaap_backlog_total", "nongaap_backlog_segment", "book_to_bill_only",
}
_CSV_PATH = CACHE_DIR.parent / "backlog_quarterly.csv"

# Resolved once from https://www.sec.gov/files/company_tickers.json (Task 6 Step 0).
CIK_BY_TICKER: dict[str, str] = {
    # "PWR": "0001050915", ...  # fill during implementation for the verified universe
}


def _companyconcept(cik: str, tag: str) -> pd.DataFrame:
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    r = requests.get(url, headers=_SEC_HEADERS, timeout=30)
    if r.status_code == 404:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "metric_value"])
    r.raise_for_status()
    units = r.json().get("units", {})
    rows = []
    for _unit, facts in units.items():
        for f in facts:
            end = f.get("end")
            filed = f.get("filed")
            val = f.get("val")
            if end and filed and val is not None:
                rows.append((end, filed, float(val)))
    if not rows:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "metric_value"])
    df = pd.DataFrame(rows, columns=["quarter_end", "availability_date", "metric_value"])
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])
    df["availability_date"] = pd.to_datetime(df["availability_date"])
    df = (df.sort_values("availability_date")
            .drop_duplicates("quarter_end", keep="first")
            .sort_values("quarter_end")
            .reset_index(drop=True))
    return df


def fetch_rpo(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    cache = CACHE_DIR / f"rpo_{ticker}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    cik = CIK_BY_TICKER.get(ticker)
    if cik is None:
        raise KeyError(f"no CIK for {ticker}; add it to CIK_BY_TICKER")
    df = _companyconcept(cik, "RevenueRemainingPerformanceObligation")
    if df.empty:
        cur = _companyconcept(cik, "RevenueRemainingPerformanceObligationCurrent")
        non = _companyconcept(cik, "RevenueRemainingPerformanceObligationNoncurrent")
        if not cur.empty and not non.empty:
            df = cur.merge(non, on="quarter_end", suffixes=("_c", "_n"))
            df["availability_date"] = df[["availability_date_c", "availability_date_n"]].max(axis=1)
            df["metric_value"] = df["metric_value_c"] + df["metric_value_n"]
            df = df[["quarter_end", "availability_date", "metric_value"]]
        time.sleep(0.2)
    if use_cache and not df.empty:
        df.to_parquet(cache)
    return df


def load_backlog_csv(path: str | None = None) -> pd.DataFrame:
    p = path or _CSV_PATH
    df = pd.read_csv(p, dtype={"ticker": str})
    bad = set(df["disclosure_type"]) - _VALID_DISCLOSURE
    if bad:
        raise ValueError(f"invalid disclosure_type(s): {sorted(bad)}")
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])
    df["availability_date"] = pd.to_datetime(df["availability_date"])
    return df.sort_values(["ticker", "quarter_end"]).reset_index(drop=True)


def backlog_growth_signal(df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    asof = pd.Timestamp(asof)
    vis = df[df["availability_date"] <= asof]
    out: dict[str, float] = {}
    for tkr, g in vis.groupby("ticker"):
        g = g.sort_values("quarter_end")
        if (g["disclosure_type"] == "book_to_bill_only").any():
            out[tkr] = float(g.iloc[-1]["metric_value"]) - 1.0
            continue
        if len(g) < 5:
            out[tkr] = float("nan")
            continue
        latest = float(g.iloc[-1]["metric_value"])
        year_ago = float(g.iloc[-5]["metric_value"])
        out[tkr] = (latest / year_ago - 1.0) if year_ago else float("nan")
    return pd.Series(out, dtype=float)


def backlog_ranks(signal: pd.Series) -> pd.Series:
    return signal.rank(method="dense", ascending=True)
EOF
```

- [ ] **Step 4: Resolve CIKs into `CIK_BY_TICKER`**

```bash
.venv/bin/python -c "
import json, requests
m = requests.get('https://www.sec.gov/files/company_tickers.json',
                 headers={'User-Agent': 'acadia-research sand.gh1902@gmail.com'}).json()
want = set(__import__('grid_equipment_basket.config', fromlist=['UNIVERSE']).UNIVERSE)
for row in m.values():
    if row['ticker'] in want:
        print(f\"    \\\"{row['ticker']}\\\": \\\"{int(row['cik_str']):010d}\\\",\")
"
```
Paste the printed lines into the `CIK_BY_TICKER` dict in `grid_equipment_basket/backlog_data.py`. If any verified name is missing from the SEC map (e.g. a very recent spinoff), find its CIK from its filing-index page on `sec.gov` and add it by hand.

- [ ] **Step 5: Create the CSV and populate it**

```bash
cat > grid_equipment_basket/data/backlog_quarterly.csv << 'EOF'
ticker,quarter_end,availability_date,metric_value,metric_unit,disclosure_type,segment_scope,source_url,notes
EOF
```
Then populate it: run `fetch_rpo(t)` for each verified name to seed `xbrl_rpo` rows, and hand-add `nongaap_backlog_*` / `book_to_bill_only` rows from each name's earnings releases (primary source URL required per row; do not coerce definitions). Record negative results (name discloses nothing usable) as a `notes`-only line or in `candidate_research.md`.

- [ ] **Step 6: Run the tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_backlog_data.py -q`
Expected: 6 passed.

- [ ] **Step 7: One live fetch sanity check**

Run: `.venv/bin/python -c "from grid_equipment_basket.backlog_data import fetch_rpo; print(fetch_rpo('PWR').tail(6))"`
Expected: a small frame of quarter-end / filed / value rows (or empty with a note if PWR is untagged — then rely on hand-collected rows).

- [ ] **Step 8: Commit**

```bash
git add grid_equipment_basket/backlog_data.py grid_equipment_basket/data/backlog_quarterly.csv \
        tests/grid_equipment_basket/test_backlog_data.py
git commit -m "feat(grid-equipment): SEC XBRL + hand-collected backlog data module"
```

---

## Task 7: Backlog weight tilt + tilted-vs-equal-weight comparison (GATED)

**Files:**
- Modify: `grid_equipment_basket/basket.py` (add `backlog_tilt_targets`)
- Modify: `docs/grid-equipment-basket-step1-results.md` (add a Step 2 section) and `grid_equipment_basket/README.md`
- Test: `tests/grid_equipment_basket/test_backlog_tilt.py`

**Interfaces:**
- Consumes: `grid_equipment_basket.backlog_data.backlog_growth_signal`, `backlog_ranks`; `grid_equipment_basket.basket.apply_cap`.
- Produces:
  - `backlog_tilt_targets(available: list[str], asof: pd.Timestamp, backlog_df: pd.DataFrame, cap: float = 0.25, top: float = 1.25, bottom: float = 0.75) -> pd.Series` — start from equal `1/n` over `sorted(available)`; compute `backlog_ranks(backlog_growth_signal(backlog_df, asof))` restricted to `available`; names in the top half of present ranks get `*top`, bottom half `*bottom`, the median name on an odd count and any name with no rank get `*1.0`; renormalize to sum 1; return `apply_cap(_, cap)`.

- [ ] **Step 1: Write the failing tests**

```bash
cat > tests/grid_equipment_basket/test_backlog_tilt.py << 'EOF'
import pandas as pd
import pytest

from grid_equipment_basket import basket as bk

_HEADER = ("ticker,quarter_end,availability_date,metric_value,metric_unit,"
           "disclosure_type,segment_scope,source_url,notes\n")


def _bdf(rows):
    from io import StringIO
    from grid_equipment_basket.backlog_data import load_backlog_csv
    txt = _HEADER + "".join(rows)
    p = StringIO(txt)
    return load_backlog_csv(p)


def _yoy_rows(ticker, latest, year_ago):
    qs = ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"]
    avs = ["2023-05-01", "2023-08-01", "2023-11-01", "2024-02-01", "2024-05-01"]
    vals = [year_ago, year_ago, year_ago, year_ago, latest]
    return [f"{ticker},{q},{a},{v},USD_million,xbrl_rpo,total,http://x,\n"
            for q, a, v in zip(qs, avs, vals)]


def test_tilt_overweights_top_half_underweights_bottom():
    rows = _yoy_rows("A", 200, 100) + _yoy_rows("B", 150, 100) + _yoy_rows("C", 130, 100) \
        + _yoy_rows("D", 90, 100) + _yoy_rows("E", 80, 100)
    bdf = _bdf(rows)
    w = bk.backlog_tilt_targets(list("ABCDE"), pd.Timestamp("2024-06-01"), bdf,
                                cap=1.0, top=1.25, bottom=0.75)
    assert w.sum() == pytest.approx(1.0)
    assert w["A"] > 0.2 and w["B"] > 0.2
    assert w["D"] < 0.2 and w["E"] < 0.2


def test_tilt_median_name_untilted_on_odd_count():
    rows = _yoy_rows("A", 200, 100) + _yoy_rows("B", 150, 100) + _yoy_rows("C", 130, 100) \
        + _yoy_rows("D", 90, 100) + _yoy_rows("E", 80, 100)
    bdf = _bdf(rows)
    w = bk.backlog_tilt_targets(list("ABCDE"), pd.Timestamp("2024-06-01"), bdf,
                                cap=1.0, top=1.25, bottom=0.75)
    assert w["C"] == pytest.approx(0.2 / (0.2 * (2 * 1.25 + 2 * 0.75 + 1.0)))


def test_tilt_name_without_signal_is_untilted():
    rows = _yoy_rows("A", 200, 100) + _yoy_rows("B", 80, 100)  # C has no rows
    bdf = _bdf(rows)
    w = bk.backlog_tilt_targets(list("ABC"), pd.Timestamp("2024-06-01"), bdf,
                                cap=1.0, top=1.25, bottom=0.75)
    assert w.sum() == pytest.approx(1.0)
    assert w["A"] > w["C"] > w["B"]


def test_tilt_respects_cap():
    rows = _yoy_rows("A", 500, 100) + _yoy_rows("B", 120, 100) + _yoy_rows("C", 80, 100) \
        + _yoy_rows("D", 70, 100)
    bdf = _bdf(rows)
    w = bk.backlog_tilt_targets(list("ABCD"), pd.Timestamp("2024-06-01"), bdf,
                                cap=0.25, top=1.25, bottom=0.75)
    assert w.max() <= 0.25 + 1e-9
    assert w.sum() == pytest.approx(1.0)
EOF
```

Note: `load_backlog_csv` must accept a file-like object as well as a path — adjust its
`pd.read_csv(p, ...)` call, which already works with a `StringIO`. Confirm the Step 6
implementation's `path or _CSV_PATH` handles a passed `StringIO` (it does — truthy).

- [ ] **Step 2: Run the tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_backlog_tilt.py -q`
Expected: FAIL — `backlog_tilt_targets` not defined.

- [ ] **Step 3: Append `backlog_tilt_targets` to `basket.py`**

```bash
cat >> grid_equipment_basket/basket.py << 'EOF'


def backlog_tilt_targets(
    available, asof, backlog_df,
    cap: float = 0.25, top: float = 1.25, bottom: float = 0.75,
) -> pd.Series:
    """Equal weight tilted by disclosed-backlog-growth rank (spec §6)."""
    from grid_equipment_basket.backlog_data import backlog_growth_signal, backlog_ranks

    names = sorted(available)
    if not names:
        return pd.Series(dtype=float)
    base = pd.Series(1.0 / len(names), index=names)

    ranks = backlog_ranks(backlog_growth_signal(backlog_df, asof)).reindex(names)
    present = ranks.dropna()
    factor = pd.Series(1.0, index=names)
    if len(present) >= 2:
        order = present.sort_values()
        k = len(order)
        half = k // 2
        bottom_names = order.index[:half]
        top_names = order.index[k - half:]
        factor.loc[bottom_names] = bottom
        factor.loc[top_names] = top
        # middle name(s) with odd k stay 1.0

    tilted = base * factor
    tilted = tilted / tilted.sum()
    return apply_cap(tilted, cap)
EOF
```

- [ ] **Step 4: Run the tilt tests + full package suite**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q`
Expected: all green (Task 1–7 tests).

- [ ] **Step 5: Run the tilted backtest and compare**

```bash
.venv/bin/python -m grid_equipment_basket --start 2023-01-01 --tilt backlog --output output_grid_equipment_tilt | tee output_grid_equipment_tilt/step2_tilt.txt
```
Record tilted vs equal-weight vs all five benchmarks. Per spec §6: if the tilt does not
improve **both** Sharpe and CAGR over equal-weight, equal-weight stays the recommendation.

- [ ] **Step 6: Update the results doc and README**

Add a `## Step 2 — backlog tilt` section to `docs/grid-equipment-basket-step1-results.md`
(tilted vs equal-weight table, the §6 keep/adopt decision) and update the README's
"Step 1 result" section to "Step 1 / Step 2 result" with the final recommended basket.

- [ ] **Step 7: Commit**

```bash
git add grid_equipment_basket/basket.py grid_equipment_basket/README.md \
        docs/grid-equipment-basket-step1-results.md \
        tests/grid_equipment_basket/test_backlog_tilt.py
git commit -m "feat(grid-equipment): backlog-growth weight tilt + tilted-vs-equal comparison"
```

---

## Task 8: Descriptive backlog-vs-forward-return check + finalize (GATED)

**Files:**
- Create: `grid_equipment_basket/backlog_forward_check.py`
- Test: `tests/grid_equipment_basket/test_backlog_forward_check.py`
- Modify: `docs/grid-equipment-basket-step1-results.md`, `grid_equipment_basket/README.md`

**Interfaces:**
- Produces:
  - `forward_return_table(backlog_df: pd.DataFrame, prices: pd.DataFrame, horizons=(63, 126)) -> pd.DataFrame` — for each ticker with ≥ 5 backlog observations, the Pearson correlation between as-of backlog-growth and the subsequent `h`-trading-day total return, one column per horizon, plus an `n` column. Exploratory only.
  - `plot_backlog_forward(backlog_df, prices, save_path: str) -> None` — scatter of backlog-growth vs next-quarter return, all names pooled, with per-name colour.

- [ ] **Step 1: Write the failing test**

```bash
cat > tests/grid_equipment_basket/test_backlog_forward_check.py << 'EOF'
import numpy as np
import pandas as pd

from grid_equipment_basket import backlog_forward_check as bfc


def _prices():
    idx = pd.bdate_range("2022-01-03", periods=700)
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx))))
         for t in ["AAA", "BBB"]},
        index=idx,
    )


def _backlog():
    rows = []
    for t, base in [("AAA", 100), ("BBB", 200)]:
        for k, qe in enumerate(pd.date_range("2022-03-31", periods=12, freq="QE")):
            rows.append({
                "ticker": t, "quarter_end": qe,
                "availability_date": qe + pd.Timedelta(days=35),
                "metric_value": base * (1 + 0.03 * k),
                "metric_unit": "USD_million", "disclosure_type": "xbrl_rpo",
                "segment_scope": "total", "source_url": "http://x", "notes": "",
            })
    return pd.DataFrame(rows)


def test_forward_return_table_has_row_per_name_and_horizons():
    tbl = bfc.forward_return_table(_backlog(), _prices(), horizons=(63, 126))
    assert set(tbl.index) == {"AAA", "BBB"}
    assert {"corr_63", "corr_126", "n"}.issubset(tbl.columns)
    assert (tbl["n"] > 0).all()
EOF
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_backlog_forward_check.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `backlog_forward_check.py`**

```bash
cat > grid_equipment_basket/backlog_forward_check.py << 'EOF'
from __future__ import annotations

"""Exploratory: does as-of backlog growth line up with subsequent stock return?
Descriptive only — informs whether a future cross-sectional factor (spec §8) is
worth a separate spec. Not a backtest, not an IC claim."""

import numpy as np
import pandas as pd

from grid_equipment_basket.backlog_data import backlog_growth_signal


def _asof_growth_series(backlog_df: pd.DataFrame, ticker: str) -> pd.Series:
    g = backlog_df[backlog_df["ticker"] == ticker].sort_values("availability_date")
    out = {}
    for _, row in g.iterrows():
        asof = row["availability_date"]
        sig = backlog_growth_signal(backlog_df[backlog_df["ticker"] == ticker], asof)
        if ticker in sig.index and pd.notna(sig[ticker]):
            out[asof] = float(sig[ticker])
    return pd.Series(out, dtype=float)


def forward_return_table(backlog_df: pd.DataFrame, prices: pd.DataFrame,
                         horizons=(63, 126)) -> pd.DataFrame:
    rows = {}
    for t in sorted(backlog_df["ticker"].unique()):
        if t not in prices.columns:
            continue
        growth = _asof_growth_series(backlog_df, t)
        if len(growth) < 3:
            continue
        px = prices[t].dropna()
        rec: dict = {}
        for h in horizons:
            xs, ys = [], []
            for asof, gval in growth.items():
                pos = px.index.get_indexer([asof], method="bfill")[0]
                if pos == -1 or pos + h >= len(px):
                    continue
                fwd = px.iloc[pos + h] / px.iloc[pos] - 1.0
                xs.append(gval)
                ys.append(fwd)
            rec[f"corr_{h}"] = (float(np.corrcoef(xs, ys)[0, 1])
                                if len(xs) >= 3 else np.nan)
            rec["n"] = len(xs)
        rows[t] = rec
    return pd.DataFrame(rows).T


def plot_backlog_forward(backlog_df, prices, save_path: str, horizon: int = 63) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for t in sorted(backlog_df["ticker"].unique()):
        if t not in prices.columns:
            continue
        growth = _asof_growth_series(backlog_df, t)
        px = prices[t].dropna()
        xs, ys = [], []
        for asof, gval in growth.items():
            pos = px.index.get_indexer([asof], method="bfill")[0]
            if pos == -1 or pos + horizon >= len(px):
                continue
            xs.append(gval)
            ys.append(px.iloc[pos + horizon] / px.iloc[pos] - 1.0)
        ax.scatter(xs, ys, label=t, alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("as-of backlog growth (YoY)")
    ax.set_ylabel(f"subsequent {horizon}-day return")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
EOF
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_backlog_forward_check.py -q`
Expected: 1 passed.

- [ ] **Step 5: Generate the real artifact**

```bash
.venv/bin/python -c "
from grid_equipment_basket.backlog_data import load_backlog_csv
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket import backlog_forward_check as bfc, config
bdf = load_backlog_csv()
px = fetch_prices(config.UNIVERSE, '2021-01-01', '2026-08-31')
print(bfc.forward_return_table(bdf, px))
bfc.plot_backlog_forward(bdf, px, 'output_grid_equipment_tilt/backlog_forward_check.png')
"
```
Paste the table into `docs/grid-equipment-basket-step1-results.md` under a
`## Descriptive backlog-vs-forward-return check` heading, with a one-paragraph read of
whether the Phase-2 factor (spec §8) looks worth a future spec. State plainly this is
descriptive, n is tiny, and no significance is claimed.

- [ ] **Step 6: Final README pass**

Ensure `grid_equipment_basket/README.md` states: final recommended basket (equal-weight
or tilted per §6), the gate outcome, the return-convention note, and the Phase-2 note.
Re-check the repo-root `README.md` pointer still matches.

- [ ] **Step 7: Run the whole suite once more and commit**

```bash
.venv/bin/python -m pytest tests/grid_equipment_basket/ -q
git add grid_equipment_basket/backlog_forward_check.py \
        tests/grid_equipment_basket/test_backlog_forward_check.py \
        docs/grid-equipment-basket-step1-results.md grid_equipment_basket/README.md README.md
git commit -m "feat(grid-equipment): descriptive backlog-vs-forward-return check + finalize"
```

---

## Self-Review (completed by plan author)

**1. Spec coverage**

| Spec section | Task(s) |
|---|---|
| §1 layered thesis, gate before Step 2 | 4, 5 (gate), 6–8 |
| §2 universe, business-desc test, FLNC kept, foreign excluded | 4 (+ Global Constraints) |
| §3 benchmarks XLI/SPY/XLU/GRID/PAVE | 1 (config), 3 (run) |
| §4 equal-weight, quarterly +6wk, drift, 25% cap, long-only, total-return, costs noted | 2, 3; costs note in README (5) |
| §5 windows, metrics, caveat, decision gate | 3 (metrics + `--prior-regime`), 5 (execute + gate) |
| §6 backlog data, schema, signal, tilt, re-validation, descriptive check | 6, 7, 8 |
| §7 bias mitigations | 2 (equal weight), 4 (business-desc rule, FLNC), 5 (README/results disclosures) |
| §8 Phase-2 note (not built) | README (5), results doc (8) |
| §9 module layout | matches File Structure table |
| §10 testing (offline, deterministic) | test_prices (1), test_basket (2), test_backtest (3), test_backlog_data (6), test_backlog_tilt (7), test_backlog_forward_check (8) |
| §11 deliverables incl. root-README pointer, results writeup in docs/ | 4, 5, 7, 8 |
| §12 out of scope | Global Constraints; no task adds z-scoring / build_factor / short leg / grid_resilience edits |

**2. Placeholder scan:** Research/analysis tasks (4, 5) carry explicit procedures + fill-in skeletons, not "TBD". CSV seeding in Task 6 and results-doc `<...>` markers are data-entry points with a stated "fill before committing" instruction, not unresolved design. No `add error handling`-style hand-waving.

**3. Type consistency:** `BasketResult.returns: pd.Series` / `.weights: pd.DataFrame` consumed as such in `run` (Task 3) and tests. `target_fn(available, asof)` signature is uniform: Task 2 default closure, Task 7 `backlog_tilt_targets(available, asof, backlog_df, cap, top, bottom)` wrapped to that shape by `__main__._backlog_target_fn` (Task 3). `apply_cap(weights, cap)` used in Tasks 2 and 7. `backlog_growth_signal(df, asof) -> Series` and `backlog_ranks(signal) -> Series` defined in Task 6, consumed in Tasks 7 and 8. `config.REBALANCE_LAG_DAYS` (42) used consistently — no `_WEEKS` variant anywhere.
