# Electrification Strategy v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `electrification_strategy/` — a rules-based long-only electrification/grid-equipment basket with a valuation-extension de-risk overlay and an optional GLD-sleeve + conditional DLR/EQIX short hedge, packaged as a comparison harness that prints a 12-cell variant grid and names a pitch winner.

**Architecture:** New top-level package parallel to `grid_equipment_basket/`. Reuses that package's proven primitives by import (`simulate_basket`, `compute_metrics`, `vol_target_scalar`, `find_drawdown_episode`, `episode_drawdown`). Three selectable universe constructions × four layer stacks {plain, +valuation, +valuation+sleeve, +valuation+sleeve+short}. All parameters frozen from the 2026-09-06 probe (`docs/electrification-short-leg-insurance-probe-results.md`). Daily simple-return convention throughout.

**Tech Stack:** Python 3.11/3.12, pandas, numpy, yfinance (via the reused price cache), pytest. FRED CSV endpoint for the one macro series (DFII10). matplotlib for the one output chart.

**Spec:** `docs/superpowers/specs/2026-09-06-electrification-strategy-design.md`

## Global Constraints

- Python `>=3.11,<3.13`. Simple returns (not log) everywhere — matches `grid_equipment_basket`.
- **Reuse, do not fork.** Import from `grid_equipment_basket`: `basket.simulate_basket`, `basket.apply_cap`, `backtest.compute_metrics`, `backtest.calendar_year_returns`, `data.prices.fetch_prices`, `overlay.vol_target_scalar`, `hedges.find_drawdown_episode`, `hedges.episode_drawdown`. Reimplement only the ~4-line month-hold locally (`overlay._month_hold` is private).
- **Frozen parameters (verbatim):** vol target 0.20, vol lookback 21, max leverage 1.5, rebalance lag 42 days, single-name cap 0.25, listing-min 252 trading days. Valuation: `EXT_LO=0.15, EXT_HI=0.35, RS_LO=0.25, RS_HI=0.50, MULT_MID=0.8, MULT_LOW=0.6, MA_DAYS=200, RS_DAYS=252`. Hedge: sleeve `("GLD","IEF")` weight `0.15`; short `("DLR","EQIX")` weight `0.25`, signal `DFII10`, lookback `126` days, lag `5` days. RF `0.04`, ANN `252`.
- **Winner rule (pre-registered):** among the 12 cells, highest full-window Sharpe subject to MaxDD ≤ −0.27, both sub-window (2019-22, 2023-26) Sharpe ≥ 0.4, feared-scenario P&L ≥ −0.02, full-period CAGR drag vs same-universe plain book ≤ 0.06.
- **Universe caveat (in code header + README):** membership + ETF snapshots + the `profitable_2026` flag are 2026-vintage, back-cast; only the listing-date gate is genuinely point-in-time. True point-in-time holdings/GICS vintages + time-varying profitability are a **pre-live gate**, not built in v1.
- No network in tests — monkeypatch `fetch_prices` / `fetch_series`. Synthetic fixtures only.
- Write markdown deliverables with `cat > file << 'EOF'` (per repo `CLAUDE.md`). After any code change that touches a signature/flag/default, update `electrification_strategy/README.md` in the same task.
- Commit after every task with `git add <explicit paths>` (never `git add -A`).

---

### Task 1: Package scaffold, config, test fixtures

**Files:**
- Create: `electrification_strategy/__init__.py` (empty)
- Create: `electrification_strategy/config.py`
- Create: `electrification_strategy/data/cache/.gitkeep` (empty)
- Create: `tests/electrification_strategy/__init__.py` (empty)
- Create: `tests/electrification_strategy/conftest.py`
- Test: `tests/electrification_strategy/test_config.py`

**Interfaces:**
- Produces: `electrification_strategy.config` module with all the module-level constants listed under Global Constraints, plus `MARQUEE_UNIVERSE: list[str]`, `BENCHMARKS: list[str]`, `FEARED_PROXY: str`, `EPISODES: dict[str, tuple[tuple[str,str], str]]`, `CALM_WINDOWS: dict[str, tuple[str,str]]`, `SUBWINDOWS: dict[str, tuple[str,str]]`, `VAL_PLATEAU_SCALES: tuple[float,...]`, `SUB_INDUSTRIES: set[str]`, `DATA_DIR: Path`, `CACHE_DIR: Path`, `START_DEFAULT: str`.
- Produces: pytest fixtures `synthetic_prices` (DataFrame, DatetimeIndex business days 2017-06-01..2026-08-31, columns = a fixed ticker list, geometric random walk, seed 0) and `synthetic_dfii10` (Series, daily, slowly varying 0.5→2.5).

- [ ] **Step 1: Write the failing test**

```python
# tests/electrification_strategy/test_config.py
from pathlib import Path
from electrification_strategy import config as c


def test_frozen_scalar_params():
    assert c.VOL_TARGET == 0.20
    assert c.VOL_LOOKBACK == 21
    assert c.VOL_MAX_LEVERAGE == 1.5
    assert c.REBALANCE_LAG_DAYS == 42
    assert c.MAX_SINGLE_NAME_WEIGHT == 0.25
    assert c.LISTING_MIN_DAYS == 252
    assert (c.VAL_EXT_LO, c.VAL_EXT_HI) == (0.15, 0.35)
    assert (c.VAL_RS_LO, c.VAL_RS_HI) == (0.25, 0.50)
    assert (c.VAL_MULT_MID, c.VAL_MULT_LOW) == (0.8, 0.6)
    assert c.VAL_MA_DAYS == 200 and c.VAL_RS_DAYS == 252
    assert c.HEDGE_SLEEVE_TICKERS == ("GLD", "IEF") and c.HEDGE_SLEEVE_WEIGHT == 0.15
    assert c.HEDGE_SHORT_TICKERS == ("DLR", "EQIX") and c.HEDGE_SHORT_WEIGHT == 0.25
    assert c.HEDGE_SHORT_SIGNAL == "DFII10"
    assert c.HEDGE_SHORT_LOOKBACK_DAYS == 126 and c.HEDGE_SHORT_LAG_DAYS == 5
    assert c.RF == 0.04 and c.ANN == 252


def test_frozen_collections():
    assert c.MARQUEE_UNIVERSE == ["ETN", "HUBB", "GEV", "VRT", "PWR", "MYRG", "NVT", "FLNC", "PRIM"]
    assert c.BENCHMARKS == ["VOLT", "PAVE", "GRID", "SPY", "XLI"]
    assert c.FEARED_PROXY == "TAN"
    assert set(c.EPISODES) == {"COVID 20", "Rate shock 22", "DeepSeek 25", "Tariff Apr-25", "Selloff 26"}
    for (pk, trough_end) in c.EPISODES.values():
        assert isinstance(pk, tuple) and len(pk) == 2 and isinstance(trough_end, str)
    assert set(c.SUBWINDOWS) == {"2019-22", "2023-26"}
    assert c.VAL_PLATEAU_SCALES == (0.8, 1.0, 1.2)
    assert c.SUB_INDUSTRIES == {
        "electrical_equipment", "heavy_electrical", "electrical_e&c",
        "dc_thermal", "grid_scale_storage",
    }


def test_winner_rule_constants():
    assert c.WINNER_MAXDD_MAX == -0.27
    assert c.WINNER_SUBWINDOW_SHARPE_MIN == 0.4
    assert c.WINNER_FEARED_PNL_MIN == -0.02
    assert c.WINNER_DRAG_MAX == 0.06


def test_paths_exist():
    assert isinstance(c.DATA_DIR, Path) and c.DATA_DIR.is_dir()
    assert isinstance(c.CACHE_DIR, Path) and c.CACHE_DIR.is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'electrification_strategy'`

- [ ] **Step 3: Create the package files**

```python
# electrification_strategy/__init__.py
```
(empty file)

```python
# electrification_strategy/config.py
"""Frozen configuration for the electrification strategy v1.

Every parameter here is frozen from the 2026-09-06 probe
(docs/electrification-short-leg-insurance-probe-results.md) and is NEVER revised
from results. Simple-return convention throughout.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RF: float = 0.04
ANN: int = 252
START_DEFAULT: str = "2017-06-01"

# ── basket construction ──────────────────────────────────────────────────
REBALANCE_LAG_DAYS: int = 42
MAX_SINGLE_NAME_WEIGHT: float = 0.25
LISTING_MIN_DAYS: int = 252          # a name enters only after this many trading days

MARQUEE_UNIVERSE: list[str] = ["ETN", "HUBB", "GEV", "VRT", "PWR", "MYRG", "NVT", "FLNC", "PRIM"]
SUB_INDUSTRIES: set[str] = {
    "electrical_equipment", "heavy_electrical", "electrical_e&c",
    "dc_thermal", "grid_scale_storage",
}

# ── vol target (reuses grid_equipment_basket.overlay.vol_target_scalar) ───
VOL_TARGET: float = 0.20
VOL_LOOKBACK: int = 21
VOL_MAX_LEVERAGE: float = 1.5

# ── valuation / extension de-risk overlay ────────────────────────────────
VAL_MA_DAYS: int = 200
VAL_RS_DAYS: int = 252
VAL_EXT_LO: float = 0.15
VAL_EXT_HI: float = 0.35
VAL_RS_LO: float = 0.25
VAL_RS_HI: float = 0.50
VAL_MULT_MID: float = 0.8
VAL_MULT_LOW: float = 0.6
VAL_PLATEAU_SCALES: tuple[float, ...] = (0.8, 1.0, 1.2)

# ── hedge overlay ───────────────────────────────────────────────────────
HEDGE_SLEEVE_TICKERS: tuple[str, ...] = ("GLD", "IEF")
HEDGE_SLEEVE_WEIGHT: float = 0.15
HEDGE_SHORT_TICKERS: tuple[str, ...] = ("DLR", "EQIX")
HEDGE_SHORT_WEIGHT: float = 0.25
HEDGE_SHORT_SIGNAL: str = "DFII10"
HEDGE_SHORT_LOOKBACK_DAYS: int = 126
HEDGE_SHORT_LAG_DAYS: int = 5

# ── evaluation ──────────────────────────────────────────────────────────
BENCHMARKS: list[str] = ["VOLT", "PAVE", "GRID", "SPY", "XLI"]
FEARED_PROXY: str = "TAN"

# (peak-search window), trough-end — same anchors as the probe
EPISODES: dict[str, tuple[tuple[str, str], str]] = {
    "COVID 20": (("2020-02-01", "2020-02-25"), "2020-04-30"),
    "Rate shock 22": (("2021-11-01", "2022-01-31"), "2022-12-31"),
    "DeepSeek 25": (("2024-11-01", "2025-01-15"), "2025-05-31"),
    "Tariff Apr-25": (("2025-03-15", "2025-04-03"), "2025-04-30"),
    "Selloff 26": (("2026-05-01", "2026-06-20"), "2026-08-31"),
}
CALM_WINDOWS: dict[str, tuple[str, str]] = {
    "pre-COVID 17-19": ("2017-06-01", "2019-12-31"),
    "AI-bull 23-24": ("2023-01-01", "2024-10-31"),
}
SUBWINDOWS: dict[str, tuple[str, str]] = {
    "2019-22": ("2019-01-01", "2022-12-31"),
    "2023-26": ("2023-01-01", "2026-08-31"),
}

# ── pre-registered winner rule (spec §6.3) ──────────────────────────────
WINNER_MAXDD_MAX: float = -0.27
WINNER_SUBWINDOW_SHARPE_MIN: float = 0.4
WINNER_FEARED_PNL_MIN: float = -0.02
WINNER_DRAG_MAX: float = 0.06
```

```python
# tests/electrification_strategy/__init__.py
```
(empty file)

```python
# tests/electrification_strategy/conftest.py
import numpy as np
import pandas as pd
import pytest

_TICKERS = [
    "ETN", "HUBB", "GEV", "VRT", "PWR", "MYRG", "NVT", "FLNC", "PRIM",
    "EMR", "AME", "RRX", "POWL", "ATKR", "AEIS", "GNRC", "AYI", "EME", "FIX", "MTZ",
    "GLD", "IEF", "DLR", "EQIX", "SPY", "XLI", "PAVE", "GRID", "VOLT", "TAN",
]


@pytest.fixture
def synthetic_prices():
    idx = pd.bdate_range("2017-06-01", "2026-08-31")
    rng = np.random.default_rng(0)
    out = {}
    for i, t in enumerate(_TICKERS):
        mu = 0.0003 + 0.00002 * (i % 5)
        steps = rng.normal(mu, 0.015, len(idx))
        # GEV lists late — leave the first ~1700 rows NaN
        series = 100.0 * np.exp(np.cumsum(steps))
        s = pd.Series(series, index=idx, name=t)
        if t == "GEV":
            s.iloc[:1700] = np.nan
        out[t] = s
    return pd.DataFrame(out)


@pytest.fixture
def synthetic_dfii10():
    idx = pd.bdate_range("2003-01-01", "2026-09-03")
    # slow sine 0.5 → 2.5 so 126-day changes flip sign over the sample
    x = np.linspace(0, 12 * np.pi, len(idx))
    return pd.Series(1.5 + 1.0 * np.sin(x), index=idx, name="DFII10")
```

```
# electrification_strategy/data/cache/.gitkeep
```
(empty file)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_config.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add electrification_strategy/__init__.py electrification_strategy/config.py \
        electrification_strategy/data/cache/.gitkeep \
        tests/electrification_strategy/__init__.py tests/electrification_strategy/conftest.py \
        tests/electrification_strategy/test_config.py
git commit -m "feat(electrification): package scaffold, frozen config, test fixtures"
```

---

### Task 2: FRED data module

**Files:**
- Create: `electrification_strategy/fred.py`
- Create: `electrification_strategy/data/fred_DFII10.csv` (populated in Step 5)
- Test: `tests/electrification_strategy/test_fred.py`

**Interfaces:**
- Consumes: `electrification_strategy.config` (`CACHE_DIR`, `DATA_DIR`).
- Produces: `fred.fetch_series(series_id: str, start: str, end: str, use_cache: bool = True) -> pd.Series` — DatetimeIndex, float dtype, `name == series_id`, sorted, NaNs dropped, sliced to `[start, end]`. Network hit only on cache miss; on `requests`/parse failure it loads `DATA_DIR/fred_{series_id}.csv` and emits a `RuntimeWarning`.

- [ ] **Step 1: Write the failing test**

```python
# tests/electrification_strategy/test_fred.py
import warnings

import pandas as pd
import pytest

from electrification_strategy import fred


_CSV = "observation_date,DFII10\n2020-01-02,0.15\n2020-01-03,.\n2020-01-06,0.10\n"


def test_parse_dot_as_nan_and_slice(monkeypatch, tmp_path):
    monkeypatch.setattr(fred.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fred, "_http_csv", lambda url: pd.read_csv(pd.io.common.StringIO(_CSV)))
    s = fred.fetch_series("DFII10", "2020-01-01", "2020-01-31")
    assert list(s.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06")]
    assert s.name == "DFII10"
    assert s.loc["2020-01-02"] == 0.15


def test_cache_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(fred.config, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_http(url):
        calls["n"] += 1
        return pd.read_csv(pd.io.common.StringIO(_CSV))

    monkeypatch.setattr(fred, "_http_csv", fake_http)
    fred.fetch_series("DFII10", "2020-01-01", "2020-01-31")
    fred.fetch_series("DFII10", "2020-01-01", "2020-01-31")
    assert calls["n"] == 1  # second call served from parquet cache


def test_offline_fallback_to_committed_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(fred.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fred.config, "DATA_DIR", tmp_path)
    (tmp_path / "fred_DFII10.csv").write_text(_CSV)

    def boom(url):
        raise RuntimeError("network down")

    monkeypatch.setattr(fred, "_http_csv", boom)
    with pytest.warns(RuntimeWarning):
        s = fred.fetch_series("DFII10", "2020-01-01", "2020-01-31")
    assert s.loc["2020-01-06"] == 0.10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_fred.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'electrification_strategy.fred'`

- [ ] **Step 3: Write the implementation**

```python
# electrification_strategy/fred.py
"""FRED series fetch with a parquet cache and a committed-CSV offline fallback.

Only DFII10 (10-year TIPS yield) is used in v1. DFII10 publishes next business
day with negligible revisions; the strategy applies a 5-day signal lag on top.
"""
from __future__ import annotations

import io
import warnings

import pandas as pd
import requests

from electrification_strategy import config

_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2000-01-01"


def _http_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def _clean(raw: pd.DataFrame, series_id: str) -> pd.Series:
    raw = raw.copy()
    raw.columns = ["date", "value"][: raw.shape[1]]
    idx = pd.to_datetime(raw["date"])
    val = pd.to_numeric(raw["value"], errors="coerce")  # "." -> NaN
    return pd.Series(val.to_numpy(), index=idx, name=series_id).sort_index().dropna()


def fetch_series(series_id: str, start: str, end: str, use_cache: bool = True) -> pd.Series:
    cache = config.CACHE_DIR / f"fred_{series_id}.parquet"
    if use_cache and cache.exists():
        s = pd.read_parquet(cache).iloc[:, 0]
        s.name = series_id
        return s.loc[start:end]

    try:
        s = _clean(_http_csv(_URL.format(sid=series_id)), series_id)
        if use_cache:
            s.to_frame().to_parquet(cache)
    except Exception as exc:  # noqa: BLE001 - any network/parse failure -> offline file
        fallback = config.DATA_DIR / f"fred_{series_id}.csv"
        warnings.warn(
            f"FRED fetch for {series_id} failed ({exc!r}); using committed {fallback.name}",
            RuntimeWarning, stacklevel=2,
        )
        s = _clean(pd.read_csv(fallback), series_id)
    return s.loc[start:end]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_fred.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Populate the committed offline fallback CSV**

Run:
```bash
.venv/bin/python -c "from electrification_strategy.fred import fetch_series; \
s = fetch_series('DFII10', '2003-01-01', '2026-08-31', use_cache=False); \
s.rename_axis('observation_date').to_frame().to_csv('electrification_strategy/data/fred_DFII10.csv')"
```
Expected: writes `electrification_strategy/data/fred_DFII10.csv` (~5900 rows). If the machine is offline, skip this step and note it — the module still works from a later online run.

- [ ] **Step 6: Commit**

```bash
git add electrification_strategy/fred.py tests/electrification_strategy/test_fred.py \
        electrification_strategy/data/fred_DFII10.csv
git commit -m "feat(electrification): FRED fetch with parquet cache + offline fallback"
```

---

### Task 3: Universe data files + loaders

**Files:**
- Create: `electrification_strategy/data/universe_seed.csv`
- Create: `electrification_strategy/data/etf_membership_2026.csv`
- Create: `electrification_strategy/universe.py` (loaders only in this task)
- Test: `tests/electrification_strategy/test_universe.py` (loader tests only in this task)

**Interfaces:**
- Consumes: `electrification_strategy.config` (`DATA_DIR`, `SUB_INDUSTRIES`).
- Produces:
  - `universe.load_seed() -> pd.DataFrame` — index `ticker`, columns `sub_industry` (str, all ∈ `config.SUB_INDUSTRIES`), `profitable_2026` (bool).
  - `universe.load_membership() -> pd.DataFrame` — index `ticker`, columns `["VOLT","ELFY","ZAP","GRID","PAVE"]`, bool.
  - `universe.THEMATIC_ETFS = ["VOLT", "ELFY", "ZAP", "GRID"]` (PAVE excluded from the ≥2 rule; counts only toward `frozen`'s ≥1).

- [ ] **Step 1: Write the failing test**

```python
# tests/electrification_strategy/test_universe.py
from electrification_strategy import config, universe


def test_load_seed_schema():
    df = universe.load_seed()
    assert df.index.name == "ticker"
    assert set(df.columns) == {"sub_industry", "profitable_2026"}
    assert df["sub_industry"].isin(config.SUB_INDUSTRIES).all()
    assert df["profitable_2026"].dtype == bool
    # the two known lossmakers are flagged False
    assert df.loc["FLNC", "profitable_2026"] is False or bool(df.loc["FLNC", "profitable_2026"]) is False
    assert bool(df.loc["STEM", "profitable_2026"]) is False
    # a marquee compounder is True
    assert bool(df.loc["ETN", "profitable_2026"]) is True


def test_load_membership_schema():
    m = universe.load_membership()
    assert m.index.name == "ticker"
    assert list(m.columns) == ["VOLT", "ELFY", "ZAP", "GRID", "PAVE"]
    assert m.to_numpy().dtype == bool
    assert bool(m.loc["ETN", "GRID"]) is True
    assert bool(m.loc["FLNC", "GRID"]) is False


def test_thematic_etfs_excludes_pave():
    assert universe.THEMATIC_ETFS == ["VOLT", "ELFY", "ZAP", "GRID"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_universe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'electrification_strategy.universe'`

- [ ] **Step 3: Create the data files**

```bash
cat > electrification_strategy/data/universe_seed.csv << 'EOF'
ticker,sub_industry,profitable_2026
ETN,electrical_equipment,true
HUBB,electrical_equipment,true
NVT,electrical_equipment,true
EMR,electrical_equipment,true
AME,electrical_equipment,true
RRX,electrical_equipment,true
POWL,electrical_equipment,true
ATKR,electrical_equipment,true
AEIS,electrical_equipment,true
GNRC,electrical_equipment,true
AYI,electrical_equipment,true
ENS,electrical_equipment,true
THR,electrical_equipment,true
GEV,heavy_electrical,true
VRT,dc_thermal,true
PWR,electrical_e&c,true
MYRG,electrical_e&c,true
PRIM,electrical_e&c,true
EME,electrical_e&c,true
FIX,electrical_e&c,true
MTZ,electrical_e&c,true
IESC,electrical_e&c,true
APG,electrical_e&c,true
TT,dc_thermal,true
CARR,dc_thermal,true
JCI,dc_thermal,true
SPXC,dc_thermal,true
MOD,dc_thermal,true
FLNC,grid_scale_storage,false
STEM,grid_scale_storage,false
EOF
```

```bash
cat > electrification_strategy/data/etf_membership_2026.csv << 'EOF'
# SNAPSHOT 2026-09 — approximate, derived from public issuer holdings pages.
# MUST be refreshed from issuer holdings files before any live use (see README).
# VOLT=Tema Electrification, ELFY=iShares Future Electrification, ZAP=Range Electrification,
# GRID=First Trust Clean Edge Smart Grid, PAVE=Global X US Infrastructure Development.
ticker,VOLT,ELFY,ZAP,GRID,PAVE
ETN,true,true,true,true,true
HUBB,true,true,true,true,true
GEV,true,true,true,true,true
VRT,true,true,true,false,true
PWR,true,true,true,true,true
MYRG,true,false,false,false,true
NVT,true,true,true,true,true
FLNC,true,false,false,false,false
PRIM,true,false,false,false,true
EMR,false,true,false,false,true
AME,false,true,false,true,false
RRX,false,true,false,false,true
POWL,true,false,true,false,true
ATKR,false,false,false,false,true
AEIS,true,false,false,true,false
GNRC,false,false,true,true,true
AYI,false,false,false,false,true
ENS,false,false,false,false,false
THR,false,false,false,false,false
EME,false,false,false,false,true
FIX,false,false,false,false,true
MTZ,false,false,false,false,true
IESC,false,false,false,false,false
APG,false,false,false,false,false
TT,false,false,false,false,true
CARR,false,false,false,false,true
JCI,false,false,false,false,true
SPXC,false,false,false,false,true
MOD,false,false,false,false,false
STEM,false,false,false,false,false
EOF
```

- [ ] **Step 4: Write the loaders**

```python
# electrification_strategy/universe.py
"""Universe constructions for the electrification strategy.

Three constructions, all quarterly-reconstituted with a 42-day reporting lag and
a 252-trading-day listing gate, all equal-weight with a 25% single-name cap:

  marquee  — grid_equipment_basket.config.UNIVERSE verbatim (reference)
  frozen   — our quality screen: seed pool ∩ sub-industry map ∩ (in ≥1 of 5 ETFs)
             ∩ profitable_2026
  thematic — thematic-ETF consensus: in ≥2 of {VOLT,ELFY,ZAP,GRID}, filtered to
             the sub-industry map + listing gate. No profitability screen.

CAVEAT: seed membership, the ETF snapshot, and profitable_2026 are 2026-vintage,
back-cast. Only the listing-date gate is genuinely point-in-time. True
point-in-time holdings/GICS + time-varying profitability are a pre-live gate.
"""
from __future__ import annotations

import pandas as pd

from electrification_strategy import config

THEMATIC_ETFS = ["VOLT", "ELFY", "ZAP", "GRID"]
_MEMBERSHIP_COLS = ["VOLT", "ELFY", "ZAP", "GRID", "PAVE"]


def load_seed() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / "universe_seed.csv", comment="#")
    df["profitable_2026"] = df["profitable_2026"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    )
    df = df.set_index("ticker")
    bad = set(df["sub_industry"]) - config.SUB_INDUSTRIES
    if bad:
        raise ValueError(f"universe_seed.csv has unknown sub_industry values: {sorted(bad)}")
    return df


def load_membership() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / "etf_membership_2026.csv", comment="#").set_index("ticker")
    for col in _MEMBERSHIP_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    return df[_MEMBERSHIP_COLS]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_universe.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add electrification_strategy/universe.py electrification_strategy/data/universe_seed.csv \
        electrification_strategy/data/etf_membership_2026.csv \
        tests/electrification_strategy/test_universe.py
git commit -m "feat(electrification): universe seed + ETF-membership snapshot + loaders"
```

---

### Task 4: Universe constructions (member sets + target_fn)

**Files:**
- Modify: `electrification_strategy/universe.py` (add `members`, `target_fn`)
- Test: `tests/electrification_strategy/test_universe.py` (add construction tests)

**Interfaces:**
- Consumes: `load_seed`, `load_membership`, `config.MARQUEE_UNIVERSE`, `config.LISTING_MIN_DAYS`, `config.MAX_SINGLE_NAME_WEIGHT`; `grid_equipment_basket.basket.apply_cap`.
- Produces:
  - `universe.members(construction: str, asof: pd.Timestamp, prices: pd.DataFrame) -> list[str]` — sorted eligible tickers for `construction ∈ {"marquee","frozen","thematic"}` at `asof`. A name is listed-eligible when `prices[ticker]` has ≥ `config.LISTING_MIN_DAYS` non-NaN observations at or before `asof`.
  - `universe.target_fn(construction: str, prices: pd.DataFrame) -> Callable[[list[str], pd.Timestamp], pd.Series]` — returns a `simulate_basket`-compatible target function: equal weight over `members(construction, asof, prices) ∩ available`, then `apply_cap(..., config.MAX_SINGLE_NAME_WEIGHT)`. Empty when no members.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/electrification_strategy/test_universe.py
import pandas as pd
import pytest
from grid_equipment_basket import basket as _bk


def _members(construction, prices, asof="2026-06-30"):
    return universe.members(construction, pd.Timestamp(asof), prices)


def test_marquee_is_the_nine(synthetic_prices):
    assert _members("marquee", synthetic_prices) == sorted(config.MARQUEE_UNIVERSE)


def test_frozen_drops_lossmakers_and_non_etf_names(synthetic_prices):
    m = _members("frozen", synthetic_prices)
    assert "FLNC" not in m and "STEM" not in m          # profitable_2026 == False
    assert "THR" not in m and "ENS" not in m            # in zero ETFs
    assert {"ETN", "HUBB", "NVT", "EMR", "AME", "POWL", "GEV", "VRT", "PWR"} <= set(m)


def test_thematic_is_the_consensus_subset(synthetic_prices):
    m = set(_members("thematic", synthetic_prices))
    # >=2 of VOLT/ELFY/ZAP/GRID
    assert {"ETN", "HUBB", "GEV", "VRT", "PWR", "NVT", "POWL", "AEIS", "GNRC", "AME"} <= m
    assert "EMR" not in m       # ELFY only -> 1
    assert "RRX" not in m       # ELFY only -> 1
    assert "MYRG" not in m      # VOLT only -> 1
    assert "PAVE" not in m and "ATKR" not in m   # ATKR: PAVE only, PAVE excluded from the >=2 rule


def test_listing_gate_excludes_young_names(synthetic_prices):
    # GEV has its first ~1700 rows NaN in the fixture; at 2019-06-30 it is not yet listed-eligible
    early = universe.members("frozen", pd.Timestamp("2019-06-30"), synthetic_prices)
    assert "GEV" not in early
    late = universe.members("frozen", pd.Timestamp("2026-06-30"), synthetic_prices)
    assert "GEV" in late


def test_target_fn_equal_weight_capped(synthetic_prices):
    fn = universe.target_fn("thematic", synthetic_prices)
    w = fn(list(synthetic_prices.columns), pd.Timestamp("2026-06-30"))
    assert w.sum() == pytest.approx(1.0)
    assert (w <= config.MAX_SINGLE_NAME_WEIGHT + 1e-9).all()
    assert w.nunique() == 1  # 10 names, cap not binding -> equal


def test_target_fn_empty_when_no_members(synthetic_prices):
    fn = universe.target_fn("thematic", synthetic_prices)
    w = fn(["NOTATICKER"], pd.Timestamp("2026-06-30"))
    assert w.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_universe.py -v`
Expected: FAIL — `AttributeError: module 'electrification_strategy.universe' has no attribute 'members'`

- [ ] **Step 3: Add `members` and `target_fn` to `universe.py`**

```python
# append to electrification_strategy/universe.py
from grid_equipment_basket.basket import apply_cap


def _listed_eligible(prices: pd.DataFrame, asof: pd.Timestamp) -> set[str]:
    upto = prices.loc[:asof]
    counts = upto.notna().sum()
    return set(counts[counts >= config.LISTING_MIN_DAYS].index)


def members(construction: str, asof: pd.Timestamp, prices: pd.DataFrame) -> list[str]:
    listed = _listed_eligible(prices, asof)

    if construction == "marquee":
        pool = set(config.MARQUEE_UNIVERSE)
        return sorted(pool & listed)

    seed = load_seed()
    memb = load_membership()
    in_map = set(seed.index)                       # sub_industry already validated in load_seed
    if construction == "frozen":
        any_etf = set(memb.index[memb[_MEMBERSHIP_COLS].any(axis=1)])
        profitable = set(seed.index[seed["profitable_2026"]])
        pool = in_map & any_etf & profitable
    elif construction == "thematic":
        ge2 = set(memb.index[memb[THEMATIC_ETFS].sum(axis=1) >= 2])
        pool = in_map & ge2
    else:
        raise ValueError(f"unknown construction {construction!r}")
    return sorted(pool & listed)


def target_fn(construction: str, prices: pd.DataFrame):
    def _fn(available, asof):
        elig = [t for t in members(construction, pd.Timestamp(asof), prices) if t in set(available)]
        if not elig:
            return pd.Series(dtype=float)
        raw = pd.Series(1.0 / len(elig), index=sorted(elig))
        return apply_cap(raw, config.MAX_SINGLE_NAME_WEIGHT)

    return _fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_universe.py -v`
Expected: PASS (loader tests + 6 construction tests)

- [ ] **Step 5: Commit**

```bash
git add electrification_strategy/universe.py tests/electrification_strategy/test_universe.py
git commit -m "feat(electrification): universe member sets + simulate_basket target_fn"
```

---

### Task 5: Month-hold util + valuation-extension overlay

**Files:**
- Create: `electrification_strategy/_util.py`
- Create: `electrification_strategy/valuation_overlay.py`
- Test: `tests/electrification_strategy/test_valuation_overlay.py`

**Interfaces:**
- Consumes: `electrification_strategy.config` (all `VAL_*`).
- Produces:
  - `_util.month_hold(daily: pd.Series, index: pd.DatetimeIndex, fill) -> pd.Series` — each calendar month's **last** value applied to **next** month's days; pre-first-month days = `fill`. (Local reimplementation of `grid_equipment_basket.overlay._month_hold`.)
  - `valuation_overlay.extension_multiplier(basket_index: pd.Series, spy_ret: pd.Series, scale: float = 1.0) -> pd.Series` — daily multiplier aligned to `basket_index.index`, values ∈ {`VAL_MULT_LOW`, `VAL_MULT_MID`, 1.0}, month-held, warm-up → 1.0. `basket_index` is a price **level** (cumprod of core returns, before vol-target); `spy_ret` is SPY daily returns. `scale` multiplies all four thresholds (for the plateau grid).

- [ ] **Step 1: Write the failing test**

```python
# tests/electrification_strategy/test_valuation_overlay.py
import numpy as np
import pandas as pd
import pytest

from electrification_strategy import config
from electrification_strategy._util import month_hold
from electrification_strategy.valuation_overlay import extension_multiplier


def test_month_hold_uses_prior_month_end():
    idx = pd.bdate_range("2021-01-01", "2021-03-31")
    daily = pd.Series(range(len(idx)), index=idx, dtype=float)
    held = month_hold(daily, idx, fill=1.0)
    jan_last = daily[idx.to_period("M") == pd.Period("2021-01")].iloc[-1]
    feb_days = held[idx.to_period("M") == pd.Period("2021-02")]
    assert (feb_days == jan_last).all()
    jan_days = held[idx.to_period("M") == pd.Period("2021-01")]
    assert (jan_days == 1.0).all()  # nothing before -> fill


def _flat_then_ramp(idx):
    # 2 years flat at 100, then +80% ramp over the last 6 months
    lvl = pd.Series(100.0, index=idx)
    tail = idx[idx >= idx[-126]]
    lvl.loc[tail] = np.linspace(100.0, 180.0, len(tail))
    return lvl


def test_full_multiplier_when_calm():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    lvl = pd.Series(100.0 * (1.0 + 0.00005) ** np.arange(len(idx)), index=idx)  # gentle drift
    spy = pd.Series(0.0003, index=idx)
    m = extension_multiplier(lvl, spy)
    assert m.iloc[-20:].eq(1.0).all()


def test_low_multiplier_when_stretched():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    lvl = _flat_then_ramp(idx)
    spy = pd.Series(0.0, index=idx)          # market flat -> rs12 huge
    m = extension_multiplier(lvl, spy)
    assert m.iloc[-1] == config.VAL_MULT_LOW


def test_warmup_is_full():
    idx = pd.bdate_range("2019-01-01", "2019-09-30")   # < 200 sessions
    lvl = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    spy = pd.Series(0.0, index=idx)
    m = extension_multiplier(lvl, spy)
    assert m.eq(1.0).all()


def test_scale_widens_thresholds():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    lvl = _flat_then_ramp(idx)
    spy = pd.Series(0.0, index=idx)
    tight = extension_multiplier(lvl, spy, scale=1.0).iloc[-1]
    loose = extension_multiplier(lvl, spy, scale=1.5).iloc[-1]
    assert tight == config.VAL_MULT_LOW
    assert loose >= tight  # wider thresholds -> same or higher multiplier
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_valuation_overlay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'electrification_strategy._util'`

- [ ] **Step 3: Write both modules**

```python
# electrification_strategy/_util.py
"""Small shared helpers."""
from __future__ import annotations

import pandas as pd


def month_hold(daily: pd.Series, index: pd.DatetimeIndex, fill) -> pd.Series:
    """Each calendar month's last value, applied to the *following* month's days."""
    periods = index.to_period("M")
    by_month = pd.Series(daily.to_numpy(), index=index).groupby(periods).last()
    held = by_month.shift(1).reindex(periods).to_numpy()
    return pd.Series(held, index=index).astype(float).fillna(fill)
```

```python
# electrification_strategy/valuation_overlay.py
"""Graduated 'sell-when-euphoric' de-risk multiplier (spec §4.1).

A trim, not a crash shield: ~2.7pp MaxDD improvement on the concentrated book,
~0 on the broadened book but +Sharpe. Better-behaved than a binary trend gate.
Decided at each month-end, applied to the following month (no lookahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from electrification_strategy import config
from electrification_strategy._util import month_hold


def extension_multiplier(basket_index: pd.Series, spy_ret: pd.Series, scale: float = 1.0) -> pd.Series:
    lvl = basket_index.astype(float).sort_index()
    spy_lvl = (1.0 + spy_ret.reindex(lvl.index).fillna(0.0)).cumprod()

    ext = lvl / lvl.rolling(config.VAL_MA_DAYS).mean() - 1.0
    rs12 = lvl.pct_change(config.VAL_RS_DAYS) - spy_lvl.pct_change(config.VAL_RS_DAYS)

    lo_e, hi_e = config.VAL_EXT_LO * scale, config.VAL_EXT_HI * scale
    lo_r, hi_r = config.VAL_RS_LO * scale, config.VAL_RS_HI * scale

    m = pd.Series(config.VAL_MULT_MID, index=lvl.index)
    m[(ext <= lo_e) & (rs12 <= lo_r)] = 1.0
    m[(ext > hi_e) | (rs12 > hi_r)] = config.VAL_MULT_LOW
    m[ext.isna() | rs12.isna()] = 1.0            # warm-up -> fully invested

    return month_hold(m, lvl.index, fill=1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_valuation_overlay.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add electrification_strategy/_util.py electrification_strategy/valuation_overlay.py \
        tests/electrification_strategy/test_valuation_overlay.py
git commit -m "feat(electrification): month-hold util + valuation-extension de-risk overlay"
```

---

### Task 6: Hedge overlay (sleeve + conditional short)

**Files:**
- Create: `electrification_strategy/hedge_overlay.py`
- Test: `tests/electrification_strategy/test_hedge_overlay.py`

**Interfaces:**
- Consumes: `electrification_strategy.config` (`HEDGE_*`), `electrification_strategy._util.month_hold`.
- Produces:
  - `hedge_overlay.sleeve_returns(prices: pd.DataFrame) -> pd.Series` — equal-weight daily simple return of `config.HEDGE_SLEEVE_TICKERS`, NaNs → 0.
  - `hedge_overlay.short_returns(prices: pd.DataFrame) -> pd.Series` — equal-weight daily simple return of `config.HEDGE_SHORT_TICKERS`, NaNs → 0.
  - `hedge_overlay.real_yield_rising_mask(dfii10: pd.Series, index: pd.DatetimeIndex) -> pd.Series` — bool, aligned to `index`: `(dfii10 - dfii10.shift(HEDGE_SHORT_LOOKBACK_DAYS)) > 0`, forward-filled daily, `.shift(HEDGE_SHORT_LAG_DAYS)` business days, then `month_hold` (held through the following month).
  - `hedge_overlay.apply_hedge(core_r, sleeve_r, short_r, mask, use_sleeve: bool, use_short: bool) -> pd.Series` — `(1 - w_s)·core_r + w_s·sleeve_r − w_h·(short_r·mask)`, with `w_s = HEDGE_SLEEVE_WEIGHT` if `use_sleeve` else 0, `w_h = HEDGE_SHORT_WEIGHT` if `use_short` else 0. Aligned to `core_r.index`.

- [ ] **Step 1: Write the failing test**

```python
# tests/electrification_strategy/test_hedge_overlay.py
import numpy as np
import pandas as pd
import pytest

from electrification_strategy import config, hedge_overlay


@pytest.fixture
def px():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    base = pd.Series(100.0 * (1.005) ** np.arange(len(idx)), index=idx)
    return pd.DataFrame({"GLD": base, "IEF": base * 0.5, "DLR": base * 2, "EQIX": base * 3})


def test_sleeve_returns_equal_weight(px):
    r = hedge_overlay.sleeve_returns(px)
    expect = px[["GLD", "IEF"]].pct_change().mean(axis=1).dropna()
    pd.testing.assert_series_equal(r.dropna(), expect, check_names=False)


def test_short_returns_equal_weight(px):
    r = hedge_overlay.short_returns(px)
    assert r.dropna().shape[0] == px.shape[0] - 1


def test_mask_true_only_when_yield_rose(px):
    idx = pd.bdate_range("2019-01-01", "2021-06-30")
    rising = pd.Series(np.linspace(0.5, 2.5, len(idx)), index=idx, name="DFII10")
    m = hedge_overlay.real_yield_rising_mask(rising, px.index)
    assert m.reindex(px.index).fillna(False).iloc[-1]        # yields rose over 126d -> on

    falling = pd.Series(np.linspace(2.5, 0.5, len(idx)), index=idx, name="DFII10")
    m2 = hedge_overlay.real_yield_rising_mask(falling, px.index)
    assert not m2.reindex(px.index).fillna(False).iloc[-1]


def test_apply_hedge_math(px):
    core = pd.Series(0.001, index=px.index)
    sleeve = pd.Series(0.002, index=px.index)
    short = pd.Series(0.003, index=px.index)
    mask = pd.Series(True, index=px.index)

    both = hedge_overlay.apply_hedge(core, sleeve, short, mask, use_sleeve=True, use_short=True)
    expect = (1 - 0.15) * 0.001 + 0.15 * 0.002 - 0.25 * 0.003
    assert both.iloc[0] == pytest.approx(expect)

    neither = hedge_overlay.apply_hedge(core, sleeve, short, mask, use_sleeve=False, use_short=False)
    assert neither.iloc[0] == pytest.approx(0.001)

    sleeve_only = hedge_overlay.apply_hedge(core, sleeve, short, mask, use_sleeve=True, use_short=False)
    assert sleeve_only.iloc[0] == pytest.approx((1 - 0.15) * 0.001 + 0.15 * 0.002)

    mask_off = hedge_overlay.apply_hedge(core, sleeve, short, pd.Series(False, index=px.index),
                                        use_sleeve=False, use_short=True)
    assert mask_off.iloc[0] == pytest.approx(0.001)  # short contributes nothing when mask is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_hedge_overlay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'electrification_strategy.hedge_overlay'`

- [ ] **Step 3: Write the implementation**

```python
# electrification_strategy/hedge_overlay.py
"""Hedge overlay (spec §4.2): a 15% GLD/IEF diversifier sleeve plus an optional
0.25x DLR+EQIX short that engages only while the 6-month change in the 10-year
real yield is positive.

The sleeve is the clean winner (raises Sharpe, +13% in the feared solar-squeeze
scenario). The short is a levered rising-real-yield bet — 2-name concentration,
~2.5%/yr dividend paid while short, value concentrated in the 2022 rate shock and
the DeepSeek drawdown, ~15%/yr drag inside a rate-rising equity bull.
"""
from __future__ import annotations

import pandas as pd

from electrification_strategy import config
from electrification_strategy._util import month_hold


def _ew_returns(prices: pd.DataFrame, tickers) -> pd.Series:
    cols = [t for t in tickers if t in prices.columns]
    return prices[cols].pct_change().mean(axis=1).fillna(0.0)


def sleeve_returns(prices: pd.DataFrame) -> pd.Series:
    return _ew_returns(prices, config.HEDGE_SLEEVE_TICKERS)


def short_returns(prices: pd.DataFrame) -> pd.Series:
    return _ew_returns(prices, config.HEDGE_SHORT_TICKERS)


def real_yield_rising_mask(dfii10: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    s = dfii10.astype(float).sort_index()
    rising = (s - s.shift(config.HEDGE_SHORT_LOOKBACK_DAYS)) > 0
    daily = rising.reindex(index.union(rising.index)).ffill().reindex(index).fillna(False)
    lagged = daily.shift(config.HEDGE_SHORT_LAG_DAYS).fillna(False)
    return month_hold(lagged.astype(float), index, fill=0.0) > 0.5


def apply_hedge(core_r: pd.Series, sleeve_r: pd.Series, short_r: pd.Series,
                mask: pd.Series, use_sleeve: bool, use_short: bool) -> pd.Series:
    w_s = config.HEDGE_SLEEVE_WEIGHT if use_sleeve else 0.0
    w_h = config.HEDGE_SHORT_WEIGHT if use_short else 0.0
    idx = core_r.index
    c = core_r.fillna(0.0)
    s = sleeve_r.reindex(idx).fillna(0.0)
    h = short_r.reindex(idx).fillna(0.0)
    m = mask.reindex(idx).fillna(False).astype(float)
    return (1.0 - w_s) * c + w_s * s - w_h * (h * m)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_hedge_overlay.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add electrification_strategy/hedge_overlay.py tests/electrification_strategy/test_hedge_overlay.py
git commit -m "feat(electrification): hedge overlay — GLD/IEF sleeve + conditional DLR/EQIX short"
```

---

### Task 7: Backtest core helpers

**Files:**
- Create: `electrification_strategy/backtest.py` (core helpers only in this task)
- Test: `tests/electrification_strategy/test_backtest.py` (helper tests only in this task)

**Interfaces:**
- Consumes: `grid_equipment_basket.basket.simulate_basket`, `grid_equipment_basket.backtest.compute_metrics`, `grid_equipment_basket.overlay.vol_target_scalar`, `grid_equipment_basket.hedges.find_drawdown_episode` / `.episode_drawdown`; `electrification_strategy.{config, universe, valuation_overlay}`.
- Produces:
  - `backtest.core_return(cand_prices, construction, start, end, spy_ret, use_valuation, val_scale=1.0) -> pd.Series` — daily return of the universe after equal-weight basket sim (`simulate_basket` with `universe.target_fn`), vol-target compose (`exposure = clip(val_mult · vol_scalar, ≤ VOL_MAX_LEVERAGE)`), slack at rf. `use_valuation=False` ⇒ `val_mult ≡ 1`.
  - `backtest.episode_drawdowns(returns, episodes) -> dict[str, float]` — per-episode max drawdown; `nan` when the window has no data.
  - `backtest.feared_pnl(cell_r, plain_r, tan_ret) -> float` — cumulative `(cell − plain)` monthly return over months where `plain` monthly < 0 and `tan` monthly > 0.
  - `backtest.calm_drag(cell_r, plain_r, windows) -> dict[str, float]` — per-window `CAGR(plain) − CAGR(cell)`.
  - `backtest._beta(r, mkt) -> float`, `backtest._corr(r, other) -> float`, `backtest._effective_n(weights_row) -> float` (= `1 / Σ wᵢ²`).

- [ ] **Step 1: Write the failing test**

```python
# tests/electrification_strategy/test_backtest.py
import numpy as np
import pandas as pd
import pytest

from electrification_strategy import backtest, config


def _cand(prices):
    from electrification_strategy import universe
    cand = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index))
    return prices[[c for c in cand if c in prices.columns]]


def test_core_return_runs_and_is_daily(synthetic_prices):
    spy = synthetic_prices["SPY"].pct_change()
    r = backtest.core_return(_cand(synthetic_prices), "marquee",
                             "2018-01-01", "2024-12-31", spy, use_valuation=False)
    assert isinstance(r, pd.Series)
    assert r.index.is_monotonic_increasing
    assert r.notna().mean() > 0.9
    assert r.abs().mean() < 0.1  # vol-targeted daily returns are small


def test_valuation_layer_only_reduces_or_holds_exposure(synthetic_prices):
    spy = synthetic_prices["SPY"].pct_change()
    plain = backtest.core_return(_cand(synthetic_prices), "frozen",
                                 "2018-01-01", "2024-12-31", spy, use_valuation=False)
    val = backtest.core_return(_cand(synthetic_prices), "frozen",
                               "2018-01-01", "2024-12-31", spy, use_valuation=True)
    # same index, and the valuation version is never *more* volatile over the full window
    assert val.index.equals(plain.index)
    assert val.std() <= plain.std() * 1.05


def test_episode_drawdowns_keys_and_sign():
    idx = pd.bdate_range("2019-06-01", "2026-08-31")
    lvl = pd.Series(100.0, index=idx)
    lvl.loc["2022-01-03":"2022-10-03"] = np.linspace(100, 70, len(lvl.loc["2022-01-03":"2022-10-03"]))
    lvl.loc["2022-10-04":] = 72.0
    r = lvl.pct_change().dropna()
    dd = backtest.episode_drawdowns(r, config.EPISODES)
    assert set(dd) == set(config.EPISODES)
    assert dd["Rate shock 22"] < -0.15
    assert np.isnan(dd["COVID 20"])  # no data before 2019-06 peak window


def test_feared_pnl_sign():
    idx = pd.bdate_range("2021-01-01", "2021-12-31")
    plain = pd.Series(0.0, index=idx)
    plain.loc["2021-03-01":"2021-03-31"] = -0.01     # March: plain book down
    cell = plain + 0.002                              # cell beats plain every day
    tan = pd.Series(0.0, index=idx)
    tan.loc["2021-03-01":"2021-03-31"] = 0.01         # March: TAN up -> feared month
    fp = backtest.feared_pnl(cell, plain, tan)
    assert fp > 0


def test_effective_n():
    assert backtest._effective_n(pd.Series([0.25, 0.25, 0.25, 0.25])) == pytest.approx(4.0)
    assert backtest._effective_n(pd.Series([0.5, 0.5, 0.0])) == pytest.approx(2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'electrification_strategy.backtest'`

- [ ] **Step 3: Write the core helpers**

```python
# electrification_strategy/backtest.py
"""Backtest comparison harness for the electrification strategy (spec §6).

Builds a {marquee, frozen, thematic} × {plain, +val, +val+sleeve, +val+sleeve+short}
grid, scores each cell on the pre-registered metrics, and names a pitch winner by
the spec §6.3 rule. Simple-return convention.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from grid_equipment_basket.backtest import compute_metrics
from grid_equipment_basket.basket import simulate_basket
from grid_equipment_basket.overlay import vol_target_scalar
from grid_equipment_basket import hedges as _geh

from electrification_strategy import config, hedge_overlay, universe, valuation_overlay

STACKS = [
    ("plain", False, False, False),
    ("+val", True, False, False),
    ("+val+sleeve", True, True, False),
    ("+val+sleeve+short", True, True, True),
]


def core_return(cand_prices, construction, start, end, spy_ret, use_valuation, val_scale=1.0):
    br = simulate_basket(cand_prices, start, end, config.REBALANCE_LAG_DAYS,
                         config.MAX_SINGLE_NAME_WEIGHT, universe.target_fn(construction, cand_prices))
    base_r = br.returns
    if base_r.empty:
        return base_r

    vscal = vol_target_scalar(base_r, config.VOL_LOOKBACK, config.VOL_TARGET,
                              config.VOL_MAX_LEVERAGE).reindex(base_r.index).fillna(1.0)
    if use_valuation:
        level = (1.0 + base_r).cumprod()
        vmult = valuation_overlay.extension_multiplier(
            level, spy_ret.reindex(base_r.index), scale=val_scale).reindex(base_r.index).fillna(1.0)
    else:
        vmult = pd.Series(1.0, index=base_r.index)

    exposure = (vmult * vscal).clip(upper=config.VOL_MAX_LEVERAGE)
    return exposure * base_r.fillna(0.0) + (1.0 - exposure) * config.RF / config.ANN


def episode_drawdowns(returns, episodes):
    out = {}
    for name, (peak_win, trough_end) in episodes.items():
        try:
            peak, trough = _geh.find_drawdown_episode(returns, peak_win, trough_end)
            out[name] = float(_geh.episode_drawdown(returns, peak, trough))
        except (ValueError, KeyError, IndexError):
            out[name] = float("nan")
    return out


def feared_pnl(cell_r, plain_r, tan_ret):
    cm = (1.0 + cell_r).resample("ME").prod() - 1.0
    pm = (1.0 + plain_r).resample("ME").prod() - 1.0
    if tan_ret is None or len(tan_ret) == 0:
        return float("nan")
    tm = (1.0 + tan_ret.reindex(cell_r.index).fillna(0.0)).resample("ME").prod() - 1.0
    bad = (pm < 0) & (tm.reindex(pm.index) > 0)
    return float((cm.reindex(pm.index)[bad] - pm[bad]).sum())


def calm_drag(cell_r, plain_r, windows):
    out = {}
    for name, (a, b) in windows.items():
        pc = compute_metrics(plain_r.loc[a:b], config.RF, config.ANN)["cagr"]
        cc = compute_metrics(cell_r.loc[a:b], config.RF, config.ANN)["cagr"]
        out[name] = float(pc - cc) if (pc == pc and cc == cc) else float("nan")
    return out


def _beta(r, mkt):
    d = pd.concat([r.rename("r"), mkt.rename("m")], axis=1).dropna()
    if len(d) < 2 or d["m"].var() == 0:
        return float("nan")
    return float(d["r"].cov(d["m"]) / d["m"].var())


def _corr(r, other):
    d = pd.concat([r.rename("r"), other.rename("o")], axis=1).dropna()
    return float(d["r"].corr(d["o"])) if len(d) > 2 else float("nan")


def _effective_n(weights_row):
    w = weights_row[weights_row > 0].astype(float)
    return float(1.0 / (w ** 2).sum()) if len(w) else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_backtest.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add electrification_strategy/backtest.py tests/electrification_strategy/test_backtest.py
git commit -m "feat(electrification): backtest core — universe+vol-target+valuation compose, episode/feared/drag metrics"
```

---

### Task 8: Comparison grid, table, winner rule, output writer

**Files:**
- Modify: `electrification_strategy/backtest.py` (add `run_comparison`, `comparison_table`, `pick_winner`, `write_outputs`)
- Test: `tests/electrification_strategy/test_backtest.py` (add grid/winner tests)

**Interfaces:**
- Consumes: everything from Task 7 + `config.BENCHMARKS`, `config.FEARED_PROXY`, `config.VAL_PLATEAU_SCALES`, `config.SUBWINDOWS`, `config.WINNER_*`.
- Produces:
  - `backtest.run_comparison(start, end, prices, dfii10, constructions=("marquee","frozen","thematic")) -> dict` with keys `start, end, cells, eff_n, benchmarks, plateau, plain_books`. `cells` is `{(construction, stack_label): {metrics, beta, corr_volt, episode_dd, feared_pnl, calm_drag, subwindow_sharpe, drag_vs_plain, returns}}` — 4 stack labels per construction.
  - `backtest.pick_winner(result) -> {"winner": (con,label)|None, "runner_up": (con,label)|None, "n_passing": int}` — spec §6.3 rule: MaxDD ≥ `WINNER_MAXDD_MAX`, every non-nan sub-window Sharpe ≥ `WINNER_SUBWINDOW_SHARPE_MIN`, `feared_pnl ≥ WINNER_FEARED_PNL_MIN`, `drag_vs_plain ≤ WINNER_DRAG_MAX`; among passers, max full-window Sharpe.
  - `backtest.comparison_table(result) -> str` — formatted multi-section text (cells, benchmarks, plateau).
  - `backtest.write_outputs(result, winner, out_dir, make_plot=True) -> None` — writes `metrics.csv`, `episode_drawdowns.csv`, `plateau.json`, `returns.csv`, and (unless `make_plot=False`) `performance.png`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/electrification_strategy/test_backtest.py
def test_run_comparison_grid_shape(synthetic_prices, synthetic_dfii10):
    res = backtest.run_comparison("2018-01-01", "2024-12-31", synthetic_prices, synthetic_dfii10)
    assert set(res["cells"]) == {
        (con, lab) for con in ("marquee", "frozen", "thematic")
        for lab in ("plain", "+val", "+val+sleeve", "+val+sleeve+short")
    }
    for cell in res["cells"].values():
        assert set(cell) >= {"metrics", "beta", "corr_volt", "episode_dd", "feared_pnl",
                             "calm_drag", "subwindow_sharpe", "drag_vs_plain", "returns"}
        assert set(cell["episode_dd"]) == set(config.EPISODES)
    assert {"marquee", "frozen", "thematic"} == set(res["eff_n"])
    assert res["eff_n"]["frozen"] > res["eff_n"]["marquee"]      # broadening cuts concentration
    assert len(res["plateau"]) == len(config.VAL_PLATEAU_SCALES)
    assert {"SPY", "XLI", "PAVE", "GRID", "VOLT"} <= set(res["benchmarks"])


def test_pick_winner_returns_a_cell_or_none(synthetic_prices, synthetic_dfii10):
    res = backtest.run_comparison("2018-01-01", "2024-12-31", synthetic_prices, synthetic_dfii10)
    win = backtest.pick_winner(res)
    assert win["winner"] is None or win["winner"] in res["cells"]
    assert 0 <= win["n_passing"] <= len(res["cells"])


def test_pick_winner_respects_maxdd_gate():
    # a hand-built result where only one cell clears every gate
    def cell(sharpe, maxdd, feared, drag, subs):
        return {"metrics": {"sharpe": sharpe, "max_dd": maxdd, "cagr": 0.1},
                "feared_pnl": feared, "drag_vs_plain": drag,
                "subwindow_sharpe": subs, "returns": None, "beta": 0.0, "corr_volt": 0.0,
                "episode_dd": {}, "calm_drag": {}}
    res = {"cells": {
        ("frozen", "plain"): cell(2.0, -0.40, 0.0, 0.0, {"2019-22": 1.0, "2023-26": 1.0}),      # maxdd fails
        ("frozen", "+val"): cell(0.9, -0.20, 0.0, 0.02, {"2019-22": 0.9, "2023-26": 1.2}),      # passes
        ("frozen", "+val+sleeve"): cell(1.5, -0.25, -0.10, 0.03, {"2019-22": 0.8, "2023-26": 1.0}),  # feared fails
    }}
    win = backtest.pick_winner(res)
    assert win["winner"] == ("frozen", "+val")
    assert win["n_passing"] == 1


def test_write_outputs(tmp_path, synthetic_prices, synthetic_dfii10):
    res = backtest.run_comparison("2018-01-01", "2024-12-31", synthetic_prices, synthetic_dfii10)
    win = backtest.pick_winner(res)
    backtest.write_outputs(res, win, tmp_path, make_plot=False)
    for f in ("metrics.csv", "episode_drawdowns.csv", "plateau.json", "returns.csv"):
        assert (tmp_path / f).exists()
    assert not (tmp_path / "performance.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_backtest.py -k "grid_shape or pick_winner or write_outputs" -v`
Expected: FAIL — `AttributeError: module 'electrification_strategy.backtest' has no attribute 'run_comparison'`

- [ ] **Step 3: Append the grid / reporting functions to `backtest.py`**

```python
# append to electrification_strategy/backtest.py

def run_comparison(start, end, prices, dfii10, constructions=("marquee", "frozen", "thematic")):
    spy_ret = prices["SPY"].pct_change()
    tan_ret = prices[config.FEARED_PROXY].pct_change() if config.FEARED_PROXY in prices.columns else None
    volt_ret = prices["VOLT"].pct_change() if "VOLT" in prices.columns else pd.Series(dtype=float)
    sleeve_r = hedge_overlay.sleeve_returns(prices)
    short_r = hedge_overlay.short_returns(prices)

    cand = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index))
    cand_prices = prices[[c for c in cand if c in prices.columns]]

    cells, plain_books, eff_n = {}, {}, {}
    for con in constructions:
        plain_r = core_return(cand_prices, con, start, end, spy_ret, use_valuation=False)
        plain_books[con] = plain_r
        plain_cagr = compute_metrics(plain_r, config.RF, config.ANN)["cagr"]
        mask = hedge_overlay.real_yield_rising_mask(dfii10, plain_r.index)

        br = simulate_basket(cand_prices, start, end, config.REBALANCE_LAG_DAYS,
                             config.MAX_SINGLE_NAME_WEIGHT, universe.target_fn(con, cand_prices))
        eff_n[con] = _effective_n(br.weights.iloc[-1]) if not br.weights.empty else 0.0

        for label, use_val, use_slv, use_sht in STACKS:
            core_r = core_return(cand_prices, con, start, end, spy_ret, use_valuation=use_val)
            cell_r = (core_r if not (use_slv or use_sht)
                      else hedge_overlay.apply_hedge(core_r, sleeve_r, short_r, mask, use_slv, use_sht))
            m = compute_metrics(cell_r, config.RF, config.ANN)
            cells[(con, label)] = {
                "metrics": m,
                "beta": _beta(cell_r, spy_ret),
                "corr_volt": _corr(cell_r, volt_ret),
                "episode_dd": episode_drawdowns(cell_r, config.EPISODES),
                "feared_pnl": feared_pnl(cell_r, plain_r, tan_ret),
                "calm_drag": calm_drag(cell_r, plain_r, config.CALM_WINDOWS),
                "subwindow_sharpe": {
                    k: compute_metrics(cell_r.loc[a:b], config.RF, config.ANN)["sharpe"]
                    for k, (a, b) in config.SUBWINDOWS.items()
                },
                "drag_vs_plain": float(plain_cagr - m["cagr"]) if m["cagr"] == m["cagr"] else float("nan"),
                "returns": cell_r,
            }

    benchmarks = {
        b: compute_metrics(prices[b].pct_change().loc[start:end].dropna(), config.RF, config.ANN)
        for b in config.BENCHMARKS if b in prices.columns
    }
    plateau = []
    for sc in config.VAL_PLATEAU_SCALES:
        r = core_return(cand_prices, "frozen", start, end, spy_ret, use_valuation=True, val_scale=sc)
        pm = compute_metrics(r, config.RF, config.ANN)
        plateau.append({"scale": sc, "sharpe": pm["sharpe"], "max_dd": pm["max_dd"], "cagr": pm["cagr"]})

    return {"start": start, "end": end, "cells": cells, "eff_n": eff_n,
            "benchmarks": benchmarks, "plateau": plateau, "plain_books": plain_books}


def pick_winner(result):
    passing = []
    for key, c in result["cells"].items():
        m = c["metrics"]
        subs = [s for s in c["subwindow_sharpe"].values() if s == s]
        ok = (m["max_dd"] == m["max_dd"] and m["max_dd"] >= config.WINNER_MAXDD_MAX
              and subs and all(s >= config.WINNER_SUBWINDOW_SHARPE_MIN for s in subs)
              and c["feared_pnl"] == c["feared_pnl"] and c["feared_pnl"] >= config.WINNER_FEARED_PNL_MIN
              and c["drag_vs_plain"] == c["drag_vs_plain"] and c["drag_vs_plain"] <= config.WINNER_DRAG_MAX)
        if ok:
            passing.append((key, m["sharpe"]))
    passing.sort(key=lambda kv: (kv[1] if kv[1] == kv[1] else -1e9), reverse=True)
    return {"winner": passing[0][0] if passing else None,
            "runner_up": passing[1][0] if len(passing) > 1 else None,
            "n_passing": len(passing)}


def _fmt_pct(x):
    return "  n/a" if x != x else f"{x * 100:6.1f}%"


def comparison_table(result):
    lines = [f"ELECTRIFICATION STRATEGY — {result['start']} → {result['end']}", ""]
    lines.append(f"{'cell':<28}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'beta':>7}"
                 f"{'drag':>8}{'feared':>8}{'corrVOLT':>9}")
    for (con, label), c in result["cells"].items():
        m = c["metrics"]
        lines.append(
            f"{con + ' ' + label:<28}{_fmt_pct(m['cagr'])}{_fmt_pct(m['ann_vol'])}"
            f"{(m['sharpe'] if m['sharpe'] == m['sharpe'] else float('nan')):>8.2f}"
            f"{_fmt_pct(m['max_dd'])}{c['beta']:>7.2f}{_fmt_pct(c['drag_vs_plain'])}"
            f"{_fmt_pct(c['feared_pnl'])}{c['corr_volt']:>9.2f}"
        )
    lines += ["", "effective #names: " + "  ".join(f"{k} {v:.1f}" for k, v in result["eff_n"].items()), ""]
    lines.append("benchmarks (CAGR / Sharpe / MaxDD):")
    for b, m in result["benchmarks"].items():
        lines.append(f"  {b:<6}{_fmt_pct(m['cagr'])}  {m['sharpe']:>5.2f}  {_fmt_pct(m['max_dd'])}")
    lines += ["", "valuation plateau (frozen +val, thresholds × scale):"]
    for row in result["plateau"]:
        lines.append(f"  ×{row['scale']:<4} Sharpe {row['sharpe']:>5.2f}  "
                     f"MaxDD {_fmt_pct(row['max_dd'])}  CAGR {_fmt_pct(row['cagr'])}")
    return "\n".join(lines)


def write_outputs(result, winner, out_dir, make_plot=True):
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for (con, label), c in result["cells"].items():
        m = c["metrics"]
        rows.append({"name": f"{con} {label}", "cagr": m["cagr"], "ann_vol": m["ann_vol"],
                     "sharpe": m["sharpe"], "max_dd": m["max_dd"], "beta": c["beta"],
                     "drag_vs_plain": c["drag_vs_plain"], "feared_pnl": c["feared_pnl"],
                     "corr_volt": c["corr_volt"]})
    for b, m in result["benchmarks"].items():
        rows.append({"name": f"BENCH {b}", "cagr": m["cagr"], "ann_vol": m["ann_vol"],
                     "sharpe": m["sharpe"], "max_dd": m["max_dd"]})
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    ep = pd.DataFrame({f"{con} {label}": c["episode_dd"]
                       for (con, label), c in result["cells"].items()}).T
    ep.to_csv(out / "episode_drawdowns.csv")

    (out / "plateau.json").write_text(json.dumps(
        {"winner": str(winner["winner"]), "runner_up": str(winner["runner_up"]),
         "n_passing": winner["n_passing"], "plateau": result["plateau"]}, indent=2))

    rets = pd.DataFrame({f"{con}|{label}": c["returns"]
                         for (con, label), c in result["cells"].items()
                         if c["returns"] is not None})
    rets.to_csv(out / "returns.csv")

    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})
        wkey = winner["winner"] or ("marquee", "plain")
        series = {
            "winner: " + " ".join(wkey): result["cells"][wkey]["returns"],
            "marquee plain": result["cells"][("marquee", "plain")]["returns"],
        }
        for name, r in series.items():
            if r is None:
                continue
            cc = (1 + r).cumprod()
            ax1.plot(cc.index, cc.values, lw=1.6, label=name)
        ax1.set_yscale("log"); ax1.legend(fontsize=8); ax1.set_ylabel("growth of 1 (log)")
        wr = result["cells"][wkey]["returns"]
        if wr is not None:
            cc = (1 + wr).cumprod()
            dd = cc / cc.cummax() - 1.0
            ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color="#8c2d04")
        ax2.set_ylabel("winner drawdown")
        fig.tight_layout(); fig.savefig(out / "performance.png", dpi=120); plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_backtest.py -v`
Expected: PASS (all Task 7 + Task 8 tests)

- [ ] **Step 5: Commit**

```bash
git add electrification_strategy/backtest.py tests/electrification_strategy/test_backtest.py
git commit -m "feat(electrification): comparison grid, pre-registered winner rule, table + output writer"
```

---

### Task 9: CLI

**Files:**
- Create: `electrification_strategy/__main__.py`
- Test: `tests/electrification_strategy/test_cli.py`

**Interfaces:**
- Consumes: `grid_equipment_basket.data.prices.fetch_prices`, `electrification_strategy.fred.fetch_series`, `electrification_strategy.{config, universe, backtest}`.
- Produces: `__main__.main(argv: list[str] | None = None) -> None`. Flags: `--start` (default `config.START_DEFAULT`), `--end` (default last month-end), `--universe {marquee,frozen,thematic,all}` (default `all`), `--output` (default `./output_electrification`), `--no-plot`. Fetches prices + DFII10, runs `run_comparison`, prints `comparison_table` + the winner line, calls `write_outputs`.

- [ ] **Step 1: Write the failing test**

```python
# tests/electrification_strategy/test_cli.py
import pandas as pd

from electrification_strategy import __main__ as cli


def test_cli_smoke(monkeypatch, tmp_path, synthetic_prices, synthetic_dfii10):
    monkeypatch.setattr(cli, "fetch_prices", lambda tickers, start, end: synthetic_prices)
    monkeypatch.setattr(cli, "fetch_series", lambda sid, start, end: synthetic_dfii10)

    cli.main(["--start", "2018-01-01", "--end", "2024-12-31",
              "--no-plot", "--output", str(tmp_path)])

    for f in ("metrics.csv", "episode_drawdowns.csv", "plateau.json", "returns.csv"):
        assert (tmp_path / f).exists()
    assert not (tmp_path / "performance.png").exists()
    m = pd.read_csv(tmp_path / "metrics.csv")
    assert (m["name"] == "frozen +val+sleeve+short").any()


def test_cli_single_universe(monkeypatch, tmp_path, synthetic_prices, synthetic_dfii10):
    monkeypatch.setattr(cli, "fetch_prices", lambda tickers, start, end: synthetic_prices)
    monkeypatch.setattr(cli, "fetch_series", lambda sid, start, end: synthetic_dfii10)
    cli.main(["--start", "2018-01-01", "--end", "2024-12-31", "--no-plot",
              "--universe", "marquee", "--output", str(tmp_path)])
    m = pd.read_csv(tmp_path / "metrics.csv")
    cells = [n for n in m["name"] if not n.startswith("BENCH")]
    assert all(n.startswith("marquee") for n in cells)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'electrification_strategy.__main__'`

- [ ] **Step 3: Write the CLI**

```python
# electrification_strategy/__main__.py
"""CLI: python -m electrification_strategy [options]"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from grid_equipment_basket.data.prices import fetch_prices

from electrification_strategy import backtest, config, universe
from electrification_strategy.fred import fetch_series


def _default_end() -> str:
    today = pd.Timestamp.today().normalize()
    return (today.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="electrification_strategy")
    ap.add_argument("--start", default=config.START_DEFAULT)
    ap.add_argument("--end", default=_default_end())
    ap.add_argument("--universe", choices=["marquee", "frozen", "thematic", "all"], default="all")
    ap.add_argument("--output", default="./output_electrification")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    cons = (("marquee", "frozen", "thematic") if args.universe == "all" else (args.universe,))

    cand = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index))
    tickers = sorted(set(cand) | set(config.BENCHMARKS) | set(config.HEDGE_SLEEVE_TICKERS)
                     | set(config.HEDGE_SHORT_TICKERS) | {config.FEARED_PROXY})

    prices = fetch_prices(tickers, args.start, args.end)
    dfii10 = fetch_series(config.HEDGE_SHORT_SIGNAL, "2003-01-01", args.end)

    result = backtest.run_comparison(args.start, args.end, prices, dfii10, constructions=cons)
    print(backtest.comparison_table(result))

    win = backtest.pick_winner(result)
    print(f"\nWINNER (pre-registered rule): {win['winner']}   runner-up: {win['runner_up']}   "
          f"({win['n_passing']}/{len(result['cells'])} cells pass)")

    out = Path(args.output)
    backtest.write_outputs(result, win, out, make_plot=not args.no_plot)
    extra = "" if args.no_plot else ", performance.png"
    print(f"\nwrote {out}/metrics.csv, episode_drawdowns.csv, plateau.json, returns.csv{extra}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/test_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the whole package test suite**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/ -v`
Expected: PASS (all files, ~30 tests)

- [ ] **Step 6: Commit**

```bash
git add electrification_strategy/__main__.py tests/electrification_strategy/test_cli.py
git commit -m "feat(electrification): CLI (python -m electrification_strategy)"
```

---

### Task 10: README, live run, results doc, housekeeping

**Files:**
- Create: `electrification_strategy/README.md`
- Create: `docs/electrification-strategy-v1-results.md`
- Modify: `docs/superpowers/specs/2026-09-06-electrification-strategy-design.md` (status line)
- Modify: `.gitignore` (only if `output_electrification/` / package parquet cache not already covered)

**Interfaces:** none (documentation + one real run).

- [ ] **Step 1: Ensure build artifacts are gitignored**

Run: `grep -nE 'output_|/cache/|\.parquet' .gitignore || true`
If `output_electrification/` and the package cache are not already matched by an existing
pattern, append:
```bash
cat >> .gitignore << 'EOF'
output_electrification/
electrification_strategy/data/cache/*.parquet
EOF
```
(Keep `electrification_strategy/data/cache/.gitkeep` tracked.)

- [ ] **Step 2: Write the README**

```bash
cat > electrification_strategy/README.md << 'EOF'
# Electrification Strategy (v1)

Rules-based long-only basket of US electrification / grid-equipment / data-center-power
companies, with a valuation-extension de-risk overlay and an optional hedge overlay
(GLD/short-duration sleeve + a conditional DLR·EQIX short). Honest **enhanced thematic
beta**, not alpha — close to the VOLT ETF on return, with risk control the ETF lacks.

Spec: `../docs/superpowers/specs/2026-09-06-electrification-strategy-design.md`
Results: `../docs/electrification-strategy-v1-results.md`
Research provenance: `../docs/electrification-short-leg-insurance-probe-results.md`

## Run

```
python -m electrification_strategy --start 2017-06-01 --end 2026-08-31 --output ./output_electrification
```

Prints the 12-cell comparison grid ({marquee, frozen, thematic} universes × {plain,
+valuation, +valuation+sleeve, +valuation+sleeve+short}) and the pitch winner by the
pre-registered rule. Writes `metrics.csv`, `episode_drawdowns.csv`, `plateau.json`,
`returns.csv`, `performance.png`.

## Universes

| name | inclusion | notes |
|---|---|---|
| `marquee` | `grid_equipment_basket.config.UNIVERSE` (the 9) | reference |
| `frozen` | seed pool ∩ sub-industry map ∩ (in ≥1 of VOLT/ELFY/ZAP/GRID/PAVE) ∩ `profitable_2026` | our quality screen (~20 names) |
| `thematic` | in ≥2 of {VOLT, ELFY, ZAP, GRID}, filtered to sub-industry + listing gate | thematic-ETF consensus (~10 names), no profitability screen |

All three: equal-weight, 25% single-name cap, quarterly reconstitution, 42-day reporting
lag, 252-trading-day listing gate.

## Overlays (frozen params — `config.py`)

- **Vol target** — 20% annualised, trailing 21-day realised, 1.5× cap, slack at rf
  (reuses `grid_equipment_basket.overlay.vol_target_scalar`).
- **Valuation / extension** — monthly multiplier 1.0 / 0.8 / 0.6 on `%-above-200d-MA`
  (15% / 35%) and `12-month return vs SPY` (25% / 50%). A graduated sell-when-euphoric
  trim, **not a crash shield**.
- **Hedge** — 15% NAV in EW GLD+IEF (the clean winner); optional −0.25× EW DLR+EQIX,
  engaged only while the 6-month change in DFII10 > 0 (5-day lag). The short is a levered
  rising-real-yield bet — 2 names, ~2.5%/yr dividend paid short, value concentrated in
  the 2022 and DeepSeek drawdowns, ~15%/yr drag in a rate-rising bull.

## Data & update path

- Prices: yfinance via the reused `grid_equipment_basket` monthly-parquet cache.
- DFII10: `fred.py` — FRED CSV endpoint, parquet cache, committed `data/fred_DFII10.csv`
  offline fallback (refresh the CSV monthly).
- `data/universe_seed.csv`, `data/etf_membership_2026.csv` — 2026 snapshots.

## Caveats (carry into any pitch)

- **Enhanced thematic beta, not alpha.** ~All the return is the electrification theme.
- **Universe is 2026-vintage, back-cast.** Membership, the ETF snapshot, and
  `profitable_2026` are not point-in-time; only the listing-date gate is. **Pre-live
  gate:** rebuild membership from point-in-time holdings / GICS vintages and make the
  profitability screen time-varying before any live capital.
- **Valuation overlay is a mild trim**, not a crash shield.
- **The DLR·EQIX short** carries borrow + dividend cost, 2-name concentration, and an
  ugly ride in rate-rising bull markets; its evidence is two rate episodes.
EOF
```

- [ ] **Step 3: Run the strategy for real and capture output**

Run: `.venv/bin/python -m electrification_strategy --start 2017-06-01 --end 2026-08-31 --output ./output_electrification | tee /tmp/elec_v1_run.txt`
Expected: prints the grid + winner; writes `output_electrification/`. (Needs network for prices + FRED on a cold cache.)

- [ ] **Step 4: Write the results doc from the real output**

```bash
cat > docs/electrification-strategy-v1-results.md << 'EOF'
# Electrification Strategy v1 — results

**Date:** <fill: run date>
**Spec:** docs/superpowers/specs/2026-09-06-electrification-strategy-design.md
**Command:** `python -m electrification_strategy --start 2017-06-01 --end 2026-08-31`

## The 12-cell grid

<paste the `comparison_table` output from /tmp/elec_v1_run.txt verbatim>

## Winner (pre-registered rule, spec §6.3)

<paste the WINNER line. State winner + runner-up in prose. If zero cells pass, say so and
report the cell with the best Sharpe among those clearing MaxDD ≤ −27% as the fallback
pitch candidate, explicitly labelled as not clearing the full bar.>

## Valuation plateau

<paste the plateau block — confirm the ×0.8 / ×1.0 / ×1.2 rows are close (not knife-edge)>

## Caveats

- Universe is 2026-vintage back-cast; pre-live gate = point-in-time membership + time-varying
  profitability (spec §1 non-goals, §3.3).
- Valuation overlay is a mild trim; the DLR·EQIX short's evidence is two rate episodes.
- One macro cycle for the part that reduces drawdowns; structural logic, not proof.
EOF
```
Then edit the `<...>` placeholders with the real captured output.

- [ ] **Step 5: Flip the spec status to BUILT**

Edit `docs/superpowers/specs/2026-09-06-electrification-strategy-design.md` line 4:
`**Status:** BUILT 2026-09-06 — results: docs/electrification-strategy-v1-results.md`

- [ ] **Step 6: Full test sweep + commit**

Run: `.venv/bin/python -m pytest tests/electrification_strategy/ -q`
Expected: all green.

```bash
git add electrification_strategy/README.md docs/electrification-strategy-v1-results.md \
        docs/superpowers/specs/2026-09-06-electrification-strategy-design.md .gitignore
git commit -m "docs(electrification): README, v1 results, spec marked BUILT"
```

---

## Self-Review

**1. Spec coverage**

| spec section | task(s) |
|---|---|
| §2.1 package layout | 1 (scaffold), files created across 2–9 |
| §2.2 reuse by import | 4, 5, 7, 9 (imports from `grid_equipment_basket`); local month-hold in 5 |
| §2.3 layer composition | 7 (`core_return`), 6 + 8 (hedge application) |
| §3 marquee / U1 frozen / U2 thematic | 3 (data + loaders), 4 (`members`, `target_fn`) |
| §3.1 seed pool | 3 (`universe_seed.csv`) |
| §3.2 fundamentals + ETF snapshots + fallbacks | 3 (`profitable_2026` flag = the documented backstop; `etf_membership_2026.csv` with the ≥1 / ≥2 rule) |
| §3.3 honesty caveat | 3 (module docstring), 10 (README) |
| §4.1 valuation overlay + plateau | 5 (`extension_multiplier`, `scale`), 8 (`plateau`) |
| §4.2 hedge overlay (sleeve + conditional short) | 6 |
| §5 FRED module | 2 |
| §6.1 `run_comparison` grid + metrics | 8 |
| §6.2 CLI + outputs | 8 (`write_outputs`), 9 (`__main__`) |
| §6.3 winner rule | 8 (`pick_winner`) |
| §7 tests | 1, 2, 3, 4, 5, 6, 7, 8, 9 (each task ships its test file) |
| §8 deliverables (README, results doc, spec status) | 10 |

No gaps.

**2. Placeholder scan** — the only `<...>` markers are in Task 10 Step 4, which is explicitly
"paste real captured output" (a live run produces it; the template + paste instruction is
concrete, not a deferred decision). All code steps contain full implementations.

**3. Type consistency** — checked: `target_fn(construction, prices)` returns `(available, asof) -> pd.Series`
matching `simulate_basket`'s `target_fn` contract; `core_return(...)` signature identical in
Tasks 7 and 8; `pick_winner` returns keys of the shape `(construction, label)` that index
`result["cells"]`; `run_comparison` return dict keys (`cells, eff_n, benchmarks, plateau,
plain_books`) match what `comparison_table` / `write_outputs` / `pick_winner` read; `month_hold`
signature identical in `_util.py` and both call sites; `_effective_n` used in Task 7 tests and
Task 8 `run_comparison` with the same `1/Σwᵢ²` definition.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-09-06-electrification-strategy.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
