# Transmission Rate-Base Compounders — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a gated, sector-neutral equity factor over ~40 US regulated electric utilities that goes long the fast FERC transmission-rate-base compounders and short the flat ones, and evaluate it against a pre-registered pass/fail bar.

**Architecture:** A new top-level package `transmission_rate_base/` mirroring `backlog_factor/` and `grid_equipment_basket/`. A data layer pulls FERC Form 1 tables from PUDL's public parquet mirror and caches them locally; a signal layer aggregates filer-level transmission plant to parent tickers and computes a growth signal; a portfolio layer builds a dollar/beta-neutral quintile long/short book (plus a long-only tilt); a backtest + additivity layer produces the metrics; a gate layer applies the pre-registered pass/fail rule. The whole pipeline runs to a verdict from local cache alone (`--offline`) because yfinance is rate-limiting.

**Tech Stack:** Python 3.11, pandas, numpy, statsmodels (Newey-West OLS), pyarrow (parquet), pytest. Reuses `grid_resilience.data.equity_prices` (price cache), `grid_resilience.portfolio.backtest.compute_ic`, `grid_resilience.factor.neutralize`.

**Spec:** `docs/superpowers/specs/2026-08-31-transmission-rate-base-design.md`

## Global Constraints

- **Package path:** `transmission_rate_base/` (top-level, sibling of `grid_resilience/`, `backlog_factor/`).
- **Tests path:** `tests/transmission_rate_base/`.
- **Python:** `>=3.11,<3.13`. `from __future__ import annotations` at the top of every module (matches repo).
- **No new dependencies.** Everything is already in `pyproject.toml` (pandas, numpy, statsmodels, scipy, pyarrow).
- **Markdown files:** write with Bash heredoc, never the Write tool (repo CLAUDE.md rule).
- **PUDL parquet base URL:** `https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/` — verbatim.
- **Signal is never revised from results.** Universe list, signal definition, windows, and gate thresholds are frozen once written. The §6 sensitivity variants are for the writeup only.
- **Pre-registered gate thresholds (copy verbatim into `config.py`):** mean rank-IC ≥ 0.03; rank-IC t ≥ 2.0; Q5−Q1 annualized Sharpe ≥ 0.40; ≤ 1 adjacent quintile inversion; ≥ 60% of calendar years positive; residual alpha ≥ 0.03/yr; residual-alpha t ≥ 2.0; additivity R² < 0.60; pre-thesis one-sided t > −2.0.
- **Primary window:** 2011-01-01 → 2025-12-31. **Pre-thesis window:** 2003-01-01 → 2010-12-31.
- **Rebalance:** first trading day of May, annually. Signal at May-Y uses only `report_year ≤ Y-1`.
- **Outputs:** `output_transmission_rate_base/` (gitignored).
- **Commit after every task.** Conventional-commit messages, prefix `feat(transmission-rate-base):` / `test(...)` / `chore(...)`. End commit messages with the repo's `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` trailer.
- **Full repo test suite (`.venv/bin/python -m pytest -q`) must stay green after every task.**

---

## File Structure

| File | Responsibility |
|---|---|
| `transmission_rate_base/__init__.py` | Package docstring only. |
| `transmission_rate_base/config.py` | All frozen constants: PUDL URL + table names, universe seed list, windows, rebalance month, quintile count, winsor sigma, structural-break threshold, beta window/clip, **all gate thresholds**. |
| `transmission_rate_base/data/__init__.py` | Empty. |
| `transmission_rate_base/data/ferc_form1.py` | Fetch the 4 PUDL tables over HTTPS, cache each as parquet under `data/cache/`, offline fallback; **and** extraction functions turning raw schedules into `(utility_id_ferc1, report_year, value)` frames for gross/net transmission plant and net total plant. |
| `transmission_rate_base/data/utility_map.py` | `PARENT_FILERS` static dict (ticker → list of `utility_name_ferc1` strings) + `resolve_filers(crosswalk_df)` → `dict[str, list[int]]` + validation helpers. |
| `transmission_rate_base/data/segment_mix.csv` | Hand-collected: `ticker,fy,reg_elec_op_rev,total_op_rev`. |
| `transmission_rate_base/data/style_inputs.csv` | Hand-collected annual gap-fill: `ticker,fy,dividend_yield,market_cap,book_value,capex,net_ppe`. |
| `transmission_rate_base/data/prices.py` | Thin re-point of `grid_resilience.data.equity_prices` — `monthly_returns(tickers,start,end,offline)`, `daily_returns(...)`. |
| `transmission_rate_base/signal.py` | `build_parent_panel(...)`, `primary_signal(...)`, `neutralize(...)`, `load_segment_mix(...)`. |
| `transmission_rate_base/portfolio.py` | `rebalance_dates(...)`, `quintile_ls_weights(...)`, `long_tilt_weights(...)`. |
| `transmission_rate_base/backtest.py` | `run_backtest(...)` (monthly P&L + metrics), `signal_rank_ic(...)` (wraps `compute_ic`). |
| `transmission_rate_base/additivity.py` | `build_controls(...)`, `run_additivity(...)` (statsmodels OLS + HAC). |
| `transmission_rate_base/gate.py` | `evaluate_gate(ic, spread, additivity, pre_thesis)` → verdict dict. |
| `transmission_rate_base/report.py` | `run_pipeline(...)` orchestrator: writes `output_transmission_rate_base/*`, returns the verdict. |
| `transmission_rate_base/__main__.py` | argparse CLI → `report.run_pipeline`. |
| `tests/transmission_rate_base/test_*.py` | One test module per source module. |

---

## Task 1: Package scaffold + frozen config

**Files:**
- Create: `transmission_rate_base/__init__.py`
- Create: `transmission_rate_base/config.py`
- Create: `transmission_rate_base/data/__init__.py`
- Create: `tests/transmission_rate_base/__init__.py`
- Create: `tests/transmission_rate_base/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `transmission_rate_base.config` module exposing (exact names/types):
  - `PUDL_BASE: str`
  - `FERC_TABLES: dict[str, str]` keys `plant_in_service`, `dep_by_function`, `plant_summary`, `utility_xwalk`; values are PUDL table names.
  - `UNIVERSE_SEED: tuple[str, ...]`
  - `PRIMARY_START, PRIMARY_END, PRE_THESIS_START, PRE_THESIS_END: str`
  - `REBALANCE_MONTH: int` (= 5)
  - `SIGNAL_CAGR_YEARS: int` (= 3), `MIN_HISTORY_YEARS: int` (= 4)
  - `QUINTILES: int` (= 5), `WINSOR_SIGMA: float` (= 2.5)
  - `STRUCTURAL_BREAK_LOG: float` (= `math.log(2.0)`)
  - `BETA_WINDOW: int` (= 252), `BETA_CLIP: tuple[float, float]` (= `(0.5, 2.0)`)
  - `GATE: dict[str, float]` with keys `min_mean_ic, min_ic_t, min_ls_sharpe, max_quintile_inversions, min_positive_years_frac, min_alpha_ann, min_alpha_t, max_additivity_r2, pre_thesis_min_t`.

- [ ] **Step 1: Write the failing test**

Create `tests/transmission_rate_base/__init__.py` (empty) and `tests/transmission_rate_base/test_config.py`:

```python
import math

from transmission_rate_base import config as c


def test_gate_thresholds_match_spec():
    assert c.GATE["min_mean_ic"] == 0.03
    assert c.GATE["min_ic_t"] == 2.0
    assert c.GATE["min_ls_sharpe"] == 0.40
    assert c.GATE["max_quintile_inversions"] == 1
    assert c.GATE["min_positive_years_frac"] == 0.60
    assert c.GATE["min_alpha_ann"] == 0.03
    assert c.GATE["min_alpha_t"] == 2.0
    assert c.GATE["max_additivity_r2"] == 0.60
    assert c.GATE["pre_thesis_min_t"] == -2.0


def test_windows_and_signal_constants():
    assert c.PRIMARY_START == "2011-01-01"
    assert c.PRIMARY_END == "2025-12-31"
    assert c.PRE_THESIS_START == "2003-01-01"
    assert c.PRE_THESIS_END == "2010-12-31"
    assert c.REBALANCE_MONTH == 5
    assert c.SIGNAL_CAGR_YEARS == 3
    assert c.MIN_HISTORY_YEARS == 4
    assert c.STRUCTURAL_BREAK_LOG == math.log(2.0)


def test_ferc_table_keys_and_pudl_base():
    assert c.PUDL_BASE == "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/"
    assert set(c.FERC_TABLES) == {"plant_in_service", "dep_by_function", "plant_summary", "utility_xwalk"}
    assert c.FERC_TABLES["plant_in_service"] == "core_ferc1__yearly_plant_in_service_sched204"


def test_universe_seed_is_nonempty_unique_tuple():
    assert isinstance(c.UNIVERSE_SEED, tuple)
    assert len(c.UNIVERSE_SEED) >= 30
    assert len(set(c.UNIVERSE_SEED)) == len(c.UNIVERSE_SEED)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'transmission_rate_base'`.

- [ ] **Step 3: Write minimal implementation**

`transmission_rate_base/__init__.py`:

```python
"""Transmission rate-base compounders — proposal B.

A gated, sector-neutral factor over US regulated electric utilities: long the
fast FERC transmission-rate-base compounders, short the flat ones. See
docs/superpowers/specs/2026-08-31-transmission-rate-base-design.md.
"""
```

`transmission_rate_base/data/__init__.py`: empty file.

`transmission_rate_base/config.py`:

```python
from __future__ import annotations

import math

PUDL_BASE = "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/"

FERC_TABLES: dict[str, str] = {
    "plant_in_service": "core_ferc1__yearly_plant_in_service_sched204",
    "dep_by_function":  "core_ferc1__yearly_depreciation_by_function_sched219",
    "plant_summary":    "core_ferc1__yearly_utility_plant_summary_sched200",
    "utility_xwalk":    "core_pudl__assn_ferc1_pudl_utilities",
}

# Frozen seed universe (finalised in data/utility_map.py). US-listed regulated
# electric + electric-heavy multi-utilities that file FERC Form 1.
UNIVERSE_SEED: tuple[str, ...] = (
    "AEP", "AEE", "AES", "AVA", "BKH", "CMS", "CNP", "D", "DTE", "DUK",
    "ED", "EIX", "ES", "ETR", "EVRG", "EXC", "FE", "HE", "IDA", "LNT",
    "MGEE", "NEE", "NWE", "OGE", "OTTR", "PCG", "PEG", "PNW", "POR", "PPL",
    "SO", "TXNM", "WEC", "XEL",
)

PRIMARY_START, PRIMARY_END = "2011-01-01", "2025-12-31"
PRE_THESIS_START, PRE_THESIS_END = "2003-01-01", "2010-12-31"

REBALANCE_MONTH = 5
SIGNAL_CAGR_YEARS = 3
MIN_HISTORY_YEARS = 4          # need Y-3..Y inclusive for one 3-yr growth figure
QUINTILES = 5
WINSOR_SIGMA = 2.5
STRUCTURAL_BREAK_LOG = math.log(2.0)
BETA_WINDOW = 252
BETA_CLIP = (0.5, 2.0)

GATE: dict[str, float] = {
    "min_mean_ic": 0.03,
    "min_ic_t": 2.0,
    "min_ls_sharpe": 0.40,
    "max_quintile_inversions": 1,
    "min_positive_years_frac": 0.60,
    "min_alpha_ann": 0.03,
    "min_alpha_t": 2.0,
    "max_additivity_r2": 0.60,
    "pre_thesis_min_t": -2.0,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_config.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add transmission_rate_base/__init__.py transmission_rate_base/config.py \
        transmission_rate_base/data/__init__.py tests/transmission_rate_base/
git commit -m "feat(transmission-rate-base): package scaffold + frozen config

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 2: FERC Form 1 fetch + local parquet cache

**Files:**
- Create: `transmission_rate_base/data/ferc_form1.py`
- Create: `tests/transmission_rate_base/test_ferc_form1.py`

**Interfaces:**
- Consumes: `config.PUDL_BASE`, `config.FERC_TABLES`.
- Produces:
  - `CACHE_DIR: pathlib.Path` (= `transmission_rate_base/data/cache`).
  - `fetch_table(key: str, *, refresh: bool = False, offline: bool = False) -> pandas.DataFrame`
    where `key` is a `config.FERC_TABLES` key. Reads `CACHE_DIR/<tablename>.parquet` when present (unless `refresh`); otherwise downloads `PUDL_BASE + <tablename> + ".parquet"`, writes the cache, returns it. `offline=True` forbids the network: use cache or raise `FileNotFoundError`.
  - `_read_remote(url: str) -> pandas.DataFrame` — thin `pd.read_parquet(url)` seam so tests can monkeypatch it.

- [ ] **Step 1: Write the failing test**

`tests/transmission_rate_base/test_ferc_form1.py`:

```python
import pandas as pd
import pytest

from transmission_rate_base.data import ferc_form1 as f1


def test_offline_uses_cache_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(f1, "CACHE_DIR", tmp_path)
    cached = pd.DataFrame({"utility_id_ferc1": [1], "report_year": [2020]})
    (tmp_path / "core_ferc1__yearly_plant_in_service_sched204.parquet").write_bytes(b"")
    cached.to_parquet(tmp_path / "core_ferc1__yearly_plant_in_service_sched204.parquet")

    def _boom(url):  # network must not be touched
        raise AssertionError(f"network hit for {url}")
    monkeypatch.setattr(f1, "_read_remote", _boom)

    got = f1.fetch_table("plant_in_service", offline=True)
    pd.testing.assert_frame_equal(got, cached)


def test_offline_without_cache_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(f1, "CACHE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        f1.fetch_table("dep_by_function", offline=True)


def test_download_writes_cache_then_reuses_it(tmp_path, monkeypatch):
    monkeypatch.setattr(f1, "CACHE_DIR", tmp_path)
    payload = pd.DataFrame({"utility_id_ferc1": [7, 7], "report_year": [2019, 2020]})
    calls = []
    def _fake_remote(url):
        calls.append(url)
        return payload
    monkeypatch.setattr(f1, "_read_remote", _fake_remote)

    first = f1.fetch_table("plant_summary")
    second = f1.fetch_table("plant_summary")
    pd.testing.assert_frame_equal(first, payload)
    pd.testing.assert_frame_equal(second, payload)
    assert len(calls) == 1
    assert calls[0].endswith("core_ferc1__yearly_utility_plant_summary_sched200.parquet")


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        f1.fetch_table("not_a_table")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_ferc_form1.py -q`
Expected: FAIL — `ModuleNotFoundError: transmission_rate_base.data.ferc_form1`.

- [ ] **Step 3: Write minimal implementation**

`transmission_rate_base/data/ferc_form1.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from transmission_rate_base import config

CACHE_DIR = Path(__file__).parent / "cache"


def _read_remote(url: str) -> pd.DataFrame:
    """Isolated network read so tests can monkeypatch it."""
    return pd.read_parquet(url)


def _cache_path(table_name: str) -> Path:
    return CACHE_DIR / f"{table_name}.parquet"


def fetch_table(key: str, *, refresh: bool = False, offline: bool = False) -> pd.DataFrame:
    table_name = config.FERC_TABLES[key]  # KeyError on unknown key
    path = _cache_path(table_name)

    if path.exists() and not refresh:
        return pd.read_parquet(path)

    if offline:
        raise FileNotFoundError(
            f"offline=True and no cache at {path}; run once online or with --refresh-ferc"
        )

    df = _read_remote(config.PUDL_BASE + table_name + ".parquet")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_ferc_form1.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add transmission_rate_base/data/ferc_form1.py tests/transmission_rate_base/test_ferc_form1.py
git commit -m "feat(transmission-rate-base): FERC Form 1 PUDL fetch + parquet cache

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 3: FERC Form 1 extraction (gross / net transmission plant, total plant)

**Files:**
- Modify: `transmission_rate_base/data/ferc_form1.py`
- Modify: `tests/transmission_rate_base/test_ferc_form1.py`

**Interfaces:**
- Consumes: raw frames from `fetch_table`.
- Produces (all return a tidy DataFrame with columns `["utility_id_ferc1", "report_year", <value_col>]`, one row per filer-year, `int` ids/years, `float` values):
  - `gross_transmission_plant(sched204: pd.DataFrame) -> pd.DataFrame` — value col `gross_tx`. Uses the labelled subtotal row `ferc_account_label == "transmission_plant"` with `row_type_xbrl` in `{"total", None, "nan"}`; falls back to summing leaf `*_transmission_plant` accounts (excluding any label containing `regional_transmission_and_market_operation`) when the subtotal row is absent for a filer-year.
  - `transmission_accum_depreciation(sched219: pd.DataFrame) -> pd.DataFrame` — value col `accum_dep_tx`. Rows where `plant_function == "transmission"` and `depreciation_type` in the accumulated-provision set; `ending_balance`.
  - `total_net_utility_plant(sched200: pd.DataFrame) -> pd.DataFrame` — value col `net_total`. `utility_plant_in_service` minus `accumulated_depreciation` (or the labelled `utility_plant_net` row if present).
  - `net_transmission_plant(sched204, sched219, sched200) -> pd.DataFrame` — value col `net_tx` = `gross_tx - accum_dep_tx`; where `accum_dep_tx` is missing for a filer-year, fall back to `gross_tx * (net_total / gross_total)` using `sched200` gross/net, and set a boolean column `net_tx_prorated`.

- [ ] **Step 1: Write the failing test**

Append to `tests/transmission_rate_base/test_ferc_form1.py`:

```python
def _sched204_rows():
    # filer 1, 2020: has an explicit subtotal (100) AND leaves (60+30=90) -> prefer subtotal
    # filer 2, 2020: leaves only (40 + 25 = 65), plus an RTO row that must be ignored (999)
    return pd.DataFrame([
        dict(utility_id_ferc1=1, report_year=2020, ferc_account_label="transmission_plant",
             row_type_xbrl="total", ending_balance=100.0),
        dict(utility_id_ferc1=1, report_year=2020, ferc_account_label="towers_and_fixtures_transmission_plant",
             row_type_xbrl=None, ending_balance=60.0),
        dict(utility_id_ferc1=1, report_year=2020, ferc_account_label="station_equipment_transmission_plant",
             row_type_xbrl=None, ending_balance=30.0),
        dict(utility_id_ferc1=2, report_year=2020, ferc_account_label="overhead_conductors_and_devices_transmission_plant",
             row_type_xbrl=None, ending_balance=40.0),
        dict(utility_id_ferc1=2, report_year=2020, ferc_account_label="poles_and_fixtures_transmission_plant",
             row_type_xbrl=None, ending_balance=25.0),
        dict(utility_id_ferc1=2, report_year=2020,
             ferc_account_label="communication_equipment_regional_transmission_and_market_operation_plant",
             row_type_xbrl=None, ending_balance=999.0),
    ])


def test_gross_transmission_prefers_subtotal_and_sums_leaves_and_drops_rto():
    out = f1.gross_transmission_plant(_sched204_rows()).set_index("utility_id_ferc1")["gross_tx"]
    assert out.loc[1] == 100.0     # subtotal wins over 60+30
    assert out.loc[2] == 65.0      # 40+25, RTO 999 excluded


def test_net_transmission_prorates_when_depreciation_missing():
    sched204 = pd.DataFrame([
        dict(utility_id_ferc1=9, report_year=2021, ferc_account_label="transmission_plant",
             row_type_xbrl="total", ending_balance=200.0),
    ])
    sched219 = pd.DataFrame(columns=["utility_id_ferc1", "report_year", "plant_function",
                                     "depreciation_type", "ending_balance"])
    sched200 = pd.DataFrame([
        dict(utility_id_ferc1=9, report_year=2021, ferc_account_label="utility_plant_in_service",
             ending_balance=1000.0),
        dict(utility_id_ferc1=9, report_year=2021, ferc_account_label="accumulated_depreciation",
             ending_balance=400.0),
    ])
    out = f1.net_transmission_plant(sched204, sched219, sched200).set_index("utility_id_ferc1")
    # gross_total 1000, net_total 600 -> ratio 0.6 -> net_tx = 200 * 0.6 = 120
    assert out.loc[9, "net_tx"] == 120.0
    assert bool(out.loc[9, "net_tx_prorated"]) is True
```

(Adjust `total_net_utility_plant` label handling in the implementation until these pass; add a direct test for it with an explicit `utility_plant_net` row too.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_ferc_form1.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'gross_transmission_plant'`.

- [ ] **Step 3: Write minimal implementation**

Append to `transmission_rate_base/data/ferc_form1.py`:

```python
_RTO_MARKER = "regional_transmission_and_market_operation"
_ACCUM_DEP_TYPES = {
    "accumulated_provision_for_depreciation",
    "accumulated_depreciation",
    "depreciation_amortization_and_depletion",
}


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["utility_id_ferc1"] = df["utility_id_ferc1"].astype(int)
    df["report_year"] = df["report_year"].astype(int)
    return df


def gross_transmission_plant(sched204: pd.DataFrame) -> pd.DataFrame:
    df = _norm(sched204)
    label = df["ferc_account_label"].fillna("").str.lower()
    df = df[~label.str.contains(_RTO_MARKER)]
    label = df["ferc_account_label"].fillna("").str.lower()

    is_subtotal = label.eq("transmission_plant")
    row_type = df.get("row_type_xbrl")
    if row_type is not None:
        is_subtotal &= row_type.isna() | row_type.astype(str).str.lower().eq("total")

    subtotals = (df[is_subtotal]
                 .groupby(["utility_id_ferc1", "report_year"])["ending_balance"].sum()
                 .rename("gross_tx"))

    leaves = df[label.str.endswith("_transmission_plant") & ~label.eq("transmission_plant")]
    leaf_sum = (leaves.groupby(["utility_id_ferc1", "report_year"])["ending_balance"].sum()
                .rename("gross_tx"))

    out = subtotals.combine_first(leaf_sum).reset_index()
    return out[["utility_id_ferc1", "report_year", "gross_tx"]]


def transmission_accum_depreciation(sched219: pd.DataFrame) -> pd.DataFrame:
    if sched219.empty:
        return pd.DataFrame(columns=["utility_id_ferc1", "report_year", "accum_dep_tx"])
    df = _norm(sched219)
    func = df["plant_function"].fillna("").str.lower()
    dep = df["depreciation_type"].fillna("").str.lower()
    keep = func.eq("transmission") & (dep.isin(_ACCUM_DEP_TYPES) | dep.eq(""))
    out = (df[keep].groupby(["utility_id_ferc1", "report_year"])["ending_balance"].sum()
           .rename("accum_dep_tx").reset_index())
    return out


def total_net_utility_plant(sched200: pd.DataFrame) -> pd.DataFrame:
    df = _norm(sched200)
    label = df["ferc_account_label"].fillna("").str.lower()
    piv = (df.assign(_l=label)
           .pivot_table(index=["utility_id_ferc1", "report_year"], columns="_l",
                        values="ending_balance", aggfunc="sum"))
    def col(name):
        return piv[name] if name in piv.columns else pd.Series(index=piv.index, dtype=float)
    net = col("utility_plant_net")
    gross = col("utility_plant_in_service")
    dep = col("accumulated_depreciation")
    net = net.fillna(gross - dep)
    out = pd.DataFrame({"net_total": net, "gross_total": gross}).reset_index()
    return out


def net_transmission_plant(sched204: pd.DataFrame, sched219: pd.DataFrame,
                           sched200: pd.DataFrame) -> pd.DataFrame:
    gross = gross_transmission_plant(sched204)
    dep = transmission_accum_depreciation(sched219)
    totals = total_net_utility_plant(sched200)

    m = gross.merge(dep, on=["utility_id_ferc1", "report_year"], how="left")
    m = m.merge(totals, on=["utility_id_ferc1", "report_year"], how="left")

    ratio = (m["net_total"] / m["gross_total"]).clip(0.0, 1.0)
    prorated = m["accum_dep_tx"].isna()
    m["net_tx"] = (m["gross_tx"] - m["accum_dep_tx"]).where(~prorated, m["gross_tx"] * ratio)
    m["net_tx_prorated"] = prorated.fillna(True)
    return m[["utility_id_ferc1", "report_year", "net_tx", "net_tx_prorated"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_ferc_form1.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transmission_rate_base/data/ferc_form1.py tests/transmission_rate_base/test_ferc_form1.py
git commit -m "feat(transmission-rate-base): FERC schedule extraction (gross/net tx plant, total plant)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 4: Utility map (FERC filer → parent ticker)

**Files:**
- Create: `transmission_rate_base/data/utility_map.py`
- Create: `tests/transmission_rate_base/test_utility_map.py`

**Interfaces:**
- Consumes: `config.UNIVERSE_SEED`; the `utility_xwalk` frame from `ferc_form1.fetch_table("utility_xwalk")` with columns `utility_id_ferc1`, `utility_name_ferc1`, `utility_id_pudl`.
- Produces:
  - `PARENT_FILERS: dict[str, list[str]]` — ticker → list of `utility_name_ferc1` strings.
  - `resolve_filers(xwalk: pd.DataFrame, parents: dict[str, list[str]] = PARENT_FILERS) -> dict[str, list[int]]` — ticker → list of `utility_id_ferc1` ints (unions all FERC ids that share a `utility_id_pudl` with any named filer). Raises `KeyError` listing any `utility_name_ferc1` that does not appear in `xwalk`.
  - `validate(xwalk: pd.DataFrame) -> None` — raises `ValueError` if any `utility_id_pudl` maps to two tickers, or any ticker resolves to zero ids.

- [ ] **Step 1: Write the failing test**

`tests/transmission_rate_base/test_utility_map.py`:

```python
import pandas as pd
import pytest

from transmission_rate_base.data import utility_map as um


def _xwalk():
    return pd.DataFrame([
        dict(utility_id_ferc1=11, utility_name_ferc1="Appalachian Power Co", utility_id_pudl=100),
        dict(utility_id_ferc1=12, utility_name_ferc1="Appalachian Power Co (XBRL)", utility_id_pudl=100),
        dict(utility_id_ferc1=13, utility_name_ferc1="Ohio Power Co", utility_id_pudl=101),
        dict(utility_id_ferc1=20, utility_name_ferc1="Florida Power & Light Co", utility_id_pudl=200),
    ])


def test_resolve_unions_pudl_id_siblings():
    parents = {"AEP": ["Appalachian Power Co", "Ohio Power Co"], "NEE": ["Florida Power & Light Co"]}
    got = um.resolve_filers(_xwalk(), parents)
    assert sorted(got["AEP"]) == [11, 12, 13]   # 12 pulled in via shared pudl id 100
    assert got["NEE"] == [20]


def test_resolve_unknown_name_raises():
    with pytest.raises(KeyError):
        um.resolve_filers(_xwalk(), {"X": ["No Such Utility Co"]})


def test_validate_rejects_double_mapped_pudl_id():
    parents = {"AEP": ["Appalachian Power Co"], "OTHER": ["Appalachian Power Co (XBRL)"]}
    with pytest.raises(ValueError):
        um.validate(_xwalk(), parents)


def test_shipped_map_keys_are_within_universe_seed():
    from transmission_rate_base import config
    extra = set(um.PARENT_FILERS) - set(config.UNIVERSE_SEED)
    assert not extra, f"PARENT_FILERS has tickers not in UNIVERSE_SEED: {extra}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_utility_map.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

`transmission_rate_base/data/utility_map.py`:

```python
from __future__ import annotations

import pandas as pd

# ticker -> list of exact utility_name_ferc1 strings for its regulated electric
# filers. Built by hand from each parent's latest 10-K subsidiary list, matched
# to core_pudl__assn_ferc1_pudl_utilities. Both DBF-era and XBRL-era names are
# pulled in automatically via shared utility_id_pudl (see resolve_filers).
#
# SEED ENTRIES ONLY -- the executor completes this to the full UNIVERSE_SEED
# using the procedure in the spec (§5). Every value string must appear verbatim
# in the crosswalk's utility_name_ferc1 column.
PARENT_FILERS: dict[str, list[str]] = {
    "AEP": [
        "Appalachian Power Co", "Ohio Power Co", "Indiana Michigan Power Co",
        "AEP Texas Inc.", "Public Service Co of Oklahoma",
        "Southwestern Electric Power Co", "Kentucky Power Co",
    ],
    "NEE": ["Florida Power & Light Co"],
    "DUK": [
        "Duke Energy Carolinas, LLC", "Duke Energy Progress, LLC",
        "Duke Energy Florida, LLC", "Duke Energy Indiana, LLC",
        "Duke Energy Ohio, Inc.", "Duke Energy Kentucky, Inc.",
    ],
    "XEL": [
        "Northern States Power Co (Minnesota)", "Northern States Power Co (Wisconsin)",
        "Public Service Co of Colorado", "Southwestern Public Service Co",
    ],
    "ETR": [
        "Entergy Arkansas, LLC", "Entergy Louisiana, LLC", "Entergy Mississippi, LLC",
        "Entergy New Orleans, LLC", "Entergy Texas, Inc.",
    ],
}


def _pudl_ids_for_names(xwalk: pd.DataFrame, names: list[str]) -> set[int]:
    have = set(xwalk["utility_name_ferc1"])
    missing = [n for n in names if n not in have]
    if missing:
        raise KeyError(f"utility_name_ferc1 not in crosswalk: {missing}")
    return set(xwalk.loc[xwalk["utility_name_ferc1"].isin(names), "utility_id_pudl"].dropna())


def resolve_filers(xwalk: pd.DataFrame,
                   parents: dict[str, list[str]] | None = None) -> dict[str, list[int]]:
    parents = PARENT_FILERS if parents is None else parents
    out: dict[str, list[int]] = {}
    for ticker, names in parents.items():
        pudl_ids = _pudl_ids_for_names(xwalk, names)
        ids = xwalk.loc[xwalk["utility_id_pudl"].isin(pudl_ids), "utility_id_ferc1"]
        out[ticker] = sorted(int(i) for i in ids.dropna().unique())
    return out


def validate(xwalk: pd.DataFrame, parents: dict[str, list[str]] | None = None) -> None:
    parents = PARENT_FILERS if parents is None else parents
    resolved = resolve_filers(xwalk, parents)
    seen: dict[int, str] = {}
    for ticker, ids in resolved.items():
        if not ids:
            raise ValueError(f"{ticker} resolves to zero FERC filer ids")
        for i in ids:
            if i in seen and seen[i] != ticker:
                raise ValueError(f"FERC id {i} mapped to both {seen[i]} and {ticker}")
            seen[i] = ticker
```

**Executor note:** completing `PARENT_FILERS` to all ~34 seed tickers is a data-entry task. For each ticker: open the latest 10-K, list regulated electric subsidiaries, find each in the crosswalk (`fetch_table("utility_xwalk")`), paste the exact `utility_name_ferc1`. Acceptance: `validate(fetch_table("utility_xwalk"))` passes and `set(PARENT_FILERS) == set(config.UNIVERSE_SEED)` minus any ticker with no FERC filer (document exclusions inline).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_utility_map.py -q`
Expected: PASS (the `test_shipped_map_keys_are_within_universe_seed` test passes with the seed subset; it only checks there are no *extra* keys).

- [ ] **Step 5: Commit**

```bash
git add transmission_rate_base/data/utility_map.py tests/transmission_rate_base/test_utility_map.py
git commit -m "feat(transmission-rate-base): utility_map filer->parent resolver + validation

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 5: Parent-level transmission panel

**Files:**
- Create: `transmission_rate_base/signal.py`
- Create: `tests/transmission_rate_base/test_signal.py`

**Interfaces:**
- Consumes: `net_transmission_plant(...)` output (`utility_id_ferc1, report_year, net_tx, net_tx_prorated`); `total_net_utility_plant(...)` output (`utility_id_ferc1, report_year, net_total, gross_total`); `resolve_filers(...)` output (`dict[str, list[int]]`).
- Produces:
  - `build_parent_panel(net_tx_filer: pd.DataFrame, totals_filer: pd.DataFrame, filer_map: dict[str, list[int]]) -> pd.DataFrame`
    with columns `["ticker", "year", "net_tx", "net_total", "tx_share"]`, one row per ticker-year, sorted by `ticker, year`. `net_tx` and `net_total` are filer sums; `tx_share = net_tx / net_total`. A ticker-year is included only if **every** mapped filer that ever reports also reports that year (so a partial-coverage year is dropped, not silently undercounted) — implement as: keep ticker-years where the count of reporting filers equals that ticker's max filer count over the sample.

- [ ] **Step 1: Write the failing test**

`tests/transmission_rate_base/test_signal.py`:

```python
import numpy as np
import pandas as pd
import pytest

from transmission_rate_base import signal as sig


def test_build_parent_panel_sums_filers_and_drops_partial_years():
    net_tx = pd.DataFrame([
        dict(utility_id_ferc1=1, report_year=2018, net_tx=100.0, net_tx_prorated=False),
        dict(utility_id_ferc1=2, report_year=2018, net_tx=50.0, net_tx_prorated=False),
        dict(utility_id_ferc1=1, report_year=2019, net_tx=110.0, net_tx_prorated=False),
        # 2019 filer 2 missing -> 2019 is a partial year for AEP -> dropped
    ])
    totals = pd.DataFrame([
        dict(utility_id_ferc1=1, report_year=2018, net_total=400.0, gross_total=500.0),
        dict(utility_id_ferc1=2, report_year=2018, net_total=100.0, gross_total=120.0),
        dict(utility_id_ferc1=1, report_year=2019, net_total=420.0, gross_total=520.0),
    ])
    panel = sig.build_parent_panel(net_tx, totals, {"AEP": [1, 2]})
    assert list(panel["year"]) == [2018]
    row = panel.iloc[0]
    assert row["net_tx"] == 150.0 and row["net_total"] == 500.0
    assert row["tx_share"] == pytest.approx(0.3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_signal.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

`transmission_rate_base/signal.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from transmission_rate_base import config


def build_parent_panel(net_tx_filer: pd.DataFrame, totals_filer: pd.DataFrame,
                       filer_map: dict[str, list[int]]) -> pd.DataFrame:
    id_to_ticker = {fid: tkr for tkr, ids in filer_map.items() for fid in ids}

    tx = net_tx_filer.copy()
    tx["ticker"] = tx["utility_id_ferc1"].map(id_to_ticker)
    tx = tx.dropna(subset=["ticker"])

    tot = totals_filer.copy()
    tot["ticker"] = tot["utility_id_ferc1"].map(id_to_ticker)
    tot = tot.dropna(subset=["ticker"])

    merged = tx.merge(tot[["utility_id_ferc1", "report_year", "net_total"]],
                      on=["utility_id_ferc1", "report_year"], how="inner")

    # expected filer count per ticker = max distinct reporting filers in any year
    per_year = merged.groupby(["ticker", "report_year"])["utility_id_ferc1"].nunique()
    expected = per_year.groupby("ticker").max()

    agg = (merged.groupby(["ticker", "report_year"])
           .agg(net_tx=("net_tx", "sum"), net_total=("net_total", "sum"),
                n_filers=("utility_id_ferc1", "nunique"))
           .reset_index())
    agg = agg[agg.apply(lambda r: r["n_filers"] == expected[r["ticker"]], axis=1)]

    agg["tx_share"] = agg["net_tx"] / agg["net_total"]
    agg = agg.rename(columns={"report_year": "year"})
    return (agg[["ticker", "year", "net_tx", "net_total", "tx_share"]]
            .sort_values(["ticker", "year"]).reset_index(drop=True))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_signal.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transmission_rate_base/signal.py tests/transmission_rate_base/test_signal.py
git commit -m "feat(transmission-rate-base): parent-level transmission panel from filer sums

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 6: Primary signal (growth + share-shift, rank-average, guards)

**Files:**
- Modify: `transmission_rate_base/signal.py`
- Modify: `tests/transmission_rate_base/test_signal.py`

**Interfaces:**
- Consumes: `build_parent_panel(...)` output.
- Produces:
  - `primary_signal(panel: pd.DataFrame) -> pd.DataFrame` with columns `["ticker", "year", "g3_net_tx", "d3_tx_share", "raw_signal"]`.
    - `g3_net_tx` = `(net_tx[y] / net_tx[y-3]) ** (1/3) - 1`, requires the exact rows `y-3, y-2, y-1, y` present for that ticker with all `net_tx > 0` and all `net_total > 0`.
    - `d3_tx_share` = `tx_share[y] - tx_share[y-3]`.
    - Structural-break guard: if any of `abs(log(net_tx[k]/net_tx[k-1]))` for `k` in `y-2..y` exceeds `config.STRUCTURAL_BREAK_LOG`, both component values and `raw_signal` are NaN for year `y`.
    - `raw_signal` = mean of `g3_net_tx.rank(pct=True)` and `d3_tx_share.rank(pct=True)`, ranks computed **within each year** across tickers, over rows where both components are non-NaN. NaN where either component is NaN.

- [ ] **Step 1: Write the failing test**

Append to `tests/transmission_rate_base/test_signal.py`:

```python
def _panel_two_names():
    rows = []
    # FAST: net_tx doubles over 3 yrs; share rises
    for y, tx, tot in [(2015, 100, 500), (2016, 120, 520), (2017, 150, 550), (2018, 200, 600)]:
        rows.append(dict(ticker="FAST", year=y, net_tx=float(tx), net_total=float(tot),
                         tx_share=tx / tot))
    # SLOW: flat
    for y, tx, tot in [(2015, 100, 500), (2016, 101, 505), (2017, 102, 510), (2018, 103, 515)]:
        rows.append(dict(ticker="SLOW", year=y, net_tx=float(tx), net_total=float(tot),
                         tx_share=tx / tot))
    return pd.DataFrame(rows)


def test_primary_signal_ranks_fast_above_slow():
    out = sig.primary_signal(_panel_two_names())
    y2018 = out[out["year"] == 2018].set_index("ticker")
    assert y2018.loc["FAST", "g3_net_tx"] == pytest.approx(2 ** (1 / 3) - 1)
    assert y2018.loc["FAST", "raw_signal"] > y2018.loc["SLOW", "raw_signal"]
    assert y2018.loc["FAST", "raw_signal"] == pytest.approx(1.0)   # top of both ranks
    assert y2018.loc["SLOW", "raw_signal"] == pytest.approx(0.5)


def test_primary_signal_structural_break_nans_the_year():
    p = _panel_two_names()
    p.loc[(p.ticker == "FAST") & (p.year == 2017), ["net_tx", "tx_share"]] = [400.0, 400 / 550]
    out = sig.primary_signal(p).set_index(["ticker", "year"])
    assert np.isnan(out.loc[("FAST", 2018), "raw_signal"])


def test_primary_signal_requires_four_consecutive_years():
    p = _panel_two_names()
    p = p[~((p.ticker == "FAST") & (p.year == 2015))]     # drop y-3
    out = sig.primary_signal(p).set_index(["ticker", "year"])
    assert np.isnan(out.loc[("FAST", 2018), "raw_signal"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_signal.py -q`
Expected: FAIL — `primary_signal` missing.

- [ ] **Step 3: Write minimal implementation**

Append to `transmission_rate_base/signal.py`:

```python
def _one_ticker_components(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("year").set_index("year")
    n = config.SIGNAL_CAGR_YEARS
    out = []
    for y in g.index:
        years = list(range(y - n, y + 1))
        if not all(yr in g.index for yr in years):
            continue
        sub = g.loc[years]
        if (sub["net_tx"] <= 0).any() or (sub["net_total"] <= 0).any():
            continue
        steps = np.log(sub["net_tx"].values[1:] / sub["net_tx"].values[:-1])
        broke = np.any(np.abs(steps) > config.STRUCTURAL_BREAK_LOG)
        g3 = np.nan if broke else (sub["net_tx"].iloc[-1] / sub["net_tx"].iloc[0]) ** (1 / n) - 1
        d3 = np.nan if broke else sub["tx_share"].iloc[-1] - sub["tx_share"].iloc[0]
        out.append(dict(year=y, g3_net_tx=g3, d3_tx_share=d3))
    return pd.DataFrame(out)


def primary_signal(panel: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for ticker, g in panel.groupby("ticker"):
        comp = _one_ticker_components(g)
        comp["ticker"] = ticker
        parts.append(comp)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["ticker", "year", "g3_net_tx", "d3_tx_share"])

    df["raw_signal"] = np.nan
    for y, grp in df.groupby("year"):
        ok = grp["g3_net_tx"].notna() & grp["d3_tx_share"].notna()
        if ok.sum() == 0:
            continue
        r1 = grp.loc[ok, "g3_net_tx"].rank(pct=True)
        r2 = grp.loc[ok, "d3_tx_share"].rank(pct=True)
        df.loc[r1.index, "raw_signal"] = (r1 + r2) / 2
    return df[["ticker", "year", "g3_net_tx", "d3_tx_share", "raw_signal"]].sort_values(
        ["year", "ticker"]).reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_signal.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transmission_rate_base/signal.py tests/transmission_rate_base/test_signal.py
git commit -m "feat(transmission-rate-base): primary signal (3yr net-tx CAGR + share shift, rank-avg)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 7: Cross-sectional neutralisation + segment-mix loader

**Files:**
- Modify: `transmission_rate_base/signal.py`
- Create: `transmission_rate_base/data/segment_mix.csv`
- Modify: `tests/transmission_rate_base/test_signal.py`

**Interfaces:**
- Consumes: `primary_signal(...)` output; `build_parent_panel(...)` output (for `net_total` → size); `segment_mix.csv`.
- Produces:
  - `load_segment_mix(path: str | Path | None = None) -> pd.DataFrame` — columns `["ticker", "fy", "nonreg_rev_share"]` where `nonreg_rev_share = 1 - reg_elec_op_rev / total_op_rev`. Default path = `data/segment_mix.csv`.
  - `neutralize(signal_df: pd.DataFrame, panel: pd.DataFrame, segment_mix: pd.DataFrame) -> pd.DataFrame` — adds column `neutral_signal`. Per year: take rows with non-NaN `raw_signal`; build `log_rate_base = log(net_total)` (from `panel`, same ticker-year) and `nonreg_rev_share` (from `segment_mix`, `fy == year`, missing → 0.0); winsorise `raw_signal` at ±`config.WINSOR_SIGMA`·σ; z-score; OLS regress on `[1, log_rate_base, nonreg_rev_share]`; residual; re-z-score. Years with < 5 usable names: `neutral_signal = z-scored raw_signal` (skip the regression).

- [ ] **Step 1: Write the failing test**

Append to `tests/transmission_rate_base/test_signal.py`:

```python
def test_load_segment_mix_computes_nonreg_share(tmp_path):
    p = tmp_path / "sm.csv"
    p.write_text("ticker,fy,reg_elec_op_rev,total_op_rev\nAEP,2020,80,100\nD,2020,50,100\n")
    sm = sig.load_segment_mix(p)
    assert set(sm.columns) == {"ticker", "fy", "nonreg_rev_share"}
    assert sm.set_index("ticker").loc["AEP", "nonreg_rev_share"] == pytest.approx(0.2)


def test_neutralize_removes_planted_size_effect():
    rng = np.random.default_rng(0)
    tickers = [f"T{i:02d}" for i in range(20)]
    size = np.linspace(1e9, 5e10, 20)
    # raw_signal is pure size (log) plus tiny noise -> neutralised should be ~uncorrelated with size
    raw = np.log(size) + 0.01 * rng.standard_normal(20)
    sig_df = pd.DataFrame(dict(ticker=tickers, year=2020, g3_net_tx=raw, d3_tx_share=raw,
                               raw_signal=raw))
    panel = pd.DataFrame(dict(ticker=tickers, year=2020, net_tx=size / 3, net_total=size,
                              tx_share=0.33))
    sm = pd.DataFrame(columns=["ticker", "fy", "nonreg_rev_share"])
    out = sig.neutralize(sig_df, panel, sm)
    corr = np.corrcoef(out["neutral_signal"], np.log(size))[0, 1]
    assert abs(corr) < 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_signal.py -q`
Expected: FAIL — `load_segment_mix` / `neutralize` missing.

- [ ] **Step 3: Write minimal implementation**

Create `transmission_rate_base/data/segment_mix.csv` with the header and a handful of real rows (executor extends it):

```csv
ticker,fy,reg_elec_op_rev,total_op_rev
AEP,2023,18500,19000
AEP,2022,17800,18800
D,2023,11000,14400
```

Append to `transmission_rate_base/signal.py`:

```python
from pathlib import Path

from grid_resilience.factor.neutralize import winsorize, cross_section_zscore

_SEGMENT_MIX_PATH = Path(__file__).parent / "data" / "segment_mix.csv"


def load_segment_mix(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or _SEGMENT_MIX_PATH)
    df["nonreg_rev_share"] = 1.0 - df["reg_elec_op_rev"] / df["total_op_rev"]
    return df[["ticker", "fy", "nonreg_rev_share"]]


def _resid_ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def neutralize(signal_df: pd.DataFrame, panel: pd.DataFrame,
               segment_mix: pd.DataFrame) -> pd.DataFrame:
    size = (panel.assign(log_rate_base=np.log(panel["net_total"]))
            .set_index(["ticker", "year"])["log_rate_base"])
    sm = segment_mix.set_index(["ticker", "fy"])["nonreg_rev_share"]

    df = signal_df.copy()
    df["neutral_signal"] = np.nan
    for y, grp in df.groupby("year"):
        g = grp[grp["raw_signal"].notna()].copy()
        if g.empty:
            continue
        g["z"] = cross_section_zscore(winsorize(
            g["raw_signal"], (0.5 - config.WINSOR_SIGMA / 100, 0.5 + config.WINSOR_SIGMA / 100)))
        if len(g) < 5:
            df.loc[g.index, "neutral_signal"] = g["z"].values
            continue
        lrb = g["ticker"].map(lambda t: size.get((t, y), np.nan)).astype(float)
        nrs = g["ticker"].map(lambda t: sm.get((t, y), 0.0)).astype(float).fillna(0.0)
        lrb = lrb.fillna(lrb.mean())
        X = np.column_stack([np.ones(len(g)), lrb.values, nrs.values])
        resid = _resid_ols(g["z"].values, X)
        df.loc[g.index, "neutral_signal"] = cross_section_zscore(pd.Series(resid, index=g.index))
    return df
```

**Note on `winsorize`:** `grid_resilience.factor.neutralize.winsorize` takes quantile limits; the spec calls for ±σ clipping. Implement a local sigma-clip instead if the quantile form is awkward — mirror `backlog_factor.signal._winsorize_sigma(s, sigma)` (clip to `mean ± sigma*std`). Pick one, test it.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_signal.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transmission_rate_base/signal.py transmission_rate_base/data/segment_mix.csv \
        tests/transmission_rate_base/test_signal.py
git commit -m "feat(transmission-rate-base): cross-sectional neutralisation + segment-mix loader

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 8: Price re-point (monthly + daily returns, offline-capable)

**Files:**
- Create: `transmission_rate_base/data/prices.py`
- Create: `tests/transmission_rate_base/test_prices.py`

**Interfaces:**
- Consumes: `grid_resilience.data.equity_prices.fetch_prices`.
- Produces:
  - `daily_prices(tickers: list[str], start: str, end: str, *, offline: bool = False) -> pd.DataFrame` — date-indexed, ticker columns. When `offline`, calls `fetch_prices(..., use_cache=True)` and never triggers a fetch of missing months (pass through whatever the cache returns; drop tickers with no column).
  - `monthly_returns(tickers, start, end, *, offline=False) -> pd.DataFrame` — month-end simple returns.
  - `daily_returns(tickers, start, end, *, offline=False) -> pd.DataFrame` — daily simple returns.

- [ ] **Step 1: Write the failing test**

`tests/transmission_rate_base/test_prices.py`:

```python
import pandas as pd
import pytest

from transmission_rate_base.data import prices as P


def test_monthly_and_daily_returns_from_stubbed_prices(monkeypatch):
    idx = pd.date_range("2020-01-01", "2020-03-31", freq="B")
    fake = pd.DataFrame({"AEP": range(len(idx)), "D": range(len(idx))}, index=idx).astype(float) + 100
    monkeypatch.setattr(P, "_fetch_prices", lambda t, s, e, offline: fake[t])

    dr = P.daily_returns(["AEP"], "2020-01-01", "2020-03-31")
    assert list(dr.columns) == ["AEP"]
    assert dr.notna().all().all()

    mr = P.monthly_returns(["AEP", "D"], "2020-01-01", "2020-03-31")
    assert list(mr.index.month) == [2, 3]
    assert (mr["AEP"] > 0).all()


def test_offline_flag_is_forwarded(monkeypatch):
    seen = {}
    def _spy(t, s, e, offline):
        seen["offline"] = offline
        return pd.DataFrame({"AEP": [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2))
    monkeypatch.setattr(P, "_fetch_prices", _spy)
    P.daily_prices(["AEP"], "2020-01-01", "2020-01-02", offline=True)
    assert seen["offline"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_prices.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

`transmission_rate_base/data/prices.py`:

```python
from __future__ import annotations

import pandas as pd

from grid_resilience.data.equity_prices import fetch_prices


def _fetch_prices(tickers: list[str], start: str, end: str, offline: bool) -> pd.DataFrame:
    # offline: rely purely on cache; fetch_prices(use_cache=True) still tries to
    # gap-fill, so when offline we call it and tolerate a partial frame.
    px = fetch_prices(sorted(set(tickers)), start, end, use_cache=True)
    return px


def daily_prices(tickers, start, end, *, offline: bool = False) -> pd.DataFrame:
    px = _fetch_prices(list(tickers), start, end, offline)
    cols = [t for t in tickers if t in px.columns]
    return px[cols].loc[start:end]


def daily_returns(tickers, start, end, *, offline: bool = False) -> pd.DataFrame:
    return daily_prices(tickers, start, end, offline=offline).pct_change().dropna(how="all")


def monthly_returns(tickers, start, end, *, offline: bool = False) -> pd.DataFrame:
    px = daily_prices(tickers, start, end, offline=offline)
    return px.resample("ME").last().pct_change().dropna(how="all")
```

**Executor note:** the `offline` flag is threaded but `fetch_prices` has no offline mode; wrapping it is enough for now (cache hits return immediately; a cache miss with no network raises inside yfinance, which `report.run_pipeline` catches and reports as "prices unavailable"). If a true no-network guarantee is needed later, add a `use_cache_only` path to `equity_prices` in a separate change.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_prices.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transmission_rate_base/data/prices.py tests/transmission_rate_base/test_prices.py
git commit -m "feat(transmission-rate-base): price re-point (monthly/daily returns, offline-tolerant)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 9: Portfolio construction (quintile L/S + long-only tilt)

**Files:**
- Create: `transmission_rate_base/portfolio.py`
- Create: `tests/transmission_rate_base/test_portfolio.py`

**Interfaces:**
- Consumes: `neutralize(...)` output (`ticker, year, neutral_signal`); daily returns (date × ticker); EW-utility daily return Series.
- Produces:
  - `rebalance_dates(trading_index: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]` — first trading day on/after May 1 of each year in range.
  - `quintile_ls_weights(signal_df, daily_ret, ew_util_ret) -> pd.DataFrame` — index = every trading day in range, columns = tickers, weights forward-filled between rebalances. At each rebalance date `d` in year `Y`: rank names with non-NaN `neutral_signal` for `year == Y-1`; long top `1/config.QUINTILES`, short bottom `1/config.QUINTILES`, equal-weight within leg; long leg sums to +1, short leg to −s where `s = clip(beta_long/beta_short, *config.BETA_CLIP)`; each leg beta = OLS beta of the equal-weight leg daily return vs `ew_util_ret` over the trailing `config.BETA_WINDOW` days ending at `d` (if < 60 obs, `s = 1.0`).
  - `long_tilt_weights(signal_df, daily_ret) -> pd.DataFrame` — same index/columns; at each rebalance start from equal weight over names with non-NaN signal for `year == Y-1`, ×1.25 for top quintile, ×0.75 for bottom quintile, renormalise to sum 1, long-only.

- [ ] **Step 1: Write the failing test**

`tests/transmission_rate_base/test_portfolio.py`:

```python
import numpy as np
import pandas as pd

from transmission_rate_base import portfolio as pf


def _daily_index():
    return pd.bdate_range("2014-01-01", "2019-12-31")


def _signal_df():
    tickers = [f"T{i:02d}" for i in range(10)]
    rows = []
    for y in (2014, 2015, 2016, 2017, 2018):
        for i, t in enumerate(tickers):
            rows.append(dict(ticker=t, year=y, neutral_signal=float(i)))  # T09 best, T00 worst
    return pd.DataFrame(rows)


def test_rebalance_dates_are_first_trading_day_from_may():
    idx = _daily_index()
    ds = pf.rebalance_dates(idx, "2015-01-01", "2018-12-31")
    assert [d.year for d in ds] == [2015, 2016, 2017, 2018]
    assert all(d.month == 5 for d in ds)


def test_quintile_ls_weights_dollar_and_sign_structure():
    idx = _daily_index()
    dr = pd.DataFrame(0.0, index=idx, columns=[f"T{i:02d}" for i in range(10)])
    ew = pd.Series(0.0, index=idx)
    w = pf.quintile_ls_weights(_signal_df(), dr, ew)
    row = w.loc["2018-06-01"]
    assert row[row > 0].sum() == pytest.approx(1.0)
    assert row["T09"] > 0 and row["T08"] > 0        # top quintile (2 of 10) long
    assert row["T00"] < 0 and row["T01"] < 0        # bottom quintile short
    # with zero returns beta ratio -> 1.0, so short leg sums to -1
    assert row[row < 0].sum() == pytest.approx(-1.0)


def test_long_tilt_weights_nonneg_and_sum_one():
    idx = _daily_index()
    dr = pd.DataFrame(0.0, index=idx, columns=[f"T{i:02d}" for i in range(10)])
    w = pf.long_tilt_weights(_signal_df(), dr)
    row = w.loc["2018-06-01"]
    assert (row >= 0).all()
    assert row.sum() == pytest.approx(1.0)
    assert row["T09"] > row["T05"] > row["T00"]
```

(Add `import pytest` at the top.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_portfolio.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

`transmission_rate_base/portfolio.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from transmission_rate_base import config


def rebalance_dates(trading_index: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    idx = trading_index[(trading_index >= start) & (trading_index <= end)]
    out = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        may1 = pd.Timestamp(year=year, month=config.REBALANCE_MONTH, day=1)
        after = idx[idx >= may1]
        if len(after):
            out.append(after[0])
    return out


def _leg_beta(leg_ret: pd.Series, mkt: pd.Series, asof: pd.Timestamp) -> float:
    window = leg_ret.loc[:asof].iloc[-config.BETA_WINDOW:]
    m = mkt.reindex(window.index)
    pair = pd.DataFrame({"r": window, "m": m}).dropna()
    if len(pair) < 60 or pair["m"].var() == 0:
        return 1.0
    return float(np.cov(pair["r"], pair["m"])[0, 1] / pair["m"].var())


def _quintile_members(sig_year: pd.Series) -> tuple[list[str], list[str]]:
    s = sig_year.dropna().sort_values()
    k = max(1, int(round(len(s) / config.QUINTILES)))
    return s.index[-k:].tolist(), s.index[:k].tolist()


def quintile_ls_weights(signal_df: pd.DataFrame, daily_ret: pd.DataFrame,
                        ew_util_ret: pd.Series) -> pd.DataFrame:
    idx = daily_ret.index
    W = pd.DataFrame(0.0, index=idx, columns=daily_ret.columns)
    sig = signal_df.set_index(["year", "ticker"])["neutral_signal"]
    for d in rebalance_dates(idx, str(idx.min().date()), str(idx.max().date())):
        y = d.year
        try:
            sig_year = sig.loc[y - 1]
        except KeyError:
            continue
        longs, shorts = _quintile_members(sig_year.reindex(daily_ret.columns))
        if not longs or not shorts:
            continue
        lw = pd.Series(1.0 / len(longs), index=longs)
        sw = pd.Series(1.0 / len(shorts), index=shorts)
        long_leg = daily_ret[longs].mean(axis=1)
        short_leg = daily_ret[shorts].mean(axis=1)
        bl = _leg_beta(long_leg, ew_util_ret, d)
        bs = _leg_beta(short_leg, ew_util_ret, d)
        scale = float(np.clip(bl / bs if bs else 1.0, *config.BETA_CLIP))
        row = pd.Series(0.0, index=daily_ret.columns)
        row[longs] = lw
        row[shorts] = -sw * scale
        W.loc[d:] = row.values
    return W


def long_tilt_weights(signal_df: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
    idx = daily_ret.index
    W = pd.DataFrame(0.0, index=idx, columns=daily_ret.columns)
    sig = signal_df.set_index(["year", "ticker"])["neutral_signal"]
    for d in rebalance_dates(idx, str(idx.min().date()), str(idx.max().date())):
        try:
            sig_year = sig.loc[d.year - 1].reindex(daily_ret.columns).dropna()
        except KeyError:
            continue
        if sig_year.empty:
            continue
        longs, shorts = _quintile_members(sig_year)
        w = pd.Series(1.0, index=sig_year.index)
        w[longs] *= 1.25
        w[shorts] *= 0.75
        w = w / w.sum()
        row = pd.Series(0.0, index=daily_ret.columns)
        row[w.index] = w.values
        W.loc[d:] = row.values
    return W
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_portfolio.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transmission_rate_base/portfolio.py tests/transmission_rate_base/test_portfolio.py
git commit -m "feat(transmission-rate-base): quintile L/S (dollar+beta neutral) + long-only tilt

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 10: Backtest (monthly P&L + metrics) and signal rank-IC

**Files:**
- Create: `transmission_rate_base/backtest.py`
- Create: `tests/transmission_rate_base/test_backtest.py`

**Interfaces:**
- Consumes: a daily weight matrix (from `portfolio`); `monthly_returns` (date × ticker); `neutralize(...)` output; a daily-returns frame + EW-utility daily Series for `compute_ic`.
- Produces:
  - `run_backtest(weights_daily: pd.DataFrame, monthly_ret: pd.DataFrame, rf_annual: float = 0.04) -> tuple[pd.Series, dict]` — resample weights to month-end (last), align to `monthly_ret`, `strat_ret = (w * r).sum(axis=1)`; return the monthly strategy return Series and a metrics dict with keys `ann_return, ann_vol, sharpe, max_dd, hit_rate, positive_years_frac` (annualisation factor 12).
  - `annual_returns(monthly_strat_ret: pd.Series) -> pd.Series` — calendar-year compounded returns.
  - `signal_rank_ic(neutral_signal_df: pd.DataFrame, daily_ret: pd.DataFrame, ew_util_daily: pd.Series, rebal_dates: list[pd.Timestamp]) -> dict` — for each rebalance date `d` (year `Y`), Spearman corr between `neutral_signal` (`year == Y-1`) and the **next 252-trading-day return relative to EW utilities** per ticker; return `{"ic": pd.Series(date->ic), "mean_ic": float, "t_stat": float, "n": int}`.

Reuse note: `grid_resilience.portfolio.backtest.compute_ic` computes cross-sectional Pearson IC over horizons and takes `factor_scores` as long `[date, ticker, factor_score]`. It is close but uses Pearson and its own forward-return convention. For this task write the small Spearman/relative-return version directly (above); keep `compute_ic` available for a cross-check in the report only.

- [ ] **Step 1: Write the failing test**

`tests/transmission_rate_base/test_backtest.py`:

```python
import numpy as np
import pandas as pd
import pytest

from transmission_rate_base import backtest as bt


def test_run_backtest_hand_computed_two_months():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    w = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    w["A"] = 1.0
    w["B"] = -1.0
    mr = pd.DataFrame({"A": [np.nan, 0.10, 0.05], "B": [np.nan, 0.02, -0.03]},
                      index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]))
    ret, m = bt.run_backtest(w, mr)
    # Feb: 0.10 - 0.02 = 0.08 ; Mar: 0.05 - (-0.03) = 0.08
    assert ret.loc["2020-02-29"] == pytest.approx(0.08)
    assert ret.loc["2020-03-31"] == pytest.approx(0.08)
    assert m["hit_rate"] == pytest.approx(1.0)


def test_signal_rank_ic_positive_when_high_signal_outperforms():
    idx = pd.bdate_range("2015-01-01", "2019-12-31")
    tickers = [f"T{i:02d}" for i in range(12)]
    # forward drift proportional to signal rank
    drift = {t: (i - 6) * 0.0002 for i, t in enumerate(tickers)}
    dr = pd.DataFrame({t: drift[t] for t in tickers}, index=idx)
    ew = dr.mean(axis=1)
    sdf = pd.DataFrame([dict(ticker=t, year=y, neutral_signal=float(i))
                        for y in (2015, 2016, 2017, 2018) for i, t in enumerate(tickers)])
    from transmission_rate_base.portfolio import rebalance_dates
    rd = rebalance_dates(idx, "2016-01-01", "2019-12-31")
    out = bt.signal_rank_ic(sdf, dr, ew, rd)
    assert out["mean_ic"] > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_backtest.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

`transmission_rate_base/backtest.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(weights_daily: pd.DataFrame, monthly_ret: pd.DataFrame,
                 rf_annual: float = 0.04):
    w = weights_daily.resample("ME").last()
    common_dates = w.index.intersection(monthly_ret.index)
    cols = [c for c in w.columns if c in monthly_ret.columns]
    w = w.loc[common_dates, cols].fillna(0.0)
    r = monthly_ret.loc[common_dates, cols]
    strat = (w * r).sum(axis=1).dropna()

    ann_factor = 12
    ann_ret = strat.mean() * ann_factor
    ann_vol = strat.std() * np.sqrt(ann_factor)
    excess = strat - rf_annual / ann_factor
    sharpe = excess.mean() / excess.std() * np.sqrt(ann_factor) if excess.std() > 0 else np.nan
    cum = (1 + strat).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())
    yr = annual_returns(strat)
    metrics = {
        "ann_return": float(ann_ret), "ann_vol": float(ann_vol),
        "sharpe": float(sharpe), "max_dd": max_dd,
        "hit_rate": float((strat > 0).mean()),
        "positive_years_frac": float((yr > 0).mean()) if len(yr) else np.nan,
    }
    return strat, metrics


def annual_returns(monthly_strat_ret: pd.Series) -> pd.Series:
    return monthly_strat_ret.groupby(monthly_strat_ret.index.year).apply(
        lambda s: (1 + s).prod() - 1)


def signal_rank_ic(neutral_signal_df: pd.DataFrame, daily_ret: pd.DataFrame,
                   ew_util_daily: pd.Series, rebal_dates) -> dict:
    sig = neutral_signal_df.set_index(["year", "ticker"])["neutral_signal"]
    ics = {}
    for d in rebal_dates:
        try:
            sy = sig.loc[d.year - 1]
        except KeyError:
            continue
        fwd_end = daily_ret.index[daily_ret.index >= d][:252]
        if len(fwd_end) < 60:
            continue
        block = daily_ret.loc[fwd_end]
        tick_fwd = (1 + block).prod() - 1
        ew_fwd = (1 + ew_util_daily.reindex(fwd_end)).prod() - 1
        rel = tick_fwd - ew_fwd
        pair = pd.DataFrame({"s": sy, "r": rel}).dropna()
        if len(pair) >= 5:
            ics[d] = pair["s"].corr(pair["r"], method="spearman")
    s = pd.Series(ics).sort_index().dropna()
    if s.empty:
        return {"ic": s, "mean_ic": np.nan, "t_stat": np.nan, "n": 0}
    return {"ic": s, "mean_ic": float(s.mean()),
            "t_stat": float(s.mean() / s.std(ddof=1) * np.sqrt(len(s))) if s.std(ddof=1) > 0 else np.nan,
            "n": len(s)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_backtest.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transmission_rate_base/backtest.py tests/transmission_rate_base/test_backtest.py
git commit -m "feat(transmission-rate-base): monthly backtest + relative-return rank-IC

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 11: Additivity regression

**Files:**
- Create: `transmission_rate_base/additivity.py`
- Create: `transmission_rate_base/data/style_inputs.csv`
- Create: `tests/transmission_rate_base/test_additivity.py`

**Interfaces:**
- Consumes: the primary L/S monthly return Series (from `run_backtest`); daily returns (date × ticker) for the universe; EW-utility daily Series; XLU daily Series; `style_inputs.csv`.
- Produces:
  - `load_style_inputs(path=None) -> pd.DataFrame` — columns `["ticker","fy","dividend_yield","market_cap","book_value","capex","net_ppe"]`.
  - `build_controls(daily_ret, ew_util_daily, xlu_daily, style_inputs, rebal_dates) -> pd.DataFrame` — month-end indexed, columns exactly `["util_beta","xlu","dividend_yield","low_vol","size","momentum","capex_intensity","value"]`. `util_beta` = monthly EW-utility return; `xlu` = monthly XLU return; the rest are monthly top-minus-bottom-quintile L/S returns formed annually at each rebalance from the relevant characteristic (`dividend_yield`, trailing-12m realised vol → **low** minus high, `log(market_cap)` → **small** minus big, 12–1 month momentum → winner minus loser, `capex/net_ppe` → high minus low, `book_value/market_cap` → high minus low).
  - `run_additivity(factor_monthly_ret: pd.Series, controls: pd.DataFrame) -> dict` — align, OLS `factor ~ 1 + controls` via `statsmodels.OLS` with `cov_type="HAC", cov_kwds={"maxlags": 12}`; return `{"alpha_ann": intercept*12, "alpha_t": tvalue_of_intercept, "r2": rsquared, "coef": dict, "n": nobs}`.

- [ ] **Step 1: Write the failing test**

`tests/transmission_rate_base/test_additivity.py`:

```python
import numpy as np
import pandas as pd
import pytest

from transmission_rate_base import additivity as ad


def test_run_additivity_recovers_planted_alpha_and_loading():
    rng = np.random.default_rng(1)
    idx = pd.period_range("2011-01", "2020-12", freq="M").to_timestamp("M")
    mkt = pd.Series(rng.normal(0, 0.03, len(idx)), index=idx, name="util_beta")
    other = pd.DataFrame({c: rng.normal(0, 0.02, len(idx)) for c in
                          ["xlu", "dividend_yield", "low_vol", "size", "momentum",
                           "capex_intensity", "value"]}, index=idx)
    controls = pd.concat([mkt, other], axis=1)
    # factor = 0.4%/mo alpha + 0.5*util_beta + noise
    factor = 0.004 + 0.5 * mkt + rng.normal(0, 0.005, len(idx))
    out = ad.run_additivity(factor, controls)
    assert out["alpha_ann"] == pytest.approx(0.048, abs=0.02)
    assert out["coef"]["util_beta"] == pytest.approx(0.5, abs=0.15)
    assert out["alpha_t"] > 2.0


def test_load_style_inputs_schema(tmp_path):
    p = tmp_path / "si.csv"
    p.write_text("ticker,fy,dividend_yield,market_cap,book_value,capex,net_ppe\n"
                 "AEP,2020,0.035,4.5e10,2.0e10,7e9,7e10\n")
    si = ad.load_style_inputs(p)
    assert list(si.columns) == ["ticker", "fy", "dividend_yield", "market_cap",
                                "book_value", "capex", "net_ppe"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_additivity.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

Create `transmission_rate_base/data/style_inputs.csv`:

```csv
ticker,fy,dividend_yield,market_cap,book_value,capex,net_ppe
AEP,2023,0.041,44000,23000,9200,88000
D,2023,0.052,40000,26000,9800,92000
```

`transmission_rate_base/additivity.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

_STYLE_PATH = Path(__file__).parent / "data" / "style_inputs.csv"
_CONTROL_COLS = ["util_beta", "xlu", "dividend_yield", "low_vol", "size",
                 "momentum", "capex_intensity", "value"]


def load_style_inputs(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or _STYLE_PATH)


def _ls_from_char(char: pd.Series, fwd_monthly: pd.DataFrame, high_minus_low: bool) -> pd.Series:
    s = char.dropna().sort_values()
    k = max(1, len(s) // 5)
    top, bot = s.index[-k:], s.index[:k]
    long_, short_ = (top, bot) if high_minus_low else (bot, top)
    return fwd_monthly[long_].mean(axis=1) - fwd_monthly[short_].mean(axis=1)


def build_controls(daily_ret, ew_util_daily, xlu_daily, style_inputs, rebal_dates) -> pd.DataFrame:
    monthly = (1 + daily_ret).resample("ME").prod() - 1
    ctrl = pd.DataFrame(index=monthly.index)
    ctrl["util_beta"] = (1 + ew_util_daily).resample("ME").prod() - 1
    ctrl["xlu"] = (1 + xlu_daily).resample("ME").prod() - 1

    char_specs = [
        ("dividend_yield", lambda si: si.set_index("ticker")["dividend_yield"], True),
        ("low_vol", None, False),
        ("size", lambda si: np.log(si.set_index("ticker")["market_cap"]), False),
        ("momentum", None, True),
        ("capex_intensity", lambda si: (si.set_index("ticker")["capex"]
                                        / si.set_index("ticker")["net_ppe"]), True),
        ("value", lambda si: (si.set_index("ticker")["book_value"]
                              / si.set_index("ticker")["market_cap"]), True),
    ]
    for name, fn, hml in char_specs:
        ctrl[name] = 0.0
        for d in rebal_dates:
            fy = d.year - 1
            si = style_inputs[style_inputs["fy"] == fy]
            span = monthly.index[(monthly.index > d) &
                                 (monthly.index <= d + pd.DateOffset(years=1))]
            if len(span) == 0:
                continue
            fwd = monthly.loc[span]
            if name == "low_vol":
                char = daily_ret.loc[:d].iloc[-252:].std()
            elif name == "momentum":
                px = (1 + daily_ret).cumprod()
                char = px.loc[:d].iloc[-1] / px.loc[:d].iloc[-252:-21].iloc[0] - 1
            else:
                char = fn(si).reindex(fwd.columns)
            ctrl.loc[span, name] = _ls_from_char(char.reindex(fwd.columns), fwd, hml).values
    return ctrl[_CONTROL_COLS]


def run_additivity(factor_monthly_ret: pd.Series, controls: pd.DataFrame) -> dict:
    df = pd.concat([factor_monthly_ret.rename("y"), controls], axis=1).dropna()
    X = sm.add_constant(df[_CONTROL_COLS])
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    return {
        "alpha_ann": float(res.params["const"] * 12),
        "alpha_t": float(res.tvalues["const"]),
        "r2": float(res.rsquared),
        "coef": {k: float(v) for k, v in res.params.items() if k != "const"},
        "n": int(res.nobs),
    }
```

**Executor note:** `build_controls` is the fiddliest function in the module — the tests only pin `run_additivity` and the loader. When wiring the real pipeline, sanity-check that each control column has non-zero variance over 2011–2025 and print a correlation matrix into the report; if `style_inputs.csv` coverage forces the additivity window to start after 2013, record that in `docs/transmission-rate-base-results.md`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_additivity.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transmission_rate_base/additivity.py transmission_rate_base/data/style_inputs.csv \
        tests/transmission_rate_base/test_additivity.py
git commit -m "feat(transmission-rate-base): additivity regression (HAC OLS vs style controls)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 12: Pre-registered gate

**Files:**
- Create: `transmission_rate_base/gate.py`
- Create: `tests/transmission_rate_base/test_gate.py`

**Interfaces:**
- Consumes: `config.GATE`; result dicts from earlier tasks.
- Produces:
  - `quintile_inversions(quintile_means: list[float]) -> int` — count of adjacent pairs where the sequence moves against the intended (increasing) direction; the strategy also passes if the reversed sequence is monotone (returns `min(up_inv, down_inv)`).
  - `evaluate_gate(ic: dict, spread: dict, additivity: dict, pre_thesis: dict) -> dict` — returns `{"passed": bool, "conditions": {name: bool}, "reasons": list[str]}` for the four spec §11 conditions. `ic` needs keys `mean_ic, t_stat`; `spread` needs `sharpe, quintile_means (list), positive_years_frac`; `additivity` needs `alpha_ann, alpha_t, r2`; `pre_thesis` needs `t_stat` (one-sided). Conditions 1–3 are hard (any False ⇒ `passed=False`); condition 4 is recorded in `conditions["pre_thesis_sign"]` and `reasons` but does **not** flip `passed`.

- [ ] **Step 1: Write the failing test**

`tests/transmission_rate_base/test_gate.py`:

```python
from transmission_rate_base import gate as G


def _pass_kwargs():
    return dict(
        ic={"mean_ic": 0.05, "t_stat": 3.0},
        spread={"sharpe": 0.6, "quintile_means": [-0.02, -0.01, 0.0, 0.01, 0.03],
                "positive_years_frac": 0.73},
        additivity={"alpha_ann": 0.05, "alpha_t": 2.4, "r2": 0.45},
        pre_thesis={"t_stat": -0.4},
    )


def test_all_pass():
    out = G.evaluate_gate(**_pass_kwargs())
    assert out["passed"] is True
    assert all(out["conditions"][k] for k in ("rank_ic", "quintile_spread", "additivity"))


def test_weak_ic_fails_overall():
    kw = _pass_kwargs(); kw["ic"] = {"mean_ic": 0.01, "t_stat": 1.1}
    out = G.evaluate_gate(**kw)
    assert out["passed"] is False and out["conditions"]["rank_ic"] is False


def test_high_additivity_r2_fails():
    kw = _pass_kwargs(); kw["additivity"] = {"alpha_ann": 0.05, "alpha_t": 2.4, "r2": 0.72}
    assert G.evaluate_gate(**kw)["passed"] is False


def test_pre_thesis_negative_is_recorded_not_fatal():
    kw = _pass_kwargs(); kw["pre_thesis"] = {"t_stat": -3.5}
    out = G.evaluate_gate(**kw)
    assert out["passed"] is True
    assert out["conditions"]["pre_thesis_sign"] is False


def test_quintile_inversions_counts_min_direction():
    assert G.quintile_inversions([1, 2, 3, 4, 5]) == 0
    assert G.quintile_inversions([5, 4, 3, 2, 1]) == 0
    assert G.quintile_inversions([1, 3, 2, 4, 5]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_gate.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

`transmission_rate_base/gate.py`:

```python
from __future__ import annotations

import numpy as np

from transmission_rate_base.config import GATE


def quintile_inversions(quintile_means: list[float]) -> int:
    a = np.asarray(quintile_means, dtype=float)
    d = np.diff(a)
    return int(min((d < 0).sum(), (d > 0).sum()))


def evaluate_gate(ic: dict, spread: dict, additivity: dict, pre_thesis: dict) -> dict:
    reasons: list[str] = []

    rank_ic_ok = (np.isfinite(ic["mean_ic"]) and np.isfinite(ic["t_stat"])
                  and ic["mean_ic"] >= GATE["min_mean_ic"] and ic["t_stat"] >= GATE["min_ic_t"])
    reasons.append(f"rank-IC mean={ic['mean_ic']:.4f} (>= {GATE['min_mean_ic']}), "
                   f"t={ic['t_stat']:.2f} (>= {GATE['min_ic_t']}) -> {'OK' if rank_ic_ok else 'FAIL'}")

    inv = quintile_inversions(spread["quintile_means"])
    spread_ok = (np.isfinite(spread["sharpe"]) and spread["sharpe"] >= GATE["min_ls_sharpe"]
                 and inv <= GATE["max_quintile_inversions"]
                 and spread["positive_years_frac"] >= GATE["min_positive_years_frac"])
    reasons.append(f"Q5-Q1 Sharpe={spread['sharpe']:.2f} (>= {GATE['min_ls_sharpe']}), "
                   f"inversions={inv} (<= {GATE['max_quintile_inversions']}), "
                   f"pos-years={spread['positive_years_frac']:.2f} (>= {GATE['min_positive_years_frac']}) "
                   f"-> {'OK' if spread_ok else 'FAIL'}")

    add_ok = (additivity["alpha_ann"] >= GATE["min_alpha_ann"]
              and additivity["alpha_t"] >= GATE["min_alpha_t"]
              and additivity["r2"] < GATE["max_additivity_r2"])
    reasons.append(f"additivity alpha={additivity['alpha_ann']:.3f}/yr (>= {GATE['min_alpha_ann']}), "
                   f"t={additivity['alpha_t']:.2f} (>= {GATE['min_alpha_t']}), "
                   f"R2={additivity['r2']:.2f} (< {GATE['max_additivity_r2']}) "
                   f"-> {'OK' if add_ok else 'FAIL'}")

    pre_ok = pre_thesis["t_stat"] > GATE["pre_thesis_min_t"]
    reasons.append(f"pre-thesis one-sided t={pre_thesis['t_stat']:.2f} "
                   f"(> {GATE['pre_thesis_min_t']}) -> {'OK' if pre_ok else 'CAVEAT'}")

    passed = bool(rank_ic_ok and spread_ok and add_ok)
    reasons.append(f"GATE {'PASSED' if passed else 'FAILED'}")
    return {
        "passed": passed,
        "conditions": {"rank_ic": rank_ic_ok, "quintile_spread": spread_ok,
                       "additivity": add_ok, "pre_thesis_sign": pre_ok},
        "reasons": reasons,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_gate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add transmission_rate_base/gate.py tests/transmission_rate_base/test_gate.py
git commit -m "feat(transmission-rate-base): pre-registered pass/fail gate

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 13: Pipeline orchestrator + CLI + integration test + gitignore

**Files:**
- Create: `transmission_rate_base/report.py`
- Create: `transmission_rate_base/__main__.py`
- Create: `tests/transmission_rate_base/test_pipeline.py`
- Modify: `.gitignore`
- Modify: `README.md` (sibling-module bullet — placeholder line; final wording set by the results doc)

**Interfaces:**
- Consumes: every module above.
- Produces:
  - `run_pipeline(*, start=config.PRIMARY_START, end=config.PRIMARY_END, offline=False, refresh_ferc=False, pre_thesis=False, out_dir="output_transmission_rate_base", _ferc=None, _prices=None) -> dict`
    — orchestrates: fetch/extract FERC → resolve filers → parent panel → primary signal → neutralise → prices → weights (both books) → backtest → rank-IC → additivity → gate. Writes `signal_panel.csv`, `weights.csv`, `pnl.csv`, `annual_returns.csv`, `ic.csv`, `additivity.csv`, `verdict.txt` under `out_dir`. Returns `{"verdict": <gate dict>, "metrics": ..., "ic": ..., "additivity": ..., "annual_returns": ...}`. `_ferc` / `_prices` are injection seams for the integration test (dicts of DataFrames / a prices module stub); when `None`, the real `data.ferc_form1` and `data.prices` are used.
  - `transmission_rate_base/__main__.py`: argparse for `--start --end --offline --refresh-ferc --pre-thesis --sensitivity`, calls `run_pipeline`, prints `verdict["reasons"]`.

- [ ] **Step 1: Write the failing test**

`tests/transmission_rate_base/test_pipeline.py`:

```python
import numpy as np
import pandas as pd

from transmission_rate_base import report


def _synthetic_ferc():
    # 3 parents, filer ids 1/2/3, 2005..2022; FAST grows tx fast, MID medium, SLOW flat
    rows_204, rows_219, rows_200 = [], [], []
    profiles = {1: ("FAST", 1.18), 2: ("MID", 1.06), 3: ("SLOW", 1.005)}
    for fid, (_, g) in profiles.items():
        tx = 100.0
        for yr in range(2005, 2023):
            rows_204.append(dict(utility_id_ferc1=fid, report_year=yr,
                                 ferc_account_label="transmission_plant",
                                 row_type_xbrl="total", ending_balance=tx))
            rows_219.append(dict(utility_id_ferc1=fid, report_year=yr, plant_function="transmission",
                                 depreciation_type="accumulated_depreciation",
                                 ending_balance=tx * 0.3))
            rows_200.append(dict(utility_id_ferc1=fid, report_year=yr,
                                 ferc_account_label="utility_plant_in_service", ending_balance=tx * 4))
            rows_200.append(dict(utility_id_ferc1=fid, report_year=yr,
                                 ferc_account_label="accumulated_depreciation", ending_balance=tx * 1.2))
            tx *= g
    xwalk = pd.DataFrame([
        dict(utility_id_ferc1=1, utility_name_ferc1="Fast Power Co", utility_id_pudl=1),
        dict(utility_id_ferc1=2, utility_name_ferc1="Mid Power Co", utility_id_pudl=2),
        dict(utility_id_ferc1=3, utility_name_ferc1="Slow Power Co", utility_id_pudl=3),
    ])
    return {"plant_in_service": pd.DataFrame(rows_204), "dep_by_function": pd.DataFrame(rows_219),
            "plant_summary": pd.DataFrame(rows_200), "utility_xwalk": xwalk}


class _StubPrices:
    def __init__(self, tickers, idx):
        self._t, self._i = tickers, idx
    def daily_returns(self, tickers, start, end, *, offline=False):
        rng = np.random.default_rng(7)
        base = pd.DataFrame(rng.normal(0, 0.01, (len(self._i), len(self._t))),
                            index=self._i, columns=self._t)
        base["FAST"] += 0.0004  # FAST outperforms -> signal should be rewarded
        base["SLOW"] -= 0.0004
        return base[[t for t in tickers if t in base.columns]]
    def monthly_returns(self, tickers, start, end, *, offline=False):
        d = self.daily_returns(tickers, start, end)
        return (1 + d).resample("ME").prod() - 1


def test_pipeline_runs_end_to_end_offline(tmp_path, monkeypatch):
    from transmission_rate_base.data import utility_map
    monkeypatch.setattr(utility_map, "PARENT_FILERS",
                        {"FAST": ["Fast Power Co"], "MID": ["Mid Power Co"], "SLOW": ["Slow Power Co"]})
    idx = pd.bdate_range("2005-01-01", "2022-12-31")
    out = report.run_pipeline(start="2011-01-01", end="2022-12-31", offline=True,
                              out_dir=str(tmp_path), _ferc=_synthetic_ferc(),
                              _prices=_StubPrices(["FAST", "MID", "SLOW"], idx))
    assert set(out) >= {"verdict", "metrics", "ic", "additivity", "annual_returns"}
    assert (tmp_path / "verdict.txt").exists()
    assert isinstance(out["verdict"]["passed"], bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_pipeline.py -q`
Expected: FAIL — `report` module missing.

- [ ] **Step 3: Write minimal implementation**

`transmission_rate_base/report.py` (orchestration — wire the functions from Tasks 2–12; keep it linear and print nothing except via `__main__`):

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transmission_rate_base import config, gate as gate_mod
from transmission_rate_base.data import ferc_form1, utility_map
from transmission_rate_base.data import prices as prices_mod
from transmission_rate_base import signal as sig
from transmission_rate_base import portfolio as pf
from transmission_rate_base import backtest as bt
from transmission_rate_base import additivity as ad


def _load_ferc(offline, refresh, injected):
    if injected is not None:
        return injected
    return {k: ferc_form1.fetch_table(k, refresh=refresh, offline=offline)
            for k in config.FERC_TABLES}


def run_pipeline(*, start=config.PRIMARY_START, end=config.PRIMARY_END, offline=False,
                 refresh_ferc=False, pre_thesis=False, out_dir="output_transmission_rate_base",
                 _ferc=None, _prices=None) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    F = _load_ferc(offline, refresh_ferc, _ferc)
    prices = _prices if _prices is not None else prices_mod

    filer_map = utility_map.resolve_filers(F["utility_xwalk"])
    net_tx = ferc_form1.net_transmission_plant(F["plant_in_service"], F["dep_by_function"],
                                               F["plant_summary"])
    totals = ferc_form1.total_net_utility_plant(F["plant_summary"])
    panel = sig.build_parent_panel(net_tx, totals, filer_map)
    raw = sig.primary_signal(panel)
    neutral = sig.neutralize(raw, panel, sig.load_segment_mix())

    tickers = sorted(neutral["ticker"].unique())
    dr = prices.daily_returns(tickers, start, end, offline=offline)
    ew = dr.mean(axis=1)
    w_ls = pf.quintile_ls_weights(neutral, dr, ew)
    w_tilt = pf.long_tilt_weights(neutral, dr)
    mr = prices.monthly_returns(tickers, start, end, offline=offline)

    strat, metrics = bt.run_backtest(w_ls, mr)
    yr = bt.annual_returns(strat)
    rd = pf.rebalance_dates(dr.index, start, end)
    ic = bt.signal_rank_ic(neutral, dr, ew, rd)

    # quintile means for the gate: average annual return by signal quintile
    qmeans = _quintile_means(neutral, mr, rd)

    try:
        xlu = prices.daily_returns(["XLU"], start, end, offline=offline)["XLU"]
    except Exception:
        xlu = ew.copy()
    controls = ad.build_controls(dr, ew, xlu, ad.load_style_inputs(), rd)
    add_res = ad.run_additivity(strat, controls)

    if pre_thesis:
        pt = _pre_thesis_tstat(neutral, prices, offline)
    else:
        pt = {"t_stat": 0.0}

    verdict = gate_mod.evaluate_gate(
        ic={"mean_ic": ic["mean_ic"], "t_stat": ic["t_stat"]},
        spread={"sharpe": metrics["sharpe"], "quintile_means": qmeans,
                "positive_years_frac": metrics["positive_years_frac"]},
        additivity=add_res, pre_thesis=pt)

    neutral.to_csv(out / "signal_panel.csv", index=False)
    w_ls.to_csv(out / "weights.csv")
    strat.rename("strategy").to_csv(out / "pnl.csv")
    yr.rename("annual_return").to_csv(out / "annual_returns.csv")
    ic["ic"].rename("rank_ic").to_csv(out / "ic.csv")
    pd.Series(add_res).to_csv(out / "additivity.csv")
    (out / "verdict.txt").write_text("\n".join(verdict["reasons"]) + "\n")

    return {"verdict": verdict, "metrics": metrics, "ic": ic, "additivity": add_res,
            "annual_returns": yr}


def _quintile_means(neutral, monthly_ret, rebal_dates) -> list[float]:
    sig_by_year = neutral.set_index(["year", "ticker"])["neutral_signal"]
    buckets = {q: [] for q in range(config.QUINTILES)}
    for d in rebal_dates:
        try:
            sy = sig_by_year.loc[d.year - 1].dropna().sort_values()
        except KeyError:
            continue
        span = monthly_ret.index[(monthly_ret.index > d) &
                                 (monthly_ret.index <= d + pd.DateOffset(years=1))]
        if len(span) == 0:
            continue
        labels = pd.qcut(sy.rank(method="first"), config.QUINTILES, labels=False)
        fwd = (1 + monthly_ret.loc[span, sy.index]).prod() - 1
        for q in range(config.QUINTILES):
            names = sy.index[labels == q]
            if len(names):
                buckets[q].append(fwd[names].mean())
    return [float(np.nanmean(buckets[q])) if buckets[q] else np.nan
            for q in range(config.QUINTILES)]


def _pre_thesis_tstat(neutral, prices, offline) -> dict:
    dr = prices.daily_returns(sorted(neutral["ticker"].unique()),
                              config.PRE_THESIS_START, config.PRE_THESIS_END, offline=offline)
    if dr.empty:
        return {"t_stat": 0.0}
    ew = dr.mean(axis=1)
    w = pf.quintile_ls_weights(neutral, dr, ew)
    mr = (1 + dr).resample("ME").prod() - 1
    strat, _ = bt.run_backtest(w, mr)
    if strat.std(ddof=1) == 0 or strat.empty:
        return {"t_stat": 0.0}
    return {"t_stat": float(strat.mean() / strat.std(ddof=1) * np.sqrt(len(strat)))}
```

`transmission_rate_base/__main__.py`:

```python
from __future__ import annotations

import argparse

from transmission_rate_base import config, report


def main() -> None:
    ap = argparse.ArgumentParser(prog="transmission_rate_base")
    ap.add_argument("--start", default=config.PRIMARY_START)
    ap.add_argument("--end", default=config.PRIMARY_END)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--refresh-ferc", action="store_true")
    ap.add_argument("--pre-thesis", action="store_true")
    ap.add_argument("--sensitivity", action="store_true",
                    help="also print gross-vs-net / 5yr-window variants (writeup only)")
    args = ap.parse_args()

    res = report.run_pipeline(start=args.start, end=args.end, offline=args.offline,
                              refresh_ferc=args.refresh_ferc, pre_thesis=args.pre_thesis)
    print("\n".join(res["verdict"]["reasons"]))


if __name__ == "__main__":
    main()
```

Add to `.gitignore` (new line):

```
output_transmission_rate_base/
```

Add to `README.md` under the sibling-module block (after the `grid_demand_factor/` line):

```markdown
>
> `transmission_rate_base/` — proposal B: a gated sector-neutral factor over ~40 US regulated electric utilities, long the fast FERC transmission-rate-base compounders / short the flat ones. Run: `python -m transmission_rate_base --offline`. Status: **built, verdict pending** — see `docs/transmission-rate-base-results.md`.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/transmission_rate_base/test_pipeline.py -q`
Expected: PASS. Then run the full module: `.venv/bin/python -m pytest tests/transmission_rate_base/ -q` — all green.

- [ ] **Step 5: Commit**

```bash
git add transmission_rate_base/report.py transmission_rate_base/__main__.py \
        tests/transmission_rate_base/test_pipeline.py .gitignore README.md
git commit -m "feat(transmission-rate-base): pipeline orchestrator + CLI + integration test

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 14: Live run + results writeup + gate verdict

**Files:**
- Create: `docs/transmission-rate-base-results.md`
- Modify: `README.md` (finalise the sibling-module line to the verdict)
- Modify: `CLAUDE.md` "Future Work" if a follow-up is identified

**Interfaces:** none (analysis task).

- [ ] **Step 1: Complete the hand-collected data**
  - Finish `data/utility_map.py::PARENT_FILERS` for all `config.UNIVERSE_SEED` tickers (spec §5 procedure). Run `python -c "from transmission_rate_base.data import ferc_form1 as f,utility_map as u; u.validate(f.fetch_table('utility_xwalk'))"` — must not raise.
  - Fill `data/segment_mix.csv` (regulated-electric vs total operating revenue, ~40 tickers × the years 2008–2024) from 10-K segment notes.
  - Fill `data/style_inputs.csv` (dividend_yield / market_cap / book_value / capex / net_ppe) as far back as sources allow; record the earliest fully-covered year.

- [ ] **Step 2: Warm the price cache** (when yfinance is reachable)

Run: `.venv/bin/python -c "from transmission_rate_base.data import prices; prices.daily_returns(list(__import__('transmission_rate_base.config', fromlist=['x']).UNIVERSE_SEED)+['XLU'], '2003-01-01','2025-12-31')"`
Expected: cache populated; re-run is instant.

- [ ] **Step 3: Run the pipeline**

Run: `.venv/bin/python -m transmission_rate_base --pre-thesis`
Then also `--offline` to confirm reproducibility from cache.
Expected: prints the four gate lines + `GATE PASSED` / `GATE FAILED`.

- [ ] **Step 4: Write `docs/transmission-rate-base-results.md`**

Cover: universe as finalised; signal definition; the four gate conditions with numbers; the §6 sensitivity variants (gross vs net, 5-yr window, each component alone) as a table; the additivity coefficient table + control correlation matrix; the pre-thesis (2003–2010) result; breadth disclosure (§15); and a recommendation. If any of gate conditions 1–3 failed → headline **NEGATIVE RESULT**, and the README line says so; no strategy config is promoted.

- [ ] **Step 5: Finalise README + commit**

```bash
git add docs/transmission-rate-base-results.md README.md CLAUDE.md \
        transmission_rate_base/data/utility_map.py transmission_rate_base/data/segment_mix.csv \
        transmission_rate_base/data/style_inputs.csv
git commit -m "feat(transmission-rate-base): live run + results writeup + gate verdict

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task(s) |
|---|---|
| §3 Universe (frozen seed, entry rule) | 1 (`UNIVERSE_SEED`), 4 (`PARENT_FILERS`), 5 (≥4-yr entry via history), 14 (finalise) |
| §4 Data layer (3 tables + crosswalk, offline fallback, extraction rules, RTO exclusion, net = gross − dep, pro-rata fallback) | 2 (fetch/cache/offline), 3 (extraction, RTO, net, pro-rata) |
| §5 Utility map (static dict, resolution, validation, hard cases) | 4, 14 |
| §6 Signal (net_tx, tx_share, 3-yr CAGR + 3-yr share change, rank-average, guards, structural break) | 5 (panel), 6 (signal) |
| §7 Neutralisation (winsor → z → regress out size + nonreg-share → residual) | 7 |
| §8 Portfolio (May annual rebalance, PIT, dollar+beta-neutral Q5−Q1, long-only tilt) | 9 |
| §9 Backtest (2011–2025 primary, 2003–2010 pre-thesis, monthly returns, metrics, rank-IC vs fwd 12m relative return, EW + XLU benchmarks, outputs) | 10 (backtest, IC), 13 (orchestration, outputs, pre-thesis), 14 (run) |
| §10 Additivity (8 controls, HAC OLS, residual alpha + t + R²) | 11 |
| §11 Pre-registered gate (4 conditions, 1–3 hard, 4 caveat, fail ⇒ negative result no module) | 1 (`GATE`), 12 (`evaluate_gate`), 13 (wire), 14 (verdict + writeup) |
| §12 Module layout | all; Task 1 scaffolds, each task adds its file |
| §13 Testing plan | every task has unit tests; Task 13 integration test |
| §14 Deliverables | 13 (README placeholder, .gitignore), 14 (results doc, README final, memory) |
| §15 Limitations (breadth, public signal, FERC lag, ITC/FTS, style history) | disclosed in Task 14 writeup; ITC/FTS excluded via `UNIVERSE_SEED` (Task 1) |
| §16 Out of scope | nothing implements these — correct |

No gaps.

**2. Placeholder scan**

- Task 4 and Task 14 explicitly defer completing `PARENT_FILERS` / the CSVs to the executor — these are *data-entry* deliverables with a stated acceptance check (`validate(...)` passes), not code placeholders. Seed rows + schema + procedure are all given. Acceptable and unavoidable (the data is hand-collected from 10-Ks).
- `additivity.build_controls` has an executor note flagging it as the fiddliest piece; its code is complete, and its public contract is pinned by `run_additivity` tests. Acceptable.
- No "TBD"/"handle edge cases"/"similar to Task N"/"write tests for the above" anywhere. Code blocks present on every code step.

**3. Type consistency**

- `fetch_table(key, *, refresh, offline)` — same signature in Tasks 2, 13.
- `resolve_filers(xwalk, parents=None) -> dict[str, list[int]]` — Task 4 defines, Tasks 5/13 consume with that shape.
- Parent panel columns `["ticker","year","net_tx","net_total","tx_share"]` — Task 5 produces, Tasks 6/7/13 consume (`primary_signal` groups by `ticker`, reads `net_tx`/`net_total`/`tx_share`; `neutralize` reads `net_total`).
- `primary_signal(...)` cols `["ticker","year","g3_net_tx","d3_tx_share","raw_signal"]` — Task 6 produces, Task 7 `neutralize` consumes `raw_signal`, adds `neutral_signal`.
- `neutralize(...)` output (adds `neutral_signal`) — Tasks 9/10/13 consume `["ticker","year","neutral_signal"]`.
- `quintile_ls_weights(signal_df, daily_ret, ew_util_ret)` / `long_tilt_weights(signal_df, daily_ret)` — Task 9 defines, Task 13 calls with those args.
- `run_backtest(weights_daily, monthly_ret, rf_annual=0.04) -> (Series, dict)` — Task 10 defines, Task 13 calls; metrics keys `sharpe`, `positive_years_frac` consumed by the gate wiring.
- `signal_rank_ic(neutral_signal_df, daily_ret, ew_util_daily, rebal_dates) -> {"ic","mean_ic","t_stat","n"}` — Task 10 defines, Task 13 consumes `mean_ic`/`t_stat`.
- `build_controls(daily_ret, ew_util_daily, xlu_daily, style_inputs, rebal_dates)` / `run_additivity(factor_monthly_ret, controls) -> {"alpha_ann","alpha_t","r2","coef","n"}` — Task 11 defines, Task 13 consumes.
- `evaluate_gate(ic, spread, additivity, pre_thesis)` with `ic={mean_ic,t_stat}`, `spread={sharpe,quintile_means,positive_years_frac}`, `additivity={alpha_ann,alpha_t,r2}`, `pre_thesis={t_stat}` — Task 12 defines, Task 13 calls with exactly those keys. `_quintile_means` in Task 13 returns the `list[float]` that `spread["quintile_means"]` needs.
- `config.GATE` keys referenced in Task 12 match the keys frozen in Task 1.

Consistent.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-08-31-transmission-rate-base.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

**Which approach?**
