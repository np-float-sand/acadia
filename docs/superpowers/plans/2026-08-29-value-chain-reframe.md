# Grid Equipment Basket — Value-Chain Reframe + Intra-Theme Hedge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a value-chain-tilted long-only basket and a market-neutral maker/contractor pair to `grid_equipment_basket/`, driven by a gross-margin + backlog-coverage signal inside frozen business-model buckets, plus the scoring machinery to judge both against pre-registered gates.

**Architecture:** Three new modules in the existing package. `margin_data.py` fetches point-in-time quarterly fundamentals from SEC XBRL (mirrors `backlog_data.py`). `value_chain.py` holds the frozen buckets, the combined signal, and the two weight builders (`value_chain_tilt_targets` for the long-only tilt, `pair_weights` for the market-neutral pair). `hedges.py` simulates the pair, applies the two overlay hedges, and scores the fixed-span drawdown episode. `backtest.py` and `__main__.py` gain a `value_chain_report` path and a `--construction` flag. Everything reuses the existing `simulate_basket` / `compute_metrics` machinery; no z-scoring, no `build_factor`.

**Tech Stack:** Python 3.11, pandas, numpy, requests (SEC XBRL), yfinance (prices, already wrapped), pytest. Run everything through the project venv: `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-29-value-chain-reframe-design.md`

## Global Constraints

- **Return convention:** simple returns throughout (`pct_change`, `(1+r).prod()`) — this package's convention, unlike `grid_resilience` which uses log returns.
- **No z-scoring, no `build_factor()`, no IC-style metrics, no significance claims** — spec §9 / base spec §12. The signal is plain arithmetic on documented constants.
- **Constants are not optimized.** All tilt / overlay / threshold values live in `config.py` as named constants with the exact values below. The results doc reports the surrounding plateau; it does not search for a best value.
- **Point-in-time discipline:** every fundamental figure is used only on/after its SEC filing date (`availability_date`), never its period-end date. A figure disclosed after a rebalance date is invisible at that rebalance.
- **Buckets are frozen before any backtest** and never revised from results: makers `["ETN", "HUBB", "GEV", "VRT", "NVT"]`, contractors `["PWR", "MYRG", "PRIM", "FLNC"]`.
- **Tests are offline and deterministic** — no network. Stub `requests.get` / price fetches with `monkeypatch`; use CSV/`StringIO`/hand-built DataFrame fixtures. The single networked step is the live run in Task 12, clearly marked.
- **Test runner:** `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q`. The existing 41 tests must stay green after every task.
- **Commit style:** `feat(grid-equipment): <what>` (docs task uses `docs(grid-equipment):`). End every commit message body with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- **File writing:** per repo `CLAUDE.md`, write/modify Markdown (`README.md`, the results doc, `CLAUDE.md`) with Bash heredocs, not an editor tool. Python files use the normal edit tools.
- **README upkeep:** per repo `CLAUDE.md`, when a code change invalidates `grid_equipment_basket/README.md` (new flag, new construction, new data source), update it in the same task.

---

## File Structure

| File | New/modified | Responsibility |
|---|---|---|
| `grid_equipment_basket/config.py` | modified | Add the frozen bucket lists and every value-chain / hedge constant. |
| `grid_equipment_basket/margin_data.py` | **new** | SEC XBRL quarterly fundamentals (`revenue`, `gross_profit`) per ticker, filing-dated; point-in-time TTM gross-margin signal and TTM revenue. |
| `grid_equipment_basket/value_chain.py` | **new** | `bucket_of`, `composite_rank`, `coverage_ratio` / `coverage_change_signal`, `value_chain_tilt_targets` (Construction 1), `pair_weights` (Construction 2). Plain arithmetic on config constants. |
| `grid_equipment_basket/hedges.py` | **new** | `simulate_pair`, `pair_overlay`, `conditional_short_mask`, `conditional_short_overlay`, `find_drawdown_episode`, `episode_drawdown`, `annualized_carry`, `risk_match_weight`. |
| `grid_equipment_basket/backtest.py` | modified | `value_chain_report()` (runs both constructions + both hedges + both gates) and `value_chain_table()` (renders it). |
| `grid_equipment_basket/__main__.py` | modified | `--construction {equal-weight,backlog-tilt,value-chain-tilt,pair}` and `--drop-winners`; dispatch to `value_chain_report`. |
| `grid_equipment_basket/README.md` | modified | Reframe, frozen buckets, both gate outcomes, new caveats. |
| `tests/grid_equipment_basket/test_value_chain.py` | **new** | Buckets, composite rank, coverage signal, tilt targets, pair weights. |
| `tests/grid_equipment_basket/test_margin_data.py` | **new** | XBRL parse + fallbacks, quarterly filtering, filing-date dating, TTM signal math, staleness/span guards. |
| `tests/grid_equipment_basket/test_hedges.py` | **new** | Pair sim, overlays, conditional mask, episode drawdown, carry, risk-match. |
| `tests/grid_equipment_basket/test_backtest.py` | modified | `value_chain_report` dict shape + gate booleans on a synthetic price_fn. |
| `docs/grid-equipment-value-chain-results.md` | **new** | Live-run numbers for both constructions, both gate evaluations, borrow-cost sensitivity, parameter plateau, all caveats. |
| `README.md` (repo root) | modified | One-line pointer to the new construction. |
| `CLAUDE.md` (repo root) | modified | "Future Work" note for the operating-margin signal variants (spec §9). |

### Canonical signatures (every task relies on these being exact)

```python
# grid_equipment_basket/value_chain.py
def bucket_of(ticker: str) -> str: ...                     # "maker" | "contractor"; raises KeyError otherwise
def composite_rank(signal_a: "pd.Series", signal_b: "pd.Series",
                   names: "list[str]") -> "pd.Series": ...  # mean of ascending dense ranks over `names`; NaN if both missing
def coverage_ratio(backlog_df: "pd.DataFrame", fund_df: "pd.DataFrame",
                   asof: "pd.Timestamp") -> "pd.Series": ...          # latest-visible backlog / TTM revenue, per ticker
def coverage_change_signal(backlog_df: "pd.DataFrame", fund_df: "pd.DataFrame",
                           asof: "pd.Timestamp") -> "pd.Series": ...  # coverage_ratio(asof) - coverage_ratio(asof - 365d)
def value_chain_tilt_targets(available: "list[str]", asof: "pd.Timestamp",
                             fund_df: "pd.DataFrame", backlog_df: "pd.DataFrame", *,
                             cap: float = 0.25, base_maker: float = 1.25,
                             base_contractor: float = 0.75, within_top: float = 1.10,
                             within_bottom: float = 0.90) -> "pd.Series": ...  # weights over `available`, sum 1, capped
def pair_weights(available: "list[str]", asof: "pd.Timestamp",
                 fund_df: "pd.DataFrame", backlog_df: "pd.DataFrame"
                 ) -> "tuple[pd.Series, pd.Series]": ...   # (long over makers-in-available sum 1, short over contractors-in-available sum 1)

# grid_equipment_basket/margin_data.py
def fetch_fundamentals(ticker: str, use_cache: bool = True) -> "pd.DataFrame": ...
#   columns: quarter_end, availability_date, revenue, gross_profit  — one row per fiscal quarter, sorted by quarter_end
def combine_fundamentals(by_ticker: "dict[str, pd.DataFrame]") -> "pd.DataFrame": ...
#   same columns + a `ticker` column; concatenated
def ttm_gross_margin_signal(fund_df: "pd.DataFrame", asof: "pd.Timestamp") -> "pd.Series": ...
#   per ticker: (sum gp[-4:]/sum rev[-4:]) - (sum gp[-8:-4]/sum rev[-8:-4]); NaN unless >=8 visible quarters,
#   quarter_end[-1]..quarter_end[-5] span in [300,430] days, and quarter_end[-1] within 200 days of asof
def ttm_revenue(fund_df: "pd.DataFrame", asof: "pd.Timestamp") -> "pd.Series": ...
#   per ticker: sum of revenue over the last 4 visible quarters; NaN unless >=4 visible and latest within 200 days of asof

# grid_equipment_basket/hedges.py
def simulate_pair(prices: "pd.DataFrame", start: str, end: str,
                  fund_df: "pd.DataFrame", backlog_df: "pd.DataFrame",
                  lag_days: int = 42) -> "pd.Series": ...   # daily pair return = long_leg.returns - short_leg.returns
def pair_overlay(base_returns: "pd.Series", pair_returns: "pd.Series", weight: float) -> "pd.Series": ...
def conditional_short_mask(basket_prices: "pd.Series", ma_days: int = 100,
                           vol_days: int = 20, vol_ref_days: int = 252) -> "pd.Series": ...  # daily bool
def conditional_short_overlay(base_returns: "pd.Series", hedge_returns: "pd.Series",
                              mask: "pd.Series", weight: float) -> "pd.Series": ...
def find_drawdown_episode(basket_returns: "pd.Series", peak_window: "tuple[str, str]",
                          trough_end: str) -> "tuple[pd.Timestamp, pd.Timestamp]": ...
def episode_drawdown(returns: "pd.Series", peak_ts: "pd.Timestamp", trough_ts: "pd.Timestamp") -> float: ...
def annualized_carry(returns: "pd.Series") -> float: ...
def risk_match_weight(base_returns: "pd.Series", pair_returns: "pd.Series",
                      target_returns: "pd.Series", lo: float = 0.0, hi: float = 3.0) -> float: ...

# grid_equipment_basket/backtest.py  (additions)
def value_chain_report(start: str, end: str, price_fn=None, drop_winners: bool = False,
                       fund_df=None, backlog_df=None) -> dict: ...
def value_chain_table(report: dict) -> str: ...
```

---

## Task 1: Config constants + `bucket_of` + `composite_rank`

**Files:**
- Modify: `grid_equipment_basket/config.py`
- Create: `grid_equipment_basket/value_chain.py`
- Test: `tests/grid_equipment_basket/test_value_chain.py`

**Interfaces:**
- Consumes: nothing (leaf task).
- Produces: `config.BUCKET_MAKERS`, `config.BUCKET_CONTRACTORS`, and the constants listed in Step 1; `value_chain.bucket_of`, `value_chain.composite_rank` (signatures above).

- [ ] **Step 1: Add constants to `config.py`**

Append to `grid_equipment_basket/config.py`:

```python
# ── Value-chain reframe (spec 2026-08-29) ──────────────────────────────────
# Frozen before any backtest, from 10-K business descriptions only. Never revised from results.
BUCKET_MAKERS: list[str] = ["ETN", "HUBB", "GEV", "VRT", "NVT"]
BUCKET_CONTRACTORS: list[str] = ["PWR", "MYRG", "PRIM", "FLNC"]

# Construction 1 — long-only tilt. Documented constants, not optimized.
VC_BASE_MAKER: float = 1.25
VC_BASE_CONTRACTOR: float = 0.75
VC_WITHIN_TOP: float = 1.10
VC_WITHIN_BOTTOM: float = 0.90

# Signal guards (mirror backlog_data's span / staleness guards).
VC_SIGNAL_MIN_QUARTERS: int = 8      # need 2 full TTM windows for a YoY margin change
VC_SPAN_MIN_DAYS: int = 300          # quarter_end[-1]..quarter_end[-5] lower bound
VC_SPAN_MAX_DAYS: int = 430          # ...upper bound (gappy series -> NaN)
VC_STALENESS_MAX_DAYS: int = 200     # latest disclosed quarter must be this fresh vs asof

# Construction 2 + hedges.
PAIR_OVERLAY_WEIGHT: float = 0.30
COND_SHORT_TICKER: str = "QQQ"
COND_SHORT_WEIGHT: float = 0.30
COND_SHORT_MA_DAYS: int = 100
COND_SHORT_VOL_DAYS: int = 20
COND_SHORT_VOL_REF_DAYS: int = 252

# Fixed drawdown episode for the Gate 2 comparison (spec §7.3): the DeepSeek scare.
DRAWDOWN_PEAK_WINDOW: tuple[str, str] = ("2024-07-01", "2024-12-31")
DRAWDOWN_TROUGH_END: str = "2025-06-30"

# Robustness pass (spec §7.1): makers to drop when --drop-winners is set.
VC_DROP_WINNERS: list[str] = ["VRT", "GEV"]
```

- [ ] **Step 2: Write the failing test**

Create `tests/grid_equipment_basket/test_value_chain.py`:

```python
import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import config, value_chain as vc


def test_buckets_are_the_frozen_split():
    assert sorted(config.BUCKET_MAKERS) == ["ETN", "GEV", "HUBB", "NVT", "VRT"]
    assert sorted(config.BUCKET_CONTRACTORS) == ["FLNC", "MYRG", "PRIM", "PWR"]
    assert set(config.BUCKET_MAKERS) & set(config.BUCKET_CONTRACTORS) == set()


def test_bucket_of_classifies_and_rejects_unknown():
    assert vc.bucket_of("ETN") == "maker"
    assert vc.bucket_of("PWR") == "contractor"
    with pytest.raises(KeyError):
        vc.bucket_of("AAPL")


def test_composite_rank_is_mean_of_ascending_ranks():
    a = pd.Series({"X": 0.05, "Y": -0.02, "Z": 0.10})   # ranks 2, 1, 3
    b = pd.Series({"X": 1.0, "Y": 3.0, "Z": 2.0})        # ranks 1, 3, 2
    out = vc.composite_rank(a, b, ["X", "Y", "Z"])
    assert out["X"] == pytest.approx(1.5)
    assert out["Y"] == pytest.approx(2.0)
    assert out["Z"] == pytest.approx(2.5)


def test_composite_rank_one_missing_component_uses_the_other():
    a = pd.Series({"X": 0.05, "Y": -0.02})               # Z missing from a
    b = pd.Series({"X": 1.0, "Y": 3.0, "Z": 2.0})
    out = vc.composite_rank(a, b, ["X", "Y", "Z"])
    assert not np.isnan(out["Z"])                        # ranked on b alone
    assert np.isnan(vc.composite_rank(pd.Series(dtype=float),
                                     pd.Series(dtype=float), ["Z"])["Z"])
```

- [ ] **Step 3: Run the test, verify it fails**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_value_chain.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'grid_equipment_basket.value_chain'`.

- [ ] **Step 4: Implement `value_chain.py`**

Create `grid_equipment_basket/value_chain.py`:

```python
from __future__ import annotations

"""Value-chain reframe (spec 2026-08-29): frozen maker/contractor buckets, a
gross-margin + backlog-coverage signal, and the two weight builders. Plain
arithmetic on config constants — no z-scoring, no build_factor."""

import numpy as np
import pandas as pd

from grid_equipment_basket import config

_BUCKET = {t: "maker" for t in config.BUCKET_MAKERS}
_BUCKET.update({t: "contractor" for t in config.BUCKET_CONTRACTORS})


def bucket_of(ticker: str) -> str:
    try:
        return _BUCKET[ticker]
    except KeyError:
        raise KeyError(f"{ticker} is not in a frozen value-chain bucket") from None


def composite_rank(signal_a: pd.Series, signal_b: pd.Series, names: list[str]) -> pd.Series:
    ra = signal_a.reindex(names).rank(method="dense", ascending=True)
    rb = signal_b.reindex(names).rank(method="dense", ascending=True)
    both = pd.concat([ra, rb], axis=1)
    return both.mean(axis=1, skipna=True).reindex(names)
```

- [ ] **Step 5: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_value_chain.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Full suite still green**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q`
Expected: `45 passed`.

- [ ] **Step 7: Commit**

```bash
git add grid_equipment_basket/config.py grid_equipment_basket/value_chain.py tests/grid_equipment_basket/test_value_chain.py
git commit -m "$(printf 'feat(grid-equipment): value-chain buckets + composite rank\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 2: `margin_data.fetch_fundamentals` — quarterly XBRL fundamentals

**Files:**
- Create: `grid_equipment_basket/margin_data.py`
- Test: `tests/grid_equipment_basket/test_margin_data.py`

**Interfaces:**
- Consumes: `grid_equipment_basket.backlog_data.CIK_BY_TICKER` (existing dict), `config.CACHE_DIR`.
- Produces: `margin_data.fetch_fundamentals(ticker, use_cache=True) -> DataFrame[quarter_end, availability_date, revenue, gross_profit]`; `margin_data.combine_fundamentals(dict) -> DataFrame` (+ `ticker` column).

**Background the implementer needs:** SEC XBRL `companyconcept` returns *facts*, each with `start`, `end`, `filed`, `val`. Flow concepts (revenue, gross profit) carry BOTH ~90-day quarterly facts AND ~365-day annual facts in the same response — you must keep only facts whose `end - start` is 80–100 days. (The existing `backlog_data._companyconcept` does not do this because RPO is an instant concept with no `start`.) `availability_date` is the `filed` date, never `end`.

- [ ] **Step 1: Write the failing test**

Create `tests/grid_equipment_basket/test_margin_data.py`:

```python
import json

import pandas as pd
import pytest

from grid_equipment_basket import margin_data as md


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def _facts(tag_rows):
    # tag_rows: list of (start, end, filed, val)
    return {"units": {"USD": [
        {"start": s, "end": e, "filed": f, "val": v} for (s, e, f, v) in tag_rows
    ]}}


def test_fetch_fundamentals_keeps_only_quarterly_facts_and_files_dates(monkeypatch):
    revenue = _facts([
        ("2023-01-01", "2023-03-31", "2023-05-01", 1000),   # quarterly - keep
        ("2023-01-01", "2023-12-31", "2024-02-01", 4300),   # annual    - drop
        ("2023-04-01", "2023-06-30", "2023-08-01", 1100),   # quarterly - keep
    ])
    gross = _facts([
        ("2023-01-01", "2023-03-31", "2023-05-01", 300),
        ("2023-04-01", "2023-06-30", "2023-08-01", 350),
    ])

    def fake_get(url, headers=None, timeout=None):
        if "Revenues" in url:
            return _Resp(revenue)
        if "GrossProfit" in url:
            return _Resp(gross)
        return _Resp({"units": {}}, status=404)

    monkeypatch.setattr(md.requests, "get", fake_get)
    df = md.fetch_fundamentals("ETN", use_cache=False)

    assert list(df.columns) == ["quarter_end", "availability_date", "revenue", "gross_profit"]
    assert len(df) == 2
    assert df["quarter_end"].tolist() == [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-06-30")]
    assert df["availability_date"].iloc[0] == pd.Timestamp("2023-05-01")
    assert df["gross_profit"].tolist() == [300.0, 350.0]


def test_fetch_fundamentals_derives_gross_profit_from_cost_when_untagged(monkeypatch):
    revenue = _facts([("2023-01-01", "2023-03-31", "2023-05-01", 1000)])
    cogs = _facts([("2023-01-01", "2023-03-31", "2023-05-01", 820)])

    def fake_get(url, headers=None, timeout=None):
        if "Revenues" in url:
            return _Resp(revenue)
        if "GrossProfit" in url:
            return _Resp({"units": {}}, status=404)
        if "CostOfGoodsAndServicesSold" in url:
            return _Resp(cogs)
        return _Resp({"units": {}}, status=404)

    monkeypatch.setattr(md.requests, "get", fake_get)
    df = md.fetch_fundamentals("PWR", use_cache=False)
    assert df["gross_profit"].iloc[0] == pytest.approx(180.0)   # 1000 - 820


def test_combine_fundamentals_adds_ticker_column():
    a = pd.DataFrame({"quarter_end": [pd.Timestamp("2023-03-31")],
                      "availability_date": [pd.Timestamp("2023-05-01")],
                      "revenue": [1.0], "gross_profit": [0.3]})
    out = md.combine_fundamentals({"ETN": a, "PWR": a})
    assert set(out["ticker"]) == {"ETN", "PWR"}
    assert len(out) == 2
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_margin_data.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'grid_equipment_basket.margin_data'`.

- [ ] **Step 3: Implement `fetch_fundamentals` / `combine_fundamentals`**

Create `grid_equipment_basket/margin_data.py`:

```python
from __future__ import annotations

"""Point-in-time quarterly fundamentals (revenue, gross profit) from SEC XBRL,
for the value-chain gross-margin signal. Mirrors backlog_data.py: structured
companyconcept fetch, filing-date dating, per-ticker parquet cache.

Flow concepts carry both quarterly (~90d) and annual (~365d) facts in one
response — only the quarterly facts are kept."""

import time

import pandas as pd
import requests

from grid_equipment_basket.backlog_data import CIK_BY_TICKER
from grid_equipment_basket.config import CACHE_DIR

_SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}
_COLS = ["quarter_end", "availability_date", "revenue", "gross_profit"]
_REVENUE_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")
_GROSS_TAGS = ("GrossProfit",)
_COST_TAGS = ("CostOfGoodsAndServicesSold", "CostOfRevenue")


def _flow_facts(cik: str, tag: str) -> pd.DataFrame:
    """Quarterly (~90-day) facts for a us-gaap flow concept, filing-dated."""
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    r = requests.get(url, headers=_SEC_HEADERS, timeout=30)
    if r.status_code == 404:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "val"])
    r.raise_for_status()
    rows = []
    for _unit, facts in r.json().get("units", {}).items():
        for f in facts:
            start, end, filed, val = f.get("start"), f.get("end"), f.get("filed"), f.get("val")
            if not (start and end and filed and val is not None):
                continue
            span = (pd.Timestamp(end) - pd.Timestamp(start)).days
            if not (80 <= span <= 100):          # keep quarterly, drop annual/semi
                continue
            rows.append((pd.Timestamp(end), pd.Timestamp(filed), float(val)))
    if not rows:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "val"])
    df = pd.DataFrame(rows, columns=["quarter_end", "availability_date", "val"])
    return (df.sort_values("availability_date")
              .drop_duplicates("quarter_end", keep="first")
              .sort_values("quarter_end")
              .reset_index(drop=True))


def _first_nonempty(cik: str, tags) -> pd.DataFrame:
    for tag in tags:
        df = _flow_facts(cik, tag)
        if not df.empty:
            return df
        time.sleep(0.2)
    return pd.DataFrame(columns=["quarter_end", "availability_date", "val"])


def fetch_fundamentals(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    cache = CACHE_DIR / f"fundamentals_{ticker}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    cik = CIK_BY_TICKER.get(ticker)
    if cik is None:
        raise KeyError(f"no CIK for {ticker}; add it to backlog_data.CIK_BY_TICKER")

    rev = _first_nonempty(cik, _REVENUE_TAGS).rename(columns={"val": "revenue"})
    gross = _flow_facts(cik, "GrossProfit").rename(columns={"val": "gross_profit"})
    if gross.empty:
        cost = _first_nonempty(cik, _COST_TAGS).rename(columns={"val": "cost"})
        if not cost.empty and not rev.empty:
            gross = rev.merge(cost[["quarter_end", "cost"]], on="quarter_end", how="inner")
            gross["gross_profit"] = gross["revenue"] - gross["cost"]
            gross = gross[["quarter_end", "availability_date", "gross_profit"]]

    if rev.empty or gross.empty:
        out = pd.DataFrame(columns=_COLS)
    else:
        out = rev.merge(gross[["quarter_end", "gross_profit"]], on="quarter_end", how="inner")
        out = out[_COLS].sort_values("quarter_end").reset_index(drop=True)

    if use_cache and not out.empty:
        out.to_parquet(cache)
    return out


def combine_fundamentals(by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for tkr, df in by_ticker.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d["ticker"] = tkr
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=_COLS + ["ticker"])
    return pd.concat(parts, ignore_index=True).sort_values(["ticker", "quarter_end"]).reset_index(drop=True)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_margin_data.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Full suite still green**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q`
Expected: `48 passed`.

- [ ] **Step 6: Commit**

```bash
git add grid_equipment_basket/margin_data.py tests/grid_equipment_basket/test_margin_data.py
git commit -m "$(printf 'feat(grid-equipment): SEC XBRL quarterly fundamentals fetch\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 3: `margin_data` — point-in-time TTM gross-margin signal + TTM revenue

**Files:**
- Modify: `grid_equipment_basket/margin_data.py`
- Modify: `tests/grid_equipment_basket/test_margin_data.py`

**Interfaces:**
- Consumes: `config.VC_SIGNAL_MIN_QUARTERS`, `config.VC_SPAN_MIN_DAYS`, `config.VC_SPAN_MAX_DAYS`, `config.VC_STALENESS_MAX_DAYS`; a combined fundamentals frame (`combine_fundamentals` output).
- Produces: `margin_data.ttm_gross_margin_signal(fund_df, asof) -> pd.Series`; `margin_data.ttm_revenue(fund_df, asof) -> pd.Series`.

- [ ] **Step 1: Write the failing test**

Add to `tests/grid_equipment_basket/test_margin_data.py`:

```python
def _fund_rows(ticker, quarters):
    # quarters: list of (quarter_end, availability_date, revenue, gross_profit)
    return pd.DataFrame(
        [(pd.Timestamp(qe), pd.Timestamp(av), float(rv), float(gp)) for qe, av, rv, gp in quarters],
        columns=["quarter_end", "availability_date", "revenue", "gross_profit"],
    ).assign(ticker=ticker)


def _eight_clean_quarters(ticker, margins):
    # margins: 8 quarterly gross-margin fractions, oldest first. Revenue fixed at 1000.
    qe = ["2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
          "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"]
    av = ["2022-11-01", "2023-02-01", "2023-05-01", "2023-08-01",
          "2023-11-01", "2024-02-01", "2024-05-01", "2024-08-01"]
    return _fund_rows(ticker, [(q, a, 1000.0, 1000.0 * m) for q, a, m in zip(qe, av, margins)])


def test_ttm_gross_margin_signal_yoy_change_in_ttm_margin():
    # prior TTM (q1..q4) avg margin 0.20 ; recent TTM (q5..q8) avg margin 0.25 -> +0.05
    df = _eight_clean_quarters("AAA", [0.20, 0.20, 0.20, 0.20, 0.25, 0.25, 0.25, 0.25])
    from grid_equipment_basket import margin_data as md
    sig = md.ttm_gross_margin_signal(df, pd.Timestamp("2024-09-15"))
    assert sig["AAA"] == pytest.approx(0.05)


def test_ttm_gross_margin_signal_needs_eight_quarters():
    df = _eight_clean_quarters("AAA", [0.2] * 8).iloc[:7]
    from grid_equipment_basket import margin_data as md
    sig = md.ttm_gross_margin_signal(df, pd.Timestamp("2024-09-15"))
    assert "AAA" not in sig.dropna().index


def test_ttm_gross_margin_signal_respects_availability_date():
    df = _eight_clean_quarters("AAA", [0.2] * 8)
    from grid_equipment_basket import margin_data as md
    # asof before the 8th quarter's 2024-08-01 filing -> only 7 visible -> NaN
    sig = md.ttm_gross_margin_signal(df, pd.Timestamp("2024-07-15"))
    assert "AAA" not in sig.dropna().index


def test_ttm_gross_margin_signal_stale_latest_quarter_is_nan():
    df = _eight_clean_quarters("AAA", [0.2] * 8)
    from grid_equipment_basket import margin_data as md
    sig = md.ttm_gross_margin_signal(df, pd.Timestamp("2025-06-01"))   # >200d past 2024-06-30
    assert "AAA" not in sig.dropna().index


def test_ttm_revenue_sums_last_four_visible_quarters():
    df = _eight_clean_quarters("AAA", [0.2] * 8)
    from grid_equipment_basket import margin_data as md
    rev = md.ttm_revenue(df, pd.Timestamp("2024-09-15"))
    assert rev["AAA"] == pytest.approx(4000.0)
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_margin_data.py -q`
Expected: FAIL — `AttributeError: module 'grid_equipment_basket.margin_data' has no attribute 'ttm_gross_margin_signal'`.

- [ ] **Step 3: Implement the two signal functions**

Append to `grid_equipment_basket/margin_data.py`:

```python
from grid_equipment_basket import config as _cfg


def _visible(fund_df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    asof = pd.Timestamp(asof)
    return fund_df[fund_df["availability_date"] <= asof]


def ttm_gross_margin_signal(fund_df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    asof = pd.Timestamp(asof)
    out: dict[str, float] = {}
    for tkr, g in _visible(fund_df, asof).groupby("ticker"):
        g = g.sort_values("quarter_end").reset_index(drop=True)
        if len(g) < _cfg.VC_SIGNAL_MIN_QUARTERS:
            out[tkr] = float("nan")
            continue
        span = (g["quarter_end"].iloc[-1] - g["quarter_end"].iloc[-5]).days
        if not (_cfg.VC_SPAN_MIN_DAYS <= span <= _cfg.VC_SPAN_MAX_DAYS):
            out[tkr] = float("nan")
            continue
        if (asof - g["quarter_end"].iloc[-1]).days > _cfg.VC_STALENESS_MAX_DAYS:
            out[tkr] = float("nan")
            continue
        recent = g.iloc[-4:]
        prior = g.iloc[-8:-4]
        m_recent = recent["gross_profit"].sum() / recent["revenue"].sum()
        m_prior = prior["gross_profit"].sum() / prior["revenue"].sum()
        out[tkr] = float(m_recent - m_prior) if prior["revenue"].sum() else float("nan")
    return pd.Series(out, dtype=float)


def ttm_revenue(fund_df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    asof = pd.Timestamp(asof)
    out: dict[str, float] = {}
    for tkr, g in _visible(fund_df, asof).groupby("ticker"):
        g = g.sort_values("quarter_end").reset_index(drop=True)
        if len(g) < 4:
            out[tkr] = float("nan")
            continue
        if (asof - g["quarter_end"].iloc[-1]).days > _cfg.VC_STALENESS_MAX_DAYS:
            out[tkr] = float("nan")
            continue
        out[tkr] = float(g.iloc[-4:]["revenue"].sum())
    return pd.Series(out, dtype=float)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_margin_data.py -q`
Expected: PASS (8 tests total in the file).

- [ ] **Step 5: Full suite still green**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q`
Expected: `53 passed`.

- [ ] **Step 6: Commit**

```bash
git add grid_equipment_basket/margin_data.py tests/grid_equipment_basket/test_margin_data.py
git commit -m "$(printf 'feat(grid-equipment): point-in-time TTM gross-margin signal + TTM revenue\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 4: `value_chain` — backlog-coverage signal

**Files:**
- Modify: `grid_equipment_basket/value_chain.py`
- Modify: `tests/grid_equipment_basket/test_value_chain.py`

**Interfaces:**
- Consumes: `margin_data.ttm_revenue`; `backlog_data.load_backlog_csv` output shape (`ticker, quarter_end, availability_date, metric_value, disclosure_type, ...`); `config.VC_SPAN_MIN_DAYS/MAX_DAYS/STALENESS_MAX_DAYS`.
- Produces: `value_chain.coverage_ratio(backlog_df, fund_df, asof) -> pd.Series`; `value_chain.coverage_change_signal(backlog_df, fund_df, asof) -> pd.Series`.

**Note:** coverage uses the *latest disclosed backlog dollar figure* visible at `asof`. Skip `book_to_bill_only` rows (a ratio, not a level). Reuse the same span/staleness discipline as `backlog_data.backlog_growth_signal`.

- [ ] **Step 1: Write the failing test**

Add to `tests/grid_equipment_basket/test_value_chain.py`:

```python
_BL_HEADER = ("ticker,quarter_end,availability_date,metric_value,metric_unit,"
              "disclosure_type,segment_scope,source_url,notes\n")


def _backlog_df(rows):
    from io import StringIO
    from grid_equipment_basket.backlog_data import load_backlog_csv
    return load_backlog_csv(StringIO(_BL_HEADER + "".join(rows)))


def _bl_row(t, qe, av, val):
    return f"{t},{qe},{av},{val},USD_million,xbrl_rpo,total,http://x,\n"


def _fund_df(ticker, quarterly_rev):
    # quarterly_rev: list of (quarter_end, availability_date, revenue)
    import pandas as pd
    return pd.DataFrame(
        [(pd.Timestamp(qe), pd.Timestamp(av), float(rv), float(rv) * 0.2) for qe, av, rv in quarterly_rev],
        columns=["quarter_end", "availability_date", "revenue", "gross_profit"],
    ).assign(ticker=ticker)


def test_coverage_ratio_is_latest_backlog_over_ttm_revenue():
    bl = _backlog_df([
        _bl_row("PWR", "2024-03-31", "2024-05-01", 8000),
    ])
    fund = _fund_df("PWR", [
        ("2023-06-30", "2023-08-01", 1000), ("2023-09-30", "2023-11-01", 1000),
        ("2023-12-31", "2024-02-01", 1000), ("2024-03-31", "2024-05-01", 1000),
    ])
    out = vc.coverage_ratio(bl, fund, pd.Timestamp("2024-06-01"))
    assert out["PWR"] == pytest.approx(2.0)          # 8000 / 4000


def test_coverage_change_signal_is_yoy_difference_in_coverage():
    bl = _backlog_df([
        _bl_row("PWR", "2023-03-31", "2023-05-01", 4000),
        _bl_row("PWR", "2024-03-31", "2024-05-01", 8000),
    ])
    fund = _fund_df("PWR", [
        ("2022-06-30", "2022-08-01", 1000), ("2022-09-30", "2022-11-01", 1000),
        ("2022-12-31", "2023-02-01", 1000), ("2023-03-31", "2023-05-01", 1000),
        ("2023-06-30", "2023-08-01", 1000), ("2023-09-30", "2023-11-01", 1000),
        ("2023-12-31", "2024-02-01", 1000), ("2024-03-31", "2024-05-01", 1000),
    ])
    out = vc.coverage_change_signal(bl, fund, pd.Timestamp("2024-06-01"))
    # coverage now 8000/4000 = 2.0 ; a year earlier 4000/4000 = 1.0 -> +1.0
    assert out["PWR"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_value_chain.py -q`
Expected: FAIL — `AttributeError: module 'grid_equipment_basket.value_chain' has no attribute 'coverage_ratio'`.

- [ ] **Step 3: Implement coverage**

Append to `grid_equipment_basket/value_chain.py`:

```python
from grid_equipment_basket import margin_data


def coverage_ratio(backlog_df: pd.DataFrame, fund_df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    asof = pd.Timestamp(asof)
    ttm_rev = margin_data.ttm_revenue(fund_df, asof)
    vis = backlog_df[(backlog_df["availability_date"] <= asof)
                     & (backlog_df["disclosure_type"] != "book_to_bill_only")]
    out: dict[str, float] = {}
    for tkr, g in vis.groupby("ticker"):
        g = g.sort_values("quarter_end")
        latest_qe = g["quarter_end"].iloc[-1]
        if (asof - latest_qe).days > config.VC_STALENESS_MAX_DAYS:
            out[tkr] = float("nan")
            continue
        rev = ttm_rev.get(tkr, float("nan"))
        latest_backlog = float(g["metric_value"].iloc[-1])
        out[tkr] = latest_backlog / rev if rev and rev == rev else float("nan")
    return pd.Series(out, dtype=float)


def coverage_change_signal(backlog_df: pd.DataFrame, fund_df: pd.DataFrame,
                           asof: pd.Timestamp) -> pd.Series:
    asof = pd.Timestamp(asof)
    now = coverage_ratio(backlog_df, fund_df, asof)
    prior = coverage_ratio(backlog_df, fund_df, asof - pd.Timedelta(days=365))
    names = sorted(set(now.index) | set(prior.index))
    return (now.reindex(names) - prior.reindex(names))
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_value_chain.py -q`
Expected: PASS (6 tests in the file).

- [ ] **Step 5: Full suite still green**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q`
Expected: `55 passed`.

- [ ] **Step 6: Commit**

```bash
git add grid_equipment_basket/value_chain.py tests/grid_equipment_basket/test_value_chain.py
git commit -m "$(printf 'feat(grid-equipment): backlog-coverage YoY-change signal\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 5: `value_chain_tilt_targets` — Construction 1 (long-only tilt)

**Files:**
- Modify: `grid_equipment_basket/value_chain.py`
- Modify: `tests/grid_equipment_basket/test_value_chain.py`

**Interfaces:**
- Consumes: `basket.apply_cap` (existing); `margin_data.ttm_gross_margin_signal`; `value_chain.coverage_change_signal`; `value_chain.composite_rank`; `value_chain.bucket_of`; `config.VC_BASE_MAKER/VC_BASE_CONTRACTOR/VC_WITHIN_TOP/VC_WITHIN_BOTTOM`.
- Produces: `value_chain.value_chain_tilt_targets(available, asof, fund_df, backlog_df, *, cap=0.25, base_maker=1.25, base_contractor=0.75, within_top=1.10, within_bottom=0.90) -> pd.Series` — weights over `available`, sum to 1, capped. Usable directly as a `simulate_basket` `target_fn` via a 2-arg closure.

**Within-bucket rule:** rank the bucket's names by composite (ascending). With `k` ranked names: the bottom `k//2` get `within_bottom`, the top `k//2` get `within_top`, any middle name (odd `k`) and any name with no composite get `1.0`. Same shape as the existing `basket.backlog_tilt_targets`.

- [ ] **Step 1: Write the failing test**

Add to `tests/grid_equipment_basket/test_value_chain.py`:

```python
def _signal_fixture():
    # 8 clean quarters each; makers get rising margins, contractors flat/falling.
    import pandas as pd
    qe = ["2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
          "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"]
    av = ["2022-11-01", "2023-02-01", "2023-05-01", "2023-08-01",
          "2023-11-01", "2024-02-01", "2024-05-01", "2024-08-01"]

    def rows(t, margins):
        return pd.DataFrame(
            [(pd.Timestamp(q), pd.Timestamp(a), 1000.0, 1000.0 * m) for q, a, m in zip(qe, av, margins)],
            columns=["quarter_end", "availability_date", "revenue", "gross_profit"],
        ).assign(ticker=t)

    fund = pd.concat([
        rows("ETN", [0.20, 0.20, 0.20, 0.20, 0.28, 0.28, 0.28, 0.28]),   # +0.08 strong maker
        rows("HUBB", [0.20, 0.20, 0.20, 0.20, 0.22, 0.22, 0.22, 0.22]),  # +0.02 weak maker
        rows("GEV", [0.20, 0.20, 0.20, 0.20, 0.24, 0.24, 0.24, 0.24]),
        rows("VRT", [0.20, 0.20, 0.20, 0.20, 0.26, 0.26, 0.26, 0.26]),
        rows("NVT", [0.20, 0.20, 0.20, 0.20, 0.21, 0.21, 0.21, 0.21]),
        rows("PWR", [0.20, 0.20, 0.20, 0.20, 0.205, 0.205, 0.205, 0.205]),
        rows("MYRG", [0.20, 0.20, 0.20, 0.20, 0.19, 0.19, 0.19, 0.19]),  # falling -> worst contractor
        rows("PRIM", [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]),
        rows("FLNC", [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]),
    ], ignore_index=True)
    return fund, _backlog_df([])   # empty backlog -> signal rides on gross margin alone


def test_tilt_overweights_makers_and_sums_to_one():
    fund, bl = _signal_fixture()
    names = ["ETN", "HUBB", "GEV", "VRT", "NVT", "PWR", "MYRG", "PRIM", "FLNC"]
    w = vc.value_chain_tilt_targets(names, pd.Timestamp("2024-09-15"), fund, bl, cap=0.25)
    assert w.sum() == pytest.approx(1.0)
    assert w.reindex(config.BUCKET_MAKERS).sum() > w.reindex(config.BUCKET_CONTRACTORS).sum()
    assert w["ETN"] > w["HUBB"]                       # strong maker over weak maker
    assert w["MYRG"] < w["PRIM"]                      # worst contractor cut hardest
    assert w.max() <= 0.25 + 1e-9


def test_tilt_no_signal_name_gets_base_bucket_multiplier_only():
    fund, bl = _signal_fixture()
    # Drop every GEV fundamental row -> GEV has no composite, takes base maker x1.0
    fund = fund[fund["ticker"] != "GEV"]
    names = ["ETN", "HUBB", "GEV", "VRT", "NVT", "PWR", "MYRG", "PRIM", "FLNC"]
    w = vc.value_chain_tilt_targets(names, pd.Timestamp("2024-09-15"), fund, bl, cap=1.0)
    # GEV weight == equal-weight * base_maker / normaliser ; still clearly a maker-side weight
    assert w["GEV"] > w.reindex(config.BUCKET_CONTRACTORS).max()
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_value_chain.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'value_chain_tilt_targets'`.

- [ ] **Step 3: Implement the tilt**

Append to `grid_equipment_basket/value_chain.py`:

```python
from grid_equipment_basket.basket import apply_cap


def _signals(available, asof, fund_df, backlog_df):
    margin_sig = margin_data.ttm_gross_margin_signal(fund_df, asof).reindex(available)
    cover_sig = coverage_change_signal(backlog_df, fund_df, asof).reindex(available)
    return margin_sig, cover_sig


def _within_bucket_factor(comp: pd.Series, within_top: float, within_bottom: float) -> pd.Series:
    factor = pd.Series(1.0, index=comp.index)
    ranked = comp.dropna().sort_values()
    k = len(ranked)
    if k >= 2:
        half = k // 2
        factor.loc[ranked.index[:half]] = within_bottom
        factor.loc[ranked.index[k - half:]] = within_top
    return factor


def value_chain_tilt_targets(available, asof, fund_df, backlog_df, *, cap: float = 0.25,
                             base_maker: float = 1.25, base_contractor: float = 0.75,
                             within_top: float = 1.10, within_bottom: float = 0.90) -> pd.Series:
    names = sorted(available)
    if not names:
        return pd.Series(dtype=float)
    base = pd.Series(1.0 / len(names), index=names)
    margin_sig, cover_sig = _signals(names, asof, fund_df, backlog_df)

    factor = pd.Series(1.0, index=names)
    for bkt, base_mult in (("maker", base_maker), ("contractor", base_contractor)):
        bkt_names = [t for t in names if bucket_of(t) == bkt]
        if not bkt_names:
            continue
        comp = composite_rank(margin_sig, cover_sig, bkt_names)
        within = _within_bucket_factor(comp, within_top, within_bottom)
        factor.loc[bkt_names] = base_mult * within.reindex(bkt_names).fillna(1.0)

    tilted = base * factor
    tilted = tilted / tilted.sum()
    return apply_cap(tilted, cap)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_value_chain.py -q`
Expected: PASS (8 tests in the file).

- [ ] **Step 5: Full suite still green**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q`
Expected: `57 passed`.

- [ ] **Step 6: Commit**

```bash
git add grid_equipment_basket/value_chain.py tests/grid_equipment_basket/test_value_chain.py
git commit -m "$(printf 'feat(grid-equipment): value-chain-tilt long-only weights (Construction 1)\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 6: `pair_weights` — Construction 2 (market-neutral legs)

**Files:**
- Modify: `grid_equipment_basket/value_chain.py`
- Modify: `tests/grid_equipment_basket/test_value_chain.py`

**Interfaces:**
- Consumes: same signal helpers as Task 5.
- Produces: `value_chain.pair_weights(available, asof, fund_df, backlog_df) -> (long_w, short_w)`. `long_w` is over the makers in `available` and sums to 1; `short_w` is over the contractors in `available` and sums to 1. Each leg weights by composite rank: long ∝ `rank`, short ∝ `(max_rank + 1 - rank)`. A name with no composite takes the bucket's mean rank before normalising. An empty bucket returns an empty Series for that leg.

- [ ] **Step 1: Write the failing test**

Add to `tests/grid_equipment_basket/test_value_chain.py`:

```python
def test_pair_weights_long_favours_strong_maker_short_favours_weak_contractor():
    fund, bl = _signal_fixture()
    names = ["ETN", "HUBB", "GEV", "VRT", "NVT", "PWR", "MYRG", "PRIM", "FLNC"]
    long_w, short_w = vc.pair_weights(names, pd.Timestamp("2024-09-15"), fund, bl)
    assert long_w.sum() == pytest.approx(1.0)
    assert short_w.sum() == pytest.approx(1.0)
    assert set(long_w.index) == set(config.BUCKET_MAKERS)
    assert set(short_w.index) == set(config.BUCKET_CONTRACTORS)
    assert long_w["ETN"] == long_w.max()             # strongest maker, biggest long
    assert short_w["MYRG"] == short_w.max()          # weakest contractor, biggest short


def test_pair_weights_empty_bucket_returns_empty_leg():
    fund, bl = _signal_fixture()
    long_w, short_w = vc.pair_weights(["ETN", "VRT"], pd.Timestamp("2024-09-15"), fund, bl)
    assert short_w.empty
    assert long_w.sum() == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_value_chain.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'pair_weights'`.

- [ ] **Step 3: Implement `pair_weights`**

Append to `grid_equipment_basket/value_chain.py`:

```python
def _leg_weights(comp: pd.Series, invert: bool) -> pd.Series:
    if comp.empty:
        return pd.Series(dtype=float)
    filled = comp.fillna(comp.mean())
    if filled.isna().all():                       # no composite anywhere in the bucket
        filled = pd.Series(1.0, index=comp.index)
    score = (filled.max() + 1.0 - filled) if invert else filled
    return score / score.sum()


def pair_weights(available, asof, fund_df, backlog_df):
    names = sorted(available)
    margin_sig, cover_sig = _signals(names, asof, fund_df, backlog_df)
    makers = [t for t in names if bucket_of(t) == "maker"]
    contractors = [t for t in names if bucket_of(t) == "contractor"]
    long_w = _leg_weights(composite_rank(margin_sig, cover_sig, makers), invert=False)
    short_w = _leg_weights(composite_rank(margin_sig, cover_sig, contractors), invert=True)
    return long_w, short_w
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_value_chain.py -q`
Expected: PASS (10 tests in the file).

- [ ] **Step 5: Full suite still green**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q`
Expected: `59 passed`.

- [ ] **Step 6: Commit**

```bash
git add grid_equipment_basket/value_chain.py tests/grid_equipment_basket/test_value_chain.py
git commit -m "$(printf 'feat(grid-equipment): market-neutral pair leg weights (Construction 2)\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 7: `hedges` — pair simulation + pair overlay

**Files:**
- Create: `grid_equipment_basket/hedges.py`
- Test: `tests/grid_equipment_basket/test_hedges.py`

**Interfaces:**
- Consumes: `basket.simulate_basket` (existing; `target_fn(available, asof) -> pd.Series`); `value_chain.pair_weights`, `value_chain.bucket_of`.
- Produces: `hedges.simulate_pair(prices, start, end, fund_df, backlog_df, lag_days=42) -> pd.Series`; `hedges.pair_overlay(base_returns, pair_returns, weight) -> pd.Series`.

**How the pair is simulated:** split `prices` columns into makers and contractors by `bucket_of`. Run `simulate_basket` on each sub-frame with a 2-arg `target_fn` that calls `pair_weights` and takes the matching leg. `pair_returns = long_leg.returns - short_leg.returns`, aligned on the date intersection. Each leg is fully invested (weights sum to 1), so the pair is dollar-neutral by construction.

- [ ] **Step 1: Write the failing test**

Create `tests/grid_equipment_basket/test_hedges.py`:

```python
import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import hedges as hg


def _flat_pair_inputs():
    # 260 business days. Makers rise 0.05%/day, contractors flat.
    idx = pd.bdate_range("2023-01-02", periods=260)
    makers = {t: 100 * np.exp(np.cumsum(np.full(len(idx), 0.0005))) for t in ["ETN", "HUBB", "GEV", "VRT", "NVT"]}
    contr = {t: np.full(len(idx), 100.0) for t in ["PWR", "MYRG", "PRIM", "FLNC"]}
    prices = pd.DataFrame({**makers, **contr}, index=idx)
    return prices, idx


def test_simulate_pair_is_long_minus_short(monkeypatch):
    prices, idx = _flat_pair_inputs()

    # Deterministic legs: equal-weight each bucket.
    def fake_pair_weights(available, asof, fund_df, backlog_df):
        mk = sorted(t for t in available if t in ["ETN", "HUBB", "GEV", "VRT", "NVT"])
        ct = sorted(t for t in available if t in ["PWR", "MYRG", "PRIM", "FLNC"])
        lw = pd.Series(1.0 / len(mk), index=mk) if mk else pd.Series(dtype=float)
        sw = pd.Series(1.0 / len(ct), index=ct) if ct else pd.Series(dtype=float)
        return lw, sw

    monkeypatch.setattr(hg.value_chain, "pair_weights", fake_pair_weights)
    pair = hg.simulate_pair(prices, "2023-01-02", str(idx[-1].date()), fund_df=None, backlog_df=None)
    # long ~ +0.05%/day, short ~ 0 -> pair ~ +0.05%/day
    assert pair.mean() == pytest.approx(0.0005, abs=5e-5)
    assert pair.std() == pytest.approx(0.0, abs=1e-6)


def test_pair_overlay_adds_weighted_pair():
    idx = pd.bdate_range("2023-01-02", periods=5)
    base = pd.Series([0.01, -0.02, 0.00, 0.03, -0.01], index=idx)
    pair = pd.Series([0.02, 0.02, 0.02, 0.02, 0.02], index=idx)
    out = hg.pair_overlay(base, pair, 0.30)
    assert out.iloc[0] == pytest.approx(0.01 + 0.30 * 0.02)
    assert len(out) == 5
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_hedges.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'grid_equipment_basket.hedges'`.

- [ ] **Step 3: Implement `simulate_pair` / `pair_overlay`**

Create `grid_equipment_basket/hedges.py`:

```python
from __future__ import annotations

"""Hedge overlays and drawdown-episode scoring for the value-chain reframe
(spec 2026-08-29 §6, §7). Simple-return convention."""

import numpy as np
import pandas as pd

from grid_equipment_basket import value_chain
from grid_equipment_basket.basket import simulate_basket


def simulate_pair(prices: pd.DataFrame, start: str, end: str,
                  fund_df: pd.DataFrame, backlog_df: pd.DataFrame,
                  lag_days: int = 42) -> pd.Series:
    cols = list(prices.columns)
    makers = [c for c in cols if value_chain.bucket_of(c) == "maker"]
    contractors = [c for c in cols if value_chain.bucket_of(c) == "contractor"]

    def _long_fn(available, asof):
        return value_chain.pair_weights(available, asof, fund_df, backlog_df)[0]

    def _short_fn(available, asof):
        return value_chain.pair_weights(available, asof, fund_df, backlog_df)[1]

    long_leg = simulate_basket(prices[makers], start, end, lag_days, cap=1.0, target_fn=_long_fn)
    short_leg = simulate_basket(prices[contractors], start, end, lag_days, cap=1.0, target_fn=_short_fn)
    pair = long_leg.returns.subtract(short_leg.returns, fill_value=0.0)
    return pair.dropna()


def pair_overlay(base_returns: pd.Series, pair_returns: pd.Series, weight: float) -> pd.Series:
    aligned = pd.concat([base_returns.rename("b"), pair_returns.rename("p")], axis=1)
    return (aligned["b"].fillna(0.0) + weight * aligned["p"].fillna(0.0)).reindex(base_returns.index)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_hedges.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Full suite still green**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q`
Expected: `61 passed`.

- [ ] **Step 6: Commit**

```bash
git add grid_equipment_basket/hedges.py tests/grid_equipment_basket/test_hedges.py
git commit -m "$(printf 'feat(grid-equipment): pair simulation + pair overlay\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 8: `hedges` — conditional QQQ short

**Files:**
- Modify: `grid_equipment_basket/hedges.py`
- Modify: `tests/grid_equipment_basket/test_hedges.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `hedges.conditional_short_mask(basket_prices, ma_days=100, vol_days=20, vol_ref_days=252) -> pd.Series[bool]`; `hedges.conditional_short_overlay(base_returns, hedge_returns, mask, weight) -> pd.Series`.

**Mask rule (spec §7.2):** the condition is evaluated at each **month-end** on the basket's own price series — `close < close.rolling(ma_days).mean()` AND `rv20 > rv20.rolling(vol_ref_days).median()`, where `rv20 = close.pct_change().rolling(vol_days).std()`. The month-end verdict is then held for every trading day of the **following** calendar month (decision known at prior month-end, applied next month). Days before the first evaluable month-end are `False`.

- [ ] **Step 1: Write the failing test**

Add to `tests/grid_equipment_basket/test_hedges.py`:

```python
def test_conditional_short_mask_true_only_when_trend_down_and_vol_high():
    idx = pd.bdate_range("2022-01-03", periods=600)
    # First ~15 months: steady uptrend, low vol. Then a sharp draw + noise spike.
    up = np.linspace(100, 200, 320)
    down = 200 + np.cumsum(np.random.default_rng(0).normal(-0.6, 4.0, len(idx) - 320))
    close = pd.Series(np.concatenate([up, down]), index=idx)

    mask = hg.conditional_short_mask(close, ma_days=100, vol_days=20, vol_ref_days=252)
    assert mask.dtype == bool
    assert not mask.loc["2022-06-01":"2022-09-30"].any()       # calm uptrend -> off
    assert mask.loc["2023-06-01":"2023-12-31"].any()           # drawdown + vol spike -> on somewhere


def test_conditional_short_mask_holds_month_end_verdict_through_next_month():
    idx = pd.bdate_range("2022-01-03", periods=600)
    close = pd.Series(np.linspace(200, 100, len(idx)), index=idx)   # persistent downtrend
    mask = hg.conditional_short_mask(close)
    # within any single month the mask value is constant
    for _, chunk in mask.loc["2023-01-01":"2023-06-30"].groupby(mask.index.to_period("M")):
        assert chunk.nunique() == 1


def test_conditional_short_overlay_subtracts_only_on_masked_days():
    idx = pd.bdate_range("2023-01-02", periods=4)
    base = pd.Series([0.01, 0.01, 0.01, 0.01], index=idx)
    qqq = pd.Series([0.02, 0.02, 0.02, 0.02], index=idx)
    mask = pd.Series([False, True, True, False], index=idx)
    out = hg.conditional_short_overlay(base, qqq, mask, 0.30)
    assert out.tolist() == pytest.approx([0.01, 0.01 - 0.006, 0.01 - 0.006, 0.01])
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_hedges.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'conditional_short_mask'`.

- [ ] **Step 3: Implement the mask + overlay**

Append to `grid_equipment_basket/hedges.py`:

```python
def conditional_short_mask(basket_prices: pd.Series, ma_days: int = 100,
                           vol_days: int = 20, vol_ref_days: int = 252) -> pd.Series:
    close = basket_prices.dropna().astype(float).sort_index()
    ma = close.rolling(ma_days).mean()
    rv = close.pct_change().rolling(vol_days).std()
    rv_ref = rv.rolling(vol_ref_days).median()
    trend_down = close < ma
    vol_high = rv > rv_ref
    daily_cond = (trend_down & vol_high).fillna(False)

    # Month-end verdict, applied to the following calendar month.
    me = daily_cond[daily_cond.index.to_series().groupby(close.index.to_period("M")).transform("max")
                    .eq(0).ne(True)]  # placeholder; replaced below
    month_end_verdict = daily_cond.groupby(close.index.to_period("M")).last()
    applied = month_end_verdict.shift(1)                     # prior month's verdict
    out = pd.Series(False, index=close.index)
    for period, val in applied.items():
        if val is True:
            out.loc[out.index.to_period("M") == period] = True
    return out.astype(bool)
```

**Note to implementer:** delete the dead `me = ...` line — it is shown only so you recognise and remove it. The working logic is `month_end_verdict` → `shift(1)` → broadcast to that month's days.

- [ ] **Step 4: Simplify — replace Step 3's function body with the clean version**

Replace the whole `conditional_short_mask` body with:

```python
def conditional_short_mask(basket_prices: pd.Series, ma_days: int = 100,
                           vol_days: int = 20, vol_ref_days: int = 252) -> pd.Series:
    close = basket_prices.dropna().astype(float).sort_index()
    ma = close.rolling(ma_days).mean()
    rv = close.pct_change().rolling(vol_days).std()
    rv_ref = rv.rolling(vol_ref_days).median()
    daily_cond = ((close < ma) & (rv > rv_ref)).fillna(False)

    periods = close.index.to_period("M")
    month_end_verdict = daily_cond.groupby(periods).last()
    prior_month_verdict = month_end_verdict.shift(1).fillna(False)
    return pd.Series(prior_month_verdict.reindex(periods).to_numpy(), index=close.index).astype(bool)


def conditional_short_overlay(base_returns: pd.Series, hedge_returns: pd.Series,
                              mask: pd.Series, weight: float) -> pd.Series:
    df = pd.concat([base_returns.rename("b"), hedge_returns.rename("h")], axis=1)
    m = mask.reindex(df.index).fillna(False)
    return (df["b"].fillna(0.0) - weight * df["h"].fillna(0.0).where(m, 0.0)).reindex(base_returns.index)
```

- [ ] **Step 5: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_hedges.py -q`
Expected: PASS (5 tests in the file).

- [ ] **Step 6: Full suite still green**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q`
Expected: `64 passed`.

- [ ] **Step 7: Commit**

```bash
git add grid_equipment_basket/hedges.py tests/grid_equipment_basket/test_hedges.py
git commit -m "$(printf 'feat(grid-equipment): conditional QQQ-short overlay\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 9: `hedges` — episode drawdown, carry, risk-match

**Files:**
- Modify: `grid_equipment_basket/hedges.py`
- Modify: `tests/grid_equipment_basket/test_hedges.py`

**Interfaces:**
- Consumes: `backtest.compute_metrics` (existing, for `cagr`).
- Produces: `hedges.find_drawdown_episode(basket_returns, peak_window, trough_end) -> (peak_ts, trough_ts)`; `hedges.episode_drawdown(returns, peak_ts, trough_ts) -> float`; `hedges.annualized_carry(returns) -> float`; `hedges.risk_match_weight(base_returns, pair_returns, target_returns, lo=0.0, hi=3.0) -> float`.

- [ ] **Step 1: Write the failing test**

Add to `tests/grid_equipment_basket/test_hedges.py`:

```python
def test_find_drawdown_episode_picks_peak_in_window_then_trough():
    idx = pd.bdate_range("2024-01-01", "2025-08-01")
    curve = pd.Series(1.0, index=idx)
    curve.loc["2024-11-15"] = np.nan   # marker only; build returns instead
    # Build a return series: flat, then +/- to make a clear Nov-2024 peak and Apr-2025 trough.
    r = pd.Series(0.0, index=idx)
    r.loc["2024-07-01":"2024-11-15"] = 0.001     # rise into a mid-Nov peak
    r.loc["2024-11-18":"2025-04-10"] = -0.002    # fall into an April trough
    r.loc["2025-04-11":] = 0.001                 # recover
    peak, trough = hg.find_drawdown_episode(r, ("2024-07-01", "2024-12-31"), "2025-06-30")
    assert pd.Timestamp("2024-11-01") <= peak <= pd.Timestamp("2024-11-30")
    assert pd.Timestamp("2025-03-15") <= trough <= pd.Timestamp("2025-04-30")


def test_episode_drawdown_is_min_over_the_fixed_span():
    idx = pd.bdate_range("2024-10-01", "2025-05-01")
    r = pd.Series(0.0, index=idx)
    r.loc["2024-11-01":"2025-03-01"] = -0.01
    dd = hg.episode_drawdown(r, pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-01"))
    assert dd < -0.5 and dd > -0.95


def test_annualized_carry_matches_closed_form():
    r = pd.Series([0.001] * 252)
    assert hg.annualized_carry(r) == pytest.approx((1.001 ** 252) - 1, rel=1e-6)


def test_risk_match_weight_scales_pair_to_target_vol():
    rng = np.random.default_rng(3)
    base = pd.Series(rng.normal(0, 0.01, 500))
    pair = pd.Series(rng.normal(0, 0.008, 500))
    target = base - 0.30 * pd.Series(rng.normal(0, 0.02, 500))   # some hedged series
    k = hg.risk_match_weight(base, pair, target)
    lhs = (base + k * pair).std()
    assert lhs == pytest.approx(target.std(), rel=0.02)
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_hedges.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'find_drawdown_episode'`.

- [ ] **Step 3: Implement**

Append to `grid_equipment_basket/hedges.py`:

```python
from grid_equipment_basket.backtest import compute_metrics


def find_drawdown_episode(basket_returns: pd.Series, peak_window, trough_end):
    curve = (1.0 + basket_returns.dropna()).cumprod()
    lo, hi = pd.Timestamp(peak_window[0]), pd.Timestamp(peak_window[1])
    peak_seg = curve.loc[lo:hi]
    if peak_seg.empty:
        raise ValueError(f"no data in peak window {peak_window}")
    peak_ts = peak_seg.idxmax()
    trough_seg = curve.loc[peak_ts:pd.Timestamp(trough_end)]
    trough_ts = trough_seg.idxmin()
    return peak_ts, trough_ts


def episode_drawdown(returns: pd.Series, peak_ts, trough_ts) -> float:
    seg = returns.loc[peak_ts:trough_ts].dropna()
    if len(seg) < 2:
        return float("nan")
    curve = (1.0 + seg).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def annualized_carry(returns: pd.Series) -> float:
    return float(compute_metrics(returns.dropna())["cagr"])


def risk_match_weight(base_returns: pd.Series, pair_returns: pd.Series,
                      target_returns: pd.Series, lo: float = 0.0, hi: float = 3.0) -> float:
    df = pd.concat([base_returns.rename("b"), pair_returns.rename("p")], axis=1).dropna()
    target_vol = float(target_returns.dropna().std())

    def vol(k):
        return float((df["b"] + k * df["p"]).std())

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if vol(mid) < target_vol:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_hedges.py -q`
Expected: PASS (9 tests in the file).

- [ ] **Step 5: Full suite still green**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q`
Expected: `68 passed`.

- [ ] **Step 6: Commit**

```bash
git add grid_equipment_basket/hedges.py tests/grid_equipment_basket/test_hedges.py
git commit -m "$(printf 'feat(grid-equipment): episode-drawdown, carry, risk-match scoring\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 10: `backtest.value_chain_report` + `value_chain_table` (both gates)

**Files:**
- Modify: `grid_equipment_basket/backtest.py`
- Modify: `tests/grid_equipment_basket/test_backtest.py`

**Interfaces:**
- Consumes: `config` (universe, benchmarks, all VC/hedge constants); `basket.simulate_basket`, `basket.equal_weight_targets`; `value_chain.value_chain_tilt_targets`; `hedges.*`; `margin_data.fetch_fundamentals` / `combine_fundamentals`; `backlog_data.load_backlog_csv`; existing `compute_metrics`, `relative_metrics`, `calendar_year_returns`, `_default_price_fn`.
- Produces: `backtest.value_chain_report(start, end, price_fn=None, drop_winners=False, fund_df=None, backlog_df=None) -> dict`; `backtest.value_chain_table(report) -> str`.

**Report dict shape:**

```python
{
  "start": str, "end": str, "drop_winners": bool,
  "equal_weight": {metrics...}, "value_chain_tilt": {metrics...},
  "benchmarks": {b: {"metrics":..., "relative_to_tilt":...} for b in XLI/SPY/XLU/GRID/PAVE},
  "pair_standalone": {metrics..., "carry": float},
  "overlays": {
     "pair_30": {metrics..., "episode_dd": float},
     "cond_short": {metrics..., "episode_dd": float, "carry": float},
     "pair_risk_matched": {metrics..., "episode_dd": float, "match_weight": float},
  },
  "episode": {"peak": Timestamp, "trough": Timestamp, "equal_weight_dd": float},
  "gate1": {"tilt_sharpe": float, "ew_sharpe": float, "tilt_cagr": float, "ew_cagr": float, "passed": bool},
  "gate2": {"pair_episode_dd": float, "cond_episode_dd": float,
            "pair_carry": float, "cond_carry": float, "passed": bool},
}
```

**Gate 1 passed** iff `tilt_sharpe > ew_sharpe and tilt_cagr > ew_cagr`.
**Gate 2 passed** iff `pair_episode_dd >= cond_episode_dd` (less negative = better protection) **and** `pair_carry >= cond_carry`.

- [ ] **Step 1: Write the failing test**

Add to `tests/grid_equipment_basket/test_backtest.py`:

```python
def _vc_price_fn():
    idx = pd.bdate_range("2023-01-02", periods=650)
    rng = np.random.default_rng(7)

    def _prices(tickers, start, end):
        data = {}
        for k, t in enumerate(sorted(tickers)):
            drift = 0.0006 if t in ("ETN", "HUBB", "GEV", "VRT", "NVT") else 0.0003
            data[t] = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, len(idx))))
        return pd.DataFrame(data, index=idx).loc[start:end]

    return _prices, idx


def _vc_fund_backlog():
    # 8 clean quarters for all 9 names; makers rising margin, contractors flat.
    from grid_equipment_basket import config
    from io import StringIO
    from grid_equipment_basket.backlog_data import load_backlog_csv
    qe = ["2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
          "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30",
          "2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30"]
    av = ["2022-11-01", "2023-02-01", "2023-05-01", "2023-08-01",
          "2023-11-01", "2024-02-01", "2024-05-01", "2024-08-01",
          "2024-11-01", "2025-02-01", "2025-05-01", "2025-08-01"]
    rows = []
    for t in config.BUCKET_MAKERS + config.BUCKET_CONTRACTORS:
        rising = t in config.BUCKET_MAKERS
        for i, (q, a) in enumerate(zip(qe, av)):
            m = 0.20 + (0.01 * i if rising else 0.0)
            rows.append((pd.Timestamp(q), pd.Timestamp(a), 1000.0, 1000.0 * m, t))
    fund = pd.DataFrame(rows, columns=["quarter_end", "availability_date", "revenue", "gross_profit", "ticker"])
    bl = load_backlog_csv(StringIO("ticker,quarter_end,availability_date,metric_value,metric_unit,"
                                   "disclosure_type,segment_scope,source_url,notes\n"))
    return fund, bl


def test_value_chain_report_shape_and_gates():
    price_fn, idx = _vc_price_fn()
    fund, bl = _vc_fund_backlog()
    rep = bt.value_chain_report("2023-01-02", str(idx[-1].date()),
                                price_fn=price_fn, fund_df=fund, backlog_df=bl)
    assert set(rep["benchmarks"]) <= {"XLI", "SPY", "XLU", "GRID", "PAVE"}
    assert isinstance(rep["gate1"]["passed"], bool)
    assert isinstance(rep["gate2"]["passed"], bool)
    assert "peak" in rep["episode"] and "trough" in rep["episode"]
    for key in ("pair_30", "cond_short", "pair_risk_matched"):
        assert "episode_dd" in rep["overlays"][key]
    tbl = bt.value_chain_table(rep)
    assert "GATE 1" in tbl and "GATE 2" in tbl


def test_value_chain_report_drop_winners_removes_vrt_gev_from_makers():
    price_fn, idx = _vc_price_fn()
    fund, bl = _vc_fund_backlog()
    rep = bt.value_chain_report("2023-01-02", str(idx[-1].date()), price_fn=price_fn,
                                fund_df=fund, backlog_df=bl, drop_winners=True)
    assert rep["drop_winners"] is True
    assert np.isfinite(rep["value_chain_tilt"]["sharpe"])
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_backtest.py -q`
Expected: FAIL — `AttributeError: module 'grid_equipment_basket.backtest' has no attribute 'value_chain_report'`.

- [ ] **Step 3: Implement `value_chain_report` / `value_chain_table`**

Append to `grid_equipment_basket/backtest.py`:

```python
def _load_vc_inputs(fund_df, backlog_df):
    if fund_df is None:
        from grid_equipment_basket.margin_data import combine_fundamentals, fetch_fundamentals
        fund_df = combine_fundamentals({t: fetch_fundamentals(t) for t in config.UNIVERSE})
    if backlog_df is None:
        from grid_equipment_basket.backlog_data import load_backlog_csv
        backlog_df = load_backlog_csv()
    return fund_df, backlog_df


def value_chain_report(start: str, end: str, price_fn=None, drop_winners: bool = False,
                       fund_df=None, backlog_df=None) -> dict:
    from grid_equipment_basket import hedges, value_chain
    from grid_equipment_basket.basket import simulate_basket

    price_fn = price_fn or _default_price_fn
    fund_df, backlog_df = _load_vc_inputs(fund_df, backlog_df)

    makers = [m for m in config.BUCKET_MAKERS if not (drop_winners and m in config.VC_DROP_WINNERS)]
    universe = makers + config.BUCKET_CONTRACTORS
    tickers = sorted(set(universe + config.BENCHMARKS + [config.COND_SHORT_TICKER]))
    prices = price_fn(tickers, start, end)
    uni_cols = [t for t in universe if t in prices.columns]

    rf, af = config.RISK_FREE_RATE, config.ANN_FACTOR

    ew = simulate_basket(prices[uni_cols], start, end,
                         config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT)
    def _tilt_fn(available, asof):
        return value_chain.value_chain_tilt_targets(
            available, asof, fund_df, backlog_df,
            cap=config.MAX_SINGLE_NAME_WEIGHT,
            base_maker=config.VC_BASE_MAKER, base_contractor=config.VC_BASE_CONTRACTOR,
            within_top=config.VC_WITHIN_TOP, within_bottom=config.VC_WITHIN_BOTTOM,
        )
    tilt = simulate_basket(prices[uni_cols], start, end,
                           config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT, _tilt_fn)

    pair_prices = prices[[c for c in uni_cols]]
    pair = hedges.simulate_pair(pair_prices, start, end, fund_df, backlog_df,
                                config.REBALANCE_LAG_DAYS)

    qqq = prices[config.COND_SHORT_TICKER].pct_change().dropna() if config.COND_SHORT_TICKER in prices else pd.Series(dtype=float)
    ew_curve_prices = (1.0 + ew.returns).cumprod()
    mask = hedges.conditional_short_mask(
        ew_curve_prices, config.COND_SHORT_MA_DAYS, config.COND_SHORT_VOL_DAYS, config.COND_SHORT_VOL_REF_DAYS)

    over_pair = hedges.pair_overlay(ew.returns, pair, config.PAIR_OVERLAY_WEIGHT)
    over_cond = hedges.conditional_short_overlay(ew.returns, qqq, mask, config.COND_SHORT_WEIGHT)
    k = hedges.risk_match_weight(ew.returns, pair, over_cond)
    over_pair_rm = hedges.pair_overlay(ew.returns, pair, k)

    peak, trough = hedges.find_drawdown_episode(
        ew.returns, config.DRAWDOWN_PEAK_WINDOW, config.DRAWDOWN_TROUGH_END)

    bench_ret = prices[[b for b in config.BENCHMARKS if b in prices.columns]].loc[start:end].pct_change().dropna(how="all")

    def M(r):
        return compute_metrics(r, rf, af)

    report = {
        "start": start, "end": end, "drop_winners": drop_winners,
        "makers": makers,
        "equal_weight": M(ew.returns),
        "value_chain_tilt": M(tilt.returns),
        "benchmarks": {
            b: {"metrics": M(bench_ret[b].dropna()),
                "relative_to_tilt": relative_metrics(tilt.returns, bench_ret[b].dropna(), rf, af)}
            for b in bench_ret.columns
        },
        "pair_standalone": {**M(pair), "carry": hedges.annualized_carry(pair)},
        "overlays": {
            "pair_30": {**M(over_pair), "episode_dd": hedges.episode_drawdown(over_pair, peak, trough)},
            "cond_short": {**M(over_cond), "episode_dd": hedges.episode_drawdown(over_cond, peak, trough),
                           "carry": hedges.annualized_carry(-config.COND_SHORT_WEIGHT * qqq.where(mask.reindex(qqq.index).fillna(False), 0.0))},
            "pair_risk_matched": {**M(over_pair_rm), "episode_dd": hedges.episode_drawdown(over_pair_rm, peak, trough),
                                  "match_weight": float(k)},
        },
        "episode": {"peak": peak, "trough": trough,
                    "equal_weight_dd": hedges.episode_drawdown(ew.returns, peak, trough)},
    }
    g1 = {"tilt_sharpe": report["value_chain_tilt"]["sharpe"], "ew_sharpe": report["equal_weight"]["sharpe"],
          "tilt_cagr": report["value_chain_tilt"]["cagr"], "ew_cagr": report["equal_weight"]["cagr"]}
    g1["passed"] = bool(g1["tilt_sharpe"] > g1["ew_sharpe"] and g1["tilt_cagr"] > g1["ew_cagr"])
    g2 = {"pair_episode_dd": report["overlays"]["pair_30"]["episode_dd"],
          "cond_episode_dd": report["overlays"]["cond_short"]["episode_dd"],
          "pair_carry": report["pair_standalone"]["carry"],
          "cond_carry": report["overlays"]["cond_short"]["carry"]}
    g2["passed"] = bool(g2["pair_episode_dd"] >= g2["cond_episode_dd"] and g2["pair_carry"] >= g2["cond_carry"])
    report["gate1"], report["gate2"] = g1, g2
    return report


def value_chain_table(report: dict) -> str:
    L = [f"Value-chain reframe  {report['start']} -> {report['end']}"
         + ("  [drop-winners]" if report["drop_winners"] else ""),
         f"  makers: {', '.join(report['makers'])}",
         f"{'':<22}{'CAGR':>9}{'Vol':>9}{'Sharpe':>9}{'MaxDD':>9}"]
    def line(name, m, extra=""):
        return f"{name:<22}{_p(m['cagr']):>9}{_p(m['ann_vol']):>9}{_f(m['sharpe']):>9}{_p(m['max_dd']):>9}{extra}"
    L.append(line("equal-weight", report["equal_weight"]))
    L.append(line("value-chain tilt", report["value_chain_tilt"]))
    for b, blk in report["benchmarks"].items():
        L.append(line(b, blk["metrics"]))
    L.append("")
    L.append(line("pair (standalone)", report["pair_standalone"],
                  f"  carry {_p(report['pair_standalone']['carry'])}"))
    for k, m in report["overlays"].items():
        L.append(line(f"EW + {k}", m, f"  episodeDD {_p(m['episode_dd'])}"))
    g1, g2 = report["gate1"], report["gate2"]
    L.append("")
    L.append(f"GATE 1 (tilt vs equal-weight):  Sharpe {_f(g1['tilt_sharpe'])} vs {_f(g1['ew_sharpe'])} | "
             f"CAGR {_p(g1['tilt_cagr'])} vs {_p(g1['ew_cagr'])}  ->  {'PASS' if g1['passed'] else 'FAIL'}")
    L.append(f"GATE 2 (pair vs conditional short):  episodeDD {_p(g2['pair_episode_dd'])} vs {_p(g2['cond_episode_dd'])} | "
             f"carry {_p(g2['pair_carry'])} vs {_p(g2['cond_carry'])}  ->  {'PASS' if g2['passed'] else 'FAIL'}")
    return "\n".join(L)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_backtest.py -q`
Expected: PASS (existing backtest tests + 2 new).

- [ ] **Step 5: Full suite still green**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q`
Expected: `70 passed`.

- [ ] **Step 6: Commit**

```bash
git add grid_equipment_basket/backtest.py tests/grid_equipment_basket/test_backtest.py
git commit -m "$(printf 'feat(grid-equipment): value_chain_report with Gate 1 / Gate 2 evaluation\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 11: `__main__` — `--construction` and `--drop-winners`

**Files:**
- Modify: `grid_equipment_basket/__main__.py`
- Modify: `grid_equipment_basket/README.md` (flag list / usage)
- Test: add `tests/grid_equipment_basket/test_cli.py`

**Interfaces:**
- Consumes: `backtest.run`, `backtest.value_chain_report`, `backtest.value_chain_table`, `backtest.results_table`; `config`.
- Produces: CLI dispatch. `--construction` accepts `equal-weight` (default), `backlog-tilt`, `value-chain-tilt`, `pair`. `--tilt backlog` stays as a hidden alias for `--construction backlog-tilt` (back-compat). `--drop-winners` only affects the value-chain constructions.

- [ ] **Step 1: Write the failing test**

Create `tests/grid_equipment_basket/test_cli.py`:

```python
import sys

import pandas as pd
import pytest

from grid_equipment_basket import __main__ as cli


def test_construction_flag_dispatches_to_value_chain_report(monkeypatch, tmp_path, capsys):
    called = {}

    def fake_report(start, end, drop_winners=False, **kw):
        called["start"] = start
        called["drop_winners"] = drop_winners
        return {"_fake": True}

    monkeypatch.setattr(cli.backtest, "value_chain_report", fake_report)
    monkeypatch.setattr(cli.backtest, "value_chain_table", lambda r: "TABLE-OK")
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--construction", "value-chain-tilt", "--drop-winners",
                         "--start", "2023-01-01", "--end", "2024-01-01",
                         "--output", str(tmp_path), "--no-plot"])
    cli.main()
    out = capsys.readouterr().out
    assert "TABLE-OK" in out
    assert called["drop_winners"] is True
    assert called["start"] == "2023-01-01"


def test_tilt_backlog_still_works_as_alias(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(cli.backtest, "run",
                        lambda s, e, target_fn=None, **k: seen.setdefault("target_fn", target_fn) or _fake_run())
    monkeypatch.setattr(cli.backtest, "results_table", lambda r: "")
    monkeypatch.setattr(cli.backtest, "plot", lambda r, p: None)
    monkeypatch.setattr(cli, "_backlog_target_fn", lambda: "BL_FN")
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--tilt", "backlog", "--output", str(tmp_path), "--no-plot"])
    cli.main()
    assert seen["target_fn"] == "BL_FN"


def _fake_run():
    idx = pd.bdate_range("2023-01-02", periods=5)
    return {"basket": {"n_obs": 5, "cagr": 0.1, "ann_vol": 0.2, "sharpe": 0.5,
                       "sortino": 0.6, "max_dd": -0.1, "hit_rate": 0.5},
            "basket_returns": pd.Series([0.0] * 5, index=idx),
            "benchmarks": {}}
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_cli.py -q`
Expected: FAIL — argparse rejects `--construction` (`error: unrecognized arguments`).

- [ ] **Step 3: Rewrite `main()` in `__main__.py`**

Replace the arg parsing + dispatch in `grid_equipment_basket/__main__.py` `main()` with:

```python
def main() -> None:
    ap = argparse.ArgumentParser(prog="grid_equipment_basket")
    ap.add_argument("--start", default=config.PRIMARY_START)
    ap.add_argument("--end", default=_default_end())
    ap.add_argument("--prior-regime", action="store_true",
                    help="run the 2020-2022 panel instead of the primary window")
    ap.add_argument("--construction",
                    choices=["equal-weight", "backlog-tilt", "value-chain-tilt", "pair"],
                    default="equal-weight")
    ap.add_argument("--tilt", choices=["none", "backlog"], default=None,
                    help="deprecated alias: --tilt backlog == --construction backlog-tilt")
    ap.add_argument("--drop-winners", action="store_true",
                    help="value-chain constructions only: drop VRT, GEV from the makers bucket")
    ap.add_argument("--output", default="./output_grid_equipment")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    construction = args.construction
    if args.tilt == "backlog":
        construction = "backlog-tilt"

    start, end = args.start, args.end
    if args.prior_regime:
        start, end = config.PRIOR_REGIME_START, config.PRIOR_REGIME_END

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if construction in ("value-chain-tilt", "pair"):
        rep = backtest.value_chain_report(start, end, drop_winners=args.drop_winners)
        print(backtest.value_chain_table(rep))
        _write_value_chain_outputs(rep, out)
        return

    target_fn = _backlog_target_fn() if construction == "backlog-tilt" else None
    res = backtest.run(start, end, target_fn=target_fn)
    print(backtest.results_table(res))
    rows = [{"name": "BASKET", **res["basket"]}]
    for b, blk in res["benchmarks"].items():
        rows.append({"name": b, **blk["metrics"]})
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    res["basket_returns"].rename("basket_return").to_csv(out / "basket_returns.csv")
    if not args.no_plot:
        backtest.plot(res, str(out / "performance.png"))
        print(f"\nwrote {out}/metrics.csv, basket_returns.csv, performance.png")


def _write_value_chain_outputs(rep: dict, out: Path) -> None:
    rows = [{"name": "equal_weight", **rep["equal_weight"]},
            {"name": "value_chain_tilt", **rep["value_chain_tilt"]},
            {"name": "pair_standalone", **rep["pair_standalone"]}]
    for b, blk in rep["benchmarks"].items():
        rows.append({"name": b, **blk["metrics"]})
    pd.DataFrame(rows).to_csv(out / "value_chain_metrics.csv", index=False)
    import json
    (out / "value_chain_gates.json").write_text(json.dumps(
        {"gate1": rep["gate1"], "gate2": rep["gate2"],
         "episode": {k: str(v) for k, v in rep["episode"].items()}}, indent=2))
    print(f"\nwrote {out}/value_chain_metrics.csv, value_chain_gates.json")
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/test_cli.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Update the README flag/usage section**

In `grid_equipment_basket/README.md`, find the CLI/usage block and add the new flag. Use Bash to rewrite the relevant lines (heredoc per `CLAUDE.md`); the block should now document:

```
python -m grid_equipment_basket [--start ...] [--end ...] [--prior-regime]
    [--construction {equal-weight,backlog-tilt,value-chain-tilt,pair}]
    [--drop-winners]        # value-chain constructions only: drop VRT, GEV from makers
    [--output DIR] [--no-plot]

--tilt backlog is retained as a deprecated alias for --construction backlog-tilt.
```

- [ ] **Step 6: Full suite still green**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q`
Expected: `72 passed`.

- [ ] **Step 7: Commit**

```bash
git add grid_equipment_basket/__main__.py grid_equipment_basket/README.md tests/grid_equipment_basket/test_cli.py
git commit -m "$(printf 'feat(grid-equipment): --construction value-chain-tilt|pair CLI + --drop-winners\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 12: Live run + results doc + README / root-README / CLAUDE.md

**Files:**
- Create: `docs/grid-equipment-value-chain-results.md`
- Modify: `grid_equipment_basket/README.md`
- Modify: `README.md` (repo root)
- Modify: `CLAUDE.md` (repo root)
- No test file — this task's deliverable is verified numbers + docs.

**Interfaces:**
- Consumes: everything built in Tasks 1–11.
- Produces: the results writeup with real numbers; the README/CLAUDE updates.

**This is the only networked task.** It hits SEC XBRL (`fetch_fundamentals` for the 9 names) and yfinance (prices for the 9 names + XLI/SPY/XLU/GRID/PAVE + QQQ). Run from the repo root with the venv.

- [ ] **Step 1: Fetch fundamentals once and eyeball coverage**

Run:

```bash
.venv/bin/python -c "
from grid_equipment_basket.margin_data import fetch_fundamentals, combine_fundamentals
from grid_equipment_basket import config
d = {t: fetch_fundamentals(t, use_cache=True) for t in config.UNIVERSE}
for t, df in d.items():
    print(f'{t:5s} rows={len(df):3d}  span={df.quarter_end.min() if len(df) else None} -> {df.quarter_end.max() if len(df) else None}')
"
```

Expected: most names return ≥ 12 quarterly rows spanning ~2021–2026. Record any name with `< 8` rows or a `GrossProfit` gap — these will have a NaN margin signal and take the base bucket multiplier only. This is the "gross margin diluted / sparse for some names" caveat becoming concrete.

- [ ] **Step 2: Primary-window run**

Run:

```bash
.venv/bin/python -m grid_equipment_basket --construction value-chain-tilt --start 2023-01-01 --end 2026-07-31 --output ./output_grid_equipment_vc --no-plot
```

Copy the printed table (both gate lines) into a scratch file. Repeat with `--construction pair` (same window) — the pair-specific rows and Gate 2 line are what matter there.

- [ ] **Step 3: Prior-regime + drop-winners runs**

Run:

```bash
.venv/bin/python -m grid_equipment_basket --construction value-chain-tilt --prior-regime --output ./output_grid_equipment_vc_prior --no-plot
.venv/bin/python -m grid_equipment_basket --construction value-chain-tilt --start 2023-01-01 --end 2026-07-31 --drop-winners --output ./output_grid_equipment_vc_drop --no-plot
```

- [ ] **Step 4: Borrow-cost sensitivity (one-liner, not a code change)**

Run:

```bash
.venv/bin/python -c "
from grid_equipment_basket import backtest, config
r = backtest.value_chain_report('2023-01-01','2026-07-31')
pc = r['pair_standalone']['carry']; cc = r['overlays']['cond_short']['carry']
print(f'pair carry {pc:.4f}  cond-short carry {cc:.4f}')
print(f'borrow cost that erases the pair-vs-cond carry edge: {(pc-cc)*100:.2f} pp/yr on the short leg')
"
```

Record the number. Note in the writeup that MYRG (~\$20–80M/day) and FLNC (hard/expensive to borrow) are where that cost concentrates.

- [ ] **Step 5: Write `docs/grid-equipment-value-chain-results.md`**

Use a Bash heredoc. Structure (fill every bracket with a real number from Steps 2–4):

```
# Grid Equipment Basket — Value-Chain Reframe: Results

Spec: docs/superpowers/specs/2026-08-29-value-chain-reframe-design.md
Plan: docs/superpowers/plans/2026-08-29-value-chain-reframe.md
Run date: <today>. Prices: yfinance auto-adjusted daily closes. Fundamentals: SEC XBRL companyconcept.

## Buckets (frozen, from 10-K business descriptions — spec §3)
Makers: ETN, HUBB, GEV, VRT, NVT
Contractors/assemblers: PWR, MYRG, PRIM, FLNC   (FLNC borderline — assembler, no pricing power)

## Signal coverage (Step 1 of Task 12)
<per-name quarter counts; which names have a live margin signal, which fall back to base multiplier>

## Construction 1 — value-chain-tilted long-only, primary window 2023-01-01 -> 2026-07-31
<table: equal-weight vs value-chain tilt vs XLI/SPY/XLU/GRID/PAVE — CAGR, Vol, Sharpe, Sortino, MaxDD>

### GATE 1 (spec §7.4)
tilt Sharpe <..> vs equal-weight <..>  |  tilt CAGR <..> vs <..>   ->  PASS / FAIL
<one paragraph: what the tilt did mechanically — which names it over/under-weighted>

## Construction 2 — market-neutral pair, primary window
<table: pair standalone; EW + pair 30%; EW + conditional short; EW + pair risk-matched — metrics + episodeDD>
Episode (fixed span, spec §7.3): peak <date>, trough <date>; equal-weight episode drawdown <..>

### GATE 2 (spec §7.4)
pair episodeDD <..> vs conditional-short episodeDD <..>  |  pair carry <..> vs conditional-short carry <..>   ->  PASS / FAIL
Borrow cost that erases the carry edge: <..> pp/yr (concentrated in MYRG, FLNC).

## Prior-regime panel 2020-2022 (context only — spec §7.1)
<value-chain tilt vs equal-weight; note GEV absent, FLNC partial, VRT SPAC-spliced>

## Drop-winners robustness (drop VRT, GEV from makers — spec §7.1)
<value-chain tilt vs equal-weight; does "makers beat contractors" survive without the two biggest winners?>

## Parameter plateau (spec Global Constraints)
<3-5 rows: base split 1.20/0.80, 1.25/0.75, 1.30/0.70 ; within-bucket 1.05/0.95, 1.10/0.90 — Sharpe/CAGR each>
(Generate by re-running value_chain_report with the constants nudged via a short inline script; do NOT add a search to the module.)

## Caveats (spec §8 — carried + new)
- ~3.56-year / ~43-obs window; +/-0.5 Sharpe standard error; every gap is a point estimate, not significant.
- GEV covers only the back ~60% of the window.
- Headline survivorship bias: universe chosen in 2026 knowing VRT/GEV/PWR won.
- The 5/4 bucket split overlaps the handoff's hindsight-flagged hand-split; mitigations = frozen business-model rule, prior-regime panel, drop-winners pass, PWR (a winner) in the short leg.
- Gross margin diluted for ETN/GEV/PRIM by non-grid segments; operating-margin variant logged for a later spec.
- Short leg not costless — gross results above; borrow-cost sensitivity stated.
- Pair adds fitted-looking DoF; plateau reported, nothing optimized.

## Decision
<one paragraph per gate: keep equal-weight vs adopt tilt; use pair vs use conditional short vs size down>
```

- [ ] **Step 6: Update `grid_equipment_basket/README.md`**

Add a "Value-chain reframe (2026-08-29)" section: the frozen buckets table, the two constructions, the two gate outcomes (with the actual PASS/FAIL and one-line result), and a pointer to `../docs/grid-equipment-value-chain-results.md`. Add the new caveats to the existing caveats section. Bash heredoc.

- [ ] **Step 7: Update repo-root `README.md`**

Add one line under the `grid_equipment_basket/` description: the `--construction value-chain-tilt|pair` option and a pointer to the results doc. Bash.

- [ ] **Step 8: Update repo-root `CLAUDE.md`**

Under "## Future Work — Signal Improvements", add:

```
- **Value-chain signal — operating-margin variants:** the shipped signal uses company-wide
  gross-margin change. Spec 2026-08-29 §9 logs two untried refinements: company-wide operating
  margin (XBRL, all 9 names) and grid-segment operating margin (hand-collected). Try after the
  gross-margin version is judged; check whether they sharpen or muddy the diversified names
  (ETN, GEV, PRIM).
```

Bash.

- [ ] **Step 9: Full suite green + no stray output dirs committed**

Run: `.venv/bin/python -m pytest tests/grid_equipment_basket/ -q` — expect `72 passed`.
Run: `git status` — confirm `output_grid_equipment_vc*` dirs are untracked/gitignored (the repo already gitignores `output_*`; verify).

- [ ] **Step 10: Commit**

```bash
git add docs/grid-equipment-value-chain-results.md grid_equipment_basket/README.md README.md CLAUDE.md
git commit -m "$(printf 'docs(grid-equipment): value-chain reframe results + gate outcomes\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task(s) |
|---|---|
| §1 thesis / reframe | Task 12 writeup framing; Task 1 buckets |
| §2 two products side by side | Tasks 5, 6, 10, 11 |
| §3 frozen 5/4 buckets, FLNC borderline, hindsight disclosure | Task 1 (`config`, `bucket_of`); Task 12 (disclosure in writeup + README) |
| §4.1 TTM gross-margin change, XBRL + derive-from-cost fallback, operating-margin logged | Tasks 2, 3; Task 12 Step 8 (CLAUDE.md log) |
| §4.2 backlog-coverage change, reuse `backlog_quarterly.csv`, annual-only names carry forward | Task 4 (annual-only → stale-guard NaN → base multiplier, noted Task 12 Step 1) |
| §4.3 combine by rank-average, no z-scoring | Task 1 `composite_rank` |
| §5 Construction 1: base ×1.25/×0.75, within ×1.10/×0.90, renormalize, 25% cap, no-signal → base only, odd median ×1.0 | Task 5 |
| §6 Construction 2: dollar-neutral, standalone / 30% overlay / risk-matched; gross results + borrow sensitivity | Tasks 6, 7, 9, 10; Task 12 Step 4 |
| §7.1 windows: primary, prior-regime panel, drop-VRT/GEV | Task 10 (`drop_winners`), Task 11 (`--prior-regime` already exists, `--drop-winners`), Task 12 Steps 2–3 |
| §7.2 conditional-QQQ-short comparator | Task 8 |
| §7.3 fixed peak/trough episode drawdown | Task 9 (`find_drawdown_episode`, `episode_drawdown`), config `DRAWDOWN_*` |
| §7.4 Gate 1 + Gate 2 definitions | Task 10 (`gate1`/`gate2` dict + booleans) |
| §7.5 hindsight controls stated | Task 12 writeup + README |
| §8 caveats carried + new | Task 12 writeup + README |
| §9 out of scope / next steps | Task 12 Step 8 |
| §10 module layout | Tasks 1–11 file-by-file |
| §11 testing list | test files in Tasks 1–11 (buckets, margin math, coverage, tilt, pair, conditional mask, episode DD, point-in-time exclusion) |
| §12 deliverables | Task 12 |

No gaps.

**2. Placeholder scan**

- Task 8 Step 3 deliberately shows a dead line (`me = ...`) then Step 4 replaces the whole function with the clean version — this is a guided refactor, not a placeholder; the final code is complete. Acceptable, and called out explicitly.
- Task 12's results-doc template has `<brackets>` — these are *data to be filled from live-run output*, which cannot exist until the code runs. Each bracket names exactly which run/step produces it. This is the one place a plan legitimately defers content to execution.
- No "TBD", no "add error handling", no "similar to Task N", no undefined symbols.

**3. Type consistency**

- `target_fn` signature is `(available, asof) -> pd.Series` everywhere (`basket.simulate_basket`, `value_chain_tilt_targets` via `_tilt_fn`, `simulate_pair` via `_long_fn`/`_short_fn`). ✔
- `fund_df` is always the *combined* frame (has a `ticker` column) wherever `ttm_gross_margin_signal` / `ttm_revenue` / `coverage_*` consume it; `fetch_fundamentals` returns the per-ticker frame and `combine_fundamentals` adds `ticker`. Task 10 calls `combine_fundamentals(...)`; Task 12 scripts call it too. ✔
- `pair_weights` returns `(long_w, short_w)`; `simulate_pair` indexes `[0]` / `[1]`; Task 6 tests assert both sum to 1. ✔
- `compute_metrics` dict keys used downstream: `cagr`, `ann_vol`, `sharpe`, `sortino`, `max_dd`, `n_obs` — all present in the existing function (confirmed against `backtest.py`). `annualized_carry` reads `["cagr"]`. ✔
- `_p` / `_f` formatting helpers exist in `backtest.py` and are reused by `value_chain_table`. ✔
- `config.UNIVERSE` (existing) vs `BUCKET_MAKERS + BUCKET_CONTRACTORS` (new): they are the same 9 tickers in a different order. `value_chain_report` builds `universe = makers + contractors` and never assumes ordering matches `config.UNIVERSE`. ✔
- Gate 2 direction: `pair_episode_dd >= cond_episode_dd` with drawdowns being negative numbers means "pair's drawdown is less deep (protects at least as well)". Matches spec §7.4 wording. ✔

Fixes applied inline: none needed beyond the Task 8 guided-refactor note, which is intentional.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-08-29-value-chain-reframe.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
