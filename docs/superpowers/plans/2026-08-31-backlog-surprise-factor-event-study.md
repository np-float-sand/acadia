# Backlog Growth Surprise — Event Study (Phase 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `backlog_factor/` module up to and including the pre-registered event-study gate — measure whether a name's backlog/RPO growth *surprise* (actual minus its own trailing trend) predicts post-disclosure abnormal returns, across a wide cross-section of industrials.

**Architecture:** New top-level module `backlog_factor/`. `data/rpo.py` discovers and fetches XBRL `RevenueRemainingPerformanceObligation` per name (mirrors `grid_equipment_basket/backlog_data.py`). `data/prices.py` fetches adjusted closes for the universe + industry-group ETFs + SPY (mirrors `grid_equipment_basket/data/prices.py`). `signal.py` builds the per-name surprise series and the cross-sectional transforms (winsorize / z-score / industry+size neutralization / staleness decay), reusing `grid_resilience.factor.neutralize`. `event_study.py` computes cumulative-abnormal-return (CAR) windows, quintile CAR tables, month-clustered t-stats, and the gate evaluation. `__main__.py` runs it and writes the results doc. **This plan stops at the gate.** If the gate passes, a follow-on plan builds `factor.py` + `backtest.py` + the §7 validation; if it fails, the deliverable is the written negative result.

**Tech Stack:** Python 3.11, pandas, numpy, requests (SEC XBRL), yfinance (prices, via the existing `grid_resilience` cache helpers), pytest. Run everything through the project venv: `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-31-backlog-surprise-factor-design.md`

## Global Constraints

- **Return convention:** simple returns for portfolio maths (`pct_change`, `(1+r).prod()`); the event study's CAR is a simple-return cumulative sum of *abnormal* daily returns (raw minus benchmark). Do NOT mix in log returns.
- **Point-in-time discipline:** every RPO figure is used only on/after its SEC filing date (`availability_date`), never its period-end date. Day 0 of an event window is the **first trading day strictly after** the filing date.
- **No fitted expectation model.** `expected_g` is the trailing 4-quarter mean of `g` — a fixed formula. The only parameters are documented constants in `config.py`.
- **Cross-sectional z-scoring IS allowed here** (this is a genuine cross-sectional factor, unlike the `grid_equipment_basket` basket module). Reuse `grid_resilience.factor.neutralize.cross_section_zscore` / `winsorize`. But the standardize / neutralize / decay steps stay separate and individually unit-tested — no opaque blob.
- **Pre-registered gate (spec §5), evaluated once, reported whatever the outcome:** Q5−Q1 abnormal CAR at its peak window is positive AND monotone across quintiles AND month-clustered t > 2; same sign and clustered t > 1 in BOTH sub-period halves; right sign in ≥ 3 of the 5 major industry groups. A FAIL is a valid deliverable — do not tune constants or re-pick a window to make it pass.
- **Guards** (mirror `grid_equipment_basket/backlog_data.py`): a name's surprise is NaN unless it has ≥ `HISTORY_MIN_QUARTERS` (6) disclosed quarters, its `iloc[-1]..iloc[-5]` quarter-end span is within `[SPAN_MIN_DAYS, SPAN_MAX_DAYS]` = [300, 430] days, and its latest disclosed quarter is < `STALENESS_MAX_DAYS` (200) days before the as-of date.
- **Tests are offline and deterministic** — stub `requests.get` / the price fetch with `monkeypatch`; use hand-built DataFrame / CSV / `StringIO` fixtures. The only networked steps are Task 1's universe discovery and Task 7's live run, both clearly marked.
- **Test runner:** `.venv/bin/python -m pytest tests/backlog_factor/ -q`. Never `python` (no pandas on PATH). Add `tests/backlog_factor/__init__.py`.
- **SEC requests** carry `User-Agent: acadia-research sand.gh1902@gmail.com` (`config.SEC_HEADERS`). Sleep 0.2s between per-name concept calls (mirror `backlog_data`).
- **Commit style:** `feat(backlog-factor): <what>` (`docs(backlog-factor):` for the results task). End every commit body with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- **File writing:** per repo `CLAUDE.md`, write/modify Markdown (`README.md`, the results doc, `candidate_research.md`, root `README.md`) with Bash heredocs, not an editor tool. Python files use the normal edit tools.

---

## File Structure

| File | New/modified | Responsibility |
|---|---|---|
| `backlog_factor/__init__.py` | **new** | empty package marker |
| `backlog_factor/config.py` | **new** | Phase-1 universe (`UNIVERSE` list of `{ticker, cik, industry_group, name}`), `INDUSTRY_GROUP_ETF` map, `STRUCTURED_START`, `HISTORY_MIN_QUARTERS`, span/staleness guards, `DRIFT_HORIZON_DAYS` (provisional 63), `CAR_WINDOWS`, `WINSOR_SIGMA`, `COST_BPS`, `RISK_FREE_RATE`, `ANN_FACTOR`, `SEC_HEADERS`, `CACHE_DIR`. Constants only, no functions. |
| `backlog_factor/data/__init__.py` | **new** | empty |
| `backlog_factor/data/rpo.py` | **new** | `discover_rpo_filers()` (SEC `frames` API → CIKs → `submissions` SIC filter), `fetch_rpo()` (one name's RPO series, mirrors `backlog_data.fetch_rpo`), `load_rpo_panel()` (long panel for the universe). Per-CIK parquet cache. |
| `backlog_factor/data/prices.py` | **new** | `fetch_prices()` — adjusted closes for a ticker list; wraps `grid_resilience.data.cache_utils` monthly-parquet helpers exactly like `grid_equipment_basket/data/prices.py`. |
| `backlog_factor/signal.py` | **new** | `name_surprise_series()` (per name: `g`, `expected_g`, `surprise` + guards), `surprise_panel()` (long panel across the universe), `neutralize_cross_section()` (winsorize → z-score → regress out industry dummies + log-mktcap → residual), `staleness_decay()`, `signal_on_date()`. |
| `backlog_factor/event_study.py` | **new** | `event_pairs()` (one row per disclosure: surprise, month, industry_group, abnormal CAR per window), `quintile_car_table()`, `clustered_tstat()`, `monotonic()`, `subperiod_split()`, `by_industry_signs()`, `evaluate_gate()`. |
| `backlog_factor/__main__.py` | **new** | `python -m backlog_factor --event-study [--start] [--end]` — runs the pipeline, prints the quintile CAR table + gate verdict, writes `output_backlog_factor/`. |
| `backlog_factor/candidate_research.md` | **new** | per-name inclusion log from Task 1. |
| `backlog_factor/README.md` | **new** | universe, signal definition, event-study result + gate outcome, §8 caveats. |
| `tests/backlog_factor/__init__.py` | **new** | empty |
| `tests/backlog_factor/test_rpo.py` | **new** | discovery parse, `fetch_rpo` XBRL parse + Current/Noncurrent merge, filing-date dating, cache. |
| `tests/backlog_factor/test_signal.py` | **new** | `g`/`expected_g`/`surprise` math, guards, winsorize, z-score, regression neutralization orthogonality, staleness decay, definition-break screen. |
| `tests/backlog_factor/test_event_study.py` | **new** | CAR window, day-0 = first day after filing, quintile assignment, clustered t-stat vs naive, monotonicity, gate evaluation on constructed panels (pass and fail). |
| `docs/backlog-surprise-factor-results.md` | **new** | Task 7: event-study numbers + the gate verdict. |
| `README.md` (repo root) | **modified** | one-line pointer to `backlog_factor/`. |

### Canonical signatures (every task relies on these being exact)

```python
# backlog_factor/data/rpo.py
def discover_rpo_filers(quarters: "list[str]", use_cache: bool = True) -> "pd.DataFrame": ...
#   columns: cik (10-digit str), ticker, sic (str), sic_description, company
def fetch_rpo(ticker: str, cik: str, use_cache: bool = True) -> "pd.DataFrame": ...
#   columns: quarter_end, availability_date, metric_value  — one row per fiscal quarter, sorted by quarter_end
def load_rpo_panel(universe: "list[dict]", use_cache: bool = True) -> "pd.DataFrame": ...
#   long: ticker, quarter_end, availability_date, metric_value

# backlog_factor/data/prices.py
def fetch_prices(tickers: "list[str]", start: str, end: str, use_cache: bool = True) -> "pd.DataFrame": ...
#   (date x ticker) adjusted close, sorted index, only columns that returned data

# backlog_factor/signal.py
def name_surprise_series(rpo_one: "pd.DataFrame", *, min_quarters: int = 6,
                         span_min: int = 300, span_max: int = 430) -> "pd.DataFrame": ...
#   input: one ticker's rows (quarter_end, availability_date, metric_value)
#   output: columns [availability_date, g, expected_g, surprise] — one row per disclosure that clears the
#           history + span guards; g = log(mv_t / mv_{t-1}); expected_g = mean of the prior 4 g's;
#           surprise = g - expected_g. A level discontinuity (|g| > log(3)) NaNs surprise for that row and
#           the next 4 rows.
def surprise_panel(rpo_panel: "pd.DataFrame", *, min_quarters: int = 6,
                   span_min: int = 300, span_max: int = 430) -> "pd.DataFrame": ...
#   long: ticker, availability_date, surprise  (drops NaN-surprise rows)
def neutralize_cross_section(raw: "pd.Series", industry_group: "pd.Series", log_mktcap: "pd.Series",
                             winsor_sigma: float = 3.0) -> "pd.Series": ...
#   winsorize at +/- winsor_sigma*std -> cross_section_zscore -> OLS residual on industry dummies + log_mktcap
def staleness_decay(days_since: "pd.Series | float", horizon_days: int) -> "pd.Series | float": ...
#   linear 1.0 at 0 days -> 0.0 at horizon_days; 0.0 beyond; clip negatives to 0
def signal_on_date(sp: "pd.DataFrame", asof: "pd.Timestamp", *, industry_group_map: "dict[str, str]",
                   log_mktcap: "pd.Series", horizon_days: int, staleness_max_days: int = 200,
                   winsor_sigma: float = 3.0) -> "pd.Series": ...
#   per ticker: most-recent surprise visible at asof, within staleness_max_days, decayed, then
#   neutralize_cross_section over the names present. Returns a Series indexed by ticker.

# backlog_factor/event_study.py
def event_pairs(sp: "pd.DataFrame", prices: "pd.DataFrame", group_etf_prices: "pd.DataFrame",
                industry_group_map: "dict[str, str]", group_etf_map: "dict[str, str]",
                car_windows: "list[int]") -> "pd.DataFrame": ...
#   one row per (ticker, availability_date): ticker, availability_date, month (Period[M]), industry_group,
#   surprise, and car_<w> for each w in car_windows  (abnormal: name simple return minus the name's
#   industry-group ETF simple return, cumulative-summed over the window starting the first trading day
#   strictly after availability_date)
def quintile_car_table(pairs: "pd.DataFrame", car_windows: "list[int]") -> "pd.DataFrame": ...
#   index 1..5 (+ a 'Q5-Q1' row), columns car_<w>_mean ; quintiles cut cross-sectionally within each month
#   then pooled
def clustered_tstat(pairs: "pd.DataFrame", value_col: str, cluster_col: str = "month") -> float: ...
#   t-stat of the mean monthly Q5-Q1 spread in value_col, SE = std(monthly means)/sqrt(n_months)
def monotonic(quintile_means: "list | pd.Series") -> bool: ...
def subperiod_split(pairs: "pd.DataFrame") -> "tuple[pd.DataFrame, pd.DataFrame]": ...   # by median month
def by_industry_signs(pairs: "pd.DataFrame", value_col: str) -> "pd.Series": ...   # {industry_group: sign}
def evaluate_gate(pairs: "pd.DataFrame", car_windows: "list[int]") -> dict: ...
#   {peak_window, q5_q1_car, monotone, tstat, sub1_sign, sub1_tstat, sub2_sign, sub2_tstat,
#    industry_signs, n_groups_right_sign, passed (bool), drift_horizon_days}
```

---

## Task 1: Config + Phase-1 universe discovery & curation

**Files:**
- Create: `backlog_factor/__init__.py`, `backlog_factor/data/__init__.py`, `backlog_factor/config.py`, `tests/backlog_factor/__init__.py`
- Create: `backlog_factor/data/rpo.py` (only `discover_rpo_filers` in this task)
- Create: `backlog_factor/candidate_research.md`
- Test: `tests/backlog_factor/test_rpo.py` (discovery parse only)

**Interfaces:**
- Consumes: SEC `frames` + `submissions` APIs.
- Produces: `config.UNIVERSE` (list of `{ticker, cik, industry_group, name}`), `config.INDUSTRY_GROUP_ETF`, all constants in Step 1; `data/rpo.discover_rpo_filers()`.

- [ ] **Step 1: Write `config.py`**

```python
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}

# Discovery: quarters (instant-period frame keys) to union RPO filers from.
DISCOVERY_QUARTERS: list[str] = ["CY2021Q4I", "CY2022Q4I", "CY2023Q4I", "CY2024Q4I", "CY2025Q4I"]

# SIC prefixes for order-driven manufacturers / contractors (industrials & capital goods).
IN_SCOPE_SIC_PREFIXES: tuple[str, ...] = (
    "15", "16", "17",              # construction / heavy construction / special trade
    "35",                          # industrial & commercial machinery
    "36",                          # electronic & electrical equipment
    "37",                          # transportation equipment (incl. aerospace, rail, defense)
    "34",                          # fabricated metal products
    "38",                          # instruments (incl. some measurement / control)
)

STRUCTURED_START: str = "2019-01-01"
HOLDOUT_MONTHS: int = 18

DRIFT_HORIZON_DAYS: int = 63       # provisional; RESET from the event study's peak window
CAR_WINDOWS: list[int] = [5, 21, 42, 63]
QUINTILE_Q: int = 5
WINSOR_SIGMA: float = 3.0
COST_BPS: float = 10.0

HISTORY_MIN_QUARTERS: int = 6
SPAN_MIN_DAYS: int = 300
SPAN_MAX_DAYS: int = 430
STALENESS_MAX_DAYS: int = 200

RISK_FREE_RATE: float = 0.04
ANN_FACTOR: int = 252

# Populated by Task 1's curation step (see candidate_research.md).
UNIVERSE: list[dict] = []          # [{"ticker": "PWR", "cik": "0001050915",
                                   #   "industry_group": "engineering_construction", "name": "Quanta Services"}]

# Hand-curated: industry group -> a liquid ETF proxy for abnormal-return benchmarking.
INDUSTRY_GROUP_ETF: dict[str, str] = {
    "machinery": "XLI",
    "electrical_equipment": "XLI",
    "engineering_construction": "PAVE",
    "aerospace_defense": "ITA",
    "building_products": "XHB",
    "semiconductor_equipment": "SOXX",
}
```

- [ ] **Step 2: Write the failing test** (`tests/backlog_factor/test_rpo.py`)

```python
import pandas as pd
import pytest

from backlog_factor.data import rpo


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status
    def json(self):
        return self._p
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_discover_rpo_filers_unions_frames_and_filters_by_sic(monkeypatch):
    frame_q4_2023 = {"data": [
        {"cik": 1050915, "entityName": "QUANTA SERVICES INC", "val": 1},
        {"cik": 320193, "entityName": "APPLE INC", "val": 1},
    ]}
    frame_q4_2024 = {"data": [
        {"cik": 1050915, "entityName": "QUANTA SERVICES INC", "val": 1},
        {"cik": 37996, "entityName": "FORD MOTOR CO", "val": 1},
    ]}
    subs = {
        "1050915": {"sic": "1731", "sicDescription": "Electrical Work", "tickers": ["PWR"], "name": "Quanta Services"},
        "320193": {"sic": "3571", "sicDescription": "Electronic Computers", "tickers": ["AAPL"], "name": "Apple"},
        "37996": {"sic": "3711", "sicDescription": "Motor Vehicles", "tickers": ["F"], "name": "Ford"},
    }

    def fake_get(url, headers=None, timeout=None):
        if "CY2023Q4I" in url:
            return _Resp(frame_q4_2023)
        if "CY2024Q4I" in url:
            return _Resp(frame_q4_2024)
        for cik10, body in subs.items():
            if f"CIK{cik10}" in url:
                return _Resp(body)
        return _Resp({}, status=404)

    monkeypatch.setattr(rpo.requests, "get", fake_get)
    out = rpo.discover_rpo_filers(["CY2023Q4I", "CY2024Q4I"], use_cache=False)

    assert set(out["ticker"]) == {"PWR", "F"}          # AAPL (SIC 3571) filtered out; PWR + F kept
    assert out.loc[out["ticker"] == "PWR", "cik"].iloc[0] == "0001050915"   # zero-padded to 10
    assert "AAPL" not in set(out["ticker"])
```

- [ ] **Step 3: Run it, verify it fails** — `ModuleNotFoundError: No module named 'backlog_factor'`.

- [ ] **Step 4: Implement `discover_rpo_filers`** in `backlog_factor/data/rpo.py`

```python
from __future__ import annotations

"""SEC XBRL RevenueRemainingPerformanceObligation discovery + per-name fetch.
Mirrors grid_equipment_basket/backlog_data.py (instant concept, filing-dated,
per-CIK parquet cache)."""

import time
import pandas as pd
import requests

from backlog_factor.config import CACHE_DIR, SEC_HEADERS, IN_SCOPE_SIC_PREFIXES

_FRAMES = "https://data.sec.gov/api/xbrl/frames/us-gaap/RevenueRemainingPerformanceObligation/USD/{q}.json"
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"


def _get(url: str) -> requests.Response:
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    return r


def discover_rpo_filers(quarters: list[str], use_cache: bool = True) -> pd.DataFrame:
    cache = CACHE_DIR / "rpo_filers.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    ciks: set[int] = set()
    for q in quarters:
        r = _get(_FRAMES.format(q=q))
        if r.status_code == 404:
            continue
        r.raise_for_status()
        for row in r.json().get("data", []):
            if row.get("cik") is not None:
                ciks.add(int(row["cik"]))
        time.sleep(0.2)

    rows = []
    for cik in sorted(ciks):
        cik10 = f"{cik:010d}"
        r = _get(_SUBMISSIONS.format(cik=cik10))
        if r.status_code != 200:
            continue
        body = r.json()
        sic = str(body.get("sic") or "")
        if not sic.startswith(IN_SCOPE_SIC_PREFIXES):
            continue
        tickers = body.get("tickers") or []
        rows.append({
            "cik": cik10,
            "ticker": tickers[0] if tickers else "",
            "sic": sic,
            "sic_description": body.get("sicDescription", ""),
            "company": body.get("name", ""),
        })
        time.sleep(0.15)

    df = pd.DataFrame(rows, columns=["cik", "ticker", "sic", "sic_description", "company"])
    df = df[df["ticker"] != ""].sort_values("ticker").reset_index(drop=True)
    if use_cache and not df.empty:
        df.to_parquet(cache)
    return df
```

- [ ] **Step 5: Run the test, verify it passes.**

- [ ] **Step 6: Live discovery + curation** (networked — the one manual step of this task)

Run:
```bash
.venv/bin/python -c "
from backlog_factor.data.rpo import discover_rpo_filers
from backlog_factor.config import DISCOVERY_QUARTERS
df = discover_rpo_filers(DISCOVERY_QUARTERS, use_cache=True)
print(len(df)); print(df.to_string())
" | tee /tmp/rpo_filers.txt
```
Then hand-curate: from that list, keep the names that a plain reading says are order-driven manufacturers or contractors (drop consumer-electronics, autos-OEM, pure instrument/measurement plays that don't disclose an order backlog concept meaningfully). Assign each kept name an `industry_group` from the `INDUSTRY_GROUP_ETF` keys. Write the kept list into `config.UNIVERSE` as `{ticker, cik, industry_group, name}` dicts, and record every keep/drop decision with a one-line reason in `backlog_factor/candidate_research.md` (Bash heredoc). Target 60–125 names; if the curated count is < 60, note in `candidate_research.md` that the breadth premise is weak (spec §9).

- [ ] **Step 7: Full suite green + commit**

Run: `.venv/bin/python -m pytest tests/backlog_factor/ -q` → `1 passed`.

```bash
git add backlog_factor/__init__.py backlog_factor/config.py backlog_factor/data/__init__.py \
        backlog_factor/data/rpo.py backlog_factor/candidate_research.md \
        tests/backlog_factor/__init__.py tests/backlog_factor/test_rpo.py
git commit -m "$(printf 'feat(backlog-factor): config + XBRL RPO filer discovery + curated Phase-1 universe\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 2: `data/rpo.py` — per-name RPO fetch + panel loader

**Files:**
- Modify: `backlog_factor/data/rpo.py`
- Modify: `tests/backlog_factor/test_rpo.py`

**Interfaces:**
- Consumes: `config.UNIVERSE`, `config.SEC_HEADERS`, `config.CACHE_DIR`.
- Produces: `rpo.fetch_rpo(ticker, cik, use_cache=True)`, `rpo.load_rpo_panel(universe, use_cache=True)` (signatures above).

- [ ] **Step 1: Write the failing test**

```python
def _facts(rows):   # rows: list of (end, filed, val)
    return {"units": {"USD": [{"end": e, "filed": f, "val": v} for e, f, v in rows]}}


def test_fetch_rpo_parses_and_files_dates(monkeypatch):
    payload = _facts([
        ("2023-03-31", "2023-05-02", 3.0e10),
        ("2023-03-31", "2023-05-09", 3.1e10),   # later-filed dup for the same quarter -> dropped
        ("2023-06-30", "2023-08-01", 3.2e10),
    ])
    def fake_get(url, headers=None, timeout=None):
        if "RevenueRemainingPerformanceObligation.json" in url and "Current" not in url and "Noncurrent" not in url:
            return _Resp(payload)
        return _Resp({"units": {}}, status=404)
    monkeypatch.setattr(rpo.requests, "get", fake_get)

    df = rpo.fetch_rpo("PWR", "0001050915", use_cache=False)
    assert list(df.columns) == ["quarter_end", "availability_date", "metric_value"]
    assert df["quarter_end"].tolist() == [pd.Timestamp("2023-03-31"), pd.Timestamp("2023-06-30")]
    assert df["availability_date"].iloc[0] == pd.Timestamp("2023-05-02")   # earliest filing wins
    assert df["metric_value"].iloc[0] == pytest.approx(3.0e10)


def test_fetch_rpo_merges_current_and_noncurrent_when_base_missing(monkeypatch):
    cur = _facts([("2023-03-31", "2023-05-02", 2.0e10)])
    non = _facts([("2023-03-31", "2023-05-02", 1.0e10)])
    def fake_get(url, headers=None, timeout=None):
        if "ObligationCurrent.json" in url:
            return _Resp(cur)
        if "ObligationNoncurrent.json" in url:
            return _Resp(non)
        return _Resp({"units": {}}, status=404)
    monkeypatch.setattr(rpo.requests, "get", fake_get)
    df = rpo.fetch_rpo("X", "0000000001", use_cache=False)
    assert df["metric_value"].iloc[0] == pytest.approx(3.0e10)
```

- [ ] **Step 2: Run, verify it fails** — `AttributeError: ... has no attribute 'fetch_rpo'`.

- [ ] **Step 3: Implement `_companyconcept`, `fetch_rpo`, `load_rpo_panel`** (append to `rpo.py`)

```python
_CONCEPT = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"


def _companyconcept(cik: str, tag: str) -> pd.DataFrame:
    r = _get(_CONCEPT.format(cik=cik, tag=tag))
    if r.status_code == 404:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "metric_value"])
    r.raise_for_status()
    rows = []
    for _unit, facts in r.json().get("units", {}).items():
        for f in facts:
            end, filed, val = f.get("end"), f.get("filed"), f.get("val")
            if end and filed and val is not None:
                rows.append((pd.Timestamp(end), pd.Timestamp(filed), float(val)))
    if not rows:
        return pd.DataFrame(columns=["quarter_end", "availability_date", "metric_value"])
    df = pd.DataFrame(rows, columns=["quarter_end", "availability_date", "metric_value"])
    return (df.sort_values("availability_date")
              .drop_duplicates("quarter_end", keep="first")
              .sort_values("quarter_end")
              .reset_index(drop=True))


def fetch_rpo(ticker: str, cik: str, use_cache: bool = True) -> pd.DataFrame:
    cache = CACHE_DIR / f"rpo_{ticker}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    df = _companyconcept(cik, "RevenueRemainingPerformanceObligation")
    if df.empty:
        cur = _companyconcept(cik, "RevenueRemainingPerformanceObligationCurrent")
        non = _companyconcept(cik, "RevenueRemainingPerformanceObligationNoncurrent")
        if not cur.empty and not non.empty:
            m = cur.merge(non, on="quarter_end", suffixes=("_c", "_n"))
            m["availability_date"] = m[["availability_date_c", "availability_date_n"]].max(axis=1)
            m["metric_value"] = m["metric_value_c"] + m["metric_value_n"]
            df = m[["quarter_end", "availability_date", "metric_value"]].sort_values("quarter_end")
        time.sleep(0.2)
    if use_cache and not df.empty:
        df.to_parquet(cache)
    return df


def load_rpo_panel(universe: list[dict], use_cache: bool = True) -> pd.DataFrame:
    parts = []
    for row in universe:
        d = fetch_rpo(row["ticker"], row["cik"], use_cache=use_cache)
        if d.empty:
            continue
        d = d.copy()
        d["ticker"] = row["ticker"]
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["ticker", "quarter_end", "availability_date", "metric_value"])
    return (pd.concat(parts, ignore_index=True)
              .sort_values(["ticker", "quarter_end"]).reset_index(drop=True))
```

- [ ] **Step 4: Run the test, verify it passes.**

- [ ] **Step 5: Full suite green + commit** (`3 passed`)

```bash
git add backlog_factor/data/rpo.py tests/backlog_factor/test_rpo.py
git commit -m "$(printf 'feat(backlog-factor): per-name XBRL RPO fetch + universe panel loader\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 3: `data/prices.py`

**Files:**
- Create: `backlog_factor/data/prices.py`
- Test: `tests/backlog_factor/test_prices.py`

**Interfaces:**
- Consumes: `grid_resilience.data.cache_utils` (`find_missing_months_wide`, `month_bounds`, `read_monthly_cache_wide`, `save_monthly_chunks_wide`); `config.CACHE_DIR`.
- Produces: `prices.fetch_prices(tickers, start, end, use_cache=True) -> (date x ticker) DataFrame`.

- [ ] **Step 1: Copy `grid_equipment_basket/data/prices.py` verbatim into `backlog_factor/data/prices.py`**, changing only the import `from grid_equipment_basket.config import CACHE_DIR` → `from backlog_factor.config import CACHE_DIR`. That module already implements the whole-span download + monthly-parquet cache + retry-on-incomplete-columns pattern this task needs; do not reinvent it.

- [ ] **Step 2: Copy `tests/grid_equipment_basket/test_prices.py`** into `tests/backlog_factor/test_prices.py`, changing the import to `from backlog_factor.data import prices as px`. These tests (`monkeypatch` on `_download` / `_download_with_retry`, `tmp_path` cache) are behavior tests, not grid-equipment-specific.

- [ ] **Step 3: Run `tests/backlog_factor/test_prices.py`** — expect all its tests to pass unchanged (the module is a straight copy). If any fail, the copy diverged — fix the copy, not the test.

- [ ] **Step 4: Full suite green + commit**

```bash
git add backlog_factor/data/prices.py tests/backlog_factor/test_prices.py
git commit -m "$(printf 'feat(backlog-factor): adjusted-close price fetcher (mirrors grid_equipment_basket)\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 4: `signal.py` — per-name surprise series

**Files:**
- Create: `backlog_factor/signal.py`
- Create: `tests/backlog_factor/test_signal.py`

**Interfaces:**
- Consumes: `config.HISTORY_MIN_QUARTERS`, `config.SPAN_MIN_DAYS`, `config.SPAN_MAX_DAYS`.
- Produces: `signal.name_surprise_series(rpo_one, *, min_quarters=6, span_min=300, span_max=430)`, `signal.surprise_panel(rpo_panel, ...)` (signatures above).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
import pytest

from backlog_factor import signal as sig


def _rpo_one(mvs, avs=None):
    qe = pd.date_range("2021-03-31", periods=len(mvs), freq="QE")
    av = avs or [q + pd.Timedelta(days=40) for q in qe]
    return pd.DataFrame({"quarter_end": qe, "availability_date": pd.to_datetime(av),
                         "metric_value": [float(x) for x in mvs]})


def test_surprise_is_growth_minus_trailing_mean():
    # 6 quarters. g between consecutive = log(1.10) each for q2..q5, then log(1.25) at q6.
    mvs = [100, 110, 121, 133.1, 146.41, 183.0125]
    out = sig.name_surprise_series(_rpo_one(mvs), min_quarters=6)
    last = out.iloc[-1]
    g6 = np.log(183.0125 / 146.41)
    exp6 = np.log(1.10)                         # trailing 4 g's are all log(1.10)
    assert last["g"] == pytest.approx(g6)
    assert last["expected_g"] == pytest.approx(exp6)
    assert last["surprise"] == pytest.approx(g6 - exp6)


def test_needs_min_quarters():
    out = sig.name_surprise_series(_rpo_one([100, 110, 121, 133.1, 146.41]), min_quarters=6)
    assert out["surprise"].dropna().empty


def test_gappy_span_is_nan():
    qe = pd.to_datetime(["2021-03-31", "2021-06-30", "2022-06-30", "2022-12-31", "2023-06-30", "2023-12-31"])
    df = pd.DataFrame({"quarter_end": qe, "availability_date": qe + pd.Timedelta(days=40),
                       "metric_value": [100.0, 110, 121, 133.1, 146.41, 161.0]})
    out = sig.name_surprise_series(df, min_quarters=6, span_min=300, span_max=430)
    assert out["surprise"].dropna().empty          # iloc[-1]..iloc[-5] spans ~30 months


def test_level_discontinuity_nans_surprise_across_break_and_next_4():
    mvs = [100, 110, 121, 133.1, 146.41, 900.0, 990, 1089, 1197.9, 1317.7, 1449.5]   # 10x jump at index 5
    out = sig.name_surprise_series(_rpo_one(mvs), min_quarters=6)
    # the jump row + next 4 -> NaN surprise; a later row is valid again
    assert out["surprise"].isna().sum() >= 5
    assert out["surprise"].notna().any()
```

- [ ] **Step 2: Run, verify it fails** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `name_surprise_series` + `surprise_panel`**

```python
from __future__ import annotations

"""Backlog/RPO growth-surprise signal (spec 2026-08-31 §4). No fitted model:
expected_g is the trailing 4-quarter mean of g."""

import numpy as np
import pandas as pd

from backlog_factor import config

_BREAK = np.log(3.0)   # |single-quarter log change| beyond this = suspected RPO-definition break


def name_surprise_series(rpo_one: pd.DataFrame, *, min_quarters: int = config.HISTORY_MIN_QUARTERS,
                         span_min: int = config.SPAN_MIN_DAYS,
                         span_max: int = config.SPAN_MAX_DAYS) -> pd.DataFrame:
    g_df = rpo_one.sort_values("quarter_end").reset_index(drop=True)
    g_df["g"] = np.log(g_df["metric_value"] / g_df["metric_value"].shift(1))
    g_df["expected_g"] = g_df["g"].shift(1).rolling(4).mean()
    g_df["surprise"] = g_df["g"] - g_df["expected_g"]

    # history + span guards: need >= min_quarters rows and a clean iloc[i-4]..iloc[i] span
    span_ok = (g_df["quarter_end"] - g_df["quarter_end"].shift(4)).dt.days.between(span_min, span_max)
    g_df.loc[(g_df.index < min_quarters - 1) | ~span_ok, "surprise"] = np.nan

    # RPO-definition-break screen: a |g| spike NaNs this row and the next 4
    brk = g_df.index[g_df["g"].abs() > _BREAK].tolist()
    for i in brk:
        g_df.loc[i: i + 4, "surprise"] = np.nan

    return g_df[["availability_date", "g", "expected_g", "surprise"]]


def surprise_panel(rpo_panel: pd.DataFrame, *, min_quarters: int = config.HISTORY_MIN_QUARTERS,
                   span_min: int = config.SPAN_MIN_DAYS,
                   span_max: int = config.SPAN_MAX_DAYS) -> pd.DataFrame:
    parts = []
    for tkr, g in rpo_panel.groupby("ticker"):
        s = name_surprise_series(g, min_quarters=min_quarters, span_min=span_min, span_max=span_max)
        s = s.dropna(subset=["surprise"]).copy()
        s["ticker"] = tkr
        parts.append(s[["ticker", "availability_date", "surprise"]])
    if not parts:
        return pd.DataFrame(columns=["ticker", "availability_date", "surprise"])
    return pd.concat(parts, ignore_index=True).sort_values(["availability_date", "ticker"]).reset_index(drop=True)
```

- [ ] **Step 4: Run the test, verify it passes.**

- [ ] **Step 5: Full suite green + commit**

```bash
git add backlog_factor/signal.py tests/backlog_factor/test_signal.py
git commit -m "$(printf 'feat(backlog-factor): per-name backlog growth-surprise series + guards\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 5: `signal.py` — cross-sectional transforms

**Files:**
- Modify: `backlog_factor/signal.py`
- Modify: `tests/backlog_factor/test_signal.py`

**Interfaces:**
- Consumes: `grid_resilience.factor.neutralize.cross_section_zscore`; `config.WINSOR_SIGMA`, `config.STALENESS_MAX_DAYS`.
- Produces: `signal.neutralize_cross_section`, `signal.staleness_decay`, `signal.signal_on_date` (signatures above).

- [ ] **Step 1: Write the failing test**

```python
def test_staleness_decay_linear_to_zero():
    assert sig.staleness_decay(0.0, 60) == pytest.approx(1.0)
    assert sig.staleness_decay(30.0, 60) == pytest.approx(0.5)
    assert sig.staleness_decay(60.0, 60) == pytest.approx(0.0)
    assert sig.staleness_decay(90.0, 60) == pytest.approx(0.0)
    s = sig.staleness_decay(pd.Series([0.0, 15.0, 60.0]), 60)
    assert list(s) == pytest.approx([1.0, 0.75, 0.0])


def test_neutralize_cross_section_residual_is_orthogonal_to_group_and_size():
    raw = pd.Series({"A": 2.0, "B": 1.0, "C": -1.0, "D": -2.0, "E": 0.5, "F": -0.5})
    grp = pd.Series({"A": "m", "B": "m", "C": "m", "D": "e", "E": "e", "F": "e"})
    lmc = pd.Series({"A": 9.0, "B": 10.0, "C": 11.0, "D": 9.5, "E": 10.5, "F": 11.5})
    out = sig.neutralize_cross_section(raw, grp, lmc, winsor_sigma=3.0)
    # residual has ~zero correlation with log-mktcap and ~zero mean within each group
    assert abs(float(np.corrcoef(out.values, lmc.reindex(out.index).values)[0, 1])) < 1e-6
    for gname in ["m", "e"]:
        assert out[grp.reindex(out.index) == gname].mean() == pytest.approx(0.0, abs=1e-9)


def test_signal_on_date_uses_latest_visible_surprise_within_staleness():
    sp = pd.DataFrame({
        "ticker": ["A", "A", "B", "C"],
        "availability_date": pd.to_datetime(["2023-02-01", "2023-05-01", "2023-04-20", "2022-01-01"]),
        "surprise": [0.1, 0.4, -0.3, 0.9],
    })
    grp = {"A": "m", "B": "m", "C": "m"}
    lmc = pd.Series({"A": 10.0, "B": 10.0, "C": 10.0})
    out = sig.signal_on_date(sp, pd.Timestamp("2023-06-01"), industry_group_map=grp,
                             log_mktcap=lmc, horizon_days=63, staleness_max_days=200)
    assert "C" not in out.index           # 2022-01 surprise is > 200 days stale
    assert set(out.index) == {"A", "B"}   # A uses its 2023-05-01 row (decayed), B its 2023-04-20 row
```

- [ ] **Step 2: Run, verify it fails.**

- [ ] **Step 3: Implement the three functions** (append to `signal.py`)

```python
from grid_resilience.factor.neutralize import cross_section_zscore


def _winsorize_sigma(s: pd.Series, sigma: float) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return s
    return s.clip(mu - sigma * sd, mu + sigma * sd)


def neutralize_cross_section(raw: pd.Series, industry_group: pd.Series, log_mktcap: pd.Series,
                             winsor_sigma: float = config.WINSOR_SIGMA) -> pd.Series:
    names = [t for t in raw.index if t in industry_group.index and t in log_mktcap.index]
    z = cross_section_zscore(_winsorize_sigma(raw.reindex(names), winsor_sigma))
    grp = industry_group.reindex(names).astype("category")
    X = pd.get_dummies(grp, drop_first=False).astype(float)
    X["_lmc"] = log_mktcap.reindex(names).astype(float).values
    X["_const"] = 1.0
    Xv = X.values
    beta, *_ = np.linalg.lstsq(Xv, z.values, rcond=None)
    resid = z.values - Xv @ beta
    return pd.Series(resid, index=names)


def staleness_decay(days_since, horizon_days: int):
    frac = 1.0 - (np.asarray(days_since, dtype=float) / float(horizon_days))
    frac = np.clip(frac, 0.0, 1.0)
    if isinstance(days_since, pd.Series):
        return pd.Series(frac, index=days_since.index)
    return float(frac) if np.ndim(frac) == 0 else frac


def signal_on_date(sp: pd.DataFrame, asof: pd.Timestamp, *, industry_group_map: dict,
                   log_mktcap: pd.Series, horizon_days: int,
                   staleness_max_days: int = config.STALENESS_MAX_DAYS,
                   winsor_sigma: float = config.WINSOR_SIGMA) -> pd.Series:
    asof = pd.Timestamp(asof)
    vis = sp[sp["availability_date"] <= asof]
    latest = (vis.sort_values("availability_date").groupby("ticker").tail(1).set_index("ticker"))
    age = (asof - latest["availability_date"]).dt.days
    keep = latest[age <= staleness_max_days].copy()
    keep["age"] = (asof - keep["availability_date"]).dt.days
    decayed = keep["surprise"] * staleness_decay(keep["age"], horizon_days)
    grp = pd.Series({t: industry_group_map.get(t) for t in decayed.index})
    return neutralize_cross_section(decayed, grp, log_mktcap, winsor_sigma).dropna()
```

- [ ] **Step 4: Run the test, verify it passes.**

- [ ] **Step 5: Full suite green + commit**

```bash
git add backlog_factor/signal.py tests/backlog_factor/test_signal.py
git commit -m "$(printf 'feat(backlog-factor): cross-sectional winsorize / z-score / industry+size neutralization / staleness decay\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 6: `event_study.py`

**Files:**
- Create: `backlog_factor/event_study.py`
- Create: `tests/backlog_factor/test_event_study.py`

**Interfaces:**
- Consumes: `signal.surprise_panel` output; `data/prices` frames; `config.CAR_WINDOWS`.
- Produces: `event_pairs`, `quintile_car_table`, `clustered_tstat`, `monotonic`, `subperiod_split`, `by_industry_signs`, `evaluate_gate` (signatures above).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
import pytest

from backlog_factor import event_study as es


def _prices(tickers, start="2023-01-02", n=200, drift=0.0):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(0)
    return pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))) for t in tickers}, index=idx)


def test_event_pairs_day0_is_first_trading_day_after_filing():
    px = pd.DataFrame({"A": [10, 11, 12, 13, 14], "ETF": [10, 10, 10, 10, 10]},
                      index=pd.to_datetime(["2023-05-10", "2023-05-11", "2023-05-12", "2023-05-15", "2023-05-16"]))
    sp = pd.DataFrame({"ticker": ["A"], "availability_date": [pd.Timestamp("2023-05-11")], "surprise": [1.0]})
    pairs = es.event_pairs(sp, px[["A"]], px[["ETF"]], {"A": "m"}, {"m": "ETF"}, car_windows=[2])
    # day 0 = 2023-05-12 (first bday strictly after 05-11); 2-day abnormal CAR = (12->13 ret) + (13->14 ret) - 0
    assert pairs["car_2"].iloc[0] == pytest.approx((13/12 - 1) + (14/13 - 1))


def test_clustered_tstat_differs_from_naive_and_gate_pass_on_monotone_panel():
    # constructed: surprise quintile strongly predicts a positive abnormal drift, monotone, both halves
    rng = np.random.default_rng(1)
    rows = []
    for m in pd.period_range("2020-01", "2023-12", freq="M"):
        for k in range(50):
            s = rng.normal()
            car = 0.02 * s + rng.normal(0, 0.01)          # monotone in s, low noise
            rows.append({"ticker": f"T{k}", "availability_date": m.to_timestamp(),
                         "month": m, "industry_group": ["machinery", "electrical_equipment",
                         "aerospace_defense", "engineering_construction", "building_products"][k % 5],
                         "surprise": s, "car_21": car, "car_63": car * 1.1})
    pairs = pd.DataFrame(rows)
    g = es.evaluate_gate(pairs, car_windows=[21, 63])
    assert g["passed"] is True
    assert g["monotone"] is True
    assert g["tstat"] > 2
    assert g["n_groups_right_sign"] >= 3


def test_gate_fails_on_noise_panel():
    rng = np.random.default_rng(2)
    rows = [{"ticker": f"T{k}", "availability_date": m.to_timestamp(), "month": m,
             "industry_group": "machinery", "surprise": rng.normal(),
             "car_21": rng.normal(0, 0.05), "car_63": rng.normal(0, 0.05)}
            for m in pd.period_range("2020-01", "2023-12", freq="M") for k in range(30)]
    g = es.evaluate_gate(pd.DataFrame(rows), car_windows=[21, 63])
    assert g["passed"] is False
```

- [ ] **Step 2: Run, verify it fails.**

- [ ] **Step 3: Implement `event_study.py`**

```python
from __future__ import annotations

"""Backlog-surprise event study (spec 2026-08-31 §5). Abnormal CAR = name simple
return minus its industry-group ETF simple return, cumulative-summed over a window
starting the first trading day strictly after the SEC filing date."""

import numpy as np
import pandas as pd

from backlog_factor import config

_MAJOR_GROUPS = ["machinery", "electrical_equipment", "aerospace_defense",
                 "engineering_construction", "building_products"]


def _window_abnormal_car(px: pd.Series, etf: pd.Series, filing: pd.Timestamp, w: int) -> float:
    r = px.pct_change()
    b = etf.pct_change()
    start = px.index.searchsorted(pd.Timestamp(filing), side="right")
    seg = (r - b.reindex(r.index)).iloc[start:start + w]
    if len(seg) < max(2, w // 2):
        return np.nan
    return float(seg.sum())


def event_pairs(sp, prices, group_etf_prices, industry_group_map, group_etf_map, car_windows):
    out = []
    for _, row in sp.iterrows():
        t = row["ticker"]
        grp = industry_group_map.get(t)
        etf_t = group_etf_map.get(grp)
        if t not in prices.columns or etf_t not in group_etf_prices.columns:
            continue
        rec = {"ticker": t, "availability_date": pd.Timestamp(row["availability_date"]),
               "month": pd.Timestamp(row["availability_date"]).to_period("M"),
               "industry_group": grp, "surprise": float(row["surprise"])}
        for w in car_windows:
            rec[f"car_{w}"] = _window_abnormal_car(prices[t].dropna(),
                                                  group_etf_prices[etf_t].dropna(),
                                                  row["availability_date"], w)
        out.append(rec)
    return pd.DataFrame(out)


def _quintile(s: pd.Series) -> pd.Series:
    try:
        return pd.qcut(s, 5, labels=[1, 2, 3, 4, 5]).astype("Int64")
    except ValueError:
        return pd.Series(pd.NA, index=s.index, dtype="Int64")


def quintile_car_table(pairs: pd.DataFrame, car_windows):
    p = pairs.copy()
    p["q"] = p.groupby("month")["surprise"].transform(_quintile)
    p = p.dropna(subset=["q"])
    rows = {}
    for q in [1, 2, 3, 4, 5]:
        rows[q] = {f"car_{w}_mean": p.loc[p["q"] == q, f"car_{w}"].mean() for w in car_windows}
    tbl = pd.DataFrame(rows).T
    tbl.loc["Q5-Q1"] = tbl.loc[5] - tbl.loc[1]
    return tbl


def _monthly_spread(pairs: pd.DataFrame, value_col: str) -> pd.Series:
    p = pairs.copy()
    p["q"] = p.groupby("month")["surprise"].transform(_quintile)
    p = p.dropna(subset=["q"])
    hi = p[p["q"] == 5].groupby("month")[value_col].mean()
    lo = p[p["q"] == 1].groupby("month")[value_col].mean()
    return (hi - lo).dropna()


def clustered_tstat(pairs: pd.DataFrame, value_col: str, cluster_col: str = "month") -> float:
    m = _monthly_spread(pairs, value_col)
    if len(m) < 3 or m.std() == 0:
        return np.nan
    return float(m.mean() / (m.std() / np.sqrt(len(m))))


def monotonic(quintile_means) -> bool:
    v = list(quintile_means)
    return all(v[i] <= v[i + 1] for i in range(len(v) - 1)) or all(v[i] >= v[i + 1] for i in range(len(v) - 1))


def subperiod_split(pairs: pd.DataFrame):
    med = pairs["month"].sort_values().iloc[len(pairs) // 2]
    return pairs[pairs["month"] <= med], pairs[pairs["month"] > med]


def by_industry_signs(pairs: pd.DataFrame, value_col: str) -> pd.Series:
    out = {}
    for g, gp in pairs.groupby("industry_group"):
        m = _monthly_spread(gp, value_col)
        out[g] = np.sign(m.mean()) if len(m) else np.nan
    return pd.Series(out)


def evaluate_gate(pairs: pd.DataFrame, car_windows=None) -> dict:
    car_windows = car_windows or config.CAR_WINDOWS
    tbl = quintile_car_table(pairs, car_windows)
    # peak window = the one with the largest |Q5-Q1|
    peak_w = max(car_windows, key=lambda w: abs(tbl.loc["Q5-Q1", f"car_{w}_mean"]))
    col = f"car_{peak_w}"
    q_means = [tbl.loc[q, f"{col}_mean"] for q in [1, 2, 3, 4, 5]]
    q5q1 = tbl.loc["Q5-Q1", f"{col}_mean"]
    t = clustered_tstat(pairs, col)
    s1, s2 = subperiod_split(pairs)
    m1, m2 = _monthly_spread(s1, col), _monthly_spread(s2, col)
    signs = by_industry_signs(pairs, col)
    want = np.sign(q5q1)
    n_right = int((signs.reindex(_MAJOR_GROUPS).dropna() == want).sum())
    passed = bool(
        q5q1 > 0 and monotonic(q_means) and (t is not None and t > 2)
        and np.sign(m1.mean()) == want and (len(m1) >= 3 and abs(m1.mean() / (m1.std() / np.sqrt(len(m1)))) > 1)
        and np.sign(m2.mean()) == want and (len(m2) >= 3 and abs(m2.mean() / (m2.std() / np.sqrt(len(m2)))) > 1)
        and n_right >= 3
    )
    return {"peak_window": peak_w, "q5_q1_car": float(q5q1), "monotone": monotonic(q_means),
            "tstat": float(t) if t == t else np.nan,
            "sub1_sign": float(np.sign(m1.mean())), "sub1_tstat": float(m1.mean() / (m1.std() / np.sqrt(len(m1)))) if len(m1) >= 3 else np.nan,
            "sub2_sign": float(np.sign(m2.mean())), "sub2_tstat": float(m2.mean() / (m2.std() / np.sqrt(len(m2)))) if len(m2) >= 3 else np.nan,
            "industry_signs": {k: (float(v) if v == v else None) for k, v in signs.items()},
            "n_groups_right_sign": n_right, "passed": passed,
            "drift_horizon_days": int(peak_w)}
```

- [ ] **Step 4: Run the test, verify it passes.**

- [ ] **Step 5: Full suite green + commit**

```bash
git add backlog_factor/event_study.py tests/backlog_factor/test_event_study.py
git commit -m "$(printf 'feat(backlog-factor): event study - quintile CARs, month-clustered t-stat, pre-registered gate\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Task 7: `__main__.py` + live run + results doc + README

**Files:**
- Create: `backlog_factor/__main__.py`
- Create: `docs/backlog-surprise-factor-results.md`
- Create: `backlog_factor/README.md`
- Modify: `README.md` (repo root)
- Test: add `tests/backlog_factor/test_cli.py` (dispatch smoke only)

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: `python -m backlog_factor --event-study`.

- [ ] **Step 1: Write `__main__.py`**

```python
from __future__ import annotations

"""CLI: python -m backlog_factor --event-study [--start YYYY-MM-DD] [--end YYYY-MM-DD]"""

import argparse
import json
from pathlib import Path

import pandas as pd

from backlog_factor import config, event_study, signal
from backlog_factor.data import prices as px
from backlog_factor.data import rpo as rpo_mod


def _run_event_study(start: str, end: str, out: Path) -> dict:
    uni = config.UNIVERSE
    if not uni:
        raise SystemExit("config.UNIVERSE is empty - run Task 1's curation step first.")
    group_map = {r["ticker"]: r["industry_group"] for r in uni}
    tickers = [r["ticker"] for r in uni]
    etfs = sorted(set(config.INDUSTRY_GROUP_ETF.values()))

    rpo_panel = rpo_mod.load_rpo_panel(uni, use_cache=True)
    sp = signal.surprise_panel(rpo_panel)
    prices = px.fetch_prices(tickers, start, end, use_cache=True)
    etf_prices = px.fetch_prices(etfs, start, end, use_cache=True)

    pairs = event_study.event_pairs(sp, prices, etf_prices, group_map,
                                    config.INDUSTRY_GROUP_ETF, config.CAR_WINDOWS)
    tbl = event_study.quintile_car_table(pairs, config.CAR_WINDOWS)
    gate = event_study.evaluate_gate(pairs, config.CAR_WINDOWS)

    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "event_pairs.csv", index=False)
    tbl.to_csv(out / "quintile_car_table.csv")
    (out / "gate.json").write_text(json.dumps(gate, indent=2, default=str))
    print(tbl.to_string())
    print()
    print(f"peak window {gate['peak_window']}d  |  Q5-Q1 abnormal CAR {gate['q5_q1_car']*100:.2f}%  "
          f"|  clustered t {gate['tstat']:.2f}  |  monotone {gate['monotone']}  "
          f"|  groups right sign {gate['n_groups_right_sign']}/5")
    print(f"GATE: {'PASS' if gate['passed'] else 'FAIL'}")
    return gate


def main() -> None:
    ap = argparse.ArgumentParser(prog="backlog_factor")
    ap.add_argument("--event-study", action="store_true")
    ap.add_argument("--start", default=config.STRUCTURED_START)
    ap.add_argument("--end", default=(pd.Timestamp.today().normalize().replace(day=1)
                                      - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    ap.add_argument("--output", default="./output_backlog_factor")
    args = ap.parse_args()
    if not args.event_study:
        raise SystemExit("nothing to do - pass --event-study")
    _run_event_study(args.start, args.end, Path(args.output))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: CLI dispatch test** (`tests/backlog_factor/test_cli.py`)

```python
import sys
import pandas as pd
from backlog_factor import __main__ as cli


def test_event_study_flag_dispatches(monkeypatch, tmp_path, capsys):
    called = {}
    monkeypatch.setattr(cli, "_run_event_study",
                        lambda s, e, out: called.setdefault("args", (s, e, str(out))) or {"passed": False})
    monkeypatch.setattr(sys, "argv", ["prog", "--event-study", "--start", "2019-01-01",
                                      "--end", "2024-01-01", "--output", str(tmp_path)])
    cli.main()
    assert called["args"][0] == "2019-01-01"


def test_no_flag_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    import pytest
    with pytest.raises(SystemExit):
        cli.main()
```

- [ ] **Step 3: Run tests, verify pass. Full suite green.**

- [ ] **Step 4: LIVE RUN (networked)**

```bash
.venv/bin/python -m backlog_factor --event-study --start 2019-01-01 --output ./output_backlog_factor
```
Capture the quintile CAR table and the gate verdict. If the curated universe from Task 1 has < ~40 names with a usable surprise history, note that the breadth premise did not hold and the event study is underpowered — that is itself the finding.

- [ ] **Step 5: Write `docs/backlog-surprise-factor-results.md`** (Bash heredoc). Fill every bracket with a real number:

```
# Backlog Growth Surprise — Event Study Results

Spec: docs/superpowers/specs/2026-08-31-backlog-surprise-factor-design.md
Plan: docs/superpowers/plans/2026-08-31-backlog-surprise-factor-event-study.md
Run date: <today>. RPO: SEC XBRL companyconcept. Prices: yfinance adjusted close.

## Phase-1 universe
<N names curated from the XBRL-RPO filer list; count by industry group; the drop list is in
backlog_factor/candidate_research.md>. <M> of them cleared the >=6-quarter + clean-span guards and
contribute at least one event.

## Method
Abnormal CAR = daily simple return minus the name's industry-group ETF, cumulative-summed over
[0,+5/+21/+42/+63] trading days from the first trading day after the SEC filing date. Surprise sorted into
cross-sectional quintiles within each calendar month, then pooled. Q5-Q1 significance = month-clustered t.

## Quintile abnormal CAR (%)
<paste the quintile_car_table: rows Q1..Q5 + Q5-Q1, columns for each window>

## Pre-registered gate (spec §5)
| Check | Value | Pass? |
|---|---|---|
| Q5-Q1 abnormal CAR at peak window (<peak>d) > 0 and monotone | <..>% , monotone <T/F> | <..> |
| month-clustered t > 2 | <..> | <..> |
| both sub-period halves same sign, clustered t > 1 | h1 <sign>/<t> , h2 <sign>/<t> | <..> |
| right sign in >= 3 of 5 industry groups | <n>/5  (<per-group signs>) | <..> |

**GATE: PASS / FAIL.**

<one paragraph: what the drift looks like - symmetric or downside-skewed (report positive vs negative
surprise CAR separately), which industry groups carry it, whether it is concentrated in one regime>

## Caveats (spec §8)
- ~<Y>-year structured span, one macro cycle. Breadth is cross-sectional (<pairs> events), not time-series.
- RPO definition drift: <count> name-quarters NaN'd by the |g|>log(3) break screen.
- Survivorship: universe assembled 2026 from current filers; delisted/acquired names absent.
- Look-ahead: expected_g is a fixed trailing mean; the peak window / drift horizon is chosen here and is
  reserved-holdout-tested only in the follow-on factor plan.

## Decision
- PASS -> DRIFT_HORIZON_DAYS := <peak>. Proceed to the follow-on factor plan (factor.py + backtest.py +
  the §7 walk-forward / holdout / additivity validation).
- FAIL -> stop. The negative result plus this doc and backlog_factor/candidate_research.md are the
  deliverable. No Phase 2 hand-collection.
```

- [ ] **Step 6: Write `backlog_factor/README.md`** (Bash heredoc): universe summary, the signal definition (surprise = g − trailing-4Q mean), the event-study result + gate verdict, a pointer to the results doc, and the spec §8 caveats. If the gate PASSED, note `DRIFT_HORIZON_DAYS` and that the follow-on factor plan is next; if FAILED, state the negative result plainly.

- [ ] **Step 7: Add a one-line pointer** to the repo-root `README.md` for `backlog_factor/` (Bash).

- [ ] **Step 8: Confirm `output_backlog_factor/` is gitignored** (repo ignores `output_*`); `git status` shows no output dir staged. Full suite green.

- [ ] **Step 9: Commit**

```bash
git add backlog_factor/__main__.py tests/backlog_factor/test_cli.py \
        docs/backlog-surprise-factor-results.md backlog_factor/README.md README.md
git commit -m "$(printf 'docs(backlog-factor): event-study CLI + live-run results + pre-registered gate verdict\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>')"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §1 thesis (surprise not raw growth) | Task 4 (`surprise = g - expected_g`); Task 7 writeup framing |
| §2 universe, Phase-1 XBRL-RPO, business-description inclusion, point-in-time membership | Task 1 (discovery + curation + `candidate_research.md`) |
| §3 benchmarks / neutralization targets (industry-group ETF, SPY, controls) | Task 1 `INDUSTRY_GROUP_ETF`; Task 6 abnormal CAR vs group ETF. (Analyst-revision / momentum / quality controls are §7 factor-plan scope, explicitly deferred — noted below.) |
| §4 signal: g, expected_g (trailing-4Q mean), surprise; winsorize; z-score; industry+size neutralize; staleness decay; guards | Tasks 4 (series + guards) and 5 (cross-sectional transforms) |
| §5 event study: CAR windows, quintile CARs, month-clustered t, monotonicity, pos/neg split, sub-period halves, by-industry; the pre-registered gate | Task 6 (`event_pairs`, `quintile_car_table`, `clustered_tstat`, `monotonic`, `subperiod_split`, `by_industry_signs`, `evaluate_gate`); Task 7 runs it + reports pos/neg split in the writeup |
| §6 factor construction | **Out of scope for this plan** — follow-on plan (see Goal + Task 7 Decision) |
| §7 factor validation (walk-forward, holdout, additivity, raw-growth compare) | **Out of scope for this plan** — follow-on plan |
| §8 caveats | Task 7 results doc + `backlog_factor/README.md` |
| §9 phased/gated data assembly | Task 1 (Phase 1 only); Phase 2 gated behind the follow-on factor gate |
| §10 module layout | Tasks 1–7 file-by-file (minus `factor.py` / `backtest.py` / `data/estimates.py` — factor-plan scope) |
| §11 testing | test files in Tasks 1–7 |
| §12 deliverables (Phase 1, event-study portion) | Task 7 |

**Deliberate scope cut:** this plan implements the spec through the §5 event-study gate only. `factor.py`, `backtest.py`, `data/estimates.py`, and the §6–§7 factor construction + validation are a separate plan, written once the gate outcome is known — mirroring how `grid_equipment_basket` split Step 1 (basket + gate) from Step 2 (tilt). The Goal statement and Task 7's Decision section both say this explicitly. No silent gap.

**2. Placeholder scan**

- `config.UNIVERSE = []` is populated by Task 1 Step 6's live curation — not a placeholder, it's a task output, and every downstream task that needs it is sequenced after Task 1. `__main__._run_event_study` raises a clear `SystemExit` if it's still empty.
- `DRIFT_HORIZON_DAYS = 63` is provisional-by-design (§4/§5 set it from the event-study peak window); Task 7 Step 5's Decision records the reset value.
- The results-doc template (Task 7 Step 5) has `<brackets>` — these are live-run outputs that cannot exist until Task 7 runs; each names exactly which artifact fills it. This is the one legitimate deferral (same pattern as the value-chain-reframe plan's Task 12).
- No "TODO", no "similar to Task N", no undefined symbols.

**3. Type consistency**

- `rpo.fetch_rpo` / `load_rpo_panel` emit `[ticker,] quarter_end, availability_date, metric_value`; `signal.name_surprise_series` consumes exactly `quarter_end, availability_date, metric_value` and emits `availability_date, g, expected_g, surprise`; `signal.surprise_panel` emits `ticker, availability_date, surprise`; `event_study.event_pairs` consumes `ticker, availability_date, surprise`. Chain matches.
- `industry_group_map` is `{ticker: group}` everywhere (`signal_on_date`, `event_pairs`); `group_etf_map` / `config.INDUSTRY_GROUP_ETF` is `{group: etf}` everywhere. `event_pairs` takes both, in that order.
- `config.CAR_WINDOWS` (`list[int]`) flows to `event_pairs`, `quintile_car_table`, `evaluate_gate` unchanged; `evaluate_gate` returns `drift_horizon_days` (int) = the peak window.
- `staleness_decay` accepts `pd.Series | float` and returns the same kind — `signal_on_date` passes a Series, `test_signal` checks both.
- `cross_section_zscore` imported from `grid_resilience.factor.neutralize` (confirmed present, public, already cross-imported by `grid_equipment_basket`). `winsorize` there is quantile-based; this plan uses its own `_winsorize_sigma` (sigma-based, per spec §4) — intentional, not a name clash.
- `evaluate_gate(...)["passed"]` is wrapped in `bool(...)`; sub-period t-stats guarded for `len < 3`.

Fixed inline: none needed.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-08-31-backlog-surprise-factor-event-study.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session with checkpoints.

**Which approach?**
