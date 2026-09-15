# Screened Electrification SOW Execution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the Addenda Capital engagement's Phase 1–3 SOW work against the Screened
Electrification Equipment strategy (`electrification_strategy/`): close the short-interest and
client-risk-data gaps, run a genuine pre-registered re-test of "stress beta" on the supplier
universe, formalize the backtest/report, and produce the handoff documentation + updated SOW.

**Architecture:** Extends the existing `electrification_strategy/` package with two new
diagnostic modules (`short_interest.py`, `risk_exposures.py`); reuses `grid_resilience/`'s
already-built and tested GSI/conditional-beta machinery unmodified via a new small package
(`electrification_stress_beta/`) rather than rebuilding it; branches Phase 2 construction on
that gate's real outcome; formalizes existing backtest/report code into named SOW deliverables.

**Tech Stack:** Python 3.11, pandas, numpy, statsmodels, yfinance, gridstatus, pytest, poetry.

**Spec:** `docs/superpowers/specs/2026-09-15-screened-electrification-sow-plan-design.md`

## Global Constraints

- Simple-return convention throughout `electrification_strategy/` (never log returns) — match
  existing code.
- All caching goes through `config.CACHE_DIR` / `config.DATA_DIR` (parquet), following the
  patterns already in `fred.py` and `grid_resilience/data/cache_utils.py`.
- Every new markdown file is created with `Bash` (`cat > file.md << 'EOF' ... EOF`), never the
  `Write` tool, per this repo's `CLAUDE.md`. Edits to an already-existing markdown file use the
  `Edit` tool.
- After any code change, check whether README.md / RESEARCH-LOG.md needs updating and do it in
  the same session (`CLAUDE.md`).
- The stress-beta gate test (Task 5) runs exactly once against real data with the pre-registered
  methodology from the spec — no iterating on GSI weights, universe cuts, or lookback windows to
  chase significance (spec §3.4 time-box).
- Gate 1's primary pass bar is the **2021-01-01-onward** window, \|t\| ≥ 2 (spec §3.4 Step B,
  the regime-based split, not the AI-boom-era `config.SUBWINDOWS`).
- Any new power/energy-market data source ships only if it demonstrably improves a gated result
  — never for narrative flavor alone (spec's data-inclusion discipline).
- `git tag` / release actions require explicit user confirmation at execution time (Task 14) —
  do not run them unattended.

---

## Task 1: Short-interest diagnostic module

**Files:**
- Create: `electrification_strategy/short_interest.py`
- Test: `tests/electrification_strategy/test_short_interest.py`
- Modify: `electrification_strategy/config.py` (add `SHORT_INTEREST_CACHE_FILE` constant)

**Interfaces:**
- Consumes: `electrification_strategy.config.CACHE_DIR`, `config.HEDGE_SHORT_TICKERS`.
- Produces: `fetch_short_interest(tickers: list[str], use_cache: bool = True) -> pd.DataFrame`
  (indexed by `ticker`, columns `shares_short, short_ratio, short_pct_float,
  date_short_interest, fetched_at`); `crowding_diagnostic(history: pd.DataFrame,
  tickers: tuple[str, ...] = config.HEDGE_SHORT_TICKERS) -> pd.DataFrame` (latest snapshot per
  ticker, sorted by `short_pct_float` descending) — both consumed by Task 7.

This closes proposal v5's own flagged gap ("Trading costs at scale are still to be estimated")
for Option B's short leg, as a cost/liquidity diagnostic — not a new alpha signal (spec §3.1).
Source: yfinance's per-ticker `Ticker(t).info` short-interest fields (already a project
dependency via `grid_equipment_basket.data.prices`; no new external endpoint to integrate). This
is a live-only snapshot (yfinance has no historical short-interest endpoint), so history
accumulates from the first pipeline run forward — document that limitation in Task 9's report,
not hidden.

- [ ] **Step 1: Add the cache-file config constant**

In `electrification_strategy/config.py`, after the existing `HEDGE_*` block, add:

```python
# -- short interest diagnostic (spec 3.1) --------------------------------
SHORT_INTEREST_CACHE_FILE: str = "short_interest_history.parquet"
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/electrification_strategy/test_short_interest.py
import pandas as pd
import pytest

from electrification_strategy import config, short_interest as si


def _fake_info(ticker):
    table = {
        "DLR": {"sharesShort": 12_000_000, "shortRatio": 2.1, "shortPercentOfFloat": 0.031,
                "dateShortInterest": 1735689600},  # 2025-01-01 UTC epoch seconds
        "EQIX": {"sharesShort": 900_000, "shortRatio": 1.4, "shortPercentOfFloat": 0.012,
                 "dateShortInterest": 1735689600},
    }
    return table[ticker]


def test_fetch_short_interest_shapes_and_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "_fetch_one", _fake_info)
    df = si.fetch_short_interest(["DLR", "EQIX"])
    assert list(df.index) == ["DLR", "EQIX"]
    assert set(["shares_short", "short_ratio", "short_pct_float",
                "date_short_interest", "fetched_at"]).issubset(df.columns)
    assert df.loc["DLR", "short_pct_float"] == 0.031
    assert (tmp_path / config.SHORT_INTEREST_CACHE_FILE).exists()


def test_cache_roundtrip_appends_without_duplicating(monkeypatch, tmp_path):
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def counting_fetch(ticker):
        calls["n"] += 1
        return _fake_info(ticker)

    monkeypatch.setattr(si, "_fetch_one", counting_fetch)
    si.fetch_short_interest(["DLR", "EQIX"])
    si.fetch_short_interest(["DLR", "EQIX"])
    history = pd.read_parquet(tmp_path / config.SHORT_INTEREST_CACHE_FILE)
    assert calls["n"] == 4  # both calls hit the network; snapshot re-fetches by design
    # but the second snapshot on the same date_short_interest doesn't duplicate rows
    assert len(history[history["ticker"] == "DLR"]) == 1


def test_crowding_diagnostic_sorts_by_short_pct_float(monkeypatch, tmp_path):
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "_fetch_one", _fake_info)
    si.fetch_short_interest(["DLR", "EQIX"])
    history = pd.read_parquet(tmp_path / config.SHORT_INTEREST_CACHE_FILE)
    diag = si.crowding_diagnostic(history, tickers=("DLR", "EQIX"))
    assert list(diag.index) == ["DLR", "EQIX"]  # DLR has the higher short_pct_float
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `poetry run pytest tests/electrification_strategy/test_short_interest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'electrification_strategy.short_interest'`

- [ ] **Step 4: Implement the module**

```python
# electrification_strategy/short_interest.py
"""Short-interest snapshot cache -- a cost/liquidity diagnostic for the hedge
short leg (config.HEDGE_SHORT_TICKERS), not a signal input.

yfinance exposes only a live snapshot (no historical short-interest endpoint),
so this accumulates history from the first pipeline run forward. See
docs/technical-reference.md and the known-limitations section of the Phase 2
findings report.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from electrification_strategy import config


def _fetch_one(ticker: str) -> dict:
    return yf.Ticker(ticker).info


def fetch_short_interest(tickers: list[str], use_cache: bool = True) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    rows = []
    for t in tickers:
        info = _fetch_one(t)
        rows.append({
            "ticker": t,
            "shares_short": info.get("sharesShort"),
            "short_ratio": info.get("shortRatio"),
            "short_pct_float": info.get("shortPercentOfFloat"),
            "date_short_interest": pd.to_datetime(info.get("dateShortInterest"), unit="s", utc=True)
                if info.get("dateShortInterest") else pd.NaT,
            "fetched_at": now,
        })
    snapshot = pd.DataFrame(rows)

    if use_cache:
        cache_path = config.CACHE_DIR / config.SHORT_INTEREST_CACHE_FILE
        if cache_path.exists():
            existing = pd.read_parquet(cache_path)
            combined = pd.concat([existing, snapshot], ignore_index=True)
            combined = combined.drop_duplicates(subset=["ticker", "date_short_interest"], keep="last")
        else:
            combined = snapshot
        combined.to_parquet(cache_path, index=False)

    return snapshot.set_index("ticker")


def crowding_diagnostic(history: pd.DataFrame, tickers: tuple[str, ...] = config.HEDGE_SHORT_TICKERS) -> pd.DataFrame:
    latest = (history[history["ticker"].isin(tickers)]
              .sort_values("fetched_at")
              .groupby("ticker").last())
    return latest.sort_values("short_pct_float", ascending=False)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/electrification_strategy/test_short_interest.py -v`
Expected: PASS (all three tests)

- [ ] **Step 6: Commit**

```bash
git add electrification_strategy/short_interest.py electrification_strategy/config.py \
        tests/electrification_strategy/test_short_interest.py
git commit -m "feat(electrification): add short-interest crowding/cost diagnostic

Closes proposal v5's flagged gap (trading costs at scale not estimated)
for the hedge short leg. Snapshot-only via yfinance; no historical
short-interest endpoint is free, so history accumulates going forward."
```

---

## Task 2: Client risk-exposures adapter

**Files:**
- Create: `electrification_strategy/risk_exposures.py`
- Create: `electrification_strategy/data/risk_exposures_sample.csv` (synthetic fixture, clearly
  marked as placeholder data)
- Test: `tests/electrification_strategy/test_risk_exposures.py`

**Interfaces:**
- Consumes: `config.DATA_DIR`, `config.REBALANCE_LAG_DAYS`.
- Produces: `load_risk_exposures(path: Path | None = None) -> pd.DataFrame` (long format,
  columns `date, ticker, factor, exposure`, empty DataFrame if the file doesn't exist);
  `align_to_universe(exposures, tickers, asof, lag_days=config.REBALANCE_LAG_DAYS) -> pd.DataFrame`
  (wide, index=ticker, columns=factor, point-in-time as of `asof - lag_days`);
  `portfolio_exposure_report(exposures, weights, asof, lag_days=config.REBALANCE_LAG_DAYS) -> pd.Series`
  (index=factor) — consumed by Task 7's diagnostic wiring.

No Addenda risk-model file exists yet (spec §3.3) — this ships against a placeholder schema with
a documented swap-in point, not blocked on the client.

- [ ] **Step 1: Write the sample fixture CSV**

```bash
mkdir -p electrification_strategy/data
cat > electrification_strategy/data/risk_exposures_sample.csv << 'CSV_EOF'
# SYNTHETIC PLACEHOLDER -- not Addenda data. Schema contract: date,ticker,factor,exposure
# (long format, one row per date x ticker x factor). Swap this file for the real
# Addenda risk-model export at electrification_strategy/data/risk_exposures.csv once
# received -- see docs/technical-reference.md for the schema contract.
date,ticker,factor,exposure
2024-12-31,ETN,value,0.42
2024-12-31,ETN,momentum,-0.15
2024-12-31,HUBB,value,0.10
2024-12-31,HUBB,momentum,0.55
2025-03-31,ETN,value,0.38
2025-03-31,ETN,momentum,0.05
2025-03-31,HUBB,value,0.08
2025-03-31,HUBB,momentum,0.61
CSV_EOF
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/electrification_strategy/test_risk_exposures.py
import pandas as pd
import pytest

from electrification_strategy import config, risk_exposures as rx


def test_load_missing_file_returns_empty(tmp_path):
    df = rx.load_risk_exposures(tmp_path / "does_not_exist.csv")
    assert df.empty
    assert list(df.columns) == ["date", "ticker", "factor", "exposure"]


def test_load_sample_fixture():
    df = rx.load_risk_exposures(config.DATA_DIR / "risk_exposures_sample.csv")
    assert len(df) == 8
    assert df["date"].dtype == "datetime64[ns]"
    assert set(df["factor"]) == {"value", "momentum"}


def test_align_to_universe_picks_most_recent_at_or_before_lagged_asof():
    df = rx.load_risk_exposures(config.DATA_DIR / "risk_exposures_sample.csv")
    # asof 2025-05-01, lag 42 days -> effective date 2025-03-20 -> most recent
    # available row at/before that is 2024-12-31 (2025-03-31 is still in the future
    # relative to the lagged point-in-time date)
    aligned = rx.align_to_universe(df, ["ETN", "HUBB"], pd.Timestamp("2025-05-01"), lag_days=42)
    assert aligned.loc["ETN", "value"] == 0.42
    assert aligned.loc["HUBB", "momentum"] == 0.55


def test_align_to_universe_missing_ticker_is_nan():
    df = rx.load_risk_exposures(config.DATA_DIR / "risk_exposures_sample.csv")
    aligned = rx.align_to_universe(df, ["ETN", "GEV"], pd.Timestamp("2025-05-01"), lag_days=42)
    assert aligned.loc["GEV"].isna().all()


def test_portfolio_exposure_report_weighted_average():
    df = rx.load_risk_exposures(config.DATA_DIR / "risk_exposures_sample.csv")
    weights = pd.Series({"ETN": 0.6, "HUBB": 0.4})
    report = rx.portfolio_exposure_report(df, weights, pd.Timestamp("2025-05-01"), lag_days=42)
    assert report["value"] == pytest.approx(0.6 * 0.42 + 0.4 * 0.10)
    assert report["momentum"] == pytest.approx(0.6 * -0.15 + 0.4 * 0.55)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `poetry run pytest tests/electrification_strategy/test_risk_exposures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'electrification_strategy.risk_exposures'`

- [ ] **Step 4: Implement the module**

```python
# electrification_strategy/risk_exposures.py
"""Adapter for Client-provided (Addenda) risk-factor exposures (spec 3.3).

No real file exists yet -- ships against a placeholder schema (long format:
date, ticker, factor, exposure) with a documented swap-in point. See
electrification_strategy/data/risk_exposures_sample.csv for the schema
contract and docs/technical-reference.md for what's needed from the Client.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from electrification_strategy import config

_COLUMNS = ["date", "ticker", "factor", "exposure"]
DEFAULT_PATH = config.DATA_DIR / "risk_exposures.csv"


def load_risk_exposures(path: Path | None = None) -> pd.DataFrame:
    path = path or DEFAULT_PATH
    if not Path(path).exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(path, comment="#")
    df["date"] = pd.to_datetime(df["date"])
    return df[_COLUMNS]


def align_to_universe(
    exposures: pd.DataFrame, tickers: list[str], asof: pd.Timestamp,
    lag_days: int = config.REBALANCE_LAG_DAYS,
) -> pd.DataFrame:
    effective_date = asof - pd.Timedelta(days=lag_days)
    eligible = exposures[exposures["date"] <= effective_date]
    if eligible.empty:
        factors = sorted(exposures["factor"].unique()) if not exposures.empty else []
        return pd.DataFrame(index=tickers, columns=factors, dtype=float)
    latest = eligible.sort_values("date").groupby(["ticker", "factor"]).last()["exposure"]
    wide = latest.unstack("factor")
    return wide.reindex(tickers)


def portfolio_exposure_report(
    exposures: pd.DataFrame, weights: pd.Series, asof: pd.Timestamp,
    lag_days: int = config.REBALANCE_LAG_DAYS,
) -> pd.Series:
    aligned = align_to_universe(exposures, list(weights.index), asof, lag_days)
    w = weights.reindex(aligned.index).fillna(0.0)
    return aligned.mul(w, axis=0).sum(axis=0)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/electrification_strategy/test_risk_exposures.py -v`
Expected: PASS (all five tests)

- [ ] **Step 6: Commit**

```bash
git add electrification_strategy/risk_exposures.py electrification_strategy/data/risk_exposures_sample.csv \
        tests/electrification_strategy/test_risk_exposures.py
git commit -m "feat(electrification): add client risk-exposure adapter against placeholder schema

No Addenda risk-model file exists yet; ships with a documented schema
contract, a synthetic fixture, and tests, so the real file is a drop-in
swap once received."
```

---

## Task 3: Document the forward-power descope decision

**Files:**
- Modify: `docs/RESEARCH-LOG.md` (§4 Open items)

No code. Records the decision from spec §3.2 so it isn't silently lost.

- [ ] **Step 1: Edit RESEARCH-LOG.md §4**

Use the `Edit` tool (existing file) to add, under §4 Open items:

```markdown
- **Forward power price data (SOW Phase 1 item 2) — DESCOPED for Screened Electrification
  (2026-09-15).** STEO `ELWHU_*` (scoped-not-run) and ERCOT RT/DA spread (feasible) remain
  available future work, but the shipped strategy doesn't need a forward-power input, and the
  stress-beta gate reuse (item below) is realized/spot GSI data, not forward-looking. Revisit
  only if a future signal specifically needs a forward-looking stress proxy. See
  `docs/steo-archive-feasibility-results.md`, `docs/rt-da-spread-feasibility-results.md`,
  `docs/superpowers/specs/2026-09-15-screened-electrification-sow-plan-design.md` §3.2.
```

- [ ] **Step 2: Commit**

```bash
git add docs/RESEARCH-LOG.md
git commit -m "docs(research-log): record forward-power descope decision for Screened Electrification"
```

---

## Task 4: National GSI blend + reusable gate-math helpers (tested)

**Files:**
- Create: `electrification_stress_beta/__init__.py`
- Create: `electrification_stress_beta/national_gsi.py`
- Test: `tests/electrification_stress_beta/__init__.py`
- Test: `tests/electrification_stress_beta/test_national_gsi.py`

**Interfaces:**
- Consumes: `grid_resilience.signals.grid_stress_index.build_multi_iso_gsi`,
  `grid_resilience.signals.stress_events.build_full_event_calendar`,
  `grid_resilience.data.grid_data.{fetch_lmp, fetch_load, daily_lmp_summary, daily_load_summary}`,
  `grid_resilience.config.SUPPORTED_ISOS`.
- Produces: `NATIONAL_KEY: str`; `build_national_gsi(daily_lmp_by_iso, daily_load_by_iso,
  isos=SUPPORTED_ISOS) -> pd.DataFrame` (single `gsi` column, equal-weighted mean across ISOs);
  `fetch_and_build_national_gsi(start, end, isos=SUPPORTED_ISOS) -> pd.DataFrame` (live/cached
  orchestration); `uniform_ticker_iso_map(tickers: list[str]) -> dict[str, str]` — all consumed
  by Task 5's gate test.

This is spec §3.4 Step A: reuse `grid_resilience`'s GSI/conditional-beta machinery unmodified.
Equipment suppliers have no home ISO (unlike a utility's service territory), so every ticker maps
to one synthetic `NATIONAL_KEY`, letting `conditional_beta.gsi_for_ticker()` resolve every
ticker to the same blended series with zero changes to `grid_resilience/`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/electrification_stress_beta/test_national_gsi.py
import pandas as pd
import pytest

from electrification_stress_beta import national_gsi


def test_build_national_gsi_averages_per_iso(monkeypatch):
    idx = pd.bdate_range("2021-01-04", periods=5)
    gsi_a = pd.DataFrame({"gsi": [0.2, 0.4, 0.6, 0.8, 1.0]}, index=idx)
    gsi_b = pd.DataFrame({"gsi": [0.0, 0.2, 0.4, 0.6, 0.8]}, index=idx)
    monkeypatch.setattr(national_gsi, "build_full_event_calendar", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(national_gsi, "build_multi_iso_gsi",
                         lambda *a, **k: {"ISO_A": gsi_a, "ISO_B": gsi_b})
    out = national_gsi.build_national_gsi(
        {"ISO_A": pd.DataFrame({"lmp_max": [1]}), "ISO_B": pd.DataFrame({"lmp_max": [1]})},
        {"ISO_A": pd.DataFrame(), "ISO_B": pd.DataFrame()},
        isos=["ISO_A", "ISO_B"],
    )
    expected = (gsi_a["gsi"] + gsi_b["gsi"]) / 2.0
    pd.testing.assert_series_equal(out["gsi"], expected, check_names=False)


def test_build_national_gsi_empty_when_no_isos_available(monkeypatch):
    monkeypatch.setattr(national_gsi, "build_full_event_calendar", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(national_gsi, "build_multi_iso_gsi", lambda *a, **k: {})
    out = national_gsi.build_national_gsi({}, {}, isos=["ERCOT"])
    assert out.empty


def test_build_national_gsi_ignores_isos_outside_the_requested_set(monkeypatch):
    idx = pd.bdate_range("2021-01-04", periods=3)
    gsi_a = pd.DataFrame({"gsi": [0.1, 0.2, 0.3]}, index=idx)
    gsi_extra = pd.DataFrame({"gsi": [0.9, 0.9, 0.9]}, index=idx)
    monkeypatch.setattr(national_gsi, "build_full_event_calendar", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(national_gsi, "build_multi_iso_gsi",
                         lambda *a, **k: {"ISO_A": gsi_a, "ISO_EXTRA": gsi_extra})
    out = national_gsi.build_national_gsi(
        {"ISO_A": pd.DataFrame({"lmp_max": [1]})}, {"ISO_A": pd.DataFrame()}, isos=["ISO_A"],
    )
    pd.testing.assert_series_equal(out["gsi"], gsi_a["gsi"], check_names=False)


def test_uniform_ticker_iso_map():
    m = national_gsi.uniform_ticker_iso_map(["ETN", "HUBB"])
    assert m == {"ETN": national_gsi.NATIONAL_KEY, "HUBB": national_gsi.NATIONAL_KEY}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/electrification_stress_beta/test_national_gsi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'electrification_stress_beta'`

- [ ] **Step 3: Implement the module**

```python
# electrification_stress_beta/__init__.py
```

```python
# electrification_stress_beta/national_gsi.py
"""National (cross-ISO) Grid Stress Index for the Screened Electrification supplier
universe -- spec docs/superpowers/specs/2026-09-15-screened-electrification-sow-plan-design.md
§3.4 Step A.

Equipment suppliers (ETN, HUBB, GEV, ...) are national/global manufacturers with no
single home ISO, unlike grid_resilience's utilities (which have a service territory).
Rather than force a ticker_iso_map onto grid_stress_index.gsi_for_ticker(), this blends
grid_resilience's existing per-ISO GSI into one national series and maps every supplier
ticker to it via a synthetic single key -- reusing conditional_beta.py unmodified.
"""
from __future__ import annotations

import pandas as pd

from grid_resilience.config import SUPPORTED_ISOS
from grid_resilience.data.grid_data import fetch_lmp, fetch_load, daily_lmp_summary, daily_load_summary
from grid_resilience.signals.stress_events import build_full_event_calendar
from grid_resilience.signals.grid_stress_index import build_multi_iso_gsi

NATIONAL_KEY = "NATIONAL"


def build_national_gsi(
    daily_lmp_by_iso: dict[str, pd.DataFrame],
    daily_load_by_iso: dict[str, pd.DataFrame],
    isos: list[str] = SUPPORTED_ISOS,
) -> pd.DataFrame:
    """Equal-weighted mean of each ISO's GSI, per day. Returns a DataFrame with a
    single 'gsi' column, or an empty DataFrame if no requested ISO has data."""
    events_df = build_full_event_calendar(daily_lmp_by_iso, isos=isos)
    gsi_by_iso = build_multi_iso_gsi(daily_lmp_by_iso, daily_load_by_iso, events_df)
    series = [df["gsi"] for iso, df in gsi_by_iso.items() if iso in isos and not df.empty]
    if not series:
        return pd.DataFrame(columns=["gsi"])
    national = pd.concat(series, axis=1).mean(axis=1, skipna=True)
    return national.to_frame("gsi")


def fetch_and_build_national_gsi(
    start: str, end: str, isos: list[str] = SUPPORTED_ISOS,
) -> pd.DataFrame:
    """Fetch raw LMP/load for each ISO (cached) and build the blended national GSI.
    Not unit tested directly -- an orchestration layer, consistent with
    grid_resilience.main.run() also being covered only by real runs, not pytest.
    Covered by Task 5's real-data gate run."""
    daily_lmp_by_iso, daily_load_by_iso = {}, {}
    for iso in isos:
        lmp_raw = fetch_lmp(iso, start, end)
        load_raw = fetch_load(iso, start, end)
        daily_lmp_by_iso[iso] = daily_lmp_summary(lmp_raw, iso)
        daily_load_by_iso[iso] = daily_load_summary(load_raw)
    return build_national_gsi(daily_lmp_by_iso, daily_load_by_iso, isos=isos)


def uniform_ticker_iso_map(tickers: list[str]) -> dict[str, str]:
    """Map every ticker to NATIONAL_KEY so conditional_beta.gsi_for_ticker() resolves
    every ticker to the same blended national series, unmodified."""
    return {t: NATIONAL_KEY for t in tickers}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/electrification_stress_beta/test_national_gsi.py -v`
Expected: PASS (all four tests)

- [ ] **Step 5: Commit**

```bash
git add electrification_stress_beta/ tests/electrification_stress_beta/
git commit -m "feat(stress-beta): national GSI blend reusing grid_resilience unmodified

Equipment suppliers have no home ISO (unlike utility service territories),
so every ticker maps to one blended national GSI via a synthetic key,
letting conditional_beta.py's gsi_for_ticker() work with zero changes to
grid_resilience/. Spec §3.4 Step A."
```

---

## Task 5: Run the stress-beta gate against real data

**Files:**
- Create: `electrification_stress_beta/gate_test.py`
- Create: `electrification_stress_beta/gate_result.json` (generated by running the script)
- Create: `docs/electrification-stress-beta-gate-results.md`

**Interfaces:**
- Consumes: `electrification_stress_beta.national_gsi.{fetch_and_build_national_gsi,
  uniform_ticker_iso_map, NATIONAL_KEY}`, `grid_resilience.signals.conditional_beta.rolling_stress_betas`,
  `electrification_strategy.universe.load_seed`, `grid_equipment_basket.data.prices.fetch_prices`.
- Produces: `gate_result.json` with keys `t_primary_2021on, n_primary, t_pre_2021, n_pre,
  gate1_pass, sign_primary` — read by Task 6 to decide which branch to execute.

Not TDD (the point is to observe a real, unknown-in-advance result) but every step is concrete.
This is the "build it properly" instruction, run exactly once per the time-box constraint.

- [ ] **Step 1: Write the gate script**

```python
# electrification_stress_beta/gate_test.py
"""Stress-beta gate test for the Screened Electrification supplier universe.
Spec: docs/superpowers/specs/2026-09-15-screened-electrification-sow-plan-design.md §3.4.

Pre-registered gate, fixed before this script is run:
  Gate 1: pooled monthly tercile-spread t-stat |t| >= 2 on the 2021-01-01-onward window
          (the tight-grid era -- see the spec's regime-dating note). The pre-2021 window
          is reported for context only, not required to match sign.

Run:
    poetry run python -m electrification_stress_beta.gate_test
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from grid_equipment_basket.data.prices import fetch_prices
from grid_resilience.config import SUPPORTED_ISOS
from grid_resilience.signals.conditional_beta import rolling_stress_betas

from electrification_strategy import universe
from electrification_stress_beta import national_gsi

START, END = "2017-06-01", "2026-08-31"
PRIMARY_WINDOW_START = "2021-01-01"
GATE_T_THRESHOLD = 2.0
OUT_PATH = Path(__file__).parent / "gate_result.json"


def _tercile_spread(betas: pd.DataFrame) -> pd.Series:
    rows = []
    for date, grp in betas.groupby(level="date"):
        g = grp.dropna(subset=["stress_beta", "fwd_ret"])
        if len(g) < 6:
            continue
        n = len(g)
        top = g.nlargest(max(1, n // 3), "stress_beta")["fwd_ret"].mean()
        bot = g.nsmallest(max(1, n // 3), "stress_beta")["fwd_ret"].mean()
        rows.append({"date": date, "spread": top - bot})
    return pd.DataFrame(rows).set_index("date")["spread"] if rows else pd.Series(dtype=float)


def _gate1_tstat(spread: pd.Series) -> tuple[float, int]:
    s = spread.dropna()
    if len(s) < 3:
        return float("nan"), len(s)
    t = float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))
    return t, len(s)


def main() -> dict:
    tickers = sorted(universe.load_seed().index)
    prices = fetch_prices(tickers + ["SPY"], START, END)
    rets = prices[tickers].pct_change()
    sector_r = rets.mean(axis=1).rename("sector_return")

    national = national_gsi.fetch_and_build_national_gsi(START, END, isos=SUPPORTED_ISOS)
    gsi_by_iso = {national_gsi.NATIONAL_KEY: national}
    ticker_iso_map = national_gsi.uniform_ticker_iso_map(tickers)

    rebalance_dates = pd.date_range(START, END, freq="MS")
    betas = rolling_stress_betas(rets, gsi_by_iso, ticker_iso_map, sector_r, rebalance_dates)

    fwd_ret = (rets.resample("MS").apply(lambda x: (1 + x).prod() - 1)).shift(-1)
    betas = betas.join(fwd_ret.stack().rename("fwd_ret"), how="left")

    spread = _tercile_spread(betas)
    primary = spread.loc[PRIMARY_WINDOW_START:]
    pre = spread.loc[:PRIMARY_WINDOW_START]

    t_primary, n_primary = _gate1_tstat(primary)
    t_pre, n_pre = _gate1_tstat(pre)

    result = {
        "t_primary_2021on": t_primary, "n_primary": n_primary,
        "t_pre_2021": t_pre, "n_pre": n_pre,
        "gate1_pass": bool(abs(t_primary) >= GATE_T_THRESHOLD) if t_primary == t_primary else False,
        "sign_primary": "positive" if (t_primary == t_primary and t_primary > 0) else "negative",
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against real data**

Run: `poetry run python -m electrification_stress_beta.gate_test`

This fetches real PJM/ERCOT/MISO/CAISO/SPP LMP+load data (cached under
`grid_resilience/data/cache/`) and real prices for the 27-name universe (cached under
`electrification_strategy/data/cache/`). Expect several minutes on a cold cache. Read the
printed/written `gate_result.json`.

- [ ] **Step 3: Write the results doc**

Fill in the four bracketed values below from the actual `gate_result.json` produced in Step 2,
then run as a `Bash` heredoc (do not use the `Write` tool, per `CLAUDE.md`):

```bash
cat > docs/electrification-stress-beta-gate-results.md << 'EOF'
# Stress-beta gate — Screened Electrification supplier universe

**Date:** 2026-09-15 (or the actual run date)
**Spec:** docs/superpowers/specs/2026-09-15-screened-electrification-sow-plan-design.md §3.4

## Methodology

Reused grid_resilience's GSI (LMP z-score 40% / congestion fraction 25% / reserve tightness
20% / named-event flag 15%) and conditional_beta.py unmodified (electrification_stress_beta/
national_gsi.py), blended equal-weighted across ERCOT/PJM/MISO/CAISO/SPP into one national
series (suppliers have no home ISO, unlike a utility's service territory), and applied
uniformly to the 27-name `screen` universe. Monthly top-tercile-minus-bottom-tercile
stress_beta spread portfolio, pooled t-stat on the forward monthly spread return.

Gate 1 (pre-registered): |t| >= 2.0 on the 2021-01-01-onward window (tight-grid era — see
below), pre-2021 reported for context only.

## Regime dating

Reuters (Maguire, 2026-07-20, "America's power grid is choking on costly congestion")
reports PJM congestion cost $777.8M in June 2026 and ~$1B in May 2026, concentrated in
PA/MD/Northern Virginia/NJ. This project's own ancillary/scarcity-pricing probe
(docs/ancillary-scarcity-pricing-feasibility-results.md) already found no timing signal on
2018-2021 (|t| < 1.1). February 2021 (ERCOT Winter Storm Uri) is the commonly-cited
tightening inflection.

## Result

| window | t-stat | n (months) |
|---|--:|--:|
| 2021-01 → present (primary) | [t_primary_2021on] | [n_primary] |
| pre-2021 (context) | [t_pre_2021] | [n_pre] |

**Gate 1: [PASS / FAIL]** (sign: [sign_primary])

## Verdict

[If PASS: describe the sign and magnitude, and that Task 6 wires it in as a scoring tilt.]
[If FAIL: state plainly that grid stress does not show a statistically significant
relationship with supplier returns on this universe/window, consistent with ~29 prior
signal-search attempts on this project. Task 6 instead formalizes the strategy's existing
built-in protections as the deliverable — see docs/electrification-stress-sensitivity-response.md.]
EOF
```

- [ ] **Step 4: Commit**

```bash
git add electrification_stress_beta/gate_test.py electrification_stress_beta/gate_result.json \
        docs/electrification-stress-beta-gate-results.md
git commit -m "research(electrification): run the stress-beta gate on the supplier universe

First test of stress beta on Screened Electrification's 27-name supplier
universe (prior tests were all on grid_resilience's utilities or the
merchant-generator panel). Pre-registered gate, run once. Verdict: [PASS/FAIL]."
```

- [ ] **Step 5: Update RESEARCH-LOG.md**

Use the `Edit` tool to add a row to `docs/RESEARCH-LOG.md` §3 "Timing signals" table (or
"Strategy construction" table if PASS) with the verdict and a link to
`docs/electrification-stress-beta-gate-results.md`, per this repo's update convention. Commit
with a `docs(research-log): ...` message.

---

## Task 6a: [IF Task 5 PASSED] Wire stress beta into the strategy

**Only execute this task if `gate_result.json["gate1_pass"] is true`. Otherwise skip to Task 6b.**

**Files:**
- Create: `electrification_strategy/stress_beta_signal.py`
- Test: `tests/electrification_strategy/test_stress_beta_signal.py`
- Modify: `electrification_strategy/config.py` (add `STRESS_BETA_TILT_STRENGTH`)
- Modify: `electrification_strategy/universe.py` (add `"screen_stress_tilt"` to `CONSTRUCTIONS`)

**Interfaces:**
- Consumes: `electrification_stress_beta` (Task 4), `grid_equipment_basket.basket.apply_cap`.
- Produces: `stress_beta_tilt_weights(base_weights: pd.Series, stress_betas: pd.Series,
  tilt_strength: float = config.STRESS_BETA_TILT_STRENGTH) -> pd.Series` — consumed by a new
  `target_fn`-compatible closure in `universe.py`.

- [ ] **Step 1: Add the config constant**

```python
# electrification_strategy/config.py, after HEDGE_* block
# -- stress-beta tilt (spec 3.4, only wired if the gate passed) ----------
STRESS_BETA_TILT_STRENGTH: float = 0.5   # starting default; not yet grid-searched
```

- [ ] **Step 2: Write the failing test**

```python
# tests/electrification_strategy/test_stress_beta_signal.py
import pandas as pd
import pytest

from electrification_strategy import stress_beta_signal as sbs


def test_tilt_weights_favor_higher_stress_beta():
    base = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    betas = pd.Series({"A": 2.0, "B": 1.0, "C": -1.0, "D": -2.0})
    tilted = sbs.stress_beta_tilt_weights(base, betas, tilt_strength=0.5)
    assert tilted["A"] > tilted["B"] > tilted["C"] > tilted["D"]
    assert tilted.sum() == pytest.approx(1.0)


def test_tilt_weights_missing_beta_falls_back_to_base():
    base = pd.Series({"A": 0.5, "B": 0.5})
    betas = pd.Series({"A": 1.0})  # B has no beta estimate
    tilted = sbs.stress_beta_tilt_weights(base, betas, tilt_strength=0.5)
    assert tilted.sum() == pytest.approx(1.0)
    assert not tilted.isna().any()
```

- [ ] **Step 3: Run to verify failure, then implement**

```python
# electrification_strategy/stress_beta_signal.py
"""Stress-beta tilt -- only wired in because the pre-registered gate in
docs/electrification-stress-beta-gate-results.md passed. See spec §3.4."""
from __future__ import annotations

import pandas as pd

from grid_equipment_basket.basket import apply_cap
from electrification_strategy import config


def stress_beta_tilt_weights(
    base_weights: pd.Series, stress_betas: pd.Series,
    tilt_strength: float = config.STRESS_BETA_TILT_STRENGTH,
) -> pd.Series:
    betas = stress_betas.reindex(base_weights.index)
    z = (betas - betas.mean()) / betas.std(ddof=0) if betas.std(ddof=0) > 0 else betas * 0.0
    z = z.fillna(0.0)
    raw = base_weights * (1.0 + tilt_strength * z)
    raw = raw.clip(lower=0.0)
    normalized = raw / raw.sum() if raw.sum() > 0 else base_weights
    return apply_cap(normalized, config.MAX_SINGLE_NAME_WEIGHT)
```

Run: `poetry run pytest tests/electrification_strategy/test_stress_beta_signal.py -v`
Expected: PASS after implementation.

- [ ] **Step 4: Wire a new construction into `universe.py`**

Add `"screen_stress_tilt"` to the `CONSTRUCTIONS` tuple and extend `target_fn` to accept an
optional `stress_betas: pd.Series | None` kwarg, applying `stress_beta_tilt_weights` to the
`"screen"` pool's equal weights when the construction name is `"screen_stress_tilt"`. Add a
corresponding test in `tests/electrification_strategy/test_universe.py` following that file's
existing patterns (read it first for the exact fixture style used there).

- [ ] **Step 5: Commit**

```bash
git add electrification_strategy/stress_beta_signal.py electrification_strategy/config.py \
        electrification_strategy/universe.py tests/electrification_strategy/test_stress_beta_signal.py \
        tests/electrification_strategy/test_universe.py
git commit -m "feat(electrification): wire stress-beta tilt into a new screen_stress_tilt construction

Gate passed (docs/electrification-stress-beta-gate-results.md). Tilt
strength is a starting default, not yet grid-searched -- flag as follow-up."
```

---

## Task 6b: [IF Task 5 FAILED] Formalize the existing protections as the deliverable

**Only execute this task if `gate_result.json["gate1_pass"] is false`. Otherwise skip (Task 6a
already ran).**

**Files:**
- Create: `docs/electrification-stress-sensitivity-response.md`

No code changes — this branch's deliverable is documentation, per spec §3.4's explicit
instruction not to force a failed signal into production.

- [ ] **Step 1: Write the reframing doc**

```bash
cat > docs/electrification-stress-sensitivity-response.md << 'EOF'
# Stress Sensitivity & Grid-Event Response

The pre-registered stress-beta gate (docs/electrification-stress-beta-gate-results.md) did
not find a statistically significant relationship between the national Grid Stress Index and
Screened Electrification supplier returns. Rather than force a null result into production,
this document is the Phase 1 "enhanced signal component" deliverable in its place: it
formalizes the three rules already built into the shipped strategy
(docs/electrification-strategy-proposal-v5.md) as the practical answer to "how sensitive is
this book to grid-stress-correlated shocks, and how is that managed."

## The three built-in protections, mapped to stress categories

| Stress category | Protection | Mechanism |
|---|---|---|
| Growth scare / credit stress, currency or geopolitical shock | 15% GLD/IEF macro reserve | Bonds rise on a growth scare; gold rises on a currency/geopolitical shock. Either way this sleeve holds value while the supplier book falls. |
| Valuation overextension ahead of a de-rate | Graduated valuation-extension de-risk | When holdings run well above trend and the wider market, exposure steps down (100% -> ~80% -> ~60%), taking money off the table before a peak. |
| Volatility regime shift | 20%-target vol-targeting | Daily re-levering keeps realized volatility near a 20% annual target, moving to cash as swings widen and back as they calm. |

## Empirical evidence: the episode-drawdown table

(Reproduce the table from docs/electrification-strategy-proposal-v5.md "How it held up in
specific market shocks" -- COVID, 2022 rate shock, DeepSeek, tariff shock, 2026 selloff --
Option A vs Option B vs S&P 500 vs VOLT.)

## Why this satisfies the SOW's intent without overstating a null result

The SOW's Phase 1 item 4 asks for an "enhanced signal component" built around measured
sensitivity to grid stress. A literal regression-based stress beta does not survive an honest
test on this universe (see the gate results doc for the full methodology, including why the
economic mechanism for equipment suppliers -- crisis creates demand for their product -- is
genuinely different from a utility's service-territory exposure, and why it was still worth
testing rather than assumed to fail). The three protections above are the strategy's actual,
already-validated response to stress-correlated shocks, quantified against five real market
episodes rather than a single regression coefficient.
EOF
```

- [ ] **Step 2: Commit**

```bash
git add docs/electrification-stress-sensitivity-response.md
git commit -m "docs(electrification): formalize built-in protections as the Phase 1 stress-sensitivity deliverable

Gate failed (docs/electrification-stress-beta-gate-results.md); this
satisfies the SOW's underlying intent without misrepresenting a negative
regression result as a working signal. Spec §3.4 FAIL branch."
```

---

## Task 7: Wire risk-exposure and short-interest diagnostics into the backtest output

**Files:**
- Modify: `electrification_strategy/backtest.py` (`write_outputs`)
- Modify: `electrification_strategy/__main__.py` (CLI wiring)
- Test: `tests/electrification_strategy/test_backtest.py` (extend)
- Test: `tests/electrification_strategy/test_cli.py` (extend)

**Interfaces:**
- Consumes: `risk_exposures.{load_risk_exposures, portfolio_exposure_report}` (Task 2),
  `short_interest.{fetch_short_interest, crowding_diagnostic}` (Task 1).
- Produces: `write_outputs(..., risk_exposures_df=None, short_interest_history=None)` writes
  two additional optional CSVs: `risk_exposure_report.csv`, `short_interest_diagnostic.csv`.

This is Phase 2 item 1's "diagnostic, not a hard constraint" wiring (spec §4) — it runs
regardless of Task 6a/6b's branch outcome, since no real Addenda risk data exists yet to
optimize against.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/electrification_strategy/test_backtest.py
def test_write_outputs_includes_diagnostics_when_provided(tmp_path, synthetic_prices, synthetic_dfii10):
    from electrification_strategy import backtest, risk_exposures, short_interest
    result = backtest.run_comparison("2018-01-01", "2024-12-31", synthetic_prices, synthetic_dfii10,
                                      constructions=("marquee",))
    winner = backtest.pick_winner(result)

    rx_df = risk_exposures.load_risk_exposures()  # sample fixture path is the default when file absent -> empty; use explicit sample
    weights = pd.Series({"ETN": 0.5, "HUBB": 0.5})
    exposures_out = risk_exposures.portfolio_exposure_report(
        risk_exposures.load_risk_exposures(risk_exposures.config.DATA_DIR / "risk_exposures_sample.csv"),
        weights, pd.Timestamp("2024-12-31"))

    si_history = pd.DataFrame([
        {"ticker": "DLR", "shares_short": 1, "short_ratio": 2.0, "short_pct_float": 0.03,
         "date_short_interest": pd.Timestamp("2024-12-01"), "fetched_at": pd.Timestamp("2024-12-15")},
        {"ticker": "EQIX", "shares_short": 1, "short_ratio": 1.0, "short_pct_float": 0.01,
         "date_short_interest": pd.Timestamp("2024-12-01"), "fetched_at": pd.Timestamp("2024-12-15")},
    ])

    backtest.write_outputs(result, winner, tmp_path, make_plot=False,
                            risk_exposures_df=exposures_out.to_frame("exposure"),
                            short_interest_history=si_history)
    assert (tmp_path / "risk_exposure_report.csv").exists()
    assert (tmp_path / "short_interest_diagnostic.csv").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `poetry run pytest tests/electrification_strategy/test_backtest.py -k diagnostics -v`
Expected: FAIL with `TypeError: write_outputs() got an unexpected keyword argument 'risk_exposures_df'`

- [ ] **Step 3: Extend `write_outputs`**

In `electrification_strategy/backtest.py`, change the signature and add at the end of the
function body (before the `if make_plot:` block):

```python
def write_outputs(result, winner, out_dir, make_plot=True,
                   risk_exposures_df=None, short_interest_history=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # ... existing body unchanged ...

    if risk_exposures_df is not None and not risk_exposures_df.empty:
        risk_exposures_df.to_csv(out / "risk_exposure_report.csv")

    if short_interest_history is not None and not short_interest_history.empty:
        from electrification_strategy.short_interest import crowding_diagnostic
        crowding_diagnostic(short_interest_history).to_csv(out / "short_interest_diagnostic.csv")
```

- [ ] **Step 4: Run to verify it passes**

Run: `poetry run pytest tests/electrification_strategy/test_backtest.py -k diagnostics -v`
Expected: PASS

- [ ] **Step 5: Wire into the CLI**

In `electrification_strategy/__main__.py`, before the `backtest.write_outputs(...)` call, add:

```python
from electrification_strategy import risk_exposures, short_interest
...
rx_df = risk_exposures.load_risk_exposures()
si_snapshot = short_interest.fetch_short_interest(list(config.HEDGE_SHORT_TICKERS))
si_history_path = config.CACHE_DIR / config.SHORT_INTEREST_CACHE_FILE
si_history = pd.read_parquet(si_history_path) if si_history_path.exists() else pd.DataFrame()
```

and pass `risk_exposures_df=rx_df if not rx_df.empty else None, short_interest_history=si_history`
into the existing `backtest.write_outputs(...)` call. Extend `test_cli.py` with a
`monkeypatch.setattr(cli, "fetch_short_interest", ...)`-style smoke test following the existing
`test_cli_smoke` pattern (read that test first — same file).

- [ ] **Step 6: Run the full electrification test suite**

Run: `poetry run pytest tests/electrification_strategy/ -v`
Expected: PASS (all tests, old and new)

- [ ] **Step 7: Commit**

```bash
git add electrification_strategy/backtest.py electrification_strategy/__main__.py \
        tests/electrification_strategy/test_backtest.py tests/electrification_strategy/test_cli.py
git commit -m "feat(electrification): wire risk-exposure and short-interest diagnostics into output bundle

Reporting-only (spec §4 item 1) -- no real Addenda data exists yet to
optimize against, so these ship as diagnostic CSVs alongside the existing
metrics/episode/returns outputs."
```

---

## Task 8: Re-run the rolling backtest end-to-end, confirm reproducibility

**Files:** none created/modified — this is a verification task.

- [ ] **Step 1: Clear the cache and rebuild from scratch**

```bash
mv electrification_strategy/data/cache electrification_strategy/data/cache.bak
mv grid_resilience/data/cache grid_resilience/data/cache.bak
poetry run python -m electrification_strategy --output output_electrification_v2
```

Expected: completes without error, `output_electrification_v2/` contains `metrics.csv`,
`episode_drawdowns.csv`, `plateau.json`, `returns.csv`, `performance.png`, and (if Task 7's
CLI wiring found any data) `risk_exposure_report.csv` / `short_interest_diagnostic.csv`.

- [ ] **Step 2: Restore the caches (this was a reproducibility check, not a cache replacement)**

```bash
rm -rf electrification_strategy/data/cache grid_resilience/data/cache
mv electrification_strategy/data/cache.bak electrification_strategy/data/cache
mv grid_resilience/data/cache.bak grid_resilience/data/cache
```

- [ ] **Step 3: Run the full test suite one more time**

Run: `poetry run pytest tests/ -v`
Expected: PASS (no regressions from Tasks 1–7)

- [ ] **Step 4: Record the winner cell for Task 9**

Open `output_electrification_v2/metrics.csv` and `plateau.json`; note the `winner` construction
name and its `cagr`, `sharpe`, `max_dd`, `beta` — needed verbatim in Task 9.

No commit needed (verification-only task; `output_electrification_v2/` is already `??` in git
status from before this plan — leave it for Task 9 to reference, don't commit generated output
directories per this repo's existing convention of `output_*/` being untracked scratch).

---

## Task 9: Write the Phase 2 findings report

**Files:**
- Create: `docs/electrification-strategy-proposal-v6.md`

- [ ] **Step 1: Write the report**

Base it on `docs/electrification-strategy-proposal-v5.md`'s structure (read it first for exact
tone/format — this repo's client-facing writing style is plain-language, tables, no jargon).
Using `Bash` heredoc (`cat > docs/electrification-strategy-proposal-v6.md << 'EOF' ... EOF`),
reproduce v5's sections and fold in:

1. All of v5's content, updated with Task 8's actual output numbers if they differ from v5's
   (they shouldn't, absent Task 6a — Task 6a/6b don't change the `screen +val+sleeve` cell
   unless Task 6a's new `screen_stress_tilt` construction outranks it under `pick_winner`).
2. A new "Stress-Beta Signal Test" section: the gate methodology, result, and verdict from
   `docs/electrification-stress-beta-gate-results.md` (either "wired in as `screen_stress_tilt`"
   or "tested, not adopted — see the built-in protections instead," linking
   `docs/electrification-stress-sensitivity-response.md`).
3. A new "Cost & Liquidity Diagnostics" section: the short-interest crowding numbers from
   `short_interest_diagnostic.csv` (Task 7/8) for DLR/EQIX, explicitly closing v5's own flagged
   "trading costs at scale still to be estimated" limitation — with the honest caveat that this
   is a live-snapshot-only history (Task 1), not a backtested cost estimate.
4. A new "Client Risk-Factor Alignment" section: state that the adapter (Task 2) is built and
   tested against a placeholder schema, pending the real Addenda file.
5. A "Market Context" callout citing the Reuters PJM-congestion figures ($777.8M June 2026,
   ~$1B May 2026) as current evidence for the physical-need thesis already in v5's Summary.
6. An updated "Method notes" section noting this is v6, listing what changed from v5.

- [ ] **Step 2: Generate the HTML/PDF (matching this repo's convention)**

Run: `poetry run python tools/md2html.py docs/electrification-strategy-proposal-v6.md --theme light --pdf`

- [ ] **Step 3: Commit**

```bash
git add docs/electrification-strategy-proposal-v6.md docs/electrification-strategy-proposal-v6.html \
        docs/electrification-strategy-proposal-v6.pdf
git commit -m "docs: Screened Electrification proposal v6 — Phase 2 written findings report

Folds in the stress-beta gate outcome, short-interest cost diagnostic,
risk-exposure adapter status, and current PJM-congestion market evidence
into v5's structure. SOW Phase 2 item 3 deliverable."
```

---

## Task 10: Write the updated SOW (client-facing)

**Files:**
- Create: `docs/scope-of-work-v2-screened-electrification.md`

- [ ] **Step 1: Write the document**

Read `docs/scope-of-work-draft.md` first (already read during brainstorming — Parties,
Compensation, Data Cost Reimbursement, IP, Scope Limitations, Term, Signatures sections). Using
`Bash` heredoc, produce a new file that:

1. Copies **verbatim** the Compensation, Data Cost Reimbursement, IP, Scope Limitations, Term,
   and Signatures sections from v1 (only update the `**Date:**` line under Parties).
2. Rewrites **Background**: replace the stress-beta/18-utility framing with a description of
   Screened Electrification Equipment (the 27-name supplier screen + vol-target + macro sleeve),
   referencing `docs/electrification-strategy-proposal-v6.md`.
3. Rewrites **Phase 1** items to match what was actually delivered (Tasks 1–7 of this plan):
   short-interest diagnostic, risk-exposure adapter against a placeholder schema, forward-power
   descope decision, and the stress-beta gate with **conditional language** — state plainly that
   the gate was pre-registered and run once, and link the results doc, rather than promising a
   specific outcome.
4. Rewrites **Phase 2** items to reference the actual backtest re-run (Task 8) and the v6
   findings report (Task 9).
5. Rewrites **Phase 3** items to reference the technical reference (Task 11), runbook (Task 12),
   manifest (Task 13), and tagged release (Task 14) as this plan defines them.

- [ ] **Step 2: Generate the HTML/PDF**

Run: `poetry run python tools/md2html.py docs/scope-of-work-v2-screened-electrification.md --theme light --pdf`

- [ ] **Step 3: Commit**

```bash
git add docs/scope-of-work-v2-screened-electrification.md docs/scope-of-work-v2-screened-electrification.html \
        docs/scope-of-work-v2-screened-electrification.pdf
git commit -m "docs: SOW v2 — Screened Electrification Equipment engagement scope

Keeps v1's commercial/legal skeleton unchanged; rewrites Background and
Phase 1-3 bodies to match what's actually being delivered, including
honest conditional language for the stress-beta gate."
```

---

## Task 11: Docstring pass + technical reference

**Files:**
- Modify: any new module from Tasks 1, 2, 4, 6a lacking a module-level docstring (they already
  have one from their own implementation steps above — this task is an audit, not a rewrite)
- Create: `docs/technical-reference.md`

- [ ] **Step 1: Audit new modules for docstring completeness**

Run: `poetry run python -c "import ast, sys
for f in ['electrification_strategy/short_interest.py', 'electrification_strategy/risk_exposures.py', 'electrification_stress_beta/national_gsi.py']:
    tree = ast.parse(open(f).read())
    missing = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and not ast.get_docstring(n) and not n.name.startswith('_')]
    print(f, 'missing:', missing)"`

For any public function reported missing, add a one-line docstring (the function's purpose,
not a restatement of its name) directly in its source file.

- [ ] **Step 2: Write the technical reference**

```bash
cat > docs/technical-reference.md << 'EOF'
# Technical Reference — Screened Electrification Pipeline

One row per module: what it does, its key inputs/outputs, and its role in the pipeline.

## electrification_strategy/

| Module | Role | Key inputs | Key outputs |
|---|---|---|---|
| `config.py` | Frozen parameters (spec 6) | — | constants consumed by every other module |
| `universe.py` | Universe constructions (marquee/frozen/thematic/screen[_stress_tilt]) | `data/universe_seed.csv`, `data/etf_membership_2026.csv`, prices | `target_fn(construction, prices)` |
| `fred.py` | DFII10 (10yr real yield) fetch + cache | FRED API | `fetch_series` |
| `short_interest.py` | Short-interest crowding/cost diagnostic (Task 1) | yfinance `Ticker.info` | `fetch_short_interest`, `crowding_diagnostic` |
| `risk_exposures.py` | Client risk-factor adapter (Task 2) | `data/risk_exposures.csv` (placeholder schema — see below) | `load_risk_exposures`, `align_to_universe`, `portfolio_exposure_report` |
| `valuation_overlay.py` | Graduated de-risk multiplier | basket level, SPY returns | `extension_multiplier` |
| `hedge_overlay.py` | 15% GLD/IEF sleeve + conditional DLR/EQIX short | prices, DFII10 | `sleeve_returns`, `short_returns`, `apply_hedge` |
| `stress_beta_signal.py` | Stress-beta tilt (Task 6a, only if gate passed) | `electrification_stress_beta` output | `stress_beta_tilt_weights` |
| `backtest.py` | Signal -> weights -> P&L -> metrics | prices, DFII10 | `run_comparison`, `write_outputs` |
| `__main__.py` | CLI entry point | argv | `output_electrification*/` file bundle |

## electrification_stress_beta/ (Task 4-5)

| Module | Role | Key inputs | Key outputs |
|---|---|---|---|
| `national_gsi.py` | Blends grid_resilience's per-ISO GSI into one national series | `grid_resilience.data.grid_data` | `build_national_gsi`, `fetch_and_build_national_gsi`, `uniform_ticker_iso_map` |
| `gate_test.py` | Pre-registered stress-beta gate (run once) | prices, national GSI | `gate_result.json` |

## Reused, unmodified from grid_resilience/

| Module | Role |
|---|---|
| `signals/grid_stress_index.py` | GSI construction (LMP z-score / congestion / reserve tightness / event flag) |
| `signals/conditional_beta.py` | Rolling OLS stress-beta estimation |
| `signals/stress_events.py` | Named + algo-detected stress event calendar |
| `data/grid_data.py` | LMP/load fetch + daily summaries, per ISO |

## Schema contract: risk_exposures.csv (Client-provided, Task 2)

Long format, one row per date x ticker x factor: `date, ticker, factor, exposure`. Place at
`electrification_strategy/data/risk_exposures.csv` (not committed — Client-provided). See
`electrification_strategy/data/risk_exposures_sample.csv` for a synthetic example of the exact
shape expected.
EOF
```

- [ ] **Step 3: Commit**

```bash
git add docs/technical-reference.md electrification_strategy/short_interest.py \
        electrification_strategy/risk_exposures.py electrification_stress_beta/national_gsi.py
git commit -m "docs: technical reference covering every module's inputs/outputs/role

SOW Phase 3 item 1 deliverable."
```

---

## Task 12: Operational runbook

**Files:**
- Create: `docs/runbook.md`

- [ ] **Step 1: Write the runbook**

```bash
cat > docs/runbook.md << 'EOF'
# Operational Runbook — Screened Electrification Pipeline

## Setup

```bash
poetry install
```

Environment variables needed (see `grid_resilience/config.py` for the full list; only
required if re-running the stress-beta gate / GSI, not for the day-to-day strategy):
`PJM_API_KEY` (free, pjm.com/api). No key is required for MISO, CAISO, SPP, NYISO, ISO-NE, or
for the core `electrification_strategy` pipeline (equity prices via yfinance, DFII10 via FRED,
short interest via yfinance).

## Refreshing data

All fetches are cached to parquet under `electrification_strategy/data/cache/` (prices, FRED)
and `grid_resilience/data/cache/` (LMP/load, only touched if re-running the gate). To force a
full refresh, delete the relevant cache directory before running — see Task 8 of the
implementation plan for the exact clear/rebuild/restore sequence used to verify reproducibility.

## Running the pipeline

```bash
poetry run python -m electrification_strategy --start 2017-06-01 --output output_electrification
```

Flags:
- `--universe {marquee,frozen,thematic,screen,all}` — restrict to one construction (default `all`)
- `--no-plot` — skip the performance.png chart (faster, useful for CI/smoke runs)
- `--output PATH` — output directory (default `./output_electrification`)

## Rebalance cadence

Monthly, with a 42-day reporting lag (`config.REBALANCE_LAG_DAYS`) — a holding entering the
universe today is reflected in the portfolio 42 calendar days later, matching realistic
data-availability lag for the fundamental screen inputs. Vol-target and valuation-overlay
exposure are recalculated daily; the underlying holdings list changes roughly monthly.

## Interpreting output

| File | Contents |
|---|---|
| `metrics.csv` | CAGR / vol / Sharpe / MaxDD / beta / drag-vs-plain / feared-scenario P&L / corr-to-VOLT, one row per construction x overlay-stack cell, plus benchmark rows |
| `episode_drawdowns.csv` | Peak-to-trough drawdown per named market-shock episode, per cell |
| `plateau.json` | The pre-registered winner (per `config.WINNER_*` criteria) and the valuation-overlay sensitivity plateau |
| `returns.csv` | Daily return series per cell |
| `performance.png` | Growth-of-$1 + drawdown chart for the winning cell vs. `marquee plain` |
| `risk_exposure_report.csv` | Portfolio-level factor exposure (only written once a real `risk_exposures.csv` is in place — empty/absent otherwise) |
| `short_interest_diagnostic.csv` | Latest short-interest/crowding snapshot for the hedge short leg (DLR, EQIX) |

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `yfinance` returns all-NaN columns for some tickers | Rate limiting / partial batch download | Re-run; `fetch_prices` retries with backoff, see `grid_equipment_basket/data/prices.py` |
| `RuntimeWarning: FRED fetch ... failed` | FRED API unreachable | Falls back automatically to the committed `electrification_strategy/data/fred_DFII10.csv` — update that file periodically if running live for extended periods |
| `PJM_API_KEY` errors when re-running the stress-beta gate | Missing/expired key | Get a free key at pjm.com/api; only needed for `electrification_stress_beta` re-runs, not the core strategy |
| Stale cache producing outdated results | Cache never expires automatically | Delete the specific ticker's cache chunk under `data/cache/`, or the whole directory for a full refresh (see Task 8) |
| `universe_seed.csv` raises `ValueError: unknown sub_industry values` | A typo or a new sub-industry added without updating `config.SUB_INDUSTRIES` | Fix the CSV or extend `config.SUB_INDUSTRIES` |
EOF
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbook.md
git commit -m "docs: operational runbook — setup, refresh, rebalance, output, failure modes

SOW Phase 3 item 2 deliverable."
```

---

## Task 13: Deliverable manifest

**Files:**
- Create: `docs/deliverable-manifest.md`

- [ ] **Step 1: Write the manifest**

```bash
cat > docs/deliverable-manifest.md << 'EOF'
# Deliverable Manifest — Screened Electrification Engagement

What ships to the Client vs. what stays internal research history, per SOW Phase 3 item 4.

## Ships

- `electrification_strategy/` (the full package)
- `electrification_stress_beta/` (national GSI blend + gate test + `gate_result.json`)
- `grid_resilience/signals/grid_stress_index.py`, `conditional_beta.py`, `stress_events.py`,
  `data/grid_data.py` — reused unmodified by `electrification_stress_beta/`
- `tests/electrification_strategy/`, `tests/electrification_stress_beta/`
- `docs/electrification-strategy-proposal-v6.md` (+ .html/.pdf) — the findings report
- `docs/electrification-stress-beta-gate-results.md`
- `docs/electrification-stress-sensitivity-response.md` (if the gate failed)
- `docs/technical-reference.md`
- `docs/runbook.md`
- `docs/scope-of-work-v2-screened-electrification.md` (+ .html/.pdf)
- `pyproject.toml`, `poetry.lock`

## Stays internal

- `research/` (all ~29 tracked signal-search attempts — negative-result research history, not
  client deliverables)
- `docs/RESEARCH-LOG.md` and every other `docs/*-results.md` / `docs/handoff*.md` / `docs/compact*.md`
  (internal research trail; the v6 report and gate-results doc are the client-facing distillation)
- `output_*/` directories (scratch backtest output; the Client re-generates these via the runbook)
- `grid_resilience/` beyond the four reused signal modules listed above (the rest of that
  package belongs to the original, unshipped Grid Resilience strategy)
- `grid_equipment_basket/`, `backlog_factor/`, `dc_demand_factor/`, `transmission_rate_base/`,
  and other sibling probe packages not part of Screened Electrification
- `.venv/`, `.superpowers/`, `.claude/`, `scratchpad/`, `pitch/`

## Release step

Once the Client confirms this manifest, proceed to Task 14 (`git tag`).
EOF
```

- [ ] **Step 2: Commit**

```bash
git add docs/deliverable-manifest.md
git commit -m "docs: deliverable manifest — what ships vs. what stays internal research

SOW Phase 3 item 4 deliverable, precedes the tagged release."
```

---

## Task 14: Tagged release

**This step is a release action visible to the Client — do not run it without explicit
confirmation from the user at execution time, even though it's written here as a plan step.**

- [ ] **Step 1: Confirm readiness**

Verify: all of Tasks 1–13 are committed, `poetry run pytest tests/ -v` passes in full, and the
user/Client has confirmed `docs/deliverable-manifest.md`.

- [ ] **Step 2: STOP — get explicit user confirmation before tagging**

Ask the user to confirm the exact tag name and that they want to proceed (e.g.
"screened-electrification-v1.0"). Do not proceed on inference alone.

- [ ] **Step 3: Tag (only after explicit confirmation)**

```bash
git tag -a screened-electrification-v1.0 -m "Screened Electrification Equipment — Phase 1-3 delivery to Addenda Capital"
```

Do not push the tag to any remote without a separate, explicit instruction to do so.

---

## Self-Review Notes

- **Spec coverage:** §3.1 short interest → Task 1; §3.2 forward-power descope → Task 3; §3.3
  risk adapter → Task 2; §3.4 stress-beta gate (both branches) → Tasks 4/5/6a/6b; Phase 2 items
  1–3 → Tasks 7/8/9; Phase 3 items 1–4 → Tasks 11/12/13/14 (handoff sessions themselves are
  scheduling-dependent, noted in Task 12/13's framing rather than a standalone task); the SOW v2
  document → Task 10; the data-inclusion discipline and Reuters grounding → Tasks 5/9.
- **Placeholder scan:** the only bracketed values in the plan (`[t_primary_2021on]`, `[PASS/FAIL]`,
  etc.) are explicitly live-data insertion points filled from a prior step's real output, not
  unresolved TODOs — each is annotated with exactly which file/key it comes from.
- **Type/signature consistency:** `stress_beta_tilt_weights`, `crowding_diagnostic`,
  `portfolio_exposure_report`, `build_national_gsi`, `NATIONAL_KEY`, and `gate_result.json`'s
  keys are used identically wherever referenced across tasks.
