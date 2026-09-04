# Utility Capex-Guidance Revision Signal ("Deliverable D") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and gate-test an aggregate utility capex-guidance-revision signal (handoff
`docs/handoff_2026-09-03-transmission-project-filings.md` §6 Deliverable D) for the
grid-equipment basket, as both a monthly rank-IC timing signal and an exposure scaler.

**Architecture:** A hand/web-assembled seed CSV feeds a data-layer module
(`grid_resilience/data/utility_capex_guidance.py`: load, feasibility count, point-in-time
imputation of the DC-attributed sub-figure, panel aggregation) which in turn feeds a
signal-layer module (`grid_equipment_basket/capex_guidance_signal.py`: daily z-scored
composite, one-directional de-risk multiplier, two-sided scaler, and the pre-registered gate
report) built on the exact seams `ftr_signal.py` / `grid_regime.py` / `capex_signal.py` already
established in this package — `broadcast_daily`, `_trailing_zscore`, `overlay.apply_overlay_l2`,
and `grid_regime.gate_check`/`final_verdict` are reused, not reimplemented.

**Tech Stack:** Python, pandas, numpy, `statsmodels` (HAC/Newey-West OLS), `scipy.stats`
(Spearman), pytest. No new dependencies — both `statsmodels` and `scipy` are already used
elsewhere in this repo (`grid_resilience/signals/conditional_beta.py`,
`grid_resilience/portfolio/backtest.py`).

**Spec:** `docs/superpowers/specs/2026-09-04-capex-guidance-signal-design.md`

## Global Constraints

- Every seed-CSV row carries a `confidence` tag (H/M/L) — never asserted without one (spec §3.2).
- No figure in the seed panel is adjusted after seeing a backtest result — frozen once
  feasibility (§3.3) is scored (spec §3.2).
- Point-in-time discipline throughout: no function may use information dated after the row/date
  it is computing a value for. This is a correctness property, not a style preference — it is
  unit-tested explicitly in Tasks 6 and 8.
- `DC_GUIDANCE_PRIMARY_WINDOW = ("2023-01-01", "2026-08-31")`,
  `DC_GUIDANCE_PRIOR_WINDOW = ("2021-01-01", "2022-12-31")` — these are Deliverable D's own
  frozen pair per the handoff, **not** `config.REGIME_PRIMARY_WINDOW`/`REGIME_PRIOR_WINDOW`
  (spec §7).
- Feasibility kill runs before any backtest: < 8 usable utilities routes to the 3-name fallback;
  the fallback itself unusable stops the report with `verdict: "not_yet_testable"` rather than
  running a backtest on data too thin to trust (spec §3.3).
- Per CLAUDE.md: markdown files (`docs/*.md`, `README.md` edits) are written via `Bash cat >` /
  `Edit`, never the `Write` tool. After any code change, check whether it invalidates anything in
  `grid_equipment_basket/README.md` and update it in the same session if so.

---

## File Structure

| File | Responsibility |
|---|---|
| `grid_equipment_basket/config.py` | new `DC_GUIDANCE_*` / `DER_SHORT_SLEEVE` constants (edit, append) |
| `grid_resilience/data/seed/utility_capex_guidance.csv` | hand/web-assembled guidance panel |
| `grid_resilience/data/utility_capex_guidance.py` | load, feasibility count, point-in-time imputation, panel aggregation |
| `grid_equipment_basket/capex_guidance_signal.py` | composite, multiplier, scaler, stats helpers, the gate report, printable table |
| `grid_equipment_basket/__main__.py` | `--capex-guidance` CLI flag (edit) |
| `grid_equipment_basket/README.md` | doc update (edit) |
| `tests/data/test_utility_capex_guidance.py` | data-layer tests |
| `tests/grid_equipment_basket/test_capex_guidance_signal.py` | signal-layer tests |
| `docs/capex-guidance-signal-results.md` | results write-up (last task, after the live run) |

---

### Task 1: Config constants

**Files:**
- Modify: `grid_equipment_basket/config.py` (append after `REGIME_LADDER`, i.e. after line 162)

**Interfaces:**
- Produces: `config.DC_GUIDANCE_UNIVERSE`, `config.DC_GUIDANCE_MIN_UTILITIES`,
  `config.DC_GUIDANCE_FALLBACK`, `config.DER_SHORT_SLEEVE`, `config.DC_GUIDANCE_ZSCORE_WINDOW`,
  `config.DC_GUIDANCE_ZSCORE_MINP`, `config.DC_GUIDANCE_ZSCORE_WINSOR`,
  `config.DC_GUIDANCE_PRIMARY_WINDOW`, `config.DC_GUIDANCE_PRIOR_WINDOW`,
  `config.DC_GUIDANCE_HOLDOUT_WINDOW`, `config.DC_GUIDANCE_TTM_QUARTERS`,
  `config.DC_GUIDANCE_DERISK_FLOOR_Z`, `config.DC_GUIDANCE_DERISK_LO_MULT`,
  `config.DC_GUIDANCE_DERISK_GRID_FLOOR`, `config.DC_GUIDANCE_DERISK_GRID_LO`,
  `config.DC_GUIDANCE_SCALER_K`, `config.DC_GUIDANCE_SCALER_LO`, `config.DC_GUIDANCE_SCALER_HI`,
  `config.DC_GUIDANCE_SCALER_GRID_K`, `config.DC_GUIDANCE_SCALER_GRID_HI`,
  `config.DC_GUIDANCE_RANK_IC_MIN_T` — used by name in every later task.

- [ ] **Step 1: Append the constants**

Add to the end of `grid_equipment_basket/config.py`:

```python
# ── Utility capex-guidance revision signal ("Deliverable D", 2026-09-04) ───
# Spec: docs/superpowers/specs/2026-09-04-capex-guidance-signal-design.md
DC_GUIDANCE_UNIVERSE: list[str] = ["D", "AEP", "NEE", "SO", "ETR", "XEL", "DUK",
                                   "PCG", "EIX", "PPL", "FE", "AEE", "WEC", "CMS", "DTE"]
DC_GUIDANCE_MIN_UTILITIES: int = 8      # feasibility kill (spec s3.3)
DC_GUIDANCE_FALLBACK: list[str] = ["D", "AEP", "NEE"]

DER_SHORT_SLEEVE: list[str] = ["ENPH", "SEDG", "CHPT", "RUN", "BLNK", "STEM"]

DC_GUIDANCE_ZSCORE_WINDOW: int = REGIME_ZSCORE_WINDOW      # reuse 756/252/3.0, not re-tuned
DC_GUIDANCE_ZSCORE_MINP: int = REGIME_ZSCORE_MINP
DC_GUIDANCE_ZSCORE_WINSOR: float = REGIME_ZSCORE_WINSOR

DC_GUIDANCE_PRIMARY_WINDOW: tuple[str, str] = ("2023-01-01", "2026-08-31")
DC_GUIDANCE_PRIOR_WINDOW: tuple[str, str] = ("2021-01-01", "2022-12-31")
DC_GUIDANCE_HOLDOUT_WINDOW: tuple[str, str] = ("2026-03-01", "2026-08-31")
DC_GUIDANCE_TTM_QUARTERS: int = 4

# s6.1 one-directional de-risk -- fixed, plateau-probed, not fitted
DC_GUIDANCE_DERISK_FLOOR_Z: float = -0.5
DC_GUIDANCE_DERISK_LO_MULT: float = 0.6
DC_GUIDANCE_DERISK_GRID_FLOOR: tuple[float, ...] = (-0.25, -0.5, -0.75)
DC_GUIDANCE_DERISK_GRID_LO: tuple[float, ...] = (0.5, 0.6, 0.7)

# s6.2 two-sided scaler
DC_GUIDANCE_SCALER_K: float = 0.35
DC_GUIDANCE_SCALER_LO: float = 0.5
DC_GUIDANCE_SCALER_HI: float = 1.5
DC_GUIDANCE_SCALER_GRID_K: tuple[float, ...] = (0.25, 0.35, 0.45)
DC_GUIDANCE_SCALER_GRID_HI: tuple[float, ...] = (1.25, 1.5)

DC_GUIDANCE_RANK_IC_MIN_T: float = 2.0   # spec s5.1/5.2/5.4
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "from grid_equipment_basket import config; print(config.DC_GUIDANCE_PRIMARY_WINDOW, config.DER_SHORT_SLEEVE)"`
Expected: prints `('2023-01-01', '2026-08-31') ['ENPH', 'SEDG', 'CHPT', 'RUN', 'BLNK', 'STEM']` with no error.

- [ ] **Step 3: Commit**

```bash
git add grid_equipment_basket/config.py
git commit -m "feat(capex-guidance): add config constants for Deliverable D"
```

---

### Task 2: Seed CSV schema + loader

**Files:**
- Create: `grid_resilience/data/seed/utility_capex_guidance.csv` (header + 4 fixture rows only —
  real data population is Task 3)
- Create: `grid_resilience/data/utility_capex_guidance.py`
- Test: `tests/data/test_utility_capex_guidance.py`

**Interfaces:**
- Produces: `load_capex_guidance(seed: Path = SEED_CSV) -> pd.DataFrame` with columns
  `utility, report_date (datetime64), plan_start_year, plan_end_year, capex_plan_usd_m,
  revision_vs_prior_usd_m, revision_quality, prior_capex_plan_usd_m, dc_attributed_usd_m,
  dc_basis, source_type, source_detail, confidence`. Used by every later data-layer task.

- [ ] **Step 1: Write the CSV header + 4 fixture rows**

```bash
cat > grid_resilience/data/seed/utility_capex_guidance.csv << 'CSV_EOF'
utility,report_date,plan_start_year,plan_end_year,capex_plan_usd_m,revision_vs_prior_usd_m,dc_attributed_usd_m,dc_basis,source_type,source_detail,confidence
FIXTURE,2023-01-01,2023,2027,10000,,,none,earnings_call,fixture row - first vintage no prior,H
FIXTURE,2023-07-01,2023,2027,11000,,,none,earnings_call,fixture row - revision derived not stated,H
FIXTURE,2024-01-01,2024,2028,13000,2000,1500,stated,earnings_call,fixture row - stated revision + stated DC $,H
FIXTURE,2024-07-01,2024,2028,13500,500,,qualitative,earnings_call,fixture row - stated revision qualitative DC mention,M
CSV_EOF
```

- [ ] **Step 2: Write the failing test**

```python
# tests/data/test_utility_capex_guidance.py
import numpy as np
import pandas as pd
import pytest

from grid_resilience.data import utility_capex_guidance as udg


@pytest.fixture
def fixture_csv(tmp_path):
    src = udg.SEED_CSV
    dst = tmp_path / "fixture.csv"
    dst.write_text(src.read_text())
    return dst


def test_load_capex_guidance_parses_dates_and_sorts(fixture_csv):
    df = udg.load_capex_guidance(fixture_csv)
    assert list(df["utility"].unique()) == ["FIXTURE"]
    assert df["report_date"].is_monotonic_increasing
    assert df["report_date"].dtype.kind == "M"


def test_load_capex_guidance_keeps_stated_revision_as_is(fixture_csv):
    df = udg.load_capex_guidance(fixture_csv)
    row = df[df["report_date"] == "2024-01-01"].iloc[0]
    assert row["revision_vs_prior_usd_m"] == pytest.approx(2000.0)
    assert row["revision_quality"] == "stated"


def test_load_capex_guidance_derives_blank_revision_from_consecutive_levels(fixture_csv):
    df = udg.load_capex_guidance(fixture_csv)
    row = df[df["report_date"] == "2023-07-01"].iloc[0]
    # 11000 - 10000 (the prior row's level) = 1000, derived not stated
    assert row["revision_vs_prior_usd_m"] == pytest.approx(1000.0)
    assert row["revision_quality"] == "derived"


def test_load_capex_guidance_first_vintage_has_no_revision(fixture_csv):
    df = udg.load_capex_guidance(fixture_csv)
    row = df[df["report_date"] == "2023-01-01"].iloc[0]
    assert pd.isna(row["revision_vs_prior_usd_m"])
    assert row["revision_quality"] == "n/a"
    assert pd.isna(row["prior_capex_plan_usd_m"])


def test_load_capex_guidance_prior_capex_plan_is_the_immediately_prior_level(fixture_csv):
    df = udg.load_capex_guidance(fixture_csv)
    row = df[df["report_date"] == "2024-01-01"].iloc[0]
    assert row["prior_capex_plan_usd_m"] == pytest.approx(11000.0)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/data/test_utility_capex_guidance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'grid_resilience.data.utility_capex_guidance'`

- [ ] **Step 4: Write the implementation**

```python
# grid_resilience/data/utility_capex_guidance.py
"""Utility 5-yr capex-guidance revision panel -- "Deliverable D" data layer.

A hand/web-assembled table of forward-multi-year capital-program guidance for
15 large US electric utilities, one row per distinct dated point where the
company stated or revised its plan. See
docs/superpowers/specs/2026-09-04-capex-guidance-signal-design.md s3.2 for the
full column contract and sourcing discipline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED_CSV = Path(__file__).parent / "seed" / "utility_capex_guidance.csv"

DC_BASIS = ("stated", "derived", "qualitative", "none")


def load_capex_guidance(seed: Path = SEED_CSV) -> pd.DataFrame:
    """Load the seed panel. Adds `revision_quality` ('stated' when the CSV
    itself carried a revision figure, 'derived' when computed here from
    consecutive plan levels, 'n/a' for a utility's first vintage) and
    `prior_capex_plan_usd_m` (the immediately-prior plan level, the base a
    revision is measured against -- used by aggregate_revision_series's
    percent form)."""
    df = pd.read_csv(seed)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values(["utility", "report_date"]).reset_index(drop=True)

    stated = df["revision_vs_prior_usd_m"].notna()
    prior_plan = df.groupby("utility")["capex_plan_usd_m"].shift(1)
    derived_ok = prior_plan.notna() & df["capex_plan_usd_m"].notna()
    derived_fill = df["capex_plan_usd_m"] - prior_plan

    df["prior_capex_plan_usd_m"] = prior_plan
    df["revision_vs_prior_usd_m"] = df["revision_vs_prior_usd_m"].where(
        stated, derived_fill.where(derived_ok))
    df["revision_quality"] = np.select(
        [stated, ~stated & derived_ok], ["stated", "derived"], default="n/a")
    return df
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/data/test_utility_capex_guidance.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add grid_resilience/data/seed/utility_capex_guidance.csv grid_resilience/data/utility_capex_guidance.py tests/data/test_utility_capex_guidance.py
git commit -m "feat(capex-guidance): seed CSV schema + point-in-time loader"
```

---

### Task 3: Populate the real seed panel (research task)

**Files:**
- Modify: `grid_resilience/data/seed/utility_capex_guidance.csv` (replace the 4 fixture rows with
  the real panel; the fixture rows move into a small inline fixture string in the test file so
  Task 2's tests keep passing without depending on the live file's exact contents)
- Modify: `tests/data/test_utility_capex_guidance.py` (point the fixture at an inline CSV string
  instead of reading `udg.SEED_CSV`)

**Interfaces:**
- Consumes: `load_capex_guidance`, nothing new produced — this is a data-only task, gated by
  Task 2's loader and tested by Task 4's `feasibility_summary`.

- [ ] **Step 1: Decouple Task 2's tests from the live seed file**

Replace the `fixture_csv` fixture in `tests/data/test_utility_capex_guidance.py` with an inline
string so populating the real panel can't break it:

```python
_FIXTURE_CSV = """utility,report_date,plan_start_year,plan_end_year,capex_plan_usd_m,revision_vs_prior_usd_m,dc_attributed_usd_m,dc_basis,source_type,source_detail,confidence
FIXTURE,2023-01-01,2023,2027,10000,,,none,earnings_call,fixture row - first vintage no prior,H
FIXTURE,2023-07-01,2023,2027,11000,,,none,earnings_call,fixture row - revision derived not stated,H
FIXTURE,2024-01-01,2024,2028,13000,2000,1500,stated,earnings_call,fixture row - stated revision + stated DC $,H
FIXTURE,2024-07-01,2024,2028,13500,500,,qualitative,earnings_call,fixture row - stated revision qualitative DC mention,M
"""


@pytest.fixture
def fixture_csv(tmp_path):
    dst = tmp_path / "fixture.csv"
    dst.write_text(_FIXTURE_CSV)
    return dst
```

Run: `pytest tests/data/test_utility_capex_guidance.py -v` — still 5 passed (now independent of
the live seed file's contents).

- [ ] **Step 2: Research and populate the real panel**

For each of the 15 utilities in `config.DC_GUIDANCE_UNIVERSE`
(D, AEP, NEE, SO, ETR, XEL, DUK, PCG, EIX, PPL, FE, AEE, WEC, CMS, DTE), research its stated
forward capital-program guidance at each distinct point 2021→2026 it was given (annual-update
earnings call, analyst day, or 10-K capital-program table) via web search, and add one row per
vintage to `grid_resilience/data/seed/utility_capex_guidance.csv` following Task 2's schema.
DC-heaviest names first (D, AEP, NEE, SO, DUK, ETR, XEL, PCG) since those are most likely to
clear the feasibility bar and most likely to explicitly attribute revisions to data centers /
large load. For each row:

- `report_date`: the date the figure became public.
- `capex_plan_usd_m`: the total $ (millions) over the stated horizon.
- `revision_vs_prior_usd_m`: fill in directly ONLY if the company itself stated a delta
  ("raised the plan by $X billion"); otherwise leave blank and let `load_capex_guidance` derive
  it from consecutive `capex_plan_usd_m` levels.
- `dc_attributed_usd_m` / `dc_basis`: fill in the $ figure only when the company gave one
  (`stated`) or one can be reasonably derived from disclosed incremental MW × a stated or
  typical $/MW (`derived`); otherwise `qualitative` (mentioned, no $) or `none`.
- `source_type`, `source_detail`, `confidence`: every row gets all three — never leave
  `confidence` blank.

- [ ] **Step 3: Check feasibility on the populated panel**

Run:
```bash
python -c "
from grid_resilience.data import utility_capex_guidance as udg
from grid_equipment_basket import config
df = udg.load_capex_guidance()
feas = udg.feasibility_summary(df, window=config.DC_GUIDANCE_PRIMARY_WINDOW)
print('n_usable:', feas['n_usable'], '/', feas['n_total'])
for u, v in feas['per_utility'].items():
    print(f\"  {u:<6} usable={v['usable']} n_revisions={v['n_revisions']}\")
"
```
Expected: `n_usable >= 8`. If it's below 8, keep researching the DC-heaviest names not yet
usable before moving on — this is the feasibility kill from spec §3.3, worth getting right before
building the rest of the signal on top of it. If after a genuine attempt across all 15 it's still
below 8, that's a real result: proceed anyway (the report function's own fallback logic, built in
Task 12, handles it), and say so plainly in the eventual results doc (Task 17).

- [ ] **Step 4: Commit**

```bash
git add grid_resilience/data/seed/utility_capex_guidance.csv tests/data/test_utility_capex_guidance.py
git commit -m "data(capex-guidance): populate the 15-utility capex-guidance panel"
```

---

### Task 4: `feasibility_summary`

**Files:**
- Modify: `grid_resilience/data/utility_capex_guidance.py`
- Test: `tests/data/test_utility_capex_guidance.py`

**Interfaces:**
- Consumes: `load_capex_guidance`'s output shape.
- Produces: `feasibility_summary(df, window=("2023-01-01","2026-08-31")) -> dict` with keys
  `per_utility` (`{ticker: {has_plan, n_revisions, usable}}`), `n_usable`, `n_total`. Used by
  Task 3 Step 3 and Task 12.

- [ ] **Step 1: Write the failing test**

```python
def test_feasibility_summary_counts_usable_utilities():
    df = pd.DataFrame({
        "utility":                ["AAA", "AAA", "BBB", "CCC"],
        "report_date": pd.to_datetime(["2022-01-01", "2023-06-01", "2023-01-01", "2020-01-01"]),
        "capex_plan_usd_m":       [1000, 1200, 500, 300],
        "revision_vs_prior_usd_m": [np.nan, 200, np.nan, np.nan],
    })
    feas = udg.feasibility_summary(df, window=("2023-01-01", "2026-08-31"))
    # AAA: has a plan and one in-window revision (2023-06-01) -> usable
    assert feas["per_utility"]["AAA"]["usable"] is True
    # BBB: has a plan but no revision anywhere -> not usable
    assert feas["per_utility"]["BBB"]["usable"] is False
    # CCC: has a plan, and a "revision" but it's before the window (2020) and
    # blank anyway -> not usable
    assert feas["per_utility"]["CCC"]["usable"] is False
    assert feas["n_usable"] == 1
    assert feas["n_total"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_utility_capex_guidance.py::test_feasibility_summary_counts_usable_utilities -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'feasibility_summary'`

- [ ] **Step 3: Write the implementation**

```python
def feasibility_summary(df: pd.DataFrame,
                        window: tuple[str, str] = ("2023-01-01", "2026-08-31")) -> dict:
    """Per-utility usable-plan / in-window-revision counts (spec s3.3). A
    utility is "usable" if it has >=1 row with a non-null `capex_plan_usd_m`
    AND >=1 row with a non-null `revision_vs_prior_usd_m` whose `report_date`
    falls inside `window`."""
    ws, we = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    per_utility: dict[str, dict] = {}
    for u, g in df.groupby("utility"):
        has_plan = bool(g["capex_plan_usd_m"].notna().any())
        in_window = g[(g["report_date"] >= ws) & (g["report_date"] <= we)]
        n_revisions = int(in_window["revision_vs_prior_usd_m"].notna().sum())
        per_utility[u] = {"has_plan": has_plan, "n_revisions": n_revisions,
                          "usable": bool(has_plan and n_revisions >= 1)}
    n_usable = sum(1 for v in per_utility.values() if v["usable"])
    return {"per_utility": per_utility, "n_usable": n_usable, "n_total": len(per_utility)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_utility_capex_guidance.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add grid_resilience/data/utility_capex_guidance.py tests/data/test_utility_capex_guidance.py
git commit -m "feat(capex-guidance): feasibility_summary"
```

---

### Task 5: `panel_asof`

**Files:**
- Modify: `grid_resilience/data/utility_capex_guidance.py`
- Test: `tests/data/test_utility_capex_guidance.py`

**Interfaces:**
- Produces: `panel_asof(df, asof) -> pd.DataFrame` — most-recent row per utility with
  `report_date <= asof`. Used by Task 12 (not strictly required by the aggregate-series path but
  kept as a reusable point-in-time-slice primitive per spec §8).

- [ ] **Step 1: Write the failing test**

```python
def test_panel_asof_returns_latest_row_per_utility_not_after_asof():
    df = pd.DataFrame({
        "utility":     ["AAA", "AAA", "BBB"],
        "report_date": pd.to_datetime(["2022-01-01", "2023-01-01", "2023-06-01"]),
        "capex_plan_usd_m": [1000, 1200, 500],
    })
    out = udg.panel_asof(df, "2023-03-01")
    # AAA's 2023-01-01 row is the latest not after asof; BBB has none yet
    assert set(out["utility"]) == {"AAA"}
    assert out.iloc[0]["report_date"] == pd.Timestamp("2023-01-01")


def test_panel_asof_empty_when_nothing_known_yet():
    df = pd.DataFrame({
        "utility": ["AAA"], "report_date": pd.to_datetime(["2023-01-01"]),
        "capex_plan_usd_m": [1000],
    })
    out = udg.panel_asof(df, "2020-01-01")
    assert out.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_utility_capex_guidance.py -k panel_asof -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write the implementation**

```python
def panel_asof(df: pd.DataFrame, asof) -> pd.DataFrame:
    """Most-recent row per utility with `report_date <= asof` (point-in-time
    slice). A utility with no row on or before `asof` is simply absent."""
    asof_ts = pd.Timestamp(asof)
    known = df[df["report_date"] <= asof_ts]
    if known.empty:
        return known.iloc[0:0]
    idx = known.groupby("utility")["report_date"].idxmax()
    return known.loc[idx].sort_values("utility").reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_utility_capex_guidance.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add grid_resilience/data/utility_capex_guidance.py tests/data/test_utility_capex_guidance.py
git commit -m "feat(capex-guidance): panel_asof point-in-time slice"
```

---

### Task 6: `impute_dc_attributed`

**Files:**
- Modify: `grid_resilience/data/utility_capex_guidance.py`
- Test: `tests/data/test_utility_capex_guidance.py`

**Interfaces:**
- Produces: `impute_dc_attributed(df, *, max_iter=5, tol=0.005, imputed_weight=0.5) ->
  pd.DataFrame` — the input df plus `dc_attributed_usd_m_filled` (float) and `dc_imputed`
  (bool) columns. Used by Task 7 / Task 12 (the `dc_filled_usd`/`dc_filled_pct` series).

- [ ] **Step 1: Write the failing tests**

```python
def test_impute_dc_attributed_keeps_stated_rows_unchanged():
    df = pd.DataFrame({
        "utility": ["AAA"], "report_date": pd.to_datetime(["2023-01-01"]),
        "revision_vs_prior_usd_m": [1000.0], "dc_attributed_usd_m": [500.0],
        "dc_basis": ["stated"],
    })
    out = udg.impute_dc_attributed(df)
    assert out.iloc[0]["dc_attributed_usd_m_filled"] == pytest.approx(500.0)
    assert out.iloc[0]["dc_imputed"] == False


def test_impute_dc_attributed_fills_qualitative_row_from_prior_observed_ratio():
    df = pd.DataFrame({
        "utility":     ["AAA", "BBB"],
        "report_date": pd.to_datetime(["2023-01-01", "2023-06-01"]),
        "revision_vs_prior_usd_m": [1000.0, 800.0],
        "dc_attributed_usd_m":     [500.0, np.nan],
        "dc_basis":                ["stated", "qualitative"],
    })
    out = udg.impute_dc_attributed(df)
    bbb = out[out["utility"] == "BBB"].iloc[0]
    # ratio from AAA alone = 500/1000 = 0.5; BBB's revision 800 * 0.5 = 400
    assert bbb["dc_attributed_usd_m_filled"] == pytest.approx(400.0)
    assert bbb["dc_imputed"] == True


def test_impute_dc_attributed_is_point_in_time_safe():
    base_rows = [
        {"utility": "AAA", "report_date": "2023-01-01", "revision_vs_prior_usd_m": 1000.0,
         "dc_attributed_usd_m": 500.0, "dc_basis": "stated"},
        {"utility": "BBB", "report_date": "2023-06-01", "revision_vs_prior_usd_m": 800.0,
         "dc_attributed_usd_m": np.nan, "dc_basis": "qualitative"},
    ]
    later_row = {"utility": "CCC", "report_date": "2024-01-01", "revision_vs_prior_usd_m": 2000.0,
                "dc_attributed_usd_m": 100.0, "dc_basis": "stated"}

    df_without = pd.DataFrame(base_rows)
    df_without["report_date"] = pd.to_datetime(df_without["report_date"])
    out_without = udg.impute_dc_attributed(df_without)

    df_with = pd.DataFrame(base_rows + [later_row])
    df_with["report_date"] = pd.to_datetime(df_with["report_date"])
    out_with = udg.impute_dc_attributed(df_with)

    v_without = out_without.loc[out_without["utility"] == "BBB", "dc_attributed_usd_m_filled"].iloc[0]
    v_with = out_with.loc[out_with["utility"] == "BBB", "dc_attributed_usd_m_filled"].iloc[0]
    # a LATER observation (CCC, 2024) must not change BBB's already-computed
    # (2023-06) imputed value -- that would be a future-information leak.
    assert v_without == pytest.approx(v_with)
    assert v_without == pytest.approx(400.0)


def test_impute_dc_attributed_before_any_observation_stays_nan():
    rows = [
        {"utility": "AAA", "report_date": "2021-01-01", "revision_vs_prior_usd_m": 500.0,
         "dc_attributed_usd_m": np.nan, "dc_basis": "none"},
        {"utility": "AAA", "report_date": "2023-01-01", "revision_vs_prior_usd_m": 1000.0,
         "dc_attributed_usd_m": 600.0, "dc_basis": "stated"},
    ]
    df = pd.DataFrame(rows)
    df["report_date"] = pd.to_datetime(df["report_date"])
    out = udg.impute_dc_attributed(df)
    first = out.iloc[0]
    assert pd.isna(first["dc_attributed_usd_m_filled"])
    assert first["dc_imputed"] == False


def test_impute_dc_attributed_converges_within_max_iter():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        stated = i % 3 == 0
        rows.append({
            "utility": f"U{i % 5}",
            "report_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=30 * i),
            "revision_vs_prior_usd_m": float(rng.uniform(100, 1000)),
            "dc_attributed_usd_m": float(rng.uniform(50, 400)) if stated else np.nan,
            "dc_basis": "stated" if stated else "qualitative",
        })
    df = pd.DataFrame(rows)
    out5 = udg.impute_dc_attributed(df, max_iter=5)
    out20 = udg.impute_dc_attributed(df, max_iter=20)
    assert out5["dc_attributed_usd_m_filled"].notna().sum() > 0
    pd.testing.assert_series_equal(
        out5["dc_attributed_usd_m_filled"].fillna(-1.0),
        out20["dc_attributed_usd_m_filled"].fillna(-1.0),
        check_exact=False, atol=1.0, rtol=0.02)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_utility_capex_guidance.py -k impute_dc_attributed -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write the implementation**

```python
def impute_dc_attributed(df: pd.DataFrame, *, max_iter: int = 5,
                         tol: float = 0.005, imputed_weight: float = 0.5) -> pd.DataFrame:
    """Point-in-time, EM-style fill of the data-center-attributed dollar figure
    for rows where the utility only qualitatively mentioned data centers (or
    not at all). Spec s4.2.

    For a row with `dc_basis` in {"qualitative", "none"}, the fill is
    `revision_vs_prior_usd_m * ratio(t)`, where `ratio(t)` is the mean of
    `dc_attributed_usd_m / revision_vs_prior_usd_m` over every "stated"/
    "derived" row with `report_date <= t` -- only already-disclosed data as of
    that date, never a later quarter's. After the first pass, `ratio(t)` is
    recomputed including the freshly-imputed rows (down-weighted
    `imputed_weight` vs. a genuinely observed row) and the fill redone, for up
    to `max_iter` rounds or until the total imputed $ changes by less than
    `tol` (relative) between rounds. A row with no stated/derived observation
    anywhere in the panel before its own date keeps `dc_attributed_usd_m_filled`
    NaN -- there is nothing yet to calibrate the ratio from.
    """
    out = df.sort_values("report_date").reset_index(drop=True).copy()
    observed = out["dc_basis"].isin(["stated", "derived"]) & out["dc_attributed_usd_m"].notna()
    has_rev = out["revision_vs_prior_usd_m"].notna() & (out["revision_vs_prior_usd_m"] != 0)
    needs_fill = out["dc_basis"].isin(["qualitative", "none"]) & has_rev

    out["dc_attributed_usd_m_filled"] = np.where(observed, out["dc_attributed_usd_m"], np.nan)
    out["dc_imputed"] = False

    pool_mask = observed & has_rev
    ratio_val = (out["dc_attributed_usd_m"] / out["revision_vs_prior_usd_m"]).where(pool_mask)
    ratio_wt = pd.Series(np.where(pool_mask, 1.0, np.nan), index=out.index)

    prev_total = None
    for _ in range(max_iter):
        weighted_sum = (ratio_val.fillna(0.0) * ratio_wt.fillna(0.0)).cumsum()
        weight_sum = ratio_wt.fillna(0.0).cumsum()
        ratio_asof = weighted_sum / weight_sum.replace(0.0, np.nan)

        fill_val = out["revision_vs_prior_usd_m"] * ratio_asof
        new_filled = out["dc_attributed_usd_m_filled"].where(observed, fill_val.where(needs_fill))
        out["dc_attributed_usd_m_filled"] = new_filled
        out["dc_imputed"] = needs_fill & new_filled.notna()

        total_now = float(np.nansum(new_filled[out["dc_imputed"]]))

        ratio_val = (out["dc_attributed_usd_m_filled"] / out["revision_vs_prior_usd_m"]).where(
            observed | out["dc_imputed"])
        ratio_wt = pd.Series(
            np.where(observed, 1.0, np.where(out["dc_imputed"], imputed_weight, np.nan)),
            index=out.index)

        if prev_total is not None and abs(prev_total) > 1e-9 and \
                abs(total_now - prev_total) / abs(prev_total) < tol:
            break
        prev_total = total_now

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_utility_capex_guidance.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add grid_resilience/data/utility_capex_guidance.py tests/data/test_utility_capex_guidance.py
git commit -m "feat(capex-guidance): point-in-time EM-style DC-attributed imputation"
```

---

### Task 7: `aggregate_revision_series`

**Files:**
- Modify: `grid_resilience/data/utility_capex_guidance.py`
- Test: `tests/data/test_utility_capex_guidance.py`

**Interfaces:**
- Consumes: a df with `report_date`, a value column, optionally a denom column (both from
  `load_capex_guidance` or `impute_dc_attributed`'s output).
- Produces: `aggregate_revision_series(df, *, value_col="revision_vs_prior_usd_m",
  denom_col=None, ttm_quarters=4) -> pd.Series` indexed by the panel's distinct `report_date`
  values (event-dated, not daily). Used by Task 12 to build all six series from spec §4.1/§4.2
  with one function.

- [ ] **Step 1: Write the failing tests**

```python
def test_aggregate_revision_series_usd_is_trailing_sum_within_window():
    df = pd.DataFrame({
        "utility":     ["AAA", "BBB", "AAA"],
        "report_date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-11-01"]),
        "revision_vs_prior_usd_m": [100.0, 50.0, 200.0],
    })
    s = udg.aggregate_revision_series(df, value_col="revision_vs_prior_usd_m", ttm_quarters=4)
    # at 2023-02-01, trailing 4Q (~365 days) window includes both prior events -> 150
    assert s.loc[pd.Timestamp("2023-02-01")] == pytest.approx(150.0)
    # at 2023-11-01, the Jan event (>365 days back is not yet true, ~304 days,
    # still inside the window) plus itself -> all three sum to 350
    assert s.loc[pd.Timestamp("2023-11-01")] == pytest.approx(350.0)


def test_aggregate_revision_series_drops_out_after_ttm_window():
    df = pd.DataFrame({
        "utility":     ["AAA", "AAA"],
        "report_date": pd.to_datetime(["2022-01-01", "2023-06-01"]),
        "revision_vs_prior_usd_m": [100.0, 50.0],
    })
    s = udg.aggregate_revision_series(df, value_col="revision_vs_prior_usd_m", ttm_quarters=4)
    # by 2023-06-01 the 2022-01-01 event is >365 days old -> only the new one counts
    assert s.loc[pd.Timestamp("2023-06-01")] == pytest.approx(50.0)


def test_aggregate_revision_series_pct_is_size_weighted_not_mean_of_percents():
    df = pd.DataFrame({
        "utility":     ["AAA", "BBB"],
        "report_date": pd.to_datetime(["2023-01-01", "2023-01-15"]),
        "revision_vs_prior_usd_m": [100.0, 10.0],
        "prior_capex_plan_usd_m":  [1000.0, 20.0],   # AAA 10%, BBB 50%
    })
    s = udg.aggregate_revision_series(df, value_col="revision_vs_prior_usd_m",
                                      denom_col="prior_capex_plan_usd_m", ttm_quarters=4)
    # size-weighted: (100+10)/(1000+20) = 0.1078..., NOT mean(10%, 50%) = 30%
    assert s.loc[pd.Timestamp("2023-01-15")] == pytest.approx(110.0 / 1020.0)


def test_aggregate_revision_series_drops_rows_with_missing_value():
    df = pd.DataFrame({
        "utility": ["AAA", "BBB"], "report_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        "dc_attributed_usd_m": [np.nan, 40.0],
    })
    s = udg.aggregate_revision_series(df, value_col="dc_attributed_usd_m", ttm_quarters=4)
    assert list(s.index) == [pd.Timestamp("2023-01-02")]
    assert s.iloc[0] == pytest.approx(40.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/test_utility_capex_guidance.py -k aggregate_revision_series -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write the implementation**

```python
def aggregate_revision_series(df: pd.DataFrame, *, value_col: str = "revision_vs_prior_usd_m",
                              denom_col: str | None = None, ttm_quarters: int = 4) -> pd.Series:
    """Trailing-`ttm_quarters`-quarter (~91.3 days/quarter) rolling sum of
    `value_col` across every panel row with a non-null value, evaluated at
    each of the panel's own distinct `report_date` values (event-dated --
    callers broadcast this to a daily index via `guidance_composite`).

    `denom_col=None` -> the $ series (spec s4.1's `agg_revision_ttm_usd`).
    `denom_col="prior_capex_plan_usd_m"` -> the size-weighted percent series
    (spec s4.1's `agg_revision_ttm_pct`): at each event date, the ratio of the
    trailing-window SUM of `value_col` to the trailing-window SUM of
    `denom_col` -- not a mean of each row's own percentage."""
    s = df.dropna(subset=[value_col]).sort_values("report_date")
    dates = pd.DatetimeIndex(s["report_date"])
    values = s[value_col].to_numpy()
    denom_values = s[denom_col].fillna(0.0).to_numpy() if denom_col else None
    window = pd.Timedelta(days=round(91.3 * ttm_quarters))

    event_dates = pd.DatetimeIndex(sorted(dates.unique()))
    num_out: list[float] = []
    den_out: list[float] = []
    for d in event_dates:
        mask = (dates > d - window) & (dates <= d)
        num_out.append(float(values[mask].sum()))
        if denom_col:
            den_out.append(float(denom_values[mask].sum()))

    num = pd.Series(num_out, index=event_dates)
    if not denom_col:
        result = num
    else:
        den = pd.Series(den_out, index=event_dates)
        result = num / den.replace(0.0, np.nan)
    name = value_col + ("_ttm_pct" if denom_col else "_ttm")
    return result.rename(name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_utility_capex_guidance.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add grid_resilience/data/utility_capex_guidance.py tests/data/test_utility_capex_guidance.py
git commit -m "feat(capex-guidance): aggregate_revision_series (usd + size-weighted pct)"
```

---

### Task 8: `guidance_composite` — daily broadcast + trailing z-score

**Files:**
- Create: `grid_equipment_basket/capex_guidance_signal.py`
- Test: `tests/grid_equipment_basket/test_capex_guidance_signal.py`

**Interfaces:**
- Consumes: `ftr_signal.broadcast_daily(available, start, end)`,
  `grid_regime._trailing_zscore(s, window, min_periods, winsor)` (both existing, reused
  unchanged), `config.DC_GUIDANCE_ZSCORE_WINDOW/MINP/WINSOR`.
- Produces: `guidance_composite(events, start, end, *, zscore_window=..., zscore_minp=...,
  winsor=...) -> pd.Series` named `"guidance_composite"`. Used by Tasks 9, 12, 13.

- [ ] **Step 1: Write the failing test**

```python
# tests/grid_equipment_basket/test_capex_guidance_signal.py
import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import capex_guidance_signal as cgs


def test_guidance_composite_empty_events_is_empty_series():
    result = cgs.guidance_composite(pd.Series(dtype=float), "2023-01-01", "2023-06-01")
    assert result.empty


def test_guidance_composite_no_future_leak():
    events = pd.Series([10.0, 50.0],
                       index=[pd.Timestamp("2023-01-01"), pd.Timestamp("2023-04-01")])
    full = cgs.guidance_composite(events, "2022-01-01", "2023-12-31",
                                  zscore_window=300, zscore_minp=5, winsor=3.0)
    truncated_events = events.loc[:"2023-01-01"]
    truncated = cgs.guidance_composite(truncated_events, "2022-01-01", "2023-03-31",
                                       zscore_window=300, zscore_minp=5, winsor=3.0)
    common = truncated.index.intersection(full.index)
    assert len(common) > 0
    pd.testing.assert_series_equal(full.loc[common], truncated.loc[common])


def test_guidance_composite_holds_value_between_events():
    events = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0],
                       index=pd.date_range("2022-01-01", periods=5, freq="90D"))
    result = cgs.guidance_composite(events, "2022-01-01", "2022-12-31",
                                    zscore_window=1000, zscore_minp=1, winsor=3.0)
    # a day between two events holds the earlier one's (already z-scored) value
    d_after_first = events.index[0] + pd.Timedelta(days=5)
    d_of_first = events.index[0]
    assert result.loc[d_after_first] == pytest.approx(result.loc[d_of_first])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'grid_equipment_basket.capex_guidance_signal'`

- [ ] **Step 3: Write the implementation**

```python
# grid_equipment_basket/capex_guidance_signal.py
"""Utility capex-guidance revision signal ("Deliverable D").

Spec: docs/superpowers/specs/2026-09-04-capex-guidance-signal-design.md

Aggregate US electric utilities revise their forward 5-yr capital-program
guidance at each earnings call / 10-K / analyst day; since ~2023 those
revisions are increasingly data-center-driven and increasingly stated as
such. This module turns the hand/web-assembled panel in
`grid_resilience.data.utility_capex_guidance` into (1) a daily z-scored
composite, (2) a one-directional de-risk multiplier and a two-sided scaler
for the grid-equipment basket's exposure, and (3) the pre-registered gate
report testing both as a timing signal and as an exposure scaler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from grid_equipment_basket import config
from grid_equipment_basket.ftr_signal import broadcast_daily
from grid_equipment_basket.grid_regime import _trailing_zscore


def guidance_composite(events: pd.Series, start: str, end: str, *,
                       zscore_window: int = config.DC_GUIDANCE_ZSCORE_WINDOW,
                       zscore_minp: int = config.DC_GUIDANCE_ZSCORE_MINP,
                       winsor: float = config.DC_GUIDANCE_ZSCORE_WINSOR) -> pd.Series:
    """`events` indexed by `report_date` (already point-in-time -- see
    `utility_capex_guidance.aggregate_revision_series`). Forward-filled onto a
    daily calendar index over `[start, end]`, then trailing-z-scored
    (`grid_regime._trailing_zscore`'s rolling window only uses
    past-and-current observations, so this composite carries the same
    no-future-leak contract as `ftr_signal.ftr_composite` /
    `grid_regime.regime_composite`)."""
    if events.empty:
        return pd.Series(dtype=float, name="guidance_composite")
    daily = broadcast_daily(events, start, end)
    return _trailing_zscore(daily, zscore_window, zscore_minp, winsor).rename("guidance_composite")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add grid_equipment_basket/capex_guidance_signal.py tests/grid_equipment_basket/test_capex_guidance_signal.py
git commit -m "feat(capex-guidance): guidance_composite (daily broadcast + trailing z-score)"
```

---

### Task 9: `guidance_derisk_multiplier` + `guidance_scaler`

**Files:**
- Modify: `grid_equipment_basket/capex_guidance_signal.py`
- Test: `tests/grid_equipment_basket/test_capex_guidance_signal.py`

**Interfaces:**
- Consumes: `guidance_composite`'s output shape (a daily z-score `pd.Series`, may contain NaN).
- Produces: `guidance_derisk_multiplier(composite, *, floor_z=..., lo_mult=...) -> pd.Series` and
  `guidance_scaler(composite, *, k=..., lo=..., hi=...) -> pd.Series`, both daily, both feedable
  directly into `overlay.apply_overlay_l2(basket_returns, mult, rf)`. Used by Task 13.

- [ ] **Step 1: Write the failing tests**

```python
def test_guidance_derisk_multiplier_steps_down_below_floor_and_back():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    composite = pd.Series([0.2, -0.6, -0.7, 0.1, np.nan], index=idx)
    mult = cgs.guidance_derisk_multiplier(composite, floor_z=-0.5, lo_mult=0.6)
    assert list(mult) == [1.0, 0.6, 0.6, 1.0, 1.0]


def test_guidance_derisk_multiplier_never_exceeds_one():
    idx = pd.date_range("2023-01-01", periods=3, freq="D")
    composite = pd.Series([5.0, -5.0, 0.0], index=idx)
    mult = cgs.guidance_derisk_multiplier(composite, floor_z=-0.5, lo_mult=0.6)
    assert mult.max() <= 1.0


def test_guidance_scaler_clips_and_defaults_to_one_when_nan():
    idx = pd.date_range("2023-01-01", periods=4, freq="D")
    composite = pd.Series([0.0, 2.0, -2.0, np.nan], index=idx)
    scaled = cgs.guidance_scaler(composite, k=0.35, lo=0.5, hi=1.5)
    assert scaled.iloc[0] == pytest.approx(1.0)
    assert scaled.iloc[1] == pytest.approx(1.5)   # 1+0.35*2=1.7 -> clipped to hi
    assert scaled.iloc[2] == pytest.approx(0.5)   # 1-0.7=0.3 -> clipped to lo
    assert scaled.iloc[3] == pytest.approx(1.0)   # NaN -> default
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -k "derisk or scaler" -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write the implementation**

Append to `grid_equipment_basket/capex_guidance_signal.py`:

```python
def guidance_derisk_multiplier(composite: pd.Series, *,
                               floor_z: float = config.DC_GUIDANCE_DERISK_FLOOR_Z,
                               lo_mult: float = config.DC_GUIDANCE_DERISK_LO_MULT) -> pd.Series:
    """`{lo_mult, 1.0}` -- `lo_mult` on any day the composite z-score is below
    `floor_z` (revision-flow decelerating/reversing), back to 1.0 once it
    clears. NaN days (not-yet-warm) default to 1.0. Never exceeds 1.0. Same
    one-directional shape as `capex_signal.capex_derisk_multiplier` (spec s6.1)."""
    lo = composite < floor_z
    return pd.Series(np.where(lo.fillna(False), lo_mult, 1.0),
                     index=composite.index).rename("guidance_derisk_mult")


def guidance_scaler(composite: pd.Series, *,
                    k: float = config.DC_GUIDANCE_SCALER_K,
                    lo: float = config.DC_GUIDANCE_SCALER_LO,
                    hi: float = config.DC_GUIDANCE_SCALER_HI) -> pd.Series:
    """`clip(1 + k*z, lo, hi)` -- two-sided: leans in when the composite is
    positive (revision-flow accelerating), leans out when negative. NaN days
    default to 1.0. Same shape as `grid_regime`'s continuous-mode multiplier
    (spec s6.2)."""
    z = composite.fillna(0.0)
    raw = (1.0 + k * z).clip(lower=lo, upper=hi)
    return raw.where(composite.notna(), 1.0).rename("guidance_scaler")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add grid_equipment_basket/capex_guidance_signal.py tests/grid_equipment_basket/test_capex_guidance_signal.py
git commit -m "feat(capex-guidance): one-directional de-risk multiplier + two-sided scaler"
```

---

### Task 10: `_hac_ols` + `_rank_ic` stats helpers

**Files:**
- Modify: `grid_equipment_basket/capex_guidance_signal.py`
- Test: `tests/grid_equipment_basket/test_capex_guidance_signal.py`

**Interfaces:**
- Consumes: `statsmodels.api`, `scipy.stats.spearmanr` (both already dependencies elsewhere in
  this repo).
- Produces: `_hac_ols(y, X, *, lag) -> dict` (`{"coef": {col: float}, "t": {col: float}, "n":
  int}`) and `_rank_ic(signal, fwd_return, *, lag) -> dict` (`{"ic": float, "t": float, "n":
  int}`). Used by Task 12.

- [ ] **Step 1: Write the failing tests**

```python
def test_hac_ols_recovers_a_strong_known_relationship():
    rng = np.random.default_rng(0)
    n = 60
    x = pd.Series(rng.normal(size=n), index=pd.date_range("2020-01-31", periods=n, freq="ME"))
    y = 2.0 * x + rng.normal(scale=0.1, size=n)
    out = cgs._hac_ols(y, pd.DataFrame({"x": x}), lag=1)
    assert out["coef"]["x"] == pytest.approx(2.0, abs=0.2)
    assert abs(out["t"]["x"]) >= 2.0


def test_hac_ols_no_relationship_gives_small_t():
    rng = np.random.default_rng(1)
    n = 60
    x = pd.Series(rng.normal(size=n), index=pd.date_range("2020-01-31", periods=n, freq="ME"))
    y = pd.Series(rng.normal(size=n), index=x.index)
    out = cgs._hac_ols(y, pd.DataFrame({"x": x}), lag=1)
    assert abs(out["t"]["x"]) < 3.0   # not a hard bound, just "not obviously significant"


def test_hac_ols_too_few_observations_returns_nan_not_a_crash():
    x = pd.Series([1.0, 2.0], index=pd.date_range("2020-01-31", periods=2, freq="ME"))
    y = pd.Series([1.0, 2.0], index=x.index)
    out = cgs._hac_ols(y, pd.DataFrame({"x": x}), lag=1)
    assert np.isnan(out["t"]["x"])


def test_rank_ic_positive_when_signal_leads_forward_return():
    idx = pd.date_range("2020-01-31", periods=40, freq="ME")
    rng = np.random.default_rng(2)
    signal = pd.Series(rng.normal(size=40), index=idx)
    fwd = signal + rng.normal(scale=0.3, size=40)
    out = cgs._rank_ic(signal, pd.Series(fwd.values, index=idx), lag=1)
    assert out["ic"] > 0.5
    assert out["t"] >= 2.0


def test_rank_ic_too_few_pairs_returns_nan():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    out = cgs._rank_ic(pd.Series([1.0, 2.0, 3.0], index=idx),
                       pd.Series([1.0, np.nan, np.nan], index=idx), lag=1)
    assert np.isnan(out["ic"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -k "hac_ols or rank_ic" -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write the implementation**

Append to `grid_equipment_basket/capex_guidance_signal.py`:

```python
def _hac_ols(y: pd.Series, X: pd.DataFrame, *, lag: int) -> dict:
    """OLS of `y` on `X` (a constant is added automatically) with Newey-West
    HAC standard errors at `lag`. Used both for the rank-IC t-stat (on ranked
    series) and the multi-control regression (on raw series) -- spec s5.1/s5.2.
    Returns `{"coef": {col: value}, "t": {col: value}, "n": n_obs}`; a NaN dict
    when there aren't enough observations to fit, rather than raising."""
    import statsmodels.api as sm

    frame = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    cols = list(X.columns)
    if len(frame) < len(cols) + 3:
        return {"coef": {c: np.nan for c in cols}, "t": {c: np.nan for c in cols}, "n": len(frame)}
    yy = frame["__y__"]
    XX = sm.add_constant(frame[cols])
    fit = sm.OLS(yy, XX).fit(cov_type="HAC", cov_kwds={"maxlags": max(int(lag), 1)})
    return {"coef": {c: float(fit.params[c]) for c in cols},
           "t": {c: float(fit.tvalues[c]) for c in cols}, "n": int(len(frame))}


def _rank_ic(signal: pd.Series, fwd_return: pd.Series, *, lag: int) -> dict:
    """Spearman rank-IC (point estimate via `scipy.stats.spearmanr`) plus a
    serial-correlation-robust t-stat: `_hac_ols` of `rank(fwd_return)` on
    `rank(signal)` with Newey-West lag=`lag` (the overlapping-forward-window
    horizons h=3/6 have serially correlated residuals month to month, which a
    plain Spearman significance test would understate). Spec s5.1."""
    from scipy.stats import spearmanr

    pair = pd.concat([signal.rename("s"), fwd_return.rename("r")], axis=1).dropna()
    if len(pair) < 5:
        return {"ic": np.nan, "t": np.nan, "n": len(pair)}
    ic, _ = spearmanr(pair["s"], pair["r"])
    ranks = pair.rank()
    hac = _hac_ols(ranks["r"], ranks[["s"]].rename(columns={"s": "signal"}), lag=lag)
    return {"ic": float(ic), "t": hac["t"]["signal"], "n": len(pair)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add grid_equipment_basket/capex_guidance_signal.py tests/grid_equipment_basket/test_capex_guidance_signal.py
git commit -m "feat(capex-guidance): HAC-OLS + rank-IC stats helpers"
```

---

### Task 11: `_monthly_nav` + `_forward_return`

**Files:**
- Modify: `grid_equipment_basket/capex_guidance_signal.py`
- Test: `tests/grid_equipment_basket/test_capex_guidance_signal.py`

**Interfaces:**
- Produces: `_monthly_nav(daily_ret) -> pd.Series` (month-end NAV, base 1.0) and
  `_forward_return(monthly_nav, h) -> pd.Series` (realized return over the next `h` months,
  NaN for the final `h` month-ends). Used by Task 12/13.

- [ ] **Step 1: Write the failing test**

```python
def test_forward_return_computes_next_h_month_realized_return():
    idx = pd.period_range("2023-01", periods=4, freq="M").to_timestamp("M")
    nav = pd.Series([1.0, 1.1, 1.21, 1.331], index=idx)
    fwd1 = cgs._forward_return(nav, 1)
    assert fwd1.iloc[0] == pytest.approx(0.10)
    assert pd.isna(fwd1.iloc[-1])
    fwd3 = cgs._forward_return(nav, 3)
    assert fwd3.iloc[0] == pytest.approx(0.331)
    assert fwd3.iloc[1:].isna().all()


def test_monthly_nav_compounds_daily_returns_to_month_end():
    idx = pd.date_range("2023-01-01", "2023-02-28", freq="D")
    ret = pd.Series(0.0, index=idx)
    ret.loc["2023-01-15"] = 0.10
    ret.loc["2023-02-10"] = 0.05
    nav = cgs._monthly_nav(ret)
    assert nav.loc["2023-01-31"] == pytest.approx(1.10)
    assert nav.loc["2023-02-28"] == pytest.approx(1.10 * 1.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -k "monthly_nav or forward_return" -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write the implementation**

Append to `grid_equipment_basket/capex_guidance_signal.py`:

```python
def _monthly_nav(daily_ret: pd.Series) -> pd.Series:
    """Month-end NAV level (base 1.0) from a daily simple-return series."""
    nav = (1.0 + daily_ret.fillna(0.0)).cumprod()
    return nav.resample("ME").last()


def _forward_return(monthly_nav: pd.Series, h: int) -> pd.Series:
    """At each month-end, the realized return over the NEXT `h` months
    (`nav[t+h] / nav[t] - 1`); NaN for the trailing `h` month-ends where the
    future NAV isn't known yet."""
    return (monthly_nav.shift(-h) / monthly_nav - 1.0).rename(f"fwd_{h}m")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add grid_equipment_basket/capex_guidance_signal.py tests/grid_equipment_basket/test_capex_guidance_signal.py
git commit -m "feat(capex-guidance): monthly NAV + forward-return helpers"
```

---

### Task 12: `feasibility_gate` + `timing_report` (Test 1 orchestration)

**Files:**
- Modify: `grid_equipment_basket/capex_guidance_signal.py`
- Test: `tests/grid_equipment_basket/test_capex_guidance_signal.py`

**Interfaces:**
- Consumes: `utility_capex_guidance.feasibility_summary`, `_hac_ols`, `_rank_ic`, `_monthly_nav`,
  `_forward_return`, `config.DC_GUIDANCE_MIN_UTILITIES/FALLBACK/RANK_IC_MIN_T`.
- Produces: `feasibility_gate(panel_df, primary_window) -> tuple[pd.DataFrame | None, dict, str]`
  (`universe_used` one of `"full"/"fallback"/"not_testable"`) and `timing_report(composites,
  basket_ret, controls, *, primary, holdout, horizons=(1,3,6)) -> dict` (`{series_name: {horizon:
  {rank_ic_primary, rank_ic_holdout, control_without_hyperscaler, control_with_hyperscaler,
  passed}}}`). Both used by Task 13's top-level `guidance_signal_report`.

- [ ] **Step 1: Write the failing tests**

```python
def _panel(rows):
    df = pd.DataFrame(rows)
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def test_feasibility_gate_uses_full_panel_when_enough_usable(monkeypatch):
    monkeypatch.setattr(cgs.config, "DC_GUIDANCE_MIN_UTILITIES", 1)
    df = _panel([{"utility": "D", "report_date": "2023-06-01",
                  "capex_plan_usd_m": 1000.0, "revision_vs_prior_usd_m": 50.0}])
    panel, feas, used = cgs.feasibility_gate(df, ("2023-01-01", "2026-08-31"))
    assert used == "full"
    assert feas["n_usable"] == 1


def test_feasibility_gate_falls_back_when_not_enough_usable():
    rows = [{"utility": "D", "report_date": "2023-06-01",
            "capex_plan_usd_m": 1000.0, "revision_vs_prior_usd_m": 50.0},
           {"utility": "AEP", "report_date": "2023-06-01",
            "capex_plan_usd_m": 900.0, "revision_vs_prior_usd_m": 40.0},
           {"utility": "ZZZ", "report_date": "1999-01-01",
            "capex_plan_usd_m": 10.0, "revision_vs_prior_usd_m": np.nan}]
    df = _panel(rows)
    panel, feas, used = cgs.feasibility_gate(df, ("2023-01-01", "2026-08-31"))
    assert used == "fallback"
    assert set(panel["utility"]) == {"D", "AEP"}


def test_feasibility_gate_not_testable_when_fallback_also_empty():
    df = _panel([{"utility": "ZZZ", "report_date": "1999-01-01",
                 "capex_plan_usd_m": 10.0, "revision_vs_prior_usd_m": np.nan}])
    panel, feas, used = cgs.feasibility_gate(df, ("2023-01-01", "2026-08-31"))
    assert used == "not_testable"
    assert panel is None


def test_timing_report_combines_rank_ic_and_control_regression_pass_conditions(monkeypatch):
    idx = pd.date_range("2023-01-31", periods=12, freq="ME")
    composite = pd.Series(np.arange(12, dtype=float), index=idx)
    basket_ret = pd.Series(0.001, index=pd.date_range("2023-01-01", "2023-12-31", freq="D"))
    controls = pd.DataFrame({"d10y": 0.0, "smh": 0.0}, index=idx)

    monkeypatch.setattr(cgs, "_rank_ic", lambda signal, fwd, lag: {"ic": 0.9, "t": 5.0, "n": 10})
    monkeypatch.setattr(cgs, "_hac_ols",
                        lambda y, X, lag: {"coef": {c: 0.0 for c in X.columns},
                                           "t": {c: 5.0 for c in X.columns}, "n": 10})
    rep = cgs.timing_report({"strong": composite}, basket_ret, controls,
                            primary=("2023-01-01", "2023-12-31"),
                            holdout=("2023-10-01", "2023-12-31"), horizons=(1,))
    assert rep["strong"][1]["passed"] is True

    monkeypatch.setattr(cgs, "_rank_ic", lambda signal, fwd, lag: {"ic": 0.0, "t": 0.5, "n": 10})
    rep2 = cgs.timing_report({"strong": composite}, basket_ret, controls,
                             primary=("2023-01-01", "2023-12-31"),
                             holdout=("2023-10-01", "2023-12-31"), horizons=(1,))
    assert rep2["strong"][1]["passed"] is False


def test_timing_report_adds_hyperscaler_control_only_when_column_present():
    idx = pd.date_range("2023-01-31", periods=6, freq="ME")
    composite = pd.Series(np.arange(6, dtype=float), index=idx)
    basket_ret = pd.Series(0.001, index=pd.date_range("2023-01-01", "2023-06-30", freq="D"))
    controls_without = pd.DataFrame({"d10y": 0.0, "smh": 0.0}, index=idx)
    rep = cgs.timing_report({"c": composite}, basket_ret, controls_without,
                            primary=("2023-01-01", "2023-06-30"),
                            holdout=("2023-05-01", "2023-06-30"), horizons=(1,))
    assert rep["c"][1]["control_with_hyperscaler"]["n"] == 0

    controls_with = controls_without.assign(bigfour=1.0)
    rep_w = cgs.timing_report({"c": composite}, basket_ret, controls_with,
                              primary=("2023-01-01", "2023-06-30"),
                              holdout=("2023-05-01", "2023-06-30"), horizons=(1,))
    assert "bigfour" in rep_w["c"][1]["control_with_hyperscaler"]["t"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -k "feasibility_gate or timing_report" -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write the implementation**

Append to `grid_equipment_basket/capex_guidance_signal.py`:

```python
_HORIZONS: tuple[int, ...] = (1, 3, 6)


def feasibility_gate(panel_df: pd.DataFrame,
                     primary_window: tuple[str, str] = config.DC_GUIDANCE_PRIMARY_WINDOW
                     ) -> tuple[pd.DataFrame | None, dict, str]:
    """Spec s3.3: <8 usable utilities in `primary_window` -> fall back to
    `config.DC_GUIDANCE_FALLBACK`; the fallback itself unusable (0 usable) ->
    `(None, feas, "not_testable")` so the caller stops before backtesting on
    data too thin to trust. Returns `(panel_to_use, feasibility_dict,
    universe_used)` with `universe_used` one of `"full"/"fallback"/"not_testable"`."""
    from grid_resilience.data import utility_capex_guidance as udg

    feas = udg.feasibility_summary(panel_df, window=primary_window)
    if feas["n_usable"] >= config.DC_GUIDANCE_MIN_UTILITIES:
        return panel_df, feas, "full"

    fallback = panel_df[panel_df["utility"].isin(config.DC_GUIDANCE_FALLBACK)]
    feas_fb = udg.feasibility_summary(fallback, window=primary_window)
    if feas_fb["n_usable"] == 0:
        return None, feas_fb, "not_testable"
    return fallback, feas_fb, "fallback"


def timing_report(composites: dict[str, pd.Series], basket_ret: pd.Series,
                  controls: pd.DataFrame, *,
                  primary: tuple[str, str] = config.DC_GUIDANCE_PRIMARY_WINDOW,
                  holdout: tuple[str, str] = config.DC_GUIDANCE_HOLDOUT_WINDOW,
                  horizons: tuple[int, ...] = _HORIZONS) -> dict:
    """Spec s5: for each named composite series and each forward horizon, the
    rank-IC (primary + holdout windows, s5.1) and the with/without-hyperscaler
    control regression (primary window, s5.2), plus the combined pass/fail
    (s5.4: BOTH the primary-window rank-IC and the without-hyperscaler control
    must clear `|t| >= config.DC_GUIDANCE_RANK_IC_MIN_T`).

    `controls` is a monthly-indexed DataFrame with columns `d10y`, `smh`, and
    optionally `bigfour`; the without-hyperscaler regression uses `d10y` +
    `smh` only, the with-hyperscaler regression adds `bigfour` when present."""
    basket_nav = _monthly_nav(basket_ret)
    out: dict = {}
    for name, comp in composites.items():
        comp_monthly = comp.resample("ME").last()
        out[name] = {}
        for h in horizons:
            fwd = _forward_return(basket_nav, h)
            rank_ic_primary = _rank_ic(comp_monthly.loc[primary[0]:primary[1]],
                                       fwd.loc[primary[0]:primary[1]], lag=h)
            rank_ic_holdout = _rank_ic(comp_monthly.loc[holdout[0]:holdout[1]],
                                       fwd.loc[holdout[0]:holdout[1]], lag=h)

            X = pd.DataFrame({"signal": comp_monthly, "d10y": controls.get("d10y"),
                              "smh": controls.get("smh")}).loc[primary[0]:primary[1]]
            fwd_primary = fwd.loc[primary[0]:primary[1]]
            ctrl_wo = _hac_ols(fwd_primary, X, lag=h)
            if "bigfour" in controls.columns:
                X_w = X.assign(bigfour=controls["bigfour"].loc[primary[0]:primary[1]])
                ctrl_w = _hac_ols(fwd_primary, X_w, lag=h)
            else:
                ctrl_w = {"coef": {}, "t": {}, "n": 0}

            sig_t = ctrl_wo["t"].get("signal", np.nan)
            passed = (abs(rank_ic_primary["t"]) >= config.DC_GUIDANCE_RANK_IC_MIN_T
                     and abs(sig_t) >= config.DC_GUIDANCE_RANK_IC_MIN_T)
            out[name][h] = {
                "rank_ic_primary": rank_ic_primary, "rank_ic_holdout": rank_ic_holdout,
                "control_without_hyperscaler": ctrl_wo, "control_with_hyperscaler": ctrl_w,
                "passed": bool(passed),
            }
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add grid_equipment_basket/capex_guidance_signal.py tests/grid_equipment_basket/test_capex_guidance_signal.py
git commit -m "feat(capex-guidance): feasibility_gate + timing_report (Test 1 orchestration)"
```

---

### Task 13: `derisk_scaler_report` + top-level `guidance_signal_report`

**Files:**
- Modify: `grid_equipment_basket/capex_guidance_signal.py`
- Test: `tests/grid_equipment_basket/test_capex_guidance_signal.py`

**Interfaces:**
- Consumes: `overlay.apply_overlay`, `overlay.apply_overlay_l2`, `grid_regime.gate_check`,
  `grid_regime.final_verdict`, `backtest.compute_metrics`, `backtest.calendar_year_returns`,
  `basket.simulate_basket`, `capex_signal.fetch_bigfour_capex`, plus every function from Tasks
  8–12.
- Produces: `derisk_scaler_report(ret, lvl, composite, *, primary, prior) -> dict`,
  `lead_lag_table(signal_monthly, fwd_return_monthly, ks=range(-6,7)) -> pd.Series` (spec s5.3's
  continuity table), and `guidance_signal_report(price_fn=None, panel_df=None, bigfour_fn=None,
  primary=..., prior=..., holdout=...) -> dict` (now includes a `"continuity_lead_lag"` key, one
  `lead_lag_table` per series) — the full spec deliverable. Used by Task 14's `guidance_table`
  and Task 15's CLI wiring.

- [ ] **Step 1: Write the failing tests**

```python
def test_derisk_scaler_report_builds_full_parameter_grids():
    idx = pd.bdate_range("2021-01-01", "2026-08-31")
    rng = np.random.default_rng(4)
    ret = pd.Series(rng.normal(0.0004, 0.02, size=len(idx)), index=idx)
    lvl = (1.0 + ret).cumprod()
    composite = pd.Series(rng.normal(size=len(idx)), index=idx)

    rep = cgs.derisk_scaler_report(ret, lvl, composite,
                                   primary=("2023-01-01", "2026-08-31"),
                                   prior=("2021-01-01", "2022-12-31"))
    assert len(rep["derisk"]["grid"]) == 9    # 3 floor_z x 3 lo_mult
    assert len(rep["scaler"]["grid"]) == 6    # 3 k x 2 hi
    assert rep["derisk"]["verdict"] in ("PASS", "FAIL", "marginal", "knife-edge")
    assert rep["scaler"]["verdict"] in ("PASS", "FAIL", "marginal", "knife-edge")
    assert "buy_and_hold" in rep["baselines"] and "layer1_only" in rep["baselines"]


def _flat_price_fn(tickers, start, end):
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(5)
    data = {}
    for t in tickers:
        rets = rng.normal(0.0003, 0.01, size=len(idx))
        data[t] = 100.0 * (1.0 + pd.Series(rets, index=idx)).cumprod()
    return pd.DataFrame(data)


def _synthetic_panel():
    rows = []
    for i, u in enumerate(["D", "AEP", "NEE", "SO", "ETR", "XEL", "DUK", "PCG"]):
        for q in range(6):
            rows.append({
                "utility": u,
                "report_date": pd.Timestamp("2022-01-01") + pd.DateOffset(months=6 * q),
                "capex_plan_usd_m": 10000.0 + 500.0 * q + 100.0 * i,
                "revision_vs_prior_usd_m": np.nan if q == 0 else 500.0 + 20.0 * i,
                "dc_attributed_usd_m": 200.0 if (q >= 3 and i < 3) else np.nan,
                "dc_basis": "stated" if (q >= 3 and i < 3) else "none",
            })
    df = pd.DataFrame(rows)
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def test_lead_lag_table_reports_pearson_corr_at_each_offset():
    idx = pd.date_range("2023-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(6)
    signal_monthly = pd.Series(rng.normal(size=24), index=idx)
    # "forward return" 2 months after date t is built to equal the signal's
    # own value from t -- i.e. the signal genuinely leads by 2 months. Given
    # this module's `shift(k)` convention (negative k = signal leads, matching
    # the VA-probe/Table-B9 doc convention "k=-1 (filings lead)"), the perfect
    # correlation must show up at k=-2, not k=+2.
    fwd_proxy = signal_monthly.shift(2)
    table = cgs.lead_lag_table(signal_monthly, fwd_proxy, ks=range(-3, 4))
    assert set(table.index) == set(range(-3, 4))
    assert table.loc[-2] == pytest.approx(1.0, abs=1e-6)


def test_guidance_signal_report_runs_end_to_end():
    fake_bigfour = pd.DataFrame({"decel2": [0.0, 0.0]},
                                index=pd.PeriodIndex(["2022Q1", "2022Q2"], freq="Q"))
    rep = cgs.guidance_signal_report(
        price_fn=_flat_price_fn, panel_df=_synthetic_panel(), bigfour_fn=lambda: fake_bigfour,
        primary=("2023-01-01", "2023-12-31"), prior=("2021-01-01", "2022-12-31"),
        holdout=("2023-10-01", "2023-12-31"))
    assert rep["universe_used"] == "full"
    assert "timing_basket" in rep and "all_usd" in rep["timing_basket"]
    assert "derisk_scaler_gate" in rep
    assert rep["derisk_scaler_gate"]["derisk"]["verdict"] in ("PASS", "FAIL", "marginal", "knife-edge")
    assert "continuity_lead_lag" in rep and "all_usd" in rep["continuity_lead_lag"]


def test_guidance_signal_report_not_testable_short_circuits():
    df = _panel([{"utility": "ZZZ", "report_date": "1999-01-01",
                 "capex_plan_usd_m": 10.0, "revision_vs_prior_usd_m": np.nan}])
    rep = cgs.guidance_signal_report(price_fn=_flat_price_fn, panel_df=df,
                                     bigfour_fn=lambda: pd.DataFrame({"decel2": []}))
    assert rep["verdict"] == "not_yet_testable"
    assert "timing_basket" not in rep
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -k "derisk_scaler_report or guidance_signal_report" -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write the implementation**

Add these imports to the top of `grid_equipment_basket/capex_guidance_signal.py` (alongside the
existing ones from Task 8):

```python
from grid_equipment_basket.capex_signal import fetch_bigfour_capex
```

Append the rest to `grid_equipment_basket/capex_guidance_signal.py`:

```python
def derisk_scaler_report(ret: pd.Series, lvl: pd.Series, composite: pd.Series, *,
                         primary: tuple[str, str] = config.DC_GUIDANCE_PRIMARY_WINDOW,
                         prior: tuple[str, str] = config.DC_GUIDANCE_PRIOR_WINDOW) -> dict:
    """Spec s6: both the one-directional de-risk (s6.1) and the two-sided
    scaler (s6.2), each on its parameter plateau, scored against the same
    gate `grid_regime.gate_check`/`final_verdict` already uses -- "beats
    layer-1 (price trend gate + vol target) on Sharpe AND Calmar, both
    windows, non-marginal, survives the parameter-neighbour plateau probe"."""
    from grid_equipment_basket import overlay
    from grid_equipment_basket.backtest import calendar_year_returns, compute_metrics
    from grid_equipment_basket.grid_regime import final_verdict, gate_check

    rf, af = config.RISK_FREE_RATE, config.ANN_FACTOR
    windows = {"primary": primary, "prior": prior}

    def _block(series: pd.Series) -> dict:
        return {wk: {"metrics": compute_metrics(series.loc[a:b].dropna(), rf, af),
                    "calendar": calendar_year_returns(series.loc[a:b].dropna())}
               for wk, (a, b) in windows.items()}

    bh = _block(ret)
    l1 = _block(overlay.apply_overlay(ret, lvl, rf))

    def _score(mult: pd.Series) -> dict:
        return _block(overlay.apply_overlay_l2(ret, mult, rf))

    derisk_grid = [{"floor_z": fz, "lo_mult": lm,
                    "block": _score(guidance_derisk_multiplier(composite, floor_z=fz, lo_mult=lm))}
                   for fz in config.DC_GUIDANCE_DERISK_GRID_FLOOR
                   for lm in config.DC_GUIDANCE_DERISK_GRID_LO]
    scaler_grid = [{"k": k, "hi": hi,
                    "block": _score(guidance_scaler(composite, k=k, hi=hi,
                                                    lo=config.DC_GUIDANCE_SCALER_LO))}
                   for k in config.DC_GUIDANCE_SCALER_GRID_K
                   for hi in config.DC_GUIDANCE_SCALER_GRID_HI]

    def _central(grid: list[dict], key: dict) -> dict:
        for row in grid:
            if all(row[k] == v for k, v in key.items()):
                return row
        return grid[0]

    derisk_central = _central(derisk_grid, {"floor_z": config.DC_GUIDANCE_DERISK_FLOOR_Z,
                                            "lo_mult": config.DC_GUIDANCE_DERISK_LO_MULT})
    scaler_central = _central(scaler_grid, {"k": config.DC_GUIDANCE_SCALER_K,
                                            "hi": config.DC_GUIDANCE_SCALER_HI})

    def _neighbour_passes(central: dict, grid: list[dict]) -> list[bool]:
        out = []
        for row in grid:
            if row is central:
                continue
            g = gate_check(row["block"]["primary"], row["block"]["prior"],
                           l1["primary"], l1["prior"], bh["primary"], bh["prior"])
            out.append(bool(g["G1"] and g["G2"]))
        return out or [True]

    def _gate_for(central: dict, grid: list[dict]) -> dict:
        gate = gate_check(central["block"]["primary"], central["block"]["prior"],
                          l1["primary"], l1["prior"], bh["primary"], bh["prior"])
        verdict = final_verdict(gate, _neighbour_passes(central, grid))
        return {"gate": gate, "verdict": verdict}

    return {
        "baselines": {"buy_and_hold": bh, "layer1_only": l1},
        "derisk": {"grid": derisk_grid, "central": derisk_central,
                  **_gate_for(derisk_central, derisk_grid)},
        "scaler": {"grid": scaler_grid, "central": scaler_central,
                  **_gate_for(scaler_central, scaler_grid)},
    }


def lead_lag_table(signal_monthly: pd.Series, fwd_return_monthly: pd.Series,
                   ks: range = range(-6, 7)) -> pd.Series:
    """Plain Pearson correlation between `signal_monthly` and
    `fwd_return_monthly` shifted by each offset `k` in `ks` (negative k =
    signal leads; positive k = basket leads) -- spec s5.3's continuity table,
    same shape as the VA-transmission-probe's and Table-B9's lead-lag tables,
    printed alongside the formal rank-IC/HAC test so a reader can see at a
    glance whether this result looks different from those two negatives."""
    out = {}
    for k in ks:
        pair = pd.concat([signal_monthly.rename("s"),
                          fwd_return_monthly.shift(k).rename("r")], axis=1).dropna()
        out[k] = float(pair["s"].corr(pair["r"])) if len(pair) >= 3 else float("nan")
    return pd.Series(out).rename("lead_lag_corr")


def _default_price_fn(tickers, start, end):
    from grid_equipment_basket.data.prices import fetch_prices
    return fetch_prices(tickers, start, end)


_SERIES_SPECS: tuple[tuple[str, str, str | None], ...] = (
    ("all_usd",       "revision_vs_prior_usd_m",   None),
    ("all_pct",       "revision_vs_prior_usd_m",   "prior_capex_plan_usd_m"),
    ("dc_stated_usd", "dc_attributed_usd_m",        None),
    ("dc_stated_pct", "dc_attributed_usd_m",        "prior_capex_plan_usd_m"),
    ("dc_filled_usd", "dc_attributed_usd_m_filled",  None),
    ("dc_filled_pct", "dc_attributed_usd_m_filled",  "prior_capex_plan_usd_m"),
)


def guidance_signal_report(price_fn=None, panel_df=None, bigfour_fn=None,
                           primary: tuple[str, str] = config.DC_GUIDANCE_PRIMARY_WINDOW,
                           prior: tuple[str, str] = config.DC_GUIDANCE_PRIOR_WINDOW,
                           holdout: tuple[str, str] = config.DC_GUIDANCE_HOLDOUT_WINDOW) -> dict:
    """Orchestrates the full spec: feasibility kill (s3.3) -> six series
    (s4.1/s4.2) -> daily composites (s4.3) -> timing bar (s5, basket AND
    long/short-spread) -> de-risk/scaler gate (s6, on the headline `all_usd`
    series). `price_fn(tickers, start, end) -> DataFrame`, `panel_df`, and
    `bigfour_fn() -> DataFrame` are injectable for tests; defaults are the
    live yfinance fetcher, `utility_capex_guidance.load_capex_guidance()`, and
    `capex_signal.fetch_bigfour_capex`."""
    from grid_equipment_basket.basket import simulate_basket
    from grid_resilience.data import utility_capex_guidance as udg

    price_fn = price_fn or _default_price_fn
    bigfour_fn = bigfour_fn or fetch_bigfour_capex
    panel_df = panel_df if panel_df is not None else udg.load_capex_guidance()

    used_panel, feas, universe_used = feasibility_gate(panel_df, primary)
    if universe_used == "not_testable":
        return {"universe_used": universe_used, "feasibility": feas, "verdict": "not_yet_testable"}

    panel_filled = udg.impute_dc_attributed(used_panel)

    series = {}
    for name, value_col, denom_col in _SERIES_SPECS:
        src = panel_filled if value_col.endswith("_filled") else used_panel
        series[name] = udg.aggregate_revision_series(
            src, value_col=value_col, denom_col=denom_col,
            ttm_quarters=config.DC_GUIDANCE_TTM_QUARTERS)

    span_start, span_end = prior[0], primary[1]
    composites = {name: guidance_composite(ev, span_start, span_end) for name, ev in series.items()}

    tickers = sorted(set(config.UNIVERSE + config.DER_SHORT_SLEEVE + ["SMH", "^TNX"]))
    prices = price_fn(tickers, span_start, span_end)
    basket_cols = [t for t in config.UNIVERSE if t in prices.columns]
    der_cols = [t for t in config.DER_SHORT_SLEEVE if t in prices.columns]

    basket = simulate_basket(prices[basket_cols], span_start, span_end,
                             config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT, None)
    ret, lvl = basket.returns, (1.0 + basket.returns).cumprod()

    der_ret = (prices[der_cols].pct_change().dropna(how="all").mean(axis=1)
              if der_cols else pd.Series(dtype=float))
    spread_ret = (ret - der_ret.reindex(ret.index)).dropna()

    tnx = prices["^TNX"].dropna() if "^TNX" in prices.columns else pd.Series(dtype=float)
    d10y_monthly = tnx.resample("ME").last().diff() if not tnx.empty else pd.Series(dtype=float)
    smh_monthly = ((1.0 + prices["SMH"].pct_change().dropna()).resample("ME").prod() - 1.0
                  if "SMH" in prices.columns else pd.Series(dtype=float))
    bigfour = bigfour_fn()
    if not bigfour.empty:
        bigfour_s = bigfour["decel2"].copy()
        bigfour_s.index = [p.to_timestamp(how="end") for p in bigfour_s.index]
        bigfour_monthly = bigfour_s.resample("ME").ffill()
    else:
        bigfour_monthly = pd.Series(dtype=float)

    controls = pd.DataFrame({"d10y": d10y_monthly, "smh": smh_monthly, "bigfour": bigfour_monthly})

    timing_basket = timing_report(composites, ret, controls, primary=primary, holdout=holdout)
    timing_spread = (timing_report(composites, spread_ret, controls, primary=primary, holdout=holdout)
                     if not spread_ret.empty else {})

    basket_nav = _monthly_nav(ret)
    fwd_1m = _forward_return(basket_nav, 1)
    continuity = {name: lead_lag_table(comp.resample("ME").last(), fwd_1m)
                 for name, comp in composites.items()}

    derisk_scaler = derisk_scaler_report(ret, lvl, composites["all_usd"], primary=primary, prior=prior)

    return {
        "universe_used": universe_used, "feasibility": feas,
        "windows": {"primary": primary, "prior": prior, "holdout": holdout},
        "timing_basket": timing_basket, "timing_spread": timing_spread,
        "continuity_lead_lag": continuity,
        "derisk_scaler_gate": derisk_scaler,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -v`
Expected: all passed

- [ ] **Step 5: Run the FULL test suite to check for regressions**

Run: `pytest tests/ -v`
Expected: all passed (no existing test broken by the new module or config additions)

- [ ] **Step 6: Commit**

```bash
git add grid_equipment_basket/capex_guidance_signal.py tests/grid_equipment_basket/test_capex_guidance_signal.py
git commit -m "feat(capex-guidance): derisk_scaler_report + top-level guidance_signal_report"
```

---

### Task 14: `guidance_table` — printable report

**Files:**
- Modify: `grid_equipment_basket/capex_guidance_signal.py`
- Test: `tests/grid_equipment_basket/test_capex_guidance_signal.py`

**Interfaces:**
- Consumes: `guidance_signal_report`'s return dict shape.
- Produces: `guidance_table(rep) -> str`. Used by Task 15's CLI wiring.

- [ ] **Step 1: Write the failing test**

```python
def test_guidance_table_handles_not_yet_testable_report():
    rep = {"universe_used": "not_testable",
          "feasibility": {"n_usable": 0, "n_total": 3, "per_utility": {}},
          "verdict": "not_yet_testable"}
    out = cgs.guidance_table(rep)
    assert "not_yet_testable" in out or "NOT YET TESTABLE" in out.upper()


def test_guidance_table_renders_full_report():
    fake_bigfour = pd.DataFrame({"decel2": [0.0, 0.0]},
                                index=pd.PeriodIndex(["2022Q1", "2022Q2"], freq="Q"))
    rep = cgs.guidance_signal_report(
        price_fn=_flat_price_fn, panel_df=_synthetic_panel(), bigfour_fn=lambda: fake_bigfour,
        primary=("2023-01-01", "2023-12-31"), prior=("2021-01-01", "2022-12-31"),
        holdout=("2023-10-01", "2023-12-31"))
    out = cgs.guidance_table(rep)
    assert "full" in out
    assert "all_usd" in out
    assert "DERISK" in out.upper() or "SCALER" in out.upper()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -k guidance_table -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Write the implementation**

Append to `grid_equipment_basket/capex_guidance_signal.py`:

```python
def _p(x) -> str:
    return "n/a" if x != x else f"{x * 100:.1f}%"


def _f(x) -> str:
    return "n/a" if x != x else f"{x:.2f}"


def guidance_table(rep: dict) -> str:
    L = ["CAPEX-GUIDANCE REVISION SIGNAL (Deliverable D)", ""]
    if rep.get("universe_used") == "not_testable":
        L.append("  FEASIBILITY: NOT YET TESTABLE")
        feas = rep["feasibility"]
        L.append(f"  n_usable={feas['n_usable']} / n_total={feas['n_total']}")
        return "\n".join(L)

    L.append(f"  universe_used: {rep['universe_used']}  "
             f"(n_usable={rep['feasibility']['n_usable']}/{rep['feasibility']['n_total']})")
    w = rep["windows"]
    L.append(f"  windows: primary {w['primary']}  prior {w['prior']}  holdout {w['holdout']}")
    L.append("")
    L.append("  TIMING BAR (vs basket) -- rank-IC t | control(w/o hyperscaler) t | PASS?")
    for name, by_h in rep["timing_basket"].items():
        for h, cell in by_h.items():
            ic = cell["rank_ic_primary"]
            ctrl = cell["control_without_hyperscaler"]
            L.append(f"    {name:<16} h={h}m  ic={_f(ic['ic'])} t={_f(ic['t'])}  "
                     f"ctrl_t={_f(ctrl['t'].get('signal', float('nan')))}  "
                     f"{'PASS' if cell['passed'] else 'fail'}")
    L.append("")
    if "all_usd" in rep.get("continuity_lead_lag", {}):
        ll = rep["continuity_lead_lag"]["all_usd"]
        L.append("  CONTINUITY -- lead-lag corr, all_usd vs 1m-fwd basket return "
                 "(negative k = signal leads):")
        L.append("    " + "  ".join(f"k={k}:{_f(v)}" for k, v in ll.items()))
    L.append("")
    for leg in ("derisk", "scaler"):
        g = rep["derisk_scaler_gate"][leg]
        L.append(f"  {leg.upper()} GATE: verdict={g['verdict']}  "
                 f"G1={g['gate']['G1']} G2={g['gate']['G2']} G3={g['gate']['G3']} "
                 f"marginal={g['gate']['marginal']}")
    return "\n".join(L)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/grid_equipment_basket/test_capex_guidance_signal.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add grid_equipment_basket/capex_guidance_signal.py tests/grid_equipment_basket/test_capex_guidance_signal.py
git commit -m "feat(capex-guidance): guidance_table printable report"
```

---

### Task 15: CLI wiring — `--capex-guidance`

**Files:**
- Modify: `grid_equipment_basket/__main__.py:54-58` (add the flag, next to `--overlay-l2`),
  `grid_equipment_basket/__main__.py:91-95` (add the invocation, next to the `--overlay-l2` block),
  `grid_equipment_basket/__main__.py:107-125` (add `_write_guidance_outputs` next to
  `_write_regime_outputs`)

**Interfaces:**
- Consumes: `capex_guidance_signal.guidance_signal_report`, `capex_guidance_signal.guidance_table`.
- Produces: `python -m grid_equipment_basket --capex-guidance` writes
  `<output>/capex_guidance_timing.csv` and `<output>/capex_guidance_gates.json`. Used by Task 17's
  live run.

- [ ] **Step 1: Add the flag**

In `grid_equipment_basket/__main__.py`, right after the existing `--overlay-l2` argument
(currently lines 56–58):

```python
    ap.add_argument("--overlay-l2", dest="overlay_l2", action="store_true",
                    help="also run the layer-2 grid-congestion regime ladder (both windows) "
                         "and write regime_metrics.csv / regime_timeline.csv")
    ap.add_argument("--capex-guidance", dest="capex_guidance", action="store_true",
                    help="also run the utility capex-guidance revision signal report "
                         "(Deliverable D) and write capex_guidance_timing.csv / "
                         "capex_guidance_gates.json")
```

- [ ] **Step 2: Add the invocation**

Right after the existing `if args.overlay_l2:` block (currently lines 91–95):

```python
    if args.overlay_l2:
        from grid_equipment_basket import grid_regime
        rrep = grid_regime.regime_report()
        print("\n" + grid_regime.regime_table(rrep))
        _write_regime_outputs(rrep, out)

    if args.capex_guidance:
        from grid_equipment_basket import capex_guidance_signal
        grep = capex_guidance_signal.guidance_signal_report()
        print("\n" + capex_guidance_signal.guidance_table(grep))
        _write_guidance_outputs(grep, out)
```

- [ ] **Step 3: Add the output writer**

Right after the existing `_write_regime_outputs` function (currently ends at line 125, just
before `_write_value_chain_outputs`):

```python
def _write_guidance_outputs(rep: dict, out: Path) -> None:
    import json

    if rep.get("universe_used") == "not_testable":
        (out / "capex_guidance_gates.json").write_text(json.dumps(rep, default=str, indent=2))
        print(f"\nwrote {out}/capex_guidance_gates.json (not yet testable)")
        return

    rows = []
    for name, by_h in rep["timing_basket"].items():
        for h, cell in by_h.items():
            rows.append({
                "series": name, "horizon_months": h,
                "rank_ic": cell["rank_ic_primary"]["ic"], "rank_ic_t": cell["rank_ic_primary"]["t"],
                "control_t": cell["control_without_hyperscaler"]["t"].get("signal"),
                "passed": cell["passed"],
            })
    pd.DataFrame(rows).to_csv(out / "capex_guidance_timing.csv", index=False)
    (out / "capex_guidance_gates.json").write_text(json.dumps({
        "universe_used": rep["universe_used"], "feasibility": rep["feasibility"],
        "derisk": {"verdict": rep["derisk_scaler_gate"]["derisk"]["verdict"],
                  "gate": rep["derisk_scaler_gate"]["derisk"]["gate"]},
        "scaler": {"verdict": rep["derisk_scaler_gate"]["scaler"]["verdict"],
                  "gate": rep["derisk_scaler_gate"]["scaler"]["gate"]},
    }, default=str, indent=2))
    print(f"\nwrote {out}/capex_guidance_timing.csv, capex_guidance_gates.json")
```

- [ ] **Step 4: Smoke-test the CLI wiring with injected data (no live network)**

Run:
```bash
python -c "
from grid_equipment_basket import capex_guidance_signal as cgs, __main__ as m
from pathlib import Path
import tempfile, pandas as pd, numpy as np

def flat_price_fn(tickers, start, end):
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(9)
    return pd.DataFrame({t: 100.0 * (1.0 + pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)).cumprod()
                         for t in tickers})

rows = []
for i, u in enumerate(['D','AEP','NEE','SO','ETR','XEL','DUK','PCG']):
    for q in range(6):
        rows.append({'utility': u, 'report_date': pd.Timestamp('2022-01-01') + pd.DateOffset(months=6*q),
                    'capex_plan_usd_m': 10000.0 + 500.0*q, 'revision_vs_prior_usd_m': np.nan if q==0 else 500.0,
                    'dc_attributed_usd_m': np.nan, 'dc_basis': 'none'})
panel = pd.DataFrame(rows); panel['report_date'] = pd.to_datetime(panel['report_date'])

rep = cgs.guidance_signal_report(price_fn=flat_price_fn, panel_df=panel,
                                 bigfour_fn=lambda: pd.DataFrame({'decel2': [0.0]}, index=pd.PeriodIndex(['2022Q1'], freq='Q')),
                                 primary=('2023-01-01','2023-12-31'), prior=('2021-01-01','2022-12-31'),
                                 holdout=('2023-10-01','2023-12-31'))
print(cgs.guidance_table(rep))
with tempfile.TemporaryDirectory() as d:
    m._write_guidance_outputs(rep, Path(d))
    print(sorted(Path(d).iterdir()))
"
```
Expected: prints the table, then a list containing `capex_guidance_timing.csv` and
`capex_guidance_gates.json`, no traceback.

- [ ] **Step 5: Commit**

```bash
git add grid_equipment_basket/__main__.py
git commit -m "feat(capex-guidance): --capex-guidance CLI flag"
```

---

### Task 16: README update (CLAUDE.md rule)

**Files:**
- Modify: `grid_equipment_basket/README.md` (append a new section)

**Interfaces:** none — documentation only.

- [ ] **Step 1: Append a stub section**

Per CLAUDE.md, README updates are written with Bash, not the Write tool:

```bash
cat >> grid_equipment_basket/README.md << 'EOF'

## Capex-Guidance Revision Signal (Deliverable D) — 2026-09-04

Handoff: `docs/handoff_2026-09-03-transmission-project-filings.md` §6. Design spec:
`docs/superpowers/specs/2026-09-04-capex-guidance-signal-design.md`. Results:
`docs/capex-guidance-signal-results.md`.

An aggregate timing signal (not a cross-sectional factor — it never re-weights the 9 basket
names) built from a hand/web-assembled panel of 15 large US electric utilities' forward
multi-year capex-guidance revisions (`grid_resilience/data/seed/utility_capex_guidance.csv`,
loaded by `grid_resilience/data/utility_capex_guidance.py`). Tests the revision-flow ($ and
size-weighted %, both the whole-panel total and the data-center-attributed portion, the latter
with a point-in-time EM-style fill for utilities that only qualitatively mention data centers)
as (1) a monthly rank-IC timing signal against forward basket and long/short-spread returns,
controlling for Δ10y yield, SMH, and (separately) the big-four hyperscaler capex series, and
(2) a one-directional de-risk multiplier / two-sided scaler for the basket's exposure, gated
against layer-1 (price trend gate + vol target) with the same `grid_regime.gate_check`/
`final_verdict` machinery the FTR-bid and grid-congestion signals use.

Run with `python -m grid_equipment_basket --capex-guidance` (writes
`capex_guidance_timing.csv`, `capex_guidance_gates.json` to the output dir).

**RESULT: <<filled in by the live run — see docs/capex-guidance-signal-results.md>>**
EOF
```

- [ ] **Step 2: Verify it appended cleanly**

Run: `tail -25 grid_equipment_basket/README.md`
Expected: the new section, ending with the `RESULT:` placeholder line (Task 17 replaces that
line with the actual verdict — this is the one intentional placeholder in this plan, explicitly
because the real number doesn't exist until the live run happens).

- [ ] **Step 3: Commit**

```bash
git add grid_equipment_basket/README.md
git commit -m "docs(capex-guidance): README section for Deliverable D"
```

---

### Task 17: Live run + results doc + memory update

**Files:**
- Run: `python -m grid_equipment_basket --capex-guidance` (networked — yfinance for basket + DER
  sleeve + SMH + ^TNX prices, SEC XBRL for the big-four hyperscaler control)
- Create: `docs/capex-guidance-signal-results.md`
- Modify: `grid_equipment_basket/README.md` (replace the Task 16 placeholder line with the real
  verdict)
- Create: `/Users/sandhyapersad/.claude/projects/-Users-sandhyapersad-acadia/memory/capex-guidance-signal-negative.md`
  or `-positive.md` (name reflects the actual outcome)
- Modify: `/Users/sandhyapersad/.claude/projects/-Users-sandhyapersad-acadia/memory/MEMORY.md`

**Interfaces:** none — this is the reporting task, consuming everything built in Tasks 1–15.

- [ ] **Step 1: Run the live report**

```bash
python -m grid_equipment_basket --capex-guidance --output ./output_grid_equipment
```

This is the first live (networked) run — expect it to take a few minutes (yfinance for the
basket + `DER_SHORT_SLEEVE` + SMH + ^TNX, plus `capex_signal.fetch_bigfour_capex`'s SEC XBRL
pull, which is cached after the first call). Capture the full printed table and note:
`universe_used` (full vs. fallback vs. not_testable — the feasibility-kill outcome from Task 3),
every `timing_basket` cell's rank-IC/control t-stats and `passed` flag across all six series ×
three horizons, the continuity lead-lag row, and both the `derisk` and `scaler` gate verdicts.

- [ ] **Step 2: Write the results doc**

Following the structure of `docs/ftr-bid-signal-results.md` / `docs/va-transmission-filings-
probe-results.md` (both already in this repo), write `docs/capex-guidance-signal-results.md`
via Bash (per CLAUDE.md) covering: (1) a TL;DR with the headline verdict, (2) the data-quality
table (H/M/L confidence mix from the populated seed panel, Task 3), (3) the feasibility-kill
outcome, (4) the full timing-bar table (all six series × three horizons, both control specs,
combined pass/fail per spec §5.4), (5) the continuity lead-lag table, (6) both gate tables
(derisk + scaler) with the G1/G2/G3 breakdown, (7) the honesty caveats verbatim from spec §11,
(8) if this is a FAIL: state plainly this is attempt #15 (or whatever the running count is) and
note how many of Deliverables C/D/E remain before the handoff's 17-attempt stop rule; if this is
a PASS on either the timing or the de-risk/scaler leg: state which one, on which series, and
flag the multiple-comparisons caveat (spec §11) before calling it a discovery.

- [ ] **Step 3: Replace the README placeholder with the real result**

Edit `grid_equipment_basket/README.md`, replacing the
`**RESULT: <<filled in by the live run...>>**` line from Task 16 with 2-4 sentences stating the
actual verdict and pointing at the results doc.

- [ ] **Step 4: Write the memory file**

Following the existing pattern (see `va-transmission-filings-probe-negative.md`,
`pjm-mw-revision-signal-negative.md` for a FAIL; any of the "-BUILT" memories for a PASS), write
one memory file with frontmatter (`name`, `description`, `metadata.type: project`) stating what
was built, the actual verdict with the key numbers, and linking `[[va-transmission-filings-
probe-negative]]` and `[[pjm-mw-revision-signal-negative]]` (the two prior forecast/flow
negatives this was designed to contrast with) plus `[[transmission-rate-base-negative]]` (running
attempt-count context). If FAIL: note the updated attempt count and how many of C/E remain
before the 17-attempt stop rule. If PASS: note it's the first commitment-data signal to clear a
gate in this theme and what should happen next (does it get wired live via
`config.DC_GUIDANCE_*` the way `REGIME_ENABLED` gates layer 2?).

- [ ] **Step 5: Update the memory index**

Add one line to `/Users/sandhyapersad/.claude/projects/-Users-sandhyapersad-acadia/memory/MEMORY.md`
pointing at the new memory file, in the same one-line style as the existing rows.

- [ ] **Step 6: Final commit**

```bash
git add docs/capex-guidance-signal-results.md grid_equipment_basket/README.md
git add /Users/sandhyapersad/.claude/projects/-Users-sandhyapersad-acadia/memory/
git commit -m "docs(capex-guidance): live-run results + memory update (Deliverable D)"
```
