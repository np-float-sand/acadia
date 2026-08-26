# Multi-Source Data-Center Demand Exposure Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing PJM-only DC-load signal (6 of ~18 tickers) to ~11 tickers by combining a new ERCOT large-load queue signal, a PJM Large Load Adjustment cross-check/zone-fix, and a named hyperscaler-PPA event signal for the merchant sleeve — each source picked because it's traceable (explicit DC attribution or a named contract), not a generic proxy.

**Architecture:** Three new data layers, each producing a `dates x tickers` raw-score DataFrame in the same shape as the existing `compute_dc_load_signal()` output. A new combination function (`dc_demand_combined.py`) z-scores each layer within its own covered-ticker subpopulation (never mixing raw scales before z-scoring — see Task 9) and merges them into one history, which flows into the existing `fill_with_icr()` → `build_rolling_factor()` pipeline unchanged. Wired behind a new `--regulated-signal dc-multi` CLI choice alongside the existing `icr` / `dc-queue` options.

**Tech Stack:** Python, pandas, `pd.ExcelFile`/`requests` for PJM's LAS spreadsheet, a checked-in seed CSV for ERCOT (see Task 4 rationale) and hyperscaler deals, `pytest` with `unittest.mock.patch`.

**Spec:** `docs/superpowers/specs/2026-08-26-dc-demand-exposure-signal-design.md`

## Global Constraints

- Follow existing code style: `from __future__ import annotations` at top of every new module, type hints on all function signatures, no comments explaining *what* code does (only *why*, and only when non-obvious).
- All new config constants go in `grid_resilience/config.py`, never hardcoded in the module that uses them.
- All new fetchers cache to `CACHE_DIR` — never re-hit a live API/URL inside a test; mock at the parsing-function boundary.
- **Point-in-time discipline for Layer 3 (hyperscaler deals):** every event's date must be the earliest public disclosure (press release / earnings call / SEC 8-K), never a later ISO/regulatory filing date. Any row whose date isn't confirmed against a primary source is marked `date_confidence="needs_verification"` and excluded from the computed signal until confirmed — never silently treated as confirmed.
- **No guessed ticker mappings.** Do not add a zone or TSP to `TICKER_NODE_MAP` without a corroborating source already in the codebase or a verified primary source (10-K service territory, company ownership record). Task 2 documents exactly which additions are confirmed vs. flagged for follow-up.
- No existing test may break. `TICKER_NODE_MAP["FE"]["load_zones"]` changes in Task 2 will change `compute_dc_load_signal()` output for FE — re-run `tests/data/test_dc_load_data.py` and `tests/factor/test_business_model_arch.py` after that task specifically.

---

### Task 1: PJM Large Load Adjustment fetcher (Layer 2a)

**Files:**
- Create: `grid_resilience/data/pjm_large_load_data.py`
- Test: `tests/data/test_pjm_large_load_data.py` (new)

**Interfaces:**
- Produces: `fetch_large_load_adjustment(vintage: str = "2026", use_cache: bool = True) -> pd.DataFrame` — long format `[zone, area, industry, year, mw_demand, mw_capacity]`.

The live spreadsheet (`LARGE_LOAD_SOURCE_URLS["2026"]`, confirmed live during design — 2025-11-24 PJM LAS vintage) has three sheets: `summary` (zone/area → industry tag), `2026 DEMAND Requests` and `2026 CAPACITY Requests` (zone/area × year MW grids, real header row at index 3, data from index 4, one stray no-data "AE" row to drop).

- [ ] **Step 1: Write the failing tests for the pure parsing functions**

Create `tests/data/test_pjm_large_load_data.py`:

```python
import pandas as pd
from grid_resilience.data.pjm_large_load_data import _parse_industry_tags, _parse_requests_sheet


def test_parse_industry_tags_drops_header_repeat_row_and_keeps_only_zone_rows():
    raw = pd.DataFrame([
        ["ZONENAME", "AREANAME", "Capacity", "Demand", "industry", "note"],
        ["BGE", "BGE", "x", "x", "data center", None],
        ["AEP", "APCO", "x", None, "industrial & crypto", None],
    ])
    result = _parse_industry_tags(raw)
    assert list(result.columns) == ["zone", "area", "industry"]
    assert len(result) == 2
    assert result.iloc[0].to_dict() == {"zone": "BGE", "area": "BGE", "industry": "data center"}


def test_parse_requests_sheet_melts_year_columns_to_long_format():
    raw = pd.DataFrame([
        ["Total Demand Request for Large Load Adjustment"] + [None] * 5,
        ["Note..."] + [None] * 5,
        [None] * 6,
        [None, "ZONENAME", "AREANAME", 2025.0, 2026.0, 2027.0],
        [None, None, None, None, None, None],   # the stray no-data "AE" row
        [None, "BGE", "BGE", None, 17.756, 25.369],
    ])
    result = _parse_requests_sheet(raw, "mw_demand")
    assert set(result.columns) == {"zone", "area", "year", "mw_demand"}
    bge_2026 = result[(result["zone"] == "BGE") & (result["year"] == 2026)]
    assert bge_2026["mw_demand"].iloc[0] == 25.369
    assert not (result["zone"] == "AE").any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/test_pjm_large_load_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grid_resilience.data.pjm_large_load_data'`

- [ ] **Step 3: Implement the module**

Create `grid_resilience/data/pjm_large_load_data.py`:

```python
from __future__ import annotations

import pandas as pd

from grid_resilience.config import CACHE_DIR

# One entry per PJM Load Analysis Subcommittee "Large Load Adjustment
# Requests" vintage, confirmed live during design (2026-08-26). Add new
# vintages here as PJM posts them each fall — never guess the URL pattern,
# PJM's media paths are not predictable from the date alone.
LARGE_LOAD_SOURCE_URLS = {
    "2026": (
        "https://www.pjm.com/-/media/DotCom/committees-groups/subcommittees/"
        "las/2025/20250916/20250916-post-meeting---informational-only---"
        "large-load-adjustment-requests.xlsx"
    ),
}


def _parse_industry_tags(raw_summary: pd.DataFrame) -> pd.DataFrame:
    """Parse the 'summary' sheet's ZONENAME/AREANAME/industry columns."""
    df = raw_summary.iloc[1:].copy()
    df.columns = ["zone", "area", "has_capacity", "has_demand", "industry", "note"]
    return df[df["zone"].notna()][["zone", "area", "industry"]].reset_index(drop=True)


def _parse_requests_sheet(raw: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Parse a '<vintage> DEMAND/CAPACITY Requests' sheet into long format."""
    header_row = raw.iloc[3]
    years = [int(y) for y in header_row[3:] if pd.notna(y)]
    df = raw.iloc[4:].copy()
    df.columns = ["_drop", "zone", "area"] + years
    df = df[df["zone"].notna()].drop(columns="_drop")
    long_df = df.melt(id_vars=["zone", "area"], var_name="year", value_name=value_name)
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    return long_df.dropna(subset=[value_name]).reset_index(drop=True)


def fetch_large_load_adjustment(vintage: str = "2026", use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch and parse a PJM LAS 'Large Load Adjustment Requests' vintage.
    Returns long format: [zone, area, industry, year, mw_demand, mw_capacity].
    """
    cache_file = CACHE_DIR / f"pjm_large_load_adjustment_{vintage}.parquet"
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    url = LARGE_LOAD_SOURCE_URLS[vintage]
    xls = pd.ExcelFile(url)
    industry = _parse_industry_tags(xls.parse("summary"))
    demand = _parse_requests_sheet(xls.parse(f"{vintage} DEMAND Requests"), "mw_demand")
    capacity = _parse_requests_sheet(xls.parse(f"{vintage} CAPACITY Requests"), "mw_capacity")

    merged = demand.merge(capacity, on=["zone", "area", "year"], how="outer")
    merged = merged.merge(industry, on=["zone", "area"], how="left")

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_file)
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_pjm_large_load_data.py -v`
Expected: 2 passed

- [ ] **Step 5: Manually verify against the live spreadsheet**

Run:
```bash
source ~/.zshrc
.venv/bin/python -c "
from grid_resilience.data.pjm_large_load_data import fetch_large_load_adjustment
df = fetch_large_load_adjustment(use_cache=False)
print(df.shape)
print(df['industry'].value_counts())
print(df[df['zone']=='BGE'].sort_values('year').head())
"
```
Expected: non-empty; industry value counts dominated by "data center"; BGE rows show increasing MW by year matching the 2025-11-24 deck's BGE chart (17.756 in 2026 rising toward 358.4 by 2033+).

- [ ] **Step 6: Commit**

```bash
git add grid_resilience/data/pjm_large_load_data.py tests/data/test_pjm_large_load_data.py
git commit -m "feat: add PJM Large Load Adjustment fetcher (Layer 2a)"
```

---

### Task 2: FE zone-map fix + verified-mapping documentation

**Files:**
- Modify: `grid_resilience/data/utility_node_map.py:81`
- Test: `tests/data/test_dc_load_data.py` (existing — add one test)

**Interfaces:** none new — this changes `compute_dc_load_signal()`'s output for FE by widening the zones it aggregates over.

`TICKER_NODE_MAP["FE"]["nodes"]` already lists `["ATSI", "JCPL", "METED", "PENELEC"]`, but `compute_dc_load_signal()` reads `load_zones`, currently `["ATSI", "JCPL"]` — the exact gap `docs/compact_2026-08-13-dc-load-signal-results.md` flagged ("this score reflects roughly half of FE's actual footprint"). METED and PENELEC are safe to add: they're already present in FE's own `nodes` field and match FE's documented PA territory.

**Do NOT add an "APS" (Allegheny Power System) zone in this task.** FE's West Virginia/Maryland footprint (from the old Allegheny Energy acquisition) may also map to PJM's "APS" zone, but this hasn't been verified against FE's 10-K service-territory map in this session — flag it in a code comment as a follow-up, don't guess it in.

- [ ] **Step 1: Write the failing test**

Add to `tests/data/test_dc_load_data.py`:

```python
def test_fe_load_zones_include_meted_and_penelec():
    from grid_resilience.data.utility_node_map import TICKER_NODE_MAP
    assert set(TICKER_NODE_MAP["FE"]["load_zones"]) >= {"ATSI", "JCPL", "METED", "PENELEC"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v -k fe_load_zones`
Expected: FAIL — `load_zones` is currently `["ATSI", "JCPL"]`

- [ ] **Step 3: Make the fix**

In `grid_resilience/data/utility_node_map.py`, change the `"FE"` entry:

```python
    "FE": {
        "name": "FirstEnergy",
        "iso": "PJM",
        "nodes": ["ATSI", "JCPL", "METED", "PENELEC"],
        # NOTE: FE's WV/MD territory (ex-Allegheny Energy) may also map to
        # PJM's "APS" (Allegheny Power System) zone — not verified against
        # FE's 10-K service-territory map yet. Confirm before adding "APS"
        # here; don't guess it in.
        "load_zones": ["ATSI", "JCPL", "METED", "PENELEC"],
        "service_territory": "OH, PA, NJ, WV, MD T&D",
        "notes": "ATSI zone most material. Known seam congestion with MISO.",
        "business_model": "regulated",
        "pass_through": 0.05,
    },
```

- [ ] **Step 4: Run test to verify it passes, then the full DC-signal and business-model-arch suites**

Run:
```bash
.venv/bin/python -m pytest tests/data/test_dc_load_data.py -v
.venv/bin/python -m pytest tests/factor/test_business_model_arch.py -v
```
Expected: all pass. If any `test_business_model_arch.py` case hardcodes FE's factor score, update the expected value with a comment explaining the zone-map fix caused the change — don't silently adjust without noting why.

- [ ] **Step 5: Commit**

```bash
git add grid_resilience/data/utility_node_map.py tests/data/test_dc_load_data.py
git commit -m "fix: add METED/PENELEC to FE's DC-signal load_zones"
```

---

### Task 3: PJM Large Load cross-check (Layer 2b)

**Files:**
- Modify: `grid_resilience/data/pjm_large_load_data.py`
- Test: `tests/data/test_pjm_large_load_data.py`

**Interfaces:**
- Consumes: `fetch_large_load_adjustment()` output (Task 1); `compute_dc_load_signal()` output (existing); `TICKER_NODE_MAP`
- Produces: `cross_check_dc_signal(dc_history: pd.DataFrame, large_load: pd.DataFrame, node_map: dict) -> pd.DataFrame` — one row per ticker: `[ticker, latest_dc_queue_score, industries, mw_demand_2030, agrees]`, where `agrees` is `False` when the generation-queue signal is materially positive but the LAS industry tag for that ticker's zones is *not* "data center"-flavored (the exact failure mode that made the original FE result ambiguous).

- [ ] **Step 1: Write the failing test**

Add to `tests/data/test_pjm_large_load_data.py`:

```python
def test_cross_check_flags_disagreement_when_industry_tag_is_not_data_center():
    import pandas as pd
    from grid_resilience.data.pjm_large_load_data import cross_check_dc_signal

    dc_history = pd.DataFrame({"AEP": [0.8]}, index=[pd.Timestamp("2025-12-31")])
    large_load = pd.DataFrame([
        {"zone": "AEP", "area": "APCO", "industry": "industrial & crypto", "year": 2030, "mw_demand": 500.0, "mw_capacity": 600.0},
    ])
    node_map = {"AEP": {"iso": "PJM", "load_zones": ["AEP", "DAYTON"]}}

    result = cross_check_dc_signal(dc_history, large_load, node_map)
    row = result[result["ticker"] == "AEP"].iloc[0]
    assert row["agrees"] == False
    assert "industrial & crypto" in row["industries"]


def test_cross_check_agrees_when_industry_tag_is_data_center():
    import pandas as pd
    from grid_resilience.data.pjm_large_load_data import cross_check_dc_signal

    dc_history = pd.DataFrame({"EXC": [0.5]}, index=[pd.Timestamp("2025-12-31")])
    large_load = pd.DataFrame([
        {"zone": "PECO", "area": "PECO", "industry": "data center", "year": 2030, "mw_demand": 800.0, "mw_capacity": 900.0},
    ])
    node_map = {"EXC": {"iso": "PJM", "load_zones": ["PECO", "BGE", "PEPCO", "COMED"]}}

    result = cross_check_dc_signal(dc_history, large_load, node_map)
    assert result[result["ticker"] == "EXC"].iloc[0]["agrees"] == True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/test_pjm_large_load_data.py -v -k cross_check`
Expected: FAIL — function doesn't exist

- [ ] **Step 3: Implement `cross_check_dc_signal`**

Add to `grid_resilience/data/pjm_large_load_data.py`:

```python
def cross_check_dc_signal(
    dc_history: pd.DataFrame,
    large_load: pd.DataFrame,
    node_map: dict,
) -> pd.DataFrame:
    """
    Diagnostic (not blended into the factor score): flags tickers where the
    existing generation-queue DC signal is positive but PJM's own
    industry-tagged Large Load data doesn't call the driving zone(s) a
    data center — the exact ambiguity that made the original FE result
    unverifiable (see docs/compact_2026-08-13-dc-load-signal-results.md).
    """
    if dc_history.empty:
        return pd.DataFrame(columns=["ticker", "latest_dc_queue_score", "industries", "mw_demand_2030", "agrees"])

    latest_date = dc_history.index.max()
    rows = []
    for ticker, info in node_map.items():
        if info.get("iso") != "PJM" or ticker not in dc_history.columns:
            continue
        score = dc_history.loc[latest_date, ticker]
        zones = info.get("load_zones", [])
        zone_rows = large_load[large_load["zone"].isin(zones)]
        industries = sorted(zone_rows["industry"].dropna().unique().tolist())
        mw_2030 = float(zone_rows[zone_rows["year"] == 2030]["mw_demand"].sum())
        is_dc_tagged = any("data center" in i for i in industries)
        agrees = (pd.isna(score) or score <= 0) or is_dc_tagged
        rows.append({
            "ticker": ticker,
            "latest_dc_queue_score": score,
            "industries": industries,
            "mw_demand_2030": mw_2030,
            "agrees": agrees,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_pjm_large_load_data.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add grid_resilience/data/pjm_large_load_data.py tests/data/test_pjm_large_load_data.py
git commit -m "feat: add PJM Large Load cross-check diagnostic for DC-queue signal"
```

---

### Task 4: ERCOT seed data + loader (Layer 1a)

**Files:**
- Create: `grid_resilience/data/seed/ercot_large_load_monthly.csv`
- Create: `grid_resilience/data/ercot_large_load_data.py`
- Test: `tests/data/test_ercot_large_load_data.py` (new)

**Why a seed CSV, not a live fetcher:** ERCOT's monthly "Large Load Interconnection Status Update" decks (confirmed individual PDFs back to 2024-03) render their MW figures as chart labels inside slide images, not as machine-readable tables or a downloadable dataset — confirmed during design: ERCOT's own Large Load Integration page states there is no public downloadable per-request dataset. A `pd.read_excel`-style automated parser (as used for PJM in Task 1) isn't available here. This module instead loads a checked-in, source-cited CSV that's extended by hand each time a new monthly deck is read (Task 5 seeds it further) — this is slower to maintain than Task 1 but honest about what ERCOT actually publishes.

**Interfaces:**
- Produces: `load_ercot_large_load_seed(path: Path | None = None) -> pd.DataFrame` — columns `[snapshot_date, scope, tsp, standalone_mw, co_located_mw, total_mw, source_url]`.

- [ ] **Step 1: Create the seed CSV with the 11 confirmed system-wide monthly totals**

Create `grid_resilience/data/seed/ercot_large_load_monthly.csv` (values read directly off ERCOT's "Large Load Queue – Past 12 Months" chart, `March-TAC-Report.pdf` page 2, confirmed live during design):

```csv
snapshot_date,scope,tsp,standalone_mw,co_located_mw,total_mw,source_url
2025-05-01,system_wide,,144571,25044,169615,https://www.ercot.com/files/docs/2026/03/12/March-TAC-Report.pdf
2025-06-01,system_wide,,147784,25043,172828,https://www.ercot.com/files/docs/2026/03/12/March-TAC-Report.pdf
2025-07-01,system_wide,,155964,26443,182408,https://www.ercot.com/files/docs/2026/03/12/March-TAC-Report.pdf
2025-08-01,system_wide,,163851,25477,189328,https://www.ercot.com/files/docs/2026/03/12/March-TAC-Report.pdf
2025-09-01,system_wide,,168009,27413,195423,https://www.ercot.com/files/docs/2026/03/12/March-TAC-Report.pdf
2025-10-01,system_wide,,180187,31907,212095,https://www.ercot.com/files/docs/2026/03/12/March-TAC-Report.pdf
2025-11-01,system_wide,,198649,32150,230799,https://www.ercot.com/files/docs/2026/03/12/March-TAC-Report.pdf
2025-12-01,system_wide,,202475,34714,237189,https://www.ercot.com/files/docs/2026/03/12/March-TAC-Report.pdf
2026-01-01,system_wide,,204448,28677,233125,https://www.ercot.com/files/docs/2026/03/12/March-TAC-Report.pdf
2026-02-01,system_wide,,213105,28643,241748,https://www.ercot.com/files/docs/2026/03/12/March-TAC-Report.pdf
2026-03-01,system_wide,,210114,28515,238629,https://www.ercot.com/files/docs/2026/03/12/March-TAC-Report.pdf
```

Note: `tsp` is blank for these rows — this chart is ERCOT-system-wide, not broken out by TSP. Task 5 adds TSP-level rows (`scope=tsp_level`, `tsp` populated) from individual monthly decks, which is what the actual per-ticker (AEP, CNP) signal in Task 6 will use — the system-wide rows above are context/normalization only.

- [ ] **Step 2: Write the failing test**

Create `tests/data/test_ercot_large_load_data.py`:

```python
import pandas as pd
from grid_resilience.data.ercot_large_load_data import load_ercot_large_load_seed


def test_load_ercot_large_load_seed_reads_system_wide_rows():
    df = load_ercot_large_load_seed()
    assert set(df.columns) == {"snapshot_date", "scope", "tsp", "standalone_mw", "co_located_mw", "total_mw", "source_url"}
    system_wide = df[df["scope"] == "system_wide"]
    assert len(system_wide) == 11
    assert pd.api.types.is_datetime64_any_dtype(df["snapshot_date"])
    row = system_wide[system_wide["snapshot_date"] == pd.Timestamp("2026-03-01")].iloc[0]
    assert row["total_mw"] == 238629
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/data/test_ercot_large_load_data.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 4: Implement the loader**

Create `grid_resilience/data/ercot_large_load_data.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

_SEED_DIR = Path(__file__).parent / "seed"
_DEFAULT_SEED_PATH = _SEED_DIR / "ercot_large_load_monthly.csv"


def load_ercot_large_load_seed(path: Path | None = None) -> pd.DataFrame:
    """Load the checked-in ERCOT large-load seed CSV (see Tasks 4-5 for why
    this is a maintained seed file, not a live fetcher)."""
    df = pd.read_csv(path or _DEFAULT_SEED_PATH)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/data/test_ercot_large_load_data.py -v`
Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add grid_resilience/data/seed/ercot_large_load_monthly.csv grid_resilience/data/ercot_large_load_data.py tests/data/test_ercot_large_load_data.py
git commit -m "feat: add ERCOT large-load seed data + loader (Layer 1a)"
```

---

### Task 5: ERCOT TSP-level historical data collection (Layer 1b)

**Files:**
- Modify: `grid_resilience/data/seed/ercot_large_load_monthly.csv`

**Interfaces:** none new — extends the seed file Task 4 created with `scope=tsp_level` rows for AEP and CenterPoint specifically.

This is a data-collection task, not a code task. ERCOT's monthly decks each include a "Large Load Project Distribution - TSP" chart (cumulative MW by TSP, by queue status) — but only the *current* snapshot's cumulative view was read during design (from `March-TAC-Report.pdf`, showing AEP and Oncor bars without precise per-status values transcribed). Building a real monthly AEP/CenterPoint time series requires reading that specific chart from each individual monthly deck.

- [ ] **Step 1: Fetch and read the TSP-distribution chart from each confirmed historical deck**

URLs confirmed to exist during design (fetch each with WebFetch, then use the Read tool's PDF page-image rendering on the saved binary to view the "Large Load Project Distribution - TSP" chart — WebFetch's text extraction fails on these chart-image PDFs, same as observed for `March-TAC-Report.pdf` during design):

```
https://www.ercot.com/files/docs/2024/03/26/LLI%20Queue%20Status%20Update%20-%202024-4-1.pdf
https://www.ercot.com/files/docs/2024/06/02/LLI%20Queue%20Status%20Update%20-%202024-6-3.pdf
https://www.ercot.com/files/docs/2024/07/05/LLI%20Queue%20Status%20Update%20-%202024-7-8.pdf
https://www.ercot.com/files/docs/2024/08/07/04-lli-queue-status-update-2024-8-5.pdf
```

Search `ercot.com Large Load Interconnection Status Update` (or check the document archive linked from `https://www.ercot.com/services/rq/large-load-integration`) for the remaining months between August 2024 and March 2026 to fill the gap — transcribe every month found, don't stop at just these four.

- [ ] **Step 2: Transcribe AEP and CenterPoint MW figures into the seed CSV**

For each deck read, append one row per TSP with the exact MW figures shown on the "Large Load Project Distribution - TSP" chart (read the bar-length labels precisely — do not estimate from bar length alone; if a chart doesn't print an exact number for AEP/CenterPoint, skip that month for that TSP rather than guessing):

```csv
2024-08-01,tsp_level,AEP,<value>,,<value>,https://www.ercot.com/files/docs/2024/08/07/04-lli-queue-status-update-2024-8-5.pdf
2024-08-01,tsp_level,CenterPoint,<value>,,<value>,https://www.ercot.com/files/docs/2024/08/07/04-lli-queue-status-update-2024-8-5.pdf
```

(`<value>` placeholders above are for this plan step's instructions only — the executor must fill real transcribed numbers, never leave a placeholder in the committed CSV.)

- [ ] **Step 3: Validate the extended seed file**

Run:
```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('grid_resilience/data/seed/ercot_large_load_monthly.csv')
tsp = df[df['scope'] == 'tsp_level']
print(tsp['tsp'].value_counts())
print(tsp.sort_values('snapshot_date'))
"
```
Expected: at least 4 months of AEP and CenterPoint rows, values monotonically non-decreasing within each TSP (a large load queue only grows or a project gets withdrawn — a sharp unexplained drop signals a transcription error, not real data; re-check the source chart if so).

- [ ] **Step 4: Commit**

```bash
git add grid_resilience/data/seed/ercot_large_load_monthly.csv
git commit -m "data: add ERCOT TSP-level historical rows for AEP/CenterPoint"
```

---

### Task 6: ERCOT level+momentum signal (Layer 1c)

**Files:**
- Modify: `grid_resilience/data/ercot_large_load_data.py`
- Modify: `grid_resilience/data/utility_node_map.py`
- Test: `tests/data/test_ercot_large_load_data.py`

**Interfaces:**
- Consumes: `load_ercot_large_load_seed()` output (Task 4-5)
- Produces: `compute_ercot_signal(tickers: list[str], tsp_map: dict, ercot_history: pd.DataFrame, as_of_dates: list[pd.Timestamp]) -> pd.DataFrame` — same `dates x tickers` shape as `compute_dc_load_signal()`. Level only in v1 (no momentum parameter) — there's no zone-size denominator yet to normalize a momentum ratio against, and an unused parameter would just be YAGNI; add momentum once Task 5's TSP-level history is long enough to support it.

ERCOT's queue "TSP" categories (Oncor, AEP, CenterPoint, ...) don't line up with `TICKER_NODE_MAP`'s `iso`/`load_zones` fields (AEP's ERCOT sub-entity, AEP Texas, isn't reflected there — only its PJM `load_zones` are). Add a small, separate mapping rather than overloading the PJM-oriented fields.

- [ ] **Step 1: Add `ERCOT_TSP_MAP` to `utility_node_map.py`**

Add near the top of `grid_resilience/data/utility_node_map.py`, after `TICKER_NODE_MAP`:

```python
# ERCOT "Large Load" queue TSP categories don't match TICKER_NODE_MAP's
# iso/load_zones fields (e.g. AEP Texas, an ERCOT TDU, isn't reflected in
# AEP's PJM-oriented load_zones). Kept separate rather than overloading
# those fields. Only entries independently verified against ERCOT's public
# TSP list belong here — see Task 6 of the DC-demand-exposure plan.
ERCOT_TSP_MAP = {
    "AEP": "AEP",
    "CNP": "CenterPoint",
}
```

- [ ] **Step 2: Write the failing test**

Add to `tests/data/test_ercot_large_load_data.py`:

```python
def test_compute_ercot_signal_level_and_momentum_for_mapped_ticker():
    import pandas as pd
    from grid_resilience.data.ercot_large_load_data import compute_ercot_signal

    history = pd.DataFrame([
        {"snapshot_date": pd.Timestamp("2025-06-01"), "scope": "tsp_level", "tsp": "AEP", "total_mw": 30000, "standalone_mw": 30000, "co_located_mw": 0, "source_url": "x"},
        {"snapshot_date": pd.Timestamp("2025-08-01"), "scope": "tsp_level", "tsp": "AEP", "total_mw": 34000, "standalone_mw": 34000, "co_located_mw": 0, "source_url": "x"},
    ])
    tsp_map = {"AEP": "AEP"}

    result = compute_ercot_signal(
        tickers=["AEP"], tsp_map=tsp_map, ercot_history=history,
        as_of_dates=[pd.Timestamp("2025-09-01")],
    )
    assert result.loc[pd.Timestamp("2025-09-01"), "AEP"] == 34000.0  # latest known level, no denominator yet in v1


def test_compute_ercot_signal_unmapped_ticker_is_nan():
    import pandas as pd
    from grid_resilience.data.ercot_large_load_data import compute_ercot_signal

    history = pd.DataFrame(columns=["snapshot_date", "scope", "tsp", "total_mw", "standalone_mw", "co_located_mw", "source_url"])
    result = compute_ercot_signal(
        tickers=["FE"], tsp_map={"AEP": "AEP"}, ercot_history=history,
        as_of_dates=[pd.Timestamp("2025-09-01")],
    )
    assert pd.isna(result.loc[pd.Timestamp("2025-09-01"), "FE"])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/test_ercot_large_load_data.py -v -k compute_ercot_signal`
Expected: FAIL — function doesn't exist

- [ ] **Step 4: Implement `compute_ercot_signal`**

Add to `grid_resilience/data/ercot_large_load_data.py`:

```python
def compute_ercot_signal(
    tickers: list[str],
    tsp_map: dict[str, str],
    ercot_history: pd.DataFrame,
    as_of_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    Raw (pre-z-score) ERCOT large-load score per ticker per date: latest
    known TSP-level total_mw as of `as_of` (level only in v1 — no
    zone-size denominator exists yet for ERCOT TSPs, unlike the PJM
    signal's zone_size()). Only tickers in `tsp_map` get real values.
    """
    tsp_rows = ercot_history[ercot_history["scope"] == "tsp_level"]
    rows = {}
    for date in as_of_dates:
        row = {}
        for ticker in tickers:
            tsp = tsp_map.get(ticker)
            if tsp is None:
                row[ticker] = float("nan")
                continue
            known = tsp_rows[(tsp_rows["tsp"] == tsp) & (tsp_rows["snapshot_date"] <= date)]
            row[ticker] = float(known.sort_values("snapshot_date")["total_mw"].iloc[-1]) if not known.empty else float("nan")
        rows[date] = row
    return pd.DataFrame.from_dict(rows, orient="index")[tickers]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_ercot_large_load_data.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add grid_resilience/data/ercot_large_load_data.py grid_resilience/data/utility_node_map.py tests/data/test_ercot_large_load_data.py
git commit -m "feat: add ERCOT large-load level signal (Layer 1c)"
```

---

### Task 7: Hyperscaler PPA deal seed table (Layer 3a)

**Files:**
- Create: `grid_resilience/data/seed/hyperscaler_deals.csv`

**Interfaces:** none new — pure data, consumed by Task 8.

Every row's `first_disclosure_date` is the earliest public disclosure (press release/8-K/earnings call), per the spec's point-in-time rule — never an ISO filing date. `sign` is `+1` for MW that adds to a ticker's disclosed DC exposure (announcement, expansion) and `-1` for a regulatory setback that removes it until resolved (Task 8 sums `sign * mw` cumulatively).

- [ ] **Step 1: Create the seed CSV with verified deals (confirmed live during design, 2026-08-26)**

```csv
ticker,counterparty,mw,sign,event_type,first_disclosure_date,date_confidence,source_url,note
CEG,Microsoft,835,1,announcement,2024-09-20,confirmed_primary_source,https://www.sec.gov/Archives/edgar/data/1868275/000186827524000058/ceg-202409208kexh991.htm,"Three Mile Island Unit 1 restart (Crane Clean Energy Center); 20-year PPA; SEC 8-K exhibit"
TLN,Amazon,1920,1,announcement,2024-03-01,needs_verification,https://www.datacenterdynamics.com/en/news/aws-acquires-talens-nuclear-data-center-campus-in-pennsylvania/,"Susquehanna nuclear campus data-center co-location + $650M campus sale; exact day not yet confirmed against Talen's own 8-K/10-Q - verify before use"
TLN,Amazon,1920,-1,regulatory_setback,2024-11-04,confirmed_primary_source,https://www.cnbc.com/2024/11/04/tech-partnerships-with-power-companies-for-ai-in-doubt-after-ferc-order.html,"FERC rejected the amended interconnection service agreement for the co-located arrangement"
TLN,Amazon,1920,1,resolution,2025-06-11,confirmed_primary_source,https://ir.talenenergy.com/news-releases/news-release-details/talen-energy-expands-nuclear-energy-relationship-amazon,"Talen Energy Expands Nuclear Energy Relationship with Amazon - press release"
CEG,Meta,1121,1,announcement,2025-06-03,needs_verification,https://www.constellationenergy.com/newsroom.html,"Clinton Clean Energy Center; 20-year PPA; year inferred as 2025 from context (preceded Meta's 2026-01-09 Vistra deal, followed Meta's Dec-2024 RFP) - confirm exact source before use"
VST,Meta,2176,1,announcement,2026-01-09,confirmed_primary_source,https://about.fb.com/news/2026/01/meta-nuclear-energy-projects-power-american-ai-leadership/,"20-year PPAs for Perry/Davis-Besse plants (OH) plus Beaver Valley (PA) uprate purchase; part of Meta's up-to-6.6GW multi-party nuclear commitment"
```

- [ ] **Step 2: Verify the two `needs_verification` rows before they're used in any live signal**

For the TLN/Amazon initial announcement date and the CEG/Meta Clinton deal date: search SEC EDGAR full-text search (`efts.sec.gov`) for the relevant 8-K, or the company's own investor-relations press release archive, and update `date_confidence` to `confirmed_primary_source` with the exact date once verified. Do not skip this — Task 8's signal computation excludes `needs_verification` rows by default (see Task 8 Step 4), so leaving them unverified silently drops real signal rather than risking a wrong date.

- [ ] **Step 3: Commit**

```bash
git add grid_resilience/data/seed/hyperscaler_deals.csv
git commit -m "data: add verified hyperscaler PPA deal seed table (Layer 3a)"
```

---

### Task 8: Hyperscaler cumulative-MW signal (Layer 3b)

**Files:**
- Create: `grid_resilience/data/hyperscaler_deals.py`
- Test: `tests/data/test_hyperscaler_deals.py` (new)

**Interfaces:**
- Produces:
  - `load_hyperscaler_deals(path: Path | None = None, include_unverified: bool = False) -> pd.DataFrame`
  - `compute_hyperscaler_signal(tickers: list[str], deals: pd.DataFrame, as_of_dates: list[pd.Timestamp]) -> pd.DataFrame` — same `dates x tickers` shape as the other layers; cumulative `sign * mw` per ticker as of each date.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_hyperscaler_deals.py`:

```python
import pandas as pd


def test_load_hyperscaler_deals_excludes_unverified_by_default(tmp_path):
    from grid_resilience.data.hyperscaler_deals import load_hyperscaler_deals
    csv_path = tmp_path / "deals.csv"
    csv_path.write_text(
        "ticker,counterparty,mw,sign,event_type,first_disclosure_date,date_confidence,source_url,note\n"
        "CEG,Microsoft,835,1,announcement,2024-09-20,confirmed_primary_source,x,\n"
        "TLN,Amazon,1920,1,announcement,2024-03-01,needs_verification,x,\n"
    )
    result = load_hyperscaler_deals(path=csv_path)
    assert list(result["ticker"]) == ["CEG"]

    result_all = load_hyperscaler_deals(path=csv_path, include_unverified=True)
    assert set(result_all["ticker"]) == {"CEG", "TLN"}


def test_compute_hyperscaler_signal_cumulative_sum_with_setback_and_resolution():
    from grid_resilience.data.hyperscaler_deals import compute_hyperscaler_signal
    deals = pd.DataFrame([
        {"ticker": "TLN", "mw": 1920, "sign": 1, "first_disclosure_date": pd.Timestamp("2024-03-01")},
        {"ticker": "TLN", "mw": 1920, "sign": -1, "first_disclosure_date": pd.Timestamp("2024-11-04")},
        {"ticker": "TLN", "mw": 1920, "sign": 1, "first_disclosure_date": pd.Timestamp("2025-06-11")},
    ])
    result = compute_hyperscaler_signal(
        tickers=["TLN"], deals=deals,
        as_of_dates=[pd.Timestamp("2024-06-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")],
    )
    assert result.loc[pd.Timestamp("2024-06-01"), "TLN"] == 1920
    assert result.loc[pd.Timestamp("2025-01-01"), "TLN"] == 0
    assert result.loc[pd.Timestamp("2025-12-31"), "TLN"] == 1920


def test_compute_hyperscaler_signal_uncovered_ticker_is_nan():
    from grid_resilience.data.hyperscaler_deals import compute_hyperscaler_signal
    deals = pd.DataFrame(columns=["ticker", "mw", "sign", "first_disclosure_date"])
    result = compute_hyperscaler_signal(tickers=["FE"], deals=deals, as_of_dates=[pd.Timestamp("2025-01-01")])
    assert pd.isna(result.loc[pd.Timestamp("2025-01-01"), "FE"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/test_hyperscaler_deals.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement the module**

Create `grid_resilience/data/hyperscaler_deals.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

_SEED_DIR = Path(__file__).parent / "seed"
_DEFAULT_SEED_PATH = _SEED_DIR / "hyperscaler_deals.csv"


def load_hyperscaler_deals(path: Path | None = None, include_unverified: bool = False) -> pd.DataFrame:
    """
    Load the curated hyperscaler-PPA seed table. Excludes
    date_confidence='needs_verification' rows by default — see Task 7
    Step 2; pass include_unverified=True only for exploratory analysis,
    never for a result reported as final.
    """
    df = pd.read_csv(path or _DEFAULT_SEED_PATH)
    df["first_disclosure_date"] = pd.to_datetime(df["first_disclosure_date"])
    if not include_unverified:
        df = df[df["date_confidence"] == "confirmed_primary_source"]
    return df.reset_index(drop=True)


def compute_hyperscaler_signal(
    tickers: list[str],
    deals: pd.DataFrame,
    as_of_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    Raw (pre-z-score) cumulative disclosed DC-linked MW under contract per
    ticker, as of each date: running sum of sign*mw across all rows up to
    that date. `event_type` is a descriptive label only, not read by this
    function — a `regulatory_setback` row's sign=-1 subtracts its own mw,
    but nothing here assumes later rows "resolve" it or that adjacent rows
    for the same ticker form a clean +/-/+ cycle of one MW figure. Real
    deal chains can be several distinct, differently-sized instruments in
    sequence (see grid_resilience/data/seed/hyperscaler_deals.csv's TLN
    rows) — this function sums whatever `sign*mw` values it's given, and
    it's each row's own accuracy (not this function) that has to earn that.
    """
    rows = {}
    for date in as_of_dates:
        row = {}
        for ticker in tickers:
            ticker_deals = deals[(deals["ticker"] == ticker) & (deals["first_disclosure_date"] <= date)]
            if ticker_deals.empty and (deals.empty or ticker not in deals["ticker"].unique()):
                row[ticker] = float("nan")
                continue
            row[ticker] = float((ticker_deals["sign"] * ticker_deals["mw"]).sum())
        rows[date] = row
    return pd.DataFrame.from_dict(rows, orient="index")[tickers]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_hyperscaler_deals.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add grid_resilience/data/hyperscaler_deals.py tests/data/test_hyperscaler_deals.py
git commit -m "feat: add hyperscaler cumulative-MW signal (Layer 3b)"
```

---

### Task 9: Signal combination

**Files:**
- Create: `grid_resilience/data/dc_demand_combined.py`
- Test: `tests/data/test_dc_demand_combined.py` (new)

**Interfaces:**
- Consumes: `compute_dc_load_signal()` (existing), `compute_ercot_signal()` (Task 6), `compute_hyperscaler_signal()` (Task 8) — all `dates x tickers` DataFrames, disjoint ticker coverage in v1.
- Produces: `combine_dc_demand_layers(pjm_history: pd.DataFrame, ercot_history: pd.DataFrame, hyperscaler_history: pd.DataFrame) -> pd.DataFrame` — one merged `dates x tickers` DataFrame, each column z-scored within its own source layer before merging.

Reuses the subpopulation-z-score discipline `fill_with_icr()` already established — concatenating raw values across layers before a single z-score is the exact bug already found and fixed once in `dc_load_data.py` (see spec's Signal Combination section).

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_dc_demand_combined.py`:

```python
import pandas as pd
import numpy as np


def test_combine_dc_demand_layers_zscores_each_layer_separately():
    from grid_resilience.data.dc_demand_combined import combine_dc_demand_layers

    dates = [pd.Timestamp("2025-12-31")]
    pjm = pd.DataFrame({"AEP": [10.0], "FE": [20.0]}, index=dates)       # raw scale ~0-100s
    ercot = pd.DataFrame({"CNP": [34000.0]}, index=dates)                # raw scale ~10,000s
    hyperscaler = pd.DataFrame({"CEG": [835.0], "TLN": [1920.0]}, index=dates)

    result = combine_dc_demand_layers(pjm, ercot, hyperscaler)

    assert set(result.columns) == {"AEP", "FE", "CNP", "CEG", "TLN"}
    # single-column-per-layer or single-value populations z-score to 0 (no within-layer spread)
    assert result.loc[dates[0], "CNP"] == 0.0
    # two-ticker PJM population: AEP (lower) should be negative, FE positive
    assert result.loc[dates[0], "AEP"] < 0 < result.loc[dates[0], "FE"]


def test_combine_dc_demand_layers_ticker_with_no_layer_is_absent():
    from grid_resilience.data.dc_demand_combined import combine_dc_demand_layers
    dates = [pd.Timestamp("2025-12-31")]
    pjm = pd.DataFrame({"AEP": [10.0]}, index=dates)
    ercot = pd.DataFrame(index=dates)
    hyperscaler = pd.DataFrame(index=dates)
    result = combine_dc_demand_layers(pjm, ercot, hyperscaler)
    assert "PCG" not in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/test_dc_demand_combined.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement the module**

Create `grid_resilience/data/dc_demand_combined.py`:

```python
from __future__ import annotations

import pandas as pd


def _zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row across this layer's own tickers (its subpopulation),
    never mixed with another layer's raw scale before scoring."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0).replace(0, pd.NA)
    return df.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


def combine_dc_demand_layers(
    pjm_history: pd.DataFrame,
    ercot_history: pd.DataFrame,
    hyperscaler_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the three DC-demand layers into one dates x tickers score.
    Each layer is z-scored within its own covered tickers before merging —
    concatenating raw values across layers before a single z-score is the
    bug already found and fixed once in fill_with_icr() (see spec).
    """
    layers = [df for df in (pjm_history, ercot_history, hyperscaler_history) if not df.empty and not df.columns.empty]
    if not layers:
        return pd.DataFrame()
    zscored = [_zscore_columns(df) for df in layers]
    return pd.concat(zscored, axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/data/test_dc_demand_combined.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add grid_resilience/data/dc_demand_combined.py tests/data/test_dc_demand_combined.py
git commit -m "feat: add per-layer z-scored DC-demand signal combination"
```

---

### Task 10: Wire into `main.py`

**Files:**
- Modify: `grid_resilience/config.py`
- Modify: `grid_resilience/main.py`

**Interfaces:**
- Produces: `--regulated-signal dc-multi` CLI choice alongside existing `icr`/`dc-queue`.

- [ ] **Step 1: Widen the config type**

In `grid_resilience/config.py`, change:
```python
REGULATED_SIGNAL: Literal["icr", "dc_queue"] = "dc_queue"
```
to:
```python
REGULATED_SIGNAL: Literal["icr", "dc_queue", "dc_multi"] = "dc_queue"
```

- [ ] **Step 2: Add the imports**

In `grid_resilience/main.py`, add alongside the existing `dc_load_data` import:
```python
from grid_resilience.data.pjm_large_load_data import fetch_large_load_adjustment, cross_check_dc_signal
from grid_resilience.data.ercot_large_load_data import load_ercot_large_load_seed, compute_ercot_signal
from grid_resilience.data.hyperscaler_deals import load_hyperscaler_deals, compute_hyperscaler_signal
from grid_resilience.data.dc_demand_combined import combine_dc_demand_layers
from grid_resilience.data.utility_node_map import ERCOT_TSP_MAP
```

- [ ] **Step 3: Update the validation check and add the `dc_multi` branch**

Change the validation at `main.py:312`:
```python
    if regulated_signal not in ("icr", "dc_queue", "dc_multi"):
        raise ValueError(
            f"Unrecognized regulated_signal={regulated_signal!r}. "
            "Expected 'icr', 'dc_queue', or 'dc_multi' (note: underscore form, not the "
            "CLI's hyphenated 'dc-multi')."
        )
```

Add a new `elif` branch after the existing `if regulated_signal == "dc_queue":` block (main.py:329-360), at the same indentation level:
```python
        elif regulated_signal == "dc_multi":
            queue = fetch_interconnection_queue()
            zonal_load_start = (pd.Timestamp(start) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
            zonal_load = fetch_zonal_load(zonal_load_start, end)
            pjm_dc_history = compute_dc_load_signal(
                tickers=universe_tickers, node_map=TICKER_NODE_MAP,
                queue=queue, zonal_load=zonal_load, as_of_dates=list(rebalance_dates),
            ) if not (queue.empty or zonal_load.empty) else pd.DataFrame()

            ercot_seed = load_ercot_large_load_seed()
            ercot_history = compute_ercot_signal(
                tickers=universe_tickers, tsp_map=ERCOT_TSP_MAP,
                ercot_history=ercot_seed, as_of_dates=list(rebalance_dates),
            )

            deals = load_hyperscaler_deals()
            hyperscaler_history = compute_hyperscaler_signal(
                tickers=universe_tickers, deals=deals, as_of_dates=list(rebalance_dates),
            )

            combined = combine_dc_demand_layers(pjm_dc_history, ercot_history, hyperscaler_history)
            merged_history = fill_with_icr(combined, icr_history, lag_days=_ICR_LAG_DAYS)
            n_real = int(combined.notna().any().sum()) if not combined.empty else 0
            print(f"  Multi-source DC demand signal: {n_real} tickers with real layer data "
                  f"(PJM/ERCOT/hyperscaler), rest filled from ICR")
            icr_history = merged_history
            regulated_signal_lag_days = 0

            # Layer 2's cross-check is a diagnostic, not blended into the score
            # (see spec) — still run it and write the output so a reviewer can
            # actually see where the generation-queue signal and PJM's own
            # industry tags disagree, rather than importing it unused.
            if not pjm_dc_history.empty:
                large_load = fetch_large_load_adjustment()
                cross_check = cross_check_dc_signal(pjm_dc_history, large_load, TICKER_NODE_MAP)
                cross_check.to_csv(output_dir / "dc_signal_cross_check.csv", index=False)
                n_disagree = int((~cross_check["agrees"]).sum())
                if n_disagree:
                    print(f"  [WARNING] {n_disagree} ticker(s) disagree between the generation-queue "
                          f"DC signal and PJM's own industry tags — see output/dc_signal_cross_check.csv")
```

- [ ] **Step 4: Add the CLI choice**

In `_parse_args()`, change:
```python
    p.add_argument(
        "--regulated-signal",
        dest="regulated_signal",
        default=REGULATED_SIGNAL.replace("_", "-"),
        choices=["icr", "dc-queue"],
        help="Signal used for the regulated path of --arch hard-switch "
             f"(default: {REGULATED_SIGNAL.replace('_', '-')})",
    )
```
to:
```python
    p.add_argument(
        "--regulated-signal",
        dest="regulated_signal",
        default=REGULATED_SIGNAL.replace("_", "-"),
        choices=["icr", "dc-queue", "dc-multi"],
        help="Signal used for the regulated path of --arch hard-switch "
             f"(default: {REGULATED_SIGNAL.replace('_', '-')})",
    )
```

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all passing, zero failures.

- [ ] **Step 6: Manually verify the new path runs end-to-end**

Run:
```bash
source ~/.zshrc
.venv/bin/python -m grid_resilience.main --no-plot --no-zone-gsi --arch hard-switch --regulated-signal dc-multi
```
Expected: completes without errors, prints "Multi-source DC demand signal: N tickers with real layer data" with N around 11 (6 PJM + CNP + up to 4 hyperscaler, fewer if any Task 7 rows are still `needs_verification`), and writes `output/dc_signal_cross_check.csv` (Layer 2's diagnostic — inspect it for any `agrees=False` rows).

- [ ] **Step 7: Commit**

```bash
git add grid_resilience/config.py grid_resilience/main.py
git commit -m "feat: wire multi-source DC demand signal into main.py as --regulated-signal dc-multi"
```

---

### Task 11: Short-sample backtest harness

**Files:**
- Create: `grid_resilience/analysis/__init__.py`
- Create: `grid_resilience/analysis/short_sample_report.py`
- Test: `tests/analysis/test_short_sample_report.py` (new)

**Interfaces:**
- Produces:
  - `window_sensitivity_report(period_returns: pd.Series) -> pd.DataFrame` — one row per trimming variant (`full`, `drop_first`, `drop_last`, `drop_first_and_last`) with `n_periods` and `mean_return` columns.
  - `sample_size_caveat(n_periods: int, minimum_trusted: int = 12) -> str` — a plain-English caveat string when `n_periods < minimum_trusted`, empty string otherwise.

Per the spec's validation-approach section: never report a point estimate without `n` and a sensitivity check next to it, given this project's history with the PEG-claim and peer-group-Sharpe retractions.

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_short_sample_report.py` (create `tests/analysis/__init__.py` if the `tests/` tree needs it — check `tests/data/__init__.py` first for the existing convention):

```python
import pandas as pd
from grid_resilience.analysis.short_sample_report import window_sensitivity_report, sample_size_caveat


def test_window_sensitivity_report_drops_first_and_last():
    returns = pd.Series([0.10, 0.02, 0.03, 0.04, -0.20], index=pd.date_range("2025-01-01", periods=5, freq="ME"))
    result = window_sensitivity_report(returns)
    full_row = result[result["variant"] == "full"].iloc[0]
    assert full_row["n_periods"] == 5
    drop_first = result[result["variant"] == "drop_first"].iloc[0]
    assert drop_first["n_periods"] == 4
    assert abs(drop_first["mean_return"] - pd.Series([0.02, 0.03, 0.04, -0.20]).mean()) < 1e-9
    drop_both = result[result["variant"] == "drop_first_and_last"].iloc[0]
    assert drop_both["n_periods"] == 3


def test_sample_size_caveat_below_threshold():
    msg = sample_size_caveat(n_periods=5, minimum_trusted=12)
    assert "5" in msg and "12" in msg


def test_sample_size_caveat_above_threshold_is_empty():
    assert sample_size_caveat(n_periods=24, minimum_trusted=12) == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/analysis/test_short_sample_report.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement the module**

Create `grid_resilience/analysis/__init__.py` (empty).

Create `grid_resilience/analysis/short_sample_report.py`:

```python
from __future__ import annotations

import pandas as pd


def window_sensitivity_report(period_returns: pd.Series) -> pd.DataFrame:
    """
    Report the point estimate under the full sample and under three
    trimmed variants, so a short-sample result's stability to the exact
    start/end date is visible next to the headline number, not hidden
    behind it.
    """
    variants = {
        "full": period_returns,
        "drop_first": period_returns.iloc[1:],
        "drop_last": period_returns.iloc[:-1],
        "drop_first_and_last": period_returns.iloc[1:-1],
    }
    rows = [
        {"variant": name, "n_periods": len(series), "mean_return": series.mean() if len(series) else float("nan")}
        for name, series in variants.items()
    ]
    return pd.DataFrame(rows)


def sample_size_caveat(n_periods: int, minimum_trusted: int = 12) -> str:
    """Plain-English caveat for reporting alongside any short-sample result."""
    if n_periods >= minimum_trusted:
        return ""
    return (
        f"Only {n_periods} independent period(s) observed (below the "
        f"{minimum_trusted}-period bar treated as minimally trustworthy here) — "
        "treat this result as directional, not confirmed."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/analysis/test_short_sample_report.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add grid_resilience/analysis/ tests/analysis/
git commit -m "feat: add short-sample backtest window-sensitivity report"
```

---

### Task 12: Record results

**Files:**
- Create: `docs/compact_<today>-dc-multi-source-signal-results.md` (check `docs/` for the latest compact-doc filename first and follow its pattern — see `docs/compact_2026-08-13-dc-load-signal-results.md` as the closest structural precedent)

**Interfaces:** none — documentation task.

- [ ] **Step 1: Run the `dc-multi` backtest and the window-sensitivity report on its output**

Run the full pipeline (Task 10 Step 6), then feed the resulting per-period returns into `window_sensitivity_report()` and `sample_size_caveat()` from Task 11. Record: n_periods per layer (PJM cross-sectional, ERCOT ~monthly, hyperscaler event-driven), point estimates, and how much each moves under the drop-first/drop-last variants.

- [ ] **Step 2: Write the session summary**

Follow `docs/compact_2026-08-13-dc-load-signal-results.md`'s structure (TL;DR, results table, qualitative check, open items). Explicitly state, per ticker-coverage layer: how many tickers got real data (target ~11), whether the two `needs_verification` Task 7 rows were resolved, and whether the result clears the spec's "material, not just sign-flip" bar — if it doesn't, say so as plainly as the DC-load and peer-group compact docs did.

- [ ] **Step 3: Commit**

```bash
git add docs/compact_*.md
git commit -m "docs: record multi-source DC demand exposure signal backtest results"
```
