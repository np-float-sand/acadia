# PJM Signal — Problem, Investigation & Proposed Fix

**Branch:** res2  
**Last commit:** 9ae1016 (before pjm fix) through 5a357e5 (ffill fix)  
**Date:** 2026-06-22  
**Status:** Data quality fixed; signal quality problem remains. Next step: per-ticker zone GSI.

---

## TL;DR

Adding PJM to the strategy dropped Sharpe from **0.319 → -1.677**. We fixed the data pipeline (Sharpe recovered to **-0.043**) but PJM signal quality is still bad. Root cause: all 5 PJM tickers share a single system-wide hub GSI — they see identical stress signals, so the only differentiator is their beta to that signal, which clusters tightly and produces low cross-sectional dispersion (score std 0.718 vs ERCOT's 1.292).

---

## What We Already Fixed (Commits on res2)

| Commit | Change |
|--------|--------|
| `c06311f` | PJM API retries: 4→6, initial backoff: 5s→15s, inter-month sleep: 3s→12s |
| `3057eff` | GSI forward-fill up to 10 days for API gap bridging |
| `5a357e5` | Simplified ffill to spec (reverted scope creep) |

These fixes resolved the data quality problem: 18/48 missing LMP months and 29/48 missing load months were caused by rate-limit exhaustion during the initial PJM fetch. All months now cached. No more 429s.

---

## Performance Numbers

| State | Sharpe | Ann Ret | Max DD |
|-------|--------|---------|--------|
| Without PJM (ERCOT/MISO/CAISO/SPP) | **0.315** | 8.03% | -13.66% |
| With PJM — broken data | -1.677 | -5.92% | -22.66% |
| With PJM — fixed data | -0.043 | 3.53% | -22.25% |

Data fix helped but PJM still drags badly.

---

## Root Cause: One GSI for All PJM Tickers

`config.py:ISO_BENCHMARK_NODES["PJM"]` pulls four system hubs:
```python
["WESTERN HUB", "EASTERN HUB", "AEP GEN HUB", "APS GEN HUB"]
```

`build_multi_iso_gsi()` builds **one** GSI DataFrame for PJM. `gsi_for_ticker()` returns `gsi_by_iso["PJM"]["gsi"]` for every PJM ticker — AEP, D, EXC, FE, PPL all see the same stress series.

The only differentiation between PJM tickers is their **stress_beta** (OLS slope of excess_return ~ GSI). PJM is a 13-state RTO — system-wide stress events affect all utilities similarly, so betas cluster. No beta dispersion → no factor score dispersion → broken rankings.

By contrast, ERCOT works because Texas has meaningful within-ISO variation (HB_NORTH vs HB_HOUSTON vs HB_WEST have very different congestion patterns), and the TICKER_NODE_MAP points each ERCOT ticker at its relevant hub.

---

## The Existing Zone Mapping (Already in Code, Not Yet Used)

`grid_resilience/data/utility_node_map.py` already maps each PJM ticker to its home zones:

| Ticker | Company | PJM Zones (nodes) | Notes |
|--------|---------|-------------------|-------|
| AEP | American Electric Power | AEP GEN HUB, AEP-DAYTON HUB | Also has SPP (secondary_iso) |
| EXC | Exelon | PECO, BGE, PEPCO, COMED | 4 zones post-Constellation spinoff |
| PPL | PPL Corp | PPL | PA, some KY (MISO) |
| FE | FirstEnergy | ATSI, JCPL, METED, PENELEC | ATSI most material; MISO seam congestion |
| D | Dominion | EASTERN HUB | Should be DOM zone — EASTERN HUB is a proxy |

These zone assignments are UNUSED by the GSI pipeline. The pipeline only reads `iso` (not `nodes`) from the node map.

---

## Proposed Three-Part Fix

### 1. Per-Ticker Zone GSI (Primary Fix — High Impact, Medium Effort)

Compute a separate GSI for each PJM ticker using its home zone LMPs instead of the shared system hub signal.

**Architecture change:**
- `gsi_by_iso: dict[str, pd.DataFrame]` becomes `gsi_by_iso: dict[str, pd.DataFrame]` with additional keys `"PJM:AEP"`, `"PJM:EXC"`, `"PJM:PPL"`, `"PJM:FE"`, `"PJM:D"` alongside the existing `"PJM"` key
- `gsi_for_ticker()` checks for `f"PJM:{ticker}"` key first, falls back to `"PJM"` for non-PJM tickers
- Existing `"PJM"` key (system hub GSI) stays for event detection / stress calendar
- `conditional_beta.py:compute_stress_betas()` accesses `gsi_by_iso[iso]["gsi"]` — needs to become `gsi_for_ticker(ticker, gsi_by_iso, ticker_iso_map)` call instead of direct dict access

**Data needed:**
- PJM zone-level LMPs (`type=ZONE` from DataMiner 2, not `type=HUB`)
- `_fetch_pjm_lmp_direct()` currently hardcodes `"type": "HUB"` — needs a zone-fetch variant
- New cache files: `pjm_lmp_zone_YYYY-MM.parquet` (separate from hub cache)
- Load data: still uses system-wide PJM load (no per-zone load needed)

**D (Dominion) note:** The `nodes` field says `EASTERN HUB` which is a hub, not a zone. The correct PJM zone is `DOM`. This needs to be updated in `utility_node_map.py`.

### 2. Add PEG (PSEG) — Low Effort, Medium Impact

PSEG (Public Service Enterprise Group) is a major NJ T&D+gen utility, ~$35B market cap, primarily in the PSEG zone within PJM. Currently absent from the universe.

Add to `TICKER_NODE_MAP`:
```python
"PEG": {
    "name": "Public Service Enterprise Group",
    "iso": "PJM",
    "nodes": ["PSEG"],
    "load_zones": ["PSEG"],
    "service_territory": "New Jersey T&D + nuclear generation",
    "notes": "PJM PSEG zone. Mix of T&D and nuclear gen. Sensitive to NJ/NYC corridor congestion.",
}
```

After adding: 6 PJM tickers total (AEP, D, EXC, FE, PPL, PEG) out of ~23 universe names.

### 3. Inter-Zonal Spread as 5th GSI Sub-Signal (Medium Impact, Higher Effort)

Builds on top of Fix 1 (needs per-ticker zone LMPs). Add a 5th component to `build_gsi`:

**Signal:** For a given ticker's zone, compute daily `(zone_lmp - pjm_system_mean_lmp) / pjm_system_std_lmp`. This measures how much the ticker's territory is deviating from the PJM system average on each day.

**Config changes needed:**
```python
GSI_WEIGHTS = {
    "lmp_zscore":        0.35,  # was 0.40
    "congestion_frac":   0.20,  # was 0.25
    "reserve_tightness": 0.20,  # unchanged
    "event_flag":        0.10,  # was 0.15
    "zone_spread":       0.15,  # new
}
```
(Weights need re-validation via grid search after adding signal.)

**Critical notes:**
- Sign of equity response varies by business model: generators benefit from high zone LMP; T&D utilities (EXC, PPL, CNP) are hurt. The `stress_beta` OLS should capture this automatically — no special handling needed.
- AEP spans PJM + SPP; zone spread only covers PJM portion. Accept as known limitation.

---

## Known Issues / Harsh Reviewer Concerns

1. **Zone LMP data not yet fetched.** `_fetch_pjm_lmp_direct` uses `type=HUB` only. The congestion spread for PJM currently reuses hub data (PJM has no entry in `ISO_ZONE_LOCATION_TYPE`). Zone fetch is new API work with new cache files.

2. **D's node mapping is wrong.** `EASTERN HUB` is a system hub, not the DOM zone. Fix: update `utility_node_map.py` to `nodes=["DOM"]` (or verify the PJM API zone name for Dominion territory).

3. **AEP multi-ISO coverage.** ~40% of AEP territory is SPP. Per-ticker zone GSI only captures PJM portion. Document as limitation; SPP signal still flows through system-level ERCOT/MISO/CAISO/SPP GSI.

4. **PJM concentration.** 6 PJM tickers / 23 total = 26% of universe. With 3L/5S portfolio, possible to get 3 PJM longs and 3 PJM shorts, reducing diversification. Monitor post-launch; consider capping per-ISO allocation at 2 names per book if IC doesn't improve.

5. **GSI weights need re-tuning** after adding zone_spread signal. Grid search currently optimized for 4-signal GSI.

---

## Key Files

| File | Relevant Lines | Role |
|------|---------------|------|
| `grid_resilience/config.py:54-60` | `ISO_BENCHMARK_NODES` | PJM system hub list → needs per-ticker zone routing |
| `grid_resilience/config.py:88-93` | `GSI_WEIGHTS` | Add `zone_spread` key for Fix 3 |
| `grid_resilience/signals/grid_stress_index.py:117-154` | `build_multi_iso_gsi`, `gsi_for_ticker` | Core change for per-ticker GSI lookup |
| `grid_resilience/signals/conditional_beta.py:180` | `gsi_series = gsi_by_iso[iso]["gsi"]` | Must change to use `gsi_for_ticker()` |
| `grid_resilience/data/grid_data.py:122-166` | `_fetch_pjm_lmp_direct` | Add `type=ZONE` variant |
| `grid_resilience/data/utility_node_map.py:38-72,151-158` | PJM ticker entries | Fix D's node; add PEG |
| `grid_resilience/main.py:184-214` | `_fetch_iso_data` | Wire per-ticker zone fetch into pipeline |

---

## Next Steps (When Resuming)

1. Invoke `superpowers:brainstorming` to finish the design session (was mid-flow when compact happened)
2. Clarify: is the inter-zonal spread 5th signal scoped to PJM only or all ISOs?
3. Write spec → `docs/superpowers/specs/2026-06-22-pjm-zone-gsi-design.md`
4. Write plan → `docs/superpowers/plans/2026-06-22-pjm-zone-gsi.md`
5. Execute plan via `superpowers:subagent-driven-development`

---

## Test Validation Targets

After implementation, run:
```bash
.venv/bin/python -m grid_resilience.main --no-plot 2>&1 | grep -E "Sharpe|Ann Ret|Max DD|Signal"
```

Success = Sharpe ≥ 0.25 with all ISOs including PJM (baseline without PJM: 0.315).

Also check factor score dispersion by ISO:
```python
fs.groupby('iso')['factor_score'].std()  # PJM target: > 0.90 (currently 0.718)
```
