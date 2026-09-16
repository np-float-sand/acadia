# Deliverable Manifest — Screened Electrification Engagement

What ships to the Client vs. what stays internal research history, per SOW Phase 3 item 4.

## Ships

- `electrification_strategy/` (the full package)
- `electrification_stress_beta/` (national GSI blend + gate test + `gate_result.json`)
- `grid_resilience/signals/grid_stress_index.py`, `conditional_beta.py`, `stress_events.py`,
  `data/grid_data.py`, `data/cache_utils.py`, `data/utility_node_map.py`, `data/ercot_auth.py`,
  `data/ercot_bulk.py`, `config.py` — reused unmodified by `electrification_stress_beta/`.
  Traced to a fixed point past the module-level imports: `cache_utils.py` (parquet cache I/O)
  and `utility_node_map.py` (via `stress_events.py`'s `STRESS_EVENTS` table) are needed at
  import time; `ercot_auth.py`/`ercot_bulk.py` are deferred imports inside `grid_data.py`'s
  `fetch_lmp()` (ERCOT token auth, historical backfill) needed at call time; `config.py` backs
  all of the above. These five were missing from an earlier draft of this manifest that only
  listed four modules — verified by importing every module in both shipped packages fresh.
- `grid_equipment_basket/basket.py`, `config.py`, `backtest.py`, `hedges.py`, `value_chain.py`,
  `margin_data.py`, `backlog_data.py`, `data/prices.py` — the traced transitive-import closure
  actually reused by `electrification_strategy/` (`backtest.py`, `universe.py`, `__main__.py`)
  and `electrification_stress_beta/` (`gate_test.py`). This is NOT the
  whole `grid_equipment_basket/` package (see Stays internal) — an earlier draft of this
  manifest incorrectly listed the entire package as internal-only, which would have broken
  `electrification_strategy` for the Client. `data/prices.py` itself imports
  `grid_resilience.data.cache_utils` (listed above), so that module ships too.
- `tests/electrification_strategy/`, `tests/electrification_stress_beta/`
- `docs/electrification-strategy-proposal-v6.md` (+ .html/.pdf) — the findings report
- `docs/electrification-stress-beta-gate-results.md`
- `docs/electrification-stress-sensitivity-response.md`
- `docs/electrification-reit-breadth-probe-results.md`
- `docs/technical-reference.md`
- `docs/runbook.md`
- `docs/scope-of-work-v2-screened-electrification.md` (+ .html/.pdf)
- `pyproject.toml`, `poetry.lock`

## Stays internal

- `research/` (all ~29 tracked signal-search attempts — negative-result research history, not
  client deliverables)
- `docs/RESEARCH-LOG.md`, plus every `docs/*-results.md` / `docs/handoff*.md` / `docs/compact*.md`
  file **not explicitly listed above under Ships** (internal research trail; the v6 report and
  the two results docs named under Ships — the stress-beta gate and the REIT-breadth probe —
  are the client-facing distillation)
- `output_*/` directories (scratch backtest output; the Client re-generates these via the runbook)
- `grid_resilience/` beyond the nine reused modules listed above (the rest of that
  package belongs to the original, unshipped Grid Resilience strategy)
- `grid_equipment_basket/` beyond the eight reused modules listed above — the rest is that
  package's own thematic-basket research/strategy code, not reused by Screened Electrification:
  `__main__.py`, `backlog_forward_check.py`, `capex_guidance_signal.py`, `capex_signal.py`,
  `ftr_signal.py`, `grid_regime.py`, `overlay.py`, `data/category_demand.py`
- `backlog_factor/`, `dc_demand_factor/`, `transmission_rate_base/`, and other sibling probe
  packages not part of Screened Electrification
- `.venv/`, `.superpowers/`, `.claude/`, `scratchpad/`, `pitch/`

## Release step

Once the Client confirms this manifest, proceed to Task 15 (`git tag`).
