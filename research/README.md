# research/

Exploratory probe scripts from the AI-power / electrification research line.

**Not production code.** Each script re-fetches data live (yfinance / FRED / EIA / SEC),
runs one analysis, and prints tables. Re-run to regenerate. All verdicts are indexed in
**`docs/RESEARCH-LOG.md`**.

| dir | investigation | verdict |
|---|---|---|
| `va_transmission_probe/` | VA/Dominion SCC transmission-filing demand DB (Deliverable A) + work-type tilt | DB built & kept (`grid_resilience/data/va_transmission_data.py`); tilt FAILED |
| `category_demand_rotation/` | Census-M3 / PPI equipment-category demand → maker rotation | FAILED (no cross-sectional IC in the AI regime) |
| `forward_gas_power/` | Henry Hub / forward power-gas signal; GBM/RF kitchen-sink; IS/OOS split | FAILED (data-blocked for the true long end; proxies show nothing) |
| `xsec_factors_ls/` | wide-universe cross-sectional factors, momentum stress test, cointegration pairs, rules-based long/short construction | momentum = 2023-24 burst only; pairs weak; L/S = no durable short |
| `capex_cycle_pair_classifier/` | 4-criterion "customer + 2 of 3" classifier + generalization (EV / nuclear / space / hydrogen / cannabis / genomics), grid/DER spread attribution, capacity-auction & book-to-bill timing, US-industrial-policy L/S, backward winner analysis | classifier = KEEPER (reproduces the book); everything else FAILED — see `docs/RESEARCH-LOG.md` §6 |
