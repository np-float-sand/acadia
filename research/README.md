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
| `xsec_alpha_probe/` | within-universe selection signals (point-in-time EDGAR revenue/NI/capex, PJM+ERCOT energy/congestion/scarcity exposure, price baselines); winsorized IC + equal/inv-vol/semivol quintile books; reads caches from the main checkout (`ACADIA_DATA_ROOT`) | no signal PASSES; capex_light CLOSE (info in the bottom), rev_accel stable top book; post-hoc composite best long-side selector — `docs/xsec-alpha-probe-results.md` |
| `xsec_factors_ls/` | wide-universe cross-sectional factors, momentum stress test, cointegration pairs, rules-based long/short construction | momentum = 2023-24 burst only; pairs weak; L/S = no durable short |
| `capex_cycle_pair_classifier/` | 4-criterion "customer + 2 of 3" classifier + generalization (EV / nuclear / space / hydrogen / cannabis / genomics), grid/DER spread attribution, capacity-auction & book-to-bill timing, US-industrial-policy L/S, backward winner analysis | classifier = KEEPER (reproduces the book); everything else FAILED — see `docs/RESEARCH-LOG.md` §6 |
| `electricity_earnings_exposure/` | "electricity as a company-specific economic input → equity signal" pivot: merchant-generation margin nowcast (Probe A) + power-cost exposure cross-section over ~18 power-intensive industrials (Probe B). Reusable EDGAR XBRL `companyconcept` helper. | BOTH FAILED pre-registered gates — `docs/electricity-earnings-exposure-probe-results.md` |
