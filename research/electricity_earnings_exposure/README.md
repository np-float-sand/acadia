# electricity_earnings_exposure/

Two pre-registered spikes testing the "electricity as a company-specific economic input
→ equity signal" pivot (external "dislocation alpha" / "Electricity Intelligence for
Equity Alpha" critique, 2026-09-07). **Both FAILED.** Full write-up:
`docs/electricity-earnings-exposure-probe-results.md`; ledger + session block in
`docs/RESEARCH-LOG.md`.

Throwaway. Re-run to regenerate. Needs `.env` (EIA_API_KEY) sourced; run via `poetry run`.

| file | what |
|---|---|
| `edgar.py` | SEC EDGAR XBRL `companyconcept` fetcher — discrete-quarter extraction for flow concepts (Revenues / COGS / OperatingIncomeLoss / GrossProfit) with FY−9M Q4 derivation. Reusable. Caches JSON under `_cache/`. |
| `probe_a.py` | Merchant-generation margin retrodiction (VST/NRG/CEG/TLN): hand-coded fleet × cached ERCOT/PJM DA hub price − Henry Hub gas vs reported revenue / operating income. FAIL — tracks Δrevenue, not Δoperating-income. |
| `universe.py` | Probe B universe — ~18 power-intensive industrials with hand-coded `{state: weight}` exposure vectors + electricity-intensity constants. |
| `probe_b.py` | Electricity-cost exposure cross-section: `−intensity·%Δ(power price)` signal vs Δmargin (EDGAR), fwd returns (yfinance), FF5+MOM+sector neutralisation (Ken French). `--wholesale` swaps EIA state retail price for ERCOT/PJM DA hub. FAIL — 6-gate tree, only breadth passes. |
| `probe_c_inputcost_hedge.py` | Input-cost hedge screen: beta of 184 all-sector stocks to copper (`HG=F`) / electrical steel (`CLF`) / labour (FRED AHE + staffing basket). Gate 0 FAILED — the supplier book has ~0 beta to all three (passes costs through) and *outperforms* in cost-shock months. No hedge warranted. |
| `probe_b_focus.py` | Confirmatory kill of the one t≈1.96 cell from `probe_b` — 2022–26, ex-crypto, sector-neutral, Newey–West t. Dead: sector-neutral IC t −2.12 (wrong sign), panel-OLS t −0.39. |
| `proposal_v3/` | **Build kit for `docs/electrification-strategy-proposal-v3.html`** (the investment-committee write-up + 9-candidate short-offset analysis). `build_figures.py` → `assemble.py [--deploy --light --pdf]`; `template.html` is the editable source; `light_mode.py` casts it to a light-locked print/PDF doc. See `proposal_v3/README.md`. |
