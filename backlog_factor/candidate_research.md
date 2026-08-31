# Backlog-Surprise Factor — Phase-1 Candidate Verification Log

Spec: `docs/superpowers/specs/2026-08-31-backlog-surprise-factor-design.md` (§2 universe, §9 phased assembly).

## Method

1. **Discovery.** `discover_rpo_filers(DISCOVERY_QUARTERS)` unions every filer of the XBRL concept
   `us-gaap:RevenueRemainingPerformanceObligation` across `CY2021Q4I`…`CY2025Q4I` (SEC `frames` API),
   then keeps only CIKs whose `submissions` SIC starts with a construction / machinery / electrical /
   transportation-equipment / industrial-instrument prefix (`config.IN_SCOPE_SIC_PREFIXES`).
   Result: **207 in-scope filers** (SIC 15:3, 16:15, 17:7, 34:12, 35:50, 36:15, 37:37, 38:68).
2. **Curation (this document).** From those 207, keep a name only if a plain reading says it is an
   **order-driven manufacturer of capital equipment or an infrastructure contractor that discloses a
   genuine forward order backlog** (not subscription / service-contract deferred revenue). Inclusion is
   never decided on stock returns.

**Result: 109 names**, by industry group:
machinery 46 · aerospace_defense 21 · engineering_construction 20 · electrical_equipment 11 ·
semiconductor_equipment 8 · building_products 3. Full list in `config.UNIVERSE`.

## Systematic drops (by category)

| Category | Examples dropped | Reason |
|---|---|---|
| **Medical / dental / surgical / lab-genomics** (most of SIC 38) | MDT, ZBH, STE, BAX, ILMN, TMO, DHR, WAT, PACB, BRKR, GEHC-adjacent… (~45 names) | Their RPO is service-contract / consumables deferred revenue, a different economic signal from a manufacturer's order backlog (spec §2 warns against blending these). Kept only genuine industrial-instrument / test-equipment makers: AME, ITRI, KEYS, KLAC, TER, FTV, VNT, VLTO, TRMB, BMI, GEOS, RAL, ROK, CDRE, GEHC. |
| **Pre-revenue / SPAC EV-charging, hydrogen, battery, eVTOL, space startups** | BLNK, CHPT, PLUG, NVVE, AMPX, NRGV, SES, ADNH, BETA, AIRO, FLY, VOYG, RDW, TPICQ(bankrupt) | No established order-backlog series; RPO is speculative or thin. Kept the ones with a real disclosed book: FLNC, EOSE, GNRC, ENS, POWL, AVAV, RKLB, KTOS. |
| **Autos / auto-parts / consumer** | TSLA, APTV, AEVA, INVZ, LIDR, FOXF, BC, MCFT, REEAF | Order backlog is not a meaningful disclosed metric for auto OEMs / consumer products. |
| **Software-heavy "industrial tech"** | ROP | Roper is now majority application software (SaaS RPO), not equipment. |
| **Foreign OTC ADR / micro / recent-IPO with no history** | ABLZF (ABB OTC), SKK, MTEN, INLF, CDNL, TRT, TPCS, SIF, PRZO, RVSN, SPWR (bankruptcy remnant), SMR (pre-revenue SMR dev) | Thin data / no usable multi-year RPO series. Kept liquid foreign primaries CNH, FTI, ESLT, JBTM. |
| **Homebuilders** | GRBK | "Backlog" here is home-sale orders, outside the capital-goods scope. |

## Flagged inclusions (kept, but note)

- **MOG-A** (Moog) — dual class; SEC/yfinance ticker is `MOG-A`.
- **ESLT** (Elbit), **CNH**, **FTI** (TechnipFMC), **KRNT** / **AZTA** / **JBTM** — foreign primary listings but liquid US lines with clean history.
- **RGR** (Sturm Ruger) — consumer firearms, but reports a genuine unit/dollar order backlog each quarter.
- **GEHC**, **TRMB**, **CDRE** — borderline hardware-vs-service; kept because a hardware order backlog is disclosed.
- **KRMN**, **LGN**, **ECG** — 2024-25 IPOs / spin-offs; short history, will clear the ≥6-quarter guard only late in the window.

## Breadth note (spec §9)

109 names clears the "≥ 60" breadth bar comfortably. The event study will report how many of the 109
actually produce ≥ 1 valid surprise event after the ≥6-quarter + clean-span guards (spec §4) — the
newer IPOs and any name that only recently began tagging RPO will not contribute until late in the window.
