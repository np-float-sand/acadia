# Design Spec — Multi-Source Data-Center Demand Exposure Signal

**Status:** Approved design, not yet built.
**Context:** Every variant of the grid-resilience factor tried to date (stress-beta, ICR, single-source DC-queue, peer-group construction) has failed to beat XLU on a long-history backtest — see `docs/compact_2026-08-19-peer-group-construction-results.md` and `docs/handoff_2026-08-15-outage-reserve-margin-signal.md`. The existing DC-load signal (`grid_resilience/data/dc_load_data.py`) uses PJM's *generation* interconnection queue as an indirect proxy for data-center demand, covers only 6 of 18 tickers (PJM-only), and its one real result (FE scoring top) was flagged as ambiguous — queue MW that "reads as large utility-scale generation buildout at least as plausibly as co-located data-center demand" (`docs/compact_2026-08-13-dc-load-signal-results.md`).

**Reframed bar (this spec):** the client's actual ask is not a "responsible investing" narrative — it's evidence of a material, tradeable edge over XLU, even from a short sample, that justifies allocating away from a safer sector bet. Short backtests are acceptable; thin ticker coverage (e.g., 2 names) is not — the signal needs breadth across the existing universe to be useful.

## Goal

Build per-ticker data-center demand exposure from **traceable** sources — sources where the DC-attribution is explicit or contractually named, not inferred from a generic proxy — combining three complementary layers. Concretely: Layer 2 keeps the existing 6 PJM tickers (AEP, D, EXC, FE, PPL, PEG); Layer 1 adds real (if noisier) coverage for CNP (AEP already covered); Layer 3 adds the merchant sleeve (CEG, TLN, VST, NRG). That's ~11 of the ~18-ticker universe with real DC-specific data, up from 6 — a meaningful expansion, not "most of the universe," and the remaining ~7 (other MISO/SPP/CAISO-zone regulated names) stay on the existing ICR fallback until those ISOs publish comparable data.

## Data landscape (confirmed by live spikes, 2026-08-26)

| ISO | Public large-load data? | Granularity | History | Verdict |
|---|---|---|---|---|
| PJM | Yes — Load Analysis Subcommittee "Large Load Adjustment Requests" | Zone-level, annual, explicitly industry-tagged ("data center" for 20 of 22 zone/area rows in the 2025-11-24 vintage; 2 tagged industrial/crypto) | Recurring since ~2024; forecast horizon to 2046 | **Use — Layer 2** |
| ERCOT | Yes — monthly "Large Load Interconnection Status Update" decks | TSP-level (AEP, CenterPoint/CNP directly map to universe tickers) | Individual monthly PDFs confirmed back to 2024-03 | **Use — Layer 1** |
| MISO | No comparable dataset | Only system-wide TWh growth ranges (149–241 TWh by 2044) and a generic *generator* queue by state, not by utility/load-type | N/A | **Defer** — revisit if MISO publishes utility-level large-load data |
| SPP | Board-approved large-load process (RR 696/"HILL"), Sept 2025 | Unknown yet | Only months old | **Defer** |
| CAISO | Still at issue-paper stage (Jan 2026) | Unknown yet | ~4.5 GW under study, no utility breakdown found | **Defer** |

## Layer 1 — ERCOT monthly large-load queue

**New module:** `grid_resilience/data/ercot_large_load_data.py`

Fetches and parses ERCOT's recurring monthly Large Load Integration Team PDF decks (e.g., `ercot.com/files/docs/2026/03/12/March-TAC-Report.pdf`), extracting the TSP-level MW-by-status table (No Studies Submitted / Under ERCOT Review / Planning Studies Approved / Approved to Energize / Observed Energized).

- Covers **AEP** (AEP Texas) and **CenterPoint (CNP)** directly. Other TSPs (Oncor, TNMP, Brazos, CPS, Rayburn, LCRA) are municipal/co-op/complex-ownership and are not ticker-mapped in v1 — flagged for follow-up if a clean public-company mapping is confirmed (do not guess; verify ownership before adding a ticker).
- Point-in-time discipline: each snapshot is dated by the deck's own "Snapshot Date," giving a real monthly time series (~24 observations back to 2024-03) — no reconstruction needed, unlike the PJM queue's point-in-time logic.
- Signal shape matches the existing pattern: level (current queue MW / TSP size) + momentum (MW added in trailing N months), same as `queued_mw()`/`new_mw_since()` in `dc_load_data.py`.
- **Limitation:** ERCOT's queue is not explicitly DC-labeled per project (unlike PJM's), but public reporting puts DC share at 70%+ of ERCOT large-load requests — treat as a noisier proxy than Layer 1's PJM counterpart, not a clean one.

## Layer 2 — PJM Large Load Adjustment (cross-check + zone-map fix)

Two uses, not a replacement for the existing generation-queue signal:

1. **Enrichment/cross-check.** The LAS data is explicitly industry-tagged per zone ("data center" vs "industrial & crypto" vs "industrial"). Use it to flag where the existing generation-queue-based DC signal and the LAS-labeled data disagree — a free consistency check that would have caught the original FE ambiguity earlier.
2. **Concrete zone-mapping fix.** The LAS zone list includes **METED**, which `docs/compact_2026-08-13-dc-load-signal-results.md` already identified as missing from FE's `TICKER_NODE_MAP` entry (FE currently only maps to ATSI/JCPL, "roughly half of FE's actual footprint"). Add METED to FE's zone list. Check DUQ and APS ownership before adding — do not assume they map to an existing ticker without verifying (Duquesne Light appears to be privately held; confirm before use).
3. **Not a standalone backtest.** Only ~2 annual vintages exist so far (Nov 2024, Nov 2025) — not enough for a time-series signal on its own. Use as a cross-sectional, current-state enrichment layer.

## Layer 3 — named hyperscaler PPA/deal events (merchant sleeve)

A curated event table: `ticker, counterparty_hyperscaler, mw, first_disclosure_date, source_url, iso_corroboration_date (optional)`.

Covers the merchant/IPP sleeve (CEG, TLN, VST, NRG) — the cleanest exposure in the universe, since each entry is a named, MW-quantified, dated transaction (Microsoft–CEG Three Mile Island restart; Amazon–TLN Susquehanna; Meta's multi-GW nuclear commitments across VST/CEG), not an inference from aggregate queue data.

**Point-in-time discipline (critical — per 2026-08-26 review):**
- `first_disclosure_date` must be the **earliest public disclosure** — press release, earnings call transcript, investor day, or SEC 8-K/10-Q — never the date a deal later appears in a PJM/ERCOT filing.
- Before accepting any date sourced from an ISO/regulatory document (e.g., a utility's own "Large Load Request" filing), explicitly check whether the underlying project was already covered in media/press. If it was, use the earlier media date — the ISO filing is corroboration, not the event.
- If an ISO filing is genuinely the *first* public mention of a project (no prior press coverage found), track it separately as a distinct leading-indicator case rather than folding it into the general rule — this is the rarer, higher-value scenario and deserves its own note in the entry, not silent conflation.
- Rationale: dating by ISO-filing date when press already covered the deal means the backtest trades on stale, already-priced information, which would understate or misattribute the signal's real predictive power.

Raw score: cumulative disclosed DC-linked MW under contract per ticker, as of each date (event-study style, not a continuous flow like Layers 1–2).

## Signal combination

Layers 1–3 are on different scales (queue-MW ratios, LAS-forecast ratios, cumulative contracted MW) and cover non-overlapping ticker subsets with no ticker getting more than one real-data layer in v1 (PJM tickers get Layer 2, CNP gets Layer 1, merchant sleeve gets Layer 3). Combination is therefore per-ticker selection, not blending: each ticker uses whichever layer covers it, z-scored against *its own subpopulation* (tickers on that layer) before entering `build_factor()` — reusing the subpopulation-z-score pattern `fill_with_icr()` already established in `dc_load_data.py`, specifically because concatenating raw values across layers before a single z-score is the exact bug already found and fixed once in this codebase (the DC-ratio/ICR mixing bug in `docs/compact_2026-08-13-dc-load-signal-results.md`). Tickers with no layer coverage keep the existing ICR fallback.

## Backtest / validation approach

Given the short-sample nature of Layers 1 and 3:
- Report point estimate **and** n (independent rebalance periods) together — never the point estimate alone.
- Test sensitivity to window start/end date (does the result survive dropping the first or last observation?).
- Pre-state the bar for "material" before running the test (economically material return spread, not just a sign flip on Sharpe) — this project has been burned twice already by short-sample-looking-good results that didn't survive scrutiny (the retracted PEG claim; the peer-group Sharpe-got-worse-despite-lower-vol artifact). Structure the test so a positive result is trustworthy by construction, not just optimistic.

## Out of scope / deferred

- MISO, SPP, CAISO large-load layers — no usable public utility-level data yet (confirmed by spike). Revisit if/when they publish comparable reports.
- Any ticker mapping not independently verified (Oncor, TNMP, DUQ, APS ownership) — do not guess a parent-company ticker; confirm or omit.
