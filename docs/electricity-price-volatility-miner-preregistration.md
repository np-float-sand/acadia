# Pre-registration — electricity-price-volatility miner/generator signal, multi-hub scan

**Date:** 2026-09-16
**Status:** PRE-REGISTERED, ABOUT TO RUN — written before touching any hub/t-stat output.
**Origin:** `docs/handoff-2026-09-16-crypto-miner-power-volatility.md`, extended per user
instruction: scan all ERCOT/PJM hubs instead of guessing one, widen the miner set, and add a
parallel arm testing the same mechanism on merchant generators.

## Correction to the source handoff

Mid-Columbia ("Mid-C") is a WECC/Pacific-Northwest bilateral hub, not a CAISO settlement point
(CAISO's own hubs are SP15/NP15/ZP26) — it sits *adjacent* to CAISO on the intertie, which is
probably why the pasted summary blurred the two. Neither Mid-C nor CAISO is fetchable in this
repo today, so this doesn't change feasibility: the scan below is scoped to what's actually
cached (ERCOT + PJM). NYISO — the hub that would best match WULF's disclosed Lake Mariner (NY)
site — also has **no fetcher in this repo**; building one is out of scope for this run and is
logged as a follow-up, not silently skipped.

## Decision 1 — hub universe

**19 hubs, all already cached at native (hourly, resampled to daily) resolution — no new
fetcher needed:**
- ERCOT (7): `HB_BUSAVG, HB_HOUSTON, HB_HUBAVG, HB_NORTH, HB_PAN, HB_SOUTH, HB_WEST`
- PJM (12): `AEP GEN HUB, AEP-DAYTON HUB, ATSI GEN HUB, CHICAGO GEN HUB, CHICAGO HUB,
  DOMINION HUB, EASTERN HUB, N ILLINOIS HUB, NEW JERSEY HUB, OHIO HUB, WEST INT HUB,
  WESTERN HUB`

NYISO and CAISO/Mid-C are excluded from this run (infeasible without new fetcher work) — this
means the WULF (NY-heavy) leg is tested only against the weaker PJM proxy (its 20% PA exposure),
not its actual settlement market. This is stated up front, not discovered after the fact.

## Decision 2 — the multiple-comparison guardrail on "pick the most predictive hub"

Scanning 19 hubs and reporting whichever fits best is a data-snooping trap this project has been
burned by repeatedly (Henry Hub GBM, narrative-mispricing Approach C, category-demand rotation —
all "found" a relationship in-sample that died out-of-sample). **Hub selection is done by
split-sample, never by full-sample fit:**

- **Train window (hub selection only): 2023-01-01 → 2024-12-31.** For each name, compute the
  GARCH-style variance-regression |t| against all 19 hubs on this window only; the hub with the
  largest |t| (correct predicted sign) is selected. If no hub has the correct sign, record "no
  hub selected" for that name rather than picking the least-wrong one.
- **Test window (verdict-bearing, never used for selection): 2025-01-01 → present**, with
  2026-01-01 → present reported separately as the required regime-fragility check per the
  original handoff's Decision 4.
- **A selected hub that is geographically implausible for the name (e.g., CIFR — 100% ERCOT —
  selected via a PJM hub) is reported as a red flag, not suppressed.** Plausibility is judged
  against each name's disclosed site list (already in `universe.py` comments / the source
  handoff), fixed before results are seen.
- Gate is evaluated **only on the held-out test-window t-stat**, not the train-window number that
  drove selection.

## Decision 3 — miner universe (widened)

**Primary (named in the paper summary): `CIFR`, `WULF`.**
**Widened for robustness, reported separately, not pooled:** `MARA`, `RIOT`, `IREN`, `BTDR`
(all already in `research/electricity_earnings_exposure/universe.py`). **`CLSK` excluded** — its
disclosed fleet (GA/MS/TN/WY) sits in SERC/WAPA territory, not ERCOT or PJM, so no hub in the
Decision-1 universe is a defensible proxy for it; including it would force a meaningless
best-of-19 pick.

## Decision 4 — parallel arm: merchant generators (the "original names")

**`VST`, `NRG`, `CEG`, `TLN`** — the merchant-generator names this project already tracks
extensively (stress-beta, ICR, narrative-mispricing threads; see `docs/RESEARCH-LOG.md`
"Merchant-generation margin nowcast" and related rows). Unlike the electrification-book equipment
makers (ETN, GEV, HUBB, …), these names have *direct, disclosed* wholesale-price revenue
exposure — the closest economic analog to the miners' cost exposure that already exists in this
project, and VST/NRG are already known to be ERCOT-heavy (per `docs/rt-da-spread-feasibility-
results.md`), so the same 19-hub scan + split-sample selection applies unchanged.

**Sign prediction differs from the miner arm and must be stated now, not fit after the fact:**
for a merchant generator, higher realized wholesale-price volatility plausibly *raises*
subsequent equity-return variance (more volatile revenue), the opposite direction from the
miners' predicted curtailment-hedging (lower variance). Both arms use the same regression
spec and gate mechanics; only the predicted sign flips. A generator result of the miner's
predicted sign (vol-lowering) is not a "bonus pass" — it's a different, unexplained mechanism and
is reported as such.

## Decision 5 — spec, controls, windows (unchanged from source handoff)

- Linear: GARCH-style variance regression (equity return variance regressed on lagged hub
  price-return variance) + plain OLS of forward return on lagged hub price return.
- Non-linear: tree-based fit (GBM/RF) + tail-only cut, checked out-of-sample (the train/test
  split from Decision 2 doubles as this OOS check — no separate split needed).
- **BTC return/variance is a mandatory control for the miner arm** (both primary and widened
  names) — this is the exact confound `docs/electricity-earnings-exposure-probe-results.md`
  Probe B already caught once (sector-bloc move masquerading as a price relationship). The
  generator arm controls for SPY/XLU instead (no BTC channel).
- Windows reported: train (2023-2024, selection only, not gate-bearing), test (2025-present,
  gate-bearing), and 2026-YTD (fragility check) — three numbers per name, all shown.

## Decision 6 — gate

**|t| ≥ 2.0 on the test-window (2025-present) regression, correct pre-registered sign, for the
selected hub only.** 2026-YTD is reported for consistency but does not itself gate (per source
handoff Decision 4, a window mismatch is flagged as regime-fragile, not an automatic fail) —
however a PASS on the 2025-present window that inverts sign in 2026-YTD is called out explicitly
as fragile, not presented as a clean pass. Primary miners (CIFR, WULF) and generators (VST, NRG,
CEG, TLN) are the names that count toward the headline verdict; widened miners (MARA, RIOT, IREN,
BTDR) are robustness-only and cannot rescue a primary-name failure.

## Decision 7 — reporting commitment

All four groups (primary miners, widened miners, generators, and the train-vs-test hub-selection
table for every name) are reported regardless of outcome, including "no hub cleared the sign
requirement" as an explicit, nameable result — not folded into a silent average. Implausible
winning hubs (Decision 2) are called out by name.

## Implementation

New module `research/electricity_price_volatility/` (daily hub price series already cached under
`grid_resilience/data/cache/{ercot_lmp_settlement point,pjm_lmp_DAY_AHEAD_HOURLY}_*.parquet`;
stock/BTC prices via the existing `fetch_prices` helper). Results go to
`docs/electricity-price-volatility-miner-results.md`; `docs/RESEARCH-LOG.md` §3 gets a row on
completion per this project's standing convention.
