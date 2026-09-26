# Real-time ancillary/scarcity pricing — feasibility check results (2026-09-08)

**Status: PJM feasible now (full history, fast); ERCOT's closest equivalents blocked to a
rolling window.** Untried-tasks-log item 7 — this idea had never been scoped at all before
this session.

---

## TL;DR

- **PJM ancillary-services market clearing prices: PASS, cleanly.**
  `pjm.get_real_time_as_market_results()` returns Regulation, Synchronized Reserve, and
  Primary Reserve clearing prices (system-wide + a mid-Atlantic "MAD" sub-region) back to
  at least January 2018, fast (a full month in 1.5 seconds — this is a market-level
  aggregate, not per-node, so it avoids the pnode-count problem that blocks PJM's
  real-time LMP for old dates in item 4). Sanity check: Jan 1, 2018 hour-0 Regulation
  cleared at $230/MWh vs. a typical $5–90 — a real scarcity spike during the January 2018
  cold snap, exactly the kind of event this signal should catch.
- **ERCOT's closest equivalents: blocked, same pattern as item 5's CRR finding.** Both
  `get_real_time_adders()` (the actual ORDC price adder — the purpose-built scarcity
  signal I originally had in mind) and `get_as_reports_dam()` (day-ahead ancillary
  clearing prices) are served from ERCOT's rolling "2-Day Disclosure" report family —
  live-tested, both return only the most recent document (2026-09), nothing from 2019.
- **Net:** this is a real, previously-unscoped, mechanistically-distinct signal —
  purpose-built system-tightness pricing rather than a repurposed energy price — and it's
  immediately buildable for PJM's regulated names (AEP, EXC, PPL, FE, D, PEG, CEG, TLN),
  which is most of the universe. ERCOT (VST, NRG, CNP) would need the same kind of
  special-request workaround flagged for CRR in item 5.

---

## What was checked

**ERCOT** (via `gridstatus.Ercot`):
- `get_real_time_adders(date, end)` — the actual ORDC (Operating Reserve Demand Curve)
  price adder, ERCOT's purpose-built real-time scarcity signal. Live-tested for
  2019-01-01→08: **failed** — underlying report (`Real-Time Price Adders by SCED Interval`,
  type 13221) only lists one document, dated 2026-09-03. Same rolling-window shape as
  item 5's CRR finding.
- `get_as_reports_dam(date, end)` — day-ahead ancillary services clearing prices.
  Live-tested for the same range: **failed** the same way (`2-Day DAM AS Disclosure`
  report, only recent documents listed).

**PJM** (via `gridstatus.PJM`):
- `get_real_time_as_market_results(date, end)` — real-time ancillary market clearing
  prices. Live-tested for 2019-01-01→08 (1 week) and 2018-01-01→02-01 (1 month): **both
  succeeded**, 1–1.5 seconds, full expected schema (`Ancillary Service`, `Locale`,
  `Market Clearing Price`, capacity-required/cleared MW columns).

## Disposition

**Feasibility: mixed, PASS for PJM.** This closes the "never scoped" gap for a genuinely
new signal idea. Unlike every LMP-based candidate in this whole research line, ancillary
pricing is a **system-operator-computed scarcity measure**, not a settled energy price —
mechanistically distinct from the GSI's existing `lmp_z` component.

**Recommended next step, if picked up:** design and gate-test a PJM ancillary-scarcity
sub-signal (e.g. Synchronized/Primary Reserve clearing-price z-score) against the existing
regulated-name universe, following the same pre-registered-gate discipline as every other
signal in this log — this session was feasibility-only. The ERCOT side would need the same
special-data-request workaround already flagged in item 5, so treat it as PJM-first, or
skip ERCOT-side entirely for a first pass (accepting the signal covers only regulated
PJM names, not VST/NRG).

---

## Follow-up (2026-09-08, same session): a first honest timing-IC check — no signal yet

Pulled the full `PJM_RTO`-wide Regulation / Synchronized Reserve / Primary Reserve
clearing prices for 2018-2021 (175,320 rows; 2022-2025 hit sustained 429s on this
much-larger post-market-redesign dataset and needs monthly chunking with more patience,
not attempted this session), built a rolling-756-day z-scored composite, and ran it as a
**basket-level timing signal** — it's an ISO-wide number, identical across every PJM name
on a given day, so it can only ever work as a timing/overlay signal (like the GSI's
congestion regime), not a cross-sectional ranking signal.

**First pass (naive, daily-sampled, overlapping windows) looked promising — and was an
artifact.** Spearman IC vs. the PJM-regulated basket's forward return: 21d t=-3.85, 63d
t=-0.54; vs. PJM-regulated minus other-ISO names: 63d t=+4.61. **Properly corrected**
(monthly-resampled, non-overlapping, HAC standard errors — the same discipline this whole
research log uses everywhere else) those numbers **evaporate**: 1-month-forward HAC-OLS
t=+1.04 (vs. basket) / t=-0.40 (vs. relative); 3-month t=+0.18 / t=+0.20. The daily
overlapping-window t-stats were inflated by autocorrelation, not a real effect — the
correct read is a null result on 44 monthly observations.

**Honest disposition:** feasibility stays a clean PASS; a first quick look at *predictive
power* on the available (small, 4-year) sample shows nothing, but this isn't a kill —
n=44 months is thin, the 2022-2025 window (which includes the DeepSeek-era demand-narrative
regime that mattered for VST/NRG elsewhere) is still unpulled, and only the RTO-wide
composite was tried, not the `MAD` sub-zone series or a decomposition by individual
ancillary product. Logged as "feasible, not yet promising, sample too small to conclude
either way" rather than "feasible and promising" — the earlier framing overstated it.
