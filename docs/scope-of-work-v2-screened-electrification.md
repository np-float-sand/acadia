# Scope of Work — Screened Electrification Equipment Strategy
**Draft for Discussion (v2)**

---

## Parties

**Consultant:** Sandhya Persad
**Client:** Addenda Capital
**Date:** 2026-09-15

---

## Background

The Consultant has designed and tested a quantitative equity strategy — **Screened
Electrification Equipment** — that invests in the companies supplying the equipment and
construction services for the U.S. electricity grid build-out. The thesis rests on physical
need (AI data-center load growth, electrification of heating and transport, and reshoring of
domestic manufacturing) rather than investor sentiment. A fixed, rules-based screen selects
profitable, reasonably-valued equipment and construction suppliers — currently 27 names, with
an adopted extension to 29 (see Phase 1) — and the resulting book is managed with a set of
automatic risk overlays: a gold/bond macro reserve, a valuation-extension de-risk ladder, and
volatility targeting. Two variants are offered: Option A (buy-only) and Option B (Option A plus
a hedge against a broad U.S. industrials basket). Full construction detail, performance, and
known limitations are documented in `docs/electrification-strategy-proposal-v6.md`, the Phase 2
findings report referenced throughout this Scope of Work.

This document supersedes the original draft's framing around a grid-resilience "stress beta"
long/short book (an 18-utility strategy that was explored in earlier research but not carried
into delivery). The Phase 1–3 scope below describes the work actually performed, and remaining,
against Screened Electrification Equipment.

The Client will provide access to risk model data. The Consultant will source all other required
data from public and commercially available sources as described herein.

---

## Scope of Work

### Phase 1 — Data Integration & Signal Development
*Estimated: 50–70 hours*

The Consultant will:

1. **Establish and validate the data pipeline; integrate short interest.** Consolidated the
   existing equity-price, fundamental, and macro (FRED) data pipeline into a reproducible,
   cached research environment, confirmed to rebuild cleanly from a cold cache. Closed a real
   gap by adding short-interest tracking (`short_interest.py`): a live yfinance snapshot of
   DLR/EQIX short interest captured on every pipeline run. No free historical short-interest
   endpoint exists, so this accumulates history going forward rather than backfilling; it is
   used as a cost/liquidity diagnostic for the strategy's hedge short leg, not as a signal
   input.

2. **Forward power price data — descoped.** The shipped strategy does not require a
   forward-looking power-price input. Feasibility work on two candidate sources (the EIA STEO
   `ELWHU_*` forward series and the ERCOT real-time/day-ahead LMP spread) is complete and
   documented as available future work, not exercised further. See
   `docs/RESEARCH-LOG.md` §4 item 20, `docs/steo-archive-feasibility-results.md`, and
   `docs/rt-da-spread-feasibility-results.md`.

3. **Integrate Client-provided risk data.** Built `risk_exposures.py`, an adapter against a
   placeholder schema (long-format `date, ticker, factor, exposure`), tested independent of
   real data, with a documented swap-in point for the Client's risk-model file. No file has
   been received from the Client as of this writing — this remains genuinely pending on the
   Client, not a gap in delivery.

4. **Develop enhanced signal component — stress beta.** A pre-registered, run-once statistical
   test — never previously conducted on this supplier universe — examined whether the
   strategy's sensitivity to grid-stress events (reusing the Grid Stress Index and
   conditional-beta methodology built for the earlier grid-resilience research) could be
   turned into a scoring signal. The pre-registered gate required a pooled t-statistic of at
   least 2.0 on the 2021-present window; the observed result was **t = 0.144**, and the gate
   did not pass. Per the pre-registered protocol, the signal was **not** wired into the
   strategy. In its place, this deliverable formalizes the three risk-management rules already
   built into the strategy — the 15% GLD/IEF macro reserve, the valuation-extension de-risk
   ladder, and 20%-target volatility targeting — as the practical, evidence-backed answer to
   grid-stress sensitivity, each mapped to a specific stress category and supported by an
   episode-drawdown table across five real market shocks (COVID, the 2022 rate shock, the
   DeepSeek sell-off, the 2025 tariff shock, and the 2026 sell-off). Full methodology and
   results: `docs/electrification-stress-beta-gate-results.md` and
   `docs/electrification-stress-sensitivity-response.md`.

5. **Additional, beyond the original scope: REIT-breadth universe extension.** A probe testing
   whether adding two data-center-exposed REITs (IRM, AMT) to the supplier universe improves
   the strategy was run under a pre-registered gate. Result: **adopted.** The resulting
   29-name `screen_reit` construction beats the base 27-name screen on both Sharpe and maximum
   drawdown in every overlay stack tested, and is retained as an available, documented
   construction. See `docs/electrification-reit-breadth-probe-results.md`.

*Deliverable: reproducible data pipeline with short-interest and risk-exposure diagnostics;
pre-registered stress-beta gate results and the resulting stress-sensitivity response; an
adopted REIT-breadth universe extension.*

---

### Phase 2 — Backtest & Validation
*Estimated: 40–60 hours*

The Consultant will:

1. **Create Strategy** — extended `backtest.py`/`__main__.py` to wire Phase 1's risk-exposure
   and short-interest diagnostics into the output bundle as reporting, not as hard portfolio
   constraints, since no real Client risk data exists yet to optimize against.
2. **Run rolling backtest** — re-ran the full backtest end to end from a genuinely cold cache
   and confirmed it is reproducible (515 tests passing; clean output bundle of `metrics.csv`,
   `episode_drawdowns.csv`, `plateau.json`, `returns.csv`, and `performance.png`).
3. **Produce written findings report** — `docs/electrification-strategy-proposal-v6.md`
   (with accompanying HTML/PDF), summarizing performance results, signal construction,
   the stress-beta gate outcome, the short-interest and risk-data integration status, known
   limitations, and the recommended configuration.

*Deliverable: backtest results (code + output files), written findings report
(`docs/electrification-strategy-proposal-v6.md`).*

---

### Phase 3 — Handoff & Documentation
*Estimated: 20–30 hours*

The following items remain outstanding as of this document's date and will follow Client
review of the Phase 2 findings report. The Consultant will:

1. **Document all code** — provide inline documentation and function-level docstrings across
   `electrification_strategy/`, including the short-interest and risk-exposure modules added
   in Phase 1, and a technical reference (`docs/technical-reference.md`) covering each module's
   inputs, outputs, and role in the pipeline — including the placeholder risk-data schema
   contract for the Client's future risk-model file.

2. **Write operational runbook** — `docs/runbook.md`: step-by-step instructions for
   environment setup, refreshing data caches, running the pipeline, rebalancing, interpreting
   output, and handling common failure modes.

3. **Conduct handoff sessions** — participate in working sessions with the Client's designated
   technical contact to walk through the codebase, answer questions, and confirm the Client
   team can operate and extend the system independently. Scheduling depends on Client
   availability.

4. **Deliver final codebase** — provide a deliverable manifest (`docs/deliverable-manifest.md`)
   listing exactly what ships versus what remains internal research, followed by a clean,
   tagged release of the full repository in a format specified by the Client.

*Deliverable: documented codebase, runbook, handoff session(s), final code delivery.*

---

## Compensation

### Hourly Rate
**$150 per hour**, billed against actual hours worked.

### Monthly Hour Cap
Hours are capped at **30 hours per month**. Work exceeding 30 hours in any calendar month requires
written pre-approval from the Client before the hours are incurred.

### Invoicing & Timesheets
The Consultant will submit an itemized timesheet by the 5th business day of each month covering the
prior month's work. Invoices are due net 30 days from submission. The Client may request a brief
check-in at any frequency to review hours against deliverables progress.

### Data Cost Reimbursement
Out-of-pocket data costs (e.g., CME DataMine electricity futures subscription) will be billed as
pass-through at actual cost, with receipts, and are **capped at $600 total** across the full
engagement. The Consultant will obtain written approval before incurring any single data expense
exceeding $100.

---

## Data Responsibilities

The Client will provide data from commercial risk model.
All data sourced by the Consultant is drawn from public or commercially available sources. The
Consultant will not use any material non-public information in the course of this work.

---

## Intellectual Property

Upon receipt of full payment for each phase, the Consultant assigns to the Client all right, title,
and interest in the work product produced under that phase, including source code, documentation,
backtest output, and written findings. The Client will own all deliverables outright and may use,
modify, and extend them without restriction.

The Consultant retains no rights to the deliverables and will not use Client-provided data for any purpose outside this engagement.

---

## Scope Limitations

The following are explicitly outside the scope of this engagement:

- **Live trading or portfolio management.** The Consultant's role is limited to historical research
  and backtesting. All trading decisions are made exclusively by the Client.
- **Investment advice.** Nothing in this engagement constitutes investment advice or a recommendation
  to buy or sell any security.
- **Access to current or live positions.** The Consultant will work with historical and publicly
  available data only and will not have access to the Client's current portfolio, positions, or
  live trading systems.
- **Performance guarantees.** Backtest results are historical simulations. Past simulated performance
  is not indicative of future results. No representation is made regarding live performance.

---

## Term

This engagement begins upon execution and continues until completion of Phase 3 or mutual written
agreement to terminate, whichever comes first. Either party may terminate with 30 days written notice.
Upon termination, the Client will owe fees for all hours worked and data costs incurred through the
notice period, and the Consultant will deliver all work product completed to that point.

---

## Signatures

**Consultant**

Signature:

Name: Sandhya Persad

Date: 

---

**Client**

Signature: 

Name: 

Title: 

Date: 

---

*This document is a draft scope of work and does not constitute a binding agreement until signed by
both parties.*
