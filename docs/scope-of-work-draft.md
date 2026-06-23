# Scope of Work — Grid Resilience Equity Strategy
**Draft for Discussion**

---

## Parties

**Consultant:** Sandhya Persad
**Client:** Addenda Capital
**Date:** June 2026

---

## Background

The Consultant has proposed a quantitative strategy that trades stocks/options
based on their measured sensitivity to electricity grid stress events. 


The Client will provide access to risk model data. The Consultant will source all other required data from public and commercially available sources as described herein. 

---

## Scope of Work

### Phase 1 — Data Integration & Signal Development
*Estimated: 50–70 hours*

The Consultant will:

1. **Establish and validate the data pipeline** — integrate public grid data,
   equity price, fundamental data, and short interest history into a
   reproducible research environment.

2. **Integrate augmenting data sources** — source forward power price data and wire into the signal layer. 

3. **Integrate Client-provided risk data** — align risk factor exposures to the universe and confirm data format compatibility.

4. **Develop enhanced signal components** — build and validate stress beta

*Deliverable: working data pipeline, documented signal layer.*

---

### Phase 2 — Backtest &  Validation 
*Estimated: 40–60 hours*

The Consultant will:
1. **Create Strategy** - mapping of signal into portfolio weights over time.
2. **Run rolling backtest** — produce P&L output, performance stats (Sharpe, maximum drawdown, etc).
2. **Produce written findings report** — summarize performance results, signal construction, known limitations, and recommended configuration.

*Deliverable: backtest results (code + output files),  written findings
report.*

---

### Phase 3 — Handoff & Documentation
*Estimated: 20–30 hours*

The Consultant will:

1. **Document all code** — provide inline documentation, function-level docstrings, and a technical
   reference covering each module, its inputs/outputs, and its role in the pipeline.

2. **Write operational runbook** — step-by-step instructions for running the pipeline, refreshing
   data, rebalancing, interpreting output, and handling common failure modes.

3. **Conduct handoff sessions** — participate in working sessions with the Client's designated
   technical contact to walk through the codebase, answer questions, and confirm the Client team
   can operate and extend the system independently.

4. **Deliver final codebase** — provide a clean, tagged release of the full repository in a format
   specified by the Client.

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
