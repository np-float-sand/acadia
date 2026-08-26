# Handoff — Outage / Reserve-Margin Signal for Merchant Generator Differentiation

**Status: SHELVED (2026-08-22).** A live spike test disproved the core premise. Do not pick
this back up without new data — see "Why this is shelved" below. If a new outage-availability
data source is ever found, start there, not from the design spec (`docs/superpowers/specs/2026-08-20-outage-availability-signal-design.md`,
now superseded/dead).

---

## Why this is shelved

A design spec was written (`docs/superpowers/specs/2026-08-20-outage-availability-signal-design.md`)
proposing to modulate merchant stress-beta by a 3-month-trailing EIA-860 `status`-based
availability ratio. An independent harsh review of that spec surfaced several serious issues
(directional-logic bug in the modulation formula, a code path — `dual_track` — the fix silently
never reached, no empirical validation plan) — see git history of the design spec doc for the
full review if needed. Before fixing any of that, a cheap spike was run to test the review's
most damaging finding: that ~90-100 days of EIA reporting lag plus a 3-month trailing average
might make the signal too stale to ever see the events it exists for.

**The spike found something worse than staleness: the data source doesn't capture the
phenomenon at all, at any lag.** Live-pulled EIA `operating-generator-capacity` status history
for VST's and NRG's curated ERCOT entities (`NRG Energy Inc`, `NRG Texas Power LLC`,
`NRG Cedar Bayou Development Company LLC`, `Luminant Generation Company LLC`) for all of
2020–2021. Result: **`OP` ratio = 1.0 in every single month for both tickers — including
February 2021, the month of Winter Storm Uri**, one of the most severe ERCOT generation-outage
events on record (widely documented ~40-50GW of ERCOT capacity offline at the storm's peak).
Zero status change, for either company, through the single most extreme stress event in the
entire backtest window.

Sanity-checked this wasn't a fetch or entity-mapping bug: the dataset *can* show non-`OP` status
in February 2021 — 41 of 1330 ERCOT generators did (3%) — just none of them are VST/NRG
entities, and 3% is nowhere near the real scale of Uri's outages.

**Root cause, visible in hindsight from the status-code descriptions already pulled during the
original research pass:** EIA-860's `status` facet is annual-scale, not event-scale. `OS` =
"out of service and **not expected to return to service in next calendar year**"; `OA` = "out of
service but **expected to return to service in next calendar year**." These codes track
retirements and extended mothballing, not the days-to-weeks forced outages that actually happen
during a winter storm or heat wave. A unit that trips for 3–5 days during Uri and is back online
before month-end never touches this field — the monthly categorical snapshot simply doesn't
register it. This is not a lag or granularity problem that a shorter lag/lookback window could
fix; the field itself doesn't carry the information at any lag.

Combined with the earlier finding that ERCOT's actual event-scale data source
(`gridstatus.Ercot.get_hourly_resource_outage_capacity`) has no historical archive (rolling
window only, nothing before ~last month), **there is currently no available data source — free
or otherwise checked — that can support this signal's original thesis** (differentiate which
merchant generator can capture a specific stress event's price spike based on real-time
availability). Both realistic candidates have been tried and ruled out.

## What would need to be true to revisit this

- A new, event-scale, historically-archived generator-outage data source is found (NERC GADS
  remains theoretically possible but was already deprioritized as subscription-restricted with
  no free/public access — would need someone to actually check institutional access, not just
  assume).
- Or the thesis itself changes scope — e.g., accepting that this can only ever be a
  structural/annual-scale signal (fleet retirements, long-term mothballing) rather than an
  event-response signal, which is a fundamentally different and much weaker claim than what
  motivated the idea originally ("a generator with a unit down for maintenance during a heat
  wave"). Worth an honest gut-check on whether that weaker claim is worth anything before
  reviving this.

## Original research trail (for context only — superseded by the shelving above)

Full prior history: EIA-860 route/frequency/pagination bugs were found and fixed in
`grid_resilience/data/eia_data.py::fetch_plant_capacity()` along the way (that fix stands on its
own merits, unrelated to this idea's fate, and is covered by `tests/data/test_eia_data.py`).
Entity-name curation findings (Vistra appears only as "Luminant Generation Company LLC" in EIA
data, no "Vistra" substring at all) are preserved in the design spec's Background section if
ever useful for a different signal touching the same data.

## What's still open, unaffected by this

- **RT/DA LMP spread** (`docs/handoff_2026-08-15-rt-da-spread-signal.md`) — a separate,
  not-yet-brainstormed idea, still worth pursuing. Its own recommended first step (verify
  `gridstatus` actually returns clean historical RT LMP for PJM/ERCOT) hasn't been done yet.
- **Reserve-margin/scarcity-market data** (PJM/ERCOT capacity auctions) — noted as out of scope
  in the now-dead design spec, never live-checked. A genuinely different, forward-looking thesis
  from generator-level outages; not disproven by this spike. Would need its own research pass if
  picked up.
