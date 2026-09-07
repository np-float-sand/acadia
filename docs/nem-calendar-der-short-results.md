# Regulatory-calendar DER short trigger (open item #15) — FAILED

**Date:** 2026-09-06
**Question:** does shorting the resi-solar / DER sleeve (ENPH SEDG RUN NOVA) in a
pre-registered window around **scheduled adverse net-metering (NEM) decisions** add value —
as an event study, and as a conditional short overlay on the shipped `screen +val+sleeve`?
**Throwaway:** `scratchpad/nem_calendar_der_short.py`. **Calendar:**
`electrification_strategy/data/nem_calendar.csv` (10 dated state NEM events, hand-built from
public record; kept as a reference, not wired).

## Small-sample caveat (stated up front)

In 2017–2026 with liquid resi-solar names there is **one major clean adverse event — CA NEM
3.0** — whose three milestone dates (proposal 2021-12-13, final vote 2022-12-15, effective
2023-04-15) fall inside a single ~16-month decline. The other in-window adverse milestones
(NC HB 589 2017, IN SEA 309 2017, FL HB 741 2022, CA VNEM 2023-11) are minor. So n ≈ 1
cluster + noise — the same wall every signal in this project has hit.

## Event study — no consistent sign

DER sleeve cumulative return over [−90d, +90d] around each adverse milestone, and vs the
electrification long book:

| milestone | sev | DER −90..−1d | DER 0..+90d | DER −90..+90d | (DER − book) |
|---|---|---|---|---|---|
| NC HB 589, 2017-07-27 | minor | +38.9% | +6.9% | **+48.6%** | +49.2% |
| CA NEM 3.0 proposal, 2021-12-13 | major | +16.2% | −6.4% | **+8.7%** | +14.6% |
| FL HB 741, 2022-03-09 | minor | −8.6% | +4.7% | −4.3% | +6.9% |
| CA NEM 3.0 vote, 2022-12-15 | major | +1.3% | −22.0% | **−21.0%** | −35.2% |
| CA NEM 3.0 effective, 2023-04-15 | major | −17.2% | −4.0% | **−20.5%** | −42.7% |
| CA VNEM, 2023-11-16 | minor | −33.3% | **+32.9%** | −11.3% | −35.6% |

- **Mean across all 6: DER +0.0% relative to the book** — no edge.
- **Major only (3): −10.9%** — but that is the 2022-H2→2023 NEM 3.0 cluster (two milestones
  ~4 months apart in one continuous decline); the **Dec-2021 proposal milestone is *positive*
  (+8.7%)** — you'd have been short into a still-rising market — and the Nov-2023 minor event
  bounced +33% in the 90 days after.
- The "signal" reduces to "resi-solar fell hard 2022-H2 through 2023," which the price already
  showed and the always-on DER short already captured.

## Overlay — short DER during [milestone −120d, +45d] windows

On `screen +val+sleeve` (mask covers 24% of days, concentrated 2021–2023):

| variant | full Sharpe | CAGR | MaxDD | 2019–22 Sh | feared P&L vs ship |
|---|---|---|---|---|---|
| shipped (no NEM short) | 0.69 | 14.4% | −24.0% | +0.51 | — |
| − 0.15× DER in adverse windows | 0.73 | 14.9% | **−27.6%** | +0.55 | **−7%** |
| − 0.25× DER in adverse windows | 0.73 | 15.1% | −30.9% | +0.55 | −11% |
| − 0.40× DER in adverse windows | 0.68 | 15.2% | −35.6% | +0.53 | −18% |

- Modest Sharpe lift (+0.04) but **MaxDD deepens 4–7pp** and **feared-scenario P&L is negative**
  — still an anti-hedge in the squeeze scenario, because the windows include the Dec-2021
  proposal (short into a rip) and the 2023 bounce.
- **The timed-short sleeve loses −1.9%/yr standalone** — the small gain it caught around the
  Dec-2022 vote / Apr-2023 effective date is more than offset by the 2021 and late-2023 windows.
- Shorting *tightly* around just the vote + effective date would have worked — but that is
  hindsight window selection on one episode, not a rule.

## Verdict: FAILED — into the graveyard

A genuinely different signal *class* (regulatory calendar, not a physical/price series), but it
fails for the same reasons as everything before it: (1) the DER short is regime-dependent and
only "works" in the 2022–23 rate-shock window; (2) telegraphed policy is already in the price
months ahead, so the calendar adds no timing precision — and it puts you short *before* the
proposal, at the top; (3) n ≈ 1 event. Nothing wired. `nem_calendar.csv` kept as a reference
for a future discretionary risk monitor (open item #5), not as a signal.

**If ever revisited:** would need (a) more adverse events — wait for the ~2027–2028 wave of
state NEM reviews (several are docketed), and/or backfill pre-2015 (CA NEM 1.0 cap, early AZ/HI)
with a resi-solar proxy; (b) a *tight* window (vote → effective only, not proposal); (c) test it
as a *scale-down of a long solar tilt* (open item #17) rather than an outright short.
