# Pre-registration — CEP as a data-center-REIT signal

**Date:** 2026-09-16
**Status:** PRE-REGISTERED, ABOUT TO RUN — written before touching any output.
**Origin:** `docs/handoff-2026-09-16-datacenter-reit-cep-signal.md`, picked up per user
instruction ("ok let's try those").

## Decision 1 — universe

`DLR`, `EQIX`, `IRM` (all three currently-public pure-play/near-pure-play DC REITs available to
this project; `COR`/`QTS` excluded, delisted/acquired). Primary target is the **equal-weighted
portfolio** of the three, not a per-name panel — matches how the existing `hedge_overlay.py`
sleeve is constructed (equal-weight `DLR`+`EQIX`) and the handoff's own framing (§"Statistical
power note": this is a time-series signal, n=months not n=names). A 3-ticker OLS panel with
ticker fixed effects is reported as a secondary robustness check only, not a gate.

## Decision 2 — CEP series

**National EIA commercial retail price** (`sectorid=COM`, `stateid=US`, confirmed to exist,
monthly, Jan 2015 → Jun 2026, `https://api.eia.gov/v2/electricity/retail-sales/data/`).
State-weighted version explicitly skipped for this pass (no state-exposure map exists for these
3 REITs in this repo; would need hand-collected 10-K property lists — out of scope for "a little
time looking at small variations").

**Primary transform: month-over-month % change** in CEP (matches this project's own signal
convention — `probe_b.py`'s `pct_change`), not level. Level is reported as a secondary check
only, since the paper's own functional form (levels vs. changes) isn't confirmed from the
abstract.

## Decision 3 — controls

Match the paper's set at reduced availability, noted explicitly rather than silently dropped:
- NGP -> Henry Hub front-month (already available in this project, per the Henry Hub handoff)
- COP -> FRED `DCOILWTICO` (WTI)
- MV -> FRED `VIXCLS`
- MR -> `SPY` monthly return
- GS10 -> FRED `GS10` (change), following the same convention as `DFII10` elsewhere in this repo
- GPR -> **dropped**. The Caldara-Iacoviello GPR index has no FRED mirror and would require a
  new fetcher against a non-EIA/non-FRED host; out of scope for a first pass. Flagged as a
  limitation on the "real replication" claim, not silently omitted.

## Decision 4 — target / horizon

Own-horizon replication first: **monthly** REIT portfolio return regressed contemporaneously on
monthly CEP change + controls (the paper's horizon). This project's preferred 1–3mo *forward*
framing is reported as a secondary rank-IC check only if the contemporaneous spec clears gate.

## Decision 5 — linear spec / gate

OLS: `portfolio_ret ~ cep_chg + hh_chg + wti_chg + vix_chg + spy_ret + gs10_chg`, HAC (Newey-West)
standard errors with **maxlags=6** (fixed before running, not tuned post-hoc). **Gate: |t| on
`cep_chg` >= 2.0**, matching this project's standard threshold
(`docs/electrification-stress-beta-decile-preregistration.md` Decision 5, `probe_b.py`'s G1-G3).
Sign expected negative per the paper (higher CEP -> lower REIT return).

## Decision 6 — non-linear spec

**Time-series tercile spread**, not cross-sectional (n=3 names is too thin for cross-section,
per handoff item 5 / statistical-power note): rank all months in-window by CEP MoM change into
terciles; portfolio-return spread = mean(top tercile) − mean(bottom tercile) months; one-sample
t-test on the spread. This is the direct analogue of the stress-beta decile follow-up's method,
adapted from cross-sectional buckets (many names, one month) to time-series buckets (one
portfolio, many months) since that's the axis with real sample size here (n≈132 months in the
full window). **Gate: |t| >= 2.0**, same threshold as Decision 5.

Tree-based (GBM/RF) fit — listed as optional in the handoff — is **not run** this pass: with
only ~3 controls + CEP and ~130 monthly observations, a tree fit's own overfitting risk (flagged
explicitly in the crypto-miner handoff) outweighs what it would add over the tercile spread at
this sample size. Can be added later if the linear/tercile specs show something worth chasing.

## Decision 7 — windows (all three reported, not just one)

1. Paper's own **Dec 2015 – Dec 2025** (primary replication window).
2. **2023-01-01 → present** (project standard recent window).
3. **2026-01-01 → present** (2026-YTD; CEP data currently runs through 2026-06, so this window
   has ~6 monthly observations — reported for context, not gated, per the handoff's own
   instruction to report it but flag consistency, not treat a 6-point sample as decisive).

A result that holds only in one window is flagged regime-fragile / not adopted, matching every
other gate in this project.

## Decision 8 — combination test (only if standalone clears)

Per the handoff's "Portfolio role" section: (a) CEP gate alone vs. real-yield gate alone, (b)
both-must-fire vs. either alone, (c) CEP as a continuous 0–1x scaler, evaluated on the
**combined `screen +val+sleeve` book's** Sharpe/Calmar/MaxDD and feared-scenario P&L — run only
if Decisions 5/6 produce a standalone signal that clears its gate in at least the 2 non-YTD
windows. If the standalone signal fails, the combination test is not run (a gate change to the
existing sleeve is not evaluated off a signal that doesn't independently work).
