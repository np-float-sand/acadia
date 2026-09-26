# Untried-tasks log — electricity-market-data edge search

**Purpose:** a working checklist of the electricity-market-data angles identified in a
2026-09-08 review of `docs/RESEARCH-LOG.md` and `docs/grid-stress-signal-evidence.md` that
are either (a) still genuinely untried, or (b) built but never closed out with a verdict.
This is a **to-do list**, not the canonical research record — once a task reaches a verdict,
also add the row to `docs/RESEARCH-LOG.md` §3 per its own convention, and update this file's
status in place.

**Status legend:** `UNTRIED` · `IN PROGRESS` · `VERDICT REACHED` (link to detail).

**Convention:** work top to bottom (priority order = cost-adjusted plausibility, cheapest
"close out existing machinery" items first). Each task keeps a one-line status row in the
ledger below plus its own `###` section with what was actually done.

---

## Ledger

| # | task | priority | status | one-line finding |
|---|---|---|---|---|
| 1 | Congestion-spread Phase 1 validation + Phase 2 go/no-go | high (already built) | **VERDICT REACHED** | Fix only ever mattered for ERCOT (PJM was never broken); works as designed there. Phase 2 **not recommended**. |
| 2 | ICR retest via EDGAR XBRL (reuse `edgar.py`) | high (data blocker now solved) | **VERDICT REACHED** | Feasibility PASS (63 median quarters vs yfinance's 4-12) but hypothesis FAIL (full-sample rank-IC t=-0.95, wrong sign) — the two big-looking window correlations are PCG/EIX wildfire-liability and VST/CEG AI re-rating confounds, not a leverage effect. `USE_ICR` stays off. |
| 3 | Merchant stress-beta orthogonalization (5 fixes, §5 of `grid-stress-signal-evidence.md`) | high (root-cause fix, cheap) | **VERDICT REACHED, ALL 5 TESTED (2026-09-18, fix #5 numbers corrected 2026-09-20)** | #1 (filter) works as diagnosed but too small to matter, NOT ADOPTED. #2, #4, #5 FAILED. #5's original worktree run had a bad cache (caught via benchmark sanity check); corrected numbers still FAIL but show IC improving/Sharpe collapsing, not uniform degradation. #3 redirected into item 6 (PARTIAL — see item 6). Series closed. |
| 4 | RT/DA LMP spread — live feasibility check | medium | **VERDICT REACHED** | ERCOT: fully feasible, easier than assumed (RT already the production fetch; DA is one cheap call). PJM: confirmed blocked pre-2024. VST/NRG (the actual problem names) are both ERCOT — doesn't matter. |
| 5 | ERCOT CRR true clearing prices | medium | **VERDICT REACHED — FAILED (feasibility)** | Live API only exposes a rolling ~12-month window (confirmed, params don't unlock more); no public bulk archive. Needs a special ERCOT data request, not a free spike. |
| 6 | STEO forward-power series, reframed as beta-stabilizer (not basket-timing) | medium | **VERDICT REACHED, PARTIAL (2026-09-19)** | PUDL/Zenodo shortcuts ruled out, but found a better one: Wayback Machine has 63 point-in-time XLSX crawls of EIA's current-release companion workbook (2016-12→2026-08), no PDF extraction needed. Spark-spread regressor gets IC@21d t 1.352→2.427 (first fix in the series to clear \|t\|≥2.0) and MaxDD −31.4%→−15.9%, but full/stress Sharpe both tick slightly worse — strict gate not fully met. NOT ADOPTED as-is; most promising lead in the whole fix series. `docs/steo-spark-spread-results.md`. |
| 7 | Real-time ancillary/scarcity pricing (ERCOT ORDC adder, PJM sync reserve) | low-medium (needs scoping) | **IN PROGRESS — feasible, not yet promising** | PJM data feasible + full history to 2018. First honest timing-IC check (monthly, HAC-corrected, 2018-2021 only): no signal (t<1.1 both horizons) — an earlier daily-overlapping-window pass looked strong but was an autocorrelation artifact. n=44 months is thin; 2022-2025 not yet pulled (rate-limited, needs chunking). ERCOT side still FAIL (blocked, item-5-style). |
| 8 | Interconnection-queue-velocity capture — process check | housekeeping | **VERDICT REACHED** | No automation exists anywhere (no cron, no workflow file) — only one static cached snapshot. Open item #10's "keep running" is aspirational, not real. |
| 9 | Electricity narrative/mispricing (esp. short side) — equity-mispricing thesis | A FAILED, C FAILED, B not started | **IN PROGRESS** | Approach A: not significant at any horizon (best t=-1.15), not adopted. Approach C (broadened per user request to a 10-feature walk-forward GBM/RF on every alignable RESEARCH-LOG feature): **FAILED** — recent-window (2023-25) IC looks real (t up to 3.73, perm p as low as 0.000) but full-window (2018-25) is flat-to-negative (|t|<1 every cell) — same overfit shape as the Henry Hub GBM work; narrative never ranks in the top-3 feature importances even in the inflated recent fits. `docs/narrative-mispricing-approach-c-results.md`. Approach B handoff still unstarted. |

---

## 1. Congestion-spread Phase 1 validation + Phase 2 go/no-go — VERDICT REACHED (2026-09-08)

**Question:** `CLAUDE.md` / `docs/superpowers/specs/2026-05-28-congestion-spread-design.md`
claimed `congestion_frac` "arrives as NaN for all ISOs," motivating an inter-zonal-spread
fallback (built for ERCOT + PJM, merged `f3f1ed0`, 2026-05-28). Promoting it to a separate
5th GSI sub-signal (Phase 2) was deferred "until the spread signal is validated in
backtesting" — that validation was never actually done or written down.

**What was checked (offline, cached data only, no new fetches):** ran
`daily_lmp_summary()` / `daily_spread_summary()` / `fill_congestion_from_spread()` directly
against the real production caches — PJM's primary hub fetch (`pjm_lmp_DAY_AHEAD_HOURLY_*`,
filtered to `ISO_BENCHMARK_NODES["PJM"]`) and ERCOT's (`ercot_lmp_settlement point_*`,
filtered to `ISO_BENCHMARK_NODES["ERCOT"]`), both full 2018–2026.

**Finding — the original diagnosis was only half right:**
- **PJM:** the raw gridstatus feed *already carries* a `congestion` column at the hub level.
  `congestion_frac` is **0% NaN before any fix**, every day, both windows checked. The
  spread-fill function changes **0 of 2,922 days** — it is a complete no-op for PJM. PJM's
  congestion component was never broken.
- **ERCOT:** the raw feed has no congestion component at all — `congestion_frac` is **100%
  NaN before the fix**, confirmed across 2,922 days. The spread fallback genuinely fires,
  filling all of them with a real, non-trivial series (mean 0.85, std 0.82, weak/slightly
  negative same-day correlation with `lmp_mean`, −0.078 — not just a repackaged price level).
  This is the one ISO where the fix does real work, and it works as designed.
- Git history shows the fix (`f3f1ed0`, 2026-05-28) predates the 243-combo grid search
  (`6a7399e`, 2026-06-18) and every subsequent GSI backtest — every Sharpe/IC number in
  `grid-stress-signal-evidence.md` already reflects the ERCOT-only fix being live. There is
  no "before" state left in the repo's history to A/B against; the fix has effectively been
  soak-tested for 3+ months across every later probe (all weak/negative) without breaking
  anything.

**Phase 2 recommendation: do not build it.** Phase 2 (LMP energy/congestion/loss
decomposition + promoting inter-zonal spread to an independent 5th sub-signal) would only
ever matter for ERCOT — PJM's component data is already correct and complete. Given the
2026-08-26 closeout audit's conclusion that the GSI's problem is signal strength / sample
size, not construction, and that the ERCOT congestion leg is already live and contributing,
there's no live bug left to fix and no strong prior that decomposing it further changes the
verdict. Closing this out is more valuable than building it.

**Action taken:** corrected `CLAUDE.md`'s "Future Work — GSI congestion signal Phase 2" note
and added a verdict row to `docs/RESEARCH-LOG.md` §3.

---

## 2. ICR retest via EDGAR XBRL — VERDICT REACHED (2026-09-08)

`USE_ICR = False` in `config.py` because yfinance only returns ~4–12 quarters of history per
ticker (documented limitation, `CLAUDE.md`). Reused `research/electricity_earnings_exposure/edgar.py`'s
`quarterly_flow()` helper against a freshly-resolved CIK map for all 25 `grid_resilience`
universe tickers.

**Feasibility: PASS.** Median 63 quarters/ticker (vs yfinance's 4–12), 22/25 tickers back to
2019 or earlier, most to 2008. The depth blocker is fully solved.

**Hypothesis: FAIL.** Full-sample monthly rank-IC (regulated-only, 102 months): mean IC
−0.025, t = −0.95 — null, wrong sign. The targeted 2022-23 rate-hiking-cycle test is also
flat (t = −0.24, regulated-only). Two windows *look* like a huge effect (t > 4) but both
trace to PCG/EIX's wildfire-liability litigation (unrelated to leverage) and VST/CEG's
AI-power re-rating (the same merchant-narrative contamination documented in
`grid-stress-signal-evidence.md` §5) — not a genuine ICR relationship.

**Disposition:** `USE_ICR` stays off. Full writeup: `docs/icr-edgar-retest-results.md`.
Reusable: `CIK_BY_TICKER` (25 names) + `research/icr_edgar_retest/icr_probe.py` for any
future fundamentals question on this universe.

## 3. Merchant stress-beta orthogonalization — VERDICT REACHED, 3/5 tested (2026-09-08)

Five candidate fixes for the VST/NRG whipsaw were designed in June/July 2026 and never
built. Tested three of them for real, via monkeypatching the actual production pipeline
(no toy approximations):

1. **XLU/XNG correlation filter** — TESTED, works as diagnosed. 100% of the 63 flagged
   ticker-months were the 4 known merchant names (NRG/VST/CEG/TLN), zero false positives on
   regulated names. Stress-period Sharpe improves −0.145→−0.085, full Sharpe −0.007→+0.050.
   **NOT ADOPTED** — real effect, but nowhere near enough to beat XLU (0.261).
2. **Orthogonalize stress beta to ERCOT/PJM power-price returns** — TESTED, **FAILED**.
   Barely engages (VST's beta correlates 0.996 with baseline); IC gets slightly worse, not
   better. Trailing realized price is the wrong proxy for a forward-looking narrative
   repricing.
3. **Power-price momentum / forward-curve regime flag** — not tested standalone; **folded
   into item 6** (the STEO forward-power work), since it's the forward-looking version of
   the same idea fix #2 already tested and failed in its trailing form.
4. **Structural-break-adaptive (exponentially-weighted) beta window** — TESTED, **FAILED
   decisively**. Full Sharpe −0.007→−0.448, IC worse at every horizon. Recency-weighting an
   already-thin stress-day sample trades bias for ruinous variance.
5. **Fundamentally-anchored beta** (revenue % spot-exposed, fuel mix, region) — **not
   tested**. Needs hand-collected per-company-quarter data, not a price-data spike like the
   other three. Left open as a real, bigger-lift candidate.

**Bottom line:** consistent with the 2026-08-26 closeout audit — construction fixes aimed
at the known whipsaw mechanism top out at a few points of Sharpe (fix #1), nowhere near the
gap to XLU, and more aggressive fixes (#2, #4) actively hurt. Full detail:
`docs/merchant-beta-orthogonalization-results.md`.

## 4. RT/DA LMP spread — live feasibility check — VERDICT REACHED (2026-09-08)

Recommended first step in three separate handoffs over a month, never executed until now.

**ERCOT: PASS, and easier than every prior doc assumed.** `ercot.get_lmp()` — already the
production fetch feeding the live GSI — turns out to already be the **real-time** leg
(SCED, 5-min), not day-ahead. The missing piece is day-ahead (`get_dam_spp(year)`, full
year in ~8s, 2011->present, identical location names to what's cached). No new "RT"
fetcher needs building at all.

**PJM: FAIL for the bulk of the backtest window, confirmed live.** `market="REAL_TIME_HOURLY"`
for any date before PJM's ~2-year archive cutoff forces a full ~10,000-pnode download before
filtering — reproduced twice (1-day and 1-week pre-2024 ranges both 429'd within 6 pages);
the identical call for a recent (2025-06) week succeeded in under 5 seconds.

**Net read:** VST and NRG — the two names actually driving the documented stress-beta
whipsaw (item 3) — are both ERCOT. PJM's blocker doesn't touch the part of this idea that
matters most. Full detail: `docs/rt-da-spread-feasibility-results.md`.

**Next step if picked up:** build the ERCOT RT/DA spread signal itself (design + gate) —
this session was feasibility-only, per every prior handoff's own recommendation. Worth a
15-minute check for a PJM DataMiner bulk historical export (like the one already used for
FTR bids) before writing PJM off completely.

## 5. ERCOT CRR true clearing prices — VERDICT REACHED, FAILED (2026-09-08)

The FTR-style signal that failed (`docs/ftr-bid-signal-results.md`) used PJM **bid** data
(a sentiment/demand proxy, not a clearing price). ERCOT's CRR (Congestion Revenue Rights)
auction results are genuinely public with real clearing prices, but checked live this
session: the document API for both CRR report products (Monthly Auction Results, Long-Term
Auction Results) only ever returns a rolling ~12-month window — confirmed empirically,
including that date-range/paging/size query parameters don't unlock older files — and
ERCOT's public CRR page has no historical-archive or bulk-download section. Getting full
2018-2026 history needs a special request through ERCOT client services, not a free
self-serve pull. **FAILED at feasibility**, same shape as CME/ICE power forwards and NERC
GADS — the genuinely forward-looking instruments in this space keep landing behind a
paywall or manual request, not a quick API call. `docs/ercot-crr-feasibility-results.md`.

## 6. STEO forward-power series, reframed — SCOPED, NOT RUN, corrected (2026-09-08)

`docs/hhub-forward-gas-signal-results.md` §Round 3 found `EIA STEO ELWHU_*` — a free,
hub-level, genuinely-forward (~16 mo) power price forecast, 2010→present. Its real purpose
here is `docs/handoff-forward-power-price-signal.md`'s two-variable model meant to
**stabilize the VST/NRG stress beta** (item 3's redirect target) — not a repeat of the
basket-timing framing that already failed with the Henry Hub gas proxy.

**Checked, then corrected.** First pass this session thought PUDL (Catalyst Cooperative,
already used in this repo for FERC Form 1) might have this pre-parsed. **It doesn't.**
Live-checked PUDL's actual processed database (the public S3 nightly mirror,
`s3://pudl.catalyst.coop/nightly/`, the same source `transmission_rate_base/` reads) —
421 tables total, zero match STEO/ELWHU/wholesale/outlook. The Zenodo record that looked
promising is PUDL's *raw-input* archive (a preservation mirror of EIA's original files,
confirmed by inspection to be year-zipped raw material, "not a cleaned/processed dataset"
per its own description) — a more stable place to fetch the same PDFs from, not a shortcut
past parsing them.

**Disposition: back to the original estimate.** Extracting `ELWHU_PJ`/`ELWHU_TX` from
each of ~90 monthly STEO releases still needs real PDF-table extraction, ~half a day per
the original scoping. Not done this session. Detail: `docs/steo-archive-feasibility-results.md`.

## 7. Real-time ancillary/scarcity pricing — IN PROGRESS: feasible, not yet promising (2026-09-08)

Never mentioned anywhere in the prior research trail before this session.

- **PJM: feasible, full history.** `get_real_time_as_market_results()` returns Regulation /
  Synchronized Reserve / Primary Reserve clearing prices back to Jan 2018 (2022-01 onward
  uses a different, much larger post-market-redesign schema and hit sustained 429s this
  session — needs monthly chunking with more patience, not yet pulled). Sanity check
  passed: the Jan 2018 cold snap shows a real $230/MWh Regulation price spike.
- **ERCOT: FAIL, same shape as item 5.** `get_real_time_adders()` (the ORDC price adder)
  and `get_as_reports_dam()` are both rolling "2-Day Disclosure" products with no historical
  depth via the live API.
- **First honest promise check (2018-2021 only, 175,320 rows pulled):** built an RTO-wide
  ancillary-price composite z-score and tested it as a basket-timing signal (it's identical
  across every PJM name on a given day, so it can only ever be a timing/overlay signal, not
  a cross-sectional ranker). A naive daily-overlapping-window pass looked strong (up to
  t=4.61) — but that's an autocorrelation artifact. Properly corrected (monthly,
  non-overlapping, HAC standard errors, the discipline used everywhere else in this log):
  **no signal** — 1-month t=+1.04, 3-month t=+0.18 (vs. the PJM-regulated basket's own
  forward return); similarly flat vs. basket-minus-other-ISO. n=44 months, so this isn't a
  kill, just "nothing found yet on a thin sample."

**Disposition: feasible, genuinely untested at real scale as a standalone timing signal**
(2018-2021 leg). **2022-2025 was subsequently pulled** (45 months, monthly-chunked,
`research/narrative_mispricing/pjm_ancillary_2022_2025.parquet`) and used as one of 7
features in the ML combined retest below — it earned real feature importance there but did
not rescue the combined-signal hypothesis (see `docs/ml-combined-retest-results.md`).
Standalone timing-IC on the 2022-2025 leg specifically, and the `MAD` sub-zone
decomposition, remain untried. Detail: `docs/ancillary-scarcity-pricing-feasibility-results.md`.


---

## 9. Electricity narrative/mispricing (short side) — A + C FAILED, B open (2026-09-08)

Raised by an external source alongside the earnings-exposure idea (already FAILED as
Probes A/B — a different, fundamentals-based channel). Brainstormed properly (per
`superpowers:brainstorming`): classified as a spike first (can "narrative" even be
operationalized in a testable way), then an architectural design once a source was picked.
User chose **both EDGAR filing mentions and GDELT news volume, combined** (EDGAR as
disclosure signal, GDELT as attention/intensity), sequenced **A then B then C**.

**Approach A — narrative-extremity short (crowding/overhype) — DONE, then REFINED, still
not significant.** Composite = avg(z-scored trailing EDGAR mentions, z-scored GDELT
volume), ranked cross-sectionally across the 25-name universe. Original pass ("data
center" keyword only + cross-sectional GDELT z-score): right sign for a short thesis
(negative) but not significant — 1mo t=-0.39, 3mo t=-1.15, 6mo t=-0.55. VST's composite was
*consistently negative* (under-narrated) through its whole 2023-25 narrative run — a real
measurement gap, not a disproof (it rarely says "data center" in its own filings, being a
generator not a data-center operator).

**Refined same session:** added "large load" (PJM/ISO planning terminology — VST's count
went 1→13; "hyperscaler" tried too, near-useless, dropped) and switched GDELT to each
ticker's own trailing-756d self-baseline z-score instead of cross-sectional. **Fixed the
measurement cleanly** — VST's composite now genuinely tracks known events (1.83 at the
Sep-2023 narrative build, 1.31 at the Feb-2025 momentum peak) instead of sitting flat
negative. **Did not move significance** — 1mo flipped positive (t=+1.40, momentum-shaped,
not overhype-shaped), 3mo/6mo stayed weak-negative (t=-1.09/-0.57). Nothing clears |t|≥2
before or after. Honest read: better data ruled out "just bad measurement," the underlying
relationship (if any) is genuinely weak. **Not adopted.** Detail:
`docs/narrative-mispricing-approach-a-results.md`.

**Approach B — narrative-surprise event study — handoff written, not started.** Treat each
new EDGAR keyword-mention event as a discrete trigger, GDELT volume as the decay curve
after it, look for a pop-then-reverse pattern to short into. Flagged risk going in: close
in shape to open item #15 (regulatory-calendar DER short — FAILED, n≈1, telegraphed news
already priced). Full handoff for picking this up in a fresh session:
`docs/handoff-narrative-mispricing-approach-b.md`.

**Approach C — narrative-fundamentals divergence, broadened to all-features GBM/RF —
RUN, FAILED.** Per user instruction, skipped straight to the ML angle rather than the plain
linear divergence sort, and broadened the feature set beyond ICR alone to **every reasonably
alignable feature already built in this research line** on the 25-name universe: narrative
composite, ICR, merchant stress-beta, the production GSI factor score (fresh run, current
config), own price momentum, plus four market-wide series (utility capex-guidance revisions,
category-demand momentum, PJM ancillary pricing, real-yield change, gas momentum). Walk-forward
`TimeSeriesSplit(5)` GBM + RF, full vs. lean feature sets, 1mo/3mo forward return, pre-registered
bar = |t|≥2 **and** perm-null p<0.05 in **both** a FULL (2018-25) and RECENT (2023-25) window.
**Result: exactly the Henry Hub overfit shape.** RECENT window looks strong (t up to 3.73,
perm p as low as 0.000, half the cells clear the bar); FULL window is flat-to-negative in
every single cell (|t|<1, perm p 0.20-0.92). Fails the pre-registered both-windows bar
outright. Feature importances also show narrative_z never in the top 3 of any full-model
cell — even the inflated recent-window fits are driven by momentum/GSI-factor/stress-beta,
not narrative. **FAILED, closed.** `docs/narrative-mispricing-approach-c-results.md`,
harness kept at `research/narrative_mispricing/approach_c/`.

---

## 10. ML two-fold walk-forward retest — REAL LEAD FOUND, not yet actionable (2026-09-08)

Combined ICR, narrative (EDGAR+GDELT), merchant stress-beta, XLU/XNG spread, momentum,
and PJM ancillary pricing into RF/GBM/classifier models, walk-forward: Fold A (train
2022-23, test 2024) weak; **Fold B (train 2024-25, test 2026) GBM classifier gets OOS
AUC=0.758**, stable across seeds, clears a shuffled-label null test, consistent across all
6 test months. Ablation shows the driver is **stress-beta + ICR + EDGAR disclosure counts
combined — not the narrative/GDELT attention data** this thread started with. n=54 test
observations (6 months) is too small to trust alone. GDELT could not be extended into
2026 (rate-limit hang); its contribution to this result is unreliable either way.
**Addendum (different session, same day): merchant-name exclusion check — the result
is NOT the VST/CEG/TLN/NRG confound.** Dropping those 4 names (the recurring confound
behind several other broken signals on this universe) weakens but does not kill it:
OOS AUC 0.755→0.719 on the remaining 19 regulated names, still 100th-percentile vs.
its own shuffled-label null, still not one-month-driven. Raises, doesn't lower, this
finding's credibility — narrows why it might be real without making it actionable.
**Second addendum, same day: isolated the 3-feature (stress-beta/ICR/EDGAR) core —
mixed, not confirmatory.** On the full 23-name universe the 3-feature-only model is
much weaker than the 7-feature one (AUC 0.567, only 78th-pct vs. its null) — the
other four features (including the ones that looked like pure noise in the drop-one
ablation) are doing real work in combination. On the 19 regulated-only names the
3-feature model holds up close to the 7-feature result (0.645 vs 0.670). So
stress-beta+ICR+EDGAR is necessary but not sufficient, not the whole mechanism.
**Not wired anywhere. Decision: KEEP and PURSUE this lead** (user instruction,
2026-09-08) — tracked as RESEARCH-LOG §4 item 18. Next: fix GDELT, let the 2026 test
window grow, re-test the isolation finding as more months accrue. Detail:
`docs/ml-two-fold-walkforward-results.md` (now incl. both addenda),
`docs/handoff-narrative-mispricing-portal.md`.

---

## Session log

### 2026-09-08 — log created; items 1 and 8 resolved

Created this file from the review requested in the prior turn. Resolved #1 (congestion-spread
Phase 1/2) with a full offline data check against production caches — see §1 above; corrected
`CLAUDE.md`'s overbroad "NaN for all ISOs" claim and added the verdict to
`docs/RESEARCH-LOG.md` §3. Resolved #8 (queue-velocity capture) as a process-check negative —
no automation exists. Items 2–7 logged with next steps, not yet started.


### 2026-09-08 (cont.) — item 9 added at user request (external source), queued behind item 3

User asked whether two ideas from an external critique were covered: the earnings/mispricing
channel (already tested and FAILED as Probes A/B, 2026-09-07 — not re-added) and
narrative/mispricing on the short side (genuinely new — added as item 9). Explicitly queued
behind finishing item 3's remaining fixes and the items-4-7 spikes, per user instruction.

### 2026-09-08 (cont.) — item 9 Approach C run, broadened per user request: FAILED

User asked to broaden Approach C beyond ICR-only to "any feature we created in
RESEARCH-LOG.md that might help in tandem... even if it was only t-stat 1 or so," and to use
GBM/RF rather than a linear interaction sort. Classified as a spike (superpowers:brainstorming),
flagged the precedent risk up front (the Henry Hub "kitchen-sink GBM/RF" result that vanished
out-of-sample), got a nod, then built a 10-feature monthly panel on the 25-name universe
(narrative composite, ICR, stress-beta, a **freshly re-run** production GSI factor score,
momentum, plus capex-guidance/category-demand/ancillary-pricing/real-yield/gas as market-wide
conditioning variables) and ran a pre-registered walk-forward GBM/RF test.

**Result: FAILED**, reproducing the exact overfit shape the guardrail warned about — a
recent-window (2023-25) result that clears |t|≥2 and perm p<0.05 in several cells (up to
t=3.73) goes flat-to-negative on the full 2018-25 window (|t|<1 in all 8 cells). Narrative
never ranks in the top-3 feature importances even where the model found *something*.
Detail: `docs/narrative-mispricing-approach-c-results.md`. Approach B (event study) remains
the only unstarted thread in item 9.

### 2026-09-18 — session queue: satellite sleeve, solar rotation, STEO spark-spread, fix #5

User reviewed the crypto-miner probe's FAIL and asked what's left to try with the shipped
`screen +val+sleeve` book, "combining with miners, generators, anything." Queued and launched
four items in this order: (1) satellite-sleeve diversification test (miner/generator/combined
sleeves at 5-15% NAV, non-signal-driven — new open item 25 in `docs/RESEARCH-LOG.md` §4,
in progress); (2) wire the already-tested solar-rotation conditional long (§4 item 17) from
candidate to a selectable construction; (3) STEO forward-power PDF extraction (this item,
`ELWHU_PJ`/`ELWHU_TX`) to unblock the merchant-stress-beta stabilizer reframing; (4) merchant
stress-beta orthogonalization fix #5 (fundamentally-anchored beta, item 3 above) — hand-collected
fuel-mix / spot-exposure data for VST/NRG/CEG/TLN. All four dispatched in parallel; results
pending.

### 2026-09-19 — items 3 and 6 both closed out

Both agents queued the prior session hit a session-wide rate limit mid-run, then stalled
twice more (watchdog timeouts from a passive-wait pattern) after being resumed. Both had
actually completed their computation before stalling — the orchestrator read the
completed backtest output directly from each abandoned worktree and wrote up the results
docs/log updates itself rather than continuing to bounce the stalled agents.

Item 3 (all 5 merchant stress-beta fixes) is now fully closed: fix #5
(fundamentally-anchored beta, hand-extracted 10-K fuel-mix/spot-exposure data for
VST/NRG/CEG/TLN) FAILED both tested variants. Item 6 (STEO forward-power spark-spread)
reached a PARTIAL verdict — the first fix in the whole 5-fix-plus-redirect series with a
real, gate-clearing IC improvement, though the realized-Sharpe half of the gate isn't met.
See `docs/merchant-fundamental-beta-results.md`, `docs/steo-spark-spread-results.md`, and
the updated synthesis in `docs/merchant-beta-orthogonalization-results.md`.
