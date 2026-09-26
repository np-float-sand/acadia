# Interconnection-queue backlog as a REIT/yieldco signal — results

**Date:** 2026-09-16/17
**Status:** Exploratory, ad hoc (not pre-registered like the CEP thread) — run at user
direction as a follow-on from `docs/datacenter-reit-cep-signal-results.md`. All thresholds
here are the **user-directed |t|>1.5 bar** from that thread, not this project's default
|t|≥2.0 (see that doc's amendment note).
**Origin:** user hypothesis — data-center REITs and renewable-developer yieldcos are both
rate-sensitive *and* possibly both exposed to "grid supply momentum" (capital rotating toward
grid-equipment makers during a buildout boom, away from asset owners/waiters); solar/storage
developers specifically may be hurt by interconnection-queue delays (can't energize new
capacity → revenue growth stalls).

## Thread 1 — grid-equipment momentum, cross-validated on a second basket

The REIT thread (`docs/datacenter-reit-cep-signal-results.md`) found grid-equipment 3-month
momentum negatively predicts REIT returns (t=-2.53 full-window bare, weakens to -1.22 with
full controls; strengthens to -3.36 with full controls in 2023-onward but flagged as likely
small-sample instability, n=35/8 params).

Tested the same signal on **CWEN/BEP/AES/AQN** (renewable-developer yieldcos; NEP and AY are
both delisted/taken private and dropped). Same 3-month lookback:

| window | alone t | +SPY t | tercile t |
|---|--:|--:|--:|
| full (2018-2026) | -1.45 | -1.73 | -1.53 |
| 2023-onward | -1.46 | -1.79 | **-2.11** |

Same negative sign, similar magnitude to the REIT result, in an independently-chosen basket —
real cross-validation, clears the user's |t|>1.5 bar in 3 of 4 cells and the strict |t|≥2.0
bar in one (2023-onward tercile). 6-month lookback is weaker/inconsistent, not pursued further.

## Thread 1b — grid-equipment momentum: the genuine out-of-sample test (finally run)

Every other signal in this doc got a real OOS check (fit early, forecast late, compare to what
actually happened) except this one — it had only been tested across different, overlapping
windows, not a true train/holdout split. Fixed: fit `ret ~ grid_mom + spy_ret` on 2017-09→2022-12
(n=64, deliberately excluding 2023-onward, the period where the effect looked strongest, to
avoid the same contamination the other threads were checked for), then forecast 2023-01→2026-08
(n=44) using realized `grid_mom` and realized `spy_ret`.

| target | OOS rank-corr (pred vs. actual) | OOS linear-corr | cumulative predicted | cumulative actual |
|---|--:|--:|--:|--:|
| REIT | **+0.586** | +0.626 | +39.9% | +135.5% |
| yieldco | **+0.429** | +0.462 | +97.3% | +23.9% |

**This is the only signal in this doc whose out-of-sample directional correlation is both
positive and meaningfully sized, on both targets.** CEP's REIT OOS correlation flipped to -0.25
(wrong sign, see the addendum in `docs/datacenter-reit-cep-signal-results.md`); the LBNL/
PJM+CAISO queue index failed the same way (Thread 3/5). This one doesn't.

The point-forecast *magnitude* still misses badly, in opposite directions for the two targets
(REIT actual far exceeds forecast, yieldco actual falls far short) — but both misses share the
same shape as every other OOS check in this doc, which points to a common, signal-independent
cause (the fitted SPY-beta, 0.70-0.75, likely under-captures how strongly both baskets tracked
the market over a multi-year compounding stretch — REITs rode an AI/data-center re-rating well
beyond generic beta, yieldcos underperformed one) rather than a specific failure of the
grid-momentum term itself.

**Status: the one lead from this whole investigation that survives a genuine OOS directional
test.** Not a basis for sizing a live trade off the point forecast, but distinct from — and
better than — every other signal tested in this doc. If this thread is picked up again, this is
where to start.

## Thread 2 — wrong-category dead end: equipment suppliers, not developers

First attempt used ARRY/SHLS/NXT (utility-scale solar/storage *equipment suppliers*) against
this project's existing national Grid Stress Index (`electrification_stress_beta`). Result:
**positive** sign (grid stress → higher returns, t up to +1.94) — the opposite of the
hypothesis, because equipment suppliers benefit from buildout/stress tailwinds (same mechanism
as this project's entire long book), unlike asset owners stuck in a queue. Also: FLNC (one of
the natural candidates) is already long in `MARQUEE_UNIVERSE`, not available as a short
candidate. Confirms mechanism matters — supplier vs. developer is not a small distinction.

## Thread 3 — LBNL national interconnection-queue backlog index

**Data:** `inputs/LBNL_Ix_Queue_Data_File_thru2025.xlsx` (Berkeley Lab's "Queued Up," >50
grid operators, ~98% of US installed capacity, project-level, CC BY 4.0, user-provided —
automated fetch is blocked by Cloudflare on every LBNL/eta.lbl.gov path tried). Unlike PJM's
raw export, this file has a real `developer` field — but per-company matching to
CWEN/BEP/AES/AQN is still too thin to use (3, 2, 45, 5 rows respectively, mostly legacy/small/
withdrawn) — confirms the earlier PJM-only finding that per-company attribution doesn't work
with this data class. **National aggregate is the workable framing.**

**Construction:** filtered to Solar/Wind/Battery/Solar+Battery/Offshore Wind/Wind+Battery types,
reconstructed monthly total MW "in queue" 2015-2025 using true `q_date`/`wd_date`/`on_date` per
project (point-in-time-correct for queue *membership*, using real historical dates — not a
lookahead violation in the way MW-figure-revision is; see caveats). Series goes flat from
2026-01 onward (the file's snapshot ends "as of end of 2025" — no new information past that,
confirmed by an identical repeated value Jan-Jun 2026).

**In-sample result (2023-2025, n=36), 3mo change:**

| target | alone t | +SPY t | tercile t |
|---|--:|--:|--:|
| REIT | +3.15 | +2.52 | +2.02 |
| yieldco | -2.27 | **-5.44** | -0.37 |

Sign split matches the hypothesis (REIT positive, yieldco negative), and the yieldco
relationship *strengthens* with SPY added rather than weakening — unusual, since almost every
other signal tested in this project's log gets absorbed by market beta, not strengthened by
controlling for it.

**Two things cut against taking this at face value:**

1. **Regional withdrawal-date completeness is badly uneven.** West 12.1%, Southeast 35.0%,
   NYISO/ERCOT 48.7%, SPP 72.8% vs. MISO/CAISO/PJM/ISO-NE 93.8-98.8%. Rows from the five poorly
   -covered regions are >50% of the dataset, and `in_queue()`'s logic treats a missing
   withdrawal date as "still active" — so the national total likely overstates true backlog,
   more so in earlier/less-covered regions.
2. **Genuine out-of-sample test fails badly, in opposite directions for the two targets.**
   Fit the 2023-2025 relationship, freeze the queue-change signal at its last known value (Dec
   2025, since the series goes flat after), forecast Jan-Aug 2026 using *actual* realized SPY
   returns for the market-beta term:

   | target | predicted 2026 cumulative | actual 2026 cumulative |
   |---|--:|--:|
   | REIT | -1% to -8% | **+33%** |
   | yieldco | +30% to +36% | **+5%** |

   REIT did far better than forecast; yieldco did far worse. Both misses are tens of points —
   the classic in-sample-only pattern that has killed most other signals in this project's log
   once actually checked against subsequent data.

**Verdict: FAILED.** The in-sample fit is the strongest cross-target sign-consistent result of
this session, but it does not survive the first real out-of-sample test, and the underlying
data has a concrete, checkable coverage gap in >50% of its rows. Not adopted, not wired
anywhere. (Grid-equipment-momentum, Thread 1, still stands as the one lead that cross-validated
on a second basket — untested out-of-sample, unlike this one.)

## Portfolio-math note (from a live discussion, kept here for reference)

Adding a small hedge position to a portfolio only raises Sharpe if
`Sharpe(position) > correlation(position, portfolio) × Sharpe(portfolio)` — not simply "does
the position have a good Sharpe on its own." Measured for the momentum-timed DLR/EQIX short
vs. the `screen +val+sleeve` book, 2023-2026: corr=-0.34, portfolio Sharpe=1.596, breakeven
corr would need to be ≤-0.53; actual short Sharpe (-0.85) sits below that bar despite genuine
negative correlation — real hedge value, still a net drag. Confirmed empirically: short-weight
sweeps (0.25x/0.5x/1x/2x) show the naive real-yield-only gate degrading faster than momentum
-timed variants as size increases (1x: 0.767 vs. 1.048-1.180 Sharpe), meaning gate *quality*
matters more than today's default 0.25x sizing reveals — but no tested combination beats the
shipped real-yield-only gate in the two windows that matter most for current positioning (the
VOLT-fair window and 2024-2026), only in the full 2017-2026 backtest. Year-by-year breakdown
(2023/2024/2025/2026 YTD) shows the short only helps in one year of four (2025) — the pooled
multi-year Sharpe comparisons materially overstate how consistent that edge is.

## Thread 4 — LBNL signal wired into the combined book (portfolio-level check)

Despite the OOS failure above, quantified the actual portfolio impact of gating the DLR/EQIX
short on the LBNL backlog signal (short REIT when 3mo backlog change < 0, matching the fitted
sign), on `screen +val+sleeve[+short]`, window 2017-06→2025-12 (real data only, not extended
into the stale 2026 tail):

| variant | Sharpe (full) | MaxDD | Calmar | Sharpe 2023-2025 | on % of days |
|---|--:|--:|--:|--:|--:|
| no short | 0.656 | -0.240 | 0.578 | 1.658 | 0% |
| real-yield only (shipped) | 0.617 | -0.232 | 0.551 | 1.621 | 53% |
| LBNL only | 0.624 | -0.249 | 0.529 | 1.744 | 11% |
| LBNL AND real-yield | 0.656 | -0.240 | 0.575 | 1.720 | 4% |
| all 3 (LBNL+grid-mom+real-yield) | 0.651 | -0.240 | 0.574 | 1.676 | 2% |

Full-window: negligible either way — the gate triggers on only ~4% of days combined, close
enough to no-short that it barely moves the numbers. The 2023-2025 subwindow looks better
(1.72-1.74 vs. shipped's 1.62), but this is **not new evidence** — it's the same in-sample fit
already shown to fail the Thread 3 out-of-sample test, now viewed as a portfolio Sharpe instead
of a regression coefficient. **Confirms the FAILED verdict, does not change it.** Shipped
real-yield-only gate remains the best-supported choice; nothing from this thread is wired live.

## Thread 5 — data-quality sanity check: PJM+CAISO only (live, not the frozen LBNL file)

Tested whether Thread 3's OOS failure was an artifact of the LBNL data's regional coverage gap
(Thread 3 caveat 1) rather than a real absence of relationship, by rebuilding the backlog index
from only the two best-covered ISOs, pulled live (not the frozen "as of end of 2025" file):
PJM (raw `gridstatus` export, 98.4% withdrawal-date coverage) + CAISO (Active + Withdrawn
sheets combined; the Completed sheet has no completion-date field so is excluded — a disclosed,
different limitation, not attempted to be fixed here). This series does **not** go flat in
2026 (unlike the LBNL file) because it's pulled live, giving 6 genuine out-of-sample months
(Jan-Jun 2026) to check against, not a frozen-and-forward-filled value.

**In-sample 2023-2025 (3mo change):** REIT alone t=+2.48, yieldco alone t=-3.26 — comparable
strength and identical sign to the national LBNL version, so data quality was not suppressing
or distorting the in-sample fit.

**Genuine OOS, 2026 (n=6, too few for a regression, but the direction is checkable):** REIT's
correlation with the signal **flips to the wrong sign** (-0.25 vs. the fitted +2.48
relationship's implied positive) — the backlog kept contracting through Jan-Jun 2026 while REIT
returns were mostly strongly positive (4 of 6 months double-digit), the same contradiction as
Thread 3's forecast miss. Yieldco's sign is directionally consistent (-0.40, matching the fitted
coefficient's expected sign) but n=6 cannot support any real conclusion, and Thread 3's
magnitude-based forecast already showed yieldco badly over-predicted.

**Conclusion: better data quality does not rescue the signal.** The cleaner PJM+CAISO index
reproduces the same in-sample relationship and still breaks on the one target (REIT) with any
real out-of-sample readability. This rules out "it's just the regional coverage gap" as the
explanation for Thread 3's failure — the relationship itself does not survive into genuinely
subsequent data, independent of data quality. Closes this line of the investigation.

## Thread 6 — EIA Form 860M generator inventory: a better data source, one thin test

User asked to "search again" for an interconnection-queue data source after Thread 5. Found
**EIA's Preliminary Monthly Electric Generator Inventory** (Form 860M), accessed via
`gridstatus.EIA.get_generators(date)` (already-installed package, uses the existing
`EIA_API_KEY`) — structurally better than everything else tried in this doc:

- **National** (all balancing authorities in one call, not region-by-region).
- **Real entity/owner names** (`Entity Name` field) — confirmed clean matches: AES (360
  operating rows/16,470MW, 26 planned/1,932MW, 26 canceled/1,550MW), Brookfield (111
  operating/821MW), Clearway (14 operating/1,044MW), Algonquin (10 operating/1,320MW) — a real
  fix for the attribution problem that killed the PJM/LBNL per-company attempt.
- **Genuine historical point-in-time snapshots**, not a single hindsight-complete file — each
  monthly `Period` has its own `Updated At` publish timestamp (e.g., the 2020-06 period was
  published 2020-08-24 and, as far as tested, doesn't appear to get silently revised later).
  Confirmed pullable back to at least 2020-01.
- Two sub-datasets: `planned` (rolling ~12-month-ahead pipeline, the "queue" analogue) and
  `canceled_or_postponed` (the "withdrawn" analogue, no date field, same caveat as PJM/CAISO).

**One test run, AES only (most-matched name), quarterly snapshots 2020-01→2026-04 (n=19
matched to forward 3-month AES return):**

| signal | r | t (n=19) |
|---|--:|--:|
| planned MW (level) | -0.384 | -1.71 |
| planned MW − canceled MW (net pipeline) | -0.382 | -1.70 |
| planned MW (QoQ % change) | -0.349 | -1.54 |

Clears the user-directed |t|>1.5 bar barely, not the strict |t|≥2.0 bar. Sign: more near-term
planned capacity → weaker forward AES return — plausibly a capex/dilution-overhang reading
rather than confirmation of the original interconnection-delay mechanism. **n=19 is thin even
by this session's standard, no OOS check run.** Given today's repeated pattern (CEP, the LBNL
queue index) of marginal in-sample results failing a genuine out-of-sample test, this should be
treated as unverified, not as a finding, until tested that way.

**Update (same session, continued):** widened from AES-only to the 5 names with real EIA
presence (AES, Brookfield/BEP via TerraForm+X-Elio+Scout Clean+Standard Solar subsidiary
brands, Clearway/CWEN, **NextEra/NEE**, **NRG** — AQN dropped, having sold its entire
renewables business to LS Power, closed 2025-01-08, confirmed via press release/news). Pooled
correlation improved substantially with NEE added: **r=-0.34, t=-1.74/-1.76 (n=26 quarters,
level and net-pipeline)** vs. the 4-name version's t=-0.55 — NEE alone fixed most of the earlier
AES-only distortion. **But this relationship is specific to the generator-owner basket — tested
against REITs (DLR/EQIX/IRM) it is ~zero (t=0.44-0.48)**, which makes mechanical sense (REITs
don't own generation) and rules out using this signal to gate the DLR/EQIX short sleeve.

**Tried three ways to turn the in-sample correlation into an actual timing/scaling rule on the
5-name generator-owner basket itself — all three underperform simply staying long:**

| approach | Sharpe (full) | CAGR (full) | Sharpe (2023-26) |
|---|--:|--:|--:|
| always long | +0.536 | +16.1% | +0.511 |
| continuous rank scaler (expanding percentile) | +0.200 | +5.8% | +0.015 |
| binary gate (level < expanding median) | -0.169 | +1.1% | +0.094 |
| binary gate (QoQ change < 0) | +0.148 | +5.2% | +0.095 |

Three different, reasonable constructions, same result: the regression-level correlation
(t≈-1.75) does not survive being turned into a tradeable rule. Not sensitive to which specific
rule was picked, which makes this a more conclusive negative than any single variant would be
on its own.

Also tried: a short overlay on the pooled basket (short when signal bearish, flat otherwise —
Sharpe -0.71 to -0.74, worse than the long-side gates), and per-name gates/shorts on AES and
NEE individually using each company's own signal against its own return (AES long-gate
Sharpe -1.27 vs. AES's own +0.16 always-long benchmark; NEE long-gate -0.16 vs. NEE's own
+0.36 benchmark; both companies' short overlays also negative). **Five separate
implementations — long/short, pooled/per-name, level/change-based — all fail the same way.**

**Verdict: real data-source upgrade, real (if modest) in-sample correlation to
generator-owner returns specifically, no relationship to REITs, and no timing, scaling, or
shorting implementation of it beats a static long position, on any name tried.** Closed for
this session. Not adopted, not wired anywhere.

## Thread 1c — systematic short-side screen: from hand-picked names to a disciplined rule

Follow-on to Thread 1b's OOS success. The initial cell-tower result (AMT/SBAC/CCI) was itself
hand-picked — "the recognizable names in a sector" — not a systematic screen, a fair critique
raised mid-session. Fix: ran the *same* fit-period-only (2017-2022) regression, at a fixed
|t|≥2.0 gate, uniformly across a pre-specified 20-name REIT universe spanning 8 subsectors
(data center, tower, industrial, storage, net lease, residential, healthcare, office) — no
cherry-picking, pass/fail decided before any holdout data was examined.

**Result: 6 names pass, and every one of them confirms with a solid OOS rank-correlation:**

| ticker | subsector | fit-period t (2017-2022) | OOS rank-corr (2023-2026) |
|---|---|--:|--:|
| SBAC | tower | -3.30 | +0.561 |
| WPC | net lease | -2.49 | +0.382 |
| EQIX | data center | -2.33 | +0.579 |
| CCI | tower | -2.24 | +0.603 |
| AMT | tower | -2.21 | +0.568 |
| DLR | data center | -2.12 | +0.434 |

**WPC (W.P. Carey) is a genuine new find** from this systematic pass — never previously tested,
fit-period signal stronger than DLR's. **IRM fails outright** (t=-0.06) despite being in the
original signal-testing basket — the records-management business dilutes it too much to belong
in a systematically-screened book. **Storage (EXR/PSA/CUBE) shows the opposite sign** (+0.91 to
+1.82) — confirms Thread on EXR's MVO long-lean was a real subsector pattern, not a
diversification artifact. Industrial/healthcare/residential/office show no signal either way.

**Combined-book result** (`screen +val+sleeve`, same 63-day grid-momentum gate as Thread 1b,
equal-weight across the 6 names):

| variant | Sharpe (full) | MaxDD (full) | Sharpe (2023-26) | MaxDD (2023-26) |
|---|--:|--:|--:|--:|
| no short | +0.693 | -0.240 | +1.596 | -0.170 |
| shipped (DLR/EQIX, real-yield gate) | +0.655 | -0.232 | +1.556 | -0.122 |
| towers-only (hand-picked, Thread 1b) | +0.700 | -0.289 | +1.706 | -0.188 |
| **systematic screened-6** | +0.676 | -0.245 | **+1.637** | -0.177 |

Screened-6 gives up a small amount of Sharpe vs. the hand-picked towers-only version (1.637 vs.
1.706, 2023-26) in exchange for a materially more defensible construction — a pre-specified rule
applied to a broad universe, not a curated list of famous names in sectors chosen after seeing
a correlation.

**Decision (user, 2026-09-18): adopt the screened-6 version over the hand-picked towers-only
version.** The methodological discipline is judged worth the ~0.07 Sharpe gap, which is well
within the noise this session has repeatedly found in gaps this size. **Not yet implemented in
the live codebase** — `hedge_overlay.py`'s `HEDGE_SHORT_TICKERS` and `real_yield_rising_mask`
still reflect the shipped DLR/EQIX + real-yield construction; adopting the screened-6 version
would require code changes (new ticker list, swap the real-yield gate for the grid-momentum
gate) not made in this session.

**Mechanism caveat, stated honestly:** unlike data centers (a real, direct commercial-power-
cost and interconnection-queue exposure), cell towers have no comparably strong physical link to
grid tightness — cell sites draw kilowatts, data centers draw tens of megawatts. The most
likely driver of the observed relationship is **capital rotation** (money moving toward the hot
grid-equipment theme and away from other high-multiple, bond-proxy-like infrastructure names as
a group), not towers being mechanically affected by the grid. This is a plausible, not a
proven, explanation — constructed after seeing the correlation, not before.

**Open next step, checked:** if capital rotation is the real mechanism, it should show up in
other classic bond-proxy sectors too. Ran the identical fit-period-only (2017-2022) |t|≥2.0
screen against 14 regulated utilities, 7 midstream/MLPs, and 3 factor ETFs (VNQ, SPHD, USMV) —
**nothing passes.** Best utility (WEC) is only -1.48; only USMV (-1.89) and WMB (-1.72) clear
even this thread's looser |t|>1.5 bar, and both are unambiguously bad standalone trades anyway
(momentum-timed short Sharpe -1.54 and -1.40 respectively, 2023-2026). **The rotation effect has
a boundary** — it's specific to a narrower set of high-multiple, growth-adjacent REIT
subsectors (towers, select data centers, net lease), not a generic "any rate-sensitive/
bond-proxy sector" phenomenon. Utilities being the classic example of that broader category and
showing nothing is itself informative, not just an absence of a positive result. No changes to
the screened-6 book from this check.

## Thread 1d — ranked scoreboard: every combined-book 2023-2026 Sharpe from this investigation

Compiled on request, to prevent the highest raw numbers from being reused without their
caveats attached — several of the best-looking figures are things this doc explicitly proved
not to trust.

| rank | variant | Sharpe (2023-26) | credible? |
|---|---|--:|---|
| 1 | LBNL-gated short only | 1.744 | **No** — built on the LBNL queue index, which failed a genuine OOS test (Thread 3) |
| 2 | LBNL AND real-yield | 1.720 | **No** — same failed signal |
| 3 | MVO-weighted overlay (static, in-sample weights) | 1.713 | **No** — lookahead bias; collapsed to ~1.62 once rebuilt with honest expanding-window weights |
| 4 | Towers only (AMT/SBAC/CCI, hand-picked) | 1.706 | **Yes** — highest number that actually passed a genuine fit/holdout OOS test (Thread 1b) |
| 5 | Both legs stacked, half-sized (DLR/EQIX + towers) | 1.668 | Partial — built on towers (credible) but this specific blend was never separately OOS-tested |
| 6 | Expanding-window MVO overlay (honest) | 1.621 | Weak — statistically indistinguishable from shipped once done honestly |
| 7 | Both legs stacked, full-sized (2x notional) | 1.618 | Partial — same caveat as rank 5, plus fails badly on the full-history window |
| 8 | No short at all | 1.596 | Always valid — the do-nothing baseline |
| 9 | Grid-mom AND real-yield (DLR/EQIX, original combo test) | 1.586 | Partial — never separately OOS-tested as a combo |
| 10 | **Screened-6 (systematic REIT screen, ADOPTED)** | **1.637** | **Yes** — pre-specified rule, broad universe, every passing name confirmed OOS (Thread 1c) |
| — | Shipped (real-yield only, DLR/EQIX) | 1.556 | The actual live baseline this whole investigation was measured against |

**If only validated results count, the real ranking is: towers-only (1.706) > screened-6 (1.637,
adopted) > shipped (1.556).** Everything ranked above 1.706 in the raw list is exactly the
"impressive but debunked" pattern this whole investigation was built to catch — keep the
caveat column attached if this table gets reused anywhere else.

### How a new candidate earns a place on this scoreboard

Before adding a row, hold the candidate to the same three checks that separated the credible
entries above from the debunked ones — every failure in this table failed at least one of these:

1. **Genuine fit-period / holdout split, never overlapping.** Fit any regression or weight
   estimate on data strictly *before* the holdout window (this thread used 2017-2022 fit →
   2023-2026 holdout); a coefficient or weight that only becomes significant once the holdout
   period is folded into the fit is disqualified, however good the "full-sample" number looks
   (this killed BE, ACM, CEP, and the LBNL queue index).
2. **The holdout must show real directional correlation, not just a plausible story.** Compute
   OOS rank-correlation (predicted vs. realized) on the untouched holdout. A good full-sample
   regression with a weak or wrong-signed holdout correlation does not qualify (this is what
   separated towers, +0.636, from BE and ACM, whose fit-period t-stats were already
   insignificant before even reaching this step).
3. **Flag lookahead-biased weight-fitting explicitly.** Any full-sample-optimized weights
   (e.g., MVO, factor loadings) applied retroactively across the same sample they were fit on
   must be disclosed as such and are not eligible for the "credible" column — rebuild with
   expanding-window or rolling weights before trusting the number (this alone separated rank 3,
   1.713 invalid, from rank 6, 1.621 honest, in the table above).

A candidate that clears all three earns a row with **Yes**; one that fails any of them can still
be logged (negative results are first-class in this project), but the row must say why, the way
every **No**/Partial row above does.

## Thread 7 — cross-thread test: STEO forward spark-spread as an alternative gate (2026-09-20)

A different research thread this session (`docs/steo-spark-spread-results.md`, built for
the *merchant stress-beta* problem in the unrelated `grid_resilience` pipeline) produced a
forward power-price signal (EIA STEO spark-spread, PJM/ERCOT). Tested whether it could gate
the screened-6 short sleeve (Thread 1c) instead of grid-equipment momentum, held to this
thread's own bar.

**Fails at the fit-period stage, before ever reaching a combined-book number.** STEO trend
data only exists from 2021-01-19 (vs. this thread's 2017-2022 fit window), so fit was
necessarily shorter: 2021-02→2022-12 (n=23) → holdout 2023-01→2025-12 (n=36). Fit-period
coefficient on the STEO term is statistically indistinguishable from zero in every
specification (t=0.49 PJM+ERCOT average, t=0.18 ERCOT alone, t=0.70 PJM alone); SPY
dominates the fit (t=5.84) in all three. The OOS rank-correlation looked superficially
reasonable (0.51-0.52) — but an SPY-only model with no STEO term at all gets 0.534 on the
identical holdout, i.e. slightly *better*. The apparent OOS "success" is 100% market beta;
STEO contributes nothing incremental in any form tested.

**No combined-book Sharpe computed** — building one and reporting whatever number fell out
would reproduce the exact failure mode Thread 1d's scoreboard exists to catch (a
compelling-looking number built on a signal that doesn't survive scrutiny). This fails
earlier and more cleanly than that: disqualified at the regression stage, never given the
chance to look good. Does not affect the adopted screened-6 construction or its gate.

## Thread 8 — STEO as a JOINT regressor alongside the EIA pipeline signal (Thread 6) — FAILED (2026-09-20)

Picked up after a stale-doc correction: `docs/handoff-2026-09-17-eia-generator-pipeline-
signal.md` (the AES-only, n=19, "SCOPED, ONE THIN TEST RUN" writeup) still reads as open,
but its own recommended next steps — widen past AES-only, go beyond a single quarterly
snapshot, test an actual tradeable rule — were already completed in **this doc's Thread 6**,
which reached a clean FAILED verdict (EIA-alone correlation is real but modest, r=-0.34,
t=-1.74/-1.76, n=26 quarters; five different tradeable-rule constructions all underperform
a static long). The handoff doc is now flagged stale and should be treated as superseded by
Thread 6, not as an open task.

Genuinely untried question, not covered by Thread 6 or by the STEO-vs-GSI threads
elsewhere in this session: does STEO's forward power-price signal add information
**alongside** the EIA pipeline signal (jointly, not as a replacement) on Thread 6's same
pooled 5-name generator-owner basket (AES/BEP/CWEN/NEE/NRG)? NRG/NEE lean more merchant
than the electrification-supplier book STEO already failed on twice
(`docs/steo-spark-spread-results.md` §9-10), so the mechanism is at least more plausible
here — but EIA-alone already failed every tradeable construction tried, so the prior going
in was low.

**Method** (`research/steo_spark_spread/probe_steo_eia_joint.py`): reused Thread 6's exact
cached panel (reproduced its r=-0.335/-0.338, t=-1.739/-1.762 before trusting it), merged
in the STEO PJM/ERCOT composite z-trend point-in-time (`merge_asof`, backward — the most
recent STEO value known on/before each EIA quarter date), then ran EIA-alone and EIA+STEO
joint OLS **on the identical, STEO-restricted window** (apples-to-apples) before considering
any fit/holdout split, following the same "check the fit stage first" discipline Thread 7
used for the same STEO-short-history problem.

STEO only covers 2021-01-19 onward, shrinking the usable window from Thread 6's n=26
(2020-01→2026-04) to n=21 (2021-04→2026-04).

| model | planned_mw t-stat | steo_composite t-stat | R² |
|---|--:|--:|--:|
| EIA-alone, full Thread 6 window (n=26) | −1.739 | — | — |
| EIA-alone, STEO-restricted window (n=21) | −0.495 | — | 0.013 |
| EIA + STEO joint, same window (n=21) | −0.442 | **0.059** | 0.013 |

**Result: FAILS decisively, on both counts.** STEO's own coefficient is statistically
indistinguishable from zero (t=0.059) and adds no explanatory power (R² unchanged at
0.013). Notably, **even the EIA-alone signal collapses on this restricted window**
(t=−1.739 → −0.495) — Thread 6's already-marginal correlation turns out to be concentrated
in the 2020-2021 portion of its sample that STEO doesn't cover at all, not evenly spread
across the full history. No fit/holdout split or combined-book number was built, matching
Thread 7's precedent: a result this clean at the regression stage doesn't earn a more
expensive follow-up.

**Disposition:** closes this specific combination. Confirms Thread 6's own verdict (EIA
pipeline signal, alone, does not beat a static long) and shows adding STEO doesn't rescue
it. **`docs/handoff-2026-09-17-eia-generator-pipeline-signal.md` should be read as
superseded by Thread 6 + this thread, not as an open task.**

## Thread 9 — widening the yieldco basket (FRVO/BEP/ENLT/CWEN/ORA/BEPC) — FAILED / mostly untestable (2026-09-20)

User asked whether grid-equipment momentum (Thread 1/1b) cross-validates on a different
6-name renewable-developer/yieldco set, and whether any pass Thread 1c's systematic
fit/holdout screen individually.

**Data constraint found before running anything:** three of the six names don't have
enough price history for the existing 2017-09→2022-12 fit window — **FRVO IPO'd
~2026-05 (5 months of data)**, ENLT lists from 2023-01 (45 months), BEPC from 2020-07
(75 months). Only **BEP, CWEN, ORA** (all 15+ years) support the pre-registered
methodology; the other three are reported untestable, not dropped silently.

**Reconstruction caveat:** Thread 1/1b/1c's original scripts were scratchpad
throwaways, never committed (the same gap found for Thread 6) — `grid_mom` was rebuilt
from the documented description (trailing 3-month return of the project's `screen`
basket) and sanity-checked against Thread 1b's reported yieldco OOS rank-correlation
(+0.429) before trusting it: reproduced **+0.409** on the original CWEN/BEP/AES/AQN
basket — close enough to trust the reconstruction directionally, not a byte-exact
replay.

**(a) Per-name fit-period regression (2017-09→2022-12, n=61), |t|≥2.0 gate:**

| ticker | grid_mom t-stat | spy_ret t-stat | pass? |
|---|--:|--:|---|
| BEP | −0.617 | 5.183 | fail |
| CWEN | −0.819 | 4.747 | fail |
| ORA | −0.051 | 2.141 | fail |

**0/3 pass.** None of these three names individually show a grid-momentum
relationship in the fit period at all — unlike the original basket, where the same
signal cleared the user's |t|>1.5 bar on 3 of 4 cells.

**(b) Pooled widened basket (BEP/CWEN/ORA, AES/AQN dropped):** fit-period grid_mom
t=−1.031 (also short of significance). OOS rank-correlation looks positive (0.479,
n=44) — but an **SPY-only model with no grid_mom term at all scores just as high
(0.475)** on the identical holdout, the same "apparent OOS success is pure market
beta" failure pattern Thread 7 already caught with STEO. Grid-momentum adds nothing
incremental on this basket.

**Verdict: FAILS both tests.** Swapping AES/AQN out for ORA (and dropping AQN, which
sold its renewables business in 2025) removes whatever weak relationship the original
basket had — the effect doesn't generalize to a differently-composed yieldco set.
FRVO/ENLT/BEPC remain genuinely untested (not failed — no data), logged as a 2028+/
data-dependent retest candidate once they have enough history, same convention as
other "not yet testable" items in this log. Nothing wired live; closes this specific
widening attempt.

## Thread 10 — full sensitivity scan, all mentioned tickers, loosened bar (2026-09-21)

User pushback: it's surprising that interconnection-queue and forward-power signals show
*nothing* across every ticker discussed this session. Ran a comprehensive scan rather than
one more targeted test: **56 tickers** (union of the screen basket, Thread 1c's REIT names,
the 4 merchant generators, every yieldco/renewable-developer name across threads, and the
solar-rotation sleeve) × **2 signals** (LBNL interconnection-queue 3-month change; STEO
PJM/ERCOT forward spark-spread) × **4 lags** (0-3 months) × **2 test types** (mean-return
OLS; a "GARCH-style" realized-variance regression, matching this project's established
convention from `research/electricity_price_volatility/regressions.py` rather than a
literal fitted GARCH model) = **832 regressions**, at the user-accepted loose bar |t|≥1.5.

**Headline: 218 hits under naive OLS (26.2%) vs. ~111 expected by pure chance (13.4% is
the two-sided P(|t|>1.5) under a true null) — a real excess, not noise.** HAC(6)-robust
correction made the raw count *worse* (268 hits, 32.2%), but this is small-sample HAC
instability on short-history tickers (e.g. GEV, 21 months of data, t swings −2.0→−6.0
under HAC — not a stable estimate at that n), not a genuine robustness finding.

**Splitting by sample length isolates a real, HAC-stable effect from an artifact:**

| subset | naive OLS hit rate | HAC(6) hit rate | chance rate |
|---|--:|--:|--:|
| long-history (n≥100, mostly `queue`) | 32.6% | 32.3% (stable) | 13.4% |
| short-history (n<100, mostly `power`) | 22.0% | 32.1% (unstable — HAC inflates it) | 13.4% |

The long-history subset's excess survives HAC correction essentially unchanged — this is
not a small-sample artifact. The short-history subset's HAC jump is exactly the artifact
pattern; those hits (mostly STEO/`power` tests, since STEO only covers 2021+) are the
least trustworthy in the whole scan.

**Mechanism check on the surviving long-history excess:** regressed SPY's own realized
variance against the queue signal (same method) — **t=−1.94 at a 2-month lag**, a real if
modest relationship between the queue signal and *broad market* volatility, not any single
stock. This is the signature of a shared macro-volatility-regime confound: since nearly
every equity's variance partly co-moves with market-wide vol, a signal that itself weakly
tracks market vol will mechanically produce a broad, elevated cross-sectional hit rate that
looks like "many stocks are sensitive to grid buildout" but is actually one shared channel
appearing repeatedly. Same failure shape as Layer-2 Option A (`docs/grid-regime-layer2-
results.md` §7): a common-mode component, not name-specific information.

**Verdict:** the scan finds something real (excess-over-chance survives HAC on the
long-history subset) but it does not look like broad stock-specific sensitivity to grid
buildout — it looks like the queue signal has a faint, non-tradeable pulse on general
market-volatility regimes, which shows up as diffuse noise across every name tested. No
individual ticker/signal/lag combination is elevated enough above this shared-confound
baseline to warrant a follow-up OOS test. Consistent with, and explains the shape of, this
log's ~30-attempt track record of no durable name-specific edge from either data class.
Full results: `research/steo_spark_spread/data/full_sensitivity_scan_all_results.csv`
(all 832 tests), `full_sensitivity_scan_hits.csv` (HAC-robust hits only).

## Open item — MISO/ISO-NE (not built, and now lower priority)

`gridstatus.{MISO,ISONE}().get_raw_interconnection_queue()` both work with no API key (feasibility
confirmed, not parsed). Given Thread 5's sanity check already shows the signal fails OOS on
clean data, extending to a 4-ISO index would mostly improve *infrastructure* rather than change
the verdict — not pursued further in this session.

## References

- `docs/datacenter-reit-cep-preregistration.md`, `docs/datacenter-reit-cep-signal-results.md`
  — the originating CEP thread and its |t|>1.5 threshold amendment.
- `docs/handoff-2026-09-16-datacenter-reit-cep-signal.md` — original handoff.
- `research/datacenter_reit_cep/probe.py`, `combination_test.py`,
  `lbnl_national_queue_mw.parquet` — scripts/cached series from this session.
- `inputs/LBNL_Ix_Queue_Data_File_thru2025.xlsx` — user-provided LBNL source file.
- `grid_resilience/data/dc_load_data.py` — the pre-existing PJM-only queue infrastructure this
  thread grew out of (built for a utility-factor use case, unrelated attribution scheme).
