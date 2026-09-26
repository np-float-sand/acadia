# Merchant stress-beta orthogonalization, fix #5 — fundamentally-anchored beta (results, 2026-09-18, corrected 2026-09-20)

**Status: FAILED both variants. Closes untried-tasks-log item 3 — all 5 candidate fixes now tested.**
Probe: `research/merchant_fundamental_beta/probe_fix5_fundamental_beta.py`. Fundamentals:
`research/merchant_fundamental_beta/fundamentals.py`.

**CORRECTION (2026-09-20):** the numbers originally reported here (baseline Sharpe 1.182,
Variant A 0.458, Variant B 0.161) were computed inside an isolated agent worktree whose
cached price data did not match the main checkout — caught when an unrelated follow-up
test (`docs/steo-spark-spread-results.md` §7) independently re-ran the identical
2023-01-01→2025-12-31 window and got a materially different **benchmark (XLU)** return
(9.52% ann., Sharpe 0.336) than this probe's original run recorded (2.51% ann., Sharpe
−0.150) — an unambiguous, unexplainable-by-two-days-passing mismatch for the same public
ETF over the same historical window. Re-ran all three variants directly in the main
checkout; the corrected numbers are below. **The bottom-line verdict is unchanged (both
variants FAIL)**, but the absolute magnitudes are different and — importantly — the
*shape* of the result changes: IC actually improves for both variants at the 21d horizon
(it had appeared to worsen in the original, bad-data run), while realized Sharpe still
collapses. See §7 for the full erratum.

---

## TL;DR

- Replaced the rolling single-ISO trailing-price stress beta for VST/NRG/CEG/TLN with a
  beta anchored to each company's own disclosed fundamentals (10-K fuel mix, ISO/region
  capacity or revenue split, spot-vs-hedged exposure %), hand-extracted with a citation
  for every number.
- Two variants, tested separately (same discipline as fixes #1/#2/#4): **Variant A**
  (`iso_blend`) replaces the single-ISO GSI regressor with a blend weighted by each
  company's disclosed ERCOT/PJM capacity or revenue split. **Variant B**
  (`iso_blend_shrink`) additionally scales the resulting beta by the disclosed
  spot-exposure fraction (1 − hedged %).
- **Both variants make full-sample Sharpe worse, monotonically — baseline 0.120 → A
  −0.032 → B −0.584.** IC@21d t-stat actually *improves* for both (0.34 → 0.499 → 0.721),
  but IC@63d degrades for both (1.207 → 0.393 → −0.570), and Variant B's stress-period
  Sharpe flips sharply negative (0.220 → −0.504).
- **Pre-registered gate: FAILED on condition (b) for both variants; condition (a) is a
  genuine split, not a clean fail.** (a) IC@21d improves for both variants — passes; IC@63d
  does not — fails. (b) Full-sample Sharpe does not hold for either variant — fails
  decisively. Net: gate not met. Do not adopt either variant.
- **Methodology note, unchanged from the original writeup:** this probe's window
  (2023-01-01→2025-12-31) is not the same window fixes #1/#2/#4 were judged on, so
  absolute Sharpe/t-stat levels are not comparable across probes — the within-probe
  comparison (baseline vs. A vs. B, all three run identically) is what the verdict rests
  on, and that comparison is now independently reproduced (see §7).

---

## 1. What was hand-extracted, and how complete it is

Every number below has a 10-K citation (ticker, fiscal year, filing date, section) in
`research/merchant_fundamental_beta/fundamentals.py`. Granularity is **annual only** —
none of the four companies discloses fuel mix, hedge ratios, or regional segment
capacity/revenue at quarterly resolution in a usable form; this matches the task's own
expectation going in.

| Ticker | Years covered | Fuel mix source | ISO/region split source | Spot-exposure % source |
|---|---|---|---|---|
| VST | 2023–2025 | 10-K capacity table (text) | Revenue-segment note (Texas/East/West[/Sunset]) | **Tier A** (real, disclosed hedge %) for 2024/2025; **Tier A-carry** for 2023 (no 2023-specific figure found; carried from the FY2023 10-K's own forward 2024/25 estimate — flagged, not fabricated) |
| NRG | 2023–2025 | 10-K capacity table (text) | Revenue-segment note (Texas/East/West-Services-Other) | **Tier C-prior** — no clean "%-hedged" figure disclosed in any of 3 years searched (NRG describes hedging via VaR/MtM, never as a single %); falls back to the pre-existing, itself-unverified `pass_through=0.85` placeholder in `utility_node_map.py` |
| CEG | 2023–2025 | 10-K pie chart, **image only** (read via vision, not machine-extractable text) | Capacity-based 5-segment table (Mid-Atlantic/Midwest/NY/ERCOT/Other) — revenue-based split not disclosed at this granularity | **Tier C-prior** — no clean %-hedged figure; falls back to `pass_through=0.80` placeholder |
| TLN | 2024–2025 only | 10-K plant-level MW-ownership table (most precise of the four — no bundled-segment aggregation) | Same plant-level table | **Tier B-policy** — a disclosed *target range* (60–80% hedged), not an audited realized figure; identical language both years |

Notable gaps and dating issues, disclosed rather than papered over:
- **TLN has no usable fundamentals before its FY2024 10-K** (filed 2025-02-28) — the
  first ~20 months of its post-Chapter-11 re-listed trading history (from 2023-06-02)
  fall back to the original single-ISO beta by construction (no look-ahead).
- **NRG's coal is 51% of generation *capacity* but under 5% of *revenue*** every year
  checked — a capacity/revenue divergence worth flagging on its own (low utilization,
  capacity-payment-driven economics), and a reminder that "fuel mix by MW" is not the
  same statement as "fuel mix by economic exposure."
- **CEG's FY2025 10-K explicitly excludes the Calpine acquisition** (~23 GW gas/geothermal,
  closed 2026-01-07, after CEG's FY2025 year-end but before the 10-K's filing date) — CEG's
  fundamentals-anchored beta as built here runs a full fiscal year behind that deal.
  Similarly, **NRG's FY2025 10-K excludes the LSP Portfolio + CPower acquisitions**
  (closed 2026-01-30, roughly doubling NRG's generation capacity) for the same reason.
- Two of four names (NRG, CEG) have **no genuine disclosed spot-exposure number at all**
  and fall back to a placeholder this project had already flagged as unverified before
  this task — Variant B's "shrink" step is only running on real disclosed data for 2 of
  4 names (VST, TLN).

## 2. Beta construction

**Variant A (`iso_blend`):** replace the ticker's single-ISO GSI regressor with
`blended_gsi = w_ERCOT · GSI_ERCOT + w_PJM · GSI_PJM`, where the weights are the
company's disclosed ERCOT/PJM-proxy capacity-or-revenue split, renormalized to sum to 1
over the two ISOs this pipeline's GSI actually models (the disclosed "OTHER" region
fraction — CAISO/NYISO/ISO-NE/MISO/WECC/Canada — is unmodeled in both baseline and this
variant, so the comparison stays like-for-like). No change to the beta's scale.

**Variant B (`iso_blend_shrink`):** Variant A's regression, then the slope is multiplied
by the ticker-year's disclosed spot-exposure fraction (1 − hedged %), so a name entering
a year >90% hedged (VST in 2024, spot-exposure 2%) contributes proportionally less beta
to the factor than a name with a lower hedge ratio (TLN, spot-exposure 30% both years on
its policy-range disclosure).

Both variants fall back to the original single-ISO, unshrunk beta for any (ticker, date)
pair predating that ticker's first available 10-K filing date, keyed to the actual filing
date — no look-ahead. Method otherwise identical to fix #2's probe: monkeypatch
`conditional_beta.compute_stress_betas`, run the real, unmodified
`grid_resilience.main.run()` end to end, compare the same metrics fixes #1/#2/#4 used.

## 3. Results (2023-01-01 → 2025-12-31) — CORRECTED, verified in main checkout 2026-09-20

| Metric | Baseline (single-ISO) | Variant A (iso_blend) | Variant B (iso_blend + shrink) |
|---|---|---|---|
| Full Sharpe | **0.120** | −0.032 | −0.584 |
| Full ann. return | 6.4% | 3.5% | −3.0% |
| Full max DD | −31.5% | −26.1% | −25.0% |
| Stress-period Sharpe | 0.220 | 0.091 | **−0.504** |
| Calm-period Sharpe | −2.890 | −2.840 | −2.500 |
| IC@21d (mean / t-stat) | 0.0225 / 0.34 | 0.0313 / **0.499** | 0.0370 / **0.721** |
| IC@63d (mean / t-stat) | 0.0812 / **1.207** | 0.0245 / 0.393 | −0.0277 / −0.570 |

Benchmark (XLU) and EW-universe returns are identical across all three rows (9.52% /
Sharpe 0.336, and 12.91% / Sharpe 0.561, respectively) — confirming all three variants
ran against the same underlying price data, the specific check that caught the original
error. (Full metric dumps: re-run log preserved in this session; original files at
`research/merchant_fundamental_beta/results_summary.json` and `run_log.txt` are the
STALE, uncorrected numbers and should not be used — kept only for the erratum record in
§7, not as a source of truth.)

Full-sample Sharpe degrades monotonically and by a lot (0.120 → −0.032 → −0.584) — the
verdict this project cares about most doesn't change from the original (wrong-data)
writeup. What *does* change: IC@21d genuinely **improves** for both variants rather than
worsening, while IC@63d and stress-period Sharpe both degrade, most severely for Variant
B. This is a different, more nuanced shape than "everything gets worse" — closer to the
same up-IC/down-Sharpe disconnect seen in the STEO spark-spread result
(`docs/steo-spark-spread-results.md`), suggesting this may be a more general pattern in
this pipeline (a regressor change can sharpen short-horizon rank information while
degrading the realized long/short book) rather than something specific to either fix.

## 4. Pre-registered gate — FAILED

Gate (identical to fix #2's, for direct comparability):
(a) IC@21d t-stat improves vs. the single-ISO baseline; (b) full-sample Sharpe does not
get worse, and the merchant names' rolling beta shows measurably fewer/smaller sign
flips across the four documented narrative-flip windows.

**(a) PASSES for both variants at 21d, FAILS at 63d** — IC@21d t-stat rises from 0.34 to
0.499 (A) and 0.721 (B); IC@63d falls from 1.207 to 0.393 (A) and −0.570 (B, wrong sign).
A genuine split result, not a clean pass or fail.

**(b) FAILS decisively for both variants** — full Sharpe falls from 0.120 to −0.032 (A)
and to −0.584 (B), both now negative. Rolling-beta sign-flip stability at the four
narrative-flip windows was not separately re-examined given (b) already fails cleanly.

**Net: gate not met** — condition (b) alone is enough to fail this per the same "AND"
rule used throughout this fix series, regardless of the partial (a) result.

## 5. Why it failed

The diagnosis motivating fix #5 (`docs/grid-stress-signal-evidence.md` §5) was that the
whipsaw comes from treating each merchant name as 100%-exposed to a single ISO's stress
index when all four are genuinely multi-region, multi-hedge businesses. That diagnosis
may still be directionally correct as *description* — CEG is disclosed as ~66-70% PJM
capacity with a real ERCOT and NY footprint, not the pure single-ISO story the baseline
regressor assumes — but **turning the accurate multi-region picture into a *better*
regressor did not translate into a better realized book**, for a reason consistent with
this project's other findings: a slow-moving, annually-updated, capacity- or
revenue-weighted blend can sharpen short-horizon *ranking* information (the IC@21d
improvement is real) without that improvement surviving into portfolio construction,
turnover, and the specific stress-day windows the backtest weights most heavily. Variant
B's added spot-exposure scaling makes this worse, not better — likely because two of the
four names (NRG, CEG) are scaled by an unverified placeholder rather than real disclosed
data (§1), injecting a second source of noise on top of the ISO-blend itself.

Combined with fix #2's finding (trailing realized price is the wrong kind of regressor
entirely) and fix #4's finding (recency-weighting an already-thin stress-day sample
trades bias for ruinous variance), this closes out the fix series with a consistent
picture: **the mechanism driving the VST/NRG/CEG/TLN whipsaw is a forward-looking market
narrative, and every proxy tried so far that is anchored to backward-looking data
(realized price, disclosed 10-K fundamentals) either fails outright or, at best, improves
short-horizon rank information without improving the realized book.** The one remaining
structurally different idea in this project's research trail — a genuinely
*forward-looking* input (STEO's forecast power-price series, untried-tasks-log item 6) —
showed exactly this same up-IC/down-Sharpe shape, reinforcing that this may be a
pipeline-level pattern rather than a mechanism-specific one.

## 6. Disposition

- Fix #5: **FAILED, do not adopt either variant.**
- **All 5 candidate fixes for the merchant stress-beta whipsaw are now tested.** #1 (XLU/XNG
  filter) real but too small to matter (NOT ADOPTED); #2 (trailing realized power-price
  orthogonalization) FAILED; #3 was redirected into the STEO forward-power idea (item 6);
  #4 (recency-weighted window) FAILED decisively; #5 (fundamentally-anchored beta) FAILED.
  Untried-tasks-log item 3 is closed — no fix in this series clears its gate.
- Kept as reusable infra regardless of the negative result: the hand-extracted,
  cited fundamentals panel (`fundamentals.py`) — real, audited 10-K data with honest
  tier labels — for any future task that needs VST/NRG/CEG/TLN fuel mix, region split, or
  hedge disclosure (e.g. a future fundamentals-driven cross-sectional cut, distinct from
  this time-series beta application).

## 7. Erratum (2026-09-20) — how the original numbers were wrong, and what it means

**What happened:** the original probe ran to completion inside an isolated agent
worktree (`.claude/worktrees/agent-a8d0326c2ba409d6a`) on 2026-09-18. Its cached equity
price data, for reasons not fully diagnosable after the fact (the worktree has since been
removed), produced a materially different XLU benchmark return over the identical
2023-01-01→2025-12-31 window than the main checkout's cache: 2.51% ann. (Sharpe −0.150)
in the worktree vs. 9.52% ann. (Sharpe 0.336) in the main checkout. That gap is far too
large to be explained by two days of new data for a public ETF, and points to the
worktree's cache being incomplete, partially stale, or otherwise divergent — plausibly
because `grid_resilience/data/cache/` (788MB, ~1,600 files) is entirely untracked by git,
so an isolated worktree's copy of it depends on whatever the harness's snapshot timing
happened to capture, independent of the actual code (confirmed identical: zero commits
touching `grid_resilience/` exist between the worktree's base and the main checkout).

**How it was caught:** an unrelated same-session follow-up
(`docs/steo-spark-spread-results.md` §7, testing whether the STEO spark-spread fix could
improve 2023-onward Sharpe) independently re-ran a plain, unmodified baseline on the
*identical* window and got a full Sharpe of 0.120 — inconsistent with this doc's
originally-reported baseline of 1.182 for what should have been the same computation.
That inconsistency was the trigger for re-running fix #5 directly in the main checkout,
which reproduced the STEO follow-up's 0.120 baseline exactly and produced the corrected
Variant A/B numbers in §3.

**What did *not* change:** the qualitative verdict (both variants FAIL the gate) and the
5-fix-series disposition (§6). **What did change:** every absolute number in this doc,
and the shape of the result — IC@21d genuinely improves for both variants rather than
worsening (§3-4), a materially different, more nuanced finding than the original writeup
described.

**Process implication:** results computed inside an isolated agent worktree should be
treated as provisional until either (a) a sanity-check figure with a known, independently
verifiable value (like a benchmark ETF's return over a stated window) is spot-checked
against the main checkout, or (b) the result is reproduced directly in the main checkout
before being written up as final. This applies retroactively to nothing else in this
session's other worktree-computed results — the STEO spark-spread probe's full-sample
baseline was independently confirmed to match the historical fix #1/#2/#4 numbers exactly
(a de facto sanity check that happened to already be built into that probe's own
comparison-table code), and the satellite-sleeve diversification test's numbers have not
been independently re-verified but did not depend on this specific 2023-2025 grid_resilience
cache state (different universe, different pipeline). Going forward, any new
worktree-computed grid_resilience result should include a benchmark-return spot-check
before being treated as final.
