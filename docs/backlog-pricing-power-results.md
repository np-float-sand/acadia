# Backlog pricing-power language — results (2026-09-21)

**Verdict: FAILED. NOT ADOPTED.** RESEARCH-LOG open item 12
(`docs/handoff_2026-09-01-grid-buildout-long-short.md` §8.3(b)). Pre-registration:
`docs/backlog-pricing-power-preregistration.md` (written and locked before any filing was
scored). This is the *pricing-power language* mechanism, deliberately distinct from every
backlog-*size* measure already tried in this project (backlog growth — FAILED,
`docs/backlog-surprise-factor-results.md`; RPO/coverage ratio — FAILED,
`docs/backlog-coverage-signal-results.md`). No backlog-size measure was used, even as a control,
per the task constraint.

## TL;DR

- Scored all 986 10-K/10-Q filings (2017-01 -> 2026-08) for the 27-name `screen` universe against
  a locked 12-phrase keyword list, MD&A-scoped where extractable.
- **Pre-registered primary metric (keyword density per 1,000 MD&A words) fails the gate on both
  horizons.** 1-month: pooled `t=-1.05` (doesn't even clear the pooled bar). 3-month: pooled
  `t=-2.09` clears the pooled bar, but the 2019-2022 sub-window (`t=-1.26`) doesn't clear the
  required `|t|>=1.5` -- gate requires **both** sub-windows, so this fails too.
- Two non-primary diagnostics (raw count, distinct-phrase breadth -- pre-registered as
  *not* the test) *do* clear the full gate at 3 months, sign consistently negative in
  pooled + both sub-windows. Reported honestly, but **not used to override the verdict** -- that
  would be exactly the after-the-fact metric-swapping the pre-registration exists to block.
- **Where the effect is real, it's the wrong sign and looks like sparsity/size confound, not
  pricing power:** more pricing-power language predicts *worse*, not better, forward returns, and
  the passing diagnostics are the two length-unnormalized measures -- length correlates with filer
  size/complexity, not the pricing-power construct this thread was testing for.
- **Severe keyword sparsity:** 50.2% of all 986 filings score zero on every one of the 12 phrases;
  3 of 27 tickers (GEV, MTZ, POWL) score zero on *every* filing in the entire 9-year history; 3 of
  12 phrases (`backlog extends into`, `backlog margin`, `firm pricing`) never fire once across the
  whole panel; one phrase (`price increase`) supplies 66% of all hits. This is a thin signal by
  construction, as flagged in the pre-registration before scoring.

## 1. Universe

`electrification_strategy.universe.members("screen", pd.Timestamp("2026-08-31"), prices)`, 27
names, all with a domestic 10-K/10-Q filer CIK (no 20-F foreign private issuers, including ETN
despite its Irish domicile):

```
AEIS, AME, APG, ATKR, AYI, CARR, EME, EMR, ENS, ETN, FIX, GEV, HUBB, IESC, JCI, MOD,
MTZ, MYRG, NVT, POWL, PRIM, PWR, RRX, SPXC, TT, VRT, WCC
```

No name was excluded for filer-type reasons. GEV (GE Vernova, spun off April 2024) has only 10
filings / 28 months of panel history by construction, not a data gap.

## 2. Keyword list (locked before scoring)

Same 12 phrases as the pre-registration, unchanged:

| Bucket | Phrases |
|---|---|
| Lead-time / scarcity | `lead time`, `extended lead`, `backlog extends into` |
| Price-escalation / pass-through | `price escalation`, `escalation clause`, `surcharge`, `pass-through`, `pricing action` |
| Margin / backlog-quality | `book-to-bill`, `backlog margin`, `firm pricing`, `price increase` |

Case-insensitive substring count on the extracted MD&A text (10-K: Item 7->7A/8; 10-Q: Item
2->3/4; whole-document fallback, flagged, when no header match -- see section 4).

## 3. Build

`research/backlog_pricing_power/`:
- `cik_map.py` -- 27-ticker CIK lookup (SEC `company_tickers.json`).
- `edgar_filings.py` -- submissions-API filing list + primary-document fetch, generalized from
  the 4-ticker hardcoded pattern in `research/merchant_fundamental_beta/{fetch_filing_list.py,
  fetch_10k.py}` (same endpoints, same 429 backoff, same `User-Agent`) rather than rebuilt from
  scratch, per the task's reuse instruction. Disk-cached under `_cache/`.
- `keywords.py` -- the locked list above, importable so scoring can't silently drift from it.
- `score.py` -- MD&A extraction (regex, last-header-match heuristic to skip table-of-contents
  hits, 200-word floor before accepting a match) + keyword density scoring.
- `run_fetch_score.py` -- orchestrator; fetched and scored all 986 filings
  (`_cache/filing_scores.parquet`).
- `build_panel.py` -- point-in-time monthly panel (`known_date = filingDate + 1 day`,
  forward-filled).
- `ic_test.py` -- the rank-IC test, same convention as `research/narrative_mispricing/
  spike_a_ic_test.py` (monthly cross-sectional Spearman, mean/t across the monthly series).

## 4. MD&A extraction coverage

Overall 88.4% of the 986 filings extracted a real Item-7/Item-2-scoped MD&A section. The
remainder fell back to whole-document scoring (flagged `mdna_extracted=False`) because those
filers write MD&A as a cross-reference stub ("Information required by this Item is presented
in...") pointing to a same-document "Financial Review"-style section further on, rather than
inlining it directly after the Item 7 header -- a large-cap annual-report convention, not a bug
in a handful of names:

| Ticker | Extraction success | Ticker | Extraction success |
|---|---|---|---|
| EMR | 23.1% | JCI | 76.9% |
| TT | 25.6% | EME | 76.9% |
| CARR | 61.5% | ETN | 74.4% |
| GEV | 70.0% | VRT | 93.9% |
| APG | 88.5% | PWR | 94.9% |

All other 17 names: 97-100%. Whole-document fallback filings score materially *lower* density
(mean 0.079/1k words) than successfully MD&A-scoped filings (mean 0.405/1k words) despite
slightly *higher* raw counts (2.80 vs 1.18) -- diluted by ~4.5x more words on average (30,592 vs
6,803). This is a real, honest limitation: for the 6 large diversified industrials above, the
signal is systematically damped relative to what a true MD&A-only read would show.

## 5. Point-in-time panel

99 months (2018-06 -> 2026-08 build range; 2019-01 -> 2026-06 test range), forward-filled from
each ticker's most recently *known* filing (`filed + 1 day`). Every month in the test range has
>=8 tickers with a non-null score (min cross-section requirement never binds).

## 6. Rank-IC results

Monthly cross-sectional Spearman IC, signal known as of month t vs. forward return; mean/t
across the monthly IC series. **Bold** = clears `|t|>=2.0` (pooled) or `|t|>=1.5` (sub-window).

### Primary metric -- density per 1,000 MD&A words (the pre-registered test)

| Horizon | Window | n months | mean IC | t-stat | % negative |
|---|---|---|---|---|---|
| 1m | Pooled | 90 | -0.023 | -1.05 | 58.9% |
| 1m | 2019-2022 | 48 | +0.002 | +0.06 | 52.1% |
| 1m | 2023-2026 | 42 | -0.050 | -1.71 | 66.7% |
| 3m | Pooled | 89 | -0.046 | **-2.09** | 64.0% |
| 3m | 2019-2022 | 48 | -0.039 | -1.26 | 62.5% |
| 3m | 2023-2026 | 41 | -0.055 | -1.71 | 65.9% |

**Gate: FAIL on both horizons.** 1m never clears the pooled bar. 3m clears the pooled bar but
the 2019-2022 sub-window (`t=-1.26`) misses the required `|t|>=1.5` -- the gate requires **both**
sub-windows, exactly the "single-window win doesn't survive a regime split" failure mode this log
has hit repeatedly (category-demand, STEO). Where it's non-trivial, the sign is negative -- more
pricing-power language predicts *worse*, not better, forward returns.

### Diagnostics -- pre-registered as non-primary, reported for transparency, not used for the verdict

| Score | Horizon | Window | n months | mean IC | t-stat |
|---|---|---|---|---|---|
| raw total count | 3m | Pooled | 89 | -0.057 | **-2.73** |
| raw total count | 3m | 2019-2022 | 48 | -0.057 | **-1.93** |
| raw total count | 3m | 2023-2026 | 41 | -0.057 | **-1.92** |
| distinct phrases (0-12) | 3m | Pooled | 89 | -0.052 | **-2.47** |
| distinct phrases (0-12) | 3m | 2019-2022 | 48 | -0.051 | -1.67 |
| distinct phrases (0-12) | 3m | 2023-2026 | 41 | -0.054 | -1.83 |

Both length-unnormalized diagnostics technically clear the full pooled+both-sub-windows gate at 3
months (sign consistently negative, magnitude stable ~-0.05 to -0.06 across every cut -- this is
not noise, it's a real, repeatable pattern). **This is not treated as a PASS.** These are exactly
the two metrics the pre-registration named as confounded by raw filing length before any score
was computed (section 2: "Raw count and... breadth measure are carried as diagnostics, not the
primary test" -- because longer filings mechanically rack up more incidental phrase hits without
saying anything denser about pricing power). That the *unnormalized* cuts pass while the
*length-normalized* primary metric doesn't is itself informative: whatever cross-sectional
information exists here tracks filing/company size and complexity, not the density of
pricing-power language -- a different, less interesting construct than the one this thread
pre-registered to test, and still the wrong sign versus the working hypothesis ("more
pricing-power language -> better forward returns").

## 7. Why it fails -- mechanism

1. **Wrong sign, when non-trivial.** Every non-zero-ish result across every cut is negative. If
   real, this says elevated pricing-power/scarcity language coincides with worse, not better,
   forward performance -- plausibly reverse causality (companies escalate this language *after*
   margin pressure or supply-chain strain is already biting and already priced in, not as a
   leading indicator of pricing leverage), not the hypothesized mechanism. Not chased further --
   pre-registered gate already fails regardless of interpretation.
2. **Severe keyword sparsity.** 50.2% of all 986 filings score zero on all 12 phrases. 3 of 27
   tickers (GEV, MTZ, POWL) never mention a single phrase in any filing across the full
   2017-2026 history. 3 of 12 phrases never fire once in the entire panel (`backlog extends
   into`, `backlog margin`, `firm pricing` -- precisely the most specific, "backlog-quality"
   phrases from the handoff's own suggested language). `price increase` alone supplies 66% of
   all hits -- the score is close to a single generic-phrase counter, not a 12-phrase composite,
   which is a much weaker basis for cross-sectional differentiation than the pre-registration
   intended.
3. **Filing-format heterogeneity dilutes the signal for large-caps.** 6 of 27 names (notably
   EMR, TT, CARR) fall back to whole-document scoring more than a third of the time because their
   10-Ks use a cross-reference-stub MD&A pattern; whole-document density is diluted ~5x relative
   to a true MD&A-only read for those filings (section 4). This doesn't explain the sign, but it
   adds noise that works against finding anything even if the mechanism were real.
4. **27-name breadth was the fix for the original 9-name thin-sample objection** (per the
   handoff's "prior: moderate-low... breadth is thin" caveat) **and it didn't help.** This was a
   real test of the wider `screen` universe, not the original 9-name grid_equipment_basket -- the
   negative result isn't attributable to small-sample noise the way earlier 9-name attempts could
   plausibly be.

## 8. Disposition

**FAILED / NOT ADOPTED.** Close RESEARCH-LOG item 12. `research/backlog_pricing_power/` kept as
a throwaway reference build (reusable EDGAR filing-list + MD&A-extraction machinery for any
future text-mining probe on this universe) but nothing wired into `electrification_strategy` or
`grid_equipment_basket`. Do not retry with a re-tuned keyword list on this same panel -- that
would be exactly the after-the-fact tuning this pre-registration exists to prevent; a genuinely
new keyword construction would need a fresh pre-registration and, given today's sparsity finding,
should budget for the fact that specific contractual/backlog-quality phrases (escalation clauses,
firm pricing, backlog-extends-into) essentially do not appear in this universe's MD&A prose at
all -- a broader net (analyst-call transcripts, press releases) would be a different data source,
not a keyword-list fix.
