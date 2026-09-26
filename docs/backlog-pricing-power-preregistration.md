# Backlog pricing-power language — pre-registration (2026-09-21)

**Written before any filing is scored or any IC is computed.** This is RESEARCH-LOG open item 12
(`docs/handoff_2026-09-01-grid-buildout-long-short.md` §8.3(b)): a text-mining construction distinct
from every backlog *size* measure already tried and FAILED in this project (backlog growth —
`docs/backlog-surprise-factor-results.md`; backlog coverage/RPO-to-revenue —
`docs/backlog-coverage-signal-results.md`). Mechanism here is **pricing-power language**, not
backlog magnitude: does a supplier's own MD&A talk like it has leverage over its customers (long
lead times, escalation clauses, pass-through surcharges, firm-pricing/book-to-bill commentary), and
does more of that talk predict better forward returns?

## 1. Universe

`electrification_strategy.universe.members("screen", pd.Timestamp("2026-08-31"), prices)` — the
shipped `screen +val+sleeve` construction's 27-name supplier screen (customer_institutional & ≥2 of
{profitable_2026, earnings_valued, low_policy_dependence}, REIT sub-industry excluded, 252-day
listing gate). Computed once, 2026-09-21, against the cached `grid_equipment_basket` price panel:

```
AEIS, AME, APG, ATKR, AYI, CARR, EME, EMR, ENS, ETN, FIX, GEV, HUBB, IESC, JCI, MOD,
MTZ, MYRG, NVT, POWL, PRIM, PWR, RRX, SPXC, TT, VRT, WCC
```

(27 names — matches the task's expected count.) All 27 have a CIK in SEC's `company_tickers.json`
and, per the filing-list check in step 1 of the build, all 27 file domestic 10-K/10-Q (none is a
20-F foreign private issuer, including ETN despite its Irish domicile — Eaton is SEC-registered as
a US domestic filer). No exclusions expected on filer-type grounds; if any name turns out to have
no usable 10-K/10-Q text in the window, it will be dropped and reported, not silently imputed.

## 2. Keyword / phrase list — LOCKED, not tunable after this point

Twelve phrases, three buckets, drawn directly from the handoff's own §8.3(b) language ("lead times
extended", "pricing actions", "backlog margin") and the task's own example list ("price escalation",
"surcharge", "pass-through", "backlog extends into", "book-to-bill", "firm pricing"). Matching is
case-insensitive substring count on the extracted MD&A text (so "surcharges", "escalation clauses"
etc. are caught by the singular root).

| Bucket | Phrase |
|---|---|
| Lead-time / scarcity commentary | `lead time` |
| | `extended lead` |
| | `backlog extends into` |
| Price-escalation / cost pass-through clause language | `price escalation` |
| | `escalation clause` |
| | `surcharge` |
| | `pass-through` |
| | `pricing action` |
| Margin / backlog-quality language | `book-to-bill` |
| | `backlog margin` |
| | `firm pricing` |
| | `price increase` |

**Score** (per filing) = total count of all 12 phrases in the extracted MD&A text.
**Primary metric = density** = count per 1,000 MD&A words (controls for MD&A length varying widely
across filers — some 10-Ks run 3,000-word MD&As, others 15,000). Raw count and a secondary
"distinct phrases present" (0–12) breadth measure are carried as diagnostics, not the primary test.

**Pre-registered mechanism caveat (stated now, not after seeing results):** substring counting is
direction-agnostic. "We shortened lead times" and "lead times remain extended" both increment the
`lead time` count once; "price increases were insufficient to offset cost inflation" and "we
implemented price increases" both increment `price increase`. This is a known limitation of
keyword-count scoring (the project's stated convention: "deliberately unelaborate," no NLP/ML
sentiment model) and is flagged here, before scoring, as a reason a null or weak result would not be
surprising — not invoked after the fact to explain one away.

## 3. MD&A extraction

10-K → text between the `Item 7` and `Item 7A`/`Item 8` section headers. 10-Q → text between `Item 2`
and `Item 3`/`Item 4`. Regex is case-insensitive and tolerant of common HTML-to-text artifacts
(extra whitespace, `&nbsp;`, period/dash after the number). If no header match is found (non-standard
filer formatting), the filing falls back to whole-document scoring and is flagged
`mdna_extracted=False` in the output — reported as a coverage stat, not dropped silently.

## 4. Point-in-time discipline

Each filing's score becomes "known" on its actual SEC `filingDate` (from the EDGAR submissions API
— an exact public timestamp, not an estimated reporting lag like the ICR retest's `_ICR_LAG_DAYS`
placeholder in `grid_resilience/factor/resilience_score.py`, which exists because that pipeline
lacks true filed dates). A 1-calendar-day buffer is added past `filingDate` before a score is usable
in the monthly panel, to avoid same-day look-ahead. The monthly panel takes, for each ticker at each
month-end, the most recently *known* (filed + 1 day ≤ month-end) filing's density score, forward-
filled until superseded by the next filing.

## 5. Data source — reuse, not rebuild

SEC EDGAR full-text/document retrieval already has three precedents in `research/` (per this
project's convention of reusing rather than rebuilding EDGAR access): `research/
electricity_earnings_exposure/edgar.py` (XBRL companyconcept — numeric, not text), `research/
merchant_fundamental_beta/{fetch_filing_list.py,fetch_10k.py}` (submissions-API filing list +
primary-document HTML→text, used for the ICR EDGAR retest), and `research/narrative_mispricing/
{extend_2026_edgar.py,spike_a_refined_edgar.py}` (EDGAR full-text-search quarterly keyword-hit
counts, the Dispatch thread's disclosure-count feature). This build's `research/
backlog_pricing_power/edgar_filings.py` ports the submissions-API + document-fetch pattern from the
`merchant_fundamental_beta` pair (generalized from 4 hardcoded tickers to all 27, with disk caching)
rather than EDGAR full-text search, because full-text search returns filing-level hit
presence/absence only (no in-document count, no section restriction) and this design needs a true
MD&A-scoped count. **Not rebuilt from scratch** — same submissions endpoint, same header
convention, same rate-limit handling (429 backoff, `User-Agent` string) as the existing scripts.

## 6. Test — pre-registered, exact convention used throughout this log

Monthly cross-sectional rank-IC (Spearman), signal known as of month `t` vs. forward return over the
next `h` months, computed exactly as `research/narrative_mispricing/spike_a_ic_test.py`'s pattern:
per month, Spearman correlation across all tickers with both a score and a forward return that month
(minimum cross-section n ≥ 8, dropped otherwise); the monthly IC series' mean and
`t = mean / (std / sqrt(n_months))` is the headline stat.

- **Horizons:** forward 1-month and forward 3-month simple return.
- **Windows:** pooled (full sample) + two non-overlapping sub-windows on this log's now-standard
  split — **2019-01→2022-12** ("pre-AI regime") and **2023-01→2026-06** ("AI regime") — matching
  `docs/category-demand-rotation-results.md` and the STEO spark-spread work's split, per this log's
  repeated finding that a single-window result usually doesn't survive a regime split.
- **Price data:** `grid_equipment_basket.data.prices.fetch_prices` (cached adjusted close), monthly
  resample.

## 7. Gate (this project's default)

**PASS** requires: pooled `|t| ≥ 2.0` on at least one horizon (1m or 3m), **and** the same sign with
`|t| ≥ 1.5` in **both** non-overlapping sub-windows (not reversed, not one-window-only — the
category-demand and STEO precedents in this log both "won" pooled or in one window and died in the
regime split). Anything short of that is FAIL / NOT ADOPTED, reported with the actual numbers, no
post-hoc reframing of the bar.

**Explicitly excluded from this test, even as a control**, per the task instructions: any backlog
*size* measure (raw backlog growth, YoY surprise, RPO/coverage ratio). Those are the already-failed
mechanism this thread exists to be distinct from; if they were to sneak back in as a "control"
column they will be flagged, not silently included.

## 8. What's built, throwaway

`research/backlog_pricing_power/` — `cik_map.py` (27-ticker CIK lookup), `edgar_filings.py`
(submissions-API filing list + document fetch/cache), `keywords.py` (the locked list above,
importable so the score code can't silently drift from what's written here), `score.py` (MD&A
extraction + keyword density scoring), `build_panel.py` (point-in-time monthly panel), `ic_test.py`
(the rank-IC test above), `run_all.py` (orchestrator). Not production code; caches under
`research/backlog_pricing_power/_cache/`.

Results → `docs/backlog-pricing-power-results.md` after this pipeline runs, verdict reported plainly
regardless of outcome.
