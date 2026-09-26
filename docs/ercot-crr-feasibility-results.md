# ERCOT CRR clearing prices — feasibility check results (2026-09-08)

**Status: FAILED (feasibility).** Untried-tasks-log item 5.

---

## TL;DR

**No free, self-serve way to get more than ~12 months of ERCOT CRR (Congestion Revenue
Rights) auction clearing prices.** The live document API for both CRR report products
(Monthly Auction Results, EMIL `NP7-803-M`, report type 11201; Long-Term Auction Results,
`NP7-802-M`, type 11203) only lists a rolling window of recent files — checked live,
exactly 12 documents, 2025-09 through 2026-08, regardless of date-range, paging, or
result-size parameters passed. ERCOT's public CRR page (ercot.com/mktinfo/crr) has no
historical-archive or bulk-download section. Getting full 2018-2026 history would require
a special data request through ERCOT's client-services "Information Request" channel —
unknown cost, unknown turnaround, not a research-spike-scale ask.

**Verdict: do not pursue further without a funded/approved ERCOT data request.** This
closes the open question from `docs/ftr-bid-signal-results.md` ("full 2018-2026 history
would need ERCOT's separate archival portal or its own scrape, untested") with a real
negative rather than leaving it untested.

## What was checked

1. Found ERCOT's real CRR report IDs (not in `gridstatus`, which has no CRR wrapper at
   all): Monthly Auction Results = report type 11201, Long-Term Auction Results = 11203
   (via web search — https://www.ercot.com/mp/data-products/data-product-details?id=NP7-803-M,
   https://www.ercot.com/mp/data-products/data-product-details?id=NP7-802-M).
2. Queried ERCOT's live document-list API (`IceDocListJsonWS`) for report type 11201:
   returns exactly 12 documents, one per month, 2025-09-18 through 2026-08-20 — a rolling
   ~12-month window, confirmed by the same pattern already seen on the FTR-bid feed.
3. Tried `startDate`/`endDate`, `mimicKey`/`showHTMLView`, and `size` query parameters to
   unlock older documents — all three returned the identical 12-document list, no effect.
4. Checked ERCOT's public CRR landing page (ercot.com/mktinfo/crr) directly for a
   historical-archive or bulk-download link — none found; the page only surfaces the same
   two current reporting products.

## Disposition

Logged as a data-access dead end at the free/public tier, same shape as CME/ICE power
forwards, NERC GADS, and PJM's FTR clearing-price SSO wall — the genuinely
forward-looking/price-discovery instruments in this space keep landing behind a paywall or
a manual request process, not a quick API call. Not pursued further this session. If ERCOT
data access is ever funded for another reason (e.g. the RT/DA spread work in item 4, which
uses ERCOT's already-free `get_lmp`/`get_dam_spp`), ask about CRR historical bulk access at
the same time rather than as a separate request.
