# STEO point-in-time archive — feasibility check results (2026-09-08)

**Status: SCOPED, NOT RUN — same estimate as before (~half a day of PDF-table
extraction), a PUDL-hosted mirror does not shortcut it.** Untried-tasks-log item 6 (fix #3
from item 3 folded in here).

---

## TL;DR

- The original scoping (`docs/hhub-forward-gas-signal-results.md` §Round 3) assumed
  building the point-in-time `ELWHU_*` (EIA-modeled forward wholesale power price by hub)
  panel would mean scraping ~90 monthly STEO PDF releases directly, at "~half a day."
- **Checked the direct path first: it's harder than that estimate.** `eia.gov/outlooks/steo/archives/`
  blocks directory listing (403, both a default and a browser-style User-Agent), but
  individual files ARE fetchable once you know the naming pattern
  (`archives/{mon}{yy}.pdf`, e.g. `oct18.pdf` — confirmed live, 200, 1.1MB). No Excel/CSV
  companion exists at the same path (404) — every vintage is **PDF-only**, so extracting a
  specific table row (`ELWHU_PJ`/`ELWHU_TX`) from each of ~90 files means real PDF
  table-parsing, not a clean tabular pull. That's a bigger lift than "half a day."
- **Checked the PUDL angle further this session — it does NOT shortcut the parsing
  problem.** `transmission_rate_base/` does use PUDL for FERC Form 1, via PUDL's actual
  processed nightly database (`s3://pudl.catalyst.coop/nightly/`, a public S3 mirror,
  directly `pd.read_parquet()`-able). Live-checked that same bucket for STEO: **zero**
  STEO/ELWHU/wholesale/outlook tables among all 421 processed tables — PUDL has never
  finished turning STEO into a queryable table. The Zenodo record found earlier
  (zenodo.org/records/18115100) is a **different thing**: PUDL's raw-input archive — a
  preservation mirror of EIA's *original* files, confirmed (via direct inspection) to be
  year-zipped, not a cleaned dataset ("raw input data for PUDL... not a
  cleaned/processed dataset" per the archive's own description). It's a more stable,
  single-host place to *get* the same raw files EIA publishes — it does not avoid the
  table-extraction work.

## Corrected disposition

**Back to the original estimate, not better than it.** There is no shortcut past parsing
each vintage's `ELWHU_PJ`/`ELWHU_TX` table by hand or with a PDF-table extractor — whether
the source is `eia.gov/outlooks/steo/archives/{mon}{yy}.pdf` directly or the Zenodo mirror
of the same files. The Zenodo archive is worth using only for **access stability**
(EIA's own archive blocks directory listing; Zenodo doesn't), not for saving parsing
effort. Revert to the original "~half a day" estimate from
`docs/hhub-forward-gas-signal-results.md` §Round 3 for the actual point-in-time panel
build — this session upgraded *where to get the files*, not *how much work extracting them
takes*.

## Disposition

**Remains SCOPED, NOT RUN.** If picked up: extract `ELWHU_PJ`/`ELWHU_TX` from each
monthly STEO release table (PDF text/table extraction, ~90 files, ~half a day per the
original estimate — Zenodo is a fine source to pull from instead of EIA's directory-listing-
blocked archive, but doesn't reduce the extraction work) into a point-in-time panel, then
test per item 3's redirect: this is the forward-looking regressor that fix #2's *trailing*-
price version should have been, aimed at stabilizing the VST/NRG stress beta — not a
basket-timing signal (the Henry Hub *trailing*-gas-proxy version of "does forward power
predict returns" already failed at that framing, per
`docs/hhub-forward-gas-signal-results.md` — don't repeat it here).
