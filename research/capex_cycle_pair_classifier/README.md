# research/capex_cycle_pair_classifier/

Throwaway probe scripts from the 2026-09-02 -> 09-06 conversation (the one that
produced the `capex-cycle-pair-classifier` memory + `docs/handoff_2026-09-03-
transmission-project-filings.md`). **Not production code.** Verdicts:
`docs/RESEARCH-LOG.md` section 6 block "2026-09-02 -> 09-06".

## Run

    python research/capex_cycle_pair_classifier/fetch.py     # once, builds _cache/*.parquet
    python research/capex_cycle_pair_classifier/<script>.py   # from repo ROOT

All scripts read the grid-equipment price panel from
`grid_equipment_basket/data/cache/prices_*.parquet` (that package's cache) plus the
`_cache/*.parquet` that `fetch.py` builds. `_cache/` is gitignored.

`parse_b9.py` additionally needs the PJM Load Forecast Report PDFs (2021-2024) in
`_cache/pjm_lf/` and the 2025/2026 table xlsx in `_cache/pjm_xls/` -- hand-download,
URLs in `docs/pjm-large-load-vintages-2026-09-03.md`. Its output CSV is already
committed at `grid_resilience/data/seed/pjm_large_load_b9_vintages.csv`.

## Scripts

| script | what it checked | verdict |
|---|---|---|
| `repro.py` | thematic-catalyst: buildout basket vs SPY divergence, timing vs the Nov-2022 PJM filing | basket outperformed but earlier/fuzzier than the "clean 7-mo lead" story |
| `parse_b9.py` | parse 6 PJM Table B-9 vintages (PDF + xlsx) -> tidy CSV | data build (kept as seed CSV) |
| `revision_vs_basket.py` | PJM data-center MW-forecast *revisions* vs basket fwd returns | FAIL -- de-risked into a rally; quarterly corr ~0 |
| `signal_basket.py` | basket gated by the MW-revision "stall" signal | FAIL -- costs return, MaxDD unchanged |
| `bra_test.py` | PJM capacity-auction (BRA) clearing prices vs basket fwd returns | FAIL -- lags equity 18-24mo, inverted, no exit signal |
| `classify.py` | the 4-criterion "customer + 2 of 3" capex-cycle-pair classifier | KEEPER -- reproduces the hand-picked book |
| `bt_pairs.py` / `bt_pairs2.py` | hand-picked vs rules-selected long/short (bt_pairs2 adds the +/-35% return cap for reverse-split artifacts) | rules ~= hand-picked (~1.95 hedged, 2023-26) |
| `bt_strict.py` | loose (customer + 2/3) vs strict (all 4) classifier | strict ~= loose; keep loose |
| `bt_generalize.py` | classifier on EV / nuclear-SMR / space + break-gate | EV only; nuclear inverts; space self-rejects |
| `bt_all.py` | cross-instance mania/post/break-gated (hydrogen 2000 & 2020, genomics, cannabis, ...) | post-mania alpha carried by grid/DER alone; break-gate makes it worse |
| `bt_book.py` | grid/DER + EV + hydrogen + eVTOL + battery-tech concurrent book | 2 live pairs (grid/DER, EV); the rest dilute |
| `attrib.py` | regress the grid/DER dollar-neutral spread on market + quality + duration + PAVE/TAN | not a factor; ~40% PAVE/TAN; residual is one-regime concentration |
| `pitch_test.py` | does "concentrated PAVE / TAN" reproduce the book? | direction yes (~0.65 corr), the book no; PAVE/XLI itself is Sharpe -0.51 |
| `long_screen.py` | cross-industry screen for names matching the 4-criterion long profile | ~40 passers across 7 buckets (AI/DC, semicap, aero, reshoring E&C, water, grid, thermal) |
| `policy_bt.py` | US-industrial-policy long / non-policy-industrials short | Sharpe 0.81 (2023-26) / 0.18 (2021-22); one macro bet, not shipped |
| `backward.py` | reverse-engineer: what did each quarter's winners have in common ex ante? | DC-power sub-sector (rank-IC +0.27); tilt buys +0.09 Sharpe in-regime, -0.12 before |
| `longbook_dispersion.py` | per-name returns by year + within-17-name cross-sectional momentum | momentum tilt loses to equal-weight |
