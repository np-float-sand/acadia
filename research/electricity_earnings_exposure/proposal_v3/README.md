# proposal_v3/ — build kit for the "Screened Electrification Equipment" write-up

Everything needed to regenerate `docs/electrification-strategy-proposal-v3.html`
(the Claude artifact), a light-locked print HTML, and a PDF.

Published artifact: https://claude.ai/code/artifact/829dc90b-02a4-4caa-8abb-9eb8eed3c198

## Files

| file | role |
|---|---|
| `template.html` | **the editable source.** All prose, tables and CSS. Figures are `__IMG_*__` placeholders. Edit this to change the write-up. |
| `build_figures.py` | recomputes every number + the 5 PNG figures from `output_electrification/*.csv` + yfinance. Writes `out/proposal_v3.json` and `out/*.png` / `out/*.b64`. |
| `build_derisk_check.py` | the "SPY + 15% sleeve vs the book" and cash-lever comparison (the numbers in §6's verdict table). Prints a table; not wired into assemble. |
| `light_mode.py` | `to_light_print(fragment_html)` — wraps the theme-aware fragment in a full `<html data-theme="light">` document + an `@media print` stylesheet (white ground, no shadows, A4, `break-inside` rules, the wide 2-year table on a landscape page). |
| `assemble.py` | inlines the figures into `template.html`, then writes the deploy HTML / light HTML / PDF. |
| `out/` | generated: `proposal_v3.json`, figures, `proposal_v3_print.html`, `proposal_v3.pdf`, `pv3_px.parquet` (price cache — delete to refetch). |

## Re-run

```sh
# from the repo root, with .env sourced for any API keys (none strictly required)
poetry run python research/electricity_earnings_exposure/proposal_v3/build_figures.py
poetry run python research/electricity_earnings_exposure/proposal_v3/assemble.py            # = --deploy --light --pdf
```

`assemble.py` flags (default is all three):
- `--deploy` → `docs/electrification-strategy-proposal-v3.html` (re-publish this path as the artifact to keep the same URL)
- `--light`  → `out/proposal_v3_print.html` (light-locked, standalone)
- `--pdf`    → `out/proposal_v3.pdf` (needs Chrome/Chromium; falls back to "open the print HTML and Save as PDF")

Editing prose only? Skip `build_figures.py` and just re-run `assemble.py`.
Numbers changed? Delete `out/pv3_px.parquet` first to refetch prices.

## Notes

- Backtest window for all headline stats/figures is **Jan 2019 → Aug 2026** (`START` in
  `build_figures.py`); before 2019 the `screen` universe has < 8 listed names.
- Sharpe convention is `(CAGR − 3.76%) / annualised vol` (`RF` in `build_figures.py`),
  matching `metrics.csv` / the research log.
- The light-mode function keys off `<html data-theme="light">` — the fragment's own CSS
  already treats an explicit `data-theme` as the branch that beats both the OS media
  query and the dark toggle, so no dark rules need stripping.
