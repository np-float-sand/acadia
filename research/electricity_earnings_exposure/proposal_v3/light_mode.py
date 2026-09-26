"""
light_mode.py — turn the theme-aware Proposal v3 fragment into a standalone,
light-locked HTML document that prints cleanly to PDF.

The artifact HTML (docs/electrification-strategy-proposal-v3.html) is a *fragment*
(no <html>/<head>/<body>) whose CSS resolves light or dark from the viewer's theme.
`to_light_print()` wraps it in a full document, pins it to light with
<html data-theme="light"> (which the fragment's own CSS already honours as the
explicit-choice branch that beats both the OS media query and the dark toggle), and
appends a print stylesheet: white ground, no shadows, A4 margins, and break-inside
rules so figure cards, tables and pro/con blocks don't split across pages.
"""
from __future__ import annotations

_PRINT_CSS = """
/* ---- injected by light_mode.to_light_print() ---- */
html[data-theme="light"]{color-scheme:light}
body{background:var(--ground)}
@media print{
  :root{--shadow:none !important}
  html,body{background:#ffffff !important}
  .page{max-width:none; padding:0 4mm}
  .bleed{max-width:100% !important}             /* 62rem bleed containers overrun A4 → clip; pin to page */
  .fig-card{box-shadow:none; border-color:#d9d9d2}
  .cand,.tbl-scroll{box-shadow:none}
  .tbl-scroll{overflow:visible}                 /* no scrollbars on paper */
  a{color:inherit; text-decoration:none}
  h1,h2,h3,h4{break-after:avoid}
  figure,.fig-card,.cand,.pc,.pc>div,.note,.stripe{break-inside:avoid}
  .tbl-scroll{break-inside:avoid}
  li{break-inside:avoid}
  thead{display:table-header-group}
  .masthead{padding-top:0}
  /* the 8-column two-year table: rotate its page to landscape so nothing clips */
  .tbl-scroll:has(table.two-year){break-inside:auto; page:wide}
  table.two-year{font-size:8.5pt}
  table.two-year th,table.two-year td{padding:.38rem .5rem}
  table.two-year td:has(ul.u){min-width:16rem; max-width:24rem}
  table.two-year tr{break-inside:avoid}
}
@page{size:A4; margin:13mm 11mm}
@page wide{size:A4 landscape; margin:12mm}
"""


def to_light_print(fragment_html: str, *, title: str = "Screened Electrification Equipment") -> str:
    """fragment_html: the contents of docs/electrification-strategy-proposal-v3.html
    (already has its <title>/<link>/<style> at the top — browsers hoist those from
    <body> to <head>, so we just drop the fragment into <body> as-is)."""
    return (
        "<!doctype html>\n"
        '<html lang="en" data-theme="light">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f"<style>{_PRINT_CSS}</style>\n"
        "</head>\n<body>\n"
        f"{fragment_html}\n"
        "</body>\n</html>\n"
    )


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "../../../docs/electrification-strategy-proposal-v3.html"
    with open(src) as f:
        print(to_light_print(f.read()))
