"""
assemble.py — build the Proposal v3 deliverables from template.html + out/*.b64.

  poetry run python research/electricity_earnings_exposure/proposal_v3/assemble.py [flags]

flags (default = all three):
  --deploy   write docs/electrification-strategy-proposal-v3.html   (theme-aware fragment,
             the file published as the Claude artifact — keeps the same URL on republish)
  --light    write out/proposal_v3_print.html                       (light-locked full doc)
  --pdf      write out/proposal_v3.pdf                               (implies --light; needs Chrome)

Edit the prose in template.html, re-run build_figures.py if the numbers changed,
then re-run this.
"""
from __future__ import annotations
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "out")
TEMPLATE = os.path.join(HERE, "template.html")
DEPLOY = os.path.join(ROOT, "docs", "electrification-strategy-proposal-v3.html")
LIGHT = os.path.join(OUT, "proposal_v3_print.html")
PDF = os.path.join(OUT, "proposal_v3.pdf")
FIGS = ["growth_fair", "growth_full", "drawdown", "rollsharpe", "shorts_bars"]

_CHROMES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
]


def _find_chrome() -> str | None:
    from shutil import which
    for c in _CHROMES:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
        w = which(c)
        if w:
            return w
    return None


def inline() -> str:
    with open(TEMPLATE) as f:
        html = f.read()
    for name in FIGS:
        p = os.path.join(OUT, f"{name}.b64")
        if not os.path.exists(p):
            sys.exit(f"missing {p} — run build_figures.py first")
        with open(p) as f:
            html = html.replace(f"__IMG_{name}__", f"data:image/png;base64,{f.read().strip()}")
    if "__IMG_" in html:
        sys.exit("unsubstituted __IMG_ placeholder remains in template.html")
    return html


def main():
    args = set(sys.argv[1:]) or {"--deploy", "--light", "--pdf"}
    frag = inline()

    if "--deploy" in args:
        with open(DEPLOY, "w") as f:
            f.write(frag)
        print(f"wrote {os.path.relpath(DEPLOY, ROOT)}  ({len(frag)//1024} KB)")

    if "--light" in args or "--pdf" in args:
        sys.path.insert(0, HERE)
        from light_mode import to_light_print
        doc = to_light_print(frag)
        with open(LIGHT, "w") as f:
            f.write(doc)
        print(f"wrote {os.path.relpath(LIGHT, ROOT)}  ({len(doc)//1024} KB)")

    if "--pdf" in args:
        chrome = _find_chrome()
        if not chrome:
            print("!! no Chrome/Chromium found — open out/proposal_v3_print.html and File > Print > Save as PDF")
            return
        cmd = [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
               f"--print-to-pdf={PDF}", "--virtual-time-budget=10000",
               f"file://{LIGHT}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(PDF) and os.path.getsize(PDF) > 1000:
            print(f"wrote {os.path.relpath(PDF, ROOT)}  ({os.path.getsize(PDF)//1024} KB)")
        else:
            print("!! Chrome PDF failed:\n", r.stderr[-800:] or r.stdout[-800:])


if __name__ == "__main__":
    main()
