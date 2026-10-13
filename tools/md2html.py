"""
md2html — render a Markdown file to a self-contained, styled HTML document.

    from tools.md2html import md_to_html
    html = md_to_html(open("docs/RESEARCH-LOG.md").read(), theme="auto")

    # or on the command line
    poetry run python tools/md2html.py docs/RESEARCH-LOG.md -o out.html --theme auto
    poetry run python tools/md2html.py notes.md --pdf            # also writes notes.pdf

theme:
    "auto"   (default) — follows the reader's OS light/dark setting; a corner toggle
                          lets them override, remembered in localStorage.
    "light" / "dark"   — locked to that theme, no toggle.

Design: one column (~46rem measure), Newsreader headings / IBM Plex Sans body /
IBM Plex Mono code, a cobalt accent, cool-neutral paper. GFM tables, fenced code,
footnotes and definition lists supported. Google Fonts is the only external request;
everything else is inline, so the file works offline and prints cleanly to PDF.

Requires `markdown` (poetry add markdown).
"""
from __future__ import annotations

import html as _html
import os
import re
import subprocess
import sys

try:
    import markdown as _md
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("md2html needs the `markdown` package — run:  poetry add markdown")

_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "attr_list",
               "toc", "footnotes", "abbr", "def_list", "md_in_html"]

# ---------------------------------------------------------------- stylesheet ----
_CSS = r"""
:root{
  --ground:#f6f6f3; --surface:#ffffff; --surface-2:#efefe9;
  --ink:#1c2026; --ink-soft:#5a626d; --ink-faint:#8b929c;
  --line:#e3e3dc; --line-soft:#ecebe4;
  --accent:#1f5fa8; --accent-ink:#17497f; --accent-soft:#e7eef7;
  --code-bg:#f2f1ec; --mark:#fbe7b0;
  --sel:#cfe0f2;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#14171b; --surface:#1b1e23; --surface-2:#22262d;
    --ink:#e6e9ec; --ink-soft:#9aa3ad; --ink-faint:#727a84;
    --line:#2c313a; --line-soft:#242a32;
    --accent:#5c9bd6; --accent-ink:#9cc4e8; --accent-soft:#1e2b3a;
    --code-bg:#1f242b; --mark:#5a4a12; --sel:#26374a;
  }
}
:root[data-theme="dark"]{
  --ground:#14171b; --surface:#1b1e23; --surface-2:#22262d;
  --ink:#e6e9ec; --ink-soft:#9aa3ad; --ink-faint:#727a84;
  --line:#2c313a; --line-soft:#242a32;
  --accent:#5c9bd6; --accent-ink:#9cc4e8; --accent-soft:#1e2b3a;
  --code-bg:#1f242b; --mark:#5a4a12; --sel:#26374a;
}

*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  font-size:17px; line-height:1.66; letter-spacing:-.003em;
  -webkit-font-smoothing:antialiased;
}
::selection{background:var(--sel)}
.doc{max-width:46rem; margin:0 auto; padding:clamp(2rem,5vw,4rem) clamp(1.1rem,4vw,2rem) 6rem}

h1,h2,h3,h4,h5,h6{font-family:"Newsreader",Georgia,serif; font-weight:500;
  line-height:1.2; text-wrap:balance; letter-spacing:-.012em; margin:2.4em 0 .5em}
h1{font-size:clamp(2rem,5vw,2.7rem); margin:.1em 0 .5em}
h2{font-size:clamp(1.45rem,3.4vw,1.85rem); padding-bottom:.24em; border-bottom:1px solid var(--line)}
h3{font-size:1.28rem; font-weight:600}
h4{font-size:1.08rem; font-weight:600; color:var(--ink-soft)}
h5,h6{font-size:.95rem; font-weight:600; color:var(--ink-soft);
  text-transform:uppercase; letter-spacing:.04em; font-family:"IBM Plex Mono",monospace}
h1+p,h2+p,h3+p{margin-top:.2em}

p{margin:1em 0}
a{color:var(--accent); text-underline-offset:2px; text-decoration-color:color-mix(in srgb,var(--accent) 40%,transparent)}
a:hover{text-decoration-color:currentColor}
strong{font-weight:600}
small{color:var(--ink-soft)}
hr{border:none; border-top:1px solid var(--line); margin:2.8em 0}
mark{background:var(--mark); color:inherit; padding:.05em .2em; border-radius:3px}

ul,ol{margin:1em 0; padding-left:1.5em}
li{margin:.35em 0}
li>ul,li>ol{margin:.3em 0}
dl{margin:1em 0}
dt{font-weight:600; margin-top:.8em}
dd{margin:.2em 0 .2em 1.4em; color:var(--ink-soft)}

blockquote{margin:1.4em 0; padding:.4em 0 .4em 1.15em; border-left:3px solid var(--accent);
  color:var(--ink-soft)}
blockquote p{margin:.5em 0}

code,kbd,samp{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.88em}
:not(pre)>code{background:var(--code-bg); padding:.12em .38em; border-radius:4px;
  border:1px solid var(--line-soft)}
pre{background:var(--code-bg); border:1px solid var(--line); border-radius:9px;
  padding:1em 1.15em; overflow-x:auto; margin:1.3em 0; line-height:1.55}
pre code{font-size:.85em; background:none; border:none; padding:0}

.table-wrap{overflow-x:auto; margin:1.5em 0; border:1px solid var(--line);
  border-radius:9px; background:var(--surface)}
table{border-collapse:collapse; width:100%; font-size:.9rem}
th,td{text-align:left; padding:.55rem .8rem; border-bottom:1px solid var(--line-soft);
  vertical-align:top}
thead th{font-family:"IBM Plex Mono",monospace; font-weight:500; font-size:.75rem;
  text-transform:uppercase; letter-spacing:.03em; color:var(--ink-soft);
  background:var(--surface-2); border-bottom:1px solid var(--line)}
tbody tr:last-child td{border-bottom:none}
td:has(+td),th:has(+th){white-space:normal}
table td[align="right"],table th[align="right"]{text-align:right; font-variant-numeric:tabular-nums}
table td[align="center"],table th[align="center"]{text-align:center}

img{max-width:100%; height:auto; border-radius:6px}
.footnote{font-size:.9rem; color:var(--ink-soft)}
.footnote ol{padding-left:1.2em}
.headerlink{margin-left:.35em; color:var(--ink-faint); text-decoration:none; opacity:0;
  transition:opacity .12s}
h1:hover .headerlink,h2:hover .headerlink,h3:hover .headerlink,
h4:hover .headerlink{opacity:1}
.toc{background:var(--surface-2); border:1px solid var(--line); border-radius:9px;
  padding:.8em 1.1em; margin:1.6em 0; font-size:.92rem}
.toc>ul{margin:.2em 0}
.toc ul{list-style:none; padding-left:1em}
.toc>ul{padding-left:0}
.toc a{color:var(--ink-soft); text-decoration:none}
.toc a:hover{color:var(--accent)}

.theme-toggle{position:fixed; top:.9rem; right:.9rem; z-index:9;
  font:500 .72rem/1 "IBM Plex Mono",monospace; letter-spacing:.04em; text-transform:uppercase;
  color:var(--ink-soft); background:var(--surface); border:1px solid var(--line);
  border-radius:999px; padding:.5em .9em; cursor:pointer;
  box-shadow:0 1px 2px rgba(0,0,0,.06)}
.theme-toggle:hover{color:var(--ink); border-color:var(--ink-faint)}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px; border-radius:3px}

@media print{
  :root,:root[data-theme="dark"]{
    --ground:#fff; --surface:#fff; --surface-2:#f3f2ee;
    --ink:#1c2026; --ink-soft:#4a525c; --ink-faint:#8b929c;
    --line:#dcdcd4; --line-soft:#e8e7e0; --code-bg:#f4f3ee;
  }
  body{font-size:10.5pt}
  .doc{max-width:none; padding:0}
  .theme-toggle{display:none}
  a{color:inherit; text-decoration:none}
  h1,h2,h3,h4{break-after:avoid}
  pre,blockquote,img,figure{break-inside:avoid}
  .table-wrap{break-inside:auto}          /* long tables may span pages */
  tr,li{break-inside:avoid}
  thead{display:table-header-group}
  h2{border-bottom-color:#d8d8d0}
  table{font-size:.82rem}                 /* claw back width for many-column tables */
  th,td{padding:.42rem .5rem}
}
@page{size:A4; margin:14mm 11mm;
  @bottom-center{content:counter(page) " / " counter(pages); font:9pt Georgia,serif; color:#777}}
"""

# --------------------------------------------------------------- toggle JS ----
_NOFLASH_JS = """
try{var t=localStorage.getItem('md2html-theme');
if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t);}catch(e){}
"""

_TOGGLE_JS = """
(function(){
  var root=document.documentElement, btn=document.getElementById('themeToggle');
  if(!btn)return;
  function sys(){return matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';}
  function cur(){return root.getAttribute('data-theme')||sys();}
  function paint(){btn.textContent=(cur()==='dark'?'\\u2600 Light':'\\u263D Dark');}
  paint();
  btn.addEventListener('click',function(){
    var next=cur()==='dark'?'light':'dark';
    root.setAttribute('data-theme',next);
    try{localStorage.setItem('md2html-theme',next);}catch(e){}
    paint();
  });
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change',paint);
})();
"""


def _first_h1(md_text: str) -> str | None:
    m = re.search(r"^\#\s+(.+?)\s*$", md_text, flags=re.M)
    return m.group(1).strip() if m else None


def _wrap_tables(body_html: str) -> str:
    return re.sub(r"(<table\b)", r'<div class="table-wrap">\1', body_html) \
             .replace("</table>", "</table></div>")


_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp"}


def _inline_images(body_html: str, base_dir: str) -> str:
    """Replace <img src="local/path"> with a data: URI so the file is self-contained."""
    import base64

    def repl(m: re.Match) -> str:
        src = m.group("src")
        if src.startswith(("http://", "https://", "data:", "//")):
            return m.group(0)
        path = src if os.path.isabs(src) else os.path.join(base_dir, src)
        ext = os.path.splitext(path)[1].lower()
        if not os.path.isfile(path) or ext not in _MIME:
            return m.group(0)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f'{m.group("pre")}src="data:{_MIME[ext]};base64,{b64}"'

    return re.sub(r'(?P<pre><img\b[^>]*?\s)src="(?P<src>[^"]*)"', repl, body_html)


def md_to_html(md_text: str, *, title: str | None = None, theme: str = "auto",
               toggle: bool = True, toc: bool = False, base_dir: str | None = None) -> str:
    """Render Markdown text to a full standalone HTML document string.

    base_dir: directory that relative image paths in the Markdown resolve against
    (default: the current working directory). Local images are inlined as data URIs.
    """
    if theme not in ("auto", "light", "dark"):
        raise ValueError("theme must be 'auto', 'light' or 'dark'")

    src = md_text
    if toc and "[TOC]" not in src:
        src = "[TOC]\n\n" + src

    conv = _md.Markdown(extensions=_EXTENSIONS,
                        extension_configs={"toc": {"permalink": "#"}})
    body = _inline_images(_wrap_tables(conv.convert(src)), base_dir or os.getcwd())

    doc_title = title or _first_h1(md_text) or "Document"
    root_attr = "" if theme == "auto" else f' data-theme="{theme}"'
    show_toggle = toggle and theme == "auto"
    toggle_btn = ('<button id="themeToggle" class="theme-toggle" type="button" '
                  'aria-label="Toggle light or dark theme"></button>') if show_toggle else ""
    toggle_js = f"<script>{_TOGGLE_JS}</script>" if show_toggle else ""

    return f"""<!doctype html>
<html lang="en"{root_attr}>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(doc_title)}</title>
<script>{_NOFLASH_JS}</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{_CSS}</style>
</head>
<body>
{toggle_btn}
<main class="doc">
{body}
</main>
{toggle_js}
</body>
</html>
"""


# ---------------------------------------------------------------------- PDF ----
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
        if which(c):
            return which(c)
    return None


def render_pdf(md_text: str, pdf_path: str, *, title: str | None = None,
               toc: bool = False, base_dir: str | None = None) -> bool:
    """Write a PDF next to `pdf_path`. Uses a light-locked render. Returns success."""
    chrome = _find_chrome()
    tmp_html = pdf_path + ".src.html"
    with open(tmp_html, "w") as f:
        f.write(md_to_html(md_text, title=title, theme="light", toggle=False,
                           toc=toc, base_dir=base_dir))
    if not chrome:
        print(f"  no Chrome found — open {tmp_html} and Print > Save as PDF")
        return False
    r = subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", "--virtual-time-budget=10000", f"file://{os.path.abspath(tmp_html)}"],
        capture_output=True, text=True)
    ok = os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000
    if ok:
        os.remove(tmp_html)
    else:
        print("  Chrome PDF failed:\n", (r.stderr or r.stdout)[-600:])
    return ok


# ---------------------------------------------------------------------- CLI ----
def main(argv: list[str] | None = None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="Render a Markdown file to styled HTML.")
    p.add_argument("input", help="path to the .md file")
    p.add_argument("-o", "--output", help="output .html path (default: alongside input)")
    p.add_argument("--theme", choices=["auto", "light", "dark"], default="auto")
    p.add_argument("--no-toggle", action="store_true", help="omit the light/dark toggle button")
    p.add_argument("--toc", action="store_true", help="insert a table of contents")
    p.add_argument("--title", help="override the document title")
    p.add_argument("--pdf", action="store_true", help="also write a PDF (light-locked, needs Chrome)")
    a = p.parse_args(argv)

    with open(a.input) as f:
        md_text = f.read()
    base_dir = os.path.dirname(os.path.abspath(a.input))
    out = a.output or os.path.splitext(a.input)[0] + ".html"
    html = md_to_html(md_text, title=a.title, theme=a.theme,
                      toggle=not a.no_toggle, toc=a.toc, base_dir=base_dir)
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}  ({len(html) // 1024} KB, theme={a.theme})")

    if a.pdf:
        pdf = os.path.splitext(out)[0] + ".pdf"
        if render_pdf(md_text, pdf, title=a.title, toc=a.toc, base_dir=base_dir):
            print(f"wrote {pdf}  ({os.path.getsize(pdf) // 1024} KB)")


if __name__ == "__main__":
    main()
