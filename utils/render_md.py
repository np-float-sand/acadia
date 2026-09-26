#!/usr/bin/env python
"""Render the repo's markdown docs to self-contained HTML.

    python utils/render_md.py docs/capex-intensity-proposal.md
    python utils/render_md.py docs/ --out output_html
    python utils/render_md.py docs/*.md --out site --index

Output is a single HTML file per input with all CSS inlined -- no network, no
build step, opens straight from disk. Light and dark both handled via
`prefers-color-scheme`.

Built for what these docs actually contain: wide GFM tables (the research log
has cells holding whole paragraphs), fenced code, blockquotes, strikethrough,
and `\\|` escaped pipes inside table cells.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from xml.etree import ElementTree

import markdown
from markdown.extensions import Extension
from markdown.inlinepatterns import InlineProcessor

# --------------------------------------------------------------------------
# Strikethrough: python-markdown has no GFM ~~del~~, and the research log uses
# it to mark closed items. Registered as an inline pattern so it correctly does
# NOT fire inside code spans.
# --------------------------------------------------------------------------
_DEL_RE = r"(~{2})(.+?)\1"


class _DelProcessor(InlineProcessor):
    def handleMatch(self, m, data):
        el = ElementTree.Element("del")
        el.text = m.group(2)
        return el, m.start(0), m.end(0)


class StrikethroughExtension(Extension):
    def extendMarkdown(self, md):
        md.inlinePatterns.register(_DelProcessor(_DEL_RE, md), "del", 175)


EXTENSIONS = [
    "tables",           # GFM tables -- the docs are mostly tables
    "fenced_code",
    "codehilite",       # pygments; CSS for both themes is inlined below
    "toc",
    "sane_lists",
    "attr_list",
    "admonition",
    StrikethroughExtension(),
]
EXTENSION_CONFIGS = {
    "codehilite": {"guess_lang": False, "css_class": "highlight"},
    "toc": {"permalink": False, "toc_depth": "2-3"},
}

# Tables are wrapped so a 3,000-character research-log cell scrolls inside its
# own box instead of forcing the whole page sideways.
_TABLE_OPEN = re.compile(r"<table>")
_TABLE_CLOSE = re.compile(r"</table>")

CSS = """
:root{
  --bg:#fbfbfa; --surface:#fff; --surface-2:#f2f4f6; --ink:#1a1d21; --muted:#5b6672;
  --border:#e2e6ea; --rule:#cfd6dd; --accent:#2f6690; --accent-soft:#eaf1f6;
  --code-bg:#f5f7f9; --del:#96a0aa;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#11151a; --surface:#181d23; --surface-2:#1f262e; --ink:#e6ebf0; --muted:#95a1ad;
    --border:#28313a; --rule:#37424d; --accent:#7fb2d8; --accent-soft:#17242e;
    --code-bg:#161c22; --del:#6d7883;
  }
}
:root[data-theme="dark"]{
  --bg:#11151a; --surface:#181d23; --surface-2:#1f262e; --ink:#e6ebf0; --muted:#95a1ad;
  --border:#28313a; --rule:#37424d; --accent:#7fb2d8; --accent-soft:#17242e;
  --code-bg:#161c22; --del:#6d7883;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.page{max-width:none;width:min(1180px,100% - 3rem);margin:0 auto;padding:3rem 0 5rem}
h1,h2,h3,h4{line-height:1.25;text-wrap:balance;margin:2.2rem 0 .7rem;font-weight:650}
h1{font-size:2rem;margin-top:0;letter-spacing:-.015em}
h2{font-size:1.4rem;border-bottom:1px solid var(--rule);padding-bottom:.35rem;margin-top:2.6rem}
h3{font-size:1.12rem}
h4{font-size:1rem;color:var(--muted)}
p,li{max-width:80ch}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent}
a:hover{border-bottom-color:currentColor}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:2px}
hr{border:0;border-top:1px solid var(--rule);margin:2.5rem 0}
del{color:var(--del)}
strong{font-weight:650}
blockquote{margin:1.3rem 0;padding:.6rem 0 .6rem 1.1rem;border-left:3px solid var(--accent);
  background:var(--accent-soft);color:var(--ink)}
blockquote p{margin:.35rem 0}
code{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  font-size:.875em;background:var(--code-bg);border:1px solid var(--border);
  border-radius:3px;padding:.08em .34em}
pre{background:var(--code-bg);border:1px solid var(--border);border-radius:6px;
  padding:.9rem 1.1rem;overflow-x:auto;line-height:1.5}
pre code{background:none;border:0;padding:0;font-size:.85rem}
.table-wrap{overflow-x:auto;margin:1.3rem 0;border:1px solid var(--border);
  border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:.55rem .75rem;border-bottom:1px solid var(--border);
  text-align:left;vertical-align:top}
td{max-width:64ch}
th{background:var(--surface-2);font-weight:650;font-size:.78rem;letter-spacing:.04em;
  text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--rule);
  position:sticky;top:0;white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover{background:var(--surface-2)}
td code{white-space:nowrap}
img{max-width:100%;height:auto;display:block;margin:1.5rem 0;
  border:1px solid var(--border);border-radius:6px;background:var(--surface)}
figure{margin:1.5rem 0}
ul,ol{padding-left:1.3rem}
li{margin:.3rem 0}
.docmeta{color:var(--muted);font-size:.82rem;margin:0 0 2rem;padding-bottom:1rem;
  border-bottom:2px solid var(--rule);display:flex;gap:1.2rem;flex-wrap:wrap}
.toc{background:var(--surface);border:1px solid var(--border);border-radius:6px;
  padding:1rem 1.2rem;margin:0 0 2rem}
.toc-title{font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);font-weight:650;margin-bottom:.5rem}
.toc ul{padding-left:1.1rem;margin:.2rem 0}
.toc>ul{padding-left:0;list-style:none}
.toc>ul>li{margin:.25rem 0}
.toc a{color:var(--ink)}
.idx{list-style:none;padding:0;display:grid;gap:.15rem}
.idx li{margin:0}
.idx a{display:flex;justify-content:space-between;gap:1rem;padding:.5rem .75rem;
  border-radius:5px;border-bottom:0}
.idx a:hover{background:var(--surface-2)}
.idx .sz{color:var(--muted);font-size:.8rem;font-variant-numeric:tabular-nums}
@media print{body{background:#fff}.toc{break-inside:avoid}}
"""

PYGMENTS_CSS = """
.highlight pre{margin:0}
.highlight .c,.highlight .c1,.highlight .cm{color:#6b7783;font-style:italic}
.highlight .k,.highlight .kn,.highlight .kd{color:#a4459b;font-weight:600}
.highlight .s,.highlight .s1,.highlight .s2{color:#2f7d4f}
.highlight .n{color:inherit}
.highlight .nb{color:#2f6690}
.highlight .nf{color:#2f6690;font-weight:600}
.highlight .mi,.highlight .mf{color:#b5651d}
.highlight .o,.highlight .p{color:var(--muted)}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]) .highlight .c,
  :root:not([data-theme="light"]) .highlight .c1{color:#7d8894}
  :root:not([data-theme="light"]) .highlight .k,
  :root:not([data-theme="light"]) .highlight .kn{color:#d79bd0}
  :root:not([data-theme="light"]) .highlight .s,
  :root:not([data-theme="light"]) .highlight .s1,
  :root:not([data-theme="light"]) .highlight .s2{color:#83c99e}
  :root:not([data-theme="light"]) .highlight .nb,
  :root:not([data-theme="light"]) .highlight .nf{color:#7fb2d8}
  :root:not([data-theme="light"]) .highlight .mi,
  :root:not([data-theme="light"]) .highlight .mf{color:#e0a264}
}
"""


# A deliberately single-look theme: a printed offering document has one
# appearance, so this one commits to white and skips the dark-mode blocks.
# Colours are still painted explicitly so it holds on any host background.
PROSPECTUS_CSS = """
:root{
  --bg:#ffffff; --surface:#ffffff; --surface-2:#f6f7f8; --ink:#14181d; --muted:#5a636d;
  --border:#dcdfe3; --rule:#b9bfc6; --accent:#1f3d5c; --accent-soft:#f2f5f8;
  --code-bg:#f6f7f8; --del:#9aa2ab;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16.5px/1.72 Charter,"Bitstream Charter","Iowan Old Style",Georgia,serif;
  -webkit-font-smoothing:antialiased}
.page{width:min(880px,100% - 3.5rem);margin:0 auto;padding:4rem 0 6rem}
h1,h2,h3,h4{font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;
  line-height:1.25;text-wrap:balance;font-weight:600;color:var(--ink)}
h1{font-size:2.1rem;margin:0 0 .4rem;letter-spacing:-.018em;font-weight:650}
h2{font-size:1.16rem;margin:2.8rem 0 .8rem;padding-bottom:.4rem;
  border-bottom:1.5px solid var(--rule);letter-spacing:-.005em}
h3{font-size:1rem;margin:1.9rem 0 .5rem;color:var(--accent)}
h4{font-size:.94rem;margin:1.4rem 0 .4rem;color:var(--muted);font-weight:600}
p,li{max-width:74ch}
p{margin:0 0 1.05rem}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid #c9d3dd}
a:hover{border-bottom-color:var(--accent)}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
hr{border:0;border-top:1px solid var(--rule);margin:2.8rem 0}
del{color:var(--del)}
strong{font-weight:700}
em{font-style:italic}
blockquote{margin:1.5rem 0;padding:1rem 1.3rem;border-left:3px solid var(--accent);
  background:var(--accent-soft);font-size:1.02rem}
blockquote p{margin:.3rem 0}
code{font-family:"SF Mono",Menlo,Consolas,monospace;font-size:.84em;
  background:var(--code-bg);border:1px solid var(--border);border-radius:2px;
  padding:.06em .3em}
pre{background:var(--code-bg);border:1px solid var(--border);padding:.9rem 1.1rem;
  overflow-x:auto;line-height:1.5}
pre code{background:none;border:0;padding:0;font-size:.82rem}
.table-wrap{overflow-x:auto;margin:1.5rem 0;border-top:1.5px solid var(--rule);
  border-bottom:1.5px solid var(--rule)}
table{border-collapse:collapse;width:100%;
  font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;font-size:.86rem}
th,td{padding:.6rem .8rem;text-align:left;vertical-align:top;
  border-bottom:1px solid var(--border)}
td{max-width:58ch;font-variant-numeric:tabular-nums}
th{font-size:.72rem;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);
  font-weight:700;border-bottom:1px solid var(--rule);white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
img{max-width:100%;height:auto;display:block;margin:1.8rem 0;
  border:1px solid var(--border)}
ul,ol{padding-left:1.35rem;margin:0 0 1.05rem}
li{margin:.32rem 0}
.docmeta{display:none}
.toc{border-top:1.5px solid var(--rule);border-bottom:1.5px solid var(--rule);
  padding:1.1rem 0;margin:2rem 0 2.6rem}
.toc-title{font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;font-size:.7rem;
  letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
  font-weight:700;margin-bottom:.6rem}
.toc ul{padding-left:1.1rem;margin:.15rem 0}
.toc>ul{padding-left:0;list-style:none;
  font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;font-size:.88rem}
.toc>ul>li{margin:.22rem 0}
.toc a{color:var(--ink);border-bottom:0}
.toc a:hover{color:var(--accent)}
.idx{list-style:none;padding:0}
.idx a{display:flex;justify-content:space-between;gap:1rem;padding:.5rem 0;
  border-bottom:1px solid var(--border)}
.idx .sz{color:var(--muted);font-size:.82rem}
@media print{.toc{break-inside:avoid}h2{break-after:avoid}}
"""

# The prospectus theme commits to one appearance, so its syntax colours must not
# flip with the reader's OS preference -- it gets a light-only highlight block.
PROSPECTUS_PYGMENTS = PYGMENTS_CSS.split("@media")[0]


# A committed dark theme for internal working documents. Like the prospectus it
# is deliberately single-look: no media query, no stamp, every colour painted.
DARK_CSS = """
:root{
  --bg:#12171c; --surface:#181e25; --surface-2:#1f272f; --ink:#e6ebf0; --muted:#97a3af;
  --border:#2b343d; --rule:#3a4550; --accent:#6fb0dd; --accent-soft:#17242e;
  --code-bg:#161c22; --del:#6d7883;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.68 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.page{width:min(1100px,100% - 3rem);margin:0 auto;padding:3rem 0 5rem}
h1,h2,h3,h4{line-height:1.25;text-wrap:balance;font-weight:650;color:var(--ink)}
h1{font-size:2rem;margin:0 0 .5rem;letter-spacing:-.015em}
h2{font-size:1.32rem;margin:2.8rem 0 .8rem;padding-bottom:.4rem;
  border-bottom:1px solid var(--rule)}
h3{font-size:1.08rem;margin:1.9rem 0 .5rem;color:var(--accent)}
h4{font-size:.98rem;margin:1.4rem 0 .4rem;color:var(--muted)}
p,li{max-width:82ch}
p{margin:0 0 1.05rem}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent}
a:hover{border-bottom-color:currentColor}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
hr{border:0;border-top:1px solid var(--rule);margin:2.6rem 0}
del{color:var(--del)}
strong{font-weight:650;color:#f4f7fa}
blockquote{margin:1.4rem 0;padding:.9rem 1.2rem;border-left:3px solid var(--accent);
  background:var(--accent-soft)}
blockquote p{margin:.3rem 0}
code{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  font-size:.86em;background:var(--code-bg);border:1px solid var(--border);
  border-radius:3px;padding:.08em .34em;color:#cfe0ee}
pre{background:var(--code-bg);border:1px solid var(--border);border-radius:6px;
  padding:.9rem 1.1rem;overflow-x:auto;line-height:1.5}
pre code{background:none;border:0;padding:0;font-size:.85rem}
.table-wrap{overflow-x:auto;margin:1.4rem 0;border:1px solid var(--border);
  border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.89rem}
th,td{padding:.55rem .78rem;border-bottom:1px solid var(--border);
  text-align:left;vertical-align:top}
td{max-width:62ch;font-variant-numeric:tabular-nums}
th{background:var(--surface-2);font-weight:650;font-size:.74rem;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--rule);
  position:sticky;top:0;white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover{background:var(--surface-2)}
td code{white-space:nowrap}
img{max-width:100%;height:auto;display:block;margin:1.6rem 0;
  border:1px solid var(--border);border-radius:6px}
ul,ol{padding-left:1.3rem;margin:0 0 1.05rem}
li{margin:.3rem 0}
.docmeta{color:var(--muted);font-size:.8rem;margin:0 0 2rem;padding-bottom:1rem;
  border-bottom:1px solid var(--rule);display:flex;gap:1.2rem;flex-wrap:wrap}
.toc{background:var(--surface);border:1px solid var(--border);border-radius:6px;
  padding:1rem 1.2rem;margin:0 0 2.2rem}
.toc-title{font-size:.7rem;letter-spacing:.13em;text-transform:uppercase;
  color:var(--muted);font-weight:650;margin-bottom:.5rem}
.toc ul{padding-left:1.1rem;margin:.2rem 0}
.toc>ul{padding-left:0;list-style:none;font-size:.9rem}
.toc>ul>li{margin:.24rem 0}
.toc a{color:var(--ink)}
.toc a:hover{color:var(--accent)}
.idx{list-style:none;padding:0}
.idx a{display:flex;justify-content:space-between;gap:1rem;padding:.5rem .7rem;
  border-radius:5px;border-bottom:0}
.idx a:hover{background:var(--surface-2)}
.idx .sz{color:var(--muted);font-size:.8rem}
"""

DARK_PYGMENTS = """
.highlight pre{margin:0}
.highlight .c,.highlight .c1,.highlight .cm{color:#7d8894;font-style:italic}
.highlight .k,.highlight .kn,.highlight .kd{color:#d79bd0;font-weight:600}
.highlight .s,.highlight .s1,.highlight .s2{color:#83c99e}
.highlight .n{color:inherit}
.highlight .nb,.highlight .nf{color:#7fb2d8}
.highlight .mi,.highlight .mf{color:#e0a264}
.highlight .o,.highlight .p{color:var(--muted)}
"""

THEMES = {"default": None, "prospectus": PROSPECTUS_CSS, "dark": DARK_CSS}
THEME_PYGMENTS = {"prospectus": PROSPECTUS_PYGMENTS, "dark": DARK_PYGMENTS}

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="page">
{meta}{toc}{body}
</div>
</body>
</html>
"""


def wrap_tables(html_text: str) -> str:
    """Put each table in its own horizontally scrolling container."""
    out = _TABLE_OPEN.sub('<div class="table-wrap"><table>', html_text)
    return _TABLE_CLOSE.sub("</table></div>", out)


def rewrite_md_links(html_text: str) -> str:
    """Point inter-document links at the rendered siblings instead of the source.

    Only touches relative hrefs -- absolute URLs and anchors are left alone, so
    a link to https://example.com/x.md still works.
    """
    def repl(m: re.Match) -> str:
        href = m.group(1)
        if href.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        return f'href="{href[:-3]}.html{m.group(2)}"'

    return re.sub(r'href="([^"]+?\.md)((?:#[^"]*)?)"', repl, html_text)


def title_of(md_text: str, fallback: str) -> str:
    """First H1 if there is one, else the filename."""
    for line in md_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def render_markdown(md_text: str) -> tuple[str, str]:
    """Return (body_html, toc_html)."""
    md = markdown.Markdown(extensions=EXTENSIONS, extension_configs=EXTENSION_CONFIGS)
    body = md.convert(md_text)
    body = wrap_tables(body)
    body = rewrite_md_links(body)
    toc = getattr(md, "toc", "") or ""
    return body, toc


def _css(theme: str) -> str:
    """Prospectus is single-look by design; the default theme ships both modes."""
    override = THEMES.get(theme)
    if override:
        return override + THEME_PYGMENTS[theme]
    return CSS + PYGMENTS_CSS


def render_file(src: Path, out_dir: Path, with_toc: bool = True,
                toc_min_headings: int = 4, theme: str = "default") -> Path:
    md_text = src.read_text(encoding="utf-8")
    body, toc = render_markdown(md_text)
    title = title_of(md_text, src.stem)

    # A table of contents on a three-heading document is just noise.
    show_toc = with_toc and toc.count("<li>") >= toc_min_headings
    toc_html = (f'<nav class="toc"><div class="toc-title">Contents</div>{toc}</nav>'
                if show_toc else "")

    words = len(md_text.split())
    meta = (f'<div class="docmeta"><span>{html.escape(src.name)}</span>'
            f'<span>{words:,} words</span>'
            f'<span>{len(md_text.splitlines()):,} lines</span></div>')

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (src.stem + ".html")
    dest.write_text(PAGE.format(title=html.escape(title), css=_css(theme),
                                meta=meta, toc=toc_html, body=body), encoding="utf-8")
    return dest


def build_index(rendered: list[tuple[Path, Path]], out_dir: Path,
                theme: str = "default") -> Path:
    rows = []
    for src, dest in sorted(rendered, key=lambda p: p[0].name):
        title = title_of(src.read_text(encoding="utf-8"), src.stem)
        kb = src.stat().st_size / 1024
        rows.append(f'<li><a href="{dest.name}"><span>{html.escape(title)}</span>'
                    f'<span class="sz">{kb:,.0f} KB</span></a></li>')
    body = (f"<h1>Documents</h1><p>{len(rendered)} rendered from markdown.</p>"
            f'<ul class="idx">{"".join(rows)}</ul>')
    dest = out_dir / "index.html"
    dest.write_text(PAGE.format(title="Documents", css=_css(theme),
                                meta="", toc="", body=body), encoding="utf-8")
    return dest


def collect(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(p.glob("*.md")))
        elif p.suffix.lower() == ".md":
            found.append(p)
        else:
            print(f"skipping (not markdown): {p}", file=sys.stderr)
    seen, unique = set(), []
    for p in found:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)
    return unique


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="render_md", description="Render markdown docs to self-contained HTML.")
    ap.add_argument("paths", nargs="+", help="markdown files and/or directories")
    ap.add_argument("--out", default="output_html", help="output directory")
    ap.add_argument("--index", action="store_true",
                    help="also write index.html linking every rendered page")
    ap.add_argument("--no-toc", action="store_true", help="never emit a contents block")
    ap.add_argument("--theme", choices=sorted(THEMES), default="default",
                    help="'prospectus' = white serif; 'dark' = committed dark; "
                         "both are single-look (no theme switching)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    sources = collect(args.paths)
    if not sources:
        print("no markdown files found", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    rendered = []
    for src in sources:
        dest = render_file(src, out_dir, with_toc=not args.no_toc,
                           theme=args.theme)
        rendered.append((src, dest))
        if not args.quiet:
            print(f"  {src}  ->  {dest}")

    if args.index and rendered:
        idx = build_index(rendered, out_dir, theme=args.theme)
        if not args.quiet:
            print(f"  index  ->  {idx}")

    if not args.quiet:
        print(f"\n{len(rendered)} file(s) rendered to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
