# utils/

Small standalone tools. Not part of any strategy package.

## `render_md.py` — markdown to self-contained HTML

Renders the repo's docs to HTML you can open straight from disk. All CSS is
inlined, so there is no build step, no network fetch, and no broken page if you
open it offline or mail someone the file.

    python utils/render_md.py docs/capex-intensity-proposal.md
    python utils/render_md.py docs/ --out output_html --index
    python utils/render_md.py docs/*.md --out site --no-toc

| flag | effect |
|---|---|
| `--out DIR` | output directory (default `output_html`) |
| `--index` | also write `index.html` linking every rendered page |
| `--no-toc` | never emit a contents block |
| `--quiet` | suppress per-file output |

Accepts files, directories, or globs. Directories are expanded to their `*.md`
children; duplicates are dropped.

### Built for what these docs actually contain

- **Wide GFM tables.** The research log has cells holding whole paragraphs, so
  every table is wrapped in its own horizontally scrolling container and the
  page body never scrolls sideways. Headers stay pinned while a long table scrolls.
- **Escaped pipes.** `\|t\| >= 2.0` inside a table cell stays one cell.
- **Strikethrough.** `~~closed item~~` renders as `<del>` -- python-markdown has
  no GFM strikethrough, so this ships as a small inline-pattern extension. It
  correctly does *not* fire inside code spans.
- **Fenced code** with Pygments highlighting, themed for light and dark.
- **Cross-document links.** A relative `](other.md)` is rewritten to
  `other.html` so the rendered set is navigable. Absolute URLs and bare anchors
  are left untouched.
- **Contents block** auto-emitted only when a document has 4+ headings -- a
  table of contents on a three-heading memo is noise.
- **Light and dark** via `prefers-color-scheme`, with every colour token defined
  on bare `:root` so the default (un-stamped) theme renders correctly.

### Verified on

`docs/capex-intensity-proposal.md` (5 tables, blockquote, inline code) and
`docs/RESEARCH-LOG.md` (232 KB, 14 strikethroughs, 50 escaped pipes, zero
unprocessed table rows). Whole-repo render: 104 documents, ~2.5 MB.

Tests: `tests/utils/test_render_md.py` (17, no network).
