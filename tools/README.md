# tools/

Small repo-wide utilities (not part of any strategy package).

| tool | what |
|---|---|
| `md2html.py` | Render a Markdown file to a self-contained, styled HTML document (and optionally a PDF). Light / dark / auto theme. GFM tables, fenced code, footnotes. |

## md2html

```sh
poetry run python tools/md2html.py docs/RESEARCH-LOG.md            # -> docs/RESEARCH-LOG.html, theme=auto
poetry run python tools/md2html.py notes.md --theme dark --no-toggle
poetry run python tools/md2html.py report.md -o /tmp/report.html --toc --pdf
```

- `--theme auto` (default): follows the reader's OS setting; a corner button toggles and
  remembers the choice in `localStorage`. `--theme light` / `--theme dark` lock it.
- `--pdf`: also writes `<output>.pdf` via headless Chrome, always light-locked, A4, with
  page-break rules so tables and code blocks don't split. Falls back to a message if no
  Chrome/Chromium is found.
- `--toc` inserts a table of contents; headings always get anchor ids.

Importable: `from tools.md2html import md_to_html, render_pdf`.

Depends on `markdown` (added to `pyproject.toml`; the `poetry add` also bumped
`pyarrow` 24 -> 25 as a transitive re-resolve).
