import re

import pytest

from utils import render_md


def test_tables_are_wrapped_for_horizontal_scroll():
    md = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    body, _ = render_md.render_markdown(md)
    assert body.count('<div class="table-wrap"><table>') == 1
    assert body.count("</table></div>") == 1
    assert "<th>a</th>" in body


def test_strikethrough_renders_as_del():
    body, _ = render_md.render_markdown("~~closed item~~ and ~~another~~")
    assert body.count("<del>") == 2
    assert "<del>closed item</del>" in body


def test_strikethrough_does_not_fire_inside_code():
    body, _ = render_md.render_markdown("`~~not struck~~`")
    assert "<del>" not in body


def test_escaped_pipe_survives_inside_a_table_cell():
    """The research log writes t-stat bars as \\| inside table cells."""
    md = "| metric | value |\n|---|---|\n| t-stat | \\|t\\| >= 2.0 |\n"
    body, _ = render_md.render_markdown(md)
    # `>` is correctly HTML-escaped; the pipes must survive as literal characters
    assert "|t| &gt;= 2.0" in body
    assert body.count("<td>") == 2          # the escaped pipes did not split the cell


def test_fenced_code_is_highlighted():
    body, _ = render_md.render_markdown("```python\nx = 1\n```")
    assert "highlight" in body
    assert "<pre" in body


def test_blockquote_and_inline_formatting():
    body, _ = render_md.render_markdown("> **bold** and `code`\n")
    assert "<blockquote>" in body
    assert "<strong>bold</strong>" in body
    assert "<code>code</code>" in body


def test_md_links_are_rewritten_to_html():
    body, _ = render_md.render_markdown("[x](docs/other.md) [y](docs/other.md#sec)")
    assert 'href="docs/other.html"' in body
    assert 'href="docs/other.html#sec"' in body


def test_absolute_and_anchor_links_are_left_alone():
    body, _ = render_md.render_markdown(
        "[a](https://example.com/spec.md) [b](#section)")
    assert 'href="https://example.com/spec.md"' in body
    assert 'href="#section"' in body


def test_title_prefers_first_h1():
    assert render_md.title_of("# Real Title\n\ntext", "fallback") == "Real Title"
    assert render_md.title_of("no heading here", "fallback") == "fallback"


def test_render_file_writes_self_contained_html(tmp_path):
    src = tmp_path / "doc.md"
    src.write_text("# Doc\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    dest = render_md.render_file(src, tmp_path / "out")
    html = dest.read_text()
    assert dest.name == "doc.html"
    assert "<title>Doc</title>" in html
    assert "<style>" in html                      # CSS inlined
    assert "http://" not in html and "cdn" not in html.lower()   # no network deps
    assert "table-wrap" in html


def test_toc_suppressed_for_short_documents(tmp_path):
    src = tmp_path / "short.md"
    src.write_text("# One\n\ntext\n\n## Two\n\nmore\n")
    html = render_md.render_file(src, tmp_path / "out").read_text()
    assert 'class="toc"' not in html


def test_toc_emitted_for_long_documents(tmp_path):
    src = tmp_path / "long.md"
    src.write_text("# T\n\n" + "".join(f"## H{i}\n\nbody\n\n" for i in range(6)))
    html = render_md.render_file(src, tmp_path / "out").read_text()
    assert 'class="toc"' in html


def test_collect_expands_directories_and_dedupes(tmp_path):
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "b.md").write_text("# B")
    (tmp_path / "c.txt").write_text("ignored")
    found = render_md.collect([str(tmp_path), str(tmp_path / "a.md")])
    assert [p.name for p in found] == ["a.md", "b.md"]


def test_index_lists_every_rendered_page(tmp_path):
    for name in ("one", "two"):
        (tmp_path / f"{name}.md").write_text(f"# {name.title()}\n\ntext\n")
    rc = render_md.main([str(tmp_path), "--out", str(tmp_path / "out"),
                         "--index", "--quiet"])
    assert rc == 0
    index = (tmp_path / "out" / "index.html").read_text()
    assert 'href="one.html"' in index and 'href="two.html"' in index
    assert "One" in index and "Two" in index


def test_main_returns_error_when_nothing_found(tmp_path, capsys):
    assert render_md.main([str(tmp_path / "nope.txt"), "--quiet"]) == 1


def test_dark_theme_tokens_are_defined_in_base_root():
    """Every colour must exist on bare :root, or the un-stamped theme breaks."""
    base = render_md.CSS.split("@media")[0]
    for token in ("--bg", "--ink", "--border", "--accent", "--code-bg", "--del"):
        assert token in base, f"{token} missing from base :root"


@pytest.mark.parametrize("doc", ["docs/capex-intensity-proposal.md"])
def test_real_repo_document_renders(tmp_path, doc):
    from pathlib import Path
    src = Path(doc)
    if not src.exists():
        pytest.skip(f"{doc} not present")
    html = render_md.render_file(src, tmp_path / "out").read_text()
    assert "<table>" in html
    assert "Capex-Intensity" in html
    assert not re.search(r"^\|", html, re.M)      # no unprocessed table rows


def test_images_are_responsive():
    """Embedded charts are 2000px+ native; they must never overflow the page."""
    assert "img{max-width:100%" in render_md.CSS.replace(" ", "")


def test_prospectus_theme_is_single_look(tmp_path):
    """A printed offering document has one appearance -- no theme switching."""
    src = tmp_path / "p.md"
    src.write_text("# P\n\n```python\nx=1\n```\n\n| a |\n|---|\n| 1 |\n")
    html = render_md.render_file(src, tmp_path / "out", theme="prospectus").read_text()
    assert "--bg:#ffffff" in html                      # genuinely white
    assert "prefers-color-scheme" not in html          # including the code block
    assert "Charter" in html                           # serif body
    assert "table-wrap" in html


def test_default_theme_still_ships_both_modes(tmp_path):
    src = tmp_path / "d.md"
    src.write_text("# D\n\ntext\n")
    html = render_md.render_file(src, tmp_path / "out").read_text()
    assert "prefers-color-scheme" in html


def test_dark_theme_is_committed_dark(tmp_path):
    """Internal docs render dark unconditionally, not on OS preference."""
    src = tmp_path / "d.md"
    src.write_text("# D\n\n```python\nx=1\n```\n\n| a |\n|---|\n| 1 |\n")
    html = render_md.render_file(src, tmp_path / "out", theme="dark").read_text()
    assert "--bg:#12171c" in html
    assert "prefers-color-scheme" not in html      # committed, not switching
    assert "table-wrap" in html


def test_every_theme_defines_its_own_highlight_colours():
    """A theme must not inherit the other theme's syntax colours."""
    for name in ("prospectus", "dark"):
        assert name in render_md.THEME_PYGMENTS
        assert "@media" not in render_md.THEME_PYGMENTS[name]
