"""
MD&A extraction + locked-keyword density scoring for one filing's text.

Per docs/backlog-pricing-power-preregistration.md §3: 10-K -> text between
"Item 7" and "Item 7A"/"Item 8"; 10-Q -> text between "Item 2" and "Item 3"/
"Item 4". Falls back to whole-document scoring (flagged) when no header match
is found -- some filers use non-standard header formatting.

Not production code. Throwaway research script.
"""
from __future__ import annotations

import re

from keywords import ALL_PHRASES

# Item-header regexes. Tolerant of "Item 7.", "Item  7 -", "ITEM 7:", etc.
# \b boundaries + optional letter suffix (7A) handled by the explicit alt patterns.
_ITEM7 = re.compile(r"\bItem\s*7\.?\s*[-:—]?\s*Management", re.IGNORECASE)
_ITEM7A_OR_8 = re.compile(r"\bItem\s*7A\.?\b|\bItem\s*8\.?\s*[-:—]?\s*Financial", re.IGNORECASE)
_ITEM2 = re.compile(r"\bItem\s*2\.?\s*[-:—]?\s*Management", re.IGNORECASE)
_ITEM3_OR_4 = re.compile(r"\bItem\s*3\.?\s*[-:—]?\s*Quantitative|\bItem\s*4\.?\s*[-:—]?\s*Controls",
                          re.IGNORECASE)


def extract_mdna(text: str, form: str) -> tuple[str, bool]:
    """Return (mdna_text, extracted_ok). Falls back to whole doc if no match."""
    if form == "10-K":
        start_re, end_re = _ITEM7, _ITEM7A_OR_8
    else:
        start_re, end_re = _ITEM2, _ITEM3_OR_4

    starts = list(start_re.finditer(text))
    if not starts:
        return text, False
    # Real MD&A section is usually the LAST match of the start header (the
    # first is typically the table-of-contents entry, which also matches).
    start = starts[-1].start()
    ends = [m.start() for m in end_re.finditer(text) if m.start() > start]
    if not ends:
        return text[start:], False
    end = min(ends)
    section = text[start:end]
    # Sanity floor: a real MD&A section is not a few words (TOC mismatch).
    if len(section.split()) < 200:
        return text, False
    return section, True


def score_text(mdna_text: str) -> dict:
    """Keyword counts, total count, word count, density per 1,000 words."""
    lower = mdna_text.lower()
    counts = {p: lower.count(p) for p in ALL_PHRASES}
    total = sum(counts.values())
    n_distinct = sum(1 for c in counts.values() if c > 0)
    n_words = max(1, len(mdna_text.split()))
    density = total / n_words * 1000.0
    out = {"total_count": total, "n_distinct_phrases": n_distinct,
           "n_words": n_words, "density_per_1k_words": density}
    out.update({f"kw__{p.replace(' ', '_').replace('-', '_')}": c for p, c in counts.items()})
    return out
