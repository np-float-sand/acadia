"""
LOCKED keyword/phrase list for the backlog pricing-power text-mining signal.

Written into docs/backlog-pricing-power-preregistration.md BEFORE any filing was
scored. This module just makes that list importable so score.py can't silently
drift from what was pre-registered -- do not add/remove/edit phrases here after
scoring has run; if the list needs to change, that is a new pre-registration.
"""
from __future__ import annotations

KEYWORDS: dict[str, list[str]] = {
    "lead_time": ["lead time", "extended lead", "backlog extends into"],
    "price_escalation": ["price escalation", "escalation clause", "surcharge",
                          "pass-through", "pricing action"],
    "margin_quality": ["book-to-bill", "backlog margin", "firm pricing", "price increase"],
}

ALL_PHRASES: list[str] = [p for bucket in KEYWORDS.values() for p in bucket]
assert len(ALL_PHRASES) == 12
