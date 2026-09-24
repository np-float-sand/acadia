"""Out-of-sample log for the frozen composite (plan step 1).

    python -m research.xsec_alpha_probe.live_log

Appends, for every month-end not yet logged, each `screen_auto` member's frozen
composite z and whether it is in the TOP_HALF book. The file is tracked in git,
so rows written on the day they are computed are the out-of-sample record:
never rewrite past rows. Refresh the caches first so the latest month exists.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.xsec_alpha_probe import data
from research.xsec_alpha_probe.tilt import book_membership, composite

LOG = Path(__file__).parent / "live" / "composite_scores.csv"
FIRST = pd.Timestamp("2026-08-31")      # last in-sample month; the log starts here


def rows_for(dt, book: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    members = list(book.columns[book.loc[dt]])
    z = comp.loc[dt].reindex(members).fillna(0.0).sort_values(ascending=False)
    top = set(z.index[: max(1, len(z) // 2)])
    return pd.DataFrame({"date": dt.date().isoformat(), "ticker": z.index, "composite_z": z.round(4).values,
                         "top_half": [t in top for t in z.index],
                         "logged_on": pd.Timestamp.today().date().isoformat()})


def main() -> pd.DataFrame:
    univ = data.universe()
    book = book_membership(univ.index)
    comp = composite(univ, book)
    done = set(pd.read_csv(LOG)["date"]) if LOG.exists() else set()
    new = [rows_for(dt, book, comp) for dt in univ.index
           if dt >= FIRST and dt.date().isoformat() not in done and book.loc[dt].any()]
    if not new:
        print("nothing new to log")
        return pd.DataFrame()
    out = pd.concat(new)
    out.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
