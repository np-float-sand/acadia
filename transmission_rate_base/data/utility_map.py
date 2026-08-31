from __future__ import annotations

import pandas as pd

# ticker -> list of exact utility_name_ferc1 strings for its regulated electric
# filers. Built by hand from each parent's latest 10-K subsidiary list, matched
# to core_pudl__assn_ferc1_pudl_utilities. Both DBF-era and XBRL-era filer ids
# are pulled in automatically via shared utility_id_pudl (see resolve_filers).
#
# SEED ENTRIES ONLY -- the executor completes this to the full config.UNIVERSE_SEED
# using the procedure in the spec (§5). Every value string must appear verbatim
# in the crosswalk's utility_name_ferc1 column (case-sensitive).
PARENT_FILERS: dict[str, list[str]] = {
    "AEP": [
        "Appalachian Power Company", "Ohio Power Company", "Indiana Michigan Power Company",
        "AEP Texas Central Company", "AEP Texas North Company", "aep texas (pudl determined)",
        "Southwestern Electric Power Company", "Kentucky Power Company",
        "Wheeling Power Company", "Kingsport Power Company",
    ],
    "NEE": ["Florida Power & Light Company"],
    "DUK": [
        "Duke Energy Carolinas, LLC", "Duke Energy Progress, Inc.", "Duke Energy Florida, Inc.",
        "Duke Energy Indiana, Inc.", "Duke Energy Ohio, Inc.", "Duke Energy Kentucky, Inc.",
    ],
    "XEL": [
        "Northern States Power Company (Minnesota)", "Northern States Power Company (Wisconsin)",
        "Southwestern Public Service Company",
    ],
    "ETR": [
        "Entergy Arkansas, Inc.", "Entergy Louisiana, LLC", "entergy mississippi, llc",
        "entergy new orleans, llc", "Entergy Texas, Inc.",
    ],
}


def _pudl_ids_for_names(xwalk: pd.DataFrame, names: list[str]) -> set:
    have = set(xwalk["utility_name_ferc1"])
    missing = [n for n in names if n not in have]
    if missing:
        raise KeyError(f"utility_name_ferc1 not in crosswalk: {missing}")
    return set(xwalk.loc[xwalk["utility_name_ferc1"].isin(names), "utility_id_pudl"].dropna())


def resolve_filers(xwalk: pd.DataFrame,
                   parents: dict[str, list[str]] | None = None) -> dict[str, list[int]]:
    """ticker -> sorted list of utility_id_ferc1 ints. Unions every FERC id that
    shares a utility_id_pudl with any named filer for that ticker."""
    parents = PARENT_FILERS if parents is None else parents
    out: dict[str, list[int]] = {}
    for ticker, names in parents.items():
        pudl_ids = _pudl_ids_for_names(xwalk, names)
        ids = xwalk.loc[xwalk["utility_id_pudl"].isin(pudl_ids), "utility_id_ferc1"]
        out[ticker] = sorted(int(i) for i in ids.dropna().unique())
    return out


def validate(xwalk: pd.DataFrame, parents: dict[str, list[str]] | None = None) -> None:
    """Raise ValueError if any ticker resolves to zero FERC filer ids, or any
    FERC id is claimed by two tickers."""
    parents = PARENT_FILERS if parents is None else parents
    resolved = resolve_filers(xwalk, parents)
    seen: dict[int, str] = {}
    for ticker, ids in resolved.items():
        if not ids:
            raise ValueError(f"{ticker} resolves to zero FERC filer ids")
        for i in ids:
            if i in seen and seen[i] != ticker:
                raise ValueError(f"FERC id {i} mapped to both {seen[i]} and {ticker}")
            seen[i] = ticker
