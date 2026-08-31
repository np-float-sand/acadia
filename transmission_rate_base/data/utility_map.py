from __future__ import annotations

import pandas as pd

# ticker -> list of exact utility_name_ferc1 strings for its regulated electric
# filers. Matched against core_pudl__assn_ferc1_pudl_utilities (case-sensitive).
# Both DBF-era and XBRL-era filer ids are pulled in automatically via shared
# utility_id_pudl (see resolve_filers).
#
# Scope choices (documented, frozen -- never revised from results):
#  - Generation-only LLCs (PSEG Power, Westar Generating, Allegheny Generating)
#    are excluded: they hold little/no transmission rate base.
#  - Pure transmission affiliates (AEP * Transmission Company, TrAIL, Ameren
#    Transmission of Illinois, NextEra Energy Transmission) ARE included -- they
#    are the cleanest transmission-rate-base exposure.
#  - Transource (AEP/Evergy JV) excluded to avoid a double claim.
#  - Gulf Power excluded: ownership moved Southern -> NextEra (2019) -> merged
#    into FPL (2021); the filer-count guard would blank the transition years
#    anyway.
#  - Montana Power (NWE predecessor) excluded: pre-2002 it was a very different
#    (part-unregulated) company.
PARENT_FILERS: dict[str, list[str]] = {
    "AEP": [
        "Appalachian Power Company", "Ohio Power Company", "Indiana Michigan Power Company",
        "AEP Texas Central Company", "AEP Texas North Company", "aep texas (pudl determined)",
        "Southwestern Electric Power Company", "Kentucky Power Company",
        "Wheeling Power Company", "Kingsport Power Company",
        "AEP Appalachian Transmission Company, Inc.", "AEP Indiana Michigan Transmission Company, Inc.",
        "AEP Kentucky Transmission Company, Inc.", "AEP Ohio Transmission Company, Inc.",
        "AEP Oklahoma Transmission Company, Inc.", "AEP Southwestern Transmission Company, Inc.",
        "AEP West Virginia Transmission Company, Inc.",
    ],
    "AEE": [
        "UNION ELECTRIC COMPANY", "Ameren Illinois Company", "Central Illinois Light Company",
        "Central Illinois Public Service Company", "Ameren Transmission Company of Illinois",
    ],
    "AES": ["Indianapolis Power & Light Company", "The Dayton Power and Light Company"],
    "AVA": ["Avista Corporation"],
    "BKH": [
        "Black Hills Power, Inc.", "Black Hills/Colorado Electric Utility Company, LP",
        "Cheyenne Light, Fuel and Power Company",
    ],
    "CMS": ["Consumers Energy Company"],
    "CNP": ["CenterPoint Energy Houston Electric, LLC"],
    "D": ["VIRGINIA ELECTRIC AND POWER COMPANY", "South Carolina Electric & Gas Company"],
    "DTE": ["DTE Electric Company"],
    "DUK": [
        "Duke Energy Carolinas, LLC", "Duke Energy Progress, Inc.", "Duke Energy Florida, Inc.",
        "Duke Energy Indiana, Inc.", "Duke Energy Ohio, Inc.", "Duke Energy Kentucky, Inc.",
    ],
    "ED": [
        "Consolidated Edison Company of New York, Inc.", "Orange and Rockland Utilities, Inc",
        "Rockland Electric Company",
    ],
    "EIX": ["Southern California Edison Company"],
    "ES": [
        "Connecticut Light and Power Company, The", "Western Massachusetts Electric Company",
        "Public Service Company of New Hampshire", "NSTAR Electric Company",
        "Boston Edison Company", "Commonwealth Electric Company", "Cambridge Electric Light Company",
    ],
    "ETR": [
        "Entergy Arkansas, Inc.", "Entergy Louisiana, LLC", "entergy mississippi, llc",
        "entergy new orleans, llc", "Entergy Texas, Inc.", "Entergy Gulf States Louisiana, L.L.C.",
        "System Energy Resources, Inc.",
    ],
    "EVRG": [
        "Kansas City Power & Light Company", "KCP&L Greater Missouri Operations Company",
        "Westar Energy, Inc.", "evergy kansas south, inc.",
    ],
    "EXC": [
        "Commonwealth Edison Company", "PECO Energy Company", "Baltimore Gas and Electric Company",
        "Potomac Electric Power Company", "Delmarva Power & Light Company",
        "Atlantic City Electric Company", "Commonwealth Edison Company of Indiana, Inc.",
    ],
    "FE": [
        "Ohio Edison Company", "Cleveland Electric Illuminating Company, The",
        "Toledo Edison Company, The", "Pennsylvania Power Company", "Metropolitan Edison Company",
        "Pennsylvania Electric Company", "Jersey Central Power & Light Company",
        "WEST PENN POWER COMPANY", "MONONGAHELA POWER COMPANY", "THE POTOMAC EDISON COMPANY",
        "Trans-Allegheny Interstate Line Company", "firstenergy pennsylvania electric company",
    ],
    "HE": [
        "Hawaiian Electric Company, Inc.", "Hawaii Electric Light Company, Inc.",
        "MAUI ELECTRIC COMPANY, LIMITED",
    ],
    "IDA": ["Idaho Power Company"],
    "LNT": [
        "Interstate Power and Light Company", "Wisconsin Power and Light Company",
        "Interstate Power Company",
    ],
    "MGEE": ["Madison Gas and Electric Company"],
    "NEE": [
        "Florida Power & Light Company", "NextEra Energy Transmission New York, Inc.",
        "nextera energy transmission midatlantic indiana, inc.",
    ],
    "NWE": ["NorthWestern Corporation", "northwestern energy public service corporation"],
    "OGE": ["Oklahoma Gas and Electric Company"],
    "OTTR": ["Otter Tail Power Company"],
    "PCG": ["PACIFIC GAS AND ELECTRIC COMPANY"],
    "PEG": ["Public Service Electric and Gas Company"],
    "PNW": ["Arizona Public Service Company"],
    "POR": ["Portland General Electric Company"],
    "PPL": [
        "PPL Electric Utilities Corporation", "Kentucky Utilities Company",
        "Louisville Gas and Electric Company",
    ],
    "SO": [
        "ALABAMA POWER COMPANY", "Georgia Power Company", "Mississippi Power Company",
        "Savannah Electric and Power Company",
    ],
    "TXNM": ["Texas-New Mexico Power Company"],
    "WEC": [
        "Wisconsin Electric Power Company", "Wisconsin Public Service Corporation",
        "Upper Peninsula Power Company",
    ],
    "XEL": [
        "Northern States Power Company (Minnesota)", "Northern States Power Company (Wisconsin)",
        "Public Service Company of Colorado", "Southwestern Public Service Company",
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
