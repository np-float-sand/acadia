"""Point-in-time hedge-adjusted open share per (ticker, filing date, year, hub).

VST / TLN: parsed from the cached 8-K earnings-release text ("hedged approximately X% ... for YYYY").
CEG: hand-transcribed from the 2023 "Generation and Hedges" tables (midpoint of each range, capped at
100%); CEG stopped publishing the table in 2024. NRG publishes no hedge % and is excluded.
Region weights per docs/forward-power-hedge-ic-preregistration.md.
"""
from __future__ import annotations

import pathlib
import re

import pandas as pd

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "_cache"

# VST generation share by hub (10-K production volumes; East + Sunset -> PJM, West dropped)
VST_REGION = [("1900-01-01", {"TX": 0.53, "PJ": 0.47}), ("2024-03-01", {"TX": 0.46, "PJ": 0.54})]
TLN_REGION = {"PJ": 1.0}

# CEG: {filed: {year: {region: (GWh, hedge_mid)}}}; Midwest CMC plants are 100% hedged per CEG.
_CEG_RAW = {
    "2023-02-16": {2023: dict(mwx=(41300, 92.5), cmc=(54200, 100), ma=(54800, 101.5), tx=(20400, 91.5), ny=(25800, 80.5)),
                   2024: dict(mwx=(42300, 64.5), cmc=(54100, 100), ma=(56400, 74.5), tx=(20100, 62.5), ny=(25300, 61.5))},
    "2023-05-04": {2023: dict(mwx=(41200, 94.5), cmc=(54400, 100), ma=(54700, 99.5), tx=(20800, 89.5), ny=(25700, 90.5)),
                   2024: dict(mwx=(42300, 65.5), cmc=(54100, 100), ma=(56300, 73.5), tx=(20500, 64.5), ny=(25300, 71.5))},
    "2023-08-03": {2023: dict(mwx=(40900, 98.5), cmc=(54500, 100), ma=(54800, 99.5), tx=(20300, 94.5), ny=(25700, 93.5)),
                   2024: dict(mwx=(42300, 64.5), cmc=(54100, 100), ma=(56100, 75.5), tx=(21000, 73.5), ny=(25300, 81.5))},
    "2023-11-06": {2023: dict(mwx=(40900, 98.5), cmc=(54600, 100), ma=(55300, 98.5), tx=(19100, 97.5), ny=(25500, 97.5)),
                   2024: dict(mwx=(42200, 55.5), cmc=(53800, 100), ma=(55600, 78.5), tx=(21600, 89.5), ny=(25000, 83.5))},
}
_CEG_HUB = {"mwx": "PJ", "cmc": "PJ", "ma": "PJ", "ny": "PJ", "tx": "TX"}

_PAT_A = re.compile(r"(?:approximately |over )?(\d{1,3})%\s+(?:of (?:its|our) expected generation volumes\s+)?"
                    r"(?:on average\s+)?for\s+(?:the balance of\s+)?(\d{4})", re.I)
_PAT_B = re.compile(r"(\d{4}) hedged at approximately (\d{1,3})%", re.I)
_PAT_C = re.compile(r"approximately (\d{1,3})% and (\d{1,3})% for (\d{4}) and (\d{4})", re.I)


def parse_release(text: str) -> dict[int, float]:
    i = text.lower().find("hedged")
    out: dict[int, float] = {}
    while i != -1:
        seg = text[i:i + 400]
        for p1, p2, y1, y2 in _PAT_C.findall(seg):
            out.setdefault(int(y1), float(p1)); out.setdefault(int(y2), float(p2))
        for yr, pct in _PAT_B.findall(seg):
            out.setdefault(int(yr), float(pct))
        for m in _PAT_A.finditer(seg):
            if "on average" not in m.group(0).lower():  # multi-year averages are not single-year figures
                out.setdefault(int(m.group(2)), float(m.group(1)))
        i = text.lower().find("hedged", i + 1)
    # "average for the three-year period 2023 to 2025" is not a single-year figure
    return {y: p for y, p in out.items() if 2020 <= y <= 2030 and p <= 100}


def _region(ticker: str, filed: pd.Timestamp) -> dict[str, float]:
    if ticker == "TLN":
        return TLN_REGION
    return [w for d, w in VST_REGION if filed >= pd.Timestamp(d)][-1]


def build() -> pd.DataFrame:
    rows = []
    for tkr in ("VST", "TLN"):
        for f in sorted((CACHE / tkr).glob("*.txt")):
            hed = parse_release(f.read_text())
            if not hed:
                continue
            filed = pd.Timestamp(f.name[:10])
            for yr, pct in hed.items():
                for hub, w in _region(tkr, filed).items():
                    rows.append((tkr, filed, yr, hub, w * (1 - pct / 100), pct))
    for filed, years in _CEG_RAW.items():
        for yr, regs in years.items():
            tot = sum(g for g, _ in regs.values())
            for hub in ("PJ", "TX"):
                open_gwh = sum(g * max(0.0, 1 - min(h, 100) / 100) for k, (g, h) in regs.items() if _CEG_HUB[k] == hub)
                rows.append(("CEG", pd.Timestamp(filed), yr, hub, open_gwh / tot, float("nan")))
    return pd.DataFrame(rows, columns=["ticker", "filed", "year", "hub", "open_share", "hedge_pct"])


if __name__ == "__main__":
    df = build()
    pd.set_option("display.width", 200, "display.max_rows", 300)
    print(df[df.hub == "PJ"].pivot_table(index=["ticker", "filed"], columns="year", values="hedge_pct"))
    print(df.pivot_table(index=["ticker", "filed"], columns=["year", "hub"], values="open_share").round(3))
