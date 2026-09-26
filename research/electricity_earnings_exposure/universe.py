"""
PROBE B universe — power-intensive industrials with a hand-coded electricity-cost
exposure vector.

exposure : {state: weight}  weights sum ~1; the states where the company's
           power-consuming plants sit (10-K Item 2). EIA industrial retail price
           per state is the price proxy.
intensity: electricity as an approx fraction of cash operating cost (COGS-ish).
sector   : bucket for sector-neutralization.

Approximate. Sources: 10-K plant tables, company fact sheets. Precise enough for a
cross-sectional rank signal; NOT a fundamentals model.
"""

UNIVERSE = [
    # ---- primary aluminum (very power-intensive) ----
    dict(ticker="AA",   cik="0001675149", sector="aluminum",
         intensity=0.32, exposure={"IN": .5, "NY": .3, "WA": .2}),
    dict(ticker="CENX", cik="0000949157", sector="aluminum",
         intensity=0.38, exposure={"KY": .6, "SC": .4}),
    dict(ticker="KALU", cik="0000811596", sector="aluminum",
         intensity=0.06, exposure={"WA": .25, "OH": .2, "IN": .2, "MI": .15, "TX": .1, "VA": .1}),
    # ---- EAF steel ----
    dict(ticker="NUE",  cik="0000073309", sector="steel",
         intensity=0.10, exposure={"TX": .18, "SC": .15, "AL": .12, "IN": .12, "NC": .1,
                                   "AR": .1, "KY": .08, "NE": .08, "UT": .07}),
    dict(ticker="STLD", cik="0001022671", sector="steel",
         intensity=0.10, exposure={"IN": .45, "TX": .3, "MS": .15, "VA": .1}),
    dict(ticker="CMC",  cik="0000022444", sector="steel",
         intensity=0.09, exposure={"TX": .4, "AZ": .2, "AL": .15, "SC": .15, "OK": .1}),
    dict(ticker="ATI",  cik="0001018963", sector="steel",
         intensity=0.12, exposure={"PA": .55, "NC": .15, "OR": .15, "NY": .15}),
    # ---- chlor-alkali / commodity chem ----
    dict(ticker="OLN",  cik="0000074303", sector="chem",
         intensity=0.28, exposure={"LA": .4, "TX": .25, "AL": .15, "TN": .1, "NY": .1}),
    dict(ticker="WLK",  cik="0001262823", sector="chem",
         intensity=0.12, exposure={"LA": .45, "TX": .25, "KY": .2, "OH": .1}),
    # ---- industrial gases (air separation ~ pure electricity) ----
    dict(ticker="APD",  cik="0000002969", sector="gas",
         intensity=0.40, exposure={"TX": .3, "LA": .2, "CA": .1, "PA": .1, "IN": .1,
                                   "OH": .1, "AL": .1}),
    dict(ticker="LIN",  cik="0001707925", sector="gas",
         intensity=0.40, exposure={"TX": .25, "LA": .25, "CA": .1, "PA": .1, "IN": .1,
                                   "OH": .1, "GA": .1}),
    # ---- bitcoin / AI-hosting miners (power = dominant opex; ERCOT-heavy) ----
    dict(ticker="MARA", cik="0001507605", sector="crypto",
         intensity=0.60, exposure={"TX": .55, "ND": .2, "OH": .15, "NE": .1}),
    dict(ticker="RIOT", cik="0001167419", sector="crypto",
         intensity=0.60, exposure={"TX": .85, "KY": .15}),
    dict(ticker="CLSK", cik="0000827876", sector="crypto",
         intensity=0.60, exposure={"GA": .5, "MS": .2, "TN": .2, "WY": .1}),
    dict(ticker="WULF", cik="0001083301", sector="crypto",
         intensity=0.60, exposure={"NY": .8, "PA": .2}),
    dict(ticker="CIFR", cik="0001819989", sector="crypto",
         intensity=0.60, exposure={"TX": 1.0}),
    dict(ticker="IREN", cik="0001878848", sector="crypto",
         intensity=0.55, exposure={"TX": 1.0}),
    dict(ticker="BTDR", cik="0001899123", sector="crypto",
         intensity=0.55, exposure={"TX": .6, "WA": .4}),
]

ALL_STATES = sorted({s for u in UNIVERSE for s in u["exposure"]})
