"""
Utility → ISO node/load-zone mapping and named stress event calendar.

TICKER_NODE_MAP ties each stock ticker to the grid nodes/zones that best
capture its physical exposure.  Validate against 10-K service-territory
maps and FERC filings before using in production.

STRESS_EVENTS is the seed event calendar.  The signals layer adds
LMP-derived events on top.
"""

TICKER_NODE_MAP = {
    # ── ERCOT (Texas) ─────────────────────────────────────────────────────────
    "NRG": {
        "name": "NRG Energy",
        "iso": "ERCOT",
        "nodes": ["HB_NORTH", "HB_HOUSTON", "HB_SOUTH"],
        "load_zones": ["NORTH", "HOUSTON"],
        "service_territory": "Texas retail/gen + national gen portfolio",
        "notes": "Large ERCOT gen fleet; HB_NORTH and HB_HOUSTON most relevant.",
    },
    "VST": {
        "name": "Vistra Energy",
        "iso": "ERCOT",
        "nodes": ["HB_NORTH", "HB_WEST", "HB_SOUTH"],
        "load_zones": ["NORTH", "WEST", "SOUTH"],
        "service_territory": "Texas (primary) + IL, OH, PA gen assets",
        "notes": "Largest ERCOT generator by capacity. Comanche Peak nuclear near HB_NORTH.",
    },
    "CNP": {
        "name": "CenterPoint Energy",
        "iso": "ERCOT",
        "nodes": ["HB_HOUSTON", "LZ_HOUSTON"],
        "load_zones": ["HOUSTON"],
        "service_territory": "Houston metro T&D (wires-only, no generation)",
        "notes": "Pure wires play in ERCOT. Stress signal reflects distribution resilience.",
    },
    # ── PJM ───────────────────────────────────────────────────────────────────
    "AEP": {
        "name": "American Electric Power",
        "iso": "PJM",
        "nodes": ["AEP GEN HUB", "AEP-DAYTON HUB"],
        "load_zones": ["AEP", "DAYTON"],
        "secondary_iso": "SPP",
        "secondary_nodes": ["AEPC.WFEC", "SPS"],
        "service_territory": "OH, WV, TX, OK, LA, AR T&D + generation",
        "notes": "Spans PJM and SPP. AEP Texas is an ERCOT TDU.",
    },
    "EXC": {
        "name": "Exelon",
        "iso": "PJM",
        "nodes": ["PECO", "BGE", "PEPCO", "COMED"],
        "load_zones": ["PECO", "BGE", "PEPCO", "COMED"],
        "service_territory": "IL, PA, MD, DC, NJ, DE T&D",
        "notes": "Wires-focused after Constellation spinoff.",
    },
    "PPL": {
        "name": "PPL Corporation",
        "iso": "PJM",
        "nodes": ["PPL"],
        "load_zones": ["PPL"],
        "service_territory": "PA, KY T&D",
        "notes": "PPL zone in PJM. Kentucky ops in MISO East.",
    },
    "FE": {
        "name": "FirstEnergy",
        "iso": "PJM",
        "nodes": ["ATSI", "JCPL", "METED", "PENELEC"],
        "load_zones": ["ATSI", "JCPL"],
        "service_territory": "OH, PA, NJ, WV, MD T&D",
        "notes": "ATSI zone most material. Known seam congestion with MISO.",
    },
    # ── MISO ──────────────────────────────────────────────────────────────────
    "ETR": {
        "name": "Entergy",
        "iso": "MISO",
        "nodes": ["ARKANSAS HUB", "LOUISIANA HUB"],
        "load_zones": ["ENTERGY"],
        "service_territory": "AR, LA, MS, TX T&D + generation",
        "notes": "MISO South. High hurricane/tropical storm exposure.",
    },
    "WEC": {
        "name": "WEC Energy Group",
        "iso": "MISO",
        "nodes": ["MINNESOTA HUB", "ILLINOIS HUB"],
        "load_zones": ["CENTRAL"],
        "service_territory": "WI, IL, MI, MN T&D + generation",
        "notes": "MISO Central. Polar vortex events most material.",
    },
    "DTE": {
        "name": "DTE Energy",
        "iso": "MISO",
        "nodes": ["MICHIGAN HUB"],
        "load_zones": ["CENTRAL"],
        "service_territory": "Southeast Michigan T&D + generation",
        "notes": "MISO Central/East. Ice storms and polar vortex.",
    },
    "CMS": {
        "name": "CMS Energy / Consumers Energy",
        "iso": "MISO",
        "nodes": ["MICHIGAN HUB"],
        "load_zones": ["CENTRAL"],
        "service_territory": "Lower Michigan T&D + generation",
        "notes": "MISO Central. Similar stress profile to DTE.",
    },
    # ── CAISO ─────────────────────────────────────────────────────────────────
    "PCG": {
        "name": "PG&E",
        "iso": "CAISO",
        "nodes": ["TH_NP15_GEN-APND"],
        "load_zones": ["NP15"],
        "service_territory": "Northern/Central California T&D + generation",
        "notes": "CAISO NP15. Wildfire risk dominates. Use PSPS event days as stress dates.",
    },
    "EIX": {
        "name": "Edison International / SCE",
        "iso": "CAISO",
        "nodes": ["TH_SP15_GEN-APND"],
        "load_zones": ["SP15"],
        "service_territory": "Southern California T&D",
        "notes": "CAISO SP15. Sep 2022 heat wave most relevant. Also PSPS wildfire events.",
    },
    # ── SPP ───────────────────────────────────────────────────────────────────
    "XEL": {
        "name": "Xcel Energy",
        "iso": "SPP",
        "nodes": ["SPPNORTH_HUB", "SPPSOUTH_HUB"],
        "load_zones": ["NSP", "SPS"],
        "secondary_iso": "WECC",
        "service_territory": "MN, CO, TX, NM T&D + generation",
        "notes": "SPS zone (TX/NM) was stressed during Winter Storm Uri.",
    },
    # ── ISO-NE ────────────────────────────────────────────────────────────────
    "ES": {
        "name": "Eversource Energy",
        "iso": "ISO-NE",
        "nodes": [".Z.CONNECTICUT", ".Z.NEMASSBOST"],
        "load_zones": ["CT", "NEMA"],
        "service_territory": "CT, MA, NH T&D",
        "notes": "ISO-NE. Nor'easters and ice storms primary risk.",
    },
    # ── NYISO ─────────────────────────────────────────────────────────────────
    "ED": {
        "name": "Consolidated Edison",
        "iso": "NYISO",
        "nodes": ["N.Y.C."],
        "load_zones": ["NYC"],
        "service_territory": "NYC + Westchester T&D",
        "notes": "NYISO Zone J (NYC). LMP spikes sharply on heat/cold days.",
    },
    # ── PJM (additional) ──────────────────────────────────────────────────────
    "D": {
        "name": "Dominion Energy",
        "iso": "PJM",
        "nodes": ["EASTERN HUB"],
        "load_zones": ["DOM"],
        "service_territory": "Virginia, North Carolina T&D + generation",
        "notes": "PJM DOM zone. EASTERN HUB is the closest benchmark proxy. ~20% of capacity is in SERC-regulated NC territory.",
    },
    # ── MISO (additional) ─────────────────────────────────────────────────────
    "AEE": {
        "name": "Ameren",
        "iso": "MISO",
        "nodes": ["ILLINOIS HUB"],
        "load_zones": ["AMIL", "AMMO"],
        "service_territory": "Illinois and Missouri T&D + generation",
        "notes": "MISO Central. ILLINOIS HUB captures Illinois operations (larger segment). Missouri ops are on the MISO/SPP seam.",
    },
    # ── SPP (additional) ──────────────────────────────────────────────────────
    "EVRG": {
        "name": "Evergy",
        "iso": "SPP",
        "nodes": ["SPPNORTH_HUB", "SPPSOUTH_HUB"],
        "load_zones": ["KCPL", "WESTAR"],
        "service_territory": "Kansas and Missouri T&D + generation",
        "notes": "SPP. Kansas City Power & Light + Westar Energy merger. Similar Uri exposure to XEL.",
    },
    # ── SERC / FRCC (no centralized LMP — use EIA-417 outage data) ────────────
    "DUK": {
        "name": "Duke Energy",
        "iso": "SERC",
        "nodes": [],
        "load_zones": [],
        "service_territory": "NC, SC, FL, IN, OH T&D + generation",
        "secondary_iso": "MISO",
        "secondary_nodes": ["ILLINOIS HUB"],
        "notes": "~62% of capacity in SERC/FRCC (no LMP). Duke Energy Indiana is MISO; Duke Energy Ohio is PJM. Use EIA-417 for primary stress signal.",
    },
    "SO": {
        "name": "Southern Company",
        "iso": "SERC",
        "nodes": [],
        "load_zones": [],
        "service_territory": "GA, AL, MS, FL T&D + generation",
        "notes": "SERC does not publish granular nodal LMPs. Use EIA-417 outage data.",
    },
    "NEE": {
        "name": "NextEra Energy",
        "iso": "FRCC",
        "nodes": [],
        "load_zones": [],
        "service_territory": "Florida T&D + national renewables generation",
        "notes": "FRCC. Use hurricane track data as resilience signal.",
    },
}

# Tickers with actionable node data (excludes SERC/FRCC which lack LMP)
LMP_MAPPED_TICKERS = [t for t, v in TICKER_NODE_MAP.items() if v["nodes"]]

# Convenience: group tickers by primary ISO
TICKERS_BY_ISO: dict[str, list[str]] = {}
for _ticker, _meta in TICKER_NODE_MAP.items():
    _iso = _meta["iso"]
    TICKERS_BY_ISO.setdefault(_iso, []).append(_ticker)


STRESS_EVENTS = [
    {
        "name": "Northeast Polar Vortex",
        "type": "cold",
        "iso": ["PJM", "ISO-NE", "NYISO"],
        "window": ("2018-01-04", "2018-01-08"),
        "context": ("2017-12-01", "2018-02-28"),
        "notes": "ISO-NE and PJM set cold records. Gas supply disruptions caused gen outages.",
    },
    {
        "name": "PJM Polar Vortex",
        "type": "cold",
        "iso": ["PJM", "MISO"],
        "window": ("2019-01-29", "2019-02-01"),
        "context": ("2019-01-01", "2019-03-01"),
        "notes": "PJM declared Maximum Emergency Generation. Relevant for FE, EXC, WEC, DTE, CMS.",
    },
    {
        "name": "Winter Storm Uri",
        "type": "cold",
        "iso": ["ERCOT", "SPP", "MISO"],
        "window": ("2021-02-10", "2021-02-20"),
        "context": ("2021-01-15", "2021-03-15"),
        "notes": "ERCOT near-collapse. Massive LMP spikes. NRG, VST, CNP, AEP most affected.",
    },
    {
        "name": "Hurricane Ida",
        "type": "hurricane",
        "iso": ["MISO"],
        "window": ("2021-08-29", "2021-09-05"),
        "context": ("2021-07-01", "2021-10-01"),
        "notes": "Cat 4 landfall Louisiana. Entergy Louisiana ~900k outages.",
    },
    {
        "name": "ERCOT West Texas Congestion",
        "type": "congestion",
        "iso": ["ERCOT"],
        "window": ("2022-03-01", "2022-03-31"),
        "context": ("2021-01-01", "2023-12-31"),
        "notes": "Chronic west-to-east congestion from wind buildout exceeding transmission.",
    },
    {
        "name": "California Heat Dome",
        "type": "heat",
        "iso": ["CAISO"],
        "window": ("2022-09-05", "2022-09-09"),
        "context": ("2022-08-01", "2022-10-01"),
        "notes": "CAISO Flex Alert x7 days. SP15 LMPs >$2,000. Relevant for PCG, EIX.",
    },
    {
        "name": "ISO-NE Winter Storm Elliott",
        "type": "cold",
        "iso": ["ISO-NE", "PJM"],
        "window": ("2022-12-22", "2022-12-26"),
        "context": ("2022-11-01", "2023-03-01"),
        "notes": "Bomb cyclone. Gas supply stress. Relevant for ES.",
    },
    {
        "name": "ERCOT Heat Emergency Jun 2023",
        "type": "heat",
        "iso": ["ERCOT"],
        "window": ("2023-06-15", "2023-06-28"),
        "context": ("2023-05-01", "2023-08-01"),
        "notes": "Multiple ERCOT conservation notices. HB_NORTH spiked above $5,000/MWh.",
    },
]
