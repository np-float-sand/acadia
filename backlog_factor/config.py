from __future__ import annotations

"""Central configuration for the backlog growth-surprise factor.

Spec: docs/superpowers/specs/2026-08-31-backlog-surprise-factor-design.md
"""

from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}

# Discovery: quarters (instant-period frame keys) to union RPO filers from.
DISCOVERY_QUARTERS: list[str] = ["CY2021Q4I", "CY2022Q4I", "CY2023Q4I", "CY2024Q4I", "CY2025Q4I"]

# SIC prefixes for order-driven manufacturers / contractors (industrials & capital goods).
# Deliberately excludes 357x (computers / office equipment), 365x-367x (consumer
# audio/video, communications, electronic components) -- a coarse pre-cut before the
# Step-6 hand-curation, not the final filter.
IN_SCOPE_SIC_PREFIXES: tuple[str, ...] = (
    "15", "16", "17",                             # construction / heavy construction / special trade
    "34",                                         # fabricated metal products
    "351", "352", "353", "354", "355", "356",     # machinery: engines, farm, construction, metalworking,
    "358", "359",                                 #   special/general industrial, refrigeration, misc
    "361", "362", "364", "369",                   # electrical: transmission & distribution apparatus,
                                                  #   industrial apparatus, lighting, misc electrical
    "37",                                         # transportation equipment (aerospace, rail, defense, autos)
    "382", "384",                                 # industrial measurement/control instruments; med/surgical
)

STRUCTURED_START: str = "2019-01-01"
HOLDOUT_MONTHS: int = 18

DRIFT_HORIZON_DAYS: int = 63       # provisional; RESET from the event study's peak window
CAR_WINDOWS: list[int] = [5, 21, 42, 63]
QUINTILE_Q: int = 5
WINSOR_SIGMA: float = 3.0
COST_BPS: float = 10.0

HISTORY_MIN_QUARTERS: int = 6
SPAN_MIN_DAYS: int = 300
SPAN_MAX_DAYS: int = 430
STALENESS_MAX_DAYS: int = 200

RISK_FREE_RATE: float = 0.04
ANN_FACTOR: int = 252

# Curated from the 207 XBRL-RPO in-scope filers discovered via the SEC frames API
# (see candidate_research.md for every keep/drop decision). 109 names.
UNIVERSE: list[dict] = [
    {"ticker": "AIR", "cik": "0000001750", "industry_group": "aerospace_defense", "name": "AAR CORP"},
    {"ticker": "AME", "cik": "0001037868", "industry_group": "machinery", "name": "AMETEK INC/"},
    {"ticker": "AMRC", "cik": "0001488139", "industry_group": "engineering_construction", "name": "Ameresco, Inc."},
    {"ticker": "ASYS", "cik": "0000720500", "industry_group": "semiconductor_equipment", "name": "AMTECH SYSTEMS INC"},
    {"ticker": "ATRO", "cik": "0000008063", "industry_group": "aerospace_defense", "name": "ASTRONICS CORP"},
    {"ticker": "AVAV", "cik": "0001368622", "industry_group": "aerospace_defense", "name": "AeroVironment Inc"},
    {"ticker": "AZTA", "cik": "0000933974", "industry_group": "semiconductor_equipment", "name": "Azenta, Inc."},
    {"ticker": "BA", "cik": "0000012927", "industry_group": "aerospace_defense", "name": "BOEING CO"},
    {"ticker": "BKR", "cik": "0001701605", "industry_group": "machinery", "name": "Baker Hughes Co"},
    {"ticker": "BMI", "cik": "0000009092", "industry_group": "electrical_equipment", "name": "BADGER METER INC"},
    {"ticker": "BW", "cik": "0001630805", "industry_group": "machinery", "name": "Babcock & Wilcox Enterprises, Inc."},
    {"ticker": "BWXT", "cik": "0001486957", "industry_group": "aerospace_defense", "name": "BWX Technologies, Inc."},
    {"ticker": "CAT", "cik": "0000018230", "industry_group": "machinery", "name": "CATERPILLAR INC"},
    {"ticker": "CDRE", "cik": "0001860543", "industry_group": "aerospace_defense", "name": "Cadre Holdings, Inc."},
    {"ticker": "CMCO", "cik": "0001005229", "industry_group": "machinery", "name": "COLUMBUS MCKINNON CORP"},
    {"ticker": "CMI", "cik": "0000026172", "industry_group": "machinery", "name": "CUMMINS INC"},
    {"ticker": "CNH", "cik": "0001567094", "industry_group": "machinery", "name": "CNH Industrial N.V."},
    {"ticker": "CR", "cik": "0001944013", "industry_group": "machinery", "name": "Crane Co"},
    {"ticker": "CVU", "cik": "0000889348", "industry_group": "aerospace_defense", "name": "CPI AEROSTRUCTURES INC"},
    {"ticker": "CVV", "cik": "0000766792", "industry_group": "semiconductor_equipment", "name": "CVD EQUIPMENT CORP"},
    {"ticker": "CW", "cik": "0000026324", "industry_group": "aerospace_defense", "name": "CURTISS WRIGHT CORP"},
    {"ticker": "DCO", "cik": "0000030305", "industry_group": "aerospace_defense", "name": "DUCOMMUN INC /DE/"},
    {"ticker": "DE", "cik": "0000315189", "industry_group": "machinery", "name": "DEERE & CO"},
    {"ticker": "DOV", "cik": "0000029905", "industry_group": "machinery", "name": "DOVER Corp"},
    {"ticker": "ECG", "cik": "0002015845", "industry_group": "engineering_construction", "name": "Everus Construction Group, Inc."},
    {"ticker": "EME", "cik": "0000105634", "industry_group": "engineering_construction", "name": "EMCOR Group, Inc."},
    {"ticker": "ENS", "cik": "0001289308", "industry_group": "electrical_equipment", "name": "EnerSys"},
    {"ticker": "EOSE", "cik": "0001805077", "industry_group": "electrical_equipment", "name": "Eos Energy Enterprises, Inc."},
    {"ticker": "EPAC", "cik": "0000006955", "industry_group": "machinery", "name": "ENERPAC TOOL GROUP CORP"},
    {"ticker": "ERII", "cik": "0001421517", "industry_group": "machinery", "name": "Energy Recovery, Inc."},
    {"ticker": "ESLT", "cik": "0001027664", "industry_group": "aerospace_defense", "name": "ELBIT SYSTEMS LTD"},
    {"ticker": "ESOA", "cik": "0001357971", "industry_group": "engineering_construction", "name": "Energy Services of America CORP"},
    {"ticker": "ETN", "cik": "0001551182", "industry_group": "electrical_equipment", "name": "Eaton Corp plc"},
    {"ticker": "FLNC", "cik": "0001868941", "industry_group": "electrical_equipment", "name": "Fluence Energy, Inc."},
    {"ticker": "FLR", "cik": "0001124198", "industry_group": "engineering_construction", "name": "FLUOR CORP"},
    {"ticker": "FLS", "cik": "0000030625", "industry_group": "machinery", "name": "FLOWSERVE CORP"},
    {"ticker": "FTEK", "cik": "0000846913", "industry_group": "machinery", "name": "FUEL TECH, INC."},
    {"ticker": "FTI", "cik": "0001681459", "industry_group": "machinery", "name": "TechnipFMC plc"},
    {"ticker": "FTV", "cik": "0001659166", "industry_group": "machinery", "name": "Fortive Corp"},
    {"ticker": "GBX", "cik": "0000923120", "industry_group": "machinery", "name": "GREENBRIER COMPANIES INC"},
    {"ticker": "GD", "cik": "0000040533", "industry_group": "aerospace_defense", "name": "GENERAL DYNAMICS CORP"},
    {"ticker": "GEHC", "cik": "0001932393", "industry_group": "machinery", "name": "GE HealthCare Technologies Inc."},
    {"ticker": "GEOS", "cik": "0001001115", "industry_group": "machinery", "name": "GEOSPACE TECHNOLOGIES CORP"},
    {"ticker": "GHM", "cik": "0000716314", "industry_group": "machinery", "name": "GRAHAM CORP"},
    {"ticker": "GNRC", "cik": "0001474735", "industry_group": "electrical_equipment", "name": "GENERAC HOLDINGS INC."},
    {"ticker": "GVA", "cik": "0000861459", "industry_group": "engineering_construction", "name": "GRANITE CONSTRUCTION INC"},
    {"ticker": "HII", "cik": "0001501585", "industry_group": "aerospace_defense", "name": "HUNTINGTON INGALLS INDUSTRIES, INC."},
    {"ticker": "HON", "cik": "0000773840", "industry_group": "aerospace_defense", "name": "HONEYWELL INTERNATIONAL INC"},
    {"ticker": "IBP", "cik": "0001580905", "industry_group": "building_products", "name": "Installed Building Products, Inc."},
    {"ticker": "IESC", "cik": "0001048268", "industry_group": "engineering_construction", "name": "IES Holdings, Inc."},
    {"ticker": "INVX", "cik": "0001042893", "industry_group": "machinery", "name": "Innovex International, Inc."},
    {"ticker": "ITRI", "cik": "0000780571", "industry_group": "electrical_equipment", "name": "ITRON, INC."},
    {"ticker": "ITT", "cik": "0000216228", "industry_group": "machinery", "name": "ITT INC."},
    {"ticker": "J", "cik": "0000052988", "industry_group": "engineering_construction", "name": "JACOBS SOLUTIONS INC."},
    {"ticker": "JBI", "cik": "0001839839", "industry_group": "building_products", "name": "Janus International Group, Inc."},
    {"ticker": "JBTM", "cik": "0001433660", "industry_group": "machinery", "name": "JBT MAREL Corp"},
    {"ticker": "JCI", "cik": "0000833444", "industry_group": "machinery", "name": "Johnson Controls International plc"},
    {"ticker": "KAI", "cik": "0000886346", "industry_group": "machinery", "name": "KADANT INC"},
    {"ticker": "KBR", "cik": "0001357615", "industry_group": "engineering_construction", "name": "KBR, INC."},
    {"ticker": "KEYS", "cik": "0001601046", "industry_group": "semiconductor_equipment", "name": "Keysight Technologies, Inc."},
    {"ticker": "KLAC", "cik": "0000319201", "industry_group": "semiconductor_equipment", "name": "KLA CORP"},
    {"ticker": "KRMN", "cik": "0002040127", "industry_group": "aerospace_defense", "name": "Karman Holdings Inc."},
    {"ticker": "KRNT", "cik": "0001625791", "industry_group": "machinery", "name": "Kornit Digital Ltd."},
    {"ticker": "KTOS", "cik": "0001069258", "industry_group": "aerospace_defense", "name": "KRATOS DEFENSE & SECURITY SOLUTIONS, INC."},
    {"ticker": "LGN", "cik": "0002052568", "industry_group": "engineering_construction", "name": "Legence Corp."},
    {"ticker": "LMT", "cik": "0000936468", "industry_group": "aerospace_defense", "name": "LOCKHEED MARTIN CORP"},
    {"ticker": "LRCX", "cik": "0000707549", "industry_group": "semiconductor_equipment", "name": "LAM RESEARCH CORP"},
    {"ticker": "MOG-A", "cik": "0000067887", "industry_group": "aerospace_defense", "name": "MOOG INC."},
    {"ticker": "MTRN", "cik": "0001104657", "industry_group": "machinery", "name": "MATERION Corp"},
    {"ticker": "MTRX", "cik": "0000866273", "industry_group": "engineering_construction", "name": "MATRIX SERVICE CO"},
    {"ticker": "MTZ", "cik": "0000015615", "industry_group": "engineering_construction", "name": "MASTEC INC"},
    {"ticker": "MYRG", "cik": "0000700923", "industry_group": "engineering_construction", "name": "MYR GROUP INC."},
    {"ticker": "NOV", "cik": "0001021860", "industry_group": "machinery", "name": "NOV Inc."},
    {"ticker": "ORN", "cik": "0001402829", "industry_group": "engineering_construction", "name": "Orion Group Holdings Inc"},
    {"ticker": "OSK", "cik": "0000775158", "industry_group": "machinery", "name": "OSHKOSH CORP"},
    {"ticker": "PH", "cik": "0000076334", "industry_group": "machinery", "name": "Parker-Hannifin Corp"},
    {"ticker": "POWL", "cik": "0000080420", "industry_group": "electrical_equipment", "name": "POWELL INDUSTRIES INC"},
    {"ticker": "PRIM", "cik": "0001361538", "industry_group": "engineering_construction", "name": "Primoris Services Corp"},
    {"ticker": "PSIX", "cik": "0001137091", "industry_group": "machinery", "name": "POWER SOLUTIONS INTERNATIONAL, INC."},
    {"ticker": "PWR", "cik": "0001050915", "industry_group": "engineering_construction", "name": "QUANTA SERVICES, INC."},
    {"ticker": "RAIL", "cik": "0001320854", "industry_group": "machinery", "name": "FreightCar America, Inc."},
    {"ticker": "RAL", "cik": "0002041385", "industry_group": "machinery", "name": "Ralliant Corp"},
    {"ticker": "RBC", "cik": "0001324948", "industry_group": "machinery", "name": "RBC Bearings INC"},
    {"ticker": "RGR", "cik": "0000095029", "industry_group": "machinery", "name": "STURM RUGER & CO INC"},
    {"ticker": "RKLB", "cik": "0001819994", "industry_group": "aerospace_defense", "name": "Rocket Lab Corp"},
    {"ticker": "ROAD", "cik": "0001718227", "industry_group": "engineering_construction", "name": "Construction Partners, Inc."},
    {"ticker": "ROK", "cik": "0001024478", "industry_group": "electrical_equipment", "name": "ROCKWELL AUTOMATION, INC"},
    {"ticker": "RRX", "cik": "0000082811", "industry_group": "electrical_equipment", "name": "REGAL REXNORD CORP"},
    {"ticker": "RTX", "cik": "0000101829", "industry_group": "aerospace_defense", "name": "RTX Corp"},
    {"ticker": "SHIM", "cik": "0001887944", "industry_group": "engineering_construction", "name": "Shimmick Corp"},
    {"ticker": "SLND", "cik": "0001883814", "industry_group": "engineering_construction", "name": "Southland Holdings, Inc."},
    {"ticker": "SPXC", "cik": "0000088205", "industry_group": "machinery", "name": "SPX Technologies, Inc."},
    {"ticker": "STRL", "cik": "0000874238", "industry_group": "engineering_construction", "name": "STERLING INFRASTRUCTURE, INC."},
    {"ticker": "SYM", "cik": "0001837240", "industry_group": "machinery", "name": "Symbotic Inc."},
    {"ticker": "TER", "cik": "0000097210", "industry_group": "semiconductor_equipment", "name": "TERADYNE, INC"},
    {"ticker": "TEX", "cik": "0000097216", "industry_group": "machinery", "name": "TEREX CORP"},
    {"ticker": "TGEN", "cik": "0001537435", "industry_group": "machinery", "name": "TECOGEN INC."},
    {"ticker": "TKR", "cik": "0000098362", "industry_group": "machinery", "name": "TIMKEN CO"},
    {"ticker": "TNC", "cik": "0000097134", "industry_group": "machinery", "name": "TENNANT CO"},
    {"ticker": "TRMB", "cik": "0000864749", "industry_group": "machinery", "name": "TRIMBLE INC."},
    {"ticker": "TXT", "cik": "0000217346", "industry_group": "aerospace_defense", "name": "TEXTRON INC"},
    {"ticker": "VECO", "cik": "0000103145", "industry_group": "semiconductor_equipment", "name": "VEECO INSTRUMENTS INC"},
    {"ticker": "VLTO", "cik": "0001967680", "industry_group": "machinery", "name": "Veralto Corp"},
    {"ticker": "VMI", "cik": "0000102729", "industry_group": "electrical_equipment", "name": "VALMONT INDUSTRIES INC"},
    {"ticker": "VNT", "cik": "0001786842", "industry_group": "machinery", "name": "Vontier Corp"},
    {"ticker": "WAB", "cik": "0000943452", "industry_group": "machinery", "name": "WESTINGHOUSE AIR BRAKE TECHNOLOGIES CORP"},
    {"ticker": "WWD", "cik": "0000108312", "industry_group": "aerospace_defense", "name": "Woodward, Inc."},
    {"ticker": "XYL", "cik": "0001524472", "industry_group": "machinery", "name": "Xylem Inc."},
    {"ticker": "ZWS", "cik": "0001439288", "industry_group": "building_products", "name": "Zurn Elkay Water Solutions Corp"},
]
                                   #   "industry_group": "engineering_construction", "name": "Quanta Services"}]

# Hand-curated: industry group -> a liquid ETF proxy for abnormal-return benchmarking.
INDUSTRY_GROUP_ETF: dict[str, str] = {
    "machinery": "XLI",
    "electrical_equipment": "XLI",
    "engineering_construction": "PAVE",
    "aerospace_defense": "ITA",
    "building_products": "XHB",
    "semiconductor_equipment": "SOXX",
}
