"""
Grid Resilience Factor Strategy
================================
Node-to-ticker mapping, LMP/congestion stress detection, and residual
return analysis for building a long/short equity signal.

SETUP:
    pip install yfinance pandas matplotlib scipy statsmodels requests

ERCOT LMP DATA:
    Free public API — no key required.
    Docs: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-188-CD
    We use the ERCOT public API v2: https://api.ercot.com/api/public-reports

PJM / MISO / SPP data sources are also noted — swap in as needed.
"""

import warnings

warnings.filterwarnings("ignore")

import os
import json
import requests
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats

try:
    import statsmodels.api as sm

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not found — OLS residuals will use numpy fallback.")

try:
    import yfinance as yf

    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print(
        "yfinance not found — equity data must be loaded from CSV (see EQUITY_CSV_PATH)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TICKER → NODE MAPPING
# ══════════════════════════════════════════════════════════════════════════════
#
# Each entry maps a stock ticker to:
#   iso      - the grid operator (ERCOT / PJM / MISO / SPP / WECC-CAISO)
#   nodes    - list of settlement point or hub names to monitor
#              Use hubs for broad exposure; load zones for geographic precision.
#   load_zones - ERCOT load zones (NORTH, SOUTH, WEST, HOUSTON) where relevant
#   service_territory - plain-English description for documentation
#
# Sources for node identification:
#   ERCOT: https://www.ercot.com/mktinfo/prices  (settlement point list)
#   PJM:   https://dataminer2.pjm.com/feed/da_hrl_lmps
#   MISO:  https://api.misoenergy.org/MISORTWDDataBroker/DataBrokerServices.asmx
#
# IMPORTANT: This mapping is a research starting point.
# Validate against each utility's 10-K service territory maps and FERC filings.
# Some utilities span multiple ISOs (e.g., AEP touches PJM and SPP).

TICKER_NODE_MAP = {
    # ── ERCOT (Texas) ────────────────────────────────────────────────────────
    "NRG": {
        "name": "NRG Energy",
        "iso": "ERCOT",
        "nodes": ["HB_NORTH", "HB_HOUSTON", "HB_SOUTH"],
        "load_zones": ["NORTH", "HOUSTON"],
        "service_territory": "Texas (retail/gen) + national gen portfolio",
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
        "notes": "Pure wires play in ERCOT. Stock performance on stress days reflects distribution resilience, not gen margin.",
    },
    "AEP": {
        "name": "American Electric Power",
        "iso": "PJM",  # primary — also SPP (AEP Texas, SWEPCO)
        "nodes": ["AEP-GEN", "AEPZ.WEST", "AEPZ.EAST"],  # PJM hubs/zones
        "load_zones": ["AEP"],
        "secondary_iso": "SPP",
        "secondary_nodes": ["AEPC.WFEC", "SPS"],  # SPP settlement points
        "service_territory": "OH, WV, TX, OK, LA, AR T&D + generation",
        "notes": "Spans PJM and SPP. Use PJM AEP zone for eastern ops; SPP SPS for Texas/Oklahoma.",
    },
    # ── PJM ──────────────────────────────────────────────────────────────────
    "SO": {
        "name": "Southern Company",
        "iso": "SERC",  # not centrally dispatched like PJM/ERCOT
        "nodes": [],  # SERC reliability region — no centralized LMP
        "load_zones": [],
        "service_territory": "GA, AL, MS, FL T&D + generation",
        "notes": "SERC does not publish granular nodal LMPs publicly. Use outage data (EIA-417) and storm day returns instead of LMP congestion signal.",
    },
    "EXC": {
        "name": "Exelon",
        "iso": "PJM",
        "nodes": ["PECO", "BGE", "PEPCO", "ComEd"],  # PJM load zones
        "load_zones": ["PECO", "BGE", "PEPCO", "COMED"],
        "service_territory": "IL, PA, MD, DC, NJ, DE T&D",
        "notes": "Wires-focused after Constellation spinoff. Use PJM zone LMPs for distribution stress signal.",
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
        "nodes": ["ATSI", "JCPL", "METED", "PENELEC", "PEPCO", "RECO"],
        "load_zones": ["ATSI", "JCPL"],
        "service_territory": "OH, PA, NJ, WV, MD T&D",
        "notes": "ATSI zone most material. Known transmission congestion issues on ATSI-MISO seam.",
    },
    "ETR": {
        "name": "Entergy",
        "iso": "MISO",
        "nodes": ["ENTERGY.GEN", "EAI", "EGSL", "ETI", "ELL"],  # MISO hubs
        "load_zones": ["ENTERGY"],
        "service_territory": "AR, LA, MS, TX T&D + generation",
        "notes": "MISO South footprint. High hurricane/tropical storm exposure. Use MISO LMP Hub_South.",
    },
    "ES": {
        "name": "Eversource Energy",
        "iso": "ISO-NE",
        "nodes": ["CT", "NEMA", "SEMA", "WCMA", "NH", "ME"],  # ISO-NE zones
        "load_zones": ["CT", "NEMA"],
        "service_territory": "CT, MA, NH T&D",
        "notes": "ISO-NE. New England grid frequently congested at NEMA-CT interface. High nor'easter/ice storm exposure.",
    },
    "ED": {
        "name": "Consolidated Edison",
        "iso": "NYISO",
        "nodes": ["NYC", "LONGIL", "MILLWD"],  # NYISO zones
        "load_zones": ["NYC"],
        "service_territory": "NYC + Westchester T&D",
        "notes": "NYISO Zone J (NYC). Extremely high load density; LMP spikes sharply on heat/cold days.",
    },
    "PCG": {
        "name": "PG&E",
        "iso": "CAISO",
        "nodes": ["PGAE_GEN-APND", "NP15_GEN-APND"],  # CAISO nodes
        "load_zones": ["NP15"],
        "service_territory": "Northern/Central California T&D + generation",
        "notes": "CAISO NP15 zone. Wildfire risk dominates stress events over weather congestion. Use PSPS event days as stress dates.",
    },
    "EIX": {
        "name": "Edison International / SCE",
        "iso": "CAISO",
        "nodes": ["SCE_GEN-APND", "SP15_GEN-APND"],
        "load_zones": ["SP15"],
        "service_territory": "Southern California T&D",
        "notes": "CAISO SP15. Heat wave stress (Sep 2022 event) most relevant. Also wildfire PSPS events.",
    },
    "XEL": {
        "name": "Xcel Energy",
        "iso": "SPP",  # Colorado is WECC; MN/TX is SPP/MISO
        "nodes": ["NSP", "SPS", "PSCO"],
        "load_zones": ["NSP", "SPS"],
        "secondary_iso": "WECC",
        "service_territory": "MN, CO, TX, NM T&D + generation",
        "notes": "Complex multi-ISO footprint. SPS (TX/NM) overlapped with Winter Storm Uri. Use SPP LMPs for southern ops.",
    },
    "WEC": {
        "name": "WEC Energy Group",
        "iso": "MISO",
        "nodes": ["WEC.GEN", "ALTE", "WPS"],
        "load_zones": ["CENTRAL"],
        "service_territory": "WI, IL, MI, MN T&D + generation",
        "notes": "MISO Central. Polar vortex events most material.",
    },
    "DTE": {
        "name": "DTE Energy",
        "iso": "MISO",
        "nodes": ["DTE.GEN", "DECO"],
        "load_zones": ["CENTRAL"],
        "service_territory": "Southeast Michigan T&D + generation",
        "notes": "MISO Central/East. Ice storms and polar vortex events. Known T&D reliability issues.",
    },
    "CMS": {
        "name": "CMS Energy / Consumers Energy",
        "iso": "MISO",
        "nodes": ["CONSUMERS.GEN", "MECS"],
        "load_zones": ["CENTRAL"],
        "service_territory": "Lower Michigan T&D + generation",
        "notes": "MISO Central. Similar stress profile to DTE.",
    },
    "AES": {
        "name": "AES Corporation",
        "iso": "PJM",  # Indiana subsidiary; also international
        "nodes": ["AEP", "ATSI"],
        "load_zones": [],
        "service_territory": "Indiana + international generation",
        "notes": "US operations mainly Indiana (MISO/PJM). International assets not captured by node mapping.",
    },
    "AWK": {
        "name": "American Water Works",
        "iso": None,
        "nodes": [],
        "load_zones": [],
        "service_territory": "Water utility — indirect grid exposure",
        "notes": "Not a generator or T&D owner. Exposure is as a large industrial electricity customer. Include as non-mapped peer for residual return analysis.",
    },
    "NEE": {
        "name": "NextEra Energy",
        "iso": "FRCC",  # FPL in Florida; also MISO/SPP/WECC for renewables
        "nodes": ["FPL.GEN"],
        "load_zones": [],
        "service_territory": "Florida T&D + national renewables generation",
        "notes": "FRCC (Florida) for regulated utility. FRCC does not publish granular LMPs. Use hurricane track data + storm prep disclosures as resilience signal.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  STRESS EVENT CALENDAR
# ══════════════════════════════════════════════════════════════════════════════
#
# Each event has:
#   type:   "weather" | "congestion" | "heat" | "cold" | "hurricane" | "wildfire"
#   iso:    affected grid operator(s)
#   window: (start, end) inclusive dates of the event core
#   context:(start, end) broader window for baseline / follow-through
#   notes:  what drove the stress

STRESS_EVENTS = [
    {
        "name": "Winter Storm Uri",
        "type": "cold",
        "iso": ["ERCOT", "SPP", "MISO"],
        "window": ("2021-02-10", "2021-02-20"),
        "context": ("2021-01-15", "2021-03-15"),
        "notes": "ERCOT near-collapse. Massive LMP spikes at all ERCOT hubs. SPP and MISO also stressed. NRG, VST, CNP, AEP most affected.",
    },
    {
        "name": "ERCOT Heat Emergency (Jun 2023)",
        "type": "heat",
        "iso": ["ERCOT"],
        "window": ("2023-06-15", "2023-06-28"),
        "context": ("2023-05-01", "2023-08-01"),
        "notes": "Multiple ERCOT conservation notices. LMPs at HB_NORTH spiked above $5,000/MWh on several afternoons.",
    },
    {
        "name": "California Heat Dome (Sep 2022)",
        "type": "heat",
        "iso": ["CAISO"],
        "window": ("2022-09-05", "2022-09-09"),
        "context": ("2022-08-01", "2022-10-01"),
        "notes": "CAISO issued Flex Alert for 7 consecutive days. SP15 LMPs hit $2,000+. Near-rolling-blackouts. Relevant for PCG, EIX.",
    },
    {
        "name": "Northeast Polar Vortex (Jan 2018)",
        "type": "cold",
        "iso": ["PJM", "ISO-NE", "NYISO"],
        "window": ("2018-01-04", "2018-01-08"),
        "context": ("2017-12-01", "2018-02-28"),
        "notes": "ISO-NE and PJM set cold weather records. Gas supply disruptions caused gen outages. Relevant for EXC, PPL, FE, ES, ED.",
    },
    {
        "name": "PJM Polar Vortex (Jan 2019)",
        "type": "cold",
        "iso": ["PJM", "MISO"],
        "window": ("2019-01-29", "2019-02-01"),
        "context": ("2019-01-01", "2019-03-01"),
        "notes": "Midwest extreme cold. PJM declared Maximum Emergency Generation. Relevant for FE, EXC, WEC, DTE, CMS.",
    },
    {
        "name": "Hurricane Ida (Aug-Sep 2021)",
        "type": "hurricane",
        "iso": ["MISO"],
        "window": ("2021-08-29", "2021-09-05"),
        "context": ("2021-07-01", "2021-10-01"),
        "notes": "Cat 4 landfall Louisiana. Entergy Louisiana had ~900k outages. Major test of ETR infrastructure resilience.",
    },
    {
        "name": "ERCOT West Texas Congestion (Ongoing)",
        "type": "congestion",
        "iso": ["ERCOT"],
        "window": ("2022-03-01", "2022-03-31"),  # representative month
        "context": ("2021-01-01", "2023-12-31"),
        "notes": "Chronic west-to-east congestion from wind buildout exceeding transmission. HB_WEST often trades at large discount to HB_NORTH. Relevant for VST, NRG (as gen owners).",
    },
    {
        "name": "ISO-NE Nor'easter Series (Winter 2022-23)",
        "type": "cold",
        "iso": ["ISO-NE"],
        "window": ("2022-12-22", "2022-12-26"),
        "context": ("2022-11-01", "2023-03-01"),
        "notes": "Winter Storm Elliott. Bomb cyclone affecting New England. Relevant for ES.",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LMP / CONGESTION DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════


class ERCOTLoader:
    """
    Fetch ERCOT Settlement Point Prices via the ERCOT Public API v2.
    No API key required for public data.

    Docs: https://developer.ercot.com/applications/pubapi/ERCOT%20Public%20API%20Registration%20and%20Authentication/
    Note: As of 2024, ERCOT requires a free registration token for API access.
          Register at: https://developer.ercot.com
          Then set: os.environ["ERCOT_API_TOKEN"] = "your_token"

    Alternatively, bulk historical data can be downloaded manually from:
    https://www.ercot.com/mp/data-products/data-product-details?id=NP4-188-CD
    """

    BASE_URL = (
        "https://api.ercot.com/api/public-reports/near-real-time/lmp_settlement_point"
    )
    HIST_URL = "https://api.ercot.com/api/public-reports/historical"

    # Hub settlement point names in ERCOT
    HUBS = {
        "HB_NORTH": "North Hub",
        "HB_SOUTH": "South Hub",
        "HB_WEST": "West Hub",
        "HB_HOUSTON": "Houston Hub",
    }

    # Load zone settlement point names
    LOAD_ZONES = {
        "LZ_NORTH": "North Load Zone",
        "LZ_SOUTH": "South Load Zone",
        "LZ_WEST": "West Load Zone",
        "LZ_HOUSTON": "Houston Load Zone",
    }

    def __init__(self):
        self.token = os.environ.get("ERCOT_API_TOKEN", None)
        if not self.token:
            print("  [ERCOT] No API token found. Set ERCOT_API_TOKEN env var.")
            print("  [ERCOT] Register free at: https://developer.ercot.com")

    def get_headers(self):
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def fetch_dam_lmp(self, settlement_point: str, date_str: str) -> pd.DataFrame:
        """
        Fetch Day-Ahead Market LMPs for a settlement point on a given date.
        Returns DataFrame with columns: [datetime, lmp, energy, congestion, loss]
        """
        url = f"{self.HIST_URL}/dam_stlmnt_pnt_prices"
        params = {
            "settlementPointName": settlement_point,
            "deliveryDateFrom": date_str,
            "deliveryDateTo": date_str,
            "size": 500,
        }
        try:
            r = requests.get(url, params=params, headers=self.get_headers(), timeout=30)
            r.raise_for_status()
            data = r.json()
            if "data" not in data or not data["data"]:
                return pd.DataFrame()
            df = pd.DataFrame(data["data"])
            df["datetime"] = pd.to_datetime(
                df["deliveryDate"]
                + " "
                + df["deliveryHour"].astype(str).str.zfill(2)
                + ":00"
            )
            df = df.rename(
                columns={
                    "settlementPointPrice": "lmp",
                    "lmpEnergyComponent": "energy",
                    "lmpCongestionComponent": "congestion",
                    "lmpLossComponent": "loss",
                }
            )
            return df[["datetime", "lmp", "energy", "congestion", "loss"]].sort_values(
                "datetime"
            )
        except Exception as e:
            print(f"  [ERCOT] Error fetching {settlement_point} on {date_str}: {e}")
            return pd.DataFrame()

    def fetch_lmp_range(
        self, settlement_point: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch daily DAM LMPs over a date range and concatenate."""
        dates = pd.date_range(start, end, freq="D")
        frames = []
        for d in dates:
            df = self.fetch_dam_lmp(settlement_point, d.strftime("%Y-%m-%d"))
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames).drop_duplicates("datetime").set_index("datetime")
        return result

    def daily_summary(
        self, settlement_point: str, start: str, end: str
    ) -> pd.DataFrame:
        """
        Returns daily summary stats: mean LMP, peak LMP, peak congestion,
        and a congestion flag (1 = day is above 90th percentile congestion).
        """
        lmp = self.fetch_lmp_range(settlement_point, start, end)
        if lmp.empty:
            return pd.DataFrame()
        daily = lmp.resample("D").agg(
            lmp_mean=("lmp", "mean"),
            lmp_max=("lmp", "max"),
            congestion_mean=("congestion", "mean"),
            congestion_max=("congestion", "max"),
            congestion_p90=("congestion", lambda x: x.quantile(0.9)),
        )
        # Flag days above rolling 90th percentile congestion
        rolling_p90 = daily["congestion_mean"].expanding().quantile(0.9)
        daily["congestion_stress"] = (daily["congestion_mean"] > rolling_p90).astype(
            int
        )
        daily["lmp_stress"] = (
            daily["lmp_mean"] > daily["lmp_mean"].expanding().quantile(0.9)
        ).astype(int)
        return daily


class PJMLoader:
    """
    Fetch PJM Day-Ahead LMPs via PJM DataMiner2 API.
    Free, no authentication required.

    Docs: https://dataminer2.pjm.com/feed/da_hrl_lmps/definition
    """

    BASE_URL = "https://api.pjm.com/api/v1"

    def fetch_da_lmp(self, pnode_id: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch hourly DA LMPs for a pnode over a date range.
        pnode_id: PJM pricing node ID (integer as string, or zone name)

        To find pnode IDs: https://dataminer2.pjm.com/feed/pnode/definition
        Common zone IDs:
            AEP:     8795689
            COMED:   33092371
            PPL:     1709329
            PEPCO:   2809705
            BGE:     12110093
            PECO:    10563783
            ATSI:    116013614
            DOM:     34964545  (Dominion/Vepco zone)
        """
        url = f"{self.BASE_URL}/da_hrl_lmps"
        params = {
            "pnode_id": pnode_id,
            "startRow": 1,
            "rowCount": 50000,
            "datetime_beginning_ept": f"{start} 00:00",
            "datetime_ending_ept": f"{end} 23:59",
            "fields": "datetime_beginning_ept,pnode_id,pnode_name,total_lmp_da,congestion_price_da,marginal_loss_price_da",
        }
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df["datetime"] = pd.to_datetime(df["datetime_beginning_ept"])
            df = df.rename(
                columns={
                    "total_lmp_da": "lmp",
                    "congestion_price_da": "congestion",
                    "marginal_loss_price_da": "loss",
                }
            )
            return df.set_index("datetime")[["lmp", "congestion", "loss"]].sort_index()
        except Exception as e:
            print(f"  [PJM] Error fetching pnode {pnode_id}: {e}")
            return pd.DataFrame()


class MISOLoader:
    """
    Fetch MISO Day-Ahead LMPs.
    Free public API.

    Docs: https://api.misoenergy.org/MISORTWDDataBroker/DataBrokerServices.asmx
    Bulk historical: https://www.misoenergy.org/markets-and-operations/real-time--market-data/market-reports/#nt=%2FMarketReportType%3ADay-Ahead%20LMP%2FMarketReportFolder%3ADay-Ahead%20LMP
    """

    BASE_URL = "https://api.misoenergy.org/MISORTWDDataBroker/DataBrokerServices.asmx"

    def fetch_da_lmp_csv(self, date_str: str) -> pd.DataFrame:
        """
        Fetch MISO DA LMP report for a date (YYYYMMDD format).
        Returns all nodes for that day.
        """
        date_fmt = date_str.replace("-", "")
        url = f"https://docs.misoenergy.org/marketreports/{date_fmt}_da_lmp.csv"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text), skiprows=4)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            return df
        except Exception as e:
            print(f"  [MISO] Error fetching {date_str}: {e}")
            return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EQUITY DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════


def load_equity_prices(
    tickers: list,
    start: str,
    end: str,
    csv_path: str = None,
) -> pd.DataFrame:
    """
    Load adjusted close prices. Tries yfinance first; falls back to CSV.

    CSV format expected:
        Date,NRG,VST,CNP,AEP,...
        2020-01-02,30.1,20.5,...
    """
    if HAS_YFINANCE:
        print(f"  Downloading equity prices for {tickers}...")
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            group_by="column",
            progress=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw.xs("Close", axis=1, level=0)
        else:
            prices = raw[["Close"]] if len(tickers) == 1 else raw
        return prices.dropna(how="all")

    elif csv_path and os.path.exists(csv_path):
        print(f"  Loading equity prices from {csv_path}...")
        prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        available = [t for t in tickers if t in prices.columns]
        return prices[available]

    else:
        print("  WARNING: No equity data source available.")
        print("  Either install yfinance or provide EQUITY_CSV_PATH.")
        return pd.DataFrame()


def compute_returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """Daily log or simple returns."""
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FACTOR STRIPPING — RESIDUAL RETURN COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


def compute_sector_factor(returns: pd.DataFrame) -> pd.Series:
    """Equal-weighted utility sector return (the industry factor to strip)."""
    return returns.mean(axis=1).rename("sector_factor")


def strip_factors(
    stock_returns: pd.Series,
    factor_df: pd.DataFrame,
    name: str = None,
) -> pd.Series:
    """
    Regress stock returns on a set of factors and return the residual (alpha).
    Uses statsmodels OLS if available, else numpy lstsq fallback.

    factor_df columns: each column is one factor (including sector).
    Returns: residual return series aligned to stock_returns index.
    """
    common = stock_returns.index.intersection(factor_df.index)
    y = stock_returns.loc[common]
    X = factor_df.loc[common]

    # Drop rows where either y or X has NaN
    mask = y.notna() & X.notna().all(axis=1)
    y, X = y[mask], X[mask]

    if len(y) < 30:
        print(f"  WARNING: Too few observations for {name} ({len(y)} days).")
        return pd.Series(dtype=float)

    if HAS_STATSMODELS:
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        residuals = model.resid
        r2 = model.rsquared
        print(
            f"  {name:<6}  R²={r2:.3f}  β_sector={model.params.get('sector_factor', np.nan):.3f}"
        )
    else:
        # Numpy fallback
        X_const = np.column_stack([np.ones(len(X)), X.values])
        coeffs, _, _, _ = np.linalg.lstsq(X_const, y.values, rcond=None)
        fitted = X_const @ coeffs
        residuals = pd.Series(y.values - fitted, index=y.index)

    return residuals.rename(f"{name}_resid" if name else "residual")


def build_residuals(
    returns: pd.DataFrame,
    extra_factors: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    For each ticker, strip the equal-weighted sector factor (and any extra
    factors supplied) and return a DataFrame of residual returns.
    """
    sector = compute_sector_factor(returns)
    factors = sector.to_frame()
    if extra_factors is not None:
        factors = factors.join(extra_factors, how="left")

    residuals = {}
    print("\nFactor stripping (OLS residuals):")
    for ticker in returns.columns:
        resid = strip_factors(returns[ticker], factors, name=ticker)
        if not resid.empty:
            residuals[ticker] = resid

    return pd.DataFrame(residuals)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  STRESS DAY IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════


def identify_stress_days(
    lmp_daily: pd.DataFrame,
    percentile: float = 90,
    method: str = "congestion",  # "congestion" | "lmp" | "either"
) -> pd.DatetimeIndex:
    """
    Given a daily LMP summary DataFrame (from ERCOTLoader.daily_summary),
    identify dates above the given percentile threshold.

    method:
        "congestion" — flag days where congestion_mean > pXX over full sample
        "lmp"        — flag days where lmp_mean > pXX
        "either"     — union of both
    """
    if lmp_daily.empty:
        return pd.DatetimeIndex([])

    p = percentile / 100
    congestion_thresh = lmp_daily["congestion_mean"].quantile(p)
    lmp_thresh = lmp_daily["lmp_mean"].quantile(p)

    if method == "congestion":
        mask = lmp_daily["congestion_mean"] > congestion_thresh
    elif method == "lmp":
        mask = lmp_daily["lmp_mean"] > lmp_thresh
    else:  # either
        mask = (lmp_daily["congestion_mean"] > congestion_thresh) | (
            lmp_daily["lmp_mean"] > lmp_thresh
        )

    return lmp_daily.index[mask]


def event_window_stress_days(event: dict) -> pd.DatetimeIndex:
    """Return the date range of a known stress event as a DatetimeIndex."""
    return pd.date_range(event["window"][0], event["window"][1], freq="D")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  RESILIENCE SIGNAL COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


def compute_resilience_signal(
    residuals: pd.DataFrame,
    stress_days: pd.DatetimeIndex,
    min_stress_days: int = 3,
) -> pd.Series:
    """
    Core signal: average residual return on stress days minus
    average residual return on non-stress days.

    Positive score = stock outperforms peers on stress days
                   → more resilient to grid fragility
    Negative score = stock underperforms on stress days
                   → more vulnerable

    Returns a Series indexed by ticker, sorted descending.
    """
    common_stress = residuals.index.intersection(stress_days)

    if len(common_stress) < min_stress_days:
        print(
            f"  WARNING: Only {len(common_stress)} stress days in residual return index."
        )
        print(f"  Consider expanding the date range or lowering min_stress_days.")

    non_stress = residuals.index.difference(stress_days)

    stress_mean = residuals.loc[common_stress].mean()
    non_stress_mean = residuals.loc[non_stress].mean()

    signal = (stress_mean - non_stress_mean).rename("resilience_signal")
    signal_zscore = ((signal - signal.mean()) / signal.std()).rename(
        "resilience_zscore"
    )

    result = pd.concat([signal, signal_zscore], axis=1)
    result["rank"] = result["resilience_signal"].rank(ascending=False).astype(int)
    return result.sort_values("resilience_signal", ascending=False)


def compute_event_returns(
    residuals: pd.DataFrame,
    event: dict,
) -> pd.DataFrame:
    """
    For a named event, compute cumulative residual return during the event
    window and the subsequent 10 trading days (follow-through).
    """
    w_start = pd.Timestamp(event["window"][0])
    w_end = pd.Timestamp(event["window"][1])
    follow_end = w_end + timedelta(days=14)

    event_resid = residuals.loc[w_start:w_end].sum()
    follow_resid = residuals.loc[w_end + timedelta(days=1) : follow_end].sum()

    result = pd.DataFrame(
        {
            "event_cumulative_resid": event_resid,
            "followthrough_resid": follow_resid,
            "total_resid": event_resid + follow_resid,
        }
    )
    result["event_rank"] = (
        result["event_cumulative_resid"].rank(ascending=False).astype(int)
    )
    return result.sort_values("event_cumulative_resid", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════


def build_long_short_portfolio(
    signal: pd.DataFrame,
    n_long: int = None,
    n_short: int = None,
    long_frac: float = 0.25,  # top quartile
    short_frac: float = 0.25,  # bottom quartile
    equal_weight: bool = True,
) -> pd.DataFrame:
    """
    Construct a long/short portfolio from the resilience signal.

    Returns a DataFrame with columns:
        ticker, signal, zscore, side (long/short/neutral), weight
    """
    n = len(signal)
    n_long = n_long or max(1, int(n * long_frac))
    n_short = n_short or max(1, int(n * short_frac))

    df = signal.copy().reset_index()
    df.columns = ["ticker"] + list(df.columns[1:])
    df = df.sort_values("resilience_signal", ascending=False).reset_index(drop=True)

    df["side"] = "neutral"
    df.loc[: n_long - 1, "side"] = "long"
    df.loc[n - n_short :, "side"] = "short"

    if equal_weight:
        df["weight"] = 0.0
        df.loc[df["side"] == "long", "weight"] = 1.0 / n_long
        df.loc[df["side"] == "short", "weight"] = -1.0 / n_short
    else:
        # Signal-weighted
        long_mask = df["side"] == "long"
        short_mask = df["side"] == "short"
        long_scores = df.loc[long_mask, "resilience_zscore"]
        short_scores = df.loc[short_mask, "resilience_zscore"]
        df.loc[long_mask, "weight"] = long_scores / long_scores.sum()
        df.loc[short_mask, "weight"] = -short_scores.abs() / short_scores.abs().sum()

    print(f"\n  Portfolio: {n_long} longs, {n_short} shorts")
    print(f"  LONGS:  {list(df.loc[df['side']=='long',  'ticker'])}")
    print(f"  SHORTS: {list(df.loc[df['side']=='short', 'ticker'])}")

    return df


def simulate_ls_returns(
    residuals: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> pd.Series:
    """
    Simulate daily P&L of the long/short portfolio using residual returns.
    Note: uses residuals (factor-stripped), not raw returns.
    """
    weights = portfolio.set_index("ticker")["weight"]
    common_tickers = residuals.columns.intersection(weights.index)
    w = weights[common_tickers]
    r = residuals[common_tickers]
    pnl = (r * w).sum(axis=1)
    pnl.name = "LS_portfolio"
    return pnl


# ══════════════════════════════════════════════════════════════════════════════
# 9.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

# Color scheme matching the presentation palette
COLORS = {
    "forest": "#1A4D2E",
    "moss": "#4A7C59",
    "sage": "#8FB996",
    "cream": "#F4F1E8",
    "gold": "#C9A84C",
    "dark": "#0D1F2D",
    "red": "#C05040",
    "light": "#E8EBE4",
    "long_fill": "#4A7C5960",
    "short_fill": "#C0504040",
}


def plot_uri_analysis(
    prices: pd.DataFrame,
    residuals: pd.DataFrame,
    event: dict,
    save_path: str = "uri_analysis.png",
):
    """
    Extended version of your original 2x2 chart, now with:
    - Panel 1: Raw normalized prices (your original chart)
    - Panel 2: Residual (factor-stripped) returns cumulative
    - Panel 3: Residual return distribution on event vs non-event days
    - Panel 4: Resilience signal bar chart
    """
    tickers = list(prices.columns)
    n = min(len(tickers), 4)
    tickers = tickers[:n]

    event_start = pd.Timestamp(event["window"][0])
    event_end = pd.Timestamp(event["window"][1])
    ctx_start = pd.Timestamp(event["context"][0])
    ctx_end = pd.Timestamp(event["context"][1])

    fig = plt.figure(figsize=(16, 12), facecolor=COLORS["cream"])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Normalized prices ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("white")

    prices_ctx = prices.loc[ctx_start:ctx_end]
    base = prices_ctx.ffill().bfill().iloc[0]
    norm = prices_ctx / base * 100

    palette = [COLORS["forest"], COLORS["moss"], COLORS["gold"], COLORS["red"]]
    for i, t in enumerate(tickers):
        if t in norm.columns:
            ax1.plot(norm.index, norm[t], label=t, color=palette[i], linewidth=1.8)

    ax1.axvspan(event_start, event_end, alpha=0.18, color=COLORS["red"], label="_event")
    ax1.axhline(100, color=COLORS["dark"], linewidth=0.5, linestyle="--", alpha=0.5)
    ax1.set_title(
        f"Normalized Prices\n{event['name']}",
        fontsize=11,
        color=COLORS["dark"],
        fontweight="bold",
    )
    ax1.set_ylabel("Base = 100")
    ax1.legend(fontsize=8, framealpha=0.7)
    ax1.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: Cumulative residual returns ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("white")

    if not residuals.empty:
        resid_ctx = residuals.loc[ctx_start:ctx_end]
        cum_resid = resid_ctx.cumsum()

        for i, t in enumerate(tickers):
            if t in cum_resid.columns:
                col = f"{t}_resid"
                if col in cum_resid.columns:
                    ax2.plot(
                        cum_resid.index,
                        cum_resid[col],
                        label=t,
                        color=palette[i],
                        linewidth=1.8,
                    )

        ax2.axvspan(event_start, event_end, alpha=0.18, color=COLORS["red"])
        ax2.axhline(0, color=COLORS["dark"], linewidth=0.5, linestyle="--", alpha=0.5)
        ax2.set_title(
            "Cumulative Residual Returns\n(Factor-Stripped)",
            fontsize=11,
            color=COLORS["dark"],
            fontweight="bold",
        )
        ax2.set_ylabel("Cum. Residual Return")
        ax2.legend(fontsize=8, framealpha=0.7)
        ax2.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax2.spines[["top", "right"]].set_visible(False)

    # ── Panel 3: Event vs non-event residual distribution ────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("white")

    if not residuals.empty:
        event_days = pd.date_range(event_start, event_end, freq="D")
        stress_cols = [
            f"{t}_resid" for t in tickers if f"{t}_resid" in residuals.columns
        ]

        for i, col in enumerate(stress_cols[:4]):
            t = col.replace("_resid", "")
            ev = residuals.loc[residuals.index.intersection(event_days), col].dropna()
            non_ev = residuals.loc[residuals.index.difference(event_days), col].dropna()

            pos = i * 0.4
            ax3.boxplot(
                [non_ev, ev],
                positions=[pos, pos + 0.18],
                widths=0.12,
                patch_artist=True,
                boxprops=dict(facecolor=palette[i], alpha=0.6),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color=palette[i]),
                capprops=dict(color=palette[i]),
                flierprops=dict(marker=".", color=palette[i], alpha=0.3),
            )

        ax3.axhline(0, color=COLORS["dark"], linewidth=0.5, linestyle="--", alpha=0.5)
        ax3.set_xticks([i * 0.4 + 0.09 for i in range(len(stress_cols[:4]))])
        ax3.set_xticklabels([c.replace("_resid", "") for c in stress_cols[:4]])
        ax3.set_title(
            "Residual Returns:\nNormal Days (left) vs Event Days (right)",
            fontsize=11,
            color=COLORS["dark"],
            fontweight="bold",
        )
        ax3.set_ylabel("Daily Residual Return")
        ax3.spines[["top", "right"]].set_visible(False)

    # ── Panel 4: Resilience signal bar chart ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("white")

    if not residuals.empty:
        all_stress = pd.date_range(event_start, event_end, freq="D")
        stress_cols = [
            f"{t}_resid" for t in tickers if f"{t}_resid" in residuals.columns
        ]

        stress_mean = residuals.loc[
            residuals.index.intersection(all_stress), stress_cols
        ].mean()
        non_stress_mean = residuals.loc[
            residuals.index.difference(all_stress), stress_cols
        ].mean()
        signal = (stress_mean - non_stress_mean).sort_values(ascending=True)
        labels = [s.replace("_resid", "") for s in signal.index]

        bar_colors = [
            COLORS["forest"] if v >= 0 else COLORS["red"] for v in signal.values
        ]
        ax4.barh(range(len(signal)), signal.values, color=bar_colors, alpha=0.85)
        ax4.set_yticks(range(len(signal)))
        ax4.set_yticklabels(labels)
        ax4.axvline(0, color=COLORS["dark"], linewidth=1)
        ax4.set_title(
            "Resilience Signal\n(Stress Day Residual - Normal Day Residual)",
            fontsize=11,
            color=COLORS["dark"],
            fontweight="bold",
        )
        ax4.set_xlabel("Signal Strength")

        long_patch = mpatches.Patch(
            color=COLORS["forest"], alpha=0.85, label="Long candidate"
        )
        short_patch = mpatches.Patch(
            color=COLORS["red"], alpha=0.85, label="Short candidate"
        )
        ax4.legend(handles=[long_patch, short_patch], fontsize=8, loc="lower right")
        ax4.spines[["top", "right"]].set_visible(False)

    event_label = f"{event['name']}\n{event['window'][0]} – {event['window'][1]}"
    fig.suptitle(
        f"Grid Resilience Factor Analysis\n{event_label}",
        fontsize=14,
        fontweight="bold",
        color=COLORS["dark"],
        y=0.98,
    )

    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=COLORS["cream"])
    print(f"\n  Chart saved: {save_path}")
    plt.show()


def plot_node_lmp(
    lmp_daily: pd.DataFrame, node: str, event: dict, save_path: str = None
):
    """Plot daily LMP and congestion for a node, with event window shaded."""
    if lmp_daily.empty:
        print("  No LMP data to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, facecolor=COLORS["cream"]
    )
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    event_start = pd.Timestamp(event["window"][0])
    event_end = pd.Timestamp(event["window"][1])

    ax1.plot(
        lmp_daily.index, lmp_daily["lmp_mean"], color=COLORS["forest"], linewidth=1.8
    )
    ax1.fill_between(
        lmp_daily.index, lmp_daily["lmp_mean"], alpha=0.15, color=COLORS["forest"]
    )
    ax1.axvspan(event_start, event_end, alpha=0.2, color=COLORS["red"])
    ax1.set_ylabel("DA LMP Mean ($/MWh)")
    ax1.set_title(
        f"{node} — Day-Ahead LMP", fontsize=11, fontweight="bold", color=COLORS["dark"]
    )
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.bar(
        lmp_daily.index,
        lmp_daily["congestion_mean"],
        color=COLORS["gold"],
        alpha=0.8,
        width=0.8,
    )
    ax2.axvspan(event_start, event_end, alpha=0.2, color=COLORS["red"])
    ax2.axhline(
        lmp_daily["congestion_mean"].quantile(0.9),
        color=COLORS["red"],
        linewidth=1,
        linestyle="--",
        label="90th percentile",
    )
    ax2.set_ylabel("Congestion Component ($/MWh)")
    ax2.set_title(
        "Congestion Component", fontsize=11, fontweight="bold", color=COLORS["dark"]
    )
    ax2.legend(fontsize=8)
    ax2.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Node LMP Analysis: {node}",
        fontsize=13,
        fontweight="bold",
        color=COLORS["dark"],
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=COLORS["cream"])
        print(f"  Node chart saved: {save_path}")
    plt.show()


def plot_resilience_scorecard(
    signal_df: pd.DataFrame, title: str = "", save_path: str = None
):
    """Summary scorecard chart of all tickers ranked by resilience signal."""
    df = signal_df.copy().reset_index()
    if "index" in df.columns:
        df = df.rename(columns={"index": "ticker"})
    df = df.sort_values("resilience_signal", ascending=True)

    fig, ax = plt.subplots(
        figsize=(10, max(4, len(df) * 0.45)), facecolor=COLORS["cream"]
    )
    ax.set_facecolor("white")

    bar_colors = [
        COLORS["forest"] if v >= 0 else COLORS["red"] for v in df["resilience_signal"]
    ]
    bars = ax.barh(
        df["ticker"], df["resilience_signal"], color=bar_colors, alpha=0.85, height=0.6
    )

    # Add z-score labels
    if "resilience_zscore" in df.columns:
        for bar, z in zip(bars, df["resilience_zscore"]):
            xpos = bar.get_width() + (0.0002 if bar.get_width() >= 0 else -0.0002)
            ax.text(
                xpos,
                bar.get_y() + bar.get_height() / 2,
                f"z={z:.2f}",
                va="center",
                ha="left" if bar.get_width() >= 0 else "right",
                fontsize=8,
                color=COLORS["dark"],
            )

    ax.axvline(0, color=COLORS["dark"], linewidth=1)
    ax.set_xlabel("Resilience Signal (stress day residual − normal day residual)")
    ax.set_title(
        f"Grid Resilience Scorecard\n{title}",
        fontsize=13,
        fontweight="bold",
        color=COLORS["dark"],
    )

    long_patch = mpatches.Patch(
        color=COLORS["forest"], alpha=0.85, label="Long candidates (resilient)"
    )
    short_patch = mpatches.Patch(
        color=COLORS["red"], alpha=0.85, label="Short candidates (fragile)"
    )
    ax.legend(handles=[long_patch, short_patch], fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=COLORS["cream"])
        print(f"  Scorecard saved: {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN — TIE IT ALL TOGETHER
# ══════════════════════════════════════════════════════════════════════════════

# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = ["NRG", "VST", "CNP", "AEP", "ETR", "EXC", "FE"]

START_DATE = "2020-01-01"
END_DATE = "2021-06-01"

# Path to a CSV of equity prices if yfinance is unavailable
EQUITY_CSV_PATH = None  # e.g., "equity_prices.csv"

# Which event to analyse in depth
FOCAL_EVENT = STRESS_EVENTS[0]  # Winter Storm Uri

# ERCOT node to pull LMP data for (requires ERCOT_API_TOKEN env var)
FOCAL_NODE = "HB_NORTH"

# Output directory
OUT_DIR = "."

# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 65)
    print("  GRID RESILIENCE FACTOR — Analysis Pipeline")
    print("=" * 65)

    # ── Step 1: Print node map for selected tickers ──────────────────────────
    print("\n[1] TICKER → NODE MAPPING")
    print("-" * 65)
    for t in TICKERS:
        if t in TICKER_NODE_MAP:
            m = TICKER_NODE_MAP[t]
            nodes_str = ", ".join(m.get("nodes", []) or ["(no central LMP)"])
            print(f"  {t:<6} | ISO: {m['iso']:<8} | Nodes: {nodes_str}")
            print(f"         | Territory: {m['service_territory']}")
            if m.get("notes"):
                print(f"         | Note: {m['notes']}")
        else:
            print(f"  {t:<6} | Not in TICKER_NODE_MAP — add manually.")
        print()

    # ── Step 2: Load equity prices ───────────────────────────────────────────
    print("[2] EQUITY DATA")
    print("-" * 65)
    prices = load_equity_prices(TICKERS, START_DATE, END_DATE, EQUITY_CSV_PATH)

    if prices.empty:
        print("  No equity data loaded. Aborting.")
        print("\n  TO RUN THIS SCRIPT:")
        print("    pip install yfinance")
        print("    python grid_resilience_analysis.py")
        return

    returns = compute_returns(prices)
    print(
        f"  Loaded {len(returns)} daily return observations for {len(returns.columns)} tickers."
    )
    print(f"  Date range: {returns.index[0].date()} → {returns.index[-1].date()}")

    # ── Step 3: Load LMP data (ERCOT) ────────────────────────────────────────
    print("\n[3] LMP / CONGESTION DATA")
    print("-" * 65)

    lmp_daily = pd.DataFrame()
    ercot = ERCOTLoader()

    if ercot.token:
        print(f"  Fetching {FOCAL_NODE} LMP from ERCOT API...")
        lmp_daily = ercot.daily_summary(FOCAL_NODE, START_DATE, END_DATE)
        if not lmp_daily.empty:
            print(f"  Loaded {len(lmp_daily)} days of LMP data.")
            print(f"  Max LMP: {lmp_daily['lmp_max'].max():.1f} $/MWh")
            print(
                f"  90th pct congestion: {lmp_daily['congestion_mean'].quantile(0.9):.2f} $/MWh"
            )
    else:
        print("  ERCOT API token not set — using known event dates as stress days.")
        print(
            "  (Set ERCOT_API_TOKEN environment variable to enable LMP-based stress detection.)"
        )

    # ── Step 4: Identify stress days ─────────────────────────────────────────
    print("\n[4] STRESS DAY IDENTIFICATION")
    print("-" * 65)

    if not lmp_daily.empty:
        # Use actual LMP/congestion data to identify stress
        lmp_stress_days = identify_stress_days(lmp_daily, 90, "lmp")
        cong_stress_days = identify_stress_days(lmp_daily, 90, "congestion")
        all_stress_days = identify_stress_days(lmp_daily, 90, "either")
        print(f"  LMP stress days (>p90 mean LMP):        {len(lmp_stress_days)}")
        print(f"  Congestion stress days (>p90 cong):     {len(cong_stress_days)}")
        print(f"  Combined (either):                      {len(all_stress_days)}")
        stress_days = all_stress_days
    else:
        # Fall back to known event calendar
        stress_days = event_window_stress_days(FOCAL_EVENT)
        print(f"  Using event calendar: {FOCAL_EVENT['name']}")
        print(
            f"  Stress window: {FOCAL_EVENT['window'][0]} → {FOCAL_EVENT['window'][1]}"
        )
        print(f"  {len(stress_days)} stress days")

    # ── Step 5: Factor stripping ─────────────────────────────────────────────
    print("\n[5] FACTOR STRIPPING")
    print("-" * 65)

    # Optional: add LMP factor as an additional regressor
    # This strips out the mechanical effect of commodity prices on all utilities
    extra_factors = None
    if not lmp_daily.empty:
        lmp_factor = lmp_daily["lmp_mean"].rename("lmp_factor")
        extra_factors = lmp_factor.to_frame()

    residuals = build_residuals(returns, extra_factors)

    # ── Step 6: Compute resilience signals ───────────────────────────────────
    print("\n[6] RESILIENCE SIGNAL")
    print("-" * 65)

    signal = compute_resilience_signal(residuals, stress_days)
    print("\n  Resilience Scorecard:")
    print(f"  {'Ticker':<10} {'Signal':>10} {'Z-Score':>10} {'Rank':>6}")
    print("  " + "-" * 40)
    for _, row in signal.iterrows():
        side = "LONG " if row["resilience_signal"] > 0 else "SHORT"
        print(
            f"  {row.name:<10} {row['resilience_signal']:>10.5f} "
            f"{row['resilience_zscore']:>10.3f} {int(row['rank']):>5}  ({side})"
        )

    # ── Step 7: Portfolio construction ───────────────────────────────────────
    print("\n[7] PORTFOLIO CONSTRUCTION")
    print("-" * 65)

    portfolio = build_long_short_portfolio(signal, long_frac=0.33, short_frac=0.33)

    # ── Step 8: Event analysis ───────────────────────────────────────────────
    print("\n[8] EVENT ANALYSIS")
    print("-" * 65)

    event_rets = compute_event_returns(residuals, FOCAL_EVENT)
    print(f"\n  Cumulative residual returns during {FOCAL_EVENT['name']}:")
    print(f"  {'Ticker':<10} {'Event':>10} {'Follow-thru':>12} {'Total':>10}")
    print("  " + "-" * 46)
    for _, row in event_rets.iterrows():
        print(
            f"  {row.name:<10} {row['event_cumulative_resid']:>10.4f} "
            f"{row['followthrough_resid']:>12.4f} {row['total_resid']:>10.4f}"
        )

    # ── Step 9: Portfolio P&L ────────────────────────────────────────────────
    print("\n[9] PORTFOLIO SIMULATION")
    print("-" * 65)

    pnl = simulate_ls_returns(residuals, portfolio)
    cum_pnl = pnl.cumsum()

    # Summary stats
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else np.nan
    print(f"\n  Annualised Sharpe (residual-based, approx): {sharpe:.3f}")
    print(f"  Cumulative PnL over period: {cum_pnl.iloc[-1]:.4f}")
    print(f"  Max daily gain:  {pnl.max():.4f}")
    print(f"  Max daily loss:  {pnl.min():.4f}")

    # ── Step 10: Visualisations ──────────────────────────────────────────────
    print("\n[10] GENERATING CHARTS")
    print("-" * 65)

    plot_uri_analysis(
        prices=prices,
        residuals=residuals,
        event=FOCAL_EVENT,
        save_path=os.path.join(OUT_DIR, "uri_resilience_analysis.png"),
    )

    plot_resilience_scorecard(
        signal_df=signal,
        title=f"Derived from {FOCAL_EVENT['name']} + Extended History",
        save_path=os.path.join(OUT_DIR, "resilience_scorecard.png"),
    )

    if not lmp_daily.empty:
        plot_node_lmp(
            lmp_daily=lmp_daily,
            node=FOCAL_NODE,
            event=FOCAL_EVENT,
            save_path=os.path.join(OUT_DIR, f"lmp_{FOCAL_NODE}.png"),
        )

    print("\n✓  Pipeline complete.")
    print(f"   Charts saved to: {os.path.abspath(OUT_DIR)}")

    return {
        "prices": prices,
        "returns": returns,
        "residuals": residuals,
        "signal": signal,
        "portfolio": portfolio,
        "pnl": pnl,
        "lmp_daily": lmp_daily,
    }


if __name__ == "__main__":
    results = main()
