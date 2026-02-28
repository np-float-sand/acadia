import requests

ERCOT_API_KEY = "423b18a2aa4b4e658a41fe3c1da2c0cc"


"""
ercot_battery_nodes.py
======================
Fetches a list of ERCOT resource nodes that are batteries (Energy Storage Resources / ESRs)
for a given month, using the ERCOT Public API.

Data source:
  - NP4-160-SG: Settlement Points List and Electrical Buses Mapping
    https://www.ercot.com/mp/data-products/data-product-details?id=NP4-160-SG
    This report contains all settlement points including resource nodes, tagged by type.
    Battery/ESR nodes have a Settlement Point Type of "ESR" or a resource type indicating storage.

Requirements:
  pip install requests pandas openpyxl

Setup:
  1. Register at https://apiexplorer.ercot.com/ and get your subscription key.
  2. Set the three variables below (or use environment variables).
"""

import os
import requests
import pandas as pd
from datetime import datetime
from io import BytesIO
from zipfile import ZipFile

# ─────────────────────────────────────────────
# CONFIGURATION — fill these in or use env vars
# ─────────────────────────────────────────────
ERCOT_USERNAME = os.getenv("ERCOT_USERNAME", "sand.gh1902@gmail.com")
ERCOT_PASSWORD = os.getenv("ERCOT_PASSWORD", "$275ZryJw#9uSEU")
ERCOT_SUBSCRIPTION_KEY = os.getenv("ERCOT_SUBSCRIPTION_KEY", ERCOT_API_KEY)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TOKEN_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com"
    "/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
BASE_API_URL = "https://api.ercot.com/api/1/services/pubapi-apim-api"

# NP4-160-SG: Settlement Points List and Electrical Buses Mapping
# This is the best source for resource node type classification
SETTLEMENT_POINTS_ENDPOINT = "/np4-160-sg/settlement_points"

# NP3-233-CD: Hourly Resource Outage Capacity — also contains unit type info
# as a secondary option if NP4-160-SG doesn't expose fuel type directly
RESOURCE_OUTAGE_ENDPOINT = "/np3-233-cd/hourly_res_outage_cap"


def get_id_token(username: str, password: str) -> str:
    """
    Authenticate with ERCOT B2C and return a short-lived ID token (valid 1 hour).
    """
    print("Authenticating with ERCOT...")
    response = requests.post(
        TOKEN_URL,
        data={
            "username": username,
            "password": password,
            "grant_type": "password",
            "scope": f"openid {CLIENT_ID} offline_access",
            "client_id": CLIENT_ID,
            "response_type": "id_token",
        },
    )
    response.raise_for_status()
    token = response.json().get("id_token")
    if not token:
        raise ValueError(f"No id_token in response: {response.json()}")
    print("  ✓ Token obtained.")
    return token


def make_headers(id_token: str) -> dict:
    return {
        "Authorization": f"Bearer {id_token}",
        "Ocp-Apim-Subscription-Key": ERCOT_SUBSCRIPTION_KEY,
    }


def fetch_all_pages(endpoint: str, headers: dict, params: dict = None) -> list:
    """
    Paginate through an ERCOT API endpoint and return all records as a list of dicts.
    ERCOT uses page/size query params; default page size is 100.
    """
    url = BASE_API_URL + endpoint
    params = params or {}
    params.setdefault("size", 1000)
    params["page"] = 1

    all_records = []
    while True:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        # ERCOT API returns {"data": [...], "_meta": {"totalPages": N, ...}}
        records = data.get("data", [])
        all_records.extend(records)

        meta = data.get("_meta", {})
        total_pages = meta.get("totalPages", 1)
        current_page = params["page"]

        print(f"  Fetched page {current_page}/{total_pages} ({len(records)} records)")

        if current_page >= total_pages:
            break
        params["page"] += 1

    return all_records


def get_battery_nodes_from_settlement_points(headers: dict) -> pd.DataFrame:
    """
    Pull the Settlement Points List (NP4-160-SG) and filter for ESR/battery nodes.

    The settlement point type for battery nodes is typically:
      - "ESR" (Energy Storage Resource)
      - or resource nodes whose name ends in "_ESR" or contains battery indicators
    """
    print("\nFetching Settlement Points List (NP4-160-SG)...")
    records = fetch_all_pages(SETTLEMENT_POINTS_ENDPOINT, headers)

    if not records:
        print("  No records returned. The endpoint may require a different approach.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"  Total settlement points: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Filter for ESR/battery nodes — check common column names ERCOT uses
    battery_df = pd.DataFrame()
    for col in df.columns:
        col_lower = col.lower()
        if "type" in col_lower or "fuel" in col_lower or "unit" in col_lower:
            unique_vals = df[col].dropna().unique()
            print(f"  Column '{col}' unique values (sample): {unique_vals[:10]}")

    # Try filtering by settlement point type = ESR
    if "settlementPointType" in df.columns:
        battery_df = df[
            df["settlementPointType"]
            .str.upper()
            .str.contains("ESR|BESS|STORAGE", na=False)
        ]
    elif "spType" in df.columns:
        battery_df = df[
            df["spType"].str.upper().str.contains("ESR|BESS|STORAGE", na=False)
        ]
    elif "resourceType" in df.columns:
        battery_df = df[
            df["resourceType"].str.upper().str.contains("ESR|BESS|STORAGE", na=False)
        ]
    else:
        # Fall back: filter by name pattern (many ERCOT battery nodes contain "ESR" or "BESS")
        name_col = next(
            (c for c in df.columns if "name" in c.lower() or "point" in c.lower()), None
        )
        if name_col:
            battery_df = df[
                df[name_col].str.upper().str.contains("ESR|BESS|BATTERY", na=False)
            ]
            print(f"  Fell back to name-based filter on column '{name_col}'")

    return battery_df


def get_battery_nodes_from_outage_report(
    headers: dict, year: int, month: int
) -> pd.DataFrame:
    """
    Pull the Hourly Resource Outage Capacity report (NP3-233-CD) for a given month
    and extract unique ESR resource nodes. This report includes unitType which
    identifies batteries.
    """
    # Build date range for the month
    start = f"{year}-{month:02d}-01T00:00:00"
    if month == 12:
        end = f"{year + 1}-01-01T00:00:00"
    else:
        end = f"{year}-{month + 1:02d}-01T00:00:00"

    print(f"\nFetching Resource Outage report for {year}-{month:02d} (NP3-233-CD)...")
    params = {
        "OutageStartFrom": start,
        "OutageStartTo": end,
    }
    records = fetch_all_pages(RESOURCE_OUTAGE_ENDPOINT, headers, params)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"  Total outage records: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Filter for ESR unit type
    for col in df.columns:
        col_upper = col.upper()
        if "UNIT" in col_upper or "TYPE" in col_upper or "FUEL" in col_upper:
            unique_vals = df[col].dropna().unique()
            print(f"  Column '{col}' sample values: {unique_vals[:10]}")

    battery_df = pd.DataFrame()
    if "unitType" in df.columns:
        battery_df = df[
            df["unitType"]
            .str.upper()
            .str.contains("ESR|BESS|STORAGE|BATTERY", na=False)
        ]
    elif "fuelType" in df.columns:
        battery_df = df[
            df["fuelType"]
            .str.upper()
            .str.contains("ESR|BESS|STORAGE|BATTERY", na=False)
        ]

    return battery_df


def get_monthly_battery_nodes(year: int, month: int) -> pd.DataFrame:
    """
    Main function: authenticate and return a DataFrame of battery resource nodes
    active during the given month.
    """
    id_token = get_id_token(ERCOT_USERNAME, ERCOT_PASSWORD)
    headers = make_headers(id_token)

    # Strategy 1: Settlement Points list (static reference data, no date filter needed)
    battery_df = get_battery_nodes_from_settlement_points(headers)

    if battery_df.empty:
        # Strategy 2: Outage report filtered to the month (includes unit type)
        battery_df = get_battery_nodes_from_outage_report(headers, year, month)

    if battery_df.empty:
        print("\n⚠️  No battery nodes found via API. See notes below.")
        return pd.DataFrame()

    # Deduplicate to unique nodes
    name_col = next(
        (
            c
            for c in battery_df.columns
            if "name" in c.lower() or "node" in c.lower() or "point" in c.lower()
        ),
        battery_df.columns[0],
    )
    battery_df = battery_df.drop_duplicates(subset=[name_col])
    battery_df = battery_df.sort_values(name_col).reset_index(drop=True)

    print(
        f"\n✓ Found {len(battery_df)} unique battery/ESR resource nodes for {year}-{month:02d}."
    )
    return battery_df


def save_results(df: pd.DataFrame, year: int, month: int, output_path: str = None):
    """Save results to a CSV file."""
    if output_path is None:
        output_path = f"ercot_battery_nodes_{year}_{month:02d}.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    return output_path


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Change year/month to the month you want
    YEAR = datetime.now().year
    MONTH = datetime.now().month

    print(f"=== ERCOT Battery Node Fetcher ===")
    print(f"Target month: {YEAR}-{MONTH:02d}\n")

    df = get_monthly_battery_nodes(YEAR, MONTH)

    if not df.empty:
        print("\nFirst 10 results:")
        print(df.head(10).to_string())
        save_results(df, YEAR, MONTH)
    else:
        print(
            "\nNOTES ON TROUBLESHOOTING:\n"
            "  1. The ERCOT API is continually evolving. If no results come back,\n"
            "     check https://apiexplorer.ercot.com/ to browse live endpoint schemas.\n"
            "  2. The NP4-160-SG settlement points file is also downloadable directly\n"
            "     from https://www.ercot.com/mp/data-products/data-product-details?id=NP4-160-SG\n"
            "     as a ZIP/XLS — the 'SettlementPointType' column will show 'ESR' for batteries.\n"
            "  3. The GIS Report (PG7-200-ER) is another good source — it lists all\n"
            "     interconnected resources with fuel type and includes ESR/battery entries.\n"
            "     Download at: https://www.ercot.com/mp/data-products/data-product-details?id=PG7-200-ER\n"
        )


# ─────────────────────────────────────────────
# ALTERNATIVE: Download NP4-160-SG directly as XLS (no API token needed)
# ─────────────────────────────────────────────
def get_battery_nodes_from_xls_download(xls_path: str) -> pd.DataFrame:
    """
    If you've manually downloaded the NP4-160-SG Excel file from the MIS portal,
    this function parses it and returns the battery nodes.

    The file typically has a sheet with columns including:
      SettlementPoint, SettlementPointType, BusName, ...

    SettlementPointType values:
      RN  = Resource Node (generation)
      ESR = Energy Storage Resource (battery)  <-- we want these
      LZ  = Load Zone
      HB  = Hub
    """
    print(f"Reading {xls_path}...")
    # Try reading the first sheet; the settlement points sheet is usually sheet 1
    df = pd.read_excel(xls_path, sheet_name=0)
    print(f"  Columns: {list(df.columns)}")

    # Find the type column
    type_col = next((c for c in df.columns if "type" in c.lower()), None)
    if type_col:
        battery_df = df[df[type_col].astype(str).str.upper() == "ESR"]
        print(f"  Found {len(battery_df)} ESR nodes in '{type_col}' column.")
        return battery_df
    else:
        print(
            "  Could not find a 'type' column. Print all columns and inspect manually."
        )
        return df
