"""
Comanche Peak Node Lookup + LMP Fetch
======================================
Step 1: Download ERCOT's Settlement Points List (public, no auth needed)
        and search for Comanche Peak / CPNPP resource nodes.

Step 2: Once the node name is confirmed, fetch its settlement point price
        and compare against HB_NORTH hub during Winter Storm Uri.

Step 3: Plot node LMP vs hub LMP — the spread is the congestion component
        for that specific generator location.

Usage:
    pip install requests pandas matplotlib
    python comanche_peak_node.py

No API key required for steps 1 and 2.
The ERCOT public reports (MIS) are freely accessible.
"""

import io
import zipfile
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ── Colour palette (matches the deck) ────────────────────────────────────────
C = dict(
    forest="#1A4D2E",
    moss="#4A7C59",
    gold="#C9A84C",
    cream="#F4F1E8",
    dark="#0D1F2D",
    red="#C05040",
    sage="#8FB996",
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  Find the exact ERCOT resource node name for Comanche Peak
# ─────────────────────────────────────────────────────────────────────────────
# ERCOT publishes the full settlement point list as a ZIP/CSV weekly.
# Report NP4-160-SG: "Settlement Points List and Electrical Buses Mapping"
# Public MIS URL (no login needed):

SP_LIST_URL = (
    "https://mis.ercot.com/misapp/GetReports.do"
    "?reportTypeId=10008"
    "&reportTitle=Settlement+Points+List+and+Electrical+Buses+Mapping"
    "&showHTMLView="
    "&mimicKey"
)

# Fallback: if the above ZIP is unavailable, we also know from ERCOT's
# network model that Comanche Peak units connect at the GLEN ROSE 345kV bus
# on Oncor's transmission system. The resource node names follow the pattern:
#
#   CPNPP_UNIT1   or   LUM_CPNPP1   or   CPNP_U1
#
# The exact string depends on the DUNS registration name that Luminant/Vistra
# used when they registered the resource with ERCOT. The lookup below will
# resolve this. If network access is unavailable the known candidates are
# listed as FALLBACK_CANDIDATES for manual verification on ERCOT MIS.

FALLBACK_CANDIDATES = [
    # Most likely forms based on ERCOT naming conventions for nuclear units:
    "CPNPP_UNIT1",  # Comanche Peak Nuclear Power Plant Unit 1
    "CPNPP_UNIT2",  # Comanche Peak Nuclear Power Plant Unit 2
    "LUM_CPNPP1",  # Luminant prefix variant
    "LUM_CPNPP2",
    "VISTRA_CPNPP1",  # Post-Vistra rebranding variant
    "VISTRA_CPNPP2",
    # The 345kV bus at the plant substation (Oncor naming):
    "GLEN_ROSE_345",  # bus-level node (less common for gen settlement)
    "COMANCHE_PK1",
    "COMANCHE_PK2",
]

SEARCH_TERMS = ["CPNPP", "COMANCHE", "GLEN ROSE", "GLENROSE", "LUMINANT_NUC"]


def fetch_settlement_point_list() -> pd.DataFrame:
    """
    Download ERCOT's Settlement Points List and return as DataFrame.
    Columns include: settlementPoint, settlementPointType, electricalBus, etc.
    """
    print("Fetching ERCOT Settlement Points List (NP4-160-SG)...")
    try:
        # ERCOT serves this as a redirect to the latest ZIP file
        r = requests.get(SP_LIST_URL, timeout=30, allow_redirects=True)
        r.raise_for_status()

        # The response is an HTML page listing available files.
        # We need to parse it to get the actual ZIP download link.
        # Files are listed as links ending in .zip or .csv
        from html.parser import HTMLParser

        class LinkParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links = []

            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    for name, val in attrs:
                        if (
                            name == "href"
                            and val
                            and (".zip" in val.lower() or ".csv" in val.lower())
                        ):
                            self.links.append(val)

        parser = LinkParser()
        parser.feed(r.text)

        if not parser.links:
            print("  No ZIP/CSV links found in response. Trying direct CSV endpoint...")
            raise ValueError("No file links found")

        # Take the most recent file (usually first or last)
        file_url = parser.links[0]
        if not file_url.startswith("http"):
            file_url = "https://mis.ercot.com" + file_url

        print(f"  Downloading: {file_url}")
        fr = requests.get(file_url, timeout=60)
        fr.raise_for_status()

        if file_url.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(fr.content)) as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_names:
                    raise ValueError("No CSV found inside ZIP")
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
        else:
            df = pd.read_csv(io.BytesIO(fr.content))

        print(f"  Loaded {len(df):,} settlement points.")
        return df

    except Exception as e:
        print(f"  Could not fetch settlement point list: {e}")
        return pd.DataFrame()


def search_for_comanche_peak(sp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Search the settlement points DataFrame for anything related to
    Comanche Peak / CPNPP.
    """
    if sp_df.empty:
        return pd.DataFrame()

    # Normalise all string columns for search
    str_cols = sp_df.select_dtypes(include="object").columns
    mask = pd.Series(False, index=sp_df.index)

    for term in SEARCH_TERMS:
        for col in str_cols:
            mask |= (
                sp_df[col].astype(str).str.upper().str.contains(term.upper(), na=False)
            )

    results = sp_df[mask]
    print(
        f"\n  Found {len(results)} settlement points matching Comanche Peak search terms:"
    )
    if not results.empty:
        print(results.to_string(index=False))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  Fetch settlement point prices for the identified node
# ─────────────────────────────────────────────────────────────────────────────
# ERCOT Report NP6-905-CD: "Settlement Point Prices at Resource Nodes,
# Hubs and Load Zones" — 15-minute interval, public.
#
# The bulk historical files are at:
#   https://mis.ercot.com/misapp/GetReports.do?reportTypeId=12301
#
# Files are named by operating day: e.g., cdr.00012301.0000000000000000.20210215.zip

SPP_BASE = (
    "https://mis.ercot.com/misapp/GetReports.do"
    "?reportTypeId=12301"
    "&reportTitle=Settlement+Point+Prices+at+Resource+Nodes%2C+Hubs+and+Load+Zones"
    "&showHTMLView="
    "&mimicKey"
)

# Direct URL pattern for a specific date's SPP report:
SPP_DATE_URL = (
    "https://mis.ercot.com/misfile/download"
    "?reportTypeId=12301"
    "&marketRunId=0"
    "&operatingDate={date}"  # YYYY-MM-DD
    "&version=1"
    "&bypass=true"
)


def fetch_spp_for_date(date_str: str, settlement_points: list) -> pd.DataFrame:
    """
    Fetch 15-minute Settlement Point Prices for a specific date.

    date_str: "YYYY-MM-DD"
    settlement_points: list of node names to extract (e.g. ["CPNPP_UNIT1", "HB_NORTH"])

    Returns DataFrame indexed by datetime with one column per settlement point.
    """
    url = SPP_DATE_URL.format(date=date_str)
    print(f"  Fetching SPP for {date_str}...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        if "zip" in r.headers.get("Content-Type", "").lower() or r.content[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_names:
                    raise ValueError("No CSV in ZIP")
                with z.open(csv_names[0]) as f:
                    raw = pd.read_csv(f)
        else:
            raw = pd.read_csv(io.StringIO(r.text))

        # Normalise column names
        raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "_")

        # Expected columns: deliverydate, deliveryhour, deliveryinterval,
        # settlementpointname, settlementpointprice
        sp_col = next(
            (c for c in raw.columns if "settlementpointname" in c or "sp_name" in c),
            None,
        )
        price_col = next((c for c in raw.columns if "price" in c), None)
        date_col = next(
            (c for c in raw.columns if "deliverydate" in c or "oper" in c), None
        )
        hour_col = next(
            (c for c in raw.columns if "deliveryhour" in c or "hour" in c), None
        )
        intv_col = next((c for c in raw.columns if "interval" in c), None)

        if not all([sp_col, price_col, date_col, hour_col]):
            print(f"    Unexpected columns: {list(raw.columns)}")
            return pd.DataFrame()

        # Filter to requested nodes
        upper_names = [s.upper() for s in settlement_points]
        filtered = raw[raw[sp_col].str.upper().isin(upper_names)].copy()

        if filtered.empty:
            print(f"    None of {settlement_points} found in this file.")
            print(f"    Available (sample): {raw[sp_col].unique()[:10]}")
            return pd.DataFrame()

        # Build datetime index (ERCOT hours are 1-24, intervals 1-4 per hour)
        def to_dt(row):
            h = int(row[hour_col]) - 1  # convert 1-based → 0-based
            intv = int(row[intv_col]) - 1 if intv_col else 0
            mins = h * 60 + intv * 15
            base = pd.Timestamp(row[date_col])
            return base + pd.Timedelta(minutes=mins)

        filtered["datetime"] = filtered.apply(to_dt, axis=1)
        pivot = filtered.pivot_table(
            index="datetime", columns=sp_col, values=price_col, aggfunc="mean"
        )
        return pivot

    except Exception as e:
        print(f"    Error fetching {date_str}: {e}")
        return pd.DataFrame()


def fetch_spp_range(
    start: str,
    end: str,
    settlement_points: list,
) -> pd.DataFrame:
    """
    Fetch SPP over a date range and concatenate.
    Returns 15-minute DataFrame with one column per settlement point.
    """
    dates = pd.date_range(start, end, freq="D")
    frames = []
    for d in dates:
        df = fetch_spp_for_date(d.strftime("%Y-%m-%d"), settlement_points)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  Plot: node LMP vs hub LMP, with Uri event window
# ─────────────────────────────────────────────────────────────────────────────


def plot_node_vs_hub(
    spp_15min: pd.DataFrame,
    node_col: str,
    hub_col: str = "HB_NORTH",
    event_start: str = "2021-02-10",
    event_end: str = "2021-02-20",
    context_start: str = "2021-01-15",
    context_end: str = "2021-03-15",
    save_path: str = "comanche_peak_node_lmp.png",
):
    """
    Three-panel chart:
      Top:    Raw 15-min LMP for node and hub (clipped to readable range)
      Middle: Congestion spread = node LMP − hub LMP  (negative = node pays less; positive = bottleneck)
      Bottom: Hourly aggregated version for clarity
    """
    # Clip to context window
    mask = (spp_15min.index >= context_start) & (spp_15min.index <= context_end)
    df = spp_15min.loc[mask].copy()

    if node_col not in df.columns:
        print(f"  Column '{node_col}' not found. Available: {list(df.columns)}")
        return

    node = df[node_col]
    hub = df[hub_col] if hub_col in df.columns else pd.Series(dtype=float)

    # Daily aggregates for the bottom panel
    node_daily = node.resample("D").agg(["mean", "max"])
    hub_daily = (
        hub.resample("D").agg(["mean", "max"]) if not hub.empty else pd.DataFrame()
    )

    # Congestion spread (15-min)
    spread_15min = node - hub if not hub.empty else pd.Series(dtype=float)
    spread_daily = (
        spread_15min.resample("D").mean()
        if not spread_15min.empty
        else pd.Series(dtype=float)
    )

    ev_start = pd.Timestamp(event_start)
    ev_end = pd.Timestamp(event_end)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 11),
        facecolor=C["cream"],
        gridspec_kw={"height_ratios": [2, 1.2, 1.5]},
    )
    fig.subplots_adjust(hspace=0.42)

    # ── Panel 1: Raw LMPs (15-min, clipped at 500 for readability) ───────────
    ax = axes[0]
    ax.set_facecolor("white")
    CLIP = 500  # $/MWh cap — Uri hit $9,000 but that compresses the rest

    node_clip = node.clip(upper=CLIP)
    ax.plot(
        node_clip.index,
        node_clip.values,
        color=C["forest"],
        linewidth=0.6,
        alpha=0.85,
        label=f"{node_col} (clipped at ${CLIP})",
    )

    if not hub.empty:
        hub_clip = hub.clip(upper=CLIP)
        ax.plot(
            hub_clip.index,
            hub_clip.values,
            color=C["gold"],
            linewidth=0.6,
            alpha=0.7,
            label=f"{hub_col} (clipped at ${CLIP})",
        )

    ax.axvspan(ev_start, ev_end, alpha=0.15, color=C["red"], zorder=0)
    ax.axhline(0, color=C["dark"], linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_ylabel("LMP ($/MWh)", fontsize=9)
    ax.set_title(
        f"Comanche Peak Resource Node vs HB_NORTH Hub\n"
        f"15-minute Settlement Point Prices  ·  Uri window shaded",
        fontsize=11,
        fontweight="bold",
        color=C["dark"],
    )
    ax.legend(fontsize=8, framealpha=0.8)
    ax.text(
        ev_start + pd.Timedelta(hours=12),
        CLIP * 0.88,
        "Winter Storm\nUri",
        color=C["red"],
        fontsize=7.5,
        va="top",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)

    # ── Panel 2: Congestion spread (node − hub), 15-min ──────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("white")

    if not spread_15min.empty:
        spread_clip = spread_15min.clip(lower=-200, upper=200)
        pos = spread_clip.clip(lower=0)
        neg = spread_clip.clip(upper=0)
        ax2.fill_between(
            spread_clip.index,
            pos,
            0,
            color=C["forest"],
            alpha=0.55,
            label="Node premium (node > hub)",
        )
        ax2.fill_between(
            spread_clip.index,
            neg,
            0,
            color=C["red"],
            alpha=0.55,
            label="Node discount (node < hub)",
        )
        ax2.axvspan(ev_start, ev_end, alpha=0.12, color=C["red"], zorder=0)
        ax2.axhline(0, color=C["dark"], linewidth=0.8)
        ax2.set_ylabel("Spread $/MWh\n(node − hub, clipped ±200)", fontsize=8)
        ax2.set_title(
            "Congestion Spread: Node LMP − HB_NORTH",
            fontsize=10,
            fontweight="bold",
            color=C["dark"],
        )
        ax2.legend(fontsize=8, framealpha=0.8, loc="upper left")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax2.tick_params(axis="x", labelrotation=30, labelsize=7)

    # ── Panel 3: Daily mean LMP ───────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("white")

    ax3.plot(
        node_daily.index,
        node_daily["mean"],
        color=C["forest"],
        linewidth=2,
        marker="o",
        markersize=3,
        label=f"{node_col} daily mean",
    )
    if not hub_daily.empty:
        ax3.plot(
            hub_daily.index,
            hub_daily["mean"],
            color=C["gold"],
            linewidth=2,
            marker="o",
            markersize=3,
            linestyle="--",
            label="HB_NORTH daily mean",
        )
    if not spread_daily.empty:
        ax3_r = ax3.twinx()
        ax3_r.bar(
            spread_daily.index,
            spread_daily.values,
            color=C["moss"],
            alpha=0.4,
            width=0.7,
            label="Daily spread (RHS)",
        )
        ax3_r.set_ylabel("Daily spread $/MWh", fontsize=8, color=C["moss"])
        ax3_r.tick_params(axis="y", colors=C["moss"])
        ax3_r.spines[["top"]].set_visible(False)

    ax3.axvspan(ev_start, ev_end, alpha=0.15, color=C["red"], zorder=0)
    ax3.set_ylabel("Daily Mean LMP ($/MWh)", fontsize=9)
    ax3.set_title(
        "Daily Mean LMP — Node vs Hub", fontsize=10, fontweight="bold", color=C["dark"]
    )
    ax3.legend(fontsize=8, framealpha=0.8, loc="upper left")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax3.tick_params(axis="x", labelrotation=30, labelsize=7)

    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=C["cream"])
    print(f"\n  Chart saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  Summary stats: what happened to Comanche Peak node during Uri?
# ─────────────────────────────────────────────────────────────────────────────


def uri_summary(spp_15min: pd.DataFrame, node_col: str, hub_col: str = "HB_NORTH"):
    """
    Print key stats comparing the node vs hub during Uri vs baseline.
    """
    uri_mask = (spp_15min.index >= "2021-02-10") & (spp_15min.index <= "2021-02-20")
    base_mask = (spp_15min.index >= "2021-01-01") & (spp_15min.index < "2021-02-10")

    def stats(series, label):
        return {
            "period": label,
            "mean": series.mean(),
            "median": series.median(),
            "p90": series.quantile(0.90),
            "max": series.max(),
            "n_hrs_above_500": (series > 500).sum() * 0.25,  # 15-min → hours
        }

    rows = []
    for col in [node_col, hub_col]:
        if col not in spp_15min.columns:
            continue
        rows.append(stats(spp_15min.loc[base_mask, col].dropna(), f"{col} baseline"))
        rows.append(stats(spp_15min.loc[uri_mask, col].dropna(), f"{col} Uri"))

    df = pd.DataFrame(rows).set_index("period")
    print("\n  ── Uri vs Baseline Summary ──────────────────────────────────")
    print(df.round(2).to_string())

    # Congestion spread during Uri
    if node_col in spp_15min.columns and hub_col in spp_15min.columns:
        spread_uri = (spp_15min[node_col] - spp_15min[hub_col]).loc[uri_mask]
        spread_base = (spp_15min[node_col] - spp_15min[hub_col]).loc[base_mask]
        print(f"\n  Congestion spread (node − hub):")
        print(f"    Baseline mean:  ${spread_base.mean():.2f}/MWh")
        print(f"    Uri mean:       ${spread_uri.mean():.2f}/MWh")
        print(f"    Uri max spread: ${spread_uri.max():.2f}/MWh")
        print(f"    Uri min spread: ${spread_uri.min():.2f}/MWh")
        print()
        if spread_uri.mean() > spread_base.mean():
            print("  → Node traded at a PREMIUM to hub during Uri")
            print("    (generation was behind a congested path INTO the node area)")
        else:
            print("  → Node traded at a DISCOUNT to hub during Uri")
            print("    (generation was behind a constrained export path)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

# Once you've run Step 1 and confirmed the node name, set it here:
# Based on ERCOT naming conventions and Luminant's QSE registration,
# the most likely candidates are listed in FALLBACK_CANDIDATES above.
# Set CONFIRMED_NODE_NAME after running the lookup, or use one of these
# directly if you have access to the ERCOT MIS settlement point list.
CONFIRMED_NODE_NAME = None  # e.g. "CPNPP_UNIT1" — set after Step 1

HUB_NODE = "HB_NORTH"

URI_START = "2021-02-10"
URI_END = "2021-02-20"
CTX_START = "2021-01-15"
CTX_END = "2021-03-15"


def main():
    print("=" * 60)
    print("  Comanche Peak Node LMP Analysis")
    print("=" * 60)

    # ── Step 1: Find the node name ────────────────────────────────────────────
    print("\n[STEP 1] Searching ERCOT settlement point list...")
    sp_df = fetch_settlement_point_list()
    candidates = search_for_comanche_peak(sp_df)

    # Determine which node name to use
    if not candidates.empty:
        # Use the first RN (Resource Node) result
        sp_col = next(
            (
                c
                for c in candidates.columns
                if "settlementpoint" in c.lower() or "sp_name" in c.lower()
            ),
            candidates.columns[0],
        )
        type_col = next((c for c in candidates.columns if "type" in c.lower()), None)
        if type_col:
            rn_candidates = candidates[
                candidates[type_col]
                .astype(str)
                .str.upper()
                .isin(["RN", "PCCRN", "LCCRN"])
            ]
            use = rn_candidates if not rn_candidates.empty else candidates
        else:
            use = candidates
        node_name = use.iloc[0][sp_col]
        print(f"\n  Using node: {node_name}")
    elif CONFIRMED_NODE_NAME:
        node_name = CONFIRMED_NODE_NAME
        print(f"\n  Using pre-configured node: {node_name}")
    else:
        # Can't connect — print instructions and exit gracefully
        print("\n  Could not reach ERCOT MIS. To find the node name manually:")
        print()
        print("  Option A — ERCOT MIS (requires free account):")
        print("    1. Go to https://mis.ercot.com")
        print("    2. Reports → Market Information → Settlement Points")
        print("       List and Electrical Buses Mapping (NP4-160-SG)")
        print("    3. Download latest ZIP, open CSV")
        print("    4. Search for: CPNPP, COMANCHE, GLEN_ROSE, LUM_NUC")
        print()
        print("  Option B — gridstatus.io Python library:")
        print("    pip install gridstatus")
        print("    import gridstatus")
        print("    iso = gridstatus.ERCOT()")
        print("    locs = iso.get_settlement_points()")
        print(
            "    locs[locs['Settlement Point'].str.contains('CPNPP|COMANCHE', case=False)]"
        )
        print()
        print("  Option C — Try these candidate names directly in Step 2:")
        for c in FALLBACK_CANDIDATES:
            print(f"    {c}")
        print()
        print("  Once confirmed, set CONFIRMED_NODE_NAME at the top of this")
        print("  script and re-run.")
        return

    # ── Step 2: Fetch LMP data ────────────────────────────────────────────────
    print(f"\n[STEP 2] Fetching LMP data for [{node_name}] and [{HUB_NODE}]")
    print(f"         Period: {CTX_START} → {CTX_END}")

    spp = fetch_spp_range(
        start=CTX_START,
        end=CTX_END,
        settlement_points=[node_name, HUB_NODE],
    )

    if spp.empty:
        print("\n  No LMP data fetched. Check node name and network access.")
        print("  You can also download the CSV directly from:")
        print(
            "  https://www.ercot.com/mp/data-products/data-product-details?id=NP6-905-CD"
        )
        print("  Then load it with: pd.read_csv('your_file.csv', parse_dates=True)")
        return

    print(f"\n  Fetched {len(spp):,} 15-minute intervals.")
    print(f"  Columns: {list(spp.columns)}")
    print(f"\n  Sample (first 5 rows):")
    print(spp.head().to_string())

    # ── Step 3: Summary stats ─────────────────────────────────────────────────
    print("\n[STEP 3] Uri event summary stats")
    uri_summary(spp, node_name, HUB_NODE)

    # ── Step 4: Plot ──────────────────────────────────────────────────────────
    print("\n[STEP 4] Generating chart...")
    plot_node_vs_hub(
        spp_15min=spp,
        node_col=node_name,
        hub_col=HUB_NODE,
        event_start=URI_START,
        event_end=URI_END,
        context_start=CTX_START,
        context_end=CTX_END,
        save_path="comanche_peak_node_lmp.png",
    )

    print("\nDone.")
    return spp


if __name__ == "__main__":
    spp = main()
