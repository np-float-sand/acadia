from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transmission_rate_base import config

CACHE_DIR = Path(__file__).parent / "cache"


def _read_remote(url: str) -> pd.DataFrame:
    """Isolated network read so tests can monkeypatch it."""
    return pd.read_parquet(url)


def _cache_path(table_name: str) -> Path:
    return CACHE_DIR / f"{table_name}.parquet"


def fetch_table(key: str, *, refresh: bool = False, offline: bool = False) -> pd.DataFrame:
    """Return a PUDL FERC Form 1 table, cached locally as parquet.

    ``key`` is a ``config.FERC_TABLES`` key. Reads the local cache when present
    (unless ``refresh``); otherwise downloads from PUDL's public parquet mirror
    and writes the cache. ``offline=True`` forbids the network: use the cache or
    raise ``FileNotFoundError``.
    """
    table_name = config.FERC_TABLES[key]  # KeyError on unknown key
    path = _cache_path(table_name)

    if path.exists() and not refresh:
        return pd.read_parquet(path)

    if offline:
        raise FileNotFoundError(
            f"offline=True and no cache at {path}; run once online or with --refresh-ferc"
        )

    df = _read_remote(config.PUDL_BASE + table_name + ".parquet")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


# ── Schedule extraction ──────────────────────────────────────────────────────

_RTO_MARKER = "regional_transmission_and_market_operation"


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["utility_id_ferc1"] = df["utility_id_ferc1"].astype(int)
    df["report_year"] = df["report_year"].astype(int)
    return df


def _drop_corrections(df: pd.DataFrame) -> pd.DataFrame:
    if "row_type_xbrl" in df.columns:
        rt = df["row_type_xbrl"].astype(str).str.lower()
        return df[~rt.str.contains("correction")]
    return df


def gross_transmission_plant(sched204: pd.DataFrame) -> pd.DataFrame:
    """Gross transmission plant in service (FERC accts 350-359) per filer-year.

    Prefers the ``transmission_plant`` calculated subtotal row; falls back to
    summing the leaf ``*_transmission_plant`` accounts. RTO / market-operation
    plant and correction rows are excluded.
    """
    df = _drop_corrections(_norm(sched204))
    label = df["ferc_account_label"].fillna("").str.lower()
    df = df[~label.str.contains(_RTO_MARKER)]
    label = df["ferc_account_label"].fillna("").str.lower()

    is_subtotal = label.eq("transmission_plant")
    if "row_type_xbrl" in df.columns:
        rtl = df["row_type_xbrl"].astype(str).str.lower()
        is_subtotal = is_subtotal & (df["row_type_xbrl"].isna() | rtl.eq("calculated_value"))
    subtotals = (df[is_subtotal].groupby(["utility_id_ferc1", "report_year"])["ending_balance"]
                 .sum().rename("gross_tx"))

    leaves = df[label.str.endswith("_transmission_plant") & ~label.eq("transmission_plant")]
    leaf_sum = (leaves.groupby(["utility_id_ferc1", "report_year"])["ending_balance"]
                .sum().rename("gross_tx"))

    out = subtotals.combine_first(leaf_sum).reset_index()
    return out[["utility_id_ferc1", "report_year", "gross_tx"]]


def transmission_accum_depreciation(sched219: pd.DataFrame) -> pd.DataFrame:
    """Accumulated depreciation for the transmission function per filer-year,
    returned as a positive number (sched219 balances are stored negative)."""
    cols = ["utility_id_ferc1", "report_year", "accum_dep_tx"]
    if sched219.empty:
        return pd.DataFrame(columns=cols)
    df = _norm(sched219)
    func = df["plant_function"].fillna("").str.lower()
    dep = df["depreciation_type"].fillna("").str.lower()
    keep = func.eq("transmission") & dep.eq("accumulated_depreciation")
    if "row_type_xbrl" in df.columns:
        rtl = df["row_type_xbrl"].astype(str).str.lower()
        keep = keep & (df["row_type_xbrl"].isna() | rtl.eq("reported_value"))
    out = (df[keep].groupby(["utility_id_ferc1", "report_year"])["ending_balance"]
           .sum().abs().rename("accum_dep_tx").reset_index())
    return out[cols]


def total_net_utility_plant(sched200: pd.DataFrame) -> pd.DataFrame:
    """Net and gross total electric utility plant per filer-year, from
    ``utility_plant_asset_type`` rows (``utility_plant_net`` /
    ``utility_plant_in_service_classified``)."""
    df = _norm(sched200)
    if "utility_type" in df.columns:
        df = df[df["utility_type"].fillna("electric").astype(str).str.lower() == "electric"]
    df = _drop_corrections(df)
    at = df["utility_plant_asset_type"].fillna("").str.lower()

    def _series(name: str) -> pd.Series:
        return (df[at == name].groupby(["utility_id_ferc1", "report_year"])["ending_balance"].sum())

    net = _series("utility_plant_net")
    gross = _series("utility_plant_in_service_classified")
    if gross.empty:
        gross = _series("utility_plant_in_service_classified_and_unclassified")
    out = pd.DataFrame({"net_total": net, "gross_total": gross}).reset_index()
    return out[["utility_id_ferc1", "report_year", "net_total", "gross_total"]]


def net_transmission_plant(sched204: pd.DataFrame, sched219: pd.DataFrame,
                           sched200: pd.DataFrame) -> pd.DataFrame:
    """net_tx = gross_tx - accum_dep_tx; where sched219 has no transmission row
    for a filer-year, fall back to gross_tx * (net_total / gross_total)."""
    gross = gross_transmission_plant(sched204)
    dep = transmission_accum_depreciation(sched219)
    totals = total_net_utility_plant(sched200)

    m = gross.merge(dep, on=["utility_id_ferc1", "report_year"], how="left")
    m = m.merge(totals, on=["utility_id_ferc1", "report_year"], how="left")

    ratio = (m["net_total"] / m["gross_total"]).clip(lower=0.0, upper=1.0)
    prorated = m["accum_dep_tx"].isna()
    m["net_tx"] = np.where(prorated, m["gross_tx"] * ratio, m["gross_tx"] - m["accum_dep_tx"])
    m["net_tx_prorated"] = prorated
    return m[["utility_id_ferc1", "report_year", "net_tx", "net_tx_prorated"]]
