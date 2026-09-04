"""Virginia SCC transmission-project filings -- "Deliverable A" (Virginia/Dominion only).

A project-level *demand* database built from Virginia State Corporation Commission
CPCN dockets (Va. Code Sec. 56-46.1). One row per Dominion Energy Virginia
transmission-line certificate case, ~2019-2026, with the driving need classified
from the Commission's final order / hearing-examiner report.

Provenance / status
-------------------
Scoped in ``docs/handoff_2026-09-03-transmission-project-filings.md`` Sec. 4 as a
1-2 day feasibility probe. Result (``docs/va-transmission-filings-probe-results.md``):

* Check (a) -- the data-center-driven project-$ fraction **does rise** 2019->2022
  (~0.27 -> ~0.85 broad) and stays ~0.8 through 2026. Good independent
  *confirmation* the buildout is real, large (~$10B), and DC-driven (~60% strict).
* Check (b) -- **fails as a timing signal.** No leading relationship at quarterly
  resolution; the annual "1-year lead" is n=7 with one common 2022-24 inflection.
  Same failure mode as PJM Table B-9.
* Check (c) -- **cannot weight names from Virginia alone.** One transmission owner
  (Dominion) in ~one PJM zone (DOM); the work-type mix is ~all
  line/rebuild/substation, so the implied tilt is a static "overweight the EPCs,
  zero the equipment makers" -- not time-varying, and it zeros VRT/GEV/NVT (the
  basket's actual winners). The HVDC/FACTS/transformer detail that *would*
  differentiate the makers is not in CPCN final orders.

**Verdict: keep as a monitored input to the discretionary dashboard
(``docs/handoff_2026-09-01-grid-buildout-long-short.md`` Sec. 7); do not extend to
the other four states or build Deliverable B on this basis.** The live SCC helpers
below are kept so the seed can be refreshed / audited, not because a signal runs.

SCC DocketSearch backend (undocumented, may change without notice)
----------------------------------------------------------------
* Breeze/OData:  ``https://www.scc.virginia.gov/DocketSearchAPI/breeze/CaseDetails``
    - ``GetDetail?$filter=Case_Number eq 'PUR-2024-00181'``  -> MATTER_NO + dates + disposition
    - ``GetDocuments?$filter=MATTER_NO eq 145555&$select=DocID,FileName,Document_Name,Date_Filed``
      (the ``$select`` is required -- without it the keyless view collapses every
      row to one ``$ref``.)
* PDF bytes:  ``http://www.scc.virginia.gov/docketsearch/DOCS/<FileName>``
"""
from __future__ import annotations

import urllib.parse
from pathlib import Path

import pandas as pd
import requests

from grid_resilience.config import CACHE_DIR

SEED_CSV = Path(__file__).parent / "seed" / "va_transmission_projects.csv"

_BREEZE = "https://www.scc.virginia.gov/DocketSearchAPI/breeze/CaseDetails"
_DOCS = "http://www.scc.virginia.gov/docketsearch/DOCS/"
_UA = {"User-Agent": "acadia-research/0.1", "Accept": "application/json"}

DRIVERS = ("data_center", "reliability_aging", "load_growth_other",
           "generation_interconnection", "policy_undergrounding")


# ── seed loader (the deliverable) ────────────────────────────────────────────

def load_va_transmission_projects(seed: Path = SEED_CSV) -> pd.DataFrame:
    """Load the hand-classified VA/Dominion transmission-project table.

    Columns: case, matter_no, region, project_name, filed_date, approved_date,
    status, disposition, kv, cost_usd_m, cost_quality (H/M/L), isd_year,
    work_type (``|``-delimited), driver (see ``DRIVERS``), dc_strict / dc_broad
    (0/1 -- strict = a data-center customer/campus/park is named in the order's
    need findings; broad = strict OR the project sits in a DC load pocket and the
    stated need is area load growth), key_doc_kind / key_doc_url, scc_case_url,
    caption.
    """
    df = pd.read_csv(seed)
    df["filed_date"] = pd.to_datetime(df["filed_date"])
    df["approved_date"] = pd.to_datetime(df["approved_date"], errors="coerce")
    df["filed_year"] = df["filed_date"].dt.year
    df["approved_year"] = df["approved_date"].dt.year
    df["cost_usd"] = df["cost_usd_m"].astype(float) * 1e6
    return df


def dc_dollar_fraction(df: pd.DataFrame | None = None, *, by: str = "filed_year",
                       basis: str = "broad", include_canceled: bool = False) -> pd.DataFrame:
    """Per-year data-center-driven project-$ and its fraction of total.

    ``by`` is ``filed_year`` or ``approved_year``; ``basis`` is ``broad`` or
    ``strict``. Returns columns [n, total_usd, dc_usd, dc_fraction].
    """
    if df is None:
        df = load_va_transmission_projects()
    if not include_canceled:
        df = df[df["disposition"] != "Canceled"]
    df = df[df[by].notna()]
    flag = "dc_broad" if basis == "broad" else "dc_strict"
    g = df.groupby(by)
    out = pd.DataFrame({
        "n": g.size(),
        "total_usd": g["cost_usd"].sum(),
        "dc_usd": g.apply(lambda x: x.loc[x[flag] == 1, "cost_usd"].sum()),
    })
    out["dc_fraction"] = out["dc_usd"] / out["total_usd"]
    return out


# ── live SCC helpers (for refreshing / auditing the seed) ────────────────────

def _get(path: str, flt: str, select: str | None = None) -> list[dict]:
    params = {"$filter": flt}
    if select:
        params["$select"] = select
    url = f"{_BREEZE}/{path}?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    r = requests.get(url, headers=_UA, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return r.json()


def fetch_case_detail(case_number: str) -> dict:
    """One case's header row: MATTER_NO, Case_Name, Caption, Status, Disposition,
    Disposition_Date, Case_Established_Date, Final_Order_Date, Closed_Date."""
    rows = _get("GetDetail", f"Case_Number eq '{case_number}'")
    return rows[0] if rows else {}


def fetch_case_documents(matter_no: int) -> pd.DataFrame:
    """Every filed document for a matter: [DocID, FileName, Document_Name, Date_Filed]."""
    rows = _get("GetDocuments", f"MATTER_NO eq {int(matter_no)}",
                "DocID,FileName,Document_Name,Date_Filed")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Date_Filed"] = pd.to_datetime(df["Date_Filed"], errors="coerce")
        df = df.sort_values("Date_Filed", ascending=False).reset_index(drop=True)
    return df


def document_url(filename: str) -> str:
    """Public PDF URL for a docket document ``FileName`` (e.g. ``86$701!.PDF``)."""
    return _DOCS + urllib.parse.quote(filename)


def download_document(filename: str, dest_dir: Path | None = None) -> Path:
    """Fetch a docket PDF to ``dest_dir`` (default: the shared cache). Cached by name."""
    dest_dir = dest_dir or (CACHE_DIR / "va_scc_docs")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / filename.replace("/", "_")
    if out.exists() and out.stat().st_size > 0:
        return out
    r = requests.get(document_url(filename), headers={"User-Agent": _UA["User-Agent"]},
                     timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out
