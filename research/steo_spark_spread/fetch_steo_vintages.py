"""
Fetch point-in-time STEO monthly data-table vintages via the Wayback Machine.

Background (docs/steo-archive-feasibility-results.md, docs/untried-tasks-log.md item 6):
EIA's own archive (eia.gov/outlooks/steo/archives/{mon}{yy}.pdf) is PDF-only for every
past vintage -- no XLSX/CSV companion exists there (confirmed 404). BUT EIA's *current*
release also publishes a machine-readable companion workbook at a fixed URL that gets
overwritten every month:

    https://www.eia.gov/outlooks/steo/xls/STEO_m.xlsx

The Wayback Machine has opportunistically crawled that exact URL ~60+ times since 2016,
which means each crawled snapshot IS a point-in-time vintage of the STEO data tables in
structured XLSX form -- no PDF-table extraction needed. Coverage is NOT monthly (crawl
frequency is opportunistic, gaps of 2-6 months are common), but per the task brief:
"prioritize getting a working point-in-time panel over exhaustive full history."

This script queries the CDX API for all 200-status snapshots of that URL (collapsed to
one per month) and downloads each one's *raw* bytes (the `id_` modifier strips Wayback's
HTML wrapper) to data/raw_xlsx/{capture_timestamp}.xlsx.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).parent
RAW_DIR = HERE / "data" / "raw_xlsx"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CDX_URL = (
    "http://web.archive.org/cdx/search/cdx"
    "?url=eia.gov/outlooks/steo/xls/STEO_m.xlsx"
    "&output=json&collapse=timestamp:6&filter=statuscode:200"
)


def list_snapshots() -> list[tuple[str, str]]:
    """Return [(timestamp, original_url), ...] for one snapshot per month."""
    with urllib.request.urlopen(CDX_URL, timeout=60) as resp:
        rows = json.load(resp)
    header, data = rows[0], rows[1:]
    ts_i = header.index("timestamp")
    url_i = header.index("original")
    return [(r[ts_i], r[url_i]) for r in data]


def fetch_one(timestamp: str, original_url: str, dest: pathlib.Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 10_000:
        return True
    raw_url = f"https://web.archive.org/web/{timestamp}id_/{original_url}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(raw_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
            if len(data) < 10_000:
                print(f"  [{timestamp}] suspiciously small ({len(data)}B), retrying...")
                time.sleep(2)
                continue
            dest.write_bytes(data)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  [{timestamp}] attempt {attempt+1} failed: {e}")
            time.sleep(3)
    return False


def main():
    snaps = list_snapshots()
    print(f"Found {len(snaps)} monthly snapshots, {snaps[0][0][:6]} -> {snaps[-1][0][:6]}")
    ok, fail = 0, 0
    for ts, url in snaps:
        dest = RAW_DIR / f"{ts}.xlsx"
        success = fetch_one(ts, url, dest)
        if success:
            ok += 1
        else:
            fail += 1
            print(f"  FAILED: {ts}")
        time.sleep(0.3)  # be polite to archive.org
    print(f"\nDone: {ok} downloaded/cached, {fail} failed. Files in {RAW_DIR}")


if __name__ == "__main__":
    sys.exit(main())
