from __future__ import annotations

from pathlib import Path

import pandas as pd

_SEED_DIR = Path(__file__).parent / "seed"
_DEFAULT_SEED_PATH = _SEED_DIR / "ercot_large_load_monthly.csv"


def load_ercot_large_load_seed(path: Path | None = None) -> pd.DataFrame:
    """Load the checked-in ERCOT large-load seed CSV (see Tasks 4-5 for why
    this is a maintained seed file, not a live fetcher)."""
    df = pd.read_csv(path or _DEFAULT_SEED_PATH)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df
