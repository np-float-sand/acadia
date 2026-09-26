"""Rerun the grid_resilience pipeline through 2026-08-31 to get stress-beta coverage
into 2026 (needed for the Fold-B train-2024-25/test-2026 walk-forward split)."""
import functools
import sys
sys.path.insert(0, ".")
import pandas as pd

_CAPTURED = []
def _capture_wrapper(fn):
    @functools.wraps(fn)
    def wrapped(*a, **k):
        out = fn(*a, **k)
        _CAPTURED.append(out)
        return out
    return wrapped

import grid_resilience.main as main_mod
main_mod.rolling_stress_betas = _capture_wrapper(main_mod.rolling_stress_betas)

result = main_mod.run(end="2026-08-31", plot=False, save_dir="/tmp/gr_2026_extend")
print("METRICS:", result.get("metrics"))
if _CAPTURED:
    rb = _CAPTURED[-1]
    rb.to_parquet("research/narrative_mispricing/rolling_betas_2018_2026.parquet")
    print("Saved rolling betas:", rb.shape, "max date:", rb.index.get_level_values("date").max())
