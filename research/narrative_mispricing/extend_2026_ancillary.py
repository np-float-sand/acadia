"""Extend PJM ancillary market results through 2026-09 (today)."""
import time
import pandas as pd
import gridstatus

pjm = gridstatus.PJM()
frames = []
periods = pd.period_range("2026-01", "2026-09", freq="M")
for p in periods:
    start = p.start_time.strftime("%Y-%m-%d")
    end = min(p.end_time + pd.Timedelta(days=1), pd.Timestamp("2026-09-08")).strftime("%Y-%m-%d")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        continue
    for attempt in range(4):
        try:
            df = pjm.get_real_time_as_market_results(date=start, end=end, verbose=False)
            frames.append(df)
            print(f"{p}: {df.shape}")
            break
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"{p}: attempt {attempt+1} failed ({str(e)[:100]}), waiting {wait}s")
            time.sleep(wait)
    time.sleep(3)

if frames:
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet("research/narrative_mispricing/pjm_ancillary_2026.parquet")
    print("TOTAL:", full.shape)
else:
    print("NO DATA COLLECTED")
