"""Pull PJM ancillary market results for 2022-2025, monthly chunks (the post-market-
redesign schema is much larger per unit time than 2018-2021 and 429'd on yearly requests
earlier this session)."""
import sys
import time

import pandas as pd
import gridstatus

pjm = gridstatus.PJM()
frames = []
periods = pd.period_range("2022-01", "2025-09", freq="M")
for p in periods:
    start = p.start_time.strftime("%Y-%m-%d")
    end = (p.end_time + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
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
    full.to_parquet("research/narrative_mispricing/pjm_ancillary_2022_2025.parquet")
    print("TOTAL:", full.shape)
else:
    print("NO DATA COLLECTED")
