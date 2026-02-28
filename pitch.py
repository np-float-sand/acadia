import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define tickers
tickers = ["NRG", "VST", "CNP", "AEP"]

# Extended history
start = "2020-01-01"
end = "2021-06-01"

# Download adjusted prices
raw = yf.download(
    tickers,
    start=start,
    end=end,
    auto_adjust=True,
    group_by="column",
    progress=False,
)

# Extract Close robustly
if isinstance(raw.columns, pd.MultiIndex):
    prices = raw.xs("Close", axis=1, level=0)
else:
    prices = raw["Close"] if "Close" in raw.columns else raw

prices = prices.dropna(how="all")

# Normalize to 100
base = prices.ffill().bfill().iloc[0]
normalized = prices / base * 100

# Define shading windows
uri_start = pd.Timestamp("2021-02-12")
uri_end = pd.Timestamp("2021-02-20")

follow_start = pd.Timestamp("2021-02-21")
follow_end = pd.Timestamp("2021-03-02")

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

for ax, ticker in zip(axes, tickers):
    if ticker not in normalized.columns:
        continue

    ax.plot(normalized.index, normalized[ticker])

    # Uri core (darker)
    ax.axvspan(uri_start, uri_end, alpha=0.25)

    # Follow-on (lighter)
    ax.axvspan(follow_start, follow_end, alpha=0.10)

    ax.set_title(ticker)
    ax.set_ylabel("Base = 100")

    # Rotate and resize x-axis labels
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)

# Formatting
fig.suptitle("Texas Utilities Around Winter Storm Uri (Extended Context)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig("utilities_uri_2x2_extended.png", dpi=300, bbox_inches="tight")
plt.show()
