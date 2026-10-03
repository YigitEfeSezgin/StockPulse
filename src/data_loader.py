import yfinance as yf

ticker = "THYAO.IS"

data = yf.download(
    ticker,
    start="2025-01-01",
    end="2026-01-01",
    auto_adjust=False
)

# Flatten MultiIndex columns
if hasattr(data.columns, "levels"):
    data.columns = data.columns.get_level_values(0)

data.columns.name = None

print("Data shape:")
print(data.shape)

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

print("\nFirst 5 rows:")
print(data.head())