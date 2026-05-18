import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

start = pd.Timestamp("2022-01-01 00:00:00")
periods = 24 * 365 * 2
dates = pd.date_range(start=start, periods=periods, freq="h")

hours = dates.hour.values
days = (dates - start).days
seasonal = 10 * np.sin(2 * np.pi * days / 365.25)
diurnal = 8 * np.sin(2 * np.pi * (hours - 6) / 24)

pm25 = np.clip(25 + seasonal + diurnal + rng.normal(0, 6, periods), 1, None)
pm10 = np.clip(pm25 * 1.6 + rng.normal(0, 8, periods), 1, None)
no2 = np.clip(30 + 6 * np.sin(2 * np.pi * (hours - 8) / 24) + rng.normal(0, 5, periods), 1, None)
o3 = np.clip(40 + 15 * np.sin(2 * np.pi * (hours - 14) / 24) + rng.normal(0, 6, periods), 1, None)

df = pd.DataFrame({
    "Date": dates,
    "PM2.5": np.round(pm25, 2),
    "PM10": np.round(pm10, 2),
    "NO2": np.round(no2, 2),
    "O3": np.round(o3, 2),
})

mask = rng.random(len(df)) < 0.01
df.loc[mask, ["PM2.5", "PM10", "NO2", "O3"]] = np.nan

out = Path("data/ancona_data.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df):,} rows to {out}")
print(df.head())
