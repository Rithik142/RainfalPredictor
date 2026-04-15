import json
import pandas as pd

INPUT_JSON = "data/stockholm_raw.json"
OUTPUT_CSV = "data/weather_hourly.csv"

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

hourly = data["hourly"]

df = pd.DataFrame({
    "time": hourly["time"],
    "precipitation": hourly["precipitation"],
    "relative_humidity_2m": hourly["relative_humidity_2m"],
    "dew_point_2m": hourly["dew_point_2m"],
    "pressure_msl": hourly["pressure_msl"],
    "temperature_2m": hourly["temperature_2m"],
    "wind_speed_10m": hourly["wind_speed_10m"],
})

df["time"] = pd.to_datetime(df["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
df = df.ffill().bfill().fillna(0)

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved CSV to {OUTPUT_CSV}")
print(df.head())