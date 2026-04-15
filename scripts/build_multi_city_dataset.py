import requests
import pandas as pd

CITIES = [
    {"region": "Stockholm", "lat": 59.3293, "lon": 18.0686},
    {"region": "Uppsala", "lat": 59.8586, "lon": 17.6389},
    {"region": "Vasteras", "lat": 59.6099, "lon": 16.5448},
    {"region": "Norrkoping", "lat": 58.5877, "lon": 16.1924},
    {"region": "Sodertalje", "lat": 59.1955, "lon": 17.6253},
]

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

HOURLY_VARS = [
    "precipitation",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "temperature_2m",
    "wind_speed_10m",
]


def fetch_city(city):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

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
    df["region"] = city["region"]
    df["lat"] = city["lat"]
    df["lon"] = city["lon"]

    df = df.ffill().bfill().fillna(0)
    return df


def main():
    frames = []

    for city in CITIES:
        print(f"Downloading {city['region']}...")
        df = fetch_city(city)
        print(f"  rows: {len(df)}")
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    full_df.to_csv("data/weather_hourly_multi_city.csv", index=False)

    print("Saved data/weather_hourly_multi_city.csv")
    print(full_df.head())
    print(full_df.shape)


if __name__ == "__main__":
    main()