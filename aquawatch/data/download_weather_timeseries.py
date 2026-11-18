"""
Download 10-year historical daily weather data for Chennai (2014-2024) from Open-Meteo Archive API.
No API key required.
"""

import json
import urllib.request
from pathlib import Path
import pandas as pd

API_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=13.0827&longitude=80.2707"
    "&start_date=2014-01-01&end_date=2024-01-01"
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean,surface_pressure_mean,wind_speed_10m_max,cloud_cover_mean"
    "&timezone=Asia%2FKolkata"
)


def download_chennai_weather(out_path: Path = None):
    if out_path is None:
        out_path = Path(__file__).resolve().parent / "chennai_weather_10yr.csv"

    if out_path.exists():
        df = pd.read_csv(out_path)
        print(f"Weather dataset already exists at {out_path} ({len(df)} records).")
        return df

    print(f"Downloading 10-year Chennai meteorological record from Open-Meteo...")
    req = urllib.request.Request(API_URL, headers={"User-Agent": "AquaWatch/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode())

    daily = data.get("daily", {})
    df = pd.DataFrame(daily)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Successfully saved {len(df)} daily observations (2014-2024) to {out_path}")
    return df


if __name__ == "__main__":
    download_chennai_weather()
