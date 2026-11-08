"""Download archived NWP forecasts by lead day from the Open-Meteo Previous Runs API.

For each hour H, column `<var>_lead{j}` holds the value forecast by the model run
issued j days before H's UTC date (j=0 is the same-day run). This lets us rebuild,
for any past issue date, exactly what the 7-day forecast looked like at the time.

Usage:
    python -m src.ingest_forecasts
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import pandas as pd

from src import config
from src.api import get_json
from src.ingest_climate import year_chunks


def _api_name(var: str, lead: int) -> str:
    return var if lead == 0 else f"{var}_previous_day{lead}"


def fetch_chunk(start: str, end: str) -> pd.DataFrame:
    names = {_api_name(v, j): f"{v}_lead{j}"
             for v in config.FORECAST_VARS for j in range(config.MAX_LEAD_DAYS + 1)}
    params = {
        "latitude": config.SITE["latitude"],
        "longitude": config.SITE["longitude"],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(names),
        "models": config.FORECAST_MODEL,
        "timezone": "GMT",
    }
    hourly = get_json(config.OPEN_METEO_PREVIOUS_RUNS_URL, params)["hourly"]
    df = pd.DataFrame({"time_utc": pd.to_datetime(hourly["time"], utc=True)})
    for api_name, col in names.items():
        df[col] = hourly.get(api_name)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=config.FORECAST_HISTORY_START)
    ap.add_argument("--end", default=(date.today() + timedelta(days=1)).isoformat())
    args = ap.parse_args()

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    print(f"previous runs {config.FORECAST_MODEL} {start} to {end}")
    frames = []
    for s, e in year_chunks(start, end):
        df = fetch_chunk(s, e)
        print(f"  {s} to {e}: {len(df)} rows")
        frames.append(df)
        time.sleep(1)

    df = pd.concat(frames, ignore_index=True).drop_duplicates("time_utc").sort_values("time_utc")
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_PREVIOUS_RUNS_CSV, index=False)
    complete = df.drop(columns="time_utc").notna().all(axis=1)
    first_full = df.loc[complete, "time_utc"].min()
    print(f"wrote {config.RAW_PREVIOUS_RUNS_CSV.relative_to(config.ROOT)}, {len(df)} rows, "
          f"all leads complete from {first_full}")


if __name__ == "__main__":
    main()
