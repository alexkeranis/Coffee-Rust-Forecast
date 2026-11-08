"""Download multi-year hourly climate history for the study site from Open-Meteo.

Fetches year-sized chunks in UTC (avoids the 2019 Brazilian DST transition creating
duplicate/missing local hours), then adds a local-time column for daily aggregation.

Usage:
    python -m src.ingest_climate                 # HISTORY_START -> today-7d
    python -m src.ingest_climate --start 2020-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import pandas as pd

from src import config
from src.api import get_json


def fetch_chunk(start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": config.SITE["latitude"],
        "longitude": config.SITE["longitude"],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(config.HOURLY_VARS),
        "timezone": "GMT",
    }
    payload = get_json(config.OPEN_METEO_ARCHIVE_URL, params)
    df = pd.DataFrame(payload["hourly"])
    df["time_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    df.attrs["elevation"] = payload.get("elevation")
    return df


def year_chunks(start: date, end: date):
    cur = start
    while cur <= end:
        chunk_end = min(date(cur.year, 12, 31), end)
        yield cur.isoformat(), chunk_end.isoformat()
        cur = chunk_end + timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=config.HISTORY_START)
    ap.add_argument("--end", default=(date.today() - timedelta(days=7)).isoformat())
    args = ap.parse_args()

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    print(f"archive {config.SITE['name']} {start} to {end}")

    frames, elevation = [], None
    for s, e in year_chunks(start, end):
        df = fetch_chunk(s, e)
        elevation = df.attrs.get("elevation", elevation)
        print(f"  {s} to {e}: {len(df)} rows")
        frames.append(df)
        time.sleep(1)

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("time_utc").sort_values("time_utc").reset_index(drop=True)
    df["time_local"] = df["time_utc"].dt.tz_convert(config.LOCAL_TZ)
    df = df[["time_utc", "time_local", *config.HOURLY_VARS]]

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_CLIMATE_CSV, index=False)
    print(f"wrote {config.RAW_CLIMATE_CSV.relative_to(config.ROOT)}, {len(df)} rows, "
          f"grid elevation {elevation} m")


if __name__ == "__main__":
    main()
