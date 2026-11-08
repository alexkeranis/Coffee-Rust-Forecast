"""Biological heuristic labeling of Hemileia vastatrix infection events.

Each noon-to-noon "infection night" is scanned for uninterrupted runs of hours that are
(a) wet (leaf-wetness proxy) and (b) within the germination temperature range. A night is an
**infection event** if it contains a run of >= `min_wet_hours` with >= `min_dark_hours` of
darkness. A continuous **infection score** in [0, 1] weights every hour of the best run by the
temperature response (1 at the optimum, 0 at the cardinal limits) and saturates with duration.

Usage:
    python -m src.label_infection
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

R = config.INFECTION_RULE


def load_hourly(path=config.RAW_CLIMATE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["time_local"] = df["time_utc"].dt.tz_convert(config.LOCAL_TZ)
    return df.sort_values("time_local").reset_index(drop=True)


def leaf_wetness(df: pd.DataFrame, rule: dict = R) -> pd.Series:
    """Hourly leaf-wetness proxy: high RH, small dew-point depression, or rain."""
    dpd = df["temperature_2m"] - df["dew_point_2m"]
    return (
        (df["relative_humidity_2m"] >= rule["rh_wet"])
        | (dpd <= rule["dpd_wet"])
        | (df["precipitation"] >= rule["rain_wet"])
    )


def temperature_response(t, t_min=R["t_min"], t_opt=R["t_opt"], t_max=R["t_max"]) -> np.ndarray:
    """Yan & Hunt (1999) beta function: 0 outside (t_min, t_max), 1 at t_opt."""
    t = np.asarray(t, dtype=float)
    inside = (t > t_min) & (t < t_max)
    tc = np.clip(t, t_min, t_max)
    expo = (t_opt - t_min) / (t_max - t_opt)
    f = ((t_max - tc) / (t_max - t_opt)) * ((tc - t_min) / (t_opt - t_min)) ** expo
    return np.where(inside, f, 0.0)


def infection_night(time_local: pd.Series) -> pd.Series:
    """Map each hour to the date of the evening that starts its noon-to-noon window."""
    return (time_local - pd.Timedelta(hours=config.NIGHT_START_HOUR)).dt.date


def hourly_flags(df: pd.DataFrame, rule: dict = R, by: tuple[str, ...] = ()) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in by:
        out[col] = df[col]
    out["night"] = infection_night(df["time_local"])
    out["wet"] = leaf_wetness(df, rule)
    out["dark"] = df["shortwave_radiation"] < rule["dark_wm2"]
    t = df["temperature_2m"]
    out["temp_ok"] = t.between(rule["t_min"], rule["t_max"])
    out["fav"] = out["wet"] & out["temp_ok"]
    out["f_temp"] = temperature_response(t, rule["t_min"], rule["t_opt"], rule["t_max"])
    out["temperature_2m"] = t
    out["precipitation"] = df["precipitation"]
    # run id increments on every non-favourable hour; grouping by (night, run) also splits
    # runs at the noon boundary
    out["run_id"] = (~out["fav"]).cumsum()
    return out


def night_stats(df: pd.DataFrame, rule: dict = R, by: tuple[str, ...] = ()) -> pd.DataFrame:
    """Per-night infection descriptors. `by` adds extra grouping keys (e.g. forecast issue
    date) so the same rule can be applied to many overlapping forecast versions at once.
    `df` must be sorted by (*by, time). Returns a frame indexed by (*by, night)."""
    h = hourly_flags(df, rule, by)
    keys = [*by, "night"]

    nightly = h.groupby(keys).agg(
        n_hours=("fav", "size"),
        wet_hours=("wet", "sum"),
        fav_hours=("fav", "sum"),
        rain_mm=("precipitation", "sum"),
        t_mean=("temperature_2m", "mean"),
    )

    runs = (
        h[h["fav"]]
        .groupby([*keys, "run_id"])
        .agg(
            run_hours=("fav", "size"),
            run_dark_hours=("dark", "sum"),
            run_t_mean=("temperature_2m", "mean"),
            run_eff_hours=("f_temp", "sum"),
        )
        .reset_index()
    )
    runs = runs[runs["run_dark_hours"] >= rule["min_dark_hours"]]

    # longest qualifying run per night (for the binary rule + descriptive stats)
    longest = (
        runs.sort_values("run_hours", kind="stable")
        .groupby(keys)
        .tail(1)
        .set_index(keys)[["run_hours", "run_dark_hours", "run_t_mean"]]
        .rename(columns={"run_hours": "max_run_hours",
                         "run_dark_hours": "max_run_dark_hours",
                         "run_t_mean": "max_run_t_mean"})
    )
    best_eff = runs.groupby(keys)["run_eff_hours"].max().rename("max_run_eff_hours")

    out = nightly.join(longest).join(best_eff)
    out[["max_run_hours", "max_run_dark_hours", "max_run_eff_hours"]] = (
        out[["max_run_hours", "max_run_dark_hours", "max_run_eff_hours"]].fillna(0)
    )
    out["infection_event"] = (out["max_run_hours"] >= rule["min_wet_hours"]).astype(int)
    out["infection_score"] = 1.0 - np.exp(-out["max_run_eff_hours"] / rule["score_tau_hours"])
    return out


def label_nights(df: pd.DataFrame, rule: dict = R) -> pd.DataFrame:
    """Observed-weather labels: one row per complete night, indexed by night date."""
    out = night_stats(df, rule)
    out = out[out["n_hours"] >= 23].drop(columns="n_hours")  # drop partial end windows
    out.index = pd.to_datetime(out.index)
    out.index.name = "night"
    return out


def main() -> None:
    daily = label_nights(load_hourly())

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    daily.round(4).to_csv(config.DAILY_LABELS_CSV)

    print(f"{len(daily)} nights, {daily.index.min().date()} to {daily.index.max().date()}, "
          f"event rate {daily['infection_event'].mean():.3f}")
    print(f"wrote {config.DAILY_LABELS_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
