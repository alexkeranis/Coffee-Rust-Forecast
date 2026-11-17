"""Feature engineering for multi-lead rust-infection forecasting.

One row = (issue date D, lead k in 1..7). Target = observed infection event on night D+k-1
(night k=1 is the noon-to-noon window starting at local noon on D).

Two feature families, both available at issue time:
  obs_*  antecedent conditions from recent weather up to the cutoff (11:00 local on D)
  fc_*   the infection rule + summary stats applied to the NWP forecast for the target night
plus calendar terms and the lead itself.

"Recent weather" is the GFS lead-0 series (each hour's same-day model value), not reanalysis:
the live job only has the forecast API's past_days, which is that same series, so training
on it keeps obs_* features identically distributed in training and live. Reanalysis is used
only for the labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.label_infection import R, infection_night, leaf_wetness, night_stats, to_local

CUTOFF_HOUR = 11  # last observed local hour before night k=1 begins at noon
# Live inference only sees ~6 weeks of past weather, so cap this to keep train/live consistent.
DAYS_SINCE_EVENT_CAP = 30
# Shortest history that yields complete obs_* features: 14-day rolling windows and the
# 30-day days-since-event cap, plus a couple of days of margin.
MIN_HISTORY_DAYS = DAYS_SINCE_EVENT_CAP + 3


def _run_length(mask: pd.Series) -> pd.Series:
    """Length of the current run of True values ending at each row."""
    return mask.groupby((~mask).cumsum()).cumsum()


def vapour_pressure_deficit(t: pd.Series, rh: pd.Series) -> pd.Series:
    """VPD in kPa from air temperature (degC) and RH (%), Tetens saturation pressure."""
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return es * (1 - rh / 100)


def check_hourly(df: pd.DataFrame, cols: list[str]) -> None:
    """Rolling features count rows, so the series must be gap-free and fully populated."""
    if df[cols].isna().any().any():
        raise ValueError("Hourly series has missing values.")
    step = df["time_utc"].diff().dropna()
    if (step != pd.Timedelta("1h")).any():
        first = df["time_utc"].iloc[1:][step.to_numpy() != pd.Timedelta("1h")].iloc[0]
        raise ValueError(f"Hourly series is not contiguous (first break at {first}).")


def observed_from_lead0(prev_runs: pd.DataFrame) -> pd.DataFrame:
    """Hourly 'observed' series from the lead-0 columns of the previous-runs archive."""
    cols = {f"{v}_lead0": v for v in config.FORECAST_VARS}
    df = prev_runs[["time_utc", *cols]].rename(columns=cols)
    df = to_local(df).dropna().sort_values("time_utc").reset_index(drop=True)
    check_hourly(df, config.FORECAST_VARS)
    return df


def antecedent_features(obs: pd.DataFrame, rule: dict = R) -> pd.DataFrame:
    """Observed-weather features at each issue date's cutoff. `obs` must be contiguous hourly.
    Returns a frame indexed by issue date (datetime64, normalized)."""
    obs = obs.sort_values("time_local").reset_index(drop=True)
    check_hourly(obs, config.FORECAST_VARS)
    t, rh = obs["temperature_2m"], obs["relative_humidity_2m"]
    wet = leaf_wetness(obs, rule)
    fav = wet & t.between(rule["t_min"], rule["t_max"])
    humid = rh >= rule["rh_wet"]
    dpd = t - obs["dew_point_2m"]

    f = pd.DataFrame(index=obs.index)
    for h in (24, 72, 168):
        f[f"obs_wet_hours_{h}h"] = wet.rolling(h).sum()
        f[f"obs_fav_hours_{h}h"] = fav.rolling(h).sum()
    for h in (24, 72, 168, 336):
        f[f"obs_rain_{h}h"] = obs["precipitation"].rolling(h).sum()
    f["obs_t_mean_24h"] = t.rolling(24).mean()
    f["obs_temp_variance_24h"] = t.rolling(24).var()
    f["obs_t_min_24h"] = t.rolling(24).min()
    f["obs_t_max_24h"] = t.rolling(24).max()
    f["obs_t_mean_168h"] = t.rolling(168).mean()
    f["obs_rh_mean_24h"] = rh.rolling(24).mean()
    f["obs_rh_mean_168h"] = rh.rolling(168).mean()
    f["obs_dpd_mean_24h"] = dpd.rolling(24).mean()
    f["obs_vpd_mean_72h"] = vapour_pressure_deficit(t, rh).rolling(72).mean()
    f["obs_radiation_mean_72h"] = obs["shortwave_radiation"].rolling(72).mean()
    f["obs_consecutive_humid_hours"] = _run_length(humid)
    f["obs_max_consecutive_wet_72h"] = _run_length(wet).rolling(72).max()

    at_cutoff = obs["time_local"].dt.hour == CUTOFF_HOUR
    f = f[at_cutoff]
    f.index = pd.to_datetime(obs.loc[at_cutoff, "time_local"].dt.date)
    f.index.name = "issue"

    # recent infection nights (strictly before night k=1)
    nights = night_stats(obs, rule)
    nights = nights[nights["n_hours"] >= 23]
    nights.index = pd.to_datetime(nights.index)
    # extend one day past the last complete night: at live time the issue date's own night
    # has not started yet, but its history features (shifted from earlier nights) must exist
    nights = nights.reindex(pd.date_range(nights.index[0], nights.index[-1] + pd.Timedelta(days=1)))
    ev, sc = nights["infection_event"], nights["infection_score"]
    hist = pd.DataFrame(index=nights.index)
    hist["obs_event_last1"] = ev.shift(1)
    hist["obs_score_last1"] = sc.shift(1)
    hist["obs_events_last7"] = ev.shift(1).rolling(7).sum()
    hist["obs_events_last14"] = ev.shift(1).rolling(14).sum()
    hist["obs_score_mean_last7"] = sc.shift(1).rolling(7).mean()
    last_event = nights.index.to_series().where(ev == 1).ffill().shift(1)
    dse = (nights.index.to_series() - last_event).dt.days
    has_history = nights.index >= nights.index[0] + pd.Timedelta(days=DAYS_SINCE_EVENT_CAP)
    hist["obs_days_since_event"] = (dse.clip(upper=DAYS_SINCE_EVENT_CAP)
                                    .where(dse.notna() | ~has_history, DAYS_SINCE_EVENT_CAP))

    return f.join(hist, how="left")


def forecast_long_from_leads(prev_runs: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide `<var>_lead{j}` archive into one row per (issue date, forecast hour)."""
    prev_runs = prev_runs.copy()
    prev_runs["time_utc"] = pd.to_datetime(prev_runs["time_utc"], utc=True)
    frames = []
    for j in range(config.MAX_LEAD_DAYS + 1):
        cols = {f"{v}_lead{j}": v for v in config.FORECAST_VARS}
        fj = prev_runs[["time_utc", *cols]].rename(columns=cols)
        fj["issue"] = (fj["time_utc"].dt.tz_localize(None).dt.normalize()
                       - pd.Timedelta(days=j))
        frames.append(fj)
    long = pd.concat(frames, ignore_index=True).dropna()
    return to_local(long)


def forecast_features(fc_long: pd.DataFrame, rule: dict = R) -> pd.DataFrame:
    """Apply the infection rule + summaries to each forecast night. `fc_long` has columns
    issue, time_utc, time_local and FORECAST_VARS. Returns rows indexed by (issue, lead)."""
    fc = fc_long.copy()
    fc["night"] = pd.to_datetime(infection_night(fc["time_local"]))
    lead = (fc["night"] - fc["issue"]).dt.days + 1
    fc = fc[(lead >= 1) & (lead <= config.MAX_LEAD_DAYS)]
    fc = fc.sort_values(["issue", "time_utc"]).reset_index(drop=True)

    stats = night_stats(fc, rule, by=("issue",))
    stats.index = stats.index.set_levels(pd.to_datetime(stats.index.levels[1]), level=1)
    stats = stats[stats["n_hours"] == 24].drop(columns="n_hours")

    dpd = fc["temperature_2m"] - fc["dew_point_2m"]
    extra = (
        fc.assign(dpd=dpd)
        .groupby(["issue", "night"])
        .agg(rh_mean=("relative_humidity_2m", "mean"),
             dpd_min=("dpd", "min"),
             dpd_mean=("dpd", "mean"),
             t_min=("temperature_2m", "min"),
             t_max=("temperature_2m", "max"),
             wind_mean=("wind_speed_10m", "mean"),
             cloud_mean=("cloud_cover", "mean"),
             radiation_sum=("shortwave_radiation", "sum"))
    )
    out = stats.join(extra, how="left").add_prefix("fc_").reset_index()
    out["lead"] = (out["night"] - out["issue"]).dt.days + 1

    # neighbouring forecast nights help absorb timing errors in the NWP
    out = out.sort_values(["issue", "lead"])
    g = out.groupby("issue")["fc_infection_score"]
    out["fc_score_prev_night"] = g.shift(1)
    out["fc_score_next_night"] = g.shift(-1)
    out["fc_events_in_week"] = out.groupby("issue")["fc_infection_event"].transform("sum")
    return out.set_index(["issue", "lead"])


def build_feature_table(obs: pd.DataFrame, fc_long: pd.DataFrame, rule: dict = R) -> pd.DataFrame:
    fc = forecast_features(fc_long, rule).reset_index()
    ante = antecedent_features(obs, rule)
    X = fc.merge(ante, left_on="issue", right_index=True, how="left")
    doy = X["night"].dt.day_of_year
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return X


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.startswith(("fc_", "obs_")) or c in ("lead", "doy_sin", "doy_cos")]


def main() -> None:
    from src.label_infection import label_nights, load_hourly

    prev = pd.read_csv(config.RAW_PREVIOUS_RUNS_CSV)
    X = build_feature_table(observed_from_lead0(prev), forecast_long_from_leads(prev))

    labels = label_nights(load_hourly())[["infection_event", "infection_score"]]
    labels = labels.rename(columns={"infection_event": "y_event", "infection_score": "y_score"})
    X = X.merge(labels, left_on="night", right_index=True, how="left")

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    X.to_csv(config.FEATURES_CSV, index=False, float_format="%.4f")
    n_lab = X["y_event"].notna().sum()
    print(f"{len(X)} rows, {len(feature_columns(X))} features, {n_lab} labeled, "
          f"issues {X.issue.min().date()} to {X.issue.max().date()}")
    print(f"wrote {config.FEATURES_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
