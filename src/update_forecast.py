"""Live 7-day coffee rust infection forecast -> docs/rust_risk.json.

Fetches recent + forecast hourly weather from Open-Meteo (same NWP model as training),
builds the exact training features, scores them with the saved XGBoost model on CPU, and
writes a JSON payload for the static dashboard. Any inconsistency in the inputs raises, so a
bad run fails the workflow and leaves the previous forecast in place.

Issue date D: today's local date once past local noon, otherwise yesterday (so night 1 is
always the noon-to-noon window currently in progress or starting today).

Usage:
    python -m src.update_forecast
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

from src import config
from src.api import get_json
from src.features import CUTOFF_HOUR, MIN_HISTORY_DAYS, build_feature_table
from src.label_infection import hourly_flags, label_nights, to_local


def fetch_live() -> pd.DataFrame:
    params = {
        "latitude": config.SITE["latitude"],
        "longitude": config.SITE["longitude"],
        "hourly": ",".join(config.FORECAST_VARS),
        "models": config.FORECAST_MODEL,
        "past_days": config.LIVE_PAST_DAYS,
        "forecast_days": config.LIVE_FORECAST_DAYS,
        "timezone": "GMT",
    }
    hourly = get_json(config.OPEN_METEO_FORECAST_URL, params)["hourly"]
    return to_local(pd.DataFrame(hourly).rename(columns={"time": "time_utc"}))


def issue_date(now_utc: pd.Timestamp) -> pd.Timestamp:
    local = now_utc.tz_convert(config.LOCAL_TZ)
    return (local - pd.Timedelta(hours=12)).normalize().tz_localize(None)


def risk_level(p: float) -> str:
    return [name for lo, name in config.RISK_LEVELS if p >= lo][-1]


def load_model():
    """Native Booster (no scikit-learn needed on the CI runner)."""
    meta = json.loads(config.MODEL_META_PATH.read_text())
    booster = xgb.Booster()
    booster.load_model(config.MODEL_PATH)
    booster.set_param({"device": "cpu"})
    return booster, meta


def predict_proba(booster: xgb.Booster, X: pd.DataFrame, features: list[str]):
    """P(infection event); binary:logistic boosters output probabilities directly."""
    return booster.predict(xgb.DMatrix(X[features], feature_names=features))


def issue_features(obs: pd.DataFrame, fc_long: pd.DataFrame, issue: pd.Timestamp,
                   past_days: int = config.LIVE_PAST_DAYS):
    """Feature rows for one issue date from hourly history and that issue's forecast.
    Hours after the cutoff are discarded. Shared by live inference and the parity test."""
    cutoff = (issue + pd.Timedelta(hours=CUTOFF_HOUR)).tz_localize(config.LOCAL_TZ)
    start = cutoff - pd.Timedelta(days=past_days)
    obs = obs[(obs["time_local"] > start) & (obs["time_local"] <= cutoff)]
    obs = obs.dropna(subset=config.FORECAST_VARS).reset_index(drop=True)
    if obs.empty or obs["time_local"].iloc[-1] != cutoff:
        raise RuntimeError(f"Recent weather does not reach the {cutoff} cutoff.")
    span = obs["time_local"].iloc[-1] - obs["time_local"].iloc[0]
    if span < pd.Timedelta(days=MIN_HISTORY_DAYS):
        raise RuntimeError(f"Only {span} of recent weather; need {MIN_HISTORY_DAYS} days.")
    X = build_feature_table(obs, fc_long)
    X = X[X["issue"] == issue].sort_values("lead").reset_index(drop=True)
    if len(X) != config.MAX_LEAD_DAYS:
        raise RuntimeError(f"Expected {config.MAX_LEAD_DAYS} forecast nights, got {len(X)}.")
    obs_cols = [c for c in X if c.startswith("obs_")]
    if X[obs_cols].isna().any().any():
        raise RuntimeError("Antecedent features are incomplete.")
    return X, obs


def build_live_features(wx: pd.DataFrame, issue: pd.Timestamp):
    fc_long = wx[["time_utc", "time_local", *config.FORECAST_VARS]].dropna().assign(issue=issue)
    return issue_features(wx, fc_long, issue)


def r(x, nd=1):
    return None if pd.isna(x) else round(float(x), nd)


def build_payload(X, probs, wx, obs, issue, meta) -> dict:
    days = []
    for row, p in zip(X.itertuples(index=False), probs):
        days.append({
            "night": row.night.date().isoformat(),
            "lead": int(row.lead),
            "probability": round(float(p), 3),
            "level": risk_level(p),
            "rule_score": r(row.fc_infection_score, 3),
            "rule_event": int(row.fc_infection_event),
            "wet_hours": int(row.fc_wet_hours),
            "favourable_hours": int(row.fc_fav_hours),
            "longest_wet_run_hours": int(row.fc_max_run_hours),
            "t_mean": r(row.fc_t_mean),
            "t_min": r(row.fc_t_min),
            "t_max": r(row.fc_t_max),
            "rh_mean": r(row.fc_rh_mean, 0),
            "rain_mm": r(row.fc_rain_mm),
        })
    peak = max(days, key=lambda d: d["probability"])

    # hourly curves: 2 days of context before night 1 through the end of night 7
    start = (issue - pd.Timedelta(days=2)).tz_localize(config.LOCAL_TZ)
    end = (issue + pd.Timedelta(days=config.MAX_LEAD_DAYS, hours=12)).tz_localize(config.LOCAL_TZ)
    h = wx[(wx["time_local"] >= start) & (wx["time_local"] < end)].reset_index(drop=True)
    flags = hourly_flags(h)
    cutoff = obs["time_local"].max()

    recent = label_nights(obs).tail(14)

    auc_by_lead = {str(c["lead"]): c["auc"] for c in meta["cv"]
                   if c["model"] == "xgboost" and c["lead"] != "all"}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "issue_date": issue.date().isoformat(),
        "site": {**config.SITE, "timezone": config.LOCAL_TZ},
        "alert": {
            "level": peak["level"],
            "max_probability": peak["probability"],
            "peak_night": peak["night"],
            "nights_high_or_above": sum(d["probability"] >= 0.5 for d in days),
        },
        "levels": [{"min": lo, "name": name} for lo, name in config.RISK_LEVELS],
        "days": days,
        "hourly": {
            "time": h["time_local"].dt.strftime("%Y-%m-%dT%H:%M").tolist(),
            "observed_until": cutoff.strftime("%Y-%m-%dT%H:%M"),
            "temperature_2m": h["temperature_2m"].round(1).tolist(),
            "relative_humidity_2m": h["relative_humidity_2m"].round(0).tolist(),
            "precipitation": h["precipitation"].round(1).tolist(),
            "leaf_wet": flags["wet"].astype(int).tolist(),
            "favourable": flags["fav"].astype(int).tolist(),
        },
        "recent_nights": [
            {"night": d.date().isoformat(), "event": int(e), "score": round(float(s), 3)}
            for d, e, s in zip(recent.index, recent["infection_event"], recent["infection_score"])
        ],
        "model": {
            "type": "XGBoost classifier",
            "forecast_model": meta["forecast_model"],
            "trained_at": meta["trained_at"],
            "train_issue_range": meta["train_issue_range"],
            "cv_auc_by_lead": auc_by_lead,
        },
    }


def main() -> None:
    model, meta = load_model()
    issue = issue_date(pd.Timestamp.now(tz="UTC"))
    wx = fetch_live()
    X, obs = build_live_features(wx, issue)

    missing = [f for f in meta["features"] if f not in X.columns]
    if missing:
        raise RuntimeError(f"Live features missing: {missing}")
    probs = predict_proba(model, X, meta["features"])
    if not (np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()):
        raise RuntimeError(f"Invalid probabilities: {probs}")

    payload = build_payload(X, probs, wx, obs, issue, meta)
    config.DOCS.mkdir(parents=True, exist_ok=True)
    config.RISK_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False))

    print(f"issue {payload['issue_date']}, alert {payload['alert']['level']}")
    for d in payload["days"]:
        print(f"  {d['night']}  p={d['probability']:.3f}  {d['level']}")
    print(f"wrote {config.RISK_JSON.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
