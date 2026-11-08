"""Checks on synthetic weather: run anywhere, no downloaded data needed (used in CI)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import check_hourly, vapour_pressure_deficit
from src.label_infection import label_nights, temperature_response
from src.update_forecast import (build_live_features, build_payload, issue_date, load_model,
                                 predict_proba, risk_level)

ISSUE = pd.Timestamp("2026-01-15")


def dew_point(t, rh):
    a, b = 17.27, 237.3
    g = np.log(rh / 100) + a * t / (b + t)
    return b * g / (a - g)


def synthetic_weather(start, end, t_mean=20.0, seed=0) -> pd.DataFrame:
    """Hourly weather with a daily cycle: warm dry afternoons, humid nights near 15-20 degC."""
    rng = np.random.default_rng(seed)
    time_utc = pd.date_range(start, end, freq="h", tz="UTC")
    local = time_utc.tz_convert(config.LOCAL_TZ)
    phase = 2 * np.pi * (local.hour.to_numpy() - 9) / 24
    t = t_mean + 5 * np.sin(phase) + rng.normal(0, 0.5, len(local))
    rh = np.clip(80 - 18 * np.sin(phase) + rng.normal(0, 3, len(local)), 20, 100)
    sun = np.clip(np.sin(np.pi * (local.hour.to_numpy() - 6) / 12), 0, None)
    return pd.DataFrame({
        "time_utc": time_utc,
        "time_local": local,
        "temperature_2m": t,
        "relative_humidity_2m": rh,
        "dew_point_2m": dew_point(t, rh),
        "precipitation": np.where(rng.random(len(local)) < 0.05, 1.0, 0.0),
        "shortwave_radiation": 800 * sun,
        "wind_speed_10m": np.full(len(local), 6.0),
        "cloud_cover": np.full(len(local), 50.0),
    })


def one_night(temp, wet_hours) -> pd.DataFrame:
    """A single noon-to-noon night with `wet_hours` saturated hours starting at 20:00."""
    local = pd.date_range("2025-01-10 12:00", periods=24, freq="h", tz=config.LOCAL_TZ)
    wet = np.zeros(24, bool)
    wet[8:8 + wet_hours] = True
    rh = np.where(wet, 98.0, 60.0)
    t = np.full(24, float(temp))
    return pd.DataFrame({
        "time_utc": local.tz_convert("UTC"),
        "time_local": local,
        "temperature_2m": t,
        "relative_humidity_2m": rh,
        "dew_point_2m": dew_point(t, rh),
        "precipitation": np.zeros(24),
        "shortwave_radiation": np.where((local.hour >= 6) & (local.hour < 18), 500.0, 0.0),
    })


def test_temperature_response_shape():
    r = config.INFECTION_RULE
    f = temperature_response([r["t_min"] - 1, r["t_min"], r["t_opt"], r["t_max"], r["t_max"] + 1])
    np.testing.assert_allclose(f, [0, 0, 1, 0, 0], atol=1e-12)
    grid = temperature_response(np.linspace(r["t_min"], r["t_max"], 200))
    assert grid.max() <= 1 and grid.min() >= 0


@pytest.mark.parametrize("temp, wet_hours, event", [
    (21, 10, 1),   # long warm wet night
    (21, 4, 0),    # too short
    (12, 10, 0),   # too cold for germination
])
def test_infection_rule(temp, wet_hours, event):
    night = label_nights(one_night(temp, wet_hours))
    assert len(night) == 1
    assert night["infection_event"].iloc[0] == event
    assert 0 <= night["infection_score"].iloc[0] <= 1


def test_vapour_pressure_deficit():
    t = pd.Series([20.0, 20.0, 30.0])
    vpd = vapour_pressure_deficit(t, pd.Series([100.0, 50.0, 50.0]))
    assert vpd[0] == pytest.approx(0)
    assert vpd[1] == pytest.approx(1.17, abs=0.01)
    assert vpd[2] > vpd[1]


def test_issue_date_rolls_at_local_noon():
    assert issue_date(pd.Timestamp("2026-11-08 21:00", tz="UTC")) == pd.Timestamp("2026-11-08")
    assert issue_date(pd.Timestamp("2026-11-08 12:00", tz="UTC")) == pd.Timestamp("2026-11-07")


@pytest.mark.parametrize("p, level", [(0.0, "Low"), (0.2499, "Low"), (0.25, "Moderate"),
                                      (0.5, "High"), (0.75, "Severe"), (1.0, "Severe")])
def test_risk_level(p, level):
    assert risk_level(p) == level


def test_gap_in_hourly_series_is_rejected():
    wx = synthetic_weather("2026-01-01", "2026-01-05").drop(index=30)
    with pytest.raises(ValueError, match="not contiguous"):
        check_hourly(wx, config.FORECAST_VARS)


def live_window():
    start = ISSUE - pd.Timedelta(days=config.LIVE_PAST_DAYS)
    end = ISSUE + pd.Timedelta(days=config.LIVE_FORECAST_DAYS)
    return synthetic_weather(start, end)


def test_live_inference_end_to_end():
    wx = live_window()
    X, obs = build_live_features(wx, ISSUE)
    assert list(X["lead"]) == list(range(1, config.MAX_LEAD_DAYS + 1))

    booster, meta = load_model()
    probs = predict_proba(booster, X, meta["features"])
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()

    payload = build_payload(X, probs, wx, obs, ISSUE, meta)
    json.dumps(payload, allow_nan=False)
    assert payload["issue_date"] == "2026-01-15"
    assert len(payload["days"]) == config.MAX_LEAD_DAYS
    assert payload["alert"]["max_probability"] == max(d["probability"] for d in payload["days"])


def test_live_inference_fails_when_recent_weather_is_missing():
    wx = live_window()
    cutoff = (ISSUE + pd.Timedelta(hours=11)).tz_localize(config.LOCAL_TZ)
    gap = (wx["time_local"] > cutoff - pd.Timedelta(hours=6)) & (wx["time_local"] <= cutoff)
    wx.loc[gap, "temperature_2m"] = np.nan
    with pytest.raises(RuntimeError, match="cutoff"):
        build_live_features(wx, ISSUE)
