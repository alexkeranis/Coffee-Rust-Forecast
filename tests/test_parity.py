"""Train/serve parity: the live inference path must reproduce the training feature table.

Feeds archived recent weather (truncated to the live look-back window) and the archived
forecast for a given issue date through `issue_features`, the function used live, and
compares against the rows written by `src.features`.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import MIN_HISTORY_DAYS, forecast_long_from_leads, observed_from_lead0
from src.update_forecast import issue_features

ISSUES = ["2024-06-15", "2024-12-03", "2025-03-20", "2025-08-10", "2026-01-27", "2026-10-15"]


@pytest.fixture(scope="module")
def data():
    needed = [config.RAW_PREVIOUS_RUNS_CSV, config.FEATURES_CSV]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        pytest.skip(f"data not generated yet ({', '.join(missing)}); see README")
    prev = pd.read_csv(config.RAW_PREVIOUS_RUNS_CSV)
    obs = observed_from_lead0(prev)
    fc_long = forecast_long_from_leads(prev)
    table = pd.read_csv(config.FEATURES_CSV, parse_dates=["issue", "night"])
    feats = json.loads(config.MODEL_META_PATH.read_text())["features"]
    return obs, fc_long, table, feats


def assert_same(live: pd.DataFrame, train: pd.DataFrame, feats: list[str]) -> None:
    a, b = live[feats].astype(float).to_numpy(), train[feats].astype(float).to_numpy()
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    bad_nan = [f for j, f in enumerate(feats) if (nan_a[:, j] != nan_b[:, j]).any()]
    assert not bad_nan, f"NaN pattern differs for {bad_nan}"
    close = np.isclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-3, rtol=1e-4)
    bad = [f for j, f in enumerate(feats) if not close[:, j].all()]
    assert not bad, f"values differ for {bad}"


@pytest.mark.parametrize("past_days", [config.LIVE_PAST_DAYS, MIN_HISTORY_DAYS + 1])
@pytest.mark.parametrize("issue", ISSUES)
def test_live_path_matches_training_features(data, issue, past_days):
    obs, fc_long, table, feats = data
    issue = pd.Timestamp(issue)
    live, _ = issue_features(obs, fc_long[fc_long["issue"] == issue], issue, past_days)
    train = table[table["issue"] == issue].sort_values("lead").reset_index(drop=True)
    assert len(train) == config.MAX_LEAD_DAYS
    assert_same(live, train, feats)


def test_short_history_is_rejected(data):
    obs, fc_long, _, _ = data
    issue = pd.Timestamp(ISSUES[-1])
    with pytest.raises(RuntimeError, match="recent weather"):
        issue_features(obs, fc_long[fc_long["issue"] == issue], issue, MIN_HISTORY_DAYS - 2)
