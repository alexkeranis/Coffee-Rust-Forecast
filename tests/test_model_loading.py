"""The CI inference path (native Booster, no scikit-learn) must match the training wrapper."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.update_forecast import load_model, predict_proba


def test_booster_matches_sklearn_wrapper():
    xgb = pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    if not config.FEATURES_CSV.exists():
        pytest.skip("feature table not generated yet; run python -m src.features")
    feats = json.loads(config.MODEL_META_PATH.read_text())["features"]
    X = pd.read_csv(config.FEATURES_CSV).dropna(subset=["y_event"]).sample(500, random_state=0)

    clf = xgb.XGBClassifier()
    clf.load_model(config.MODEL_PATH)
    clf.set_params(device="cpu")
    expected = clf.predict_proba(X[feats])[:, 1]

    booster, _ = load_model()
    got = predict_proba(booster, X, feats)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)
    assert ((got >= 0) & (got <= 1)).all()
