"""Train and evaluate the multi-lead XGBoost infection forecaster (GPU if available).

Validation:
  leave-one-season-out  each coffee season (Jul..Jun) is predicted by a model trained on all
                        other seasons, with an 8-day purge around the test block so no target
                        night is shared between train and test
  forward               each season is predicted only from earlier seasons (same purge),
                        which is how the model is actually used
Inside each training fold the chronologically last 15% is used for early stopping only.

Baselines (same rows):
  climatology  monthly event rate from observed nights before the forecast archive
  persistence  observed event last night
  fc_rule      the biological rule applied to the raw NWP forecast (0/1)
  fc_score     the rule's graded score on the raw NWP forecast, used as a probability

Usage:
    python -m src.train
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src import config
from src.features import feature_columns

PURGE_DAYS = config.MAX_LEAD_DAYS + 1


def gpu_available() -> bool:
    try:
        xgb.XGBClassifier(n_estimators=1, device="cuda").fit(np.zeros((4, 1)), [0, 1, 0, 1])
        return True
    except xgb.core.XGBoostError:
        return False


def coffee_season(d: pd.Series) -> pd.Series:
    """Season label = calendar year in which the Jul..Jun season starts."""
    return np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1)


def fit_with_early_stopping(X, y, device, params=config.XGB_PARAMS):
    n_val = max(int(len(X) * 0.15), 50)
    m = xgb.XGBClassifier(**params, device=device,
                          early_stopping_rounds=config.EARLY_STOPPING_ROUNDS)
    m.fit(X.iloc[:-n_val], y.iloc[:-n_val],
          eval_set=[(X.iloc[-n_val:], y.iloc[-n_val:])], verbose=False)
    best = m.best_iteration + 1
    final = xgb.XGBClassifier(**{**params, "n_estimators": best}, device=device)
    final.fit(X, y, verbose=False)
    final.set_params(device="cpu")  # predict on host data without device-mismatch copies
    return final, best


def forward_predictions(df: pd.DataFrame, feats: list[str], device: str,
                        min_train_rows: int = 1500) -> pd.Series:
    """Expanding window: season s is scored by a model fit only on issues before s."""
    p = pd.Series(np.nan, index=df.index)
    for season in sorted(df["season"].unique()):
        test = df["season"] == season
        train = df["issue"] < df.loc[test, "issue"].min() - pd.Timedelta(days=PURGE_DAYS)
        if train.sum() < min_train_rows:
            continue
        model, _ = fit_with_early_stopping(df.loc[train, feats], df.loc[train, "y_event"], device)
        p[test] = model.predict_proba(df.loc[test, feats])[:, 1]
    return p


def scores(y, p) -> dict:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    out = {"brier": brier_score_loss(y, p), "logloss": log_loss(y, p, labels=[0, 1])}
    out["auc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    return out


def evaluate(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for lead, g in [("all", df), *df.groupby("lead")]:
        clim_brier = brier_score_loss(g["y_event"], g["p_climatology"])
        for name, col in cols.items():
            s = scores(g["y_event"], g[col])
            rows.append({"lead": lead, "model": name, **s,
                         "bss_vs_clim": 1 - s["brier"] / clim_brier})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(config.FEATURES_CSV, parse_dates=["issue", "night"])
    df = df.dropna(subset=["y_event"]).sort_values(["issue", "lead"]).reset_index(drop=True)
    df["y_event"] = df["y_event"].astype(int)
    feats = feature_columns(df)
    df["season"] = coffee_season(df["issue"])

    # climatology from observed nights before the forecast archive (no overlap with tests)
    labels = pd.read_csv(config.DAILY_LABELS_CSV, parse_dates=["night"])
    pre = labels[labels["night"] < df["issue"].min()]
    clim = pre.groupby(pre["night"].dt.month)["infection_event"].mean()
    df["p_climatology"] = df["night"].dt.month.map(clim)
    df["p_persistence"] = df["obs_event_last1"].fillna(df["p_climatology"])
    df["p_fc_rule"] = df["fc_infection_event"]
    df["p_fc_score"] = df["fc_infection_score"]

    device = "cuda" if gpu_available() else "cpu"
    print(f"xgboost {xgb.__version__} ({device}), {len(df)} rows, {len(feats)} features")
    print("rows per season:", df.groupby("season").size().to_dict())

    df["p_model"] = np.nan
    best_iters = []
    for season in sorted(df["season"].unique()):
        test = df["season"] == season
        lo = df.loc[test, "issue"].min() - pd.Timedelta(days=PURGE_DAYS)
        hi = df.loc[test, "issue"].max() + pd.Timedelta(days=PURGE_DAYS)
        train = ~test & ~df["issue"].between(lo, hi)
        model, best = fit_with_early_stopping(df.loc[train, feats], df.loc[train, "y_event"], device)
        df.loc[test, "p_model"] = model.predict_proba(df.loc[test, feats])[:, 1]
        best_iters.append(best)
        s = scores(df.loc[test, "y_event"], df.loc[test, "p_model"])
        print(f"season {season}: n={test.sum()} trees={best} "
              f"auc={s['auc']:.3f} brier={s['brier']:.3f}")

    cols = {"xgboost": "p_model", "fc_score (raw NWP + rule)": "p_fc_score",
            "fc_rule (raw NWP + rule)": "p_fc_rule", "persistence": "p_persistence",
            "climatology": "p_climatology"}
    res = evaluate(df, cols)
    pd.set_option("display.width", 140)
    print("\nleave-one-season-out, all leads")
    print(res[res["lead"] == "all"].drop(columns="lead").set_index("model").round(3).to_string())
    by_lead = res[res["model"].isin(["xgboost", "fc_score (raw NWP + rule)", "climatology"])]
    by_lead = by_lead[by_lead["lead"] != "all"]
    print("\nauc by lead")
    print(by_lead.pivot(index="model", columns="lead", values="auc").round(3).to_string())
    print("\nbrier skill vs climatology by lead")
    print(by_lead.pivot(index="model", columns="lead", values="bss_vs_clim").round(3).to_string())

    df["p_forward"] = forward_predictions(df, feats, device)
    fwd_rows = df.dropna(subset=["p_forward"])
    fwd = evaluate(fwd_rows, {"xgboost (forward)": "p_forward", **cols})
    fwd_seasons = [int(s) for s in sorted(fwd_rows["season"].unique())]
    print(f"\nforward, test seasons {fwd_seasons}, all leads")
    print(fwd[fwd["lead"] == "all"].drop(columns="lead").set_index("model").round(3).to_string())

    # final model on all labeled rows
    n_trees = int(np.median(best_iters))
    final = xgb.XGBClassifier(**{**config.XGB_PARAMS, "n_estimators": n_trees}, device=device)
    final.fit(df[feats], df["y_event"], verbose=False)
    config.MODELS.mkdir(parents=True, exist_ok=True)
    final.save_model(config.MODEL_PATH)

    gain = pd.Series(final.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    print("\ntop features by gain")
    print((gain / gain.sum()).head(12).round(3).to_string())

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "xgboost_version": xgb.__version__,
        "device": device,
        "forecast_model": config.FORECAST_MODEL,
        "site": config.SITE,
        "features": feats,
        "n_estimators": n_trees,
        "params": config.XGB_PARAMS,
        "infection_rule": config.INFECTION_RULE,
        "train_rows": len(df),
        "train_issue_range": [str(df["issue"].min().date()), str(df["issue"].max().date())],
        "monthly_climatology": {int(k): round(float(v), 4) for k, v in clim.items()},
        "cv": json.loads(res.round(4).to_json(orient="records")),
        "cv_forward_seasons": fwd_seasons,
        "cv_forward": json.loads(fwd.round(4).to_json(orient="records")),
    }
    config.MODEL_META_PATH.write_text(json.dumps(meta, indent=2))
    out_cols = ["issue", "night", "lead", "season", "y_event", *[c for c in df if c.startswith("p_")]]
    df[out_cols].to_csv(config.OOF_PREDICTIONS_CSV, index=False, float_format="%.4f")
    print(f"\nwrote {config.MODEL_PATH.relative_to(config.ROOT)} ({n_trees} trees)")


if __name__ == "__main__":
    main()
