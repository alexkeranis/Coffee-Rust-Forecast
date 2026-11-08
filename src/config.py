"""Central project configuration: study site, data paths, and API settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"

# Study site: Varginha, South Minas Gerais (Sul de Minas) - major arabica hub,
# humid highland climate (~950 m) with recurrent coffee leaf rust pressure.
SITE = {
    "name": "Varginha",
    "state": "MG",
    "latitude": -21.5514,
    "longitude": -45.4303,
}
LOCAL_TZ = "America/Sao_Paulo"

# Reanalysis history, used for labels (the archive lags real time by ~5 days).
HISTORY_START = "2019-01-01"

# Hourly reanalysis variables.
HOURLY_VARS = [
    "temperature_2m",         # degC
    "relative_humidity_2m",   # %
    "dew_point_2m",           # degC - dew-point depression is a leaf-wetness proxy
    "precipitation",          # mm
    "shortwave_radiation",    # W/m2 - darkness favours germination
    "vapour_pressure_deficit",  # kPa
    "wind_speed_10m",         # km/h - wind dries leaves
    "cloud_cover",            # %
]

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

RAW_CLIMATE_CSV = DATA_RAW / f"{SITE['name'].lower()}_hourly_climate.csv"

# Archived NWP forecasts by lead day ("previous runs"). Lead-specific coverage for all
# variables starts ~2024-03. The model is pinned so training and live inference match.
# Lead 0 doubles as the recent-weather series for obs_* features (see features.py).
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_MODEL = "gfs_seamless"
FORECAST_HISTORY_START = "2024-01-01"
MAX_LEAD_DAYS = 7
FORECAST_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "shortwave_radiation",
    "wind_speed_10m",
    "cloud_cover",
]
RAW_PREVIOUS_RUNS_CSV = DATA_RAW / f"{SITE['name'].lower()}_previous_runs_{FORECAST_MODEL}.csv"
FEATURES_CSV = DATA_PROCESSED / f"{SITE['name'].lower()}_training_features.csv"
OOF_PREDICTIONS_CSV = DATA_PROCESSED / "oof_predictions.csv"

MODEL_PATH = MODELS / "coffee_rust_xgb.json"     # native XGBoost format (version-portable)
MODEL_META_PATH = MODELS / "model_meta.json"     # feature list, params, CV metrics

XGB_PARAMS = {
    "n_estimators": 2000,        # upper bound; actual count set by early stopping
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
}
EARLY_STOPPING_ROUNDS = 100

# Live inference
RISK_JSON = DOCS / "rust_risk.json"
LIVE_PAST_DAYS = 45        # the API keeps ~42 days of GFS history; features need 33
LIVE_FORECAST_DAYS = 9     # night 7 ends at local noon on D+7
# Alert levels on the calibrated probability of an infection night (lower bounds).
RISK_LEVELS = [(0.0, "Low"), (0.25, "Moderate"), (0.5, "High"), (0.75, "Severe")]
DAILY_LABELS_CSV = DATA_PROCESSED / f"{SITE['name'].lower()}_daily_infection_labels.csv"

# Hemileia vastatrix infection heuristic.
# Urediniospores need liquid water on the leaf to germinate, germination is inhibited by
# light, and the temperature response is roughly 15-28 degC with an optimum near 21-23 degC.
# Minimum wetness for successful infection is ~6 h (longer under suboptimal temps).
# Refs: Nutman & Roberts (1963); Kushalappa et al. (1983); Avelino et al. (2004/2015).
# Leaf wetness is not measured, so it is estimated from RH / dew-point depression / rain
# (RH>=90% and DPD<=2 degC are common LW proxies, e.g. Sentelhas et al. 2008).
INFECTION_RULE = {
    # leaf-wetness proxy (an hour is "wet" if ANY holds)
    "rh_wet": 90.0,            # %
    "dpd_wet": 2.0,            # degC, T - Td
    "rain_wet": 0.2,           # mm/h
    # cardinal temperatures for germination (beta response, Yan & Hunt 1999)
    "t_min": 15.0,
    "t_opt": 22.0,
    "t_max": 28.0,
    # darkness: Open-Meteo shortwave is the preceding-hour mean
    "dark_wm2": 10.0,          # W/m2
    # event definition
    "min_wet_hours": 6,        # consecutive wet + in-range hours
    "min_dark_hours": 4,       # of which at least this many in darkness
    # graded score: 1 - exp(-effective_hours / tau)
    "score_tau_hours": 8.0,
}

# Biological "day" runs noon-to-noon so an overnight wet period is never split at
# midnight; the label is assigned to the date of the evening it starts.
NIGHT_START_HOUR = 12
