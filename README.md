# Minas Gerais Coffee Rust Pathogen Forecaster

Seven-night forecast of *Hemileia vastatrix* (coffee leaf rust) infection risk for
**Varginha, South Minas Gerais**: hourly GFS weather, labels derived from the pathogen's biology,
and an XGBoost classifier. A GitHub Actions job refreshes the forecast daily and after every push,
and a static dashboard shows it: **https://alexkeranis.github.io/Coffee-Rust-Forecast/**

## Setup (Windows, NVIDIA GPU)

```powershell
python -m venv .venv --prompt coffee-rust
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Install only one of the two requirements files. `requirements.txt` is the lean CPU runtime for
the scheduled job (`xgboost-cpu`, no scikit-learn). `requirements-dev.txt` is the full local
environment (CUDA XGBoost, scikit-learn, notebooks).

## Reproduce from scratch

Data is not committed. It is rebuilt from the free Open-Meteo APIs (no key needed):

```powershell
python -m src.ingest_climate        # data/raw: hourly reanalysis history (labels)
python -m src.ingest_forecasts      # data/raw: archived GFS forecasts by lead day
python -m src.label_infection       # data/processed: nightly infection labels
python -m src.features              # data/processed: (issue, lead) feature table
python -m src.train                 # models/: XGBoost + cross-validation report
pytest                              # synthetic, train/serve parity and model-loading tests
python -m src.update_forecast       # docs/rust_risk.json
```

The archives grow over time, so a retrain will differ slightly. To match the committed model,
pass `--end` to both ingest scripts using `train_issue_range` from `models/model_meta.json`.

## Biological labels

No farm outbreak records are public, so labels come from what *H. vastatrix* urediniospores
need to germinate and infect:

| Requirement | Biology | Hourly proxy |
|---|---|---|
| Liquid water on the leaf | Spores only germinate in free water | RH >= 90% **or** T - Td <= 2 degC **or** rain >= 0.2 mm |
| Temperature | Germination ~15-28 degC, optimum ~22 degC | Beta response (Yan & Hunt 1999), 0 at limits, 1 at optimum |
| Darkness | Light inhibits germination | Shortwave radiation < 10 W/m2 |
| Duration | ~6 h minimum wetness for infection | Uninterrupted wet + in-range run |

Nights run noon to noon, so an overnight wet spell is never split at midnight.
`infection_event` = 1 when a night has a favourable run of >= 6 h with >= 4 h of darkness.
`infection_score` = `1 - exp(-E / 8)`, where `E` is the temperature-weighted wet hours of the best
run. Thresholds are in `INFECTION_RULE` in `src/config.py`. Labels use Open-Meteo reanalysis.

Refs: Nutman & Roberts (1963); Kushalappa et al. (1983); Avelino et al. (2004, 2015);
Sentelhas et al. (2008).

## Forecast model

One training row per (issue date D, lead k = 1..7), with the target the label of night D+k-1.
Features use only what was known on D:

- `fc_*`: the infection rule and summaries (wet hours, longest favourable run, rain, RH,
  dew-point depression, wind, cloud) on the GFS forecast issued for that night and its
  neighbours, recovered from Open-Meteo's archive of past runs.
- `obs_*`: conditions up to 11:00 local on D (rolling 24 h to 14 d wetness, rain, temperature,
  humidity, recent infection nights).
- `lead`, and day of year (sin/cos).

Because it trains on archived forecasts rather than observed weather, the model learns to correct
GFS instead of re-learning the labelling rule. `obs_*` come from the GFS lead-0 series, not
reanalysis, because that is all the live job can get (the forecast API's `past_days`). The two
sources disagree on about 15% of nightly labels here, so mixing them would shift the features
between training and live use. Reanalysis is used only for labels.

### Validation

Leave-one-season-out over coffee seasons (Jul-Jun), with an 8-day purge around each test season.
Out-of-fold, all leads, issue dates Jan 2024 to Oct 2026 (7,088 rows):

| Model | AUC | Brier | Brier skill vs climatology |
|---|---|---|---|
| **XGBoost** | **0.910** | **0.114** | **+0.36** |
| Rule on raw GFS forecast (graded score) | 0.884 | 0.130 | +0.27 |
| Rule on raw GFS forecast (0/1) | 0.830 | 0.172 | +0.03 |
| Monthly climatology | 0.789 | 0.178 | 0 |
| Persistence (last night) | 0.705 | 0.284 | -0.60 |

AUC falls from 0.92 at 1 day to 0.89 at 7 days, and XGBoost beats the raw-forecast rule at every
lead. Probabilities are well calibrated, though slightly overconfident above 0.9 (0.94 predicted
against 0.88 observed).

A forward check predicts each season only from earlier ones, as in real use. On 2025/26 and
2026/27 it scores AUC 0.930 and Brier skill +0.39, against 0.931 and +0.42 from
leave-one-season-out on the same rows. Training on later seasons therefore does not inflate the
scores. Both results are in `models/model_meta.json`.

## Live forecast and dashboard

`src.update_forecast` fetches recent and forecast weather (`gfs_seamless`, the model used in
training), rebuilds the same features, scores them on CPU and writes `docs/rust_risk.json`. The
file holds the alert summary, the 7 nights (probability, level, raw-rule score, wet hours,
temperature, RH, rain), hourly curves, the last 14 nights' labels and model info. Alert levels:
Low < 25% <= Moderate < 50% <= High < 75% <= Severe.

The job refuses to write a doubtful forecast. It stops if the weather has gaps or ends before the
11:00 cutoff, if there is under 33 days of history (Open-Meteo keeps about 42), if a night,
feature or probability is invalid. The previous forecast then stays up, and the dashboard flags
one older than three days. `tests/test_parity.py` checks that the live code path reproduces the
training features exactly, with both the full and the minimum look-back.

`docs/` is a static Chart.js site with no build step. Preview it locally with caching disabled:

```powershell
python scripts/serve_dashboard.py   # http://localhost:8000
```

## Automation

- `pathogen_inference.yml` runs daily at 21:00 UTC (18:00 in Minas Gerais), on every push to
  `main`, and on demand. It retries once after a failure and commits `docs/rust_risk.json` when
  it changes, so the commit history doubles as an archive of issued forecasts.
- `tests.yml` runs pyflakes and pytest on every push and pull request.

The model is never retrained in CI. Retrain locally and commit `models/`.

## Limitations

- Labels mark weather favourable to infection, not observed disease. Inoculum, host
  susceptibility, crop load and fungicide use are not modelled.
- Covers one site (a single Open-Meteo grid cell). Leaf wetness is inferred, not measured.
- About two and a half seasons of archived forecasts, so year-to-year variability is thinly
  sampled.

## Layout

```
src/
  config.py            site, thresholds, paths, model settings
  api.py               Open-Meteo HTTP with retry
  ingest_climate.py    hourly reanalysis history (archive API)
  ingest_forecasts.py  archived GFS forecasts by lead day (previous-runs API)
  label_infection.py   biological heuristic labels
  features.py          (issue, lead) feature table, shared by training and inference
  train.py             XGBoost, leave-one-season-out and forward validation
  update_forecast.py   live inference -> docs/rust_risk.json
  plot_style.py        matplotlib style matching the dashboard
notebooks/             01 weather -> labels; 02 features, GPU timing, skill, calibration
tests/                 synthetic pipeline, train/serve parity, model loading
models/                coffee_rust_xgb.json + model_meta.json
docs/                  dashboard (GitHub Pages)
```

## Data and license

Weather data from [Open-Meteo](https://open-meteo.com/) (Historical Weather, Previous Runs and
Forecast APIs) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). GFS forecasts are
from NOAA. Code is released under the [MIT License](LICENSE).
