# Minas Gerais Coffee Rust Pathogen Forecaster

Seven-night forecast of *Hemileia vastatrix* (coffee leaf rust) infection risk for
**Varginha, South Minas Gerais**, built from hourly weather, biologically derived labels and an
XGBoost classifier. A GitHub Actions job refreshes the forecast daily and a static dashboard in
`docs/` displays it.

## Setup (Windows, NVIDIA GPU)

```powershell
python -m venv .venv --prompt coffee-rust
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

`requirements.txt` is the lean CPU-only runtime used by the scheduled job (`xgboost-cpu`, no
scikit-learn). `requirements-dev.txt` is the full local environment (CUDA XGBoost, scikit-learn,
notebooks). Install one or the other, not both.

## Reproduce from scratch

Data files are not committed. They are rebuilt from the free Open-Meteo APIs (no key needed):

```powershell
python -m src.ingest_climate        # data/raw: hourly reanalysis history (labels)
python -m src.ingest_forecasts      # data/raw: archived GFS forecasts by lead day
python -m src.label_infection       # data/processed: nightly infection labels
python -m src.features              # data/processed: (issue, lead) feature table
python -m src.train                 # models/: XGBoost + cross-validation report
pytest                              # synthetic tests + train/serve parity tests
python -m src.update_forecast       # docs/rust_risk.json
```

Re-downloading later extends the archives, so a retrained model will differ slightly. To match
the committed model, pass `--end` to both ingest scripts using `train_issue_range` in
`models/model_meta.json`.

## Biological labels

Outbreak records from private farms are not available, so ground truth is generated from the
environmental requirements of *H. vastatrix* urediniospore germination and infection:

| Requirement | Biology | Proxy used (hourly) |
|---|---|---|
| Liquid water on the leaf | Spores only germinate in free water | RH >= 90% **or** T - Td <= 2 degC **or** rain >= 0.2 mm |
| Temperature | Germination ~15-28 degC, optimum ~22 degC | Beta response (Yan & Hunt 1999), 0 at limits, 1 at optimum |
| Darkness | Light inhibits germination | Shortwave radiation < 10 W/m2 |
| Duration | ~6 h minimum wetness for infection | Uninterrupted wet + in-range run |

Nights run **noon to noon** so an overnight wet period is never split at midnight.

- `infection_event` = 1 if the night has a favourable run of >= 6 h including >= 4 h of darkness.
- `infection_score` = `1 - exp(-E / 8)`, where `E` sums the temperature response over the best
  run (temperature-weighted wet hours).

Thresholds live in `src/config.py` (`INFECTION_RULE`). Labels are computed on Open-Meteo's
reanalysis archive.
Refs: Nutman & Roberts (1963); Kushalappa et al. (1983); Avelino et al. (2004, 2015);
Sentelhas et al. (2008).

## Forecast model

Each training row is one (issue date D, lead k = 1..7) pair, and the target is the label of
night D+k-1. The features are only what was available on D:

- `fc_*`: the infection rule and summaries (wet hours, longest favourable run, rain, RH,
  dew-point depression, wind, cloud) computed on the GFS forecast actually issued for that
  night, recovered from Open-Meteo's archive of past runs, plus the neighbouring nights.
- `obs_*`: recent conditions up to 11:00 local on D (rolling 24 h to 14 d wetness, rain,
  temperature, humidity, recent infection nights).
- `lead`, day of year (sin/cos).

Training on archived forecasts rather than observed weather means the model learns to correct
the GFS forecast instead of re-learning the labeling rule.

The `obs_*` features are built from the GFS lead-0 series (each hour's same-day model value),
not from reanalysis. The live job can only get recent weather from the forecast API's
`past_days`, which returns that same series. GFS and reanalysis disagree on about 15% of
nightly labels at this site, so mixing the two sources would shift the features between
training and live use. Reanalysis is used only for the labels.

### Validation

Leave-one-season-out over coffee seasons (Jul-Jun) with an 8-day purge around each test season.
Out-of-fold, all leads, issues Jan 2024 to Oct 2026 (7,088 rows):

| Model | AUC | Brier | Brier skill vs climatology |
|---|---|---|---|
| **XGBoost** | **0.910** | **0.114** | **+0.36** |
| Rule on raw GFS forecast (graded score) | 0.884 | 0.130 | +0.27 |
| Rule on raw GFS forecast (0/1) | 0.830 | 0.172 | +0.03 |
| Monthly climatology | 0.789 | 0.178 | 0 |
| Persistence (last night) | 0.705 | 0.284 | -0.60 |

Skill decays with lead (AUC 0.92 at 1 day, 0.89 at 7 days), and XGBoost beats the raw-forecast
rule at every lead. Probabilities are well calibrated, slightly overconfident above 0.9
(mean predicted 0.94, observed 0.88).

Forward check: each season predicted only from earlier seasons, as in real use. On the
2025/26 and 2026/27 seasons this gives AUC 0.930 and Brier skill +0.39, against 0.931 / +0.42
for leave-one-season-out on the same rows. Training on later seasons does not inflate the
scores. Both results are stored in `models/model_meta.json`.

## Live inference

```powershell
python -m src.update_forecast       # -> docs/rust_risk.json
```

Fetches recent and forecast hourly weather (`gfs_seamless`, the model used in training),
rebuilds the same features, scores them on CPU, and writes `docs/rust_risk.json`:

- `alert`: headline level, peak probability and night, number of High+ nights
- `days[]`: per night (lead 1-7) probability, level, raw-rule score, wet/favourable hours,
  longest wet run, temperature, RH, rain
- `hourly`: temperature / RH / precipitation (2 days of context + forecast), with the boundary
  between past and forecast hours
- `recent_nights[]`: the last 14 nights' infection labels, computed on GFS lead-0 weather
- `model`: forecast model, training window, CV AUC by lead

Alert levels: Low < 25% <= Moderate < 50% <= High < 75% <= Severe.

The job stops with an error rather than writing a doubtful forecast. It fails when recent
weather has gaps or stops before the 11:00 cutoff, when there is less than 33 days of history
(Open-Meteo keeps about 42), when a forecast night is incomplete, when a feature is missing,
or when a probability is outside [0, 1]. A failed run leaves the previous forecast in place,
and the dashboard flags a forecast older than three days.

`tests/test_parity.py` feeds archived data through the live code path and checks that it
reproduces the training feature table, both with the full 45-day look-back and with the
33-day minimum. `tests/test_pipeline.py` runs the labeling rule and the full live pipeline
on synthetic weather, so it needs no downloaded data and runs in CI.

## Dashboard

`docs/` is a static site (Chart.js, no build step) that reads `docs/rust_risk.json`: alert level
and peak probability, a 7-night risk chart, hourly temperature / humidity / rain panels with
infection-favourable hours shaded, and night-by-night and hourly tables.

```powershell
python scripts/serve_dashboard.py   # then open http://localhost:8000
```

The script serves `docs/` with caching disabled, so a normal refresh picks up edits.

## Automation

- `.github/workflows/pathogen_inference.yml` runs daily at 21:00 UTC (18:00 in Minas Gerais),
  installs `requirements.txt`, runs `src.update_forecast`, and commits `docs/rust_risk.json`
  when it changes. The commit history doubles as an archive of issued forecasts.
- `.github/workflows/tests.yml` runs pyflakes and pytest on every push and pull request.

The model is not retrained in CI. Retrain locally and commit `models/` when you want it to
learn from newer seasons.

## Limitations

- Labels mark weather favourable to infection, not observed disease. Inoculum, host
  susceptibility, crop load and fungicide use are not modelled.
- One site (a single Open-Meteo grid cell). Leaf wetness is inferred from RH, dew point and
  rain, not measured.
- Two and a half seasons of archived forecasts, so seasonal-scale variability is thinly sampled.

## Data

Weather data from [Open-Meteo](https://open-meteo.com/) (Historical Weather, Previous Runs and
Forecast APIs), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
GFS forecasts are from NOAA.

## License

Code is released under the [MIT License](LICENSE).

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
notebooks/
  01_biological_heuristics.ipynb      weather -> infection labels
  02_features_and_gpu_xgboost.ipynb   forecasts by lead, GPU timing, CV skill, calibration
tests/                 synthetic pipeline tests, train/serve parity, model loading
data/raw/              downloaded source data (not committed)
data/processed/        labels, features, out-of-fold predictions (not committed)
models/                coffee_rust_xgb.json + model_meta.json
docs/                  dashboard (GitHub Pages)
```
