# Pearls AQI Predictor

Predict the Air Quality Index (AQI) for Karachi for the next 3 days with a
serverless machine-learning pipeline:

External APIs -> feature pipeline -> feature store -> training pipeline ->
model registry -> inference pipeline -> Streamlit dashboard.

The project has two modes:

- Local demo mode uses an on-disk feature store/model registry so the app can
  be explained and tested without accounts.
- GitHub/serverless mode requires Hopsworks and uses the real cloud feature
  store/model registry required by the project brief.

## What Is Included

- Weather and pollutant collection from Open-Meteo, AQICN, and OpenWeather.
- Feature engineering for time, weather, derived stagnation, and AQI dynamics.
- Historical backfill for training data.
- Model training with Ridge Regression, Random Forest, optional XGBoost,
  optional LightGBM, optional TensorFlow MLP, and a weighted ensemble benchmark.
- Evaluation with RMSE, MAE, R2, and TimeSeriesSplit CV.
- Hopsworks-backed feature store/model registry for the serverless pipeline,
  plus local fallbacks for offline demos.
- 3-day batch inference with hazardous-AQI alerts.
- Streamlit dashboard with forecast cards, trend charts, leaderboard, and SHAP
  feature importance when available.
- Flask API for forecast, metrics, latest history, and health endpoints.
- GitHub Actions for CI, hourly feature/inference runs, and daily training.
- Offline sample-data generator so the viva demo works without internet keys.

## Project Structure

```text
pearls-aqi-predictor/
  config.yaml
  requirements.txt
  .env.example
  src/
    aqi.py
    config.py
    data_sources.py
    feature_engineering.py
    feature_store.py
    model_registry.py
    utils.py
  pipelines/
    backfill_pipeline.py
    feature_pipeline.py
    training_pipeline.py
    inference_pipeline.py
  app/
    streamlit_app.py
    flask_api.py
  notebooks/
    eda.py
  scripts/
    generate_sample_data.py
  tests/
    test_pipeline.py
  .github/workflows/
    ci.yml
    feature-pipeline.yml
    training-pipeline.yml
```

## Setup

Use Python 3.11 or 3.12.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

All commands below must be run from the folder that contains `config.yaml`,
`requirements.txt`, `src/`, `pipelines/`, and `app/`.

## Offline Quickstart

This path needs no accounts and no API keys.

```bash
python -m scripts.generate_sample_data
python -m pipelines.training_pipeline --quick
python -m pipelines.inference_pipeline
streamlit run app/streamlit_app.py
```

Open the Streamlit URL printed in the terminal, usually
`http://localhost:8501`.

To run the optional Flask API:

```bash
flask --app app.flask_api run --host 0.0.0.0 --port 8000
```

Useful API endpoints:

- `GET /health`
- `GET /forecast`
- `GET /predictions`
- `GET /metrics`
- `GET /history/latest?n=30`
- `POST /run-inference`

For the full model run, use:

```bash
python -m pipelines.training_pipeline
```

The full run includes optional TensorFlow and SHAP work if those packages are
installed and enabled in `config.yaml`.

## Real Data Backfill

Open-Meteo does not require a key, so the project can build a real Karachi
history without signup:

```bash
python -m pipelines.backfill_pipeline
python -m pipelines.training_pipeline --quick
python -m pipelines.inference_pipeline
streamlit run app/streamlit_app.py
```

To change the historical window:

```bash
python -m pipelines.backfill_pipeline --start 2021-01-01 --end 2026-06-07
```

The default backfill end date is yesterday in the configured timezone, so the
training data avoids incomplete current-day daily averages.

## Optional EDA Figures

```bash
python -m notebooks.eda
```

This creates:

- `figures/01_aqi_timeseries.png`
- `figures/02_seasonality.png`
- `figures/03_aqi_vs_weather.png`
- `figures/04_correlation_heatmap.png`

## Optional Cloud / Serverless Mode

1. Create a Hopsworks account at `https://app.hopsworks.ai`.
2. Create an API key.
3. Copy `.env.example` to `.env`.
4. Fill in:

```env
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT=your_project_name
AQICN_TOKEN=optional_live_aqi_token
OPENWEATHER_API_KEY=optional_alternate_provider_key
```

With `backend.mode: auto` in `config.yaml`, the code uses Hopsworks when
`HOPSWORKS_API_KEY` is present and local files when it is absent.

## GitHub Actions

The workflows are already included.

- `ci.yml`: runs unit tests on push and pull request.
- `feature-pipeline.yml`: runs hourly and on manual dispatch. It updates recent
  features and refreshes the 3-day predictions.
- `training-pipeline.yml`: runs daily and on manual dispatch. It retrains and
  registers the best model.

For the required cloud/serverless version, add these repository secrets in
GitHub before running the scheduled workflows:

- `HOPSWORKS_API_KEY`
- `HOPSWORKS_PROJECT`
- `AQICN_TOKEN` (optional)
- `OPENWEATHER_API_KEY` (optional)

The scheduled workflows intentionally fail fast when the required Hopsworks
secrets are missing. This prevents accidentally demonstrating a local-only
pipeline as the serverless feature-store pipeline.

First-run order on GitHub:

1. Add `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT` in repository secrets.
2. Open Actions -> `training-pipeline` -> Run workflow.
3. Open Actions -> `feature-pipeline` -> Run workflow.
4. Confirm `models/metrics.json` and `data/latest_predictions.json` artifacts
   are uploaded by the workflow runs.

The training workflow uses the lighter reliable training path by default. Use
the manual `full_training=true` input when you want GitHub Actions to install
the full optional stack from `requirements.txt` and attempt TensorFlow/SHAP and
boosting models.

## Streamlit Cloud Deployment

1. Push the repository to GitHub.
2. Go to `https://share.streamlit.io`.
3. Create a new app from the repository.
4. Set the main file to `app/streamlit_app.py`.
5. Add the same secrets if using Hopsworks/AQICN.
6. Deploy.

## Configuration

Change `config.yaml` to forecast another city:

- `location.city`
- `location.country`
- `location.latitude`
- `location.longitude`
- `location.timezone`
- `location.aqicn_station`

The forecast horizon, backfill start date, backend mode, model names, and
training options are also controlled from `config.yaml`.

## Tests

```bash
pytest -q
```

The tests cover AQI breakpoint math, AQI categories and alerts, feature-column
creation, and training-frame cleanup.

## Viva Talking Points

- The architecture follows the Feature / Training / Inference pattern.
- The feature store prevents training/serving feature drift.
- The model registry stores the fitted model, feature column order, and metrics.
- The 3-day forecast combines future weather with recent observed AQI dynamics.
- The local fallback makes the demo reliable without accounts.
- Hopsworks plus GitHub Actions satisfies the serverless automation requirement.
- SHAP explains which features are most important when a tree model wins.
- Hazard alerts trigger when predicted AQI crosses the configured threshold.

## Disclaimer

This project is for education and demonstration. Forecasts are estimates and
are not a substitute for official public-health air-quality advisories.
