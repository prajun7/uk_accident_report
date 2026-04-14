# UK Accident Risk Zone Model

This project builds an end-to-end UK road-accident risk-zone pipeline on STATS19-style data.  
Instead of predicting raw police severity directly, the pipeline first creates spatial accident-density clusters and then classifies each case into:

- `Low Risk`
- `Medium Risk`
- `High Risk`

The current deployable model bundle is saved at `output/7_model_bundle.joblib`.

## Pipeline Overview

The main pipeline stages are:

1. `3_data_acquisition_filtering.py`
   Loads `Accidents0515.csv`, keeps pipeline-required columns, and drops only clearly unusable high-null columns.
2. `4_data_extraction.py`
   Extracts the road, environment, temporal, and spatial fields used by the current workflow.
3. `5_data_validation_cleansing.py`
   Cleans coded sentinel values safely, preserves missing geography/time when needed, and validates speed limits.
4. `6_data_aggregation_representation.py`
   Builds spatial clusters, derives `Risk_Zone`, engineers temporal/context features, removes exact duplicate supervised rows, balances the classes, and writes the final training arrays.
5. `7_data_analysis.py`
   Trains and compares `Random Forest`, `XGBoost`, `LightGBM`, and a `Neural Network`, then saves the best model bundle.
6. `8_data_visualization.py`
   Generates the spatial map, confusion matrices, correlation heatmap, feature-importance chart, and model-metrics comparison chart.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Run the full pipeline:

```bash
./run_pipeline.sh
```

Or run the stages manually:

```bash
./.venv/bin/python 3_data_acquisition_filtering.py
./.venv/bin/python 4_data_extraction.py
./.venv/bin/python 5_data_validation_cleansing.py
./.venv/bin/python 6_data_aggregation_representation.py
./.venv/bin/python 7_data_analysis.py
./.venv/bin/python 8_data_visualization.py
```

Important modeling notes in the current version:

- the target is an engineered `Risk_Zone`, not raw `Accident_Severity`
- missing coordinates are dropped before clustering instead of being imputed
- exact duplicate supervised rows are removed before balancing
- high-cardinality road/context fields use out-of-fold target encoding
- the earlier stacking ensemble was removed and replaced by a neural-network benchmark

Important outputs:

- `output/6_X_final.csv`, `output/6_y_final.csv`: final supervised training arrays
- `output/6_feature_correlation_matrix.csv`: saved correlation matrix source
- `output/7_model_scores.csv`: holdout score table
- `output/7_predictions.csv`: saved predictions for visualization
- `output/7_feature_importances_all_models.csv`: combined feature-importance table
- `output/7_model_bundle.joblib`: full trained inference bundle

Main figures:

- `output/8_1_spatial_risk_map.png`
- `output/8_2_model_comparison_cm.png`
- `output/8_3_correlation_heatmap.png`
- `output/8_3_feature_importance.png`
- `output/8_4_model_metrics_comparison.png`

## Inference

Use the CLI:

```bash
./.venv/bin/python 10_inference.py --pretty
```

Run one custom scenario with inline JSON:

```bash
./.venv/bin/python 10_inference.py --pretty --json '{
  "Speed_limit": 60,
  "Road_Type": 6,
  "1st_Road_Class": 3,
  "1st_Road_Number": 507,
  "Junction_Detail": 2,
  "Junction_Control": 4,
  "Light_Conditions": 4,
  "Weather_Conditions": 2,
  "Road_Surface_Conditions": 2,
  "Day_of_Week": 6,
  "Urban_or_Rural_Area": 2,
  "Hour": 23
}'
```

Helpful options:

- `--show-schema`: print accepted raw fields and derived features
- `--show-defaults`: print the trained defaults used for omitted fields
- `--input path/to/payload.json`: load a JSON payload from disk

Partial payloads are allowed. The inference layer rebuilds:

- `IsNight`
- `IsRushHour`
- `Hour_Sin`
- `Hour_Cos`
- `DayNight_Context`
- `Urban_Speed_Net`
- `Junction_Complexity`

## Web UI and Local API

Start the local server:

```bash
./.venv/bin/python predict_server.py
```

Then open:

```text
http://127.0.0.1:8765/
```

Available endpoints:

- `GET /`: browser UI
- `POST /predict`: JSON prediction API
- `GET /schema`: accepted feature schema and defaults
- `GET /healthz`: health check showing the loaded model name
- `GET /output/<file>`: serve generated PNG or CSV outputs

Example API call:

```bash
curl -s http://127.0.0.1:8765/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Speed_limit": 30,
    "Road_Type": 3,
    "1st_Road_Class": 4,
    "Day_of_Week": 2,
    "Urban_or_Rural_Area": 1,
    "Hour": 14
  }'
```

The server and UI both use the current best saved model bundle and rebuild the engineered features automatically before prediction.
