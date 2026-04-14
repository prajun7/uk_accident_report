# UK Accident Risk Zone Model

This project trains a multi-model accident risk-zone classifier on UK STATS19 data and predicts:

- `Low Risk`
- `Medium Risk`
- `High Risk`

The current best model from Step 7 is saved in `output/7_model_bundle.joblib`.

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

Or run the main training stages manually:

```bash
./.venv/bin/python 6_data_aggregation_representation.py
./.venv/bin/python 7_data_analysis.py
./.venv/bin/python 8_data_visualization.py
```

Important outputs:

- `output/6_X_final.csv`, `output/6_y_final.csv`: final training arrays
- `output/7_model_bundle.joblib`: full trained inference bundle
- `output/7_model_scores.csv`: holdout accuracies
- `output/7_predictions.csv`: saved predictions for comparison plots

## Inference

Use the CLI:

```bash
./.venv/bin/python 10_inference.py --pretty
```

Predict one custom scenario with inline JSON:

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

- `--show-schema`: print accepted fields and derived features
- `--show-defaults`: print default values used for omitted fields
- `--input path/to/payload.json`: load a JSON file instead of inline JSON

## Web UI

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
- `GET /healthz`: simple health check

The UI uses the current best saved model and sends raw inputs to the backend, which rebuilds the engineered features automatically.
