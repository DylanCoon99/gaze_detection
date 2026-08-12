# Gaze Evaluation Platform

Evaluation infrastructure for driver-monitoring gaze models: a system that answers **"which model should we ship, and how do we know?"** automatically and repeatedly.

---

## What It Does

This platform evaluates gaze estimation models against a common dataset and set of metrics, tracks results over time, and blocks regressions via CI. Instead of running a notebook once and eyeballing numbers, every model evaluation is containerized, logged, and compared automatically.

**Core workflow:**
1. Define a model in a YAML config (architecture, weights, quantization, runtime)
2. Run the evaluation runner — it loads the model, runs inference on preprocessed face crops, computes metrics, and logs everything to MLflow
3. Open a PR — CI builds the Docker image, runs the eval, and blocks the merge if any metric regresses
4. Compare models in the dashboard — a FastAPI service with an interactive Pareto frontier view

---

## Design Principle

> Adding a new model should require writing one config file and changing nothing else.

A model config looks like this:

```yaml
name: mobilenet_v2_fp32_finetuned
weights: /models/mobilenet_v2_fp32_finetuned.pt
runtime: torch
input_resolution: [224, 224]
quantization: "none"
preprocessing: standard_imagenet
```

Set `quantization: "dynamic_int8"` and the runner applies post-training quantization automatically — same weights, different runtime behavior.

---

## Architecture

```
  Model configs              Preprocessed dataset
  (YAML)                     (crops + labels.csv)
          \                        /
           \                      /
            v                    v
           +----------------------+
           |     Eval Runner      |
           |   (Dockerized)       |
           +----------------------+
                      |
                      v
           +----------------------+
           |       MLflow         |
           | params, metrics,     |
           | artifacts            |
           +----------------------+
                /            \
               v              v
      CI Regression       FastAPI Dashboard
      Gate                (Pareto frontier,
      (GitHub Actions)     model comparison)
```

---

## Metrics

| Category | Metrics |
|---|---|
| Accuracy | MAE (yaw, pitch, combined), p50/p95/p99 angular error |
| Safety | Eyes-off-road false positive rate, false negative rate |
| Latency | Per-batch mean, p50, p95, p99 (ms) |
| Slicing | All accuracy metrics broken out by head-pose bin (0-15, 15-30, 30-45, 45+) |

Hardware info (device, chip) is logged alongside latency so comparisons are apples-to-apples.

---

## CI Regression Gate

Every pull request to `main` triggers a GitHub Actions workflow that:

1. Builds the Docker image
2. Runs evaluation on a test fixture (20-image subset checked into the repo)
3. Compares results against `tests/fixtures/baseline.json`
4. **Fails the build** if any metric regresses beyond its tolerance

Tolerances are defined in `scripts/check_regression.py` — for example, MAE can't increase by more than 0.5 degrees, FNR can't increase by more than 2%.

---

## Dashboard

A FastAPI service reads from MLflow and serves an interactive dashboard with:

- **Models Overview** — latest metrics for each model at a glance
- **All Runs** — sortable table of every evaluation run
- **Model Detail** — run history for a specific model
- **Pareto Plot** — interactive scatter plot comparing any two metrics, with Pareto-optimal models highlighted

---

## Stack

| Layer | Choice |
|---|---|
| Config | Dataclass + YAML |
| Data versioning | DVC |
| Preprocessing | MediaPipe BlazeFace detection + crop pipeline |
| Tracking | MLflow |
| Container | Docker |
| CI | GitHub Actions |
| API / Dashboard | FastAPI + Tailwind CSS + Chart.js |

---

## Project Structure

```
src/
  infra/
    runner/          # Evaluation runner — loads model, runs inference, logs to MLflow
      configs/       # Model YAML configs
      models.py      # BaseModel ABC + TorchModel (with PTQ support)
      main.py        # CLI entrypoint
    metrics/         # Accuracy, safety, latency, and slicing calculations
    preprocessing/   # MediaPipe face detection + crop pipeline
  services/          # FastAPI dashboard + MLflow client
scripts/
  check_regression.py   # CI regression comparison script
  download_data.sh      # Downloads 300W-LP from Kaggle + runs preprocessing
tests/
  fixtures/             # Test subset (20 images + baseline metrics + dummy model)
.github/
  workflows/
    eval-gate.yaml      # CI regression gate workflow
```

---

## Getting Started

**Download and preprocess the dataset:**

```bash
./scripts/download_data.sh
```

This downloads 300W-LP from Kaggle and runs the preprocessing pipeline to produce face crops and labels.

**Run an evaluation locally:**

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
mlflow ui --port 5001 &
python src/infra/runner/main.py \
  --config src/infra/runner/configs/mobilenet_v2_fp32_finetuned.yaml \
  --data /path/to/preprocessed/crops
```

**Run via Docker:**

```bash
docker build -t gaze-eval .
docker run --rm \
  -v /path/to/crops:/data \
  -v /path/to/models:/models \
  gaze-eval --config /app/runner/configs/mobilenet_v2_fp32_finetuned.yaml --data /data
```

**Launch the dashboard:**

```bash
cd src/services
uvicorn main:app --reload
# Open http://localhost:8000
```
