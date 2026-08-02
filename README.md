# Gaze Evaluation Platform

Evaluation infrastructure for driver-monitoring gaze models: a system that answers **"which model should we ship, and how do we know?"** repeatedly and automatically, rather than once by hand.

---

## The Problem

Benchmarking a model by hand answers the question once. When models change, datasets update, or deployment constraints shift, you run the notebook again and hope you didn't forget a step. This project replaces that with infrastructure: declarative model configs, versioned data, automated evaluation, and a CI gate that blocks regressions before they merge.

---

## Design Principle

> Adding a sixth architecture should require writing one config file and changing nothing else.

If adding a model means editing the runner, this is a benchmark script. If it means dropping in a YAML and opening a PR, it's evaluation infrastructure.

---

## Architecture

```
  Model configs              Dataset version
  (5 architectures, YAML)    (DVC-pinned crops)
          \                        /
           \                      /
            v                    v
           +----------------------+
           |     Eval runner      |
           | containerized,       |
           | runs on device       |
           +----------------------+
                      |
                      v
           +----------------------+
           |      Run store       |
           | params, metrics,     |
           | artifacts (MLflow)   |
           +----------------------+
                /            \
               v              v
      CI regression      Comparison view
      gate               (Pareto frontier)
      (blocks PR)
```

---

## Models

| # | Model | Family | Runtime | Quantization | Purpose |
|---|---|---|---|---|---|
| 1 | Geometric baseline | geometric | rule-based | n/a | Proves the runner interface is genuinely abstract, not just "PyTorch models with different configs" |
| 2 | MobileNetV2 FP32 | cnn | torch | none | CNN baseline, no quantization |
| 3 | MobileNetV3 INT8 | cnn | tflite_xnnpack | int8_ptq | Quantized CNN, different runtime — creates a clear Pareto tradeoff against #2 |
| 4 | ResNet-18 FP32 | cnn | torch | none | Larger CNN, intentionally worse on latency/size — should be dominated on the Pareto plot |
| 5 | MobileViT-small | vit | torch or tflite | none | Transformer-based — proves the platform isn't CNN-specific |

3 runtimes (rule-based, PyTorch, TFLite), 2 quantization levels (FP32, INT8), 3 architecture families (geometric, CNN, ViT).

---

## Task Definition

**Input:** Cropped and aligned face image (112×112 RGB)
**Output:** Yaw and pitch angles in degrees
**Loss:** Mean squared error on yaw and pitch
**Primary metric:** Mean absolute error (MAE) in degrees on the AFLW2000 test set

---

## Dataset

| Split | Source | Size |
|---|---|---|
| Train | 300W-LP | ~60,000 images |
| Eval | AFLW2000 | 2,000 images |

Data is pinned with DVC, **including the preprocessing output**, not just the raw archives. When a number moves, you know whether the model changed or the face-crop margin did.

---

## Metrics

| Metric | Why |
|---|---|
| Mean / p50 angular error | Baseline comparability with literature |
| **p95 / p99 angular error** | A model with 4° mean that occasionally reports 30° misses real glances away |
| **Eyes-off-road FPR / FNR** | The actual product decision: given a gaze-cone threshold + dwell time, does it fire correctly? |
| Latency p50 / p99 | Mean latency is not a deployment metric |
| Power (mW) | Pareto axis |
| Model size (MB), peak RSS | Deployment constraint |

### Slices

Every metric is broken out by:
- **Head-pose bin** — especially yaw beyond ±45°, where models fall apart and where a driver checking a blind spot lives
- **Glasses / no glasses**
- **Lighting condition**

---

## CI Regression Gate

- PR triggers eval on a held-out subset
- Results compared against the current baseline run
- Build **fails** if any tracked metric degrades past threshold
- Bot comment on the PR with the metric delta table

---

## Stack

| Layer | Choice |
|---|---|
| Config | Hydra or pydantic-settings |
| Data versioning | DVC |
| Tracking | MLflow (self-hosted) |
| Artifacts | S3 or MinIO |
| Container | Docker |
| CI | GitHub Actions |
| Serving | FastAPI |
| IaC | Terraform (optional) |

---

## Build Order

1. Runner interface + config layer
2. Dataset versioning (DVC)
3. Metrics + slicing
4. MLflow tracking
5. Dockerfile / reproducible run
6. CI regression gate
7. Pareto view (FastAPI)

Milestone check after step 6: can a stranger clone the repo, run one command, and reproduce your numbers?
