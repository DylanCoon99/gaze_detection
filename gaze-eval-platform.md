# Gaze evaluation platform — project spec

## What this is

Evaluation infrastructure for driver-monitoring gaze models: a system that answers **"which model should we ship, and how do we know?"** repeatedly and automatically, rather than once by hand.

This is a reframing and extension of the existing gaze benchmarking suite (5 architectures, 300W-LP / AFLW2000, accuracy vs. latency vs. power on Raspberry Pi). The models and measurement code already exist. What's being added is everything that turns a benchmark into a platform.

**Why this framing:** "I optimize models" invites comparison against people who write CUDA kernels. "I built the infrastructure that decides which model ships" is a direct demonstration of ML platform work — which is the role actually reachable right now.

---

## The design test

> Adding a sixth architecture should require writing one config file and changing nothing else.

If adding a model means editing the runner, this is a benchmark script. If it means dropping in a YAML and opening a PR, it's evaluation infrastructure. Every design decision below serves this property.

### Starting models (5)

| # | Model | Family | Runtime | Quantization | Purpose |
|---|---|---|---|---|---|
| 1 | Geometric baseline | geometric | rule-based | n/a | Proves the runner interface is genuinely abstract, not just "PyTorch models with different configs" |
| 2 | MobileNetV2 FP32 | cnn | torch | none | CNN baseline, no quantization — already built in refresher exercises |
| 3 | MobileNetV3 INT8 | cnn | tflite_xnnpack | int8_ptq | Quantized CNN, different runtime — creates a clear Pareto tradeoff against #2 |
| 4 | ResNet-18 FP32 | cnn | torch | none | Larger CNN, intentionally worse on latency/size — should be dominated on the Pareto plot |
| 5 | MobileViT-small | vit | torch or tflite | none | Transformer-based — proves the platform isn't CNN-specific |

This gives 3 runtimes (rule-based, PyTorch, TFLite), 2 quantization levels (FP32, INT8), and 3 architecture families (geometric, CNN, ViT).

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

Each component is small. The value is in the interfaces between them.

---

## Components

### 1. Config layer

One YAML per architecture. Fields:

```yaml
name: mobilenet_v3_int8
family: cnn                    # geometric | cnn | vit
weights: s3://gaze-models/mnv3_int8.tflite
runtime: tflite_xnnpack        # torch | tflite_xnnpack | geometric
input_resolution: [224, 224]
quantization: int8_ptq         # none | int8_ptq
preprocessing: standard_crop_v2
```

The geometric baseline and MobileViT differ enormously in implementation but must present the **same interface** to the runner: `load()`, `warmup(n)`, `infer(batch)`, `teardown()`. Defining that interface is the real design work in this project — do it first.

### 2. Dataset layer

300W-LP / AFLW2000 pinned with DVC, **including the preprocessing output**, not just the raw archives.

Why this matters more than it sounds: when a number moves, you need to know whether the model changed or the face-crop margin did. Versioning makes "run 47 used crops v2" a fact rather than an archaeology project.

### 3. Eval runner

Takes `(model_config, dataset_version, hardware_target)` → produces one run.

- Containerized; same image runs on dev machine and on device
- Logs git SHA, config hash, dataset version, random seed
- Fixed warmup iterations before timed iterations (critical for on-device latency numbers)
- Uploads artifacts (per-sample predictions, error distributions, raw timing traces)

Everything above this line is declarative. Everything below is a result.

### 4. Metrics

Mean angular error is the paper metric and the least interesting number recorded. Track:

| Metric | Why |
|---|---|
| Mean / p50 angular error | Baseline comparability with literature |
| **p95 / p99 angular error** | A model with 4° mean that occasionally reports 30° misses real glances away |
| **Eyes-off-road FPR / FNR** | The actual product decision: given a gaze-cone threshold + dwell time, does it fire correctly? False positives are what make drivers disable the system |
| Latency p50 / p99 | Mean latency is not a deployment metric |
| Power (mW) | Pareto axis |
| Model size (MB), peak RSS | Deployment constraint |

#### Slices — the part that makes it an eval harness

Overall accuracy hides everything. Break every metric out by:

- **Head-pose bin** — especially yaw beyond ±45°, where models fall apart and where a driver checking a blind spot lives
- **Glasses / no glasses**
- **Lighting condition**

A harness that reports one number per model is a benchmark. One that reports *"MobileNetV3 INT8 is best overall but degrades badly at extreme yaw"* is doing the job.

### 5. Run store

MLflow, self-hosted. Every run = params + metrics + artifacts. Gives the comparison UI for free — do not build a custom dashboard for tabular comparison.

Artifacts to S3 or MinIO.

### 6. CI regression gate

**The centerpiece.** Lead the README with this.

- PR triggers eval on a held-out subset
- Results compared against the current baseline run
- Build **fails** if any tracked metric degrades past threshold
- Bot comment on the PR with the metric delta table

Automated regression gating on model quality is precisely what evaluation infrastructure teams exist to do. Almost no portfolio project has it. Expect it to be the thing interviewers ask about.

### 7. Comparison view

MLflow covers tabular comparison. The one thing worth building yourself is the **Pareto plot** — accuracy vs. latency vs. power with dominated models greyed out — because that was the original project goal and MLflow won't render it.

A single FastAPI endpoint reading from the run store. One day of work, not a week. Build it last.

---

## Build order

1. **Runner interface + config layer** — everything depends on this abstraction
2. **Dataset versioning (DVC)**
3. **Metrics + slicing**
4. **MLflow tracking**
5. **Dockerfile / reproducible run**
6. **CI regression gate**
7. **Pareto view (FastAPI)**
8. *Optional:* Terraform for the cloud pieces

Milestone check after step 6: can a stranger clone the repo, run one command, and reproduce your numbers?

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

## Non-goals

Explicitly out of scope. Cut these if they creep in:

- Jetson port and TensorRT optimization — nice-to-have, not a prerequisite for this framing. Finish on the Pi.
- Custom CUDA/Triton kernels
- Training new architectures from scratch
- A hand-built dashboard duplicating MLflow's UI
- Kubernetes deployment
- Beating SOTA accuracy — the point is the infrastructure, not the numbers it produces

---

## README framing

The README is the deliverable a hiring manager actually reads. Structure it as:

1. **One-line pitch** — evaluation infrastructure for driver-monitoring gaze models
2. **The regression gate**, with a screenshot of a failing PR
3. **Architecture diagram**
4. **How to reproduce** — clone, one command, exact numbers
5. **Results** — Pareto plot, plus the slice table showing where models break down
6. **Design decisions** — why DVC, why the runner interface looks the way it does, what you'd do differently

Point 6 is where senior engineers decide whether you thought about the problem or followed a tutorial.
