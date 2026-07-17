# Gaze Estimation Benchmarking Suite

A systematic study of neural network optimization techniques for real-time head pose estimation on a Raspberry Pi. Eight models are trained, quantized, pruned, and distilled against the same task, dataset, and measurement harness to produce a rigorous accuracy-latency-power Pareto analysis.

---

## Motivation

Edge inference on constrained hardware forces genuine understanding of what neural network optimization techniques actually do. This project treats a single application — head pose estimation as a gaze proxy for driver monitoring — as a controlled benchmark environment. Every technique (quantization, pruning, distillation, binary networks) is applied to the same task so results are directly comparable, not isolated toy demonstrations.

---

## Task Definition

**Input:** Cropped and aligned face image (112×112 RGB)  
**Output:** Yaw and pitch angles in degrees  
**Loss:** Mean squared error on yaw and pitch  
**Primary metric:** Mean absolute error (MAE) in degrees on the AFLW2000 test set

The face crop is produced by a fixed MediaPipe BlazeFace detector upstream of all models. Preprocessing is locked before any model is trained and does not change between experiments.

---

## Dataset

| Split | Source | Size |
|---|---|---|
| Train | 300W-LP | ~60,000 images |
| Eval | AFLW2000 | 2,000 images |

The AFLW2000 test split is locked before any model touches it. No hyperparameter decisions are made using test set performance.

---

## Models

### Model 1 — Geometric baseline
OpenCV face detector → 68-point landmark detector (dlib) → `solvePnP` to recover the 3D rotation vector. No neural network. Establishes the accuracy floor and latency baseline that all learned models are compared against.

### Model 2 — Frozen MobileNetV2
MobileNetV2 pretrained on ImageNet, all conv layers frozen, small regression head trained on 300W-LP. Evaluates how well ImageNet features transfer to head pose without any task-specific fine-tuning.

### Model 3 — Fine-tuned MobileNetV2
Same architecture as Model 2 with all layers unfrozen and fine-tuned end-to-end on 300W-LP. Used as the primary baseline for quantization experiments (PTQ and QAT both applied).

### Model 4 — MobileNetV3-Small / EfficientNet-Lite0
A mobile-first backbone designed for constrained inference. Evaluates whether architecture choices (squeeze-and-excitation, hard-swish) yield better accuracy-latency tradeoffs than fine-tuning a larger backbone.

### Model 5 — Custom tiny CNN
A small CNN designed from scratch with explicit FLOP budget constraints. No pretrained weights. Forces reasoning about receptive field, depth vs width, and the minimum network capacity needed for this task.

### Model 6 — Pruned MobileNetV2
Structured channel pruning applied to Model 3 at 30% and 50% sparsity levels using `torch-pruning`, followed by retraining to recover accuracy. Structured (not unstructured) pruning is used because only channel removal produces real latency reduction on ARM.

### Model 7 — Knowledge distillation
A small student network (same architecture as Model 5) trained with a larger teacher (Model 3 or a ResNet50 teacher trained separately) providing soft supervision. Compared directly against Model 5 trained without distillation to isolate the value of the teacher signal.

### Model 8 — Binary / ternary network
BNN and TWN variants of the Model 5 architecture implemented using Larq or Brevitas. Mixed-precision variant keeps FP32 in the first and last layers. Evaluates the accuracy cost of extreme quantization and connects directly to FPGA-based BNN accelerator design (XNOR-popcount).

---

## Measurement Harness

All models pass through the same harness. Build it before training any model.

### Accuracy
- MAE on yaw and pitch separately (degrees)
- Percentage of predictions within 5° and 15° thresholds
- Error distribution histogram (tail behavior matters for safety applications)

### Latency
- Mean inference time per frame (ms)
- p95 and p99 latency
- Measured on Pi under realistic CPU load, not idle
- Whole pipeline: capture → preprocess → inference → output

### Memory
- Model size on disk (MB)
- Peak RAM during inference (`/proc/self/status`)

### Power
- Idle draw vs inference draw via USB power meter (e.g. UM25C)
- Energy per inference = power delta × mean latency

### Throughput
- Maximum sustainable FPS at full pipeline

---

## Quantization Ladder

For Models 2–6, each is evaluated at every precision level:

| Precision | Method | Notes |
|---|---|---|
| FP32 | Baseline | PyTorch native |
| FP16 | Cast | Measure whether NEON gives benefit on Pi 4 |
| INT8 PTQ | TFLite / ONNX Runtime | Calibration set = 200 training images |
| INT8 QAT | torch.ao.quantization | Fake quant nodes during training |

Per-layer sensitivity analysis is run for each model: quantization is disabled one layer at a time and the MAE recovery is recorded. This identifies which layers tolerate INT8 worst.

---

## Repository Structure

```
.
├── data/
│   ├── 300wlp/              # Training set
│   ├── aflw2000/            # Eval set (test split locked)
│   └── splits/              # Saved train/val/test indices
├── preprocessing/
│   ├── face_detector.py     # MediaPipe BlazeFace wrapper
│   ├── crop_align.py        # Face crop and alignment pipeline
│   └── normalize.py         # Normalization constants
├── models/
│   ├── geometric/           # Model 1: solvePnP baseline
│   ├── mobilenetv2/         # Models 2, 3, 6: frozen, fine-tuned, pruned
│   ├── mobilenetv3/         # Model 4: lightweight architecture
│   ├── tiny_cnn/            # Models 5, 7, 8: custom, distilled, binary
│   └── teachers/            # Larger teacher models for distillation
├── quantization/
│   ├── ptq.py               # Post-training quantization pipeline
│   ├── qat.py               # Quantization-aware training
│   ├── sensitivity.py       # Per-layer sensitivity analysis
│   └── export_tflite.py     # TFLite conversion and benchmark
├── pruning/
│   ├── structured_prune.py  # Channel pruning with torch-pruning
│   └── retrain.py           # Post-pruning accuracy recovery
├── distillation/
│   ├── output_kd.py         # Hinton-style output distillation
│   └── feature_kd.py        # FitNets-style feature distillation
├── binary/
│   ├── bnn_train.py         # Binary network training (Larq or Brevitas)
│   ├── twn_train.py         # Ternary weight network
│   └── mixed_precision.py   # FP32 first/last + binary middle
├── harness/
│   ├── benchmark.py         # Unified accuracy + latency + memory measurement
│   ├── latency.py           # p50/p95/p99 latency profiler
│   ├── power.py             # USB power meter interface
│   └── results/             # JSON results per model per precision
├── analysis/
│   ├── pareto.py            # Accuracy vs latency Pareto plot
│   ├── sensitivity_plot.py  # Per-layer sensitivity visualization
│   └── summary_table.py     # Full results table generation
└── docs/
    └── results.md           # Final write-up and analysis
```

---

## Build Order

1. Lock dataset splits and preprocessing pipeline
2. Build the measurement harness (stub with random predictions to verify it runs)
3. **Model 1** — geometric baseline through harness
4. **Models 2 & 3** — frozen and fine-tuned MobileNetV2, PTQ applied to both
5. **Model 3 (QAT)** — QAT vs PTQ comparison on same architecture
6. **Model 4** — lightweight backbone through harness
7. **Model 5** — custom tiny CNN designed and trained from scratch
8. **Model 6** — structured pruning on Model 3 at 30% and 50%
9. **Model 7** — distillation into Model 5 architecture
10. **Model 8** — BNN and TWN experiments, mixed-precision variant
11. Final Pareto analysis and write-up

Do not change the preprocessing pipeline or test split after step 1.

---

## Learning Objectives by Phase

### ML foundations (before starting)
- Backpropagation and gradient flow through conv layers
- Loss functions for regression vs classification
- CNN architecture fundamentals: receptive field, depthwise separable convolutions, residual connections
- Transfer learning mechanics: what ImageNet features generalize to and why

### Quantization (Models 2–4)
- Scale factors and zero points in INT8 quantization
- Why PTQ loses accuracy and how QAT recovers it (straight-through estimator)
- Per-layer sensitivity: which layers tolerate INT8 worst and why
- TFLite export and ONNX Runtime deployment on ARM

### Architecture design (Model 5)
- Receptive field analysis and FLOP counting by hand
- Width vs depth tradeoffs for a fixed parameter budget
- Training from scratch discipline: overfit one batch first, then generalize

### Pruning (Model 6)
- Structured vs unstructured pruning and why only structured gives real ARM speedup
- Magnitude-based channel importance criteria
- The prune → retrain cycle and iterative vs one-shot approaches

### Distillation (Model 7)
- Soft targets and the temperature hyperparameter
- Hinton distillation loss: hard label CE + soft target KL divergence
- Feature-level distillation for regression tasks (MSE against teacher intermediate features)

### Binary networks (Model 8)
- XNOR-popcount as a replacement for multiply-accumulate
- The accuracy cost of binarization and why first/last layers are most sensitive
- Mixed-precision strategies and how they map to FPGA LUT-based BNN accelerators

### Performance profiling (cross-cutting)
- C++ profiling with `perf`, `gprof`, and `Valgrind`/`Callgrind` — identifying hotspots in inference code
- Flame graphs for visualizing call stacks and time spent per function
- CPU cache behavior and memory access patterns: using `perf stat` to measure cache misses, branch mispredictions, and IPC
- ARM-specific profiling: NEON utilization, pipeline stalls, and thermal throttling on the Pi
- ML model profiling: per-layer latency breakdown using TFLite's built-in profiler and `benchmark_model --enable_op_profiling`
- PyTorch Profiler (`torch.profiler`) for training-side bottleneck analysis — CPU vs GPU time, operator-level breakdown
- Memory profiling: tracking peak RSS, heap allocations (`massif`), and memory fragmentation during inference
- Microbenchmarking discipline: warm-up runs, statistical aggregation (mean/p95/p99), and controlling for thermal throttling and CPU governor state

---

## Key Papers

| Paper | Relevance |
|---|---|
| MobileNetV2: Inverted Residuals and Linear Bottlenecks — Sandler et al. 2018 | Models 2, 3 |
| Searching for MobileNetV3 — Howard et al. 2019 | Model 4 |
| Quantization and Training of NNs for Integer-Arithmetic-Only Inference — Jacob et al. 2018 | PTQ/QAT |
| Pruning Filters for Efficient ConvNets — Li et al. 2017 | Model 6 |
| The Lottery Ticket Hypothesis — Frankle & Carlin 2019 | Model 6 |
| Distilling the Knowledge in a Neural Network — Hinton et al. 2015 | Model 7 |
| FitNets: Hints for Thin Deep Nets — Romero et al. 2015 | Model 7 |
| XNOR-Net: ImageNet Classification Using Binary CNN — Rastegari et al. 2016 | Model 8 |
| FINN: A Framework for Fast, Scalable Binarized Neural Network Inference — Umuroglu et al. 2017 | Model 8 → FPGA bridge |

---

## Runtime Stack

| Component | Choice | Reason |
|---|---|---|
| Training | PyTorch | QAT, pruning, and distillation tooling |
| On-device inference | TFLite | Best-optimized for ARM Cortex-A; XNNPACK backend |
| Face detection | MediaPipe BlazeFace | Fast enough to not dominate the latency budget |
| BNN training | Larq (TF) or Brevitas (PyTorch) | Native support for XNOR-popcount ops |
| Structured pruning | torch-pruning | Dependency graph-aware channel removal |
| Profiling | perf, `/proc/self/status`, UM25C power meter | Pi-native tooling |

Vanilla PyTorch is not used for on-device inference. It is not optimized for ARM and the overhead is significant compared to TFLite with XNNPACK.

---

## CI / GitHub Actions

### Unit tests
A GitHub Actions workflow runs `pytest` on every push and pull request. Tests cover the preprocessing pipeline, data loading, loss functions, harness utilities, and model construction — anything that does not require a GPU or Raspberry Pi hardware. Training smoke tests (1 epoch on a tiny data subset) verify that training loops don't crash due to shape mismatches or broken augmentation.

### Dependency management
Dependabot is enabled to monitor Python dependencies (PyTorch, TFLite, Brevitas, torch-pruning, etc.) and open PRs for version bumps automatically. This keeps the project current with security patches and API changes without manual tracking.

---

## Expected Outcomes

The final deliverable is a Pareto plot of MAE (degrees) vs inference latency (ms) on the Pi, with one point per model configuration. A good result is not a single winning model — it is a well-characterized frontier with a clear understanding of why each point is where it is.

The secondary deliverable is the per-layer sensitivity table across all quantized models: which layers lose the most accuracy under INT8, and whether the pattern is consistent across architectures.
