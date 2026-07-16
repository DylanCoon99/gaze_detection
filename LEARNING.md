# Project-Specific ML Learning Guide

This guide covers the specialized ML topics needed for the Gaze Estimation Benchmarking Suite, beyond general ML fundamentals. Work through each section roughly in project build order.

---

## 1. CNN Architecture Fundamentals

Before touching any model, make sure these concepts are solid.

### Concepts
- **Depthwise separable convolutions** — how MobileNets factor standard convolutions into depthwise + pointwise to cut FLOPs
- **Inverted residual blocks** — MobileNetV2's expand → depthwise → project structure and why the residual is on the narrow (bottleneck) side
- **Receptive field** — how kernel size, stride, and depth determine how much of the input each neuron "sees"
- **FLOP counting** — calculating multiply-accumulate operations per layer by hand (input size × kernel size × output channels)
- **Squeeze-and-excitation blocks** — channel attention mechanism used in MobileNetV3

### Exercises
- Draw the computation graph for a standard 3×3 conv vs a depthwise separable conv; compute the FLOP ratio for a 112×112×32 input with 64 output channels
- Read the MobileNetV2 paper and diagram every layer of the architecture, annotating parameter counts
- Given a FLOP budget of 50M MACs, design a CNN on paper that maximizes receptive field

### Resources
- **Paper:** [MobileNetV2 — Sandler et al. 2018](https://arxiv.org/abs/1801.04381)
- **Paper:** [MobileNetV3 — Howard et al. 2019](https://arxiv.org/abs/1905.02244)
- **Video:** 3Blue1Brown — "But what is a convolution?" (YouTube)
- **Blog:** [Depthwise Separable Convolutions explained](https://eli.thegreenplace.net/2018/depthwise-separable-convolutions-for-machine-learning/) — Eli Bendersky
- **Tool:** `torchinfo` (`pip install torchinfo`) — use `summary(model, input_size=(1,3,112,112))` to verify your hand-calculated FLOP counts

---

## 2. Transfer Learning

### Concepts
- **Feature reuse** — why early conv layers (edges, textures) trained on ImageNet generalize to head pose, but later layers may not
- **Frozen vs fine-tuned** — when to freeze the backbone and when to unfreeze; effect on training speed and overfitting
- **Learning rate schedules for fine-tuning** — lower LR for pretrained layers, higher for the new head (discriminative learning rates)
- **Domain gap** — ImageNet (object classification, 224×224) vs 300W-LP (face regression, 112×112); what transfers and what doesn't

### Exercises
- Freeze all layers of a pretrained MobileNetV2, train a regression head, record MAE
- Unfreeze and fine-tune end-to-end with 10× lower LR on backbone vs head; compare
- Visualize intermediate feature maps (hook into `layer[4]` and `layer[14]`) for a face image — do the features look face-relevant?

### Resources
- **Course:** Fast.ai Lesson 1–3 (transfer learning is the core pedagogical approach)
- **Paper:** [How transferable are features in deep neural networks? — Yosinski et al. 2014](https://arxiv.org/abs/1411.1792)
- **PyTorch tutorial:** [Finetuning TorchVision Models](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)

---

## 3. Quantization

This is a core focus of the project. Understand the math before running any tools.

### Concepts
- **Affine quantization scheme** — `x_int = round(x_float / scale) + zero_point`; how scale and zero_point are chosen
- **Symmetric vs asymmetric quantization** — when zero_point = 0 (symmetric) vs non-zero (asymmetric); tradeoffs
- **Per-tensor vs per-channel quantization** — per-channel gives better accuracy but more bookkeeping
- **Post-training quantization (PTQ)** — calibrate scale/zero_point on a representative dataset after training; no retraining needed
- **Quantization-aware training (QAT)** — insert fake quantization nodes during training so the network learns to tolerate discretization
- **Straight-through estimator (STE)** — how gradients flow through the non-differentiable `round()` operation during QAT
- **Per-layer sensitivity** — some layers (especially first and last) lose more accuracy under INT8; quantize one layer at a time to measure impact
- **Activation vs weight quantization** — weights are static (quantized once), activations are dynamic (quantized per inference with calibrated ranges)

### Exercises
- Manually quantize a single weight tensor to INT8: compute scale and zero_point, quantize, dequantize, measure the reconstruction error
- Apply PTQ to a trained MobileNetV2 using `torch.ao.quantization` with a 200-image calibration set; measure MAE before and after
- Apply QAT to the same model; compare MAE recovery vs PTQ
- Run per-layer sensitivity analysis: quantize all layers to INT8, then disable quantization one layer at a time; plot which layers matter most
- Export a quantized model to TFLite and run inference on desktop to verify correctness before deploying to Pi

### Resources
- **Paper:** [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference — Jacob et al. 2018](https://arxiv.org/abs/1712.05877) (the foundational INT8 quantization paper)
- **Paper:** [A White Paper on Neural Network Quantization — Nagel et al. 2021](https://arxiv.org/abs/2106.08295) (comprehensive survey, read sections 2–4)
- **PyTorch docs:** [Quantization documentation](https://pytorch.org/docs/stable/quantization.html) — practical API reference
- **PyTorch tutorial:** [Static Quantization with Eager Mode](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- **TFLite docs:** [Post-training quantization guide](https://www.tensorflow.org/lite/performance/post_training_quantization)
- **Blog:** [Lei Mao's Quantization blog posts](https://leimao.github.io/) — clear math-first explanations
- **Video:** [Quantization in Deep Learning (MIT HAN Lab)](https://www.youtube.com/watch?v=Qr2AKxsYpJQ) — Song Han's lecture from TinyML course

---

## 4. Pruning

### Concepts
- **Structured vs unstructured pruning** — unstructured (individual weight zeroing) doesn't speed up inference on ARM because the tensor shape doesn't change; structured (entire channel removal) actually reduces computation
- **Magnitude-based criteria** — rank channels by L1 or L2 norm of their weights; remove the smallest
- **Dependency graphs** — removing a channel in one layer requires adjusting adjacent layers; `torch-pruning` handles this automatically
- **Pruning schedule** — one-shot (prune all at once, retrain) vs iterative (prune a little, retrain, repeat); iterative generally preserves more accuracy
- **The Lottery Ticket Hypothesis** — sparse subnetworks exist at initialization that can train to full accuracy; relevant context but not directly used in this project

### Exercises
- Use `torch-pruning` to remove 30% of channels from a trained MobileNetV2; measure MAE and latency before and after retraining
- Repeat at 50% sparsity; plot the accuracy-sparsity curve
- Inspect which layers lost the most channels — do they correlate with the quantization-sensitive layers?
- Compare one-shot pruning at 50% vs two rounds of 30% pruning with retraining between

### Resources
- **Paper:** [Pruning Filters for Efficient ConvNets — Li et al. 2017](https://arxiv.org/abs/1608.08710)
- **Paper:** [The Lottery Ticket Hypothesis — Frankle & Carlin 2019](https://arxiv.org/abs/1803.03635) (read for context)
- **Library docs:** [torch-pruning GitHub](https://github.com/VainF/Torch-Pruning) — dependency-graph-aware structured pruning
- **Video:** [MIT HAN Lab — Pruning and Sparsity](https://www.youtube.com/watch?v=sZzc6s2lH34) — Song Han's lecture

---

## 5. Knowledge Distillation

### Concepts
- **Soft targets** — the teacher's output probabilities (softened with temperature) carry information about inter-class relationships that hard labels don't
- **Temperature scaling** — dividing logits by temperature T > 1 before softmax to produce softer probability distributions; higher T = more information transfer but noisier signal
- **Hinton distillation loss** — `L = α * CE(student, hard_label) + (1-α) * T² * KL(student_soft, teacher_soft)`; the T² factor compensates for gradient magnitude changes
- **Distillation for regression** — no softmax, so the standard approach is MSE between student and teacher outputs; optionally add feature-level distillation
- **Feature-level distillation (FitNets)** — align intermediate feature maps between student and teacher using a learned projection layer; helps when student architecture differs from teacher
- **Teacher quality** — a better teacher doesn't always produce a better student; the capacity gap matters

### Exercises
- Train Model 5 (tiny CNN) from scratch on hard labels; record MAE
- Train the same architecture with output-level distillation from Model 3 (fine-tuned MobileNetV2); compare MAE
- Experiment with feature-level distillation: pick one intermediate layer from teacher and student, add a 1×1 conv adapter, train with combined output + feature MSE loss
- Vary the loss weighting α between 0.1 and 0.9; plot the effect on student accuracy
- Try distillation from a larger teacher (ResNet50); does a bigger teacher help or hurt?

### Resources
- **Paper:** [Distilling the Knowledge in a Neural Network — Hinton et al. 2015](https://arxiv.org/abs/1503.02531) (the foundational paper; short and readable)
- **Paper:** [FitNets: Hints for Thin Deep Nets — Romero et al. 2015](https://arxiv.org/abs/1412.6550)
- **Blog:** [Knowledge Distillation: A Survey — Gou et al. 2021](https://arxiv.org/abs/2006.05525) (comprehensive survey; skim for taxonomy)
- **PyTorch tutorial:** [Knowledge Distillation Tutorial](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
- **Note:** For regression tasks, the temperature/softmax formulation doesn't apply directly. Focus on MSE between outputs and intermediate feature matching.

---

## 6. Binary & Ternary Neural Networks

This is the most exotic topic in the project. Take time here.

### Concepts
- **Binary weights** — weights are constrained to {-1, +1}; multiply-accumulate becomes XNOR + popcount, which is dramatically faster on hardware that supports it
- **Ternary weights (TWN)** — weights are {-1, 0, +1}; the zero allows the network to "ignore" connections, often giving significantly better accuracy than pure binary
- **Binary activations** — activations are also binarized using the sign function; combined with binary weights, the entire forward pass uses bitwise operations
- **The sign function and STE** — `sign()` has zero gradient almost everywhere; the straight-through estimator passes gradients through as if sign were the identity function (within a clipping range)
- **Scaling factors** — real-valued scaling factors per channel partially recover the representational loss of binarization
- **Mixed-precision BNNs** — keep the first layer (which sees raw pixels) and the last layer (which produces the final output) in FP32; binarize only the middle layers
- **XNOR-popcount** — the hardware primitive that makes BNNs fast: XNOR two bit-packed vectors, then count the 1-bits; this replaces dot product
- **FPGA deployment** — BNNs map naturally to FPGA LUTs; FINN framework compiles BNNs to streaming dataflow architectures

### Exercises
- Implement a binary linear layer from scratch: binarize weights with `sign()`, implement forward pass with XNOR logic (in Python/NumPy first, not on FPGA)
- Train a small binary network on MNIST using Brevitas; compare accuracy vs an FP32 network of the same architecture
- Implement the straight-through estimator manually: verify that gradients flow correctly by checking against numerical gradients
- Train Model 8 variants: full binary, ternary, and mixed-precision (FP32 first/last); compare all three on 300W-LP
- Calculate the theoretical speedup: for a 3×3×64×64 conv layer, how many bit operations does XNOR-popcount use vs how many multiply-accumulates does FP32 use?

### Resources
- **Paper:** [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks — Rastegari et al. 2016](https://arxiv.org/abs/1603.05279) (the foundational BNN paper)
- **Paper:** [Ternary Weight Networks — Li et al. 2016](https://arxiv.org/abs/1605.04711)
- **Paper:** [FINN: A Framework for Fast, Scalable Binarized Neural Network Inference — Umuroglu et al. 2017](https://arxiv.org/abs/1612.07119) (FPGA deployment framework)
- **Paper:** [Binary Neural Networks: A Survey — Qin et al. 2020](https://arxiv.org/abs/2004.03333) (comprehensive survey; skim for the accuracy vs efficiency landscape)
- **Library docs:** [Brevitas GitHub](https://github.com/Xilinx/brevitas) — PyTorch library for quantized and binary neural network training
- **Library docs:** [Larq GitHub](https://github.com/larq/larq) — TensorFlow/Keras library for BNNs (alternative to Brevitas)
- **Video:** [MIT HAN Lab — Binary / Ternary Quantization](https://www.youtube.com/watch?v=Qr2AKxsYpJQ) — covers BNNs in the context of extreme quantization
- **Blog:** [Understanding Binary Neural Networks](https://sushscience.wordpress.com/2017/10/01/understanding-binary-neural-networks/) — gentle introduction with diagrams

---

## 7. Edge Deployment & TFLite

### Concepts
- **TFLite conversion pipeline** — PyTorch → ONNX → TFLite (or PyTorch → TFLite via `ai_edge_torch`)
- **XNNPACK delegate** — TFLite's optimized CPU backend for ARM; uses NEON SIMD instructions
- **Operator compatibility** — not all PyTorch ops have TFLite equivalents; custom ops or op fusion may be needed
- **Profiling on-device** — using TFLite's benchmark tool and `/proc/self/status` for peak memory
- **NEON and FP16** — ARM's SIMD instruction set; some Pi 4 operations are faster in FP16 via NEON, others are not

### Exercises
- Convert a simple PyTorch model to TFLite via ONNX; run it on desktop and verify output matches
- Use TFLite's `benchmark_model` tool to profile inference time on the Pi
- Compare FP32 vs FP16 vs INT8 TFLite models on the Pi — measure actual latency, not just model size
- Profile peak memory using `/proc/self/status` VmPeak during inference

### Resources
- **TFLite docs:** [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- **Tool:** [ai_edge_torch](https://github.com/google-ai-edge/ai-edge-torch) — direct PyTorch to TFLite conversion
- **TFLite docs:** [XNNPACK delegate](https://www.tensorflow.org/lite/performance/xnnpack)
- **TFLite docs:** [Model Benchmark Tool](https://www.tensorflow.org/lite/performance/measurement)

---

## 8. Pareto Analysis & Experiment Design

### Concepts
- **Pareto frontier** — the set of model configurations where no other configuration is better on all metrics simultaneously
- **Multi-objective tradeoffs** — accuracy vs latency vs power vs memory; a 2D Pareto plot (MAE vs ms) is the minimum; 3D with power is ideal
- **Controlled comparison** — same dataset, same preprocessing, same test split, same measurement harness; the only variable is the model/technique
- **Statistical significance** — run inference multiple times, report mean ± std; a 0.5ms difference on a noisy measurement is not meaningful

### Exercises
- Generate a synthetic Pareto plot with random data points; implement the algorithm to find the Pareto frontier
- After training your first two models, plot them on the MAE vs latency plane; start building the real Pareto plot early

### Resources
- **Blog:** [Multi-Objective Optimization explained simply](https://en.wikipedia.org/wiki/Multi-objective_optimization) (Wikipedia is actually clear on this one)
- **Library:** `matplotlib` for plotting; consider `plotly` for interactive 3D Pareto plots

---

## Recommended Reading Order

1. **MobileNetV2 paper** (Section 1) → understand the backbone before using it
2. **Jacob et al. 2018** (Sections 1–3) → quantization math before running PTQ/QAT
3. **Nagel et al. 2021 white paper** (Sections 2–4) → deeper quantization understanding
4. **Li et al. 2017 pruning paper** → before pruning experiments
5. **Hinton et al. 2015 distillation paper** → short, read the whole thing
6. **Rastegari et al. 2016 XNOR-Net** → before binary network experiments
7. **Umuroglu et al. 2017 FINN** → only if pursuing FPGA deployment

---

## Video Course Recommendation

The **MIT HAN Lab TinyML and Efficient Deep Learning** course (available on YouTube) covers quantization, pruning, distillation, and binary networks in a single coherent course. It's the single best resource that aligns with this project's scope. Watch lectures 4–9.
