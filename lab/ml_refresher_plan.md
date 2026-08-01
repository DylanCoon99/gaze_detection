# ML Refresher Plan (Condensed for Gaze Detection Project)

Days 1–3 covered math foundations, linear models, classification, NumPy, floating point, and FLOP counting. All complete.

Days 4–6 focus on what's needed for the project: neural networks, PyTorch, CNNs, transfer learning, and bridging into Model 2.

---

## Day 1: Math Foundations Refresh ✓

### Exercises
- [x] Implement matrix multiplication from scratch in NumPy, then compare with `np.matmul`
- [x] Compute gradients by hand for `f(x,y) = 3x²y + y³`, verify with a numerical gradient checker: `(f(x+h) - f(x-h)) / 2h`
- [x] ~~Implement a numerical gradient checker function that works for any scalar function~~ (skipped — already solid on gradients)
- [x] Derive the posterior for a coin-flip problem using Bayes' theorem (prior = Beta(2,2), observe 7 heads in 10 flips)
- [x] Show that minimizing MSE is equivalent to maximum likelihood estimation under Gaussian noise — derive it
- [x] Compute eigenvalues/eigenvectors of a 2×2 matrix by hand, verify with `np.linalg.eig`

---

## Day 2: Linear Models ✓

### Exercises
- [x] Generate synthetic data: `y = 3x + 7 + noise`; implement linear regression with batch gradient descent from scratch (no sklearn)
- [x] Plot the loss curve over iterations; try learning rates of 0.0001, 0.01, 0.1, and 1.0 — observe convergence, slow convergence, and divergence
- [x] Implement the normal equation; verify it gives the same weights as gradient descent
- [x] Implement mini-batch gradient descent (batch size 32); compare convergence speed vs batch GD
- [x] Generate polynomial data: `y = 0.5x³ - 2x² + x + noise`; fit polynomials of degree 1, 3, 5, 9, 15 — plot training vs validation error for each degree
- [x] Implement standardization from scratch; show that gradient descent converges faster on standardized features
- [x] Compare your implementation against `sklearn.linear_model.LinearRegression` — verify identical results

---

## Day 3: Classification, NumPy, Floating Point & FLOPs ✓

### Exercises
- [x] NumPy fundamentals: array creation, indexing, broadcasting, matrix ops, axis aggregations, reshaping, random/practical patterns
- [x] Implement the sigmoid function; plot it and verify it approaches 0 and 1 at extremes
- [x] Implement binary cross-entropy loss from scratch; compute it for a few example predictions
- [x] Derive the gradient of BCE loss w.r.t. weights on paper, then implement logistic regression with gradient descent
- [x] Generate a 2D dataset (two Gaussian blobs with some overlap); train your logistic regression and plot the decision boundary
- [x] Create an imbalanced dataset (95% class 0, 5% class 1); train a model and compute accuracy, precision, recall, and F1 — observe why accuracy is misleading
- [x] Implement ROC curve plotting from scratch: vary threshold from 0 to 1, compute TPR/FPR at each point, plot, and compute AUC using the trapezoidal rule
- [x] Implement softmax and categorical cross-entropy; train a multiclass logistic regression on a 3-class dataset
- [x] Compare your implementation against `sklearn.linear_model.LogisticRegression`
- [x] Floating point behavior: precision loss, overflow/underflow, safe sigmoid, FP32 vs FP16 vs FP64 memory and precision tradeoffs
- [x] FLOP counting: dot products, matrix multiplies, FC layers, conv layers, depthwise separable conv savings

---

## Day 4: Neural Networks + PyTorch

### Topics to Cover
- **Backpropagation**
  - Forward pass → loss → backward pass (chain rule through the computation graph)
  - Gradient flow: why deep networks with sigmoid suffer from vanishing gradients
  - Computational graph: each operation has a local gradient; backprop multiplies them along paths
- **Activation Functions**
  - ReLU: `max(0, z)` — the default for hidden layers; fast, avoids vanishing gradients
  - Sigmoid/tanh — awareness of why they're problematic in deep networks
  - He initialization for ReLU networks: `sqrt(2/fan_in)`
- **PyTorch Fundamentals**
  - Tensors, autograd, `loss.backward()`, `optimizer.step()`
  - `nn.Module`, `nn.Linear`, `nn.ReLU`, `nn.Sequential`
  - DataLoader, Dataset — batched training pipeline
  - Training loop pattern: forward → loss → backward → step → zero_grad

### Exercises
- [x] Implement a 2-layer neural network from scratch (NumPy): forward pass + backprop on XOR
- [x] Swap activation functions (sigmoid vs ReLU); plot training loss curves — observe vanishing gradient effect
- [x] Reimplement the same network in PyTorch using `nn.Module`; compare the code
- [x] Train a feedforward network on MNIST in PyTorch (2 hidden layers, 128 units); aim for >95% accuracy
- [x] Implement the "overfit one batch" sanity check before full training

### Resources
- **3Blue1Brown "Neural Networks"** (YouTube, 4 videos) — best visual intro to neural networks and backpropagation
- **Andrej Karpathy: "micrograd"** (YouTube) — builds autograd + neural net from scratch
- **PyTorch: 60 Minute Blitz** — https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
- **Stanford CS231n: Backpropagation** — http://cs231n.github.io/optimization-2/

---

## Day 5: CNNs + Transfer Learning

### Topics to Cover
- **Convolutional Neural Networks**
  - Convolution: sliding kernel, weight sharing, spatial locality
  - Padding, stride, pooling — output size formula: `(W - K + 2P) / S + 1`
  - Hierarchical features: edges → textures → parts → objects
  - Depthwise separable convolutions (MobileNet) — you already calculated the FLOP savings
- **Regularization for CNNs**
  - Dropout: randomly zero activations during training
  - Batch normalization: stabilize training, allow higher LR
  - Data augmentation: random flips, crops, color jitter
- **Transfer Learning**
  - Frozen backbone + trainable head (feature extraction)
  - Fine-tuning: unfreeze with lower LR
  - This is exactly Models 2 and 3 in your project
- **Optimizers**
  - SGD with momentum vs Adam — when to use each
  - Learning rate schedulers: cosine annealing

### Exercises
- [x] Build a CNN in PyTorch for CIFAR-10 (3 conv layers + pooling + FC head); train and report accuracy
- [x] Add batch norm and dropout; measure the effect on validation accuracy
- [x] Train with SGD vs Adam; plot both training curves
- [x] Fine-tune a pretrained MobileNetV2 on a small dataset; compare accuracy vs training from scratch
- [x] Add data augmentation (random flip, crop, color jitter); measure accuracy improvement

### Resources
- **Stanford CS231n: Convolutional Neural Networks** — http://cs231n.github.io/convolutional-networks/
- **PyTorch: Training a Classifier** — https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- **PyTorch: Finetuning TorchVision Models** — https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
- **Blog: "A Recipe for Training Neural Networks"** — Karpathy: https://karpathy.github.io/2019/04/25/recipe/
- **Paper: MobileNetV2** — Sandler et al. 2018: https://arxiv.org/abs/1801.04381

---

## Day 6: Project Bridge

### Topics to Cover
- **Head Pose Regression**
  - Input: 112×112 RGB face crop → Output: yaw and pitch (2 continuous values)
  - Loss: MSE on yaw and pitch
  - This is a regression task using a CNN backbone — same architecture as classification, but linear output + MSE instead of softmax + CE
- **Model Profiling**
  - `torchinfo.summary()` — verify parameter counts and FLOP estimates
  - Verify your Day 3 FLOP calculations match real models
- **Export Pipeline**
  - PyTorch → ONNX export as a deployment sanity check

### Exercises
- [ ] Load a subset of 300W-LP (or use a face dataset); write a PyTorch Dataset and DataLoader
- [ ] Attach a regression head (2 outputs: yaw, pitch) to a frozen MobileNetV2 backbone — this is Model 2
- [ ] Train on face data with MSE loss; report MAE on a validation split
- [ ] Profile the model with `torchinfo`; verify parameter count and FLOP estimates
- [ ] Export the trained model to ONNX; reload and verify predictions match

### Resources
- **PyTorch: Custom Datasets** — https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- **torchinfo** — `pip install torchinfo`; `summary(model, input_size=(1,3,112,112))`
- **PyTorch: Export to ONNX** — https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
- **300W-LP dataset** — for face pose data

---

## General Resources (Use Anytime)

| Resource | Best For |
|---|---|
| 3Blue1Brown (YouTube) | Visual intuition for math and neural networks |
| Stanford CS231n Course Notes | CNNs, training, and practical deep learning |
| Andrej Karpathy (YouTube/blog) | Building things from scratch, practical advice |
| PyTorch Tutorials | Official tutorials for neural network code |
| Fast.ai Practical Deep Learning | Top-down, code-first deep learning |
