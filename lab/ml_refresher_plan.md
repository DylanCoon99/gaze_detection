# 1-Week ML Refresher Plan

---

## Day 1: Math Foundations Refresh

### Topics to Cover
- **Linear Algebra**
  - Vectors: dot product, cross product, norms (L1, L2)
  - Matrices: multiplication, transpose, inverse, determinant
  - Eigenvalues and eigenvectors — what they represent geometrically
  - Matrix decomposition (SVD) — high-level intuition, not full derivation
- **Calculus**
  - Partial derivatives and the gradient vector
  - The chain rule — critical for understanding backpropagation later
  - Directional derivatives and why gradients point in the direction of steepest ascent
  - Jacobians — what they are conceptually (you'll need this for backprop through layers)
- **Probability & Statistics**
  - Bayes' theorem and prior/posterior/likelihood terminology
  - Common distributions: Gaussian, Bernoulli, uniform
  - Expectation, variance, covariance
  - Maximum likelihood estimation (MLE) — connects to loss functions later

### Exercises
- [x] Implement matrix multiplication from scratch in NumPy, then compare with `np.matmul`
- [x] Compute gradients by hand for `f(x,y) = 3x²y + y³`, verify with a numerical gradient checker: `(f(x+h) - f(x-h)) / 2h`
- [x] ~~Implement a numerical gradient checker function that works for any scalar function~~ (skipped — already solid on gradients)
- [x] Derive the posterior for a coin-flip problem using Bayes' theorem (prior = Beta(2,2), observe 7 heads in 10 flips)
- [x] Show that minimizing MSE is equivalent to maximum likelihood estimation under Gaussian noise — derive it
- [x] Compute eigenvalues/eigenvectors of a 2×2 matrix by hand, verify with `np.linalg.eig`

### Resources
- **3Blue1Brown "Essence of Linear Algebra"** (YouTube, ~3 hrs total) — watch chapters 1–7 for vectors, matrices, determinants, eigenvalues
- **3Blue1Brown "Essence of Calculus"** (YouTube) — chapters on derivatives and chain rule
- **Khan Academy: Multivariable Calculus** — gradient and partial derivatives sections
- **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong) — Chapter 2 (Linear Algebra), Chapter 5 (Vector Calculus), Chapter 6 (Probability). Free PDF: https://mml-book.github.io/
- **NumPy documentation** — `np.linalg` module reference for verifying your implementations

---

## Day 2: Linear Models

### Topics to Cover
- **Linear Regression**
  - The hypothesis function: `y = Xw + b`
  - Mean Squared Error (MSE) loss and why it's convex for linear models
  - Normal equation (closed-form solution): `w = (X^T X)^{-1} X^T y` — when it works and when it doesn't (singular matrices, large feature counts)
- **Gradient Descent**
  - Batch gradient descent: compute gradient over entire dataset, update weights
  - Stochastic gradient descent (SGD): update per sample — noisier but faster
  - Mini-batch gradient descent: the practical middle ground
  - Learning rate: too high (divergence), too low (slow convergence), just right
  - Convergence criteria: loss plateau, gradient norm threshold
- **Feature Engineering**
  - Feature scaling: why it matters for gradient descent (elongated vs spherical contours)
  - Standardization (zero mean, unit variance) vs min-max normalization
  - Polynomial features: `[x] → [x, x², x³]` — linear model on non-linear features
- **Bias-Variance Tradeoff**
  - Underfitting (high bias): model too simple to capture the pattern
  - Overfitting (high variance): model memorizes training noise
  - Training error vs validation error curves as model complexity increases
  - How polynomial degree controls this tradeoff

### Exercises
- [ ] Generate synthetic data: `y = 3x + 7 + noise`; implement linear regression with batch gradient descent from scratch (no sklearn)
- [ ] Plot the loss curve over iterations; try learning rates of 0.0001, 0.01, 0.1, and 1.0 — observe convergence, slow convergence, and divergence
- [ ] Implement the normal equation; verify it gives the same weights as gradient descent
- [ ] Implement mini-batch gradient descent (batch size 32); compare convergence speed vs batch GD
- [ ] Generate polynomial data: `y = 0.5x³ - 2x² + x + noise`; fit polynomials of degree 1, 3, 5, 9, 15 — plot training vs validation error for each degree
- [ ] Implement standardization from scratch; show that gradient descent converges faster on standardized features
- [ ] Compare your implementation against `sklearn.linear_model.LinearRegression` — verify identical results

### Resources
- **Andrew Ng's Machine Learning Specialization** (Coursera) — Week 1–2 covers linear regression and gradient descent with excellent visual explanations
- **Stanford CS229 Lecture Notes** — Section 1 (Linear Regression, LMS algorithm, Normal Equations): https://cs229.stanford.edu/notes2022fall/main_notes.pdf
- **Fast.ai Practical Deep Learning** — Lesson 1 introduction covers the training loop concept
- **Blog: "Gradient Descent, Step by Step"** — StatQuest (YouTube) — visual walkthrough
- **Scikit-learn User Guide: Linear Models** — https://scikit-learn.org/stable/modules/linear_model.html — use as reference to check your work
- **Python Data Science Handbook** (VanderPlas) — Chapter 5: covers linear regression with sklearn

---

## Day 3: Classification

### Topics to Cover
- **Logistic Regression**
  - The sigmoid function: `σ(z) = 1 / (1 + e^{-z})` — maps any real number to (0, 1)
  - Decision boundary: where `σ(Xw + b) = 0.5`, i.e., `Xw + b = 0`
  - Why MSE doesn't work well for classification (non-convex loss landscape)
- **Cross-Entropy Loss**
  - Binary cross-entropy: `L = -[y log(ŷ) + (1-y) log(1-ŷ)]`
  - Why it works: large penalty when confident and wrong, small penalty when confident and right
  - Connection to maximum likelihood estimation under Bernoulli distribution
  - Multiclass extension: categorical cross-entropy with softmax
- **Optimization**
  - Gradient of cross-entropy loss w.r.t. weights — derive it
  - Gradient descent for logistic regression (same algorithm as Day 2, different gradient)
- **Evaluation Metrics**
  - Confusion matrix: TP, FP, TN, FN
  - Accuracy — and why it's misleading with imbalanced classes (99% accuracy on 99/1 split)
  - Precision: of all predicted positives, how many are correct
  - Recall: of all actual positives, how many did we catch
  - F1 score: harmonic mean of precision and recall
  - ROC curve: TPR vs FPR at all thresholds; AUC as a threshold-independent metric
  - Precision-recall curve: better than ROC for imbalanced datasets
- **Multiclass Classification**
  - Softmax function: generalizes sigmoid to K classes
  - One-vs-rest vs softmax approaches

### Exercises
- [ ] Implement the sigmoid function; plot it and verify it approaches 0 and 1 at extremes
- [ ] Implement binary cross-entropy loss from scratch; compute it for a few example predictions
- [ ] Derive the gradient of BCE loss w.r.t. weights on paper, then implement logistic regression with gradient descent
- [ ] Generate a 2D dataset (two Gaussian blobs with some overlap); train your logistic regression and plot the decision boundary
- [ ] Create an imbalanced dataset (95% class 0, 5% class 1); train a model and compute accuracy, precision, recall, and F1 — observe why accuracy is misleading
- [ ] Implement ROC curve plotting from scratch: vary threshold from 0 to 1, compute TPR/FPR at each point, plot, and compute AUC using the trapezoidal rule
- [ ] Implement softmax and categorical cross-entropy; train a multiclass logistic regression on a 3-class dataset
- [ ] Compare your implementation against `sklearn.linear_model.LogisticRegression`

### Resources
- **Andrew Ng's Machine Learning Specialization** (Coursera) — Week 3 covers logistic regression and classification
- **Stanford CS229 Lecture Notes** — Section 2 (Classification and Logistic Regression)
- **StatQuest: Logistic Regression** (YouTube) — clear visual explanation
- **StatQuest: ROC and AUC** (YouTube) — best short explanation of ROC curves
- **StatQuest: Confusion Matrix** (YouTube)
- **Scikit-learn: Metrics and Scoring** — https://scikit-learn.org/stable/modules/model_evaluation.html — reference for all metrics
- **Blog: "Understanding Binary Cross-Entropy"** — Daniel Godoy (towardsdatascience) — derives BCE from MLE
- **Google's Machine Learning Crash Course: Classification** — https://developers.google.com/machine-learning/crash-course/classification

---

## Day 4: Trees, Ensembles & Regularization

### Topics to Cover
- **Decision Trees**
  - Splitting criteria: Gini impurity and information gain (entropy)
  - Recursive binary splitting: greedily choose the split that maximizes purity
  - Tree depth as the complexity knob — deep trees overfit, shallow trees underfit
  - Feature importance: how much each feature reduces impurity across all splits
- **Random Forests**
  - Bagging (bootstrap aggregating): train many trees on random subsets of data
  - Feature randomness: each split considers only a random subset of features
  - Why combining many weak learners reduces variance without increasing bias
  - Out-of-bag (OOB) error as a free validation estimate
- **Gradient Boosting**
  - Boosting vs bagging: sequential (correcting errors) vs parallel (averaging)
  - Gradient boosting: each new tree fits the residual errors of the ensemble so far
  - Learning rate (shrinkage): how much each tree contributes — smaller = more trees needed but better generalization
  - XGBoost / LightGBM as optimized implementations
- **Regularization**
  - L2 regularization (Ridge): adds `λ||w||²` to loss — shrinks all weights toward zero
  - L1 regularization (Lasso): adds `λ||w||₁` to loss — drives some weights exactly to zero (feature selection)
  - Elastic Net: combines L1 and L2
  - The regularization parameter λ: too high = underfitting, too low = overfitting
  - Regularization as a Bayesian prior (L2 = Gaussian prior on weights)
- **Cross-Validation**
  - K-fold cross-validation: split data into K folds, train on K-1, validate on 1, rotate
  - Why we need it: single train/val split is high variance for small datasets
  - Stratified K-fold: preserve class proportions in each fold

### Exercises
- [ ] Train a decision tree on the Iris dataset; visualize it with `sklearn.tree.plot_tree`; manually trace a prediction through the tree
- [ ] Train trees of depth 1, 3, 5, 10, None (unlimited); plot training vs validation accuracy for each — observe overfitting
- [ ] Train a random forest (100 trees) on the same data; compare accuracy and variance vs a single tree
- [ ] Train gradient boosting (XGBoost or sklearn's `GradientBoostingClassifier`) on a tabular dataset (Titanic or housing prices); tune learning rate and n_estimators
- [ ] Compare single tree vs random forest vs gradient boosting on the same dataset with the same features — tabulate results
- [ ] Go back to your Day 2 linear regression; add L2 regularization to the loss and gradient; retrain and plot how coefficients shrink as λ increases
- [ ] Implement L1 regularization (use subgradient or proximal gradient); observe that some weights go exactly to zero
- [ ] Implement k-fold cross-validation manually (split indices, loop, aggregate scores); compare against `sklearn.model_selection.cross_val_score`
- [ ] Use cross-validation to select the best regularization strength λ for Ridge regression

### Resources
- **StatQuest: Decision Trees** (YouTube) — clearest explanation of Gini and entropy splitting
- **StatQuest: Random Forests** (YouTube)
- **StatQuest: Gradient Boost** (YouTube, Parts 1–4) — builds intuition step by step
- **StatQuest: Regularization** (YouTube, Ridge/Lasso/Elastic Net series)
- **An Introduction to Statistical Learning (ISLR)** — Chapter 8 (Trees), Chapter 6 (Regularization). Free PDF: https://www.statlearning.com/
- **XGBoost documentation** — https://xgboost.readthedocs.io/ — reference for gradient boosting
- **Scikit-learn: Decision Trees** — https://scikit-learn.org/stable/modules/tree.html
- **Scikit-learn: Ensemble Methods** — https://scikit-learn.org/stable/modules/ensemble.html
- **Andrew Ng's Machine Learning Specialization** (Coursera) — Week 4 covers decision trees and ensembles

---

## Day 5: Neural Networks Basics

### Topics to Cover
- **The Perceptron**
  - Single neuron: weighted sum + bias + activation → output
  - Why a single perceptron can only learn linearly separable functions (XOR problem)
- **Multi-Layer Networks (MLPs)**
  - Hidden layers: stacking neurons to learn non-linear functions
  - Universal approximation theorem: one hidden layer can approximate any continuous function (given enough neurons) — but deeper is more parameter-efficient
- **Activation Functions**
  - Sigmoid: `σ(z) = 1/(1+e^{-z})` — squashes to (0,1), suffers from vanishing gradients
  - Tanh: squashes to (-1,1), zero-centered but still saturates
  - ReLU: `max(0, z)` — simple, fast, avoids vanishing gradient for positive values, but "dying ReLU" problem
  - Leaky ReLU, ELU — fixes for dying ReLU
  - Softmax: output layer for multiclass; turns logits into a probability distribution
- **Backpropagation**
  - Forward pass: compute output layer by layer
  - Loss computation: compare output to target
  - Backward pass: chain rule applied recursively through the computation graph
  - Gradient flow: why deep networks with sigmoid activations suffer from vanishing gradients
  - Computational graph perspective: each operation has a local gradient; backprop multiplies them along paths
- **Loss Functions**
  - MSE for regression
  - Cross-entropy for classification
  - Why the choice of loss + output activation matters (softmax + CE, sigmoid + BCE, linear + MSE)
- **Weight Initialization**
  - Why zero initialization doesn't work (symmetry breaking)
  - Xavier/Glorot initialization: scale weights by `1/sqrt(fan_in)`
  - He initialization: `sqrt(2/fan_in)` — designed for ReLU

### Exercises
- [ ] Implement a single perceptron; train it on AND, OR gates; show it fails on XOR
- [ ] Implement a 2-layer neural network from scratch (NumPy only): forward pass, loss, backward pass with chain rule
- [ ] Train your network on XOR — verify it learns the correct decision boundary
- [ ] Implement backpropagation step by step: compute `dL/dW2`, `dL/db2`, `dL/dW1`, `dL/db1` by hand for a 2-layer network with ReLU hidden activation, then implement in code
- [ ] Swap activation functions (sigmoid, tanh, ReLU) in your network; plot training loss curves — observe how sigmoid trains slower due to vanishing gradients
- [ ] Implement Xavier and He initialization; compare training speed vs random uniform initialization on a 5-layer network
- [ ] Reimplement the same 2-layer network in PyTorch using `nn.Module`; compare the code side by side with your NumPy version
- [ ] Train a simple feedforward network (2 hidden layers, 128 units each) on MNIST in PyTorch; aim for >95% accuracy
- [ ] Visualize the learned first-layer weights on MNIST — they should look like stroke/edge detectors

### Resources
- **3Blue1Brown "Neural Networks"** (YouTube, 4 videos) — the best visual introduction to neural networks and backpropagation
- **Stanford CS231n: Backpropagation** — http://cs231n.github.io/optimization-2/ — computational graph explanation of backprop
- **Stanford CS231n: Neural Networks Part 1** — http://cs231n.github.io/neural-networks-1/ — architectures and activation functions
- **Michael Nielsen's "Neural Networks and Deep Learning"** — Chapter 1 and 2. Free online: http://neuralnetworksanddeeplearning.com/
- **PyTorch: 60 Minute Blitz** — https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html — official intro tutorial
- **Andrej Karpathy: "micrograd"** (YouTube) — builds a full autograd engine and neural network from scratch in ~2.5 hours; excellent for understanding backprop at a code level
- **Blog: "Yes you should understand backprop"** — Andrej Karpathy: https://karpathy.medium.com/yes-you-should-understand-backprop-e1cc5f2edf7c
- **Paper: "Understanding the difficulty of training deep feedforward neural networks"** — Glorot & Bengio 2010 (Xavier initialization)

---

## Day 6: Practical Deep Learning

### Topics to Cover
- **Convolutional Neural Networks (CNNs)**
  - Convolution operation: sliding a kernel over the input; weight sharing and spatial locality
  - Feature maps: each filter detects one type of pattern; stack many filters per layer
  - Padding (same vs valid) and stride — how they affect output spatial dimensions
  - Pooling: max pooling and average pooling — downsample spatial dimensions, add translation invariance
  - Output size formula: `(W - K + 2P) / S + 1`
  - Hierarchical feature learning: edges → textures → parts → objects as you go deeper
- **Regularization Techniques**
  - Dropout: randomly zero activations during training; forces redundancy; acts as ensemble
  - Batch normalization: normalize activations per mini-batch; stabilizes training, allows higher LR
  - Data augmentation: random flips, crops, color jitter — cheapest regularization
  - Early stopping: monitor validation loss, stop when it starts increasing
- **Optimizers**
  - SGD with momentum: exponential moving average of gradients smooths updates
  - Adam: adaptive learning rates per parameter (combines momentum + RMSprop)
  - Learning rate schedulers: step decay, cosine annealing, warmup
  - When to use SGD vs Adam: Adam converges faster but SGD often generalizes better with tuning
- **Transfer Learning**
  - Pretrained backbones: use ImageNet-trained features as a starting point
  - Feature extraction: freeze backbone, train only the head
  - Fine-tuning: unfreeze backbone with lower LR
  - When transfer learning helps vs hurts (domain similarity)
- **Practical Training Discipline**
  - Overfit one batch first: verify the model can memorize before trying to generalize
  - Monitor train AND val loss: divergence = overfitting
  - Gradient clipping: prevent exploding gradients in deep networks
  - Reproducibility: set random seeds, log hyperparameters

### Exercises
- [ ] Compute by hand: for a 32×32 input with a 3×3 kernel, stride 1, padding 1 — what's the output size? Repeat for stride 2, no padding.
- [ ] Build a small CNN in PyTorch for CIFAR-10 (3 conv layers + pooling + FC head); train it and report validation accuracy
- [ ] Add dropout (0.5) after FC layers; compare validation accuracy with and without
- [ ] Add batch normalization after each conv layer; observe that you can increase the learning rate without divergence
- [ ] Train the same CNN with SGD (momentum=0.9) vs Adam; plot both training curves — note Adam converges faster initially
- [ ] Implement a cosine annealing learning rate schedule; compare final accuracy vs constant LR
- [ ] Fine-tune a pretrained ResNet-18 on a small dataset (e.g., CIFAR-10 or a custom 5-class dataset from Kaggle); compare accuracy vs training from scratch
- [ ] Implement the "overfit one batch" sanity check: take 1 mini-batch, train until loss → 0; verify the model has capacity
- [ ] Add data augmentation (random horizontal flip, random crop) to your CIFAR-10 pipeline; measure the accuracy improvement

### Resources
- **Stanford CS231n: Convolutional Neural Networks** — http://cs231n.github.io/convolutional-networks/ — the best written explanation of CNNs
- **Stanford CS231n: Training Neural Networks** (Parts 1 & 2) — http://cs231n.github.io/neural-networks-3/ — batch norm, dropout, optimizers, LR schedules
- **Fast.ai Practical Deep Learning** — Lessons 1–5 cover transfer learning, training discipline, and CNNs
- **PyTorch: Training a Classifier** — https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- **PyTorch: Finetuning TorchVision Models** — https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
- **Blog: "A Recipe for Training Neural Networks"** — Andrej Karpathy: https://karpathy.github.io/2019/04/25/recipe/ — essential practical advice
- **Paper: "Batch Normalization: Accelerating Deep Network Training"** — Ioffe & Szegedy 2015
- **Paper: "Adam: A Method for Stochastic Optimization"** — Kingma & Ba 2015
- **Blog: "An overview of gradient descent optimization algorithms"** — Ruder (2016): https://ruder.io/optimizing-gradient-descent/

---

## Day 7: Unsupervised Learning & Putting It Together

### Topics to Cover
- **Dimensionality Reduction**
  - PCA: project data onto directions of maximum variance
  - How PCA works: center data → compute covariance matrix → eigenvectors are the principal components
  - Choosing the number of components: explained variance ratio (keep enough to explain 90–95%)
  - When PCA helps: visualization, denoising, speeding up downstream models
  - t-SNE / UMAP: non-linear dimensionality reduction for visualization (not for preprocessing)
- **Clustering**
  - K-means: assign points to nearest centroid, recompute centroids, repeat
  - K-means limitations: assumes spherical clusters, sensitive to initialization, must specify K
  - Choosing K: elbow method (inertia vs K) and silhouette score
  - Other clustering methods (awareness-level): DBSCAN (density-based, no need to specify K), hierarchical clustering
- **ML Pipeline Design**
  - Train / validation / test split: why you need all three (hyperparameter tuning on val, final number on test)
  - Data leakage: fitting preprocessing (e.g., scaler) on test data, using future information, target leakage
  - Feature engineering: domain knowledge → features that help the model
  - Model selection: compare multiple models on the same val set, pick the best, report on test
  - Reproducibility: fixed seeds, logged hyperparameters, versioned datasets
- **Putting It All Together**
  - The full ML workflow: problem definition → data collection → EDA → preprocessing → feature engineering → model selection → hyperparameter tuning → evaluation → deployment
  - When to use what: linear models (interpretable, fast, good baseline), trees/ensembles (tabular data king), neural networks (images, sequences, large data)

### Exercises
- [ ] Load a high-dimensional dataset (e.g., sklearn's digits dataset, 64 features); run PCA and plot explained variance ratio — pick the knee point
- [ ] Visualize the digits dataset in 2D using PCA (first 2 components); color by digit label — observe cluster structure
- [ ] Run t-SNE on the same dataset; compare the 2D visualization to PCA — note t-SNE separates clusters better but distances are not meaningful
- [ ] Implement K-means from scratch: random initialization → assign → update → repeat until convergence
- [ ] Run your K-means on 2D synthetic data (blobs); plot the cluster assignments and centroids at each iteration
- [ ] Implement the elbow method and silhouette score; use them to pick K on a dataset where you don't know the true number of clusters
- [ ] Compare your K-means against `sklearn.cluster.KMeans` — verify similar results
- [ ] Build an end-to-end ML pipeline on a real dataset (e.g., Kaggle's House Prices or Titanic):
   - [ ] Load and explore data (EDA: distributions, missing values, correlations)
   - [ ] Preprocess: handle missing values, encode categoricals, scale numerics
   - [ ] Engineer 2–3 features based on domain intuition
   - [ ] Train multiple models: linear regression/logistic regression, random forest, gradient boosting
   - [ ] Evaluate all models with cross-validation; select the best
   - [ ] Report final performance on the held-out test set
- [ ] Write a 1-page reflection: what do you understand now that you didn't in college? What's still fuzzy? What concepts connect to the gaze estimation project?

### Resources
- **StatQuest: PCA** (YouTube, main idea + step by step) — best visual explanation
- **StatQuest: K-means Clustering** (YouTube)
- **Stanford CS229: Unsupervised Learning** — lecture notes on PCA and clustering
- **Scikit-learn: Unsupervised Learning** — https://scikit-learn.org/stable/unsupervised_learning.html
- **Scikit-learn: Clustering** — https://scikit-learn.org/stable/modules/clustering.html — great comparison of algorithms with visuals
- **Kaggle Learn: Intro to Machine Learning** and **Intermediate ML** — short courses that walk through full pipelines
- **Blog: "A Few Useful Things to Know About Machine Learning"** — Pedro Domingos (2012 paper, very readable): https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf
- **Book: "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** — Geron, Chapter 8 (Dimensionality Reduction), Chapter 9 (Clustering)
- **Fast.ai: Practical Deep Learning** — Lesson 1 walkthrough of a full end-to-end project

---

## General Resources (Use Anytime)

| Resource | Best For |
|---|---|
| 3Blue1Brown (YouTube) | Visual intuition for math and neural networks |
| StatQuest (YouTube) | Clear, short explanations of individual algorithms |
| Andrew Ng's ML Specialization (Coursera) | Structured lectures with math derivations |
| Fast.ai Practical Deep Learning | Top-down, code-first deep learning |
| Stanford CS229 Lecture Notes | Rigorous math behind classical ML |
| Stanford CS231n Course Notes | CNNs, training, and practical deep learning |
| Andrej Karpathy (YouTube/blog) | Building things from scratch, practical advice |
| Scikit-learn User Guide | Reference docs for any sklearn algorithm |
| PyTorch Tutorials | Official tutorials for neural network code |
| Kaggle Learn | Short hands-on micro-courses |
| ISLR (free textbook) | Statistical learning theory with R examples |

The plan front-loads fundamentals (Days 1–3) so the later days build on solid ground. Adjust pacing based on what you remember — if Day 1 math feels easy, move faster into Day 2.
