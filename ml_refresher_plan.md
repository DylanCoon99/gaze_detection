# 1-Week ML Refresher Plan

## Day 1: Math Foundations Refresh
**Concepts:** Linear algebra (vectors, matrices, dot products, eigenvalues), calculus (partial derivatives, chain rule, gradients), probability (Bayes' theorem, distributions)

**Exercises:**
- Implement matrix multiplication from scratch in NumPy, then compare with `np.matmul`
- Compute gradients by hand for `f(x,y) = 3x²y + y³`, verify with a numerical gradient checker
- Derive the posterior for a coin-flip problem using Bayes' theorem

---

## Day 2: Linear Models
**Concepts:** Linear regression, gradient descent, cost functions (MSE), feature scaling, bias-variance tradeoff

**Exercises:**
- Implement linear regression with gradient descent from scratch (no sklearn)
- Plot the loss curve over iterations, experiment with learning rates (too high, too low, just right)
- Add polynomial features, observe overfitting vs underfitting on a synthetic dataset
- Compare your implementation against `sklearn.linear_model.LinearRegression`

---

## Day 3: Classification
**Concepts:** Logistic regression, sigmoid function, cross-entropy loss, decision boundaries, evaluation metrics (accuracy, precision, recall, F1, confusion matrix)

**Exercises:**
- Implement logistic regression from scratch with gradient descent
- Train on a 2D dataset and plot the decision boundary
- Evaluate with a confusion matrix on a class-imbalanced dataset — observe why accuracy alone is misleading
- Implement ROC curve plotting, compute AUC

---

## Day 4: Trees, Ensembles & Regularization
**Concepts:** Decision trees, random forests, boosting (XGBoost/gradient boosting), L1/L2 regularization, cross-validation

**Exercises:**
- Train a decision tree on the Iris dataset, visualize it with `sklearn.tree.plot_tree`
- Compare a single tree vs random forest vs gradient boosting on a tabular dataset (e.g., Titanic or housing prices)
- Add L1/L2 regularization to your Day 2 linear regression, plot how coefficients shrink
- Implement k-fold cross-validation manually, compare against `sklearn.model_selection.cross_val_score`

---

## Day 5: Neural Networks Basics
**Concepts:** Perceptrons, multi-layer networks, activation functions (ReLU, sigmoid, softmax), backpropagation, loss functions

**Exercises:**
- Implement a 2-layer neural network from scratch (forward pass + backprop) to classify XOR
- Swap activation functions and observe the effect on training
- Reimplement the same network in PyTorch, compare the code
- Train a simple feedforward net on MNIST, aim for >95% accuracy

---

## Day 6: Practical Deep Learning
**Concepts:** CNNs (convolutions, pooling, architectures), dropout, batch normalization, transfer learning, optimizers (SGD, Adam)

**Exercises:**
- Build a small CNN in PyTorch for CIFAR-10, experiment with depth and filters
- Add dropout and batch norm, measure the effect on validation accuracy
- Fine-tune a pretrained ResNet on a small custom dataset (e.g., 2-3 classes from your webcam or a Kaggle dataset)
- Compare SGD vs Adam on the same architecture

---

## Day 7: Unsupervised Learning & Putting It Together
**Concepts:** K-means, PCA, dimensionality reduction, clustering evaluation (silhouette score), ML pipeline design (train/val/test splits, data leakage, feature engineering)

**Exercises:**
- Run PCA on a high-dimensional dataset, plot explained variance and pick the right number of components
- Implement k-means from scratch, compare against sklearn
- Build an end-to-end ML pipeline: load data, explore, engineer features, train multiple models, evaluate, select the best one
- Write a 1-page summary of what you'd do differently vs. what you learned in college

---

## Recommended Resources
- **Fast.ai Practical Deep Learning** — great top-down refresher
- **3Blue1Brown "Essence of Linear Algebra"** and **"Neural Networks"** YouTube series — best visual intuition
- **Andrew Ng's Coursera ML course** — if you want structured lectures alongside this
- **Kaggle Learn** — short hands-on micro-courses for any topic above

The plan front-loads fundamentals (Days 1-3) so the later days build on solid ground. Adjust pacing based on what you remember — if Day 1 math feels easy, move faster into Day 2.
