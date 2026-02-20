# Math for AI - Example Code

This directory contains Python examples demonstrating mathematical concepts used in AI/ML.

## Files

### 01_vector_matrix_ops.py (283 lines)
**Linear Algebra Fundamentals**

Demonstrates:
- Vector space operations: basis, span, linear independence
- Matrix operations: multiplication, transpose, inverse
- Rank computation and properties
- ML application: feature matrices and weight matrices
- Visualization of vector addition and linear transformations

**Key concepts:**
- Standard basis vectors in R^n
- Linear independence checking via rank
- Matrix multiplication and properties
- Feature vectors as matrix rows in ML

**Output:** `vector_ops.png` - Vector operations visualization

---

### 02_svd_pca.py (302 lines)
**SVD and Principal Component Analysis**

Demonstrates:
- Singular Value Decomposition (SVD) with NumPy
- Low-rank matrix approximation
- PCA implementation from scratch (centering → covariance → eigen)
- Comparison with sklearn PCA
- Application to Iris dataset
- Explained variance visualization

**Key concepts:**
- SVD decomposition: A = U @ S @ V^T
- Relationship between SVD and PCA
- Principal components as eigenvectors
- Dimensionality reduction in practice

**Output:** `pca_visualization.png` - PCA results on Iris dataset

---

### 03_matrix_calculus_autograd.py (408 lines)
**Matrix Calculus and Automatic Differentiation**

Demonstrates:
- Manual gradient computation for scalar and vector functions
- Jacobian matrix computation (vector → vector)
- Hessian matrix computation (second derivatives)
- PyTorch autograd comparison
- Numerical gradient checking
- MSE loss gradients for linear regression

**Key concepts:**
- Partial derivatives and gradient vectors
- Jacobian for vector-valued functions
- Hessian for convexity analysis
- Automatic differentiation with PyTorch
- Gradient descent optimization

**Output:** `gradient_descent.png` - Gradient descent trajectory visualization

---

### 04_norms_regularization.py (400 lines)
**Norms, Distances, and Regularization**

Demonstrates:
- Lp norms (L1, L2, L∞) and their properties
- Distance metrics: Euclidean, Manhattan, Cosine, Mahalanobis
- L1 vs L2 regularization in linear regression
- Sparsity-inducing property of L1
- Unit ball visualization for different norms

**Key concepts:**
- Norm properties: non-negativity, homogeneity, triangle inequality
- L1 (Lasso) produces sparse solutions
- L2 (Ridge) shrinks coefficients smoothly
- Feature selection vs feature shrinkage
- Regularization paths

**Output:**
- `unit_balls.png` - Unit ball shapes for L1, L2, L∞
- `regularization_comparison.png` - L1 vs L2 effects

---

### 05_gradient_descent.py (271 lines)
**Gradient Descent Optimization Algorithms**

Demonstrates:
- Basic Gradient Descent (GD) optimizer
- Stochastic Gradient Descent (SGD)
- Momentum-based optimization
- Adam optimizer from scratch
- Optimization of Rosenbrock function (non-convex with narrow valley)

**Key concepts:**
- Vanilla gradient descent updates: θ = θ - lr * ∇θ
- Momentum accumulation for acceleration
- Adaptive learning rates with Adam
- Comparison of convergence behavior across optimizers

**Output:** `gradient_optimizers.png` - Optimization trajectories for different algorithms

---

### 06_optimization_constrained.py (320 lines)
**Constrained and Unconstrained Optimization**

Demonstrates:
- Unconstrained optimization with scipy.optimize methods (BFGS, CG, Newton-CG, L-BFGS-B)
- Lagrange multipliers for equality constraints
- KKT conditions for inequality constraints
- Convex vs non-convex optimization comparison

**Key concepts:**
- Analytical vs numerical optimization
- Lagrangian formulation for constraints
- First-order (gradient) and second-order (Hessian) methods
- Constraint handling in optimization problems

**Output:** `constrained_optimization.png` - Constraint visualization and optimization paths

---

### 07_probability_distributions.py (367 lines)
**Probability Distributions and Statistical Inference**

Demonstrates:
- Common probability distributions: Gaussian, Bernoulli, Poisson, Exponential
- Maximum Likelihood Estimation (MLE) for Gaussian parameters
- Maximum A Posteriori (MAP) estimation with Gaussian prior
- Bayesian update visualization

**Key concepts:**
- PDF/PMF properties for different distributions
- MLE: argmax_θ P(data | θ)
- MAP: argmax_θ P(θ | data) = argmax_θ P(data | θ) P(θ)
- Prior × Likelihood = Posterior (Bayes' rule)

**Output:** `probability_distributions.png` - Distribution PDFs and Bayesian inference

---

### 08_information_theory.py (462 lines)
**Information Theory for Machine Learning**

Demonstrates:
- Entropy as measure of uncertainty: H(X) = -Σ p(x) log p(x)
- Cross-entropy and KL divergence
- Mutual information between variables
- Connection to ML loss functions (cross-entropy loss)
- ELBO (Evidence Lower Bound) visualization for VAEs

**Key concepts:**
- Maximum entropy for uniform distributions
- KL divergence as measure of distribution difference
- Cross-entropy = Entropy + KL divergence
- ELBO = log p(x) - KL(q||p) used in variational inference

**Output:** `information_theory.png` - Entropy, KL divergence, and ELBO visualization

---

### 09_mcmc_sampling.py (308 lines)
**MCMC Sampling and Advanced Sampling Techniques**

Demonstrates:
- Rejection sampling from target distribution
- Importance sampling for expectation estimation
- Metropolis-Hastings MCMC algorithm
- Reparameterization trick (VAE-style)

**Key concepts:**
- Rejection sampling: accept/reject based on proposal distribution
- Importance sampling: weighted samples for expectation
- MCMC: Markov chain converging to target distribution
- Reparameterization: z = μ + σ * ε for gradient flow in VAEs

**Output:** `mcmc_sampling.png` - Sampling method comparisons and MCMC convergence

---

### 10_tensor_ops_einsum.py (298 lines)
**Tensor Operations and Einstein Summation**

Demonstrates:
- Tensor creation and manipulation in NumPy and PyTorch
- Einstein summation notation (einsum) for efficient operations
- Broadcasting rules and examples
- Numerical stability techniques (log-sum-exp, softmax)

**Key concepts:**
- einsum notation: implicit summation over repeated indices
- Common operations: matrix multiply, batch operations, trace, transpose
- Broadcasting: automatic shape alignment for element-wise ops
- Numerical stability: avoid overflow/underflow in exp/log

**Output:** Console output demonstrating einsum equivalences and timing comparisons

---

### 11_graph_spectral.py (372 lines)
**Graph Theory and Spectral Graph Theory**

Demonstrates:
- Graph matrix construction: adjacency, degree, Laplacian
- Spectral decomposition of graph Laplacian
- Spectral clustering algorithm
- Simple Graph Neural Network (GNN) message passing
- PageRank computation

**Key concepts:**
- Laplacian eigenvalues/eigenvectors encode graph structure
- Normalized Laplacian: L_norm = I - D^(-1/2) A D^(-1/2)
- Spectral clustering uses eigenvectors for community detection
- GNN message passing aggregates neighbor features

**Output:** `graph_spectral.png` - Graph visualization, spectral clustering, and PageRank

---

### 12_attention_math.py (419 lines)
**Attention Mechanism Mathematics**

Demonstrates:
- Scaled dot-product attention from scratch
- Multi-head attention implementation
- Positional encoding (sinusoidal)
- Attention weight visualization
- Comparison with PyTorch nn.MultiheadAttention

**Key concepts:**
- Attention formula: softmax(Q @ K^T / sqrt(d_k)) @ V
- Scaling factor sqrt(d_k) prevents softmax saturation
- Multi-head attention: parallel attention with different projections
- Positional encoding adds sequence order information to embeddings

**Output:** `attention_weights.png` - Heatmap visualization of attention weights

---

## Running the Examples

Each file is standalone and can be run independently:

```bash
python 01_vector_matrix_ops.py
python 02_svd_pca.py
# ... through 12_attention_math.py
```

## Dependencies

All examples require:
- numpy
- matplotlib
- torch (PyTorch)
- sklearn (scikit-learn)
- scipy

Install with:
```bash
pip install numpy matplotlib torch scikit-learn scipy
```

## Learning Path

**Recommended order:**
1. `01_vector_matrix_ops.py` - Linear algebra fundamentals
2. `02_svd_pca.py` - Matrix factorization and dimensionality reduction
3. `03_matrix_calculus_autograd.py` - Gradients and automatic differentiation
4. `04_norms_regularization.py` - Norms and regularization
5. `05_gradient_descent.py` - Optimization algorithms
6. `06_optimization_constrained.py` - Constrained optimization
7. `07_probability_distributions.py` - Probability and inference
8. `08_information_theory.py` - Information theory for ML
9. `09_mcmc_sampling.py` - Sampling techniques
10. `10_tensor_ops_einsum.py` - Tensor operations and einsum
11. `11_graph_spectral.py` - Graph theory and spectral methods
12. `12_attention_math.py` - Attention mechanism mathematics

## Output Files

Running the scripts will generate PNG visualizations:
- `vector_ops.png` - Vector addition and transformations
- `pca_visualization.png` - PCA projection and explained variance
- `gradient_descent.png` - Optimization trajectory
- `unit_balls.png` - Norm visualizations
- `regularization_comparison.png` - Regularization effects
- `gradient_optimizers.png` - Optimizer comparison
- `constrained_optimization.png` - Constrained optimization
- `probability_distributions.png` - Distribution PDFs
- `information_theory.png` - Entropy and KL divergence
- `mcmc_sampling.png` - Sampling methods
- `graph_spectral.png` - Graph spectral analysis
- `attention_weights.png` - Attention weight heatmaps

## Notes

- All examples include extensive comments explaining mathematical concepts
- Print statements show intermediate results for learning
- Visualizations are automatically saved to the current directory
- Each file has a `if __name__ == "__main__":` block for modularity
