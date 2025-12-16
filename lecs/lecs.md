# Computational Intelligence - Complete Lecture Notes

## Table of Contents
1. [Classical Optimization](#classical-optimization)
2. [Gradient Descent and Newton's Methods](#gradient-descent-and-newtons-methods)
3. [Regression](#regression)
4. [Bipolar Perceptron Learning & Optimization](#bipolar-perceptron-learning--optimization)
5. [Neural Architectures for Multiclass Models](#neural-architectures-for-multiclass-models)
6. [Backpropagation](#backpropagation)

---

## Classical Optimization

### Introduction to Optimization

**Definition**: Optimization is the act of obtaining the best result under given circumstances. It can be defined as the process of finding conditions that give the maximum or minimum of a function.

**Operations Research (OR)**: An interdisciplinary branch of mathematics that uses:
- Mathematical modeling
- Statistics
- Algorithms

to arrive at optimal or good decisions in complex problems concerned with optimizing maxima (profit, assembly line speed, crop yield, bandwidth) or minima (cost, loss, risk).

### Historical Development

Key contributors:
- **Isaac Newton (1642-1727)**: Differential calculus methods of optimization
- **Gottfried Leibnitz (1646-1716)**: Differential calculus methods
- **Leonhard Euler (1707-1783)**: Calculus of variations, minimization of functionals
- **Joseph-Louis Lagrange (1736-1813)**: Calculus of variations, method of optimization for constrained problems
- **Augustin-Louis Cauchy (1789-1857)**: Steepest descent method for unconstrained optimization
- **George Bernard Dantzig (1914-2005)**: Linear programming and Simplex method (1947)
- **Richard Bellman (1920-1984)**: Dynamic programming
- **Harold William Kuhn (1925-)**: Necessary and sufficient conditions for optimal solutions
- **Albert William Tucker (1905-1995)**: Nonlinear programming, game theory
- **John Von Neumann (1903-1957)**: Game theory

### Mathematical Optimization Problem

**General Form**:
```
minimize f₀(x)
subject to gᵢ(x) ≤ bᵢ, i = 1,...,m
```

Where:
- f₀: ℝⁿ → ℝ is the objective function
- x = (x₁,...,xₙ) are design variables (unknowns, must be linearly independent)
- gᵢ: ℝⁿ → ℝ are inequality constraints

**Note**: If a point x* corresponds to the minimum of f(x), it also corresponds to the maximum of -f(x). Thus, optimization can be taken to mean minimization.

### Local vs Global Extrema

- **Local Minimum**: f(x*) ≤ f(x* + h) for all sufficiently small h
- **Global Minimum**: f(x*) ≤ f(x) for all x in the domain
- **Local Maximum**: f(x*) ≥ f(x* + h) for all sufficiently small h
- **Global Maximum**: f(x*) ≥ f(x) for all x in the domain

### Linear vs Nonlinear Programming

- **Linear Programming**: Both objective function and constraints are linear
- **Nonlinear Programming**: At least one function (objective or constraint) is nonlinear

---

## Review of Mathematics

### Positive Definiteness

**Test 1 - Eigenvalues**: 
A matrix A is:
- **Positive definite** if all eigenvalues are positive
- **Negative definite** if all eigenvalues are negative

The eigenvalues satisfy: |A - λI| = 0

**Test 2 - Determinants**:
For matrix A of order n, evaluate:
- A₁ = |a₁₁|
- A₂ = |a₁₁ a₁₂; a₂₁ a₂₂|
- A₃ = |a₁₁ a₁₂ a₁₃; a₂₁ a₂₂ a₂₃; a₃₁ a₃₂ a₃₃|
- ... Aₙ

Matrix A is:
- **Positive definite** if all A₁, A₂, ..., Aₙ are positive
- **Negative definite** if sign of Aⱼ is (-1)ʲ for j = 1,2,...,n
- **Positive semidefinite** if some Aⱼ are positive and remaining are zero

**Semidefinite Matrices**:
- **Positive-semidefinite**: all eigenvalues ≥ 0
- **Negative-semidefinite**: all eigenvalues ≤ 0

### Matrix Concepts

- **Nonsingular matrix**: Determinant ≠ 0
- **Rank**: Order of the largest nonsingular square submatrix

---

## Single Variable Optimization

### Classical Optimization Techniques

**Characteristics**:
- Useful for continuous and differentiable functions
- Analytical methods using differential calculus
- Limited scope for non-continuous/non-differentiable functions

### Necessary Condition

If f(x) has a relative minimum at x = x* where a < x* < b, and if f'(x) exists and is finite at x*, then:

**f'(x*) = 0**

**Important Notes**:
- A point x* where f'(x*) = 0 is called a **stationary point**
- Not all stationary points are extrema (could be inflection points)
- Theorem doesn't apply if derivative doesn't exist at x*

### Sufficient Condition

Let f'(x*) = f''(x*) = ... = f⁽ⁿ⁻¹⁾(x*) = 0, but f⁽ⁿ⁾(x*) ≠ 0. Then f(x*) is:
- **Minimum** if f⁽ⁿ⁾(x*) > 0 and n is even
- **Maximum** if f⁽ⁿ⁾(x*) < 0 and n is even
- **Neither** if n is odd (inflection point)

### Example

Find extrema of f(x) = 12x⁵ - 45x⁴ + 40x³ + 5

**Solution**:
1. f'(x) = 60x⁴ - 180x³ + 120x² = 60x²(x-1)(x-2)
2. f'(x) = 0 at x = 0, x = 1, x = 2
3. f''(x) = 240x³ - 540x² + 240x = 60x(4x² - 9x + 4)

Analysis:
- At x = 1: f''(1) = -60 < 0 → **relative maximum**, f_max = 12
- At x = 2: f''(2) = 240 > 0 → **relative minimum**, f_min = -11
- At x = 0: f''(0) = 0, need third derivative
  - f'''(0) = 240 ≠ 0, n = 3 (odd) → **inflection point**

---

## Multivariable Optimization (Unconstrained)

### Necessary Condition

If f(X) has an extreme point at X = X* and first partial derivatives exist, then:

**∂f/∂x₁ = ∂f/∂x₂ = ... = ∂f/∂xₙ = 0** at X = X*

### Sufficient Condition

The **Hessian matrix** H of second partial derivatives evaluated at X* must be:
- **Positive definite** for relative minimum
- **Negative definite** for relative maximum

### Saddle Point

When the Hessian is neither positive nor negative definite at a stationary point (x*, y*), the point is a **saddle point**.

**Characteristic**: Corresponds to minimum with respect to one variable and maximum with respect to another.

**Example**: f(x,y) = x² - y²
- ∂f/∂x = 2x, ∂f/∂y = -2y
- Both zero at (0,0)
- Hessian: [2 0; 0 -2] → indefinite → saddle point

---

## Constrained Optimization with Equality Constraints

### Problem Statement

```
Minimize f(X)
subject to gⱼ(X) = 0, j = 1,2,...,m
```

Where m ≤ n (otherwise overdefined)

### Solution Methods

#### 1. Direct Substitution

**Procedure**:
1. Solve m equality constraints
2. Express m variables in terms of remaining (n-m) variables
3. Substitute into objective function
4. Optimize unconstrained function of (n-m) variables

**Limitations**:
- Simple in theory
- Impractical for nonlinear constraints
- Suitable only for simple problems

#### 2. Lagrange Multipliers Method

**Lagrange Function**:
```
L(x₁, x₂,..., xₙ, λ₁, λ₂,..., λₘ) = f(X) + Σⱼ λⱼgⱼ(X)
```

**Necessary Conditions**:
```
∂L/∂xᵢ = ∂f/∂xᵢ + Σⱼ λⱼ(∂gⱼ/∂xᵢ) = 0, i = 1,...,n
∂L/∂λⱼ = gⱼ(X) = 0, j = 1,...,m
```

This gives n+m equations for n+m unknowns (xᵢ and λⱼ).

**Interpretation of λ (Lagrange Multiplier)**:
- Provides sensitivity information
- Shows how optimal value changes with constraint

### Example: Box in Sphere

Find dimensions of largest box inscribed in unit sphere.

**Setup**:
- Variables: x₁, x₂, x₃ (half-dimensions)
- Objective: maximize f = 8x₁x₂x₃
- Constraint: x₁² + x₂² + x₃² = 1

**Solution**:
Using direct substitution:
- x₃ = √(1 - x₁² - x₂²)
- f(x₁,x₂) = 8x₁x₂√(1 - x₁² - x₂²)

Setting derivatives to zero:
- Result: x₁* = x₂* = x₃* = 1/√3
- Maximum volume: f_max = 8/(3√3)

### Example: Cylindrical Tin

Maximize volume of cylinder with surface area A₀ = 24.

**Setup**:
- Maximize: f = πx₁²x₂
- Constraint: 2πx₁² + 2πx₁x₂ = 24

**Lagrange Function**:
L = πx₁²x₂ + λ(24 - 2πx₁² - 2πx₁x₂)

**Necessary Conditions**:
- ∂L/∂x₁ = 2πx₁x₂ - 4πλx₁ - 2πλx₂ = 0
- ∂L/∂x₂ = πx₁² - 2πλx₁ = 0
- ∂L/∂λ = 24 - 2πx₁² - 2πx₁x₂ = 0

**Solution**:
- x₁* = 2, x₂* = 4, λ* = -1
- Maximum volume: f* = 16π

### Sufficient Condition

For constrained minimum at X*, the quadratic form:
```
Q = Σᵢ Σⱼ (∂²L/∂xᵢ∂xⱼ) dxᵢdxⱼ
```
evaluated at X* must be positive definite for all admissible variations dX (satisfying constraints).

For maximum, Q must be negative definite.

---

## Gradient Descent and Newton's Methods

### Gradient Descent Concept

**Principle**: Move in the direction opposite to the gradient (steepest descent).

**Gradient Vector**:
```
∇f(x₁,...,xₙ) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**Update Rule**:
```
Xᵢ₊₁ = Xᵢ - α∇f(Xᵢ)
```

Where α is the step size (learning rate).

### Gradient Descent Algorithm

**Steps**:
1. Initialize X₀ and set i = 0
2. Compute gradient ∇f(Xᵢ)
3. If ||∇f(Xᵢ)|| < ε, stop (converged)
4. Update: Xᵢ₊₁ = Xᵢ - α∇f(Xᵢ)
5. Set i = i + 1, go to step 2

### Steepest Descent with Line Search

**Optimal Step Size**: Instead of fixed α, find optimal step size by minimizing:
```
φ(α) = f(Xᵢ - α∇f(Xᵢ))
```

This is a 1D optimization problem along the search direction.

### Newton-Raphson Method

**Motivation**: Automatic step size selection using second-order information.

**Hessian Matrix**:
```
H = [∂²f/∂xᵢ∂xⱼ]
```

**Update Rule**:
```
Xᵢ₊₁ = Xᵢ - H⁻¹(Xᵢ)∇f(Xᵢ)
```

**Advantages**:
- Faster convergence (quadratic near optimum)
- Automatic step size

**Disadvantages**:
- Requires computing and inverting Hessian (expensive)
- May not converge if Hessian not positive definite

### Example: Newton-Raphson (1D)

f(x) = x⁴ - x³ + x² - x + 1

**Derivatives**:
- f'(x) = 4x³ - 3x² + 2x - 1
- f''(x) = 12x² - 6x + 2

**Update**: x_{i+1} = x_i - f'(x_i)/f''(x_i)

---

## Regression

### Machine Learning Pipeline

**Dataset Split**:
- **Training set**: Train classifier, measure training error
- **Validation set**: Tune hyperparameters, measure validation error
- **Test set**: Final evaluation with secret labels

**Cross-validation**: Random train/validate splits to ensure robustness.

### Features

Common feature representations:
- Raw pixels
- Histograms
- Templates
- SIFT descriptors
- GIST
- ORB
- HOG

### What is Regression?

**Goal**: Predict continuous values from noisy data.
- Predicting house prices
- Estimating human pose
- Temperature forecasting

**Contrast with Classification**: Classification predicts discrete categories/labels.

### Least Squares Regression

**Problem**: Given dataset D = {(X₁, y₁), (X₂, y₂),...,(Xₙ, yₙ)} where y ∈ ℝ

**Objective**: Minimize
```
J(W) = [Σᵢ (yᵢ - W·Xᵢ)²] / (2N)
```

### Gradient Descent Solution

**Gradient**:
```
∂J/∂W = -[Σᵢ (yᵢ - W·Xᵢ)Xᵢ] / N
```

**Update**:
```
W_{t+1} = W_t - η[∂J/∂W]_t
```

Where η is the learning rate.

### Closed-Form Solution

**Matrix Formulation**:
```
DW = Y
```

Where:
- D = [X₁ᵀ; X₂ᵀ; ...; Xₙᵀ] is N×(d+1) design matrix
- W = [w_d, w_{d-1}, ..., w₁, w₀]ᵀ
- Y = [y₁, y₂, ..., yₙ]ᵀ

**Pseudo-Inverse Solution**:
```
W = (DᵀD)⁻¹DᵀY
```

This gives the optimal solution in one step.

### Polynomial Curve Fitting

**Model**: y(x,W) = w₀ + w₁x + w₂x² + ... + w_Mx^M

**Objective**: Minimize sum of squared errors:
```
J(W) = Σᵢ [yᵢ - y(xᵢ,W)]² / (2N)
```

**Problem**: Choice of polynomial order M affects:
- Underfitting (M too small)
- Overfitting (M too large)

---

## Overfitting and Regularization

### Definition of Overfitting

**Overfitting occurs when**:
- Model describes noise rather than signal
- Model captures idiosyncrasies rather than generalities
- Too many parameters relative to training data
- Example: order-N polynomial can intersect any N+1 points

### Detection of Overfitting

**Stability Test**:
- Good model: stable under different data samples
- Overfit model: inconsistent performance

**Performance Test**:
- Good model: low test error
- Bad model: high test error

### Bias-Variance Trade-off

**Error components**:
1. **Bias**: Model is oversimplified (underfitting)
2. **Variance**: Limited data leads to unstable estimates (overfitting)
3. **Inherent**: Intrinsic data difficulty (irreducible)

**Trade-off**: Reducing bias often increases variance, and vice versa.

### Regularization

**Modified Objective**:
```
J(W) = Σᵢ [yᵢ - y(xᵢ,W)]² / (2N) + (λ/2)||W||²
```

The term (λ/2)||W||² penalizes large weights.

**Solution with Regularization**:
```
W = (DᵀD + λI)⁻¹DᵀY
```

**Effect**:
- λ = 0: No regularization (may overfit)
- λ large: Strong regularization (may underfit)
- Optimal λ: Balance between bias and variance

### Solutions to Underfitting

**Underfitting** (high bias, poor on both train and validation):
1. More features
2. More powerful model
3. Reduce regularization

### Solutions to Overfitting

**Overfitting** (low bias, high variance, poor on validation):
1. More training data
2. Less powerful model
3. Increase regularization

**Rule**: First ensure you can overfit, then prevent overfitting.

---

## Logistic Regression

### Binary Classification

**Problem**: Given D = {(X₁, y₁),...,(Xₙ, yₙ)} where y ∈ {-1, +1}

**Sigmoid Activation**:
```
ŷ = 1 / (1 + e^{-W·X})
```

**Decision Rule**:
- If ŷ > 0.5: Predict +1
- Otherwise: Predict -1

### Loss Function

**Individual Loss**:
```
Lᵢ = -log(|yᵢ/2 - 1/2 + ŷᵢ|)
```

**Alternative Form**:
```
Lᵢ = { -log(ŷᵢ)      if yᵢ = +1
     { -log(1 - ŷᵢ)  if yᵢ = -1
```

### Maximum Likelihood

**Likelihood**:
```
L = Πᵢ |yᵢ/2 - 1/2 + ŷᵢ|
```

**Negative Log-Likelihood** (to minimize):
```
L = -Σᵢ log(|yᵢ/2 - 1/2 + ŷᵢ|)
```

### Gradient Descent for Logistic Regression

**Gradient**:
```
∂Lᵢ/∂W = -yᵢXᵢ / (1 + exp(yᵢW·Xᵢ))
```

**Update**:
```
W_{t+1} = W_t - η Σᵢ ∂Lᵢ/∂W
```

---

## Bipolar Perceptron Learning & Optimization

### Biological Motivation

Neural networks simulate biological mechanisms:
- Neurons as computational units
- Weighted connections (synapses)
- Activation functions (firing patterns)

### Deep Learning Context

**Human vs Computer Strengths**:
- Computers: Numerical computation, exact calculations
- Humans: Pattern recognition, visual understanding
- Deep learning bridges the gap for some tasks

### Perceptron Architecture

**Components**:
1. Input vector: X = [x₁, x₂,..., xₙ, 1]ᵀ
2. Weight vector: W = [w₁, w₂,..., wₙ, w₀]ᵀ
3. Activation: sign(W·X)

**Geometric Interpretation**:
- 2D: Line separating classes (w₁x₁ + w₂x₂ + w₀ = 0)
- 3D: Plane separating classes
- Higher dimensions: Hyperplane

### AND Function Example

**Truth Table**:
```
x₁   x₂   y
-1   -1   -1
-1   +1   -1
+1   -1   -1
+1   +1   +1
```

**Constraints**:
- (-1)w₁ + (-1)w₂ + w₀ < 0
- (-1)w₁ + (+1)w₂ + w₀ < 0
- (+1)w₁ + (-1)w₂ + w₀ < 0
- (+1)w₁ + (+1)w₂ + w₀ ≥ 0

**Solution**: w₀ = -1, w₁ = 2, w₂ = 2

### Perceptron Training

**Objective**: Minimize
```
E(W) = Σᵢ [yᵢ - sign(W·Xᵢ)]²
```

**Problem**: sign function has zero derivative (except at origin).

**Solution**: Use smooth approximation (e.g., tanh, sigmoid).

### Training Algorithms

#### Batch Training

```
Initialize W = 0
Repeat:
  delta = 0
  For each training example (Xₘ, yₘ):
    If yₘ(W·Xₘ) ≤ 0:
      delta = delta - yₘXₘ
  delta = delta / N
  W = W - delta
Until ||delta|| < ε
```

#### Online Training

```
Initialize W = 0
Repeat:
  For each training example (Xₘ, yₘ):
    If yₘ(W·Xₘ) ≤ 0:
      delta = -yₘXₘ
      W = W - delta/N
Until converged
```

### Relation to SVM

**Hinge Loss** (soft-margin):
```
Lᵢ = max(0, 1 - yᵢ(W·Xᵢ))
```

This allows for more robust classification with margin.

### Activation Functions

#### Common Choices

1. **Sign**: sign(u) = {+1 if u ≥ 0, -1 otherwise}
2. **Sigmoid**: σ(u) = 1/(1 + e^{-u})
3. **Tanh**: tanh(u) = (e^u - e^{-u})/(e^u + e^{-u})
4. **ReLU**: ReLU(u) = max(0, u)

#### Derivatives

- **Sigmoid**: dσ/du = σ(1 - σ)
- **Tanh**: d tanh/du = 1 - tanh²(u)

### Multilayer Networks

**Limitations of Single Layer**:
- Can only learn linearly separable functions
- XOR problem requires multiple layers

**Solution**: Deep neural networks with hidden layers.

### Softmax for Multi-class

**Softmax Function**:
```
yᵣ = exp(zᵣ) / Σⱼ exp(zⱼ)
```

**Properties**:
- 0 < yᵣ < 1 for all r
- Σᵣ yᵣ = 1
- Converts scores to probability distribution

**Prediction**: Class with maximum yᵣ

---

## Neural Architectures for Multiclass Models

### Problem Definition

**Given**: Dataset D = {(X₁, y₁),...,(Xₙ, yₙ)} where y ∈ {1,2,...,k}

**Goal**: Estimate weight vectors W₁, W₂,..., W_k to minimize loss function.

### Score and Classification

**Score for class r**: sᵣ = Wᵣ·Xᵢ

**Classification Decision**: Predict class with maximum score.

**Correct Classification**: Class yᵢ should have maximum score.

### Multiclass Perceptron Loss

**Individual Losses**:
```
dᵣ = max(Wᵣ·Xᵢ - W_{yᵢ}·Xᵢ, 0) for r ≠ yᵢ
```

**Total Loss**:
```
Lᵢ = max_r(dᵣ)
```

**Learning Rule**:
```
∂Lᵢ/∂Wᵣ = { -Xᵢ     if r = yᵢ
           { +Xᵢ     if r is most incorrect prediction
           { 0       otherwise
```

**Disadvantage**: Only updates most incorrect class.

### Weston-Watkins SVM

**Improvement**: Updates all incorrectly predicted classes.

**Loss Function**:
```
Lᵢ = Σᵣ≠yᵢ max(1 + Wᵣ·Xᵢ - W_{yᵢ}·Xᵢ, 0)
```

**Learning Rule**:
```
∂Lᵢ/∂Wᵣ = { -Xᵢ Σⱼ≠ᵣ δⱼᵢ    if r = yᵢ
          { +Xᵢ δᵣᵢ         if r ≠ yᵢ
```

Where δⱼᵢ = 1 if margin violated, 0 otherwise.

**With Regularization**:
```
L = Σᵢ Σᵣ≠yᵢ dᵣ + (λ/2) Σᵣ ||Wᵣ||²
```

### Hierarchical MCSVM

**Application**: Action recognition with hierarchical categories.

**Structure**:
- Top level: Coarse categories (e.g., sports, daily activities)
- Lower levels: Fine-grained actions

**Advantages**:
- Better organization of large number of classes
- Exploits hierarchical relationships

### Multinomial Logistic Regression (Softmax)

**Score Computation**:
```
vᵣ = Wᵣ·Xᵢ
yᵣ = exp(vᵣ) / Σⱼ exp(vⱼ)
```

**Loss Function**:
```
Lᵢ = -log(y_{yᵢ}) = -v_{yᵢ} + log(Σⱼ exp(vⱼ))
```

**Gradient**:
```
∂Lᵢ/∂Wᵣ = (∂Lᵢ/∂vᵣ)(∂vᵣ/∂Wᵣ)
```

Where ∂vᵣ/∂Wᵣ = Xᵢ

### Weight Initialization

**Guidelines**:
- **Too small (near zero)**: Zero gradients, no learning
- **Too large**: Numerical instability, saturation in sigmoid/tanh
- **Recommended**: Small random values around zero
  - Example: Uniform[-0.1, 0.1] or Normal(0, 0.01)

### Data Preprocessing

**Normalization**:
- **Zero mean**: X' = X - mean(X)
- **Unit variance**: X'' = X' / std(X')

**Batch Normalization**:
- Normalize activations within each mini-batch
- Helps with training stability
- Reduces internal covariate shift

---

## Backpropagation

### Motivation

**Problem**: Compute gradients efficiently for deep networks.

**Solution**: Backpropagation algorithm - efficient computation using chain rule.

### Cost Function

**For single example**:
```
E(θ) = Cost between network output and target
```

**For all training data**:
```
E(θ) = Σᵣ Eᵣ(θ)
```

**Goal**: Find θ* that minimizes total cost.

### Simplified Problem

**Loss Function**:
```
E(θ) = (1/2)||Y(θ) - Y||²
```

**Gradient**:
```
∂E/∂θ = (Y(θ) - Y)ᵀ ∂Y(θ)/∂θ
```

### Vector Calculus Review

**Key Rules**:
- ∂(Ax)/∂x = Aᵀ
- ∂(xᵀA)/∂x = A
- ∂(xᵀAx)/∂x = (A + Aᵀ)x
- Chain rule: ∂f/∂x = (∂f/∂y)(∂y/∂x)

### Forward Pass

**Layer computations**:
```
f₀ = X (input)
Z₁ = A₀f₀ + b₀
f₁ = σ(Z₁)
Z₂ = A₁f₁ + b₁
f₂ = σ(Z₂) (output)
E = (1/2)(f₂ - Y)ᵀ(f₂ - Y)
```

### Backward Pass

**Gradient w.r.t. bias b₁**:
```
∂E/∂b₁ = (f₂ - Y)ᵀ (∂f₂/∂Z₂)(∂Z₂/∂b₁)
       = (f₂ - Y)ᵀ diag(σ'(Z₂))
```

Where ∂Z₂