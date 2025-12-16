# Custom SVM Implementation from Scratch

This project implements a Support Vector Machine (SVM) classifier from scratch using Python and NumPy, replacing the sklearn SVM implementation in the neural network project.

## Overview

Our implementation includes:

1. **Binary SVM** (`lib/svm.py`) - Core SVM implementation using Sequential Minimal Optimization (SMO)
2. **Multi-class SVM** (`lib/multiclass_svm.py`) - Extends binary SVM for multi-class classification using One-vs-Rest strategy
3. **Custom Metrics** (`lib/metrics.py`) - Evaluation metrics including accuracy, precision, recall, F1-score, and confusion matrix

## Key Features

### SVM Implementation (`lib/svm.py`)

- **Sequential Minimal Optimization (SMO)** algorithm for solving the quadratic programming problem
- **Multiple kernel support**: Linear and RBF (Radial Basis Function) kernels
- **Configurable parameters**: C (regularization), gamma (kernel coefficient), tolerance, max iterations
- **KKT conditions checking** for optimal solution verification
- **Support vector extraction** and decision function computation

### Multi-class Extension (`lib/multiclass_svm.py`)

- **One-vs-Rest (OvR) strategy** for multi-class classification
- **Automatic binary/multi-class detection**
- **Decision function aggregation** for final predictions
- **Compatible interface** with sklearn SVM

### Custom Metrics (`lib/metrics.py`)

- **Accuracy score** calculation
- **Confusion matrix** generation
- **Precision, recall, and F1-score** computation
- **Classification report** with detailed metrics
- **Support for both binary and multi-class** problems

## Algorithm Details

### Sequential Minimal Optimization (SMO)

The SMO algorithm solves the SVM dual optimization problem by:

1. **Selecting pairs of Lagrange multipliers** (alphas) to optimize
2. **Solving the two-variable quadratic subproblem** analytically
3. **Updating the bias term** based on KKT conditions
4. **Iterating until convergence** or maximum iterations reached

### Kernel Functions

#### Linear Kernel
```
K(x_i, x_j) = x_i · x_j
```

#### RBF Kernel
```
K(x_i, x_j) = exp(-γ ||x_i - x_j||²)
```

Where γ (gamma) can be:
- A fixed value
- 'scale': 1 / (n_features × X.var())

### Multi-class Strategy

For multi-class problems with K classes:

1. **Train K binary classifiers**, each distinguishing one class from all others
2. **For prediction**, compute decision scores from all classifiers
3. **Assign the class** with the highest decision score

## Usage Examples

### Basic Binary Classification

```python
from lib.svm import SVM
from lib.metrics import accuracy_score

# Create and train SVM
svm = SVM(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

### Multi-class Classification

```python
from lib.multiclass_svm import MultiClassSVM
from lib.metrics import classification_report

# Create and train multi-class SVM
svm = MultiClassSVM(kernel='rbf', C=10.0, gamma='scale')
svm.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = svm.predict(X_test)
report = classification_report(y_test, y_pred)
```

### Integration with Neural Network Project

The SVM implementation is integrated into the project's Section 4 notebook (`notebooks/section4_svm_demo.ipynb`):

```python
# Import our custom implementation
from lib.multiclass_svm import MultiClassSVM as SVC
from lib.metrics import accuracy_score, classification_report, confusion_matrix

# Use exactly like sklearn SVM
svm = SVC(kernel='rbf', C=100.0, gamma='scale', random_state=42)
svm.fit(X_train_latent, y_train)
y_pred = svm.predict(X_test_latent)
```

## Performance Considerations

### Optimizations Implemented

1. **Efficient kernel matrix computation** using vectorized operations
2. **Smart alpha pair selection** using heuristics from SMO literature
3. **Early stopping** based on KKT condition violations
4. **Memory-efficient** kernel matrix handling for large datasets

### Computational Complexity

- **Training time**: O(n²) to O(n³) depending on dataset characteristics
- **Memory usage**: O(n²) for kernel matrix storage
- **Prediction time**: O(n_support_vectors × n_features)

### Recommended Settings

For the neural network project's latent space classification:

```python
# Balanced performance and accuracy
svm = MultiClassSVM(
    kernel='rbf',
    C=10.0,
    gamma='scale',
    max_iter=500,
    tol=1e-3,
    random_state=42
)
```

## Testing

Run the test scripts to verify implementation:

```bash
# Test basic SVM functionality
python test_svm.py

# Test multi-class SVM
python test_multiclass_svm.py
```

## Comparison with sklearn

| Feature | Our Implementation | sklearn SVM |
|---------|-------------------|-------------|
| Algorithm | SMO from scratch | Optimized libsvm |
| Kernels | Linear, RBF | Linear, RBF, Poly, Sigmoid |
| Multi-class | One-vs-Rest | One-vs-Rest, One-vs-One |
| Performance | Good for learning | Production optimized |
| Dependencies | NumPy only | Cython, C libraries |

## Educational Value

This implementation demonstrates:

1. **Quadratic programming** solution using SMO
2. **Kernel methods** and their computational aspects
3. **Multi-class extension** strategies
4. **Optimization algorithms** in machine learning
5. **Software engineering** practices for ML libraries

## Files Structure

```
lib/
├── svm.py              # Core binary SVM implementation
├── multiclass_svm.py   # Multi-class SVM wrapper
└── metrics.py          # Evaluation metrics

notebooks/
└── section4_svm_demo.ipynb  # Updated notebook using custom SVM

test_svm.py             # Binary SVM tests
test_multiclass_svm.py  # Multi-class SVM tests
SVM_README.md          # This documentation
```

## Future Enhancements

Potential improvements for the implementation:

1. **Additional kernels** (polynomial, sigmoid)
2. **Sparse matrix support** for large datasets
3. **Parallel training** for multi-class problems
4. **Advanced optimization** techniques (chunking, caching)
5. **Probability estimates** for predictions

## References

1. Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines
2. Vapnik, V. (1995). The Nature of Statistical Learning Theory
3. Schölkopf, B. & Smola, A. (2002). Learning with Kernels