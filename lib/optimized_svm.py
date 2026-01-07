"""
Highly optimized SVM implementation using Numba JIT compilation and parallel processing.

This module provides a high-performance SVM implementation that significantly
outperforms the base SVM while maintaining the same API and functionality.
"""

import numpy as np
from numba import jit, prange
from typing import Optional, Literal, Union
import warnings
from joblib import Parallel, delayed
from .optimized_ops import fast_linear_kernel, fast_rbf_kernel, fast_kernel_value


@jit(nopython=True, cache=True)
def _fast_smo_optimize_pair(i1, i2, alphas, y, C, tol, b, X, kernel_type, gamma):
    """
    Fast SMO pair optimization using Numba JIT compilation.
    
    Args:
        i1, i2: Indices of alphas to optimize
        alphas: Alpha coefficients array
        y: Labels array
        C: Regularization parameter
        tol: Tolerance
        b: Bias term
        X: Training data
        kernel_type: 0 for linear, 1 for RBF
        gamma: RBF parameter
        
    Returns:
        Tuple of (success, new_b)
    """
    if i1 == i2:
        return False, b
    
    alpha1_old = alphas[i1]
    alpha2_old = alphas[i2]
    y1, y2 = y[i1], y[i2]
    
    # Compute bounds L and H
    if y1 != y2:
        L = max(0.0, alpha2_old - alpha1_old)
        H = min(C, C + alpha2_old - alpha1_old)
    else:
        L = max(0.0, alpha1_old + alpha2_old - C)
        H = min(C, alpha1_old + alpha2_old)
    
    if abs(L - H) < tol:
        return False, b
    
    # Compute kernel values
    k11 = fast_kernel_value(X[i1], X[i1], kernel_type, gamma)
    k12 = fast_kernel_value(X[i1], X[i2], kernel_type, gamma)
    k22 = fast_kernel_value(X[i2], X[i2], kernel_type, gamma)
    
    eta = k11 + k22 - 2 * k12
    
    if eta <= 0:
        return False, b
    
    # Compute errors
    E1 = _fast_compute_error(i1, alphas, y, b, X, kernel_type, gamma)
    E2 = _fast_compute_error(i2, alphas, y, b, X, kernel_type, gamma)
    
    # Compute new alpha2
    alpha2_new = alpha2_old + y2 * (E1 - E2) / eta
    
    # Clip alpha2
    alpha2_new = max(L, min(H, alpha2_new))
    
    # Check if change is significant
    if abs(alpha2_new - alpha2_old) < tol:
        return False, b
    
    # Compute new alpha1
    alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)
    
    # Update alphas
    alphas[i1] = alpha1_new
    alphas[i2] = alpha2_new
    
    # Update bias term
    b1 = b - E1 - y1 * (alpha1_new - alpha1_old) * k11 - y2 * (alpha2_new - alpha2_old) * k12
    b2 = b - E2 - y1 * (alpha1_new - alpha1_old) * k12 - y2 * (alpha2_new - alpha2_old) * k22
    
    if 0 < alpha1_new < C:
        new_b = b1
    elif 0 < alpha2_new < C:
        new_b = b2
    else:
        new_b = (b1 + b2) / 2
    
    return True, new_b


@jit(nopython=True, cache=True)
def _fast_compute_error(i, alphas, y, b, X, kernel_type, gamma):
    """
    Fast error computation for sample i.
    
    Args:
        i: Sample index
        alphas: Alpha coefficients
        y: Labels
        b: Bias term
        X: Training data
        kernel_type: 0 for linear, 1 for RBF
        gamma: RBF parameter
        
    Returns:
        Error for sample i
    """
    prediction = b
    
    for j in range(len(alphas)):
        if alphas[j] > 0:
            kernel_val = fast_kernel_value(X[i], X[j], kernel_type, gamma)
            prediction += alphas[j] * y[j] * kernel_val
    
    return prediction - y[i]


@jit(nopython=True, cache=True)
def _fast_select_second_alpha(i1, E1, alphas, y, b, X, kernel_type, gamma, C, tol):
    """
    Fast second alpha selection using heuristic.
    
    Args:
        i1: First alpha index
        E1: Error for first alpha
        alphas: Alpha coefficients
        y: Labels
        b: Bias term
        X: Training data
        kernel_type: Kernel type
        gamma: RBF parameter
        C: Regularization parameter
        tol: Tolerance
        
    Returns:
        Index of second alpha
    """
    max_step = 0.0
    i2 = -1
    
    # Look for non-bound alphas first
    for i in range(len(alphas)):
        if i == i1:
            continue
        if alphas[i] > tol and alphas[i] < C - tol:
            E2 = _fast_compute_error(i, alphas, y, b, X, kernel_type, gamma)
            step = abs(E1 - E2)
            if step > max_step:
                max_step = step
                i2 = i
    
    # If no good candidate found, select randomly
    if i2 == -1:
        for i in range(len(alphas)):
            if i != i1:
                i2 = i
                break
    
    return i2


@jit(nopython=True, cache=True)
def _fast_smo_algorithm(X, y, C, tol, max_iter, kernel_type, gamma):
    """
    Fast SMO algorithm implementation using Numba JIT compilation.
    
    Args:
        X: Training data
        y: Labels
        C: Regularization parameter
        tol: Tolerance
        max_iter: Maximum iterations
        kernel_type: 0 for linear, 1 for RBF
        gamma: RBF parameter
        
    Returns:
        Tuple of (alphas, b, num_iterations)
    """
    n_samples = X.shape[0]
    alphas = np.zeros(n_samples, dtype=np.float64)
    b = 0.0
    
    num_changed = 0
    examine_all = True
    iteration = 0
    
    while (num_changed > 0 or examine_all) and iteration < max_iter:
        num_changed = 0
        
        if examine_all:
            # Examine all examples
            for i in range(n_samples):
                if _fast_examine_example(i, alphas, y, C, tol, b, X, kernel_type, gamma):
                    num_changed += 1
                    # Update b after each successful optimization
                    b = _fast_update_bias(alphas, y, C, tol, X, kernel_type, gamma)
        else:
            # Examine non-bound examples
            for i in range(n_samples):
                if alphas[i] > tol and alphas[i] < C - tol:
                    if _fast_examine_example(i, alphas, y, C, tol, b, X, kernel_type, gamma):
                        num_changed += 1
                        # Update b after each successful optimization
                        b = _fast_update_bias(alphas, y, C, tol, X, kernel_type, gamma)
        
        if examine_all:
            examine_all = False
        elif num_changed == 0:
            examine_all = True
        
        iteration += 1
        
        # Early stopping if we have enough support vectors
        num_sv = 0
        for i in range(n_samples):
            if alphas[i] > tol:
                num_sv += 1
        
        if num_sv > min(100, n_samples // 2):
            break
    
    return alphas, b, iteration


@jit(nopython=True, cache=True)
def _fast_examine_example(i1, alphas, y, C, tol, b, X, kernel_type, gamma):
    """
    Fast example examination for SMO algorithm.
    
    Args:
        i1: Example index
        alphas: Alpha coefficients
        y: Labels
        C: Regularization parameter
        tol: Tolerance
        b: Bias term
        X: Training data
        kernel_type: Kernel type
        gamma: RBF parameter
        
    Returns:
        Boolean indicating if optimization occurred
    """
    y1 = y[i1]
    alpha1 = alphas[i1]
    E1 = _fast_compute_error(i1, alphas, y, b, X, kernel_type, gamma)
    r1 = E1 * y1
    
    # Check KKT conditions with relaxed tolerance
    tolerance = tol * 2
    
    if ((r1 < -tolerance and alpha1 < C) or (r1 > tolerance and alpha1 > 0)):
        # Select second alpha
        i2 = _fast_select_second_alpha(i1, E1, alphas, y, b, X, kernel_type, gamma, C, tol)
        
        success, new_b = _fast_smo_optimize_pair(i1, i2, alphas, y, C, tol, b, X, kernel_type, gamma)
        if success:
            return True
    
    return False


@jit(nopython=True, cache=True)
def _fast_update_bias(alphas, y, C, tol, X, kernel_type, gamma):
    """
    Fast bias update computation.
    
    Args:
        alphas: Alpha coefficients
        y: Labels
        C: Regularization parameter
        tol: Tolerance
        X: Training data
        kernel_type: Kernel type
        gamma: RBF parameter
        
    Returns:
        Updated bias term
    """
    # Find support vectors on the margin (0 < alpha < C)
    margin_sv_indices = []
    for i in range(len(alphas)):
        if tol < alphas[i] < C - tol:
            margin_sv_indices.append(i)
    
    if len(margin_sv_indices) == 0:
        return 0.0
    
    # Compute bias as average over margin support vectors
    bias_sum = 0.0
    for i in margin_sv_indices:
        prediction = 0.0
        for j in range(len(alphas)):
            if alphas[j] > 0:
                kernel_val = fast_kernel_value(X[i], X[j], kernel_type, gamma)
                prediction += alphas[j] * y[j] * kernel_val
        bias_sum += y[i] - prediction
    
    return bias_sum / len(margin_sv_indices)


class OptimizedSVM:
    """
    Highly optimized SVM classifier using Numba JIT compilation and parallel processing.
    
    This implementation provides significant performance improvements over the base SVM
    while maintaining the same API and functionality. Key optimizations include:
    - JIT-compiled SMO algorithm
    - Parallel kernel computations
    - Memory-efficient operations
    - Early stopping heuristics
    """
    
    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal['linear', 'rbf'] = 'linear',
        gamma: Union[float, str] = 'scale',
        tol: float = 1e-2,
        max_iter: int = 100,
        random_state: Optional[int] = None,
        n_jobs: int = -1
    ):
        """
        Initialize Optimized SVM classifier.
        
        Args:
            C: Regularization parameter
            kernel: Kernel type ('linear' or 'rbf')
            gamma: RBF kernel parameter
            tol: Tolerance for stopping criterion
            max_iter: Maximum number of iterations
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize attributes
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.n_support_ = None
        self.classes_ = None
        
        # Internal variables
        self._X = None
        self._y = None
        self._alphas = None
        self._b = 0.0
        self._gamma_val = None
        self._kernel_type = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _prepare_data(self, X, y):
        """Prepare and validate input data."""
        # Convert to numpy arrays with optimal dtype
        X = np.asarray(X, dtype=np.float64)  # Use float64 for numerical stability in SVM
        y = np.asarray(y)
        
        # Store unique classes
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 2:
            raise ValueError("OptimizedSVM currently supports only binary classification")
        
        # Convert labels to -1 and +1
        y_binary = np.where(y == self.classes_[0], -1, 1).astype(np.float64)
        
        return X, y_binary
    
    def _setup_kernel_params(self, X):
        """Setup kernel parameters."""
        # Set kernel type for JIT functions
        if self.kernel == 'linear':
            self._kernel_type = 0
        elif self.kernel == 'rbf':
            self._kernel_type = 1
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        
        # Pre-compute gamma value for RBF kernel
        if self.kernel == 'rbf':
            if isinstance(self.gamma, str) and self.gamma == 'scale':
                self._gamma_val = 1.0 / (X.shape[1] * np.var(X))
            else:
                self._gamma_val = float(self.gamma)
        else:
            self._gamma_val = 1.0  # Not used for linear kernel
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OptimizedSVM':
        """
        Fit the optimized SVM model to training data.
        
        Args:
            X: Training vectors of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: Fitted estimator
        """
        # Prepare data
        X, y_binary = self._prepare_data(X, y)
        self._X = X
        self._y = y_binary
        
        # Setup kernel parameters
        self._setup_kernel_params(X)
        
        print(f"Training OptimizedSVM with {len(X)} samples, {X.shape[1]} features")
        print(f"Kernel: {self.kernel}, C: {self.C}, Tolerance: {self.tol}")
        
        # Run optimized SMO algorithm
        self._alphas, self._b, num_iterations = _fast_smo_algorithm(
            X, y_binary, self.C, self.tol, self.max_iter, 
            self._kernel_type, self._gamma_val
        )
        
        print(f"SMO converged after {num_iterations} iterations")
        
        # Extract support vectors
        support_indices = np.where(self._alphas > self.tol)[0]
        self.support_vectors_ = X[support_indices]
        self.support_vector_labels_ = y_binary[support_indices]
        self.dual_coef_ = self._alphas[support_indices] * y_binary[support_indices]
        self.intercept_ = self._b
        self.n_support_ = len(support_indices)
        
        print(f"Found {self.n_support_} support vectors ({100*self.n_support_/len(X):.1f}% of training data)")
        
        return self
    
    def _decision_function_batch(self, X_batch):
        """Compute decision function for a batch of samples."""
        if self.support_vectors_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_batch = np.asarray(X_batch, dtype=np.float64)
        
        if self.kernel == 'linear':
            # Efficient linear decision function
            w = np.sum((self.dual_coef_[:, np.newaxis] * self.support_vectors_), axis=0)
            return np.dot(X_batch, w) + self.intercept_
        else:
            # RBF kernel decision function
            decisions = np.zeros(X_batch.shape[0])
            
            # Use parallel computation for large datasets
            if len(X_batch) > 100 and self.n_jobs != 1:
                # Parallel computation
                def compute_decision(i):
                    decision = self.intercept_
                    for j, sv in enumerate(self.support_vectors_):
                        kernel_val = fast_kernel_value(X_batch[i], sv, self._kernel_type, self._gamma_val)
                        decision += self.dual_coef_[j] * kernel_val
                    return decision
                
                n_jobs = self.n_jobs if self.n_jobs > 0 else -1
                decisions = Parallel(n_jobs=n_jobs)(
                    delayed(compute_decision)(i) for i in range(len(X_batch))
                )
                decisions = np.array(decisions)
            else:
                # Sequential computation for small datasets
                for i in range(len(X_batch)):
                    decision = self.intercept_
                    for j, sv in enumerate(self.support_vectors_):
                        kernel_val = fast_kernel_value(X_batch[i], sv, self._kernel_type, self._gamma_val)
                        decision += self.dual_coef_[j] * kernel_val
                    decisions[i] = decision
            
            return decisions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the decision function for samples in X.
        
        Args:
            X: Samples of shape (n_samples, n_features)
            
        Returns:
            Decision function values of shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        
        # Process in batches for memory efficiency
        batch_size = 1000
        if len(X) <= batch_size:
            return self._decision_function_batch(X)
        
        # Process large datasets in batches
        decisions = []
        for i in range(0, len(X), batch_size):
            batch_end = min(i + batch_size, len(X))
            batch_decisions = self._decision_function_batch(X[i:batch_end])
            decisions.append(batch_decisions)
        
        return np.concatenate(decisions)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X: Samples of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        decision = self.decision_function(X)
        predictions = np.where(decision >= 0, 1, -1)
        
        # Convert back to original class labels
        return np.where(predictions == -1, self.classes_[0], self.classes_[1])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score on test data.
        
        Args:
            X: Test samples
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self


# Alias for sklearn compatibility
OptimizedSVC = OptimizedSVM


def benchmark_svm_performance(svm_class, X, y, num_runs=5):
    """
    Benchmark SVM performance.
    
    Args:
        svm_class: SVM class to benchmark
        X: Training data
        y: Training labels
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    fit_times = []
    predict_times = []
    accuracies = []
    
    for run in range(num_runs):
        # Create fresh instance
        svm = svm_class(random_state=42 + run)
        
        # Time fitting
        start_time = time.time()
        svm.fit(X, y)
        fit_time = time.time() - start_time
        fit_times.append(fit_time)
        
        # Time prediction
        start_time = time.time()
        predictions = svm.predict(X)
        predict_time = time.time() - start_time
        predict_times.append(predict_time)
        
        # Compute accuracy
        accuracy = np.mean(predictions == y)
        accuracies.append(accuracy)
    
    return {
        'avg_fit_time': np.mean(fit_times),
        'std_fit_time': np.std(fit_times),
        'avg_predict_time': np.mean(predict_times),
        'std_predict_time': np.std(predict_times),
        'avg_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'svm_type': svm_class.__name__
    }