"""
Fast SVM implementation optimized for speed while maintaining educational value.

This implementation focuses on computational efficiency with simplified SMO algorithm,
better heuristics, and optimized data structures.
"""

import numpy as np
from typing import Optional, Literal, Union
import warnings


class FastSVM:
    """
    Fast SVM classifier with optimized SMO algorithm.
    
    This implementation prioritizes speed while maintaining the core SVM concepts.
    Key optimizations:
    - Simplified SMO with better heuristics
    - Early stopping conditions
    - Efficient kernel computation
    - Reduced iteration complexity
    
    Parameters:
    -----------
    C : float, default=1.0
        Regularization parameter.
    kernel : {'linear', 'rbf'}, default='linear'
        Kernel type (linear is much faster).
    gamma : float or 'scale', default='scale'
        Kernel coefficient for 'rbf'.
    tol : float, default=1e-2
        Tolerance for stopping criterion (relaxed for speed).
    max_iter : int, default=100
        Maximum number of iterations (reduced for speed).
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal['linear', 'rbf'] = 'linear',
        gamma: Union[float, str] = 'scale',
        tol: float = 1e-2,
        max_iter: int = 100,
        random_state: Optional[int] = None
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        
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
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_kernel_value(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel value between two samples."""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            diff = x1 - x2
            return np.exp(-self._gamma_val * np.dot(diff, diff))
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _compute_kernel_row(self, i: int) -> np.ndarray:
        """Compute the i-th row of the kernel matrix efficiently."""
        if self.kernel == 'linear':
            return np.dot(self._X, self._X[i])
        elif self.kernel == 'rbf':
            # Vectorized RBF computation
            diff = self._X - self._X[i]
            distances_sq = np.sum(diff**2, axis=1)
            return np.exp(-self._gamma_val * distances_sq)
    
    def _compute_prediction(self, i: int) -> float:
        """Compute prediction for sample i using current alphas."""
        prediction = self._b
        
        # Only compute for non-zero alphas (sparse computation)
        nonzero_indices = np.where(self._alphas > self.tol)[0]
        
        if len(nonzero_indices) == 0:
            return prediction
        
        if self.kernel == 'linear':
            # Efficient linear kernel computation
            w = np.sum((self._alphas[nonzero_indices] * self._y[nonzero_indices])[:, np.newaxis] * 
                      self._X[nonzero_indices], axis=0)
            prediction += np.dot(w, self._X[i])
        else:
            # RBF kernel computation
            for j in nonzero_indices:
                kernel_val = self._compute_kernel_value(self._X[i], self._X[j])
                prediction += self._alphas[j] * self._y[j] * kernel_val
        
        return prediction
    
    def _compute_error(self, i: int) -> float:
        """Compute the error for sample i."""
        return self._compute_prediction(i) - self._y[i]
    
    def _select_second_alpha_fast(self, i1: int, E1: float) -> int:
        """Fast heuristic for selecting second alpha."""
        # Simple random selection for speed
        candidates = list(range(len(self._alphas)))
        candidates.remove(i1)
        return np.random.choice(candidates)
    
    def _optimize_pair_fast(self, i1: int, i2: int) -> bool:
        """Fast pair optimization with simplified logic."""
        if i1 == i2:
            return False
        
        alpha1_old = self._alphas[i1]
        alpha2_old = self._alphas[i2]
        y1, y2 = self._y[i1], self._y[i2]
        
        # Compute bounds
        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha1_old + alpha2_old - self.C)
            H = min(self.C, alpha1_old + alpha2_old)
        
        if abs(L - H) < self.tol:
            return False
        
        # Compute kernel values
        k11 = self._compute_kernel_value(self._X[i1], self._X[i1])
        k12 = self._compute_kernel_value(self._X[i1], self._X[i2])
        k22 = self._compute_kernel_value(self._X[i2], self._X[i2])
        
        eta = k11 + k22 - 2 * k12
        
        if eta <= 0:
            return False
        
        # Compute errors
        E1 = self._compute_error(i1)
        E2 = self._compute_error(i2)
        
        # Compute new alpha2
        alpha2_new = alpha2_old + y2 * (E1 - E2) / eta
        
        # Clip alpha2
        alpha2_new = max(L, min(H, alpha2_new))
        
        # Check if change is significant
        if abs(alpha2_new - alpha2_old) < self.tol:
            return False
        
        # Compute new alpha1
        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)
        
        # Update alphas
        self._alphas[i1] = alpha1_new
        self._alphas[i2] = alpha2_new
        
        # Update bias (simplified)
        self._b = self._b - E1 - y1 * (alpha1_new - alpha1_old) * k11 - y2 * (alpha2_new - alpha2_old) * k12
        
        return True
    
    def _smo_algorithm_fast(self) -> None:
        """Fast SMO algorithm with aggressive early stopping."""
        num_changed = 0
        examine_all = True
        iteration = 0
        
        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            
            if examine_all:
                # Examine all examples
                indices = list(range(len(self._alphas)))
                np.random.shuffle(indices)  # Randomize order for better convergence
                
                for i in indices:
                    if self._examine_example_fast(i):
                        num_changed += 1
            else:
                # Examine non-bound examples
                non_bound_indices = np.where((self._alphas > self.tol) & (self._alphas < self.C - self.tol))[0]
                np.random.shuffle(non_bound_indices)
                
                for i in non_bound_indices:
                    if self._examine_example_fast(i):
                        num_changed += 1
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            iteration += 1
            
            # Early stopping if we have enough support vectors
            if np.sum(self._alphas > self.tol) > min(100, len(self._alphas) // 2):
                break
        
        if iteration >= self.max_iter:
            warnings.warn(f"Fast SMO did not fully converge after {self.max_iter} iterations", UserWarning)
    
    def _examine_example_fast(self, i1: int) -> bool:
        """Fast example examination with relaxed KKT conditions."""
        y1 = self._y[i1]
        alpha1 = self._alphas[i1]
        E1 = self._compute_error(i1)
        r1 = E1 * y1
        
        # Relaxed KKT conditions for speed
        tolerance = self.tol * 2  # More relaxed tolerance
        
        if ((r1 < -tolerance and alpha1 < self.C) or (r1 > tolerance and alpha1 > 0)):
            # Select second alpha
            i2 = self._select_second_alpha_fast(i1, E1)
            
            if self._optimize_pair_fast(i1, i2):
                return True
        
        return False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FastSVM':
        """
        Fit the SVM model to the training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values (class labels).
        
        Returns:
        --------
        self : FastSVM
            Fitted estimator.
        """
        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float32)  # Use float32 for speed
        y = np.asarray(y)
        
        # Store unique classes
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 2:
            raise ValueError("FastSVM currently supports only binary classification")
        
        # Convert labels to -1 and +1
        y_binary = np.where(y == self.classes_[0], -1, 1).astype(np.float32)
        
        # Store training data
        self._X = X
        self._y = y_binary
        
        # Pre-compute gamma value for RBF kernel
        if self.kernel == 'rbf':
            if isinstance(self.gamma, str) and self.gamma == 'scale':
                self._gamma_val = 1.0 / (X.shape[1] * np.var(X))
            else:
                self._gamma_val = self.gamma
        
        # Initialize alphas
        self._alphas = np.zeros(len(X), dtype=np.float32)
        self._b = 0.0
        
        # Run fast SMO algorithm
        self._smo_algorithm_fast()
        
        # Extract support vectors
        support_indices = np.where(self._alphas > self.tol)[0]
        self.support_vectors_ = X[support_indices]
        self.support_vector_labels_ = y_binary[support_indices]
        self.dual_coef_ = self._alphas[support_indices] * y_binary[support_indices]
        self.intercept_ = self._b
        self.n_support_ = len(support_indices)
        
        return self
    
    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the decision function for samples in X."""
        if self.support_vectors_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X, dtype=np.float32)
        
        if self.kernel == 'linear':
            # Efficient linear decision function
            w = np.sum((self.dual_coef_[:, np.newaxis] * self.support_vectors_), axis=0)
            return np.dot(X, w) + self.intercept_
        else:
            # RBF kernel decision function
            decisions = np.zeros(X.shape[0])
            
            for i, x in enumerate(X):
                decision = self.intercept_
                for j, sv in enumerate(self.support_vectors_):
                    kernel_val = self._compute_kernel_value(x, sv)
                    decision += self.dual_coef_[j] * kernel_val
                decisions[i] = decision
            
            return decisions
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        decision = self._decision_function(X)
        predictions = np.where(decision >= 0, 1, -1)
        
        # Convert back to original class labels
        return np.where(predictions == -1, self.classes_[0], self.classes_[1])
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the decision function for the samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns:
        --------
        X_new : ndarray of shape (n_samples,)
            Decision function values.
        """
        return self._decision_function(X)


# Alias for compatibility
SVM = FastSVM