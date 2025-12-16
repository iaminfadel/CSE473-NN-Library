"""
Support Vector Machine (SVM) implementation from scratch.

This module implements a basic SVM classifier using the Sequential Minimal Optimization (SMO) algorithm
for solving the quadratic programming problem. It supports both linear and RBF kernels.
"""

import numpy as np
from typing import Optional, Literal, Tuple, Union
import warnings


class SVM:
    """
    Support Vector Machine classifier implementation from scratch.
    
    This implementation uses the Sequential Minimal Optimization (SMO) algorithm
    to solve the dual optimization problem for SVM classification.
    
    Parameters:
    -----------
    C : float, default=1.0
        Regularization parameter. Higher values mean less regularization.
    kernel : {'linear', 'rbf'}, default='rbf'
        Kernel type to use in the algorithm.
    gamma : float or 'scale', default='scale'
        Kernel coefficient for 'rbf'. If 'scale', uses 1 / (n_features * X.var()).
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    max_iter : int, default=1000
        Maximum number of iterations for the SMO algorithm.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal['linear', 'rbf'] = 'rbf',
        gamma: Union[float, str] = 'scale',
        tol: float = 1e-3,
        max_iter: int = 1000,
        random_state: Optional[int] = None
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Initialize attributes that will be set during training
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.n_support_ = None
        self.classes_ = None
        
        # Internal variables for SMO algorithm
        self._X = None
        self._y = None
        self._alphas = None
        self._b = 0.0
        self._kernel_matrix = None
        self._errors = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_kernel_value(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel value between two samples."""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            if isinstance(self.gamma, str) and self.gamma == 'scale':
                gamma_val = self._gamma_val
            else:
                gamma_val = self.gamma
            
            diff = x1 - x2
            return np.exp(-gamma_val * np.dot(diff, diff))
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _compute_kernel_row(self, i: int) -> np.ndarray:
        """Compute the i-th row of the kernel matrix on-demand."""
        x_i = self._X[i]
        row = np.zeros(len(self._X))
        for j in range(len(self._X)):
            row[j] = self._compute_kernel_value(x_i, self._X[j])
        return row
    
    def _compute_kernel_matrix_batch(self, X1: np.ndarray, X2: Optional[np.ndarray] = None, batch_size: int = 1000) -> np.ndarray:
        """Compute kernel matrix in batches to save memory."""
        if X2 is None:
            X2 = X1
            
        n1, n2 = X1.shape[0], X2.shape[0]
        
        # For small matrices, compute directly
        if n1 * n2 < 10000:
            return self._compute_kernel_matrix_direct(X1, X2)
        
        # For large matrices, use batching
        K = np.zeros((n1, n2))
        
        for i in range(0, n1, batch_size):
            end_i = min(i + batch_size, n1)
            for j in range(0, n2, batch_size):
                end_j = min(j + batch_size, n2)
                
                # Compute batch
                X1_batch = X1[i:end_i]
                X2_batch = X2[j:end_j]
                K_batch = self._compute_kernel_matrix_direct(X1_batch, X2_batch)
                K[i:end_i, j:end_j] = K_batch
        
        return K
    
    def _compute_kernel_matrix_direct(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix directly (for small matrices)."""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            if isinstance(self.gamma, str) and self.gamma == 'scale':
                gamma_val = self._gamma_val
            else:
                gamma_val = self.gamma
                
            # Efficient computation of squared distances
            X1_norm = np.sum(X1**2, axis=1, keepdims=True)
            X2_norm = np.sum(X2**2, axis=1, keepdims=True)
            distances_sq = X1_norm + X2_norm.T - 2 * np.dot(X1, X2.T)
            
            return np.exp(-gamma_val * distances_sq)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _compute_error(self, i: int) -> float:
        """Compute the error for sample i."""
        # Compute prediction without storing full kernel matrix
        prediction = self._b
        x_i = self._X[i]
        
        for j in range(len(self._alphas)):
            if self._alphas[j] > 0:  # Only compute for non-zero alphas
                kernel_val = self._compute_kernel_value(x_i, self._X[j])
                prediction += self._alphas[j] * self._y[j] * kernel_val
        
        return prediction - self._y[i]
    
    def _select_second_alpha(self, i1: int, E1: float) -> int:
        """Select the second alpha using the heuristic from SMO."""
        # Find the alpha that maximizes |E1 - E2|
        max_step = 0
        i2 = -1
        
        # Look for non-bound alphas first
        non_bound_indices = np.where((self._alphas > self.tol) & (self._alphas < self.C - self.tol))[0]
        
        if len(non_bound_indices) > 1:
            for i in non_bound_indices:
                if i == i1:
                    continue
                E2 = self._compute_error(i)
                step = abs(E1 - E2)
                if step > max_step:
                    max_step = step
                    i2 = i
        
        # If no good candidate found, select randomly
        if i2 == -1:
            candidates = list(range(len(self._alphas)))
            candidates.remove(i1)
            i2 = np.random.choice(candidates)
        
        return i2
    
    def _optimize_pair(self, i1: int, i2: int) -> bool:
        """Optimize the pair of alphas (i1, i2) using SMO."""
        if i1 == i2:
            return False
        
        alpha1_old = self._alphas[i1]
        alpha2_old = self._alphas[i2]
        y1, y2 = self._y[i1], self._y[i2]
        
        # Compute bounds L and H
        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha1_old + alpha2_old - self.C)
            H = min(self.C, alpha1_old + alpha2_old)
        
        if L == H:
            return False
        
        # Compute kernel values on-demand
        x1, x2 = self._X[i1], self._X[i2]
        k11 = self._compute_kernel_value(x1, x1)
        k12 = self._compute_kernel_value(x1, x2)
        k22 = self._compute_kernel_value(x2, x2)
        
        # Compute eta (second derivative)
        eta = k11 + k22 - 2 * k12
        
        if eta <= 0:
            return False
        
        # Compute errors
        E1 = self._compute_error(i1)
        E2 = self._compute_error(i2)
        
        # Compute new alpha2
        alpha2_new = alpha2_old + y2 * (E1 - E2) / eta
        
        # Clip alpha2
        if alpha2_new >= H:
            alpha2_new = H
        elif alpha2_new <= L:
            alpha2_new = L
        
        # Check if change is significant
        if abs(alpha2_new - alpha2_old) < self.tol:
            return False
        
        # Compute new alpha1
        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)
        
        # Update alphas
        self._alphas[i1] = alpha1_new
        self._alphas[i2] = alpha2_new
        
        # Update bias term
        b1 = self._b - E1 - y1 * (alpha1_new - alpha1_old) * k11 - y2 * (alpha2_new - alpha2_old) * k12
        b2 = self._b - E2 - y1 * (alpha1_new - alpha1_old) * k12 - y2 * (alpha2_new - alpha2_old) * k22
        
        if 0 < alpha1_new < self.C:
            self._b = b1
        elif 0 < alpha2_new < self.C:
            self._b = b2
        else:
            self._b = (b1 + b2) / 2
        
        return True
    
    def _smo_algorithm(self) -> None:
        """Sequential Minimal Optimization algorithm."""
        num_changed = 0
        examine_all = True
        iteration = 0
        
        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            
            if examine_all:
                # Examine all examples
                for i in range(len(self._alphas)):
                    num_changed += self._examine_example(i)
            else:
                # Examine non-bound examples
                non_bound_indices = np.where((self._alphas > self.tol) & (self._alphas < self.C - self.tol))[0]
                for i in non_bound_indices:
                    num_changed += self._examine_example(i)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            iteration += 1
        
        if iteration >= self.max_iter:
            warnings.warn(f"SMO algorithm did not converge after {self.max_iter} iterations")
    
    def _examine_example(self, i1: int) -> int:
        """Examine example i1 and try to optimize it."""
        y1 = self._y[i1]
        alpha1 = self._alphas[i1]
        E1 = self._compute_error(i1)
        r1 = E1 * y1
        
        # Check KKT conditions
        if ((r1 < -self.tol and alpha1 < self.C) or (r1 > self.tol and alpha1 > 0)):
            # Select second alpha
            i2 = self._select_second_alpha(i1, E1)
            
            if self._optimize_pair(i1, i2):
                return 1
        
        return 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
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
        self : SVM
            Fitted estimator.
        """
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Store unique classes
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 2:
            raise ValueError("SVM currently supports only binary classification")
        
        # Convert labels to -1 and +1
        y_binary = np.where(y == self.classes_[0], -1, 1)
        
        # Store training data
        self._X = X
        self._y = y_binary
        
        # Pre-compute gamma value for RBF kernel
        if self.kernel == 'rbf' and isinstance(self.gamma, str) and self.gamma == 'scale':
            self._gamma_val = 1.0 / (X.shape[1] * np.var(X))
        else:
            self._gamma_val = self.gamma
        
        # Initialize alphas
        self._alphas = np.zeros(len(X))
        self._b = 0.0
        
        # Run SMO algorithm (no need to store full kernel matrix)
        self._smo_algorithm()
        
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
        
        # Compute decision function without storing full kernel matrix
        decision = np.full(X.shape[0], self.intercept_)
        
        for i, x in enumerate(X):
            for j, sv in enumerate(self.support_vectors_):
                kernel_val = self._compute_kernel_value(x, sv)
                decision[i] += self.dual_coef_[j] * kernel_val
        
        return decision
    
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


# Alias for sklearn compatibility
SVC = SVM