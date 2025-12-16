"""
Simple but reliable SVM implementation focusing on correctness over speed.

This implementation prioritizes getting the mathematics right and achieving
good accuracy, even if it's not the fastest implementation.
"""

import numpy as np
from typing import Optional, Literal, Union
import warnings


class SimpleSVM:
    """
    Simple SVM classifier with focus on correctness and good accuracy.
    
    This implementation uses a simplified but reliable approach to SVM training
    that should achieve good accuracy on most datasets.
    
    Parameters:
    -----------
    C : float, default=1.0
        Regularization parameter.
    kernel : {'linear', 'rbf'}, default='rbf'
        Kernel type.
    gamma : float or 'scale', default='scale'
        Kernel coefficient for 'rbf'.
    tol : float, default=1e-4
        Tolerance for stopping criterion.
    max_iter : int, default=1000
        Maximum number of iterations.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal['linear', 'rbf'] = 'rbf',
        gamma: Union[float, str] = 'scale',
        tol: float = 1e-4,
        max_iter: int = 1000,
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
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_kernel_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the full kernel matrix."""
        if X2 is None:
            X2 = X1
            
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            # Compute RBF kernel matrix
            if isinstance(self.gamma, str) and self.gamma == 'scale':
                gamma_val = 1.0 / (X1.shape[1] * np.var(X1))
            else:
                gamma_val = self.gamma
            
            # Compute squared distances efficiently
            X1_norm = np.sum(X1**2, axis=1, keepdims=True)
            X2_norm = np.sum(X2**2, axis=1, keepdims=True)
            distances_sq = X1_norm + X2_norm.T - 2 * np.dot(X1, X2.T)
            
            return np.exp(-gamma_val * distances_sq)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _simplified_smo(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Simplified SMO algorithm that focuses on correctness."""
        n_samples = len(X)
        
        # Initialize
        alphas = np.zeros(n_samples)
        b = 0.0
        
        # Compute kernel matrix once
        K = self._compute_kernel_matrix(X)
        
        # Simple SMO loop
        for iteration in range(self.max_iter):
            num_changed_alphas = 0
            
            for i in range(n_samples):
                # Compute prediction for sample i
                prediction_i = np.sum(alphas * y * K[i, :]) + b
                E_i = prediction_i - y[i]
                
                # Check KKT conditions
                if ((y[i] * E_i < -self.tol and alphas[i] < self.C) or 
                    (y[i] * E_i > self.tol and alphas[i] > 0)):
                    
                    # Select j randomly (simple heuristic)
                    j = np.random.choice([idx for idx in range(n_samples) if idx != i])
                    
                    # Compute prediction for sample j
                    prediction_j = np.sum(alphas * y * K[j, :]) + b
                    E_j = prediction_j - y[j]
                    
                    # Save old alphas
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]
                    
                    # Compute bounds
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Compute new alpha_j
                    alphas[j] = alphas[j] - (y[j] * (E_i - E_j)) / eta
                    
                    # Clip alpha_j
                    if alphas[j] > H:
                        alphas[j] = H
                    elif alphas[j] < L:
                        alphas[j] = L
                    
                    # Check if change is significant
                    if abs(alphas[j] - alpha_j_old) < self.tol:
                        continue
                    
                    # Determine alpha_i
                    alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j])
                    
                    # Compute b1 and b2
                    b1 = (b - E_i - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - 
                          y[j] * (alphas[j] - alpha_j_old) * K[i, j])
                    
                    b2 = (b - E_j - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - 
                          y[j] * (alphas[j] - alpha_j_old) * K[j, j])
                    
                    # Update b
                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            # Check convergence
            if num_changed_alphas == 0:
                break
            
            # Progress report
            if iteration % 100 == 0:
                n_sv = np.sum(alphas > self.tol)
                print(f"    Iteration {iteration}: {n_sv} support vectors")
        
        return alphas, b
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleSVM':
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
        self : SimpleSVM
            Fitted estimator.
        """
        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        # Store unique classes
        self.classes_ = np.unique(y)
        
        if len(self.classes_) != 2:
            raise ValueError("SimpleSVM currently supports only binary classification")
        
        # Convert labels to -1 and +1
        y_binary = np.where(y == self.classes_[0], -1, 1).astype(np.float64)
        
        # Train using simplified SMO
        alphas, b = self._simplified_smo(X, y_binary)
        
        # Extract support vectors
        support_indices = np.where(alphas > self.tol)[0]
        
        if len(support_indices) == 0:
            # Fallback: use all samples as support vectors with small alphas
            print("Warning: No support vectors found, using fallback method")
            support_indices = np.arange(len(X))
            alphas = np.full(len(X), 0.1)
        
        self.support_vectors_ = X[support_indices]
        self.support_vector_labels_ = y_binary[support_indices]
        self.dual_coef_ = alphas[support_indices] * y_binary[support_indices]
        self.intercept_ = b
        self.n_support_ = len(support_indices)
        
        return self
    
    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the decision function for samples in X."""
        if self.support_vectors_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Compute kernel matrix between X and support vectors
        K = self._compute_kernel_matrix(X, self.support_vectors_)
        
        # Compute decision function
        decision = np.dot(K, self.dual_coef_) + self.intercept_
        
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


# Multi-class wrapper
class SimpleMultiClassSVM:
    """Simple multi-class SVM using One-vs-Rest."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.classes_ = None
        self.classifiers_ = {}
        self.n_classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ == 2:
            print("Training binary classifier...")
            self.classifiers_['binary'] = SimpleSVM(**self.kwargs)
            self.classifiers_['binary'].fit(X, y)
        else:
            print(f"Training {self.n_classes_} binary classifiers (One-vs-Rest)...")
            
            for i, class_label in enumerate(self.classes_):
                print(f"  Training classifier {i+1}/{self.n_classes_} for class {class_label}...")
                
                # Create binary labels
                y_binary = np.where(y == class_label, 1, 0)
                
                # Check class balance
                n_positive = np.sum(y_binary)
                if n_positive < 2:
                    print(f"    Skipping class {class_label} (too few samples)")
                    continue
                
                # Train binary classifier
                classifier = SimpleSVM(**self.kwargs)
                try:
                    classifier.fit(X, y_binary)
                    self.classifiers_[class_label] = classifier
                    print(f"    Completed (support vectors: {classifier.n_support_})")
                except Exception as e:
                    print(f"    Failed: {e}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X, dtype=np.float64)
        
        if self.n_classes_ == 2:
            return self.classifiers_['binary'].predict(X)
        else:
            # Multi-class prediction
            decision_scores = np.full((X.shape[0], self.n_classes_), -np.inf)
            
            for i, class_label in enumerate(self.classes_):
                if class_label in self.classifiers_:
                    try:
                        scores = self.classifiers_[class_label].decision_function(X)
                        decision_scores[:, i] = scores
                    except Exception:
                        decision_scores[:, i] = -1.0
            
            predicted_indices = np.argmax(decision_scores, axis=1)
            return self.classes_[predicted_indices]
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.n_classes_ == 2:
            return self.classifiers_['binary'].decision_function(X)
        else:
            decision_scores = np.full((X.shape[0], self.n_classes_), -np.inf)
            
            for i, class_label in enumerate(self.classes_):
                if class_label in self.classifiers_:
                    try:
                        scores = self.classifiers_[class_label].decision_function(X)
                        decision_scores[:, i] = scores
                    except Exception:
                        decision_scores[:, i] = -1.0
            
            return decision_scores


# Alias for compatibility
SVC = SimpleMultiClassSVM