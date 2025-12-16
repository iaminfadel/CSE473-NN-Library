"""
Fast Multi-class SVM implementation using optimized binary SVM.

This implementation focuses on speed while maintaining educational value.
Key optimizations:
- Uses fast binary SVM implementation
- Parallel-friendly structure (though not parallelized here)
- Early stopping for multi-class training
- Memory-efficient prediction
"""

import numpy as np
from typing import Optional, Literal, Union
from .fast_svm import FastSVM


class FastMultiClassSVM:
    """
    Fast Multi-class SVM classifier using One-vs-Rest strategy.
    
    This class wraps our fast binary SVM implementation to handle multi-class
    classification efficiently.
    
    Parameters:
    -----------
    C : float, default=1.0
        Regularization parameter.
    kernel : {'linear', 'rbf'}, default='linear'
        Kernel type (linear is much faster).
    gamma : float or 'scale', default='scale'
        Kernel coefficient for 'rbf'.
    tol : float, default=1e-2
        Tolerance for stopping criterion.
    max_iter : int, default=100
        Maximum number of iterations.
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
        
        # Will be set during training
        self.classes_ = None
        self.classifiers_ = {}
        self.n_classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FastMultiClassSVM':
        """
        Fit the multi-class SVM model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values (class labels).
        
        Returns:
        --------
        self : FastMultiClassSVM
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        
        # Get unique classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ == 2:
            # Binary classification - use single SVM
            print("Training binary classifier...")
            self.classifiers_['binary'] = FastSVM(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                tol=self.tol,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            self.classifiers_['binary'].fit(X, y)
        else:
            # Multi-class - use One-vs-Rest with optimizations
            print(f"Training {self.n_classes_} binary classifiers (One-vs-Rest)...")
            
            for i, class_label in enumerate(self.classes_):
                print(f"  Training classifier {i+1}/{self.n_classes_} for class {class_label}...")
                
                # Create binary labels: current class vs all others
                y_binary = np.where(y == class_label, 1, 0)
                
                # Skip if class has too few samples
                if np.sum(y_binary) < 2:
                    print(f"    Skipping class {class_label} (too few samples)")
                    continue
                
                # Train fast binary SVM
                classifier = FastSVM(
                    C=self.C,
                    kernel=self.kernel,
                    gamma=self.gamma,
                    tol=self.tol,
                    max_iter=self.max_iter,
                    random_state=self.random_state
                )
                
                try:
                    classifier.fit(X, y_binary)
                    self.classifiers_[class_label] = classifier
                    print(f"    Completed (support vectors: {classifier.n_support_})")
                except Exception as e:
                    print(f"    Failed to train classifier for class {class_label}: {e}")
                    continue
        
        return self
    
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
        if self.classes_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X, dtype=np.float32)
        
        if self.n_classes_ == 2:
            # Binary classification
            return self.classifiers_['binary'].predict(X)
        else:
            # Multi-class: get decision scores from all classifiers
            decision_scores = np.full((X.shape[0], self.n_classes_), -np.inf)
            
            for i, class_label in enumerate(self.classes_):
                if class_label in self.classifiers_:
                    try:
                        scores = self.classifiers_[class_label].decision_function(X)
                        decision_scores[:, i] = scores
                    except Exception:
                        # If classifier fails, use default score
                        decision_scores[:, i] = -1.0
            
            # Predict class with highest decision score
            predicted_indices = np.argmax(decision_scores, axis=1)
            return self.classes_[predicted_indices]
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the decision function for the samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns:
        --------
        X_new : ndarray of shape (n_samples, n_classes)
            Decision function values.
        """
        if self.classes_ is None:
            raise ValueError("Model must be fitted before computing decision function")
        
        X = np.asarray(X, dtype=np.float32)
        
        if self.n_classes_ == 2:
            # Binary classification
            return self.classifiers_['binary'].decision_function(X)
        else:
            # Multi-class: return decision scores from all classifiers
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
SVC = FastMultiClassSVM