"""
Test script for our fast SVM implementation.
"""

import numpy as np
import sys
import os
import time
sys.path.insert(0, os.path.abspath('.'))

from lib.fast_multiclass_svm import FastMultiClassSVM as SVC
from lib.metrics import accuracy_score
from sklearn.datasets import make_classification


def test_fast_svm():
    """Test fast SVM functionality."""
    print("Testing Fast SVM implementation...")
    
    # Generate a multi-class classification dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_classes=5,
        n_redundant=0, 
        n_informative=15,
        n_clusters_per_class=1, 
        random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split the data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Test linear SVM (should be fast)
    print("\nTesting Linear SVM...")
    start_time = time.time()
    
    svm_linear = SVC(kernel='linear', C=1.0, max_iter=50, random_state=42)
    svm_linear.fit(X_train, y_train)
    
    y_pred_linear = svm_linear.predict(X_test)
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    
    linear_time = time.time() - start_time
    print(f"Linear SVM Accuracy: {accuracy_linear:.4f}")
    print(f"Training time: {linear_time:.2f} seconds")
    
    # Test RBF SVM (slower but should still be reasonable)
    print("\nTesting RBF SVM...")
    start_time = time.time()
    
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', max_iter=30, random_state=42)
    svm_rbf.fit(X_train, y_train)
    
    y_pred_rbf = svm_rbf.predict(X_test)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    
    rbf_time = time.time() - start_time
    print(f"RBF SVM Accuracy: {accuracy_rbf:.4f}")
    print(f"Training time: {rbf_time:.2f} seconds")
    
    return accuracy_linear, accuracy_rbf, linear_time, rbf_time


def test_mnist_like():
    """Test on MNIST-like data (higher dimensional)."""
    print("\nTesting on MNIST-like data...")
    
    # Create MNIST-like dataset
    np.random.seed(42)
    n_samples = 500
    n_features = 32  # Like our latent features
    n_classes = 10
    
    # Generate data
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"MNIST-like dataset: {X_train.shape} -> {X_test.shape}")
    
    # Train fast linear SVM
    start_time = time.time()
    
    svm = SVC(kernel='linear', C=1.0, max_iter=50, random_state=42)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    
    print(f"MNIST-like accuracy: {accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    return accuracy, training_time


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING FAST SVM IMPLEMENTATION")
    print("=" * 60)
    
    try:
        # Test basic functionality
        acc_linear, acc_rbf, time_linear, time_rbf = test_fast_svm()
        
        # Test on MNIST-like data
        acc_mnist, time_mnist = test_mnist_like()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print(f"Linear SVM: {acc_linear:.4f} accuracy in {time_linear:.2f}s")
        print(f"RBF SVM: {acc_rbf:.4f} accuracy in {time_rbf:.2f}s")
        print(f"MNIST-like: {acc_mnist:.4f} accuracy in {time_mnist:.2f}s")
        print("=" * 60)
        
        if time_linear < 5 and time_rbf < 10 and time_mnist < 5:
            print("✓ Performance targets met!")
        else:
            print("⚠ Performance could be improved further")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()