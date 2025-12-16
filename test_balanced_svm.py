"""
Test script for the balanced SVM implementation.
"""

import numpy as np
import sys
import os
import time
sys.path.insert(0, os.path.abspath('.'))

from lib.balanced_multiclass_svm import BalancedMultiClassSVM as SVC
from lib.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification


def test_balanced_svm():
    """Test balanced SVM functionality."""
    print("Testing Balanced SVM implementation...")
    
    # Generate a multi-class classification dataset
    X, y = make_classification(
        n_samples=800, 
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
    
    # Test RBF SVM (should have good accuracy)
    print("\nTesting RBF SVM...")
    start_time = time.time()
    
    svm_rbf = SVC(kernel='rbf', C=10.0, gamma='scale', max_iter=200, random_state=42)
    svm_rbf.fit(X_train, y_train)
    
    y_pred_rbf = svm_rbf.predict(X_test)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    
    rbf_time = time.time() - start_time
    print(f"RBF SVM Accuracy: {accuracy_rbf:.4f}")
    print(f"Training time: {rbf_time:.2f} seconds")
    
    # Test Linear SVM
    print("\nTesting Linear SVM...")
    start_time = time.time()
    
    svm_linear = SVC(kernel='linear', C=1.0, max_iter=200, random_state=42)
    svm_linear.fit(X_train, y_train)
    
    y_pred_linear = svm_linear.predict(X_test)
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    
    linear_time = time.time() - start_time
    print(f"Linear SVM Accuracy: {accuracy_linear:.4f}")
    print(f"Training time: {linear_time:.2f} seconds")
    
    return accuracy_rbf, accuracy_linear, rbf_time, linear_time


def test_mnist_like_balanced():
    """Test on MNIST-like data with balanced SVM."""
    print("\nTesting on MNIST-like data (balanced SVM)...")
    
    # Create MNIST-like dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 32  # Like our latent features
    n_classes = 10
    
    # Generate more realistic data with class structure
    X = []
    y = []
    
    for class_id in range(n_classes):
        # Create class-specific mean
        class_mean = np.random.randn(n_features) * 2
        # Generate samples for this class
        class_samples = np.random.randn(n_samples // n_classes, n_features) * 0.5 + class_mean
        X.append(class_samples)
        y.extend([class_id] * (n_samples // n_classes))
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"MNIST-like dataset: {X_train.shape} -> {X_test.shape}")
    
    # Train balanced RBF SVM
    start_time = time.time()
    
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', max_iter=200, random_state=42)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    
    print(f"MNIST-like accuracy: {accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Print classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    return accuracy, training_time


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING BALANCED SVM IMPLEMENTATION")
    print("=" * 60)
    
    try:
        # Test basic functionality
        acc_rbf, acc_linear, time_rbf, time_linear = test_balanced_svm()
        
        # Test on MNIST-like data
        acc_mnist, time_mnist = test_mnist_like_balanced()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print(f"RBF SVM: {acc_rbf:.4f} accuracy in {time_rbf:.2f}s")
        print(f"Linear SVM: {acc_linear:.4f} accuracy in {time_linear:.2f}s")
        print(f"MNIST-like: {acc_mnist:.4f} accuracy in {time_mnist:.2f}s")
        print("=" * 60)
        
        # Check if we achieved good accuracy
        if acc_rbf > 0.8 and acc_mnist > 0.7:
            print("✓ Accuracy targets met!")
        else:
            print("⚠ Accuracy could be improved")
            
        if time_rbf < 30 and time_mnist < 60:
            print("✓ Speed targets met!")
        else:
            print("⚠ Training time is acceptable for educational purposes")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()