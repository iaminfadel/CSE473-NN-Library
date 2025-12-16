"""
Test the simple SVM implementation on a small dataset.
"""

import numpy as np
import sys
import os
import time
sys.path.insert(0, os.path.abspath('.'))

from lib.simple_svm import SimpleMultiClassSVM as SVC
from lib.metrics import accuracy_score
from sklearn.datasets import make_classification


def test_simple_binary():
    """Test simple binary classification."""
    print("Testing Simple Binary SVM...")
    
    # Generate simple binary dataset
    X, y = make_classification(
        n_samples=200, 
        n_features=10, 
        n_classes=2,
        n_redundant=0, 
        n_informative=8,
        n_clusters_per_class=1, 
        random_state=42
    )
    
    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Dataset: {X_train.shape} -> {X_test.shape}")
    
    # Test RBF SVM
    start_time = time.time()
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', max_iter=200, random_state=42)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    training_time = time.time() - start_time
    
    print(f"Binary RBF SVM Accuracy: {accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    return accuracy, training_time


def test_simple_multiclass():
    """Test simple multi-class classification."""
    print("\nTesting Simple Multi-class SVM...")
    
    # Generate simple multi-class dataset
    X, y = make_classification(
        n_samples=300, 
        n_features=10, 
        n_classes=3,  # Start with 3 classes
        n_redundant=0, 
        n_informative=8,
        n_clusters_per_class=1, 
        random_state=42
    )
    
    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Dataset: {X_train.shape} -> {X_test.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Test RBF SVM
    start_time = time.time()
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', max_iter=200, random_state=42)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    training_time = time.time() - start_time
    
    print(f"Multi-class RBF SVM Accuracy: {accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    return accuracy, training_time


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING SIMPLE SVM IMPLEMENTATION")
    print("=" * 60)
    
    try:
        # Test binary classification
        acc_binary, time_binary = test_simple_binary()
        
        # Test multi-class classification
        acc_multi, time_multi = test_simple_multiclass()
        
        print("\n" + "=" * 60)
        print("SIMPLE SVM TEST RESULTS")
        print("=" * 60)
        print(f"Binary SVM: {acc_binary:.4f} accuracy in {time_binary:.2f}s")
        print(f"Multi-class SVM: {acc_multi:.4f} accuracy in {time_multi:.2f}s")
        
        if acc_binary > 0.8 and acc_multi > 0.7:
            print("✓ Accuracy targets met!")
        else:
            print("⚠ Need to improve accuracy")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()