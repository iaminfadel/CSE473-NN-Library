"""
Test script for our custom SVM implementation.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from lib.svm import SVM
from lib.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def test_svm_basic():
    """Test basic SVM functionality."""
    print("Testing basic SVM functionality...")
    
    # Generate a simple binary classification dataset
    X, y = make_classification(
        n_samples=200, 
        n_features=10, 
        n_redundant=0, 
        n_informative=10,
        n_clusters_per_class=1, 
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Test linear kernel
    print("\nTesting Linear SVM...")
    svm_linear = SVM(kernel='linear', C=1.0, random_state=42)
    svm_linear.fit(X_train, y_train)
    
    y_pred_linear = svm_linear.predict(X_test)
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    print(f"Linear SVM Accuracy: {accuracy_linear:.4f}")
    
    # Test RBF kernel
    print("\nTesting RBF SVM...")
    svm_rbf = SVM(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_rbf.fit(X_train, y_train)
    
    y_pred_rbf = svm_rbf.predict(X_test)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    print(f"RBF SVM Accuracy: {accuracy_rbf:.4f}")
    
    # Print classification report
    print("\nClassification Report (RBF SVM):")
    report = classification_report(y_test, y_pred_rbf)
    print(report)
    
    print("\nSVM test completed successfully!")
    return accuracy_linear, accuracy_rbf


def test_svm_mnist_subset():
    """Test SVM on a small MNIST-like dataset."""
    print("\nTesting SVM on MNIST-like data...")
    
    # Create a simple 2D dataset that mimics image features
    np.random.seed(42)
    
    # Class 0: centered around (2, 2)
    X_class0 = np.random.normal([2, 2], 1, (100, 2))
    y_class0 = np.zeros(100)
    
    # Class 1: centered around (-2, -2)
    X_class1 = np.random.normal([-2, -2], 1, (100, 2))
    y_class1 = np.ones(100)
    
    # Combine data
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([y_class0, y_class1])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train SVM
    svm = SVM(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    
    # Predict
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"MNIST-like test accuracy: {accuracy:.4f}")
    print(f"Number of support vectors: {svm.n_support_}")
    
    return accuracy


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CUSTOM SVM IMPLEMENTATION")
    print("=" * 60)
    
    try:
        # Test basic functionality
        acc_linear, acc_rbf = test_svm_basic()
        
        # Test on MNIST-like data
        acc_mnist = test_svm_mnist_subset()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print(f"Linear SVM accuracy: {acc_linear:.4f}")
        print(f"RBF SVM accuracy: {acc_rbf:.4f}")
        print(f"MNIST-like accuracy: {acc_mnist:.4f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()