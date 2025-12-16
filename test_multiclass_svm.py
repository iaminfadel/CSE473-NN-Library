"""
Test script for our multi-class SVM implementation.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from lib.multiclass_svm import MultiClassSVM
from lib.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification


def test_multiclass_svm():
    """Test multi-class SVM functionality."""
    print("Testing multi-class SVM functionality...")
    
    # Generate a multi-class classification dataset
    X, y = make_classification(
        n_samples=300, 
        n_features=10, 
        n_classes=5,
        n_redundant=0, 
        n_informative=10,
        n_clusters_per_class=1, 
        random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Test multi-class SVM
    print("\nTraining Multi-class SVM...")
    svm = MultiClassSVM(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Multi-class SVM Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    return accuracy


def test_binary_classification():
    """Test that multi-class SVM works for binary classification too."""
    print("\nTesting binary classification with MultiClassSVM...")
    
    # Generate binary classification dataset
    X, y = make_classification(
        n_samples=200, 
        n_features=10, 
        n_classes=2,
        n_redundant=0, 
        n_informative=10,
        n_clusters_per_class=1, 
        random_state=42
    )
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train SVM
    svm = MultiClassSVM(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Binary classification accuracy: {accuracy:.4f}")
    
    return accuracy


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MULTI-CLASS SVM IMPLEMENTATION")
    print("=" * 60)
    
    try:
        # Test multi-class functionality
        acc_multiclass = test_multiclass_svm()
        
        # Test binary classification
        acc_binary = test_binary_classification()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print(f"Multi-class SVM accuracy: {acc_multiclass:.4f}")
        print(f"Binary SVM accuracy: {acc_binary:.4f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()