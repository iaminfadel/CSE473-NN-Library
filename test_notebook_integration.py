"""
Test integration of custom SVM with the notebook workflow.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from lib.multiclass_svm import MultiClassSVM as SVC
from lib.metrics import accuracy_score, classification_report, confusion_matrix


def test_notebook_workflow():
    """Test the SVM workflow as used in the notebook."""
    print("Testing notebook integration workflow...")
    
    # Simulate latent features (32-dimensional like in the notebook)
    np.random.seed(42)
    n_samples = 1000
    n_features = 32
    n_classes = 10
    
    # Generate synthetic latent features
    X_train_latent = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, n_classes, n_samples)
    
    X_test_latent = np.random.randn(200, n_features)
    y_test = np.random.randint(0, n_classes, 200)
    
    print(f"Training data: {X_train_latent.shape}")
    print(f"Test data: {X_test_latent.shape}")
    print(f"Number of classes: {n_classes}")
    
    # Test the exact workflow from the notebook
    def train_svm_classifier_test(X_train_latent, y_train, X_test_latent, y_test):
        """Simplified version of the notebook function."""
        print("Training SVM classifier on latent features...")
        
        # Test different SVM configurations (simplified)
        configs = [
            {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'max_iter': 200},
            {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale', 'max_iter': 200},
        ]
        
        best_accuracy = 0
        best_config = None
        
        print("Testing different SVM configurations...")
        
        for i, config in enumerate(configs):
            print(f"  Config {i+1}/{len(configs)}: {config}")
            
            # Train SVM with this configuration
            svm = SVC(random_state=42, **config)
            svm.fit(X_train_latent, y_train.ravel())
            
            # Test on a subset for speed
            test_subset = min(100, len(X_test_latent))
            indices = np.random.choice(len(X_test_latent), test_subset, replace=False)
            X_test_subset = X_test_latent[indices]
            y_test_subset = y_test[indices]
            
            # Make predictions
            y_pred_subset = svm.predict(X_test_subset)
            accuracy = accuracy_score(y_test_subset, y_pred_subset)
            
            print(f"    Accuracy on subset: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config
        
        print(f"Best configuration: {best_config}")
        print(f"Best subset accuracy: {best_accuracy:.4f}")
        
        # Train final model
        print("Training final model...")
        final_svm = SVC(random_state=42, **best_config)
        final_svm.fit(X_train_latent, y_train.ravel())
        
        # Make predictions on full test set
        y_pred = final_svm.predict(X_test_latent)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Final Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'model': final_svm,
            'predictions': y_pred,
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'best_config': best_config
        }
    
    # Run the test
    results = train_svm_classifier_test(X_train_latent, y_train, X_test_latent, y_test)
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Confusion matrix shape: {results['confusion_matrix'].shape}")
    print(f"Best config: {results['best_config']}")
    
    return results['accuracy']


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING NOTEBOOK INTEGRATION")
    print("=" * 60)
    
    try:
        accuracy = test_notebook_workflow()
        
        print("\n" + "=" * 60)
        print("INTEGRATION TEST PASSED!")
        print(f"Final accuracy: {accuracy:.4f}")
        print("The notebook should work with our custom SVM implementation.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nINTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()