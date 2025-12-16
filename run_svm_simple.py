"""
Simple SVM demonstration for the report.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pickle
import matplotlib.pyplot as plt
from lib.multiclass_svm import MultiClassSVM as SVC
from lib.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("=" * 60)
    print("SIMPLIFIED SVM DEMONSTRATION")
    print("=" * 60)
    
    # Load autoencoder results
    print("Loading trained autoencoder...")
    with open('autoencoder_results_final.pkl', 'rb') as f:
        autoencoder_results = pickle.load(f)
    
    encoder = autoencoder_results['encoder']
    X_test_processed = autoencoder_results['data']['X_test']
    y_test = autoencoder_results['data']['y_test']
    
    # Use a small subset for demonstration
    n_samples = 1000
    indices = np.random.choice(len(X_test_processed), n_samples, replace=False)
    X_subset = X_test_processed[indices]
    y_subset = y_test[indices]
    
    print(f"Using {n_samples} samples for demonstration")
    
    # Extract latent features
    print("Extracting latent features...")
    X_latent = encoder.forward(X_subset)
    
    # Split into train/test
    split_idx = int(0.7 * len(X_latent))
    X_train_latent = X_latent[:split_idx]
    X_test_latent = X_latent[split_idx:]
    y_train = y_subset[:split_idx]
    y_test_final = y_subset[split_idx:]
    
    print(f"Training set: {X_train_latent.shape}")
    print(f"Test set: {X_test_latent.shape}")
    
    # Train simple linear SVM
    print("Training linear SVM...")
    svm = SVC(kernel='linear', C=1.0, max_iter=100, random_state=42)
    svm.fit(X_train_latent, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test_latent)
    accuracy = accuracy_score(y_test_final, y_pred)
    
    print(f"SVM Accuracy: {accuracy:.4f}")
    
    # Get classification report
    class_report = classification_report(y_test_final, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test_final, y_pred)
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title('SVM Confusion Matrix (Linear Kernel)')
    plt.colorbar()
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('report/svm_confusion_matrix_simple.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'train_size': len(X_train_latent),
        'test_size': len(X_test_latent),
        'kernel': 'linear',
        'C': 1.0
    }
    
    with open('svm_simple_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"SVM Accuracy: {accuracy:.4f}")
    print(f"Training samples: {len(X_train_latent)}")
    print(f"Test samples: {len(X_test_latent)}")
    print(f"Kernel: Linear")
    print(f"Regularization (C): 1.0")
    
    # Print per-class metrics
    print("\nPer-class Performance:")
    for class_id in range(10):
        if str(class_id) in class_report:
            metrics = class_report[str(class_id)]
            print(f"  Class {class_id}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    print(f"\nOverall Accuracy: {class_report['accuracy']:.4f}")
    print(f"Macro Average F1: {class_report['macro avg']['f1-score']:.4f}")
    
    print("\nFiles saved:")
    print("  - svm_simple_results.pkl")
    print("  - report/svm_confusion_matrix_simple.png")

if __name__ == "__main__":
    main()