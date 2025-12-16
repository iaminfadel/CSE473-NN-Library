"""
SVM demonstration using sklearn with proper preprocessing.
"""

import sys
import os
import time
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 60)
    print("SKLEARN SVM DEMONSTRATION WITH AUTOENCODER DATA")
    print("=" * 60)
    
    # Load autoencoder results
    print("Loading trained autoencoder...")
    with open('autoencoder_results_final.pkl', 'rb') as f:
        autoencoder_results = pickle.load(f)
    
    encoder = autoencoder_results['encoder']
    X_train_full = autoencoder_results['data']['X_train']
    X_test_full = autoencoder_results['data']['X_test']
    y_train_full = autoencoder_results['data']['y_train']
    y_test_full = autoencoder_results['data']['y_test']
    
    print(f"Autoencoder loaded successfully!")
    print(f"Full training data: {X_train_full.shape}")
    print(f"Full test data: {X_test_full.shape}")
    
    # Use reasonable subset for demonstration
    n_train = 5000
    n_test = 2000
    
    print(f"\nUsing subset for demonstration:")
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    
    # Sample data
    np.random.seed(42)
    train_indices = np.random.choice(len(X_train_full), n_train, replace=False)
    test_indices = np.random.choice(len(X_test_full), n_test, replace=False)
    
    X_train = X_train_full[train_indices]
    X_test = X_test_full[test_indices]
    y_train = y_train_full[train_indices]
    y_test = y_test_full[test_indices]
    
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    # Extract latent features
    print("\nExtracting latent features...")
    start_time = time.time()
    
    X_train_latent = encoder.forward(X_train)
    X_test_latent = encoder.forward(X_test)
    
    feature_time = time.time() - start_time
    print(f"Feature extraction completed in {feature_time:.2f} seconds")
    print(f"Latent features shape: {X_train_latent.shape} -> {X_test_latent.shape}")
    print(f"Compression ratio: {X_train.shape[1] / X_train_latent.shape[1]:.1f}x")
    
    # Standardize latent features
    print("Standardizing latent features...")
    scaler_latent = StandardScaler()
    X_train_latent_scaled = scaler_latent.fit_transform(X_train_latent)
    X_test_latent_scaled = scaler_latent.transform(X_test_latent)
    
    # Train SVM on latent features
    print("\n" + "=" * 40)
    print("TRAINING SVM ON LATENT FEATURES")
    print("=" * 40)
    
    start_time = time.time()
    
    svm_latent = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    svm_latent.fit(X_train_latent_scaled, y_train.ravel())
    
    y_pred_latent = svm_latent.predict(X_test_latent_scaled)
    accuracy_latent = accuracy_score(y_test, y_pred_latent)
    
    latent_time = time.time() - start_time
    
    print(f"SVM on latent features:")
    print(f"  Accuracy: {accuracy_latent:.4f}")
    print(f"  Training time: {latent_time:.2f} seconds")
    
    # Train SVM on raw pixels (smaller subset)
    print("\n" + "=" * 40)
    print("TRAINING SVM ON RAW PIXELS (BASELINE)")
    print("=" * 40)
    
    # Use smaller subset for raw pixels
    n_raw = 1000
    raw_indices = np.random.choice(len(X_train), n_raw, replace=False)
    X_train_raw = X_train[raw_indices]
    y_train_raw = y_train[raw_indices]
    
    # Scale raw pixels
    scaler_raw = StandardScaler()
    X_train_raw_scaled = scaler_raw.fit_transform(X_train_raw)
    
    start_time = time.time()
    
    svm_raw = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_raw.fit(X_train_raw_scaled, y_train_raw.ravel())
    
    # Test on subset of test data
    test_raw_indices = np.random.choice(len(X_test), 500, replace=False)
    X_test_raw = X_test[test_raw_indices]
    y_test_raw = y_test[test_raw_indices]
    
    X_test_raw_scaled = scaler_raw.transform(X_test_raw)
    y_pred_raw = svm_raw.predict(X_test_raw_scaled)
    accuracy_raw = accuracy_score(y_test_raw, y_pred_raw)
    
    raw_time = time.time() - start_time
    
    print(f"SVM on raw pixels:")
    print(f"  Accuracy: {accuracy_raw:.4f}")
    print(f"  Training time: {raw_time:.2f} seconds")
    print(f"  Training samples: {len(X_train_raw)}")
    
    # Create confusion matrix for latent features
    conf_matrix = confusion_matrix(y_test, y_pred_latent)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title('SVM Confusion Matrix (Latent Features)')
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
    plt.savefig('report/svm_confusion_matrix_sklearn.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Get classification report
    class_report = classification_report(y_test, y_pred_latent, output_dict=True)
    
    # Save results
    results = {
        'latent_svm': {
            'accuracy': accuracy_latent,
            'training_time': latent_time,
            'training_samples': len(X_train_latent),
            'test_samples': len(X_test_latent),
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        },
        'raw_svm': {
            'accuracy': accuracy_raw,
            'training_time': raw_time,
            'training_samples': len(X_train_raw),
            'test_samples': len(X_test_raw)
        },
        'feature_extraction_time': feature_time
    }
    
    with open('sklearn_svm_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Latent Features SVM:")
    print(f"  Accuracy: {accuracy_latent:.4f}")
    print(f"  Training time: {latent_time:.2f}s")
    print(f"  Samples: {len(X_train_latent)} train, {len(X_test_latent)} test")
    print()
    print(f"Raw Pixels SVM (baseline):")
    print(f"  Accuracy: {accuracy_raw:.4f}")
    print(f"  Training time: {raw_time:.2f}s")
    print(f"  Samples: {len(X_train_raw)} train, {len(X_test_raw)} test")
    print()
    print(f"Feature extraction time: {feature_time:.2f}s")
    print(f"Compression ratio: {X_train.shape[1] / X_train_latent.shape[1]:.1f}x")
    
    # Per-class performance
    print("\nPer-class Performance (Latent Features):")
    for class_id in range(10):
        if str(class_id) in class_report:
            metrics = class_report[str(class_id)]
            print(f"  Digit {class_id}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {class_report['accuracy']:.4f}")
    print(f"  Macro Avg F1: {class_report['macro avg']['f1-score']:.4f}")
    print(f"  Weighted Avg F1: {class_report['weighted avg']['f1-score']:.4f}")
    
    print("\nFiles saved:")
    print("  - sklearn_svm_results.pkl")
    print("  - report/svm_confusion_matrix_sklearn.png")

if __name__ == "__main__":
    main()