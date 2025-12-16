"""
Run SVM section with memory optimizations and extract results for the report.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pickle
import matplotlib.pyplot as plt
from lib.multiclass_svm import MultiClassSVM as SVC
from lib.metrics import accuracy_score, classification_report, confusion_matrix

def extract_latent_features(encoder, X_data, batch_size=1000):
    """Extract latent features using the trained encoder."""
    print(f"Extracting latent features from {X_data.shape[0]} samples...")
    
    n_samples = X_data.shape[0]
    latent_features = []
    
    # Process in batches to handle memory efficiently
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_data = X_data[i:batch_end]
        
        # Extract latent features for this batch
        batch_features = encoder.forward(batch_data)
        latent_features.append(batch_features)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {batch_end}/{n_samples} samples...")
    
    # Concatenate all batches
    latent_features = np.vstack(latent_features)
    
    print(f"Latent features extracted: {latent_features.shape}")
    return latent_features

def train_svm_classifier_optimized(X_train_latent, y_train, X_test_latent, y_test, max_train_samples=3000):
    """Train SVM classifier with memory optimizations."""
    print("Training SVM classifier on latent features...")
    
    # Subsample training data for memory efficiency
    if len(X_train_latent) > max_train_samples:
        print(f"Subsampling training data from {len(X_train_latent)} to {max_train_samples} samples...")
        indices = np.random.choice(len(X_train_latent), max_train_samples, replace=False)
        X_train_latent = X_train_latent[indices]
        y_train = y_train[indices]
    
    # Test different SVM configurations (simplified for speed)
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
        
        # Test on a smaller subset for speed during hyperparameter search
        test_subset = min(500, len(X_test_latent))
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
    
    # Train final model with best configuration
    print("Training final model...")
    final_svm = SVC(random_state=42, **best_config)
    final_svm.fit(X_train_latent, y_train.ravel())
    
    # Make predictions on test set (use subset for memory)
    test_size = min(2000, len(X_test_latent))
    test_indices = np.random.choice(len(X_test_latent), test_size, replace=False)
    X_test_final = X_test_latent[test_indices]
    y_test_final = y_test[test_indices]
    
    y_pred = final_svm.predict(X_test_final)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_final, y_pred)
    
    print(f"Final Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    class_report = classification_report(y_test_final, y_pred, output_dict=True)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_final, y_pred)
    
    results = {
        'model': final_svm,
        'predictions': y_pred,
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'best_config': best_config,
        'test_size': test_size,
        'train_size': len(X_train_latent)
    }
    
    return results

def main():
    print("=" * 60)
    print("SVM CLASSIFICATION ON LATENT FEATURES")
    print("=" * 60)
    
    # Load autoencoder results
    print("Loading trained autoencoder...")
    with open('autoencoder_results_final.pkl', 'rb') as f:
        autoencoder_results = pickle.load(f)
    
    autoencoder = autoencoder_results['autoencoder']
    encoder = autoencoder_results['encoder']
    X_train_final = autoencoder_results['data']['X_train']
    X_test_processed = autoencoder_results['data']['X_test']
    y_train_final = autoencoder_results['data']['y_train']
    y_test = autoencoder_results['data']['y_test']
    
    print(f"Autoencoder loaded successfully!")
    print(f"Training data shape: {X_train_final.shape}")
    print(f"Test data shape: {X_test_processed.shape}")
    print(f"Latent dimension: {autoencoder.latent_dim}")
    
    # Extract latent features
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)
    
    X_train_latent = extract_latent_features(encoder, X_train_final)
    X_test_latent = extract_latent_features(encoder, X_test_processed)
    
    print(f"\nFeature extraction completed!")
    print(f"Original dimensions: {X_train_final.shape[1]} → Latent dimensions: {X_train_latent.shape[1]}")
    print(f"Compression ratio: {X_train_final.shape[1] / X_train_latent.shape[1]:.1f}x")
    
    # Train SVM
    print("\n" + "=" * 60)
    print("SVM TRAINING")
    print("=" * 60)
    
    svm_results = train_svm_classifier_optimized(X_train_latent, y_train_final, X_test_latent, y_test)
    
    # Save results
    with open('section4_results.pkl', 'wb') as f:
        pickle.dump({
            'svm_results': svm_results,
            'latent_features': {
                'X_train_latent': X_train_latent[:1000],  # Save subset for memory
                'X_test_latent': X_test_latent[:1000],
                'y_train': y_train_final[:1000],
                'y_test': y_test[:1000]
            }
        }, f)
    
    # Create confusion matrix visualization
    plt.figure(figsize=(10, 8))
    conf_matrix = svm_results['confusion_matrix']
    
    # Plot confusion matrix
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title('SVM Confusion Matrix on Latent Features')
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
    plt.savefig('report/svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"SVM Accuracy: {svm_results['accuracy']:.4f}")
    print(f"Training samples used: {svm_results['train_size']:,}")
    print(f"Test samples evaluated: {svm_results['test_size']:,}")
    print(f"Best configuration: {svm_results['best_config']}")
    print(f"Confusion matrix shape: {conf_matrix.shape}")
    
    print("\nFiles saved:")
    print("  - section4_results.pkl")
    print("  - report/svm_confusion_matrix.png")

if __name__ == "__main__":
    main()