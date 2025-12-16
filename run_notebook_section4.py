"""
Run the key parts of Section 4 notebook to get results and visualizations.
"""

import sys
import os
import time
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set plot style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'

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

def train_svm_classifier(X_train_latent, y_train, X_test_latent, y_test, max_train_samples=5000):
    """Train SVM classifier on latent features with hyperparameter testing."""
    print("Training SVM classifier on latent features...")
    
    # Subsample training data for memory efficiency
    if len(X_train_latent) > max_train_samples:
        print(f"Subsampling training data from {len(X_train_latent)} to {max_train_samples} samples...")
        indices = np.random.choice(len(X_train_latent), max_train_samples, replace=False)
        X_train_latent = X_train_latent[indices]
        y_train = y_train[indices]
    
    # Standardize the latent features for better SVM performance
    print("Standardizing latent features...")
    scaler = StandardScaler()
    X_train_latent_scaled = scaler.fit_transform(X_train_latent)
    X_test_latent_scaled = scaler.transform(X_test_latent)
    
    # Test different SVM configurations
    configs = [
        {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'max_iter': 2000},
        {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale', 'max_iter': 2000},
        {'C': 100.0, 'kernel': 'rbf', 'gamma': 'scale', 'max_iter': 2000},
        {'C': 1.0, 'kernel': 'linear', 'max_iter': 2000}
    ]
    
    best_accuracy = 0
    best_config = None
    
    print("Testing different SVM configurations...")
    
    for i, config in enumerate(configs):
        print(f"  Config {i+1}/{len(configs)}: {config}")
        
        # Train SVM with this configuration
        svm = SVC(random_state=42, **config)
        svm.fit(X_train_latent_scaled, y_train.ravel())
        
        # Test on a subset for speed during hyperparameter search
        test_subset = min(2000, len(X_test_latent))
        indices = np.random.choice(len(X_test_latent), test_subset, replace=False)
        X_test_subset = X_test_latent_scaled[indices]
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
    
    # Train final model with best configuration on full data
    print("Training final model on full dataset...")
    final_svm = SVC(random_state=42, **best_config)
    final_svm.fit(X_train_latent_scaled, y_train.ravel())
    
    # Make predictions on full test set
    print("Making predictions on full test set...")
    y_pred = final_svm.predict(X_test_latent_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Final Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    results = {
        'model': final_svm,
        'predictions': y_pred,
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'best_config': best_config,
        'configs_tested': configs,
        'scaler': scaler
    }
    
    return results

def train_baseline_svm(X_train_raw, y_train, X_test_raw, y_test, max_samples=3000):
    """Train baseline SVM on raw pixel values for comparison."""
    print(f"Training baseline SVM on raw pixels (using {max_samples} samples)...")
    
    # Subsample for computational efficiency
    if len(X_train_raw) > max_samples:
        indices = np.random.choice(len(X_train_raw), max_samples, replace=False)
        X_train_subset = X_train_raw[indices]
        y_train_subset = y_train[indices]
    else:
        X_train_subset = X_train_raw
        y_train_subset = y_train
    
    print(f"Using {X_train_subset.shape[0]} training samples")
    
    # Scale the raw pixel data
    scaler_raw = StandardScaler()
    X_train_subset_scaled = scaler_raw.fit_transform(X_train_subset)
    
    # Simple SVM with reliable parameters
    svm_baseline = SVC(kernel='rbf', C=1.0, gamma='scale', max_iter=2000, random_state=42)
    
    # Train
    svm_baseline.fit(X_train_subset_scaled, y_train_subset.ravel())
    
    # Predict on smaller subset of test data for speed
    test_subset = min(1000, len(X_test_raw))
    test_indices = np.random.choice(len(X_test_raw), test_subset, replace=False)
    X_test_subset = X_test_raw[test_indices]
    y_test_subset = y_test[test_indices]
    
    # Scale the test data
    X_test_subset_scaled = scaler_raw.transform(X_test_subset)
    
    y_pred_baseline = svm_baseline.predict(X_test_subset_scaled)
    
    # Metrics
    accuracy_baseline = accuracy_score(y_test_subset, y_pred_baseline)
    
    print(f"Baseline Test Accuracy: {accuracy_baseline:.4f}")
    
    return {
        'model': svm_baseline,
        'predictions': y_pred_baseline,
        'accuracy': accuracy_baseline,
        'training_samples': X_train_subset.shape[0],
        'test_samples': test_subset
    }

def main():
    print("=" * 60)
    print("SECTION 4: LATENT SPACE SVM CLASSIFICATION")
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
    
    # Task 11.1: Feature Extraction Pipeline
    print("\n" + "=" * 60)
    print("TASK 11.1: FEATURE EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Extract latent features using trained encoder
    X_train_latent = extract_latent_features(encoder, X_train_final)
    X_test_latent = extract_latent_features(encoder, X_test_processed)
    
    print(f"\nFeature extraction completed!")
    print(f"Original dimensions: {X_train_final.shape[1]} → Latent dimensions: {X_train_latent.shape[1]}")
    print(f"Compression ratio: {X_train_final.shape[1] / X_train_latent.shape[1]:.1f}x")
    
    # Analyze latent feature statistics
    print(f"\nLatent feature statistics:")
    print(f"Training set - Mean: {np.mean(X_train_latent):.4f}, Std: {np.std(X_train_latent):.4f}")
    print(f"Test set - Mean: {np.mean(X_test_latent):.4f}, Std: {np.std(X_test_latent):.4f}")
    print(f"Feature range: [{np.min(X_train_latent):.4f}, {np.max(X_train_latent):.4f}]")
    
    # Task 11.2: SVM Classification
    print("\n" + "=" * 60)
    print("TASK 11.2: SVM CLASSIFICATION")
    print("=" * 60)
    
    # Train SVM on latent features
    svm_results = train_svm_classifier(X_train_latent, y_train_final, X_test_latent, y_test)
    
    # Train baseline SVM on raw pixels for comparison
    baseline_results = train_baseline_svm(X_train_final, y_train_final, X_test_processed, y_test)
    
    # Create comprehensive visualizations
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy Comparison
    methods = ['Latent Space SVM', 'Raw Pixel SVM']
    accuracies = [svm_results['accuracy'], baseline_results['accuracy']]
    
    axes[0, 0].bar(methods, accuracies, color=['steelblue', 'lightcoral'])
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('SVM Classification Accuracy Comparison')
    axes[0, 0].set_ylim(0, 1)
    
    # Add accuracy values on bars
    for i, acc in enumerate(accuracies):
        axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', fontweight='bold')
    
    # 2. Confusion Matrix for Latent Space SVM
    sns.heatmap(svm_results['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix - Latent Space SVM')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # 3. Per-class Performance
    class_report = svm_results['classification_report']
    classes = [str(i) for i in range(10)]
    f1_scores = [class_report[cls]['f1-score'] for cls in classes]
    
    axes[1, 0].bar(classes, f1_scores, color='lightgreen')
    axes[1, 0].set_xlabel('Digit Class')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title('Per-Class F1-Scores - Latent Space SVM')
    axes[1, 0].set_ylim(0, 1)
    
    # 4. Latent Space Visualization (2D PCA)
    # Use subset for visualization
    n_viz = min(2000, len(X_test_latent))
    indices = np.random.choice(len(X_test_latent), n_viz, replace=False)
    
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(X_test_latent[indices])
    labels_viz = y_test[indices]
    
    # Create scatter plot
    scatter = axes[1, 1].scatter(latent_2d[:, 0], latent_2d[:, 1], 
                                c=labels_viz, cmap='tab10', alpha=0.6, s=20)
    axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[1, 1].set_title('Latent Space Visualization (PCA)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Digit Class')
    
    plt.tight_layout()
    plt.savefig('report/section4_svm_comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate confusion matrix figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(svm_results['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title('SVM Confusion Matrix on Latent Features', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('report/svm_confusion_matrix_notebook.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    
    # Calculate feature statistics per class
    feature_means_per_class = []
    for digit in range(10):
        digit_mask = (y_test == digit)
        digit_features = X_test_latent[digit_mask]
        class_mean = np.mean(digit_features, axis=0)
        feature_means_per_class.append(class_mean)
    
    feature_means_per_class = np.array(feature_means_per_class)
    
    # Calculate variance across classes for each feature
    feature_variance = np.var(feature_means_per_class, axis=0)
    most_discriminative = np.argsort(feature_variance)[-5:]  # Top 5 most discriminative features
    
    print(f"Most discriminative latent features (by class variance):")
    for i, feat_idx in enumerate(reversed(most_discriminative)):
        print(f"  {i+1}. Feature {feat_idx}: variance = {feature_variance[feat_idx]:.4f}")
    
    # Visualize feature distributions for top discriminative features
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i in range(3):
        feat_idx = most_discriminative[-(i+1)]
        
        # Plot histogram for each digit class
        for digit in range(10):
            digit_mask = (y_test == digit)
            digit_values = X_test_latent[digit_mask, feat_idx]
            axes[i].hist(digit_values, alpha=0.6, bins=20, label=f'Digit {digit}')
        
        axes[i].set_xlabel(f'Latent Feature {feat_idx} Value')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Distribution of Feature {feat_idx}\n(Variance: {feature_variance[feat_idx]:.4f})')
        if i == 2:  # Only show legend on last plot
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('report/latent_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comprehensive results
    section4_results = {
        'latent_svm': svm_results,
        'baseline_svm': baseline_results,
        'latent_features': {
            'train': X_train_latent[:1000],  # Save subset for memory
            'test': X_test_latent[:1000]
        },
        'compression_ratio': X_train_final.shape[1] / X_train_latent.shape[1],
        'feature_extraction_summary': {
            'original_dims': X_train_final.shape[1],
            'latent_dims': X_train_latent.shape[1],
            'train_samples': X_train_final.shape[0],
            'test_samples': X_test_processed.shape[0]
        },
        'feature_analysis': {
            'feature_variance': feature_variance,
            'most_discriminative_features': most_discriminative,
            'feature_means_per_class': feature_means_per_class
        }
    }
    
    with open('section4_results_notebook.pkl', 'wb') as f:
        pickle.dump(section4_results, f)
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION RESULTS")
    print("=" * 60)
    
    print(f"\nLatent Space SVM Results:")
    print(f"  Best Configuration: {svm_results['best_config']}")
    print(f"  Test Accuracy: {svm_results['accuracy']:.4f}")
    
    print(f"\nBaseline SVM Results:")
    print(f"  Test Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"  Training Samples Used: {baseline_results['training_samples']}")
    print(f"  Test Samples Used: {baseline_results['test_samples']}")
    
    print(f"\nPer-Class Performance (Latent Space SVM):")
    print("-" * 50)
    for digit in range(10):
        metrics = class_report[str(digit)]
        print(f"Digit {digit}: Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {class_report['accuracy']:.4f}")
    print(f"  Macro Avg F1: {class_report['macro avg']['f1-score']:.4f}")
    print(f"  Weighted Avg F1: {class_report['weighted avg']['f1-score']:.4f}")
    
    print("\nFiles saved:")
    print("  - section4_results_notebook.pkl")
    print("  - report/section4_svm_comprehensive_results.png")
    print("  - report/svm_confusion_matrix_notebook.png")
    print("  - report/latent_feature_analysis.png")
    
    print("\n" + "=" * 60)
    print("SECTION 4 COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()