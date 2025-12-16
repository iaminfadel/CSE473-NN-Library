#!/usr/bin/env python3
"""
Generate all plots and visualizations for the autoencoder section of the report.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_results():
    """Load the autoencoder results."""
    with open('autoencoder_results_final.pkl', 'rb') as f:
        results = pickle.load(f)
    return results

def plot_autoencoder_architecture():
    """Create autoencoder architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Define layer positions and sizes - CORRECTED ARCHITECTURE: 784 → 256 → 128 → 64 → 32
    layers = [
        {'name': 'Input\n784', 'x': 1, 'y': 2, 'width': 1, 'height': 4, 'color': 'lightblue'},
        {'name': 'Hidden 1\n256', 'x': 3, 'y': 2.5, 'width': 1, 'height': 3, 'color': 'lightgreen'},
        {'name': 'Hidden 2\n128', 'x': 5, 'y': 3, 'width': 1, 'height': 2, 'color': 'lightgreen'},
        {'name': 'Hidden 3\n64', 'x': 7, 'y': 3.25, 'width': 1, 'height': 1.5, 'color': 'lightgreen'},
        {'name': 'Latent\n32', 'x': 9, 'y': 3.5, 'width': 1, 'height': 1, 'color': 'orange'},
        {'name': 'Hidden 4\n64', 'x': 11, 'y': 3.25, 'width': 1, 'height': 1.5, 'color': 'lightcoral'},
        {'name': 'Hidden 5\n128', 'x': 13, 'y': 3, 'width': 1, 'height': 2, 'color': 'lightcoral'},
        {'name': 'Hidden 6\n256', 'x': 15, 'y': 2.5, 'width': 1, 'height': 3, 'color': 'lightcoral'},
        {'name': 'Output\n784', 'x': 17, 'y': 2, 'width': 1, 'height': 4, 'color': 'lightblue'},
    ]
    
    # Draw layers
    for layer in layers:
        rect = Rectangle((layer['x'], layer['y']), layer['width'], layer['height'], 
                        facecolor=layer['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(layer['x'] + 0.5, layer['y'] + layer['height']/2, layer['name'], 
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    for i in range(len(layers) - 1):
        start_x = layers[i]['x'] + layers[i]['width']
        end_x = layers[i+1]['x']
        y = layers[i]['y'] + layers[i]['height']/2
        ax.annotate('', xy=(end_x, y), xytext=(start_x, y), arrowprops=arrow_props)
    
    # Add encoder/decoder labels
    ax.text(6, 1, 'Encoder', ha='center', va='center', fontsize=14, fontweight='bold', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    ax.text(14, 1, 'Decoder', ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Autoencoder Architecture (784-256-128-64-32-64-128-256-784)', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('report/autoencoder_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(results):
    """Plot training and validation loss curves."""
    history = results['history']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    epochs = np.arange(1, len(history['train_losses']) + 1)
    val_epochs = np.arange(10, len(history['train_losses']) + 1, 10)  # Every 10 epochs
    
    ax1.plot(epochs, history['train_losses'], label='Training Loss', color='blue', alpha=0.7)
    ax1.plot(val_epochs, history['val_losses'], label='Validation Loss', color='red', marker='o', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Reconstruction quality
    ax2.plot(val_epochs, history['reconstruction_quality'], color='green', marker='s', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Reconstruction Quality (1 - MSE)')
    ax2.set_title('Reconstruction Quality Over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/autoencoder_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_reconstruction_examples(results):
    """Plot original vs reconstructed images."""
    data = results['data']
    autoencoder = results['autoencoder']
    
    # Get test samples
    X_test = data['X_test'][:10]  # First 10 test samples
    
    # Generate reconstructions using reconstruct_probabilities
    reconstructions = []
    for x in X_test:
        x_reshaped = x.reshape(1, -1)
        # Use reconstruct_probabilities if available, otherwise forward
        if hasattr(autoencoder, 'reconstruct_probabilities'):
            reconstruction = autoencoder.reconstruct_probabilities(x_reshaped)
        else:
            reconstruction = autoencoder.forward(x_reshaped)
        reconstructions.append(reconstruction.flatten())
    
    reconstructions = np.array(reconstructions)
    
    # Plot
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    
    for i in range(10):
        # Original
        axes[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstructions[i].reshape(28, 28), cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle('Original vs Reconstructed MNIST Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report/autoencoder_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_latent_space_visualization(results):
    """Visualize the latent space representations."""
    data = results['data']
    encoder = results['encoder']
    
    # Get a subset of test data with labels
    X_test = data['X_test'][:1000]  # First 1000 samples
    y_test = data['y_test'][:1000]
    
    # Encode to latent space
    latent_representations = []
    for x in X_test:
        x_reshaped = x.reshape(1, -1)
        latent = encoder.forward(x_reshaped)
        latent_representations.append(latent.flatten())
    
    latent_representations = np.array(latent_representations)
    
    # Use PCA to reduce to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_representations)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for digit in range(10):
        mask = y_test == digit
        ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                  c=[colors[digit]], label=f'Digit {digit}', alpha=0.6, s=20)
    
    ax.set_xlabel(f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
    ax.set_ylabel(f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
    ax.set_title('Latent Space Visualization (PCA Projection)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/autoencoder_latent_space.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_reconstruction_error_distribution(results):
    """Plot distribution of reconstruction errors."""
    test_metrics = results['test_metrics']
    per_sample_mse = test_metrics['per_sample_mse']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax1.hist(per_sample_mse, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(per_sample_mse), color='red', linestyle='--', 
               label=f'Mean: {np.mean(per_sample_mse):.6f}')
    ax1.axvline(np.median(per_sample_mse), color='green', linestyle='--', 
               label=f'Median: {np.median(per_sample_mse):.6f}')
    ax1.set_xlabel('Per-Sample MSE')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Reconstruction Errors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(per_sample_mse, vert=True)
    ax2.set_ylabel('Per-Sample MSE')
    ax2.set_title('Reconstruction Error Statistics')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/autoencoder_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_metrics_table(results):
    """Generate metrics for the report table."""
    test_metrics = results['test_metrics']
    history = results['history']
    
    metrics = {
        'Initial Training Loss': f"{history['train_losses'][0]:.6f}",
        'Final Training Loss': f"{history['train_losses'][-1]:.6f}",
        'Final Validation Loss': f"{history['val_losses'][-1]:.6f}",
        'Test MSE': f"{test_metrics['mse']:.6f}",
        'Test MAE': f"{test_metrics['mae']:.6f}",
        'Test BCE Loss': f"{test_metrics['bce_loss']:.6f}",
        'Mean Per-Sample MSE': f"{test_metrics['mean_per_sample_mse']:.6f}",
        'Std Per-Sample MSE': f"{test_metrics['std_per_sample_mse']:.6f}",
        'Reconstruction Quality': f"{history['reconstruction_quality'][-1]:.6f}",
        'Range Preservation': f"{test_metrics['range_preservation']:.6f}",
    }
    
    print("=== AUTOENCODER METRICS TABLE ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    return metrics

def main():
    """Generate all plots and metrics."""
    print("Loading autoencoder results...")
    results = load_results()
    
    print("Generating architecture diagram...")
    plot_autoencoder_architecture()
    
    print("Generating training curves...")
    plot_training_curves(results)
    
    print("Generating reconstruction examples...")
    plot_reconstruction_examples(results)
    
    print("Generating latent space visualization...")
    plot_latent_space_visualization(results)
    
    print("Generating error distribution plots...")
    plot_reconstruction_error_distribution(results)
    
    print("Generating metrics table...")
    metrics = generate_metrics_table(results)
    
    print("\nAll plots generated successfully!")
    print("Files created:")
    print("- report/autoencoder_architecture.png")
    print("- report/autoencoder_training_curves.png") 
    print("- report/autoencoder_reconstructions.png")
    print("- report/autoencoder_latent_space.png")
    print("- report/autoencoder_error_distribution.png")

if __name__ == "__main__":
    main()