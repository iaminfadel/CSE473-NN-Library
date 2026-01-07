#!/usr/bin/env python3
"""
Enhanced Comprehensive Library Comparison with Stress Testing

This script creates an extensive comparison with actual autoencoder benchmarking,
stress testing, and detailed analysis using enhanced styling.
"""

import sys
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import psutil
import gc
from datetime import datetime

# Set plot styles as requested
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Add lib directory to path
sys.path.append('lib')

# Import custom library components
from lib.network import Sequential
from lib.layers import Dense
from lib.activations import ReLU, Sigmoid, Tanh
from lib.losses import MSELoss
from lib.optimizer import SGD
from lib.autoencoder import create_autoencoder

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.datasets import mnist
    TENSORFLOW_AVAILABLE = True
    print("✓ TensorFlow available")
    tf.get_logger().setLevel('ERROR')  # Suppress warnings
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠ TensorFlow not available - skipping TensorFlow comparisons")

# Set random seeds for reproducibility
np.random.seed(42)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(42)

# Create report directory
REPORT_DIR = Path("report")
REPORT_DIR.mkdir(exist_ok=True)

print("Enhanced Comprehensive Library Comparison with Stress Testing")
print("=" * 70)

class PerformanceMonitor:
    """Monitor system performance during benchmarks."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        
    def get_metrics(self):
        """Get performance metrics."""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            'duration': end_time - self.start_time,
            'memory_start': self.start_memory,
            'memory_end': end_memory,
            'memory_peak': self.peak_memory,
            'memory_delta': end_memory - self.start_memory
        }


def benchmark_autoencoder_implementations():
    """Comprehensive autoencoder benchmarking with actual training."""
    print("\n" + "="*70)
    print("COMPREHENSIVE AUTOENCODER BENCHMARKING")
    print("="*70)
    
    # Load MNIST data
    if TENSORFLOW_AVAILABLE:
        (X_train_full, _), (X_test_full, _) = mnist.load_data()
        X_train_full = X_train_full.astype('float32') / 255.0
        X_test_full = X_test_full.astype('float32') / 255.0
        X_train_flat = X_train_full.reshape(X_train_full.shape[0], -1)
        X_test_flat = X_test_full.reshape(X_test_full.shape[0], -1)
    else:
        # Generate dummy data
        X_train_flat = np.random.rand(5000, 784).astype(np.float32)
        X_test_flat = np.random.rand(1000, 784).astype(np.float32)
    
    # Use subset for benchmarking
    X_train = X_train_flat[:2000]
    X_test = X_test_flat[:500]
    
    print(f"Dataset: Train {X_train.shape}, Test {X_test.shape}")
    
    results = {}
    
    # 1. Custom Library Autoencoder
    print("\n1. Benchmarking Custom Library Autoencoder")
    print("-" * 50)
    
    try:
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Create and train custom autoencoder
        autoencoder = create_autoencoder(latent_dim=32)
        
        history = autoencoder.train(
            X_train, X_test[:100],  # Use small validation set
            epochs=15,  # Reduced for faster benchmarking
            learning_rate=0.01,
            batch_size=32,
            print_interval=5
        )
        
        monitor.update_peak_memory()
        metrics = monitor.get_metrics()
        
        # Evaluate final performance
        test_predictions = autoencoder.reconstruct(X_test[:100])
        final_mse = np.mean((X_test[:100] - test_predictions) ** 2)
        
        results['custom'] = {
            'losses': history.get('train_losses', []),
            'final_loss': history.get('train_losses', [0])[-1] if history.get('train_losses') else 0,
            'final_mse': final_mse,
            'duration': metrics['duration'],
            'memory_peak': metrics['memory_peak'],
            'memory_delta': metrics['memory_delta']
        }
        
        print(f"  ✓ Training completed in {results['custom']['duration']:.2f}s")
        print(f"  ✓ Final loss: {results['custom']['final_loss']:.6f}")
        print(f"  ✓ Final MSE: {results['custom']['final_mse']:.6f}")
        print(f"  ✓ Peak memory: {results['custom']['memory_peak']:.1f}MB")
        
    except Exception as e:
        print(f"  ❌ Custom autoencoder failed: {e}")
        results['custom'] = None
    
    # 2. TensorFlow Autoencoder
    if TENSORFLOW_AVAILABLE:
        print("\n2. Benchmarking TensorFlow Autoencoder")
        print("-" * 50)
        
        try:
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            # Create TensorFlow autoencoder with identical architecture
            model = models.Sequential([
                # Encoder
                layers.Dense(256, activation='relu', input_shape=(784,)),
                layers.Dense(128, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation=None),  # Latent space
                
                # Decoder
                layers.Dense(64, activation='relu'),
                layers.Dense(128, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(784, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=optimizers.SGD(learning_rate=0.01),
                loss='mse',
                metrics=['mse']
            )
            
            # Train the model
            history = model.fit(
                X_train, X_train,  # Autoencoder: input = target
                epochs=15,
                batch_size=32,
                validation_data=(X_test[:100], X_test[:100]),
                verbose=0
            )
            
            monitor.update_peak_memory()
            metrics = monitor.get_metrics()
            
            # Evaluate final performance
            test_predictions = model.predict(X_test[:100], verbose=0)
            final_mse = np.mean((X_test[:100] - test_predictions) ** 2)
            
            results['tensorflow'] = {
                'losses': history.history['loss'],
                'final_loss': history.history['loss'][-1],
                'final_mse': final_mse,
                'duration': metrics['duration'],
                'memory_peak': metrics['memory_peak'],
                'memory_delta': metrics['memory_delta']
            }
            
            print(f"  ✓ Training completed in {results['tensorflow']['duration']:.2f}s")
            print(f"  ✓ Final loss: {results['tensorflow']['final_loss']:.6f}")
            print(f"  ✓ Final MSE: {results['tensorflow']['final_mse']:.6f}")
            print(f"  ✓ Peak memory: {results['tensorflow']['memory_peak']:.1f}MB")
            
        except Exception as e:
            print(f"  ❌ TensorFlow autoencoder failed: {e}")
            results['tensorflow'] = None
    
    return results


def run_stress_tests():
    """Run comprehensive stress tests on different data sizes."""
    print("\n" + "="*70)
    print("COMPREHENSIVE STRESS TESTING")
    print("="*70)
    
    stress_results = {}
    
    # Test different data sizes
    test_sizes = [500, 1000, 2000, 3000]
    
    print("\n🔬 Custom Library Stress Test")
    print("-" * 40)
    
    custom_stress = []
    
    for size in test_sizes:
        print(f"  Testing with {size} samples...")
        
        try:
            # Generate test data
            X = np.random.rand(size, 784).astype(np.float32)
            
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            # Create and train autoencoder
            autoencoder = create_autoencoder(latent_dim=16)  # Smaller latent dim for speed
            
            history = autoencoder.train(
                X, X[:100],
                epochs=5,  # Reduced epochs for stress test
                learning_rate=0.01,
                batch_size=32,
                print_interval=10
            )
            
            monitor.update_peak_memory()
            metrics = monitor.get_metrics()
            
            custom_stress.append({
                'size': size,
                'duration': metrics['duration'],
                'memory_peak': metrics['memory_peak'],
                'memory_delta': metrics['memory_delta'],
                'final_loss': history.get('train_losses', [0])[-1] if history.get('train_losses') else 0,
                'success': True
            })
            
            print(f"    ✓ {size} samples: {metrics['duration']:.2f}s, Peak: {metrics['memory_peak']:.1f}MB")
            
        except Exception as e:
            print(f"    ❌ {size} samples: Failed - {str(e)}")
            custom_stress.append({
                'size': size,
                'duration': 0,
                'memory_peak': 0,
                'memory_delta': 0,
                'final_loss': 'Failed',
                'success': False
            })
        
        # Clean up memory
        gc.collect()
    
    stress_results['custom'] = custom_stress
    
    # TensorFlow stress test
    if TENSORFLOW_AVAILABLE:
        print("\n🔬 TensorFlow Stress Test")
        print("-" * 40)
        
        tf_stress = []
        
        for size in test_sizes:
            print(f"  Testing with {size} samples...")
            
            try:
                # Generate test data
                X = np.random.rand(size, 784).astype(np.float32)
                
                monitor = PerformanceMonitor()
                monitor.start_monitoring()
                
                # Create TensorFlow model
                model = models.Sequential([
                    layers.Dense(128, activation='relu', input_shape=(784,)),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(16, activation=None),  # Latent space
                    layers.Dense(64, activation='relu'),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(784, activation='sigmoid')
                ])
                
                model.compile(
                    optimizer=optimizers.SGD(learning_rate=0.01),
                    loss='mse'
                )
                
                # Train the model
                history = model.fit(
                    X, X,
                    epochs=5,
                    batch_size=32,
                    verbose=0
                )
                
                monitor.update_peak_memory()
                metrics = monitor.get_metrics()
                
                tf_stress.append({
                    'size': size,
                    'duration': metrics['duration'],
                    'memory_peak': metrics['memory_peak'],
                    'memory_delta': metrics['memory_delta'],
                    'final_loss': history.history['loss'][-1],
                    'success': True
                })
                
                print(f"    ✓ {size} samples: {metrics['duration']:.2f}s, Peak: {metrics['memory_peak']:.1f}MB")
                
            except Exception as e:
                print(f"    ❌ {size} samples: Failed - {str(e)}")
                tf_stress.append({
                    'size': size,
                    'duration': 0,
                    'memory_peak': 0,
                    'memory_delta': 0,
                    'final_loss': 'Failed',
                    'success': False
                })
            
            # Clean up memory
            gc.collect()
        
        stress_results['tensorflow'] = tf_stress
    
    return stress_results
def create_enhanced_visualizations(autoencoder_results, stress_results):
    """Create comprehensive visualizations with enhanced styling."""
    print("\n" + "="*70)
    print("CREATING ENHANCED VISUALIZATIONS")
    print("="*70)
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Autoencoder Performance Comparison
    plt.subplot(3, 4, 1)
    if autoencoder_results:
        implementations = []
        durations = []
        colors = sns.color_palette("husl", len(autoencoder_results))
        
        for i, (impl, results) in enumerate(autoencoder_results.items()):
            if results:
                implementations.append(impl.title())
                durations.append(results['duration'])
        
        if implementations:
            bars = plt.bar(implementations, durations, color=colors, alpha=0.8)
            plt.ylabel('Training Time (seconds)')
            plt.title('Autoencoder Training Speed', fontweight='bold')
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar, duration in zip(bars, durations):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{duration:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 2. Autoencoder Loss Comparison
    plt.subplot(3, 4, 2)
    if autoencoder_results:
        colors = sns.color_palette("husl", len(autoencoder_results))
        for i, (impl, results) in enumerate(autoencoder_results.items()):
            if results and 'losses' in results and results['losses']:
                plt.plot(results['losses'], label=impl.title(), 
                        linewidth=3, color=colors[i])
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training Curves', fontweight='bold')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    # 3. Memory Usage Comparison
    plt.subplot(3, 4, 3)
    if autoencoder_results:
        implementations = []
        memory_usage = []
        
        for impl, results in autoencoder_results.items():
            if results:
                implementations.append(impl.title())
                memory_usage.append(results['memory_peak'])
        
        if implementations:
            bars = plt.bar(implementations, memory_usage, 
                          color=sns.color_palette("viridis", len(implementations)), alpha=0.8)
            plt.ylabel('Peak Memory Usage (MB)')
            plt.title('Memory Usage Comparison', fontweight='bold')
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar, memory in zip(bars, memory_usage):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{memory:.0f}MB', ha='center', va='bottom', fontweight='bold')
    
    # 4. Scalability Test Results - Duration
    plt.subplot(3, 4, 4)
    for impl_name, results in stress_results.items():
        successful_results = [r for r in results if r['success']]
        if successful_results:
            sizes = [r['size'] for r in successful_results]
            durations = [r['duration'] for r in successful_results]
            
            plt.plot(sizes, durations, 'o-', label=impl_name.title(), 
                    linewidth=3, markersize=8)
    
    plt.xlabel('Dataset Size')
    plt.ylabel('Training Time (seconds)')
    plt.title('Scalability: Training Time', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Scalability Test Results - Memory
    plt.subplot(3, 4, 5)
    for impl_name, results in stress_results.items():
        successful_results = [r for r in results if r['success']]
        if successful_results:
            sizes = [r['size'] for r in successful_results]
            memory_peaks = [r['memory_peak'] for r in successful_results]
            
            plt.plot(sizes, memory_peaks, 's-', label=impl_name.title(), 
                    linewidth=3, markersize=8)
    
    plt.xlabel('Dataset Size')
    plt.ylabel('Peak Memory Usage (MB)')
    plt.title('Scalability: Memory Usage', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Performance Efficiency (Loss/Time)
    plt.subplot(3, 4, 6)
    if autoencoder_results:
        implementations = []
        efficiency = []
        
        for impl, results in autoencoder_results.items():
            if results and results['duration'] > 0 and results['losses']:
                implementations.append(impl.title())
                # Calculate loss reduction per second
                initial_loss = results['losses'][0]
                final_loss = results['final_loss']
                loss_reduction = initial_loss - final_loss
                eff = loss_reduction / results['duration']
                efficiency.append(eff)
        
        if implementations:
            bars = plt.bar(implementations, efficiency, 
                          color=sns.color_palette("plasma", len(implementations)), alpha=0.8)
            plt.ylabel('Loss Reduction per Second')
            plt.title('Training Efficiency', fontweight='bold')
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar, eff in zip(bars, efficiency):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{eff:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Reconstruction Quality
    plt.subplot(3, 4, 7)
    if autoencoder_results:
        implementations = []
        mse_scores = []
        
        for impl, results in autoencoder_results.items():
            if results and 'final_mse' in results:
                implementations.append(impl.title())
                mse_scores.append(results['final_mse'])
        
        if implementations:
            bars = plt.bar(implementations, mse_scores, 
                          color=sns.color_palette("coolwarm", len(implementations)), alpha=0.8)
            plt.ylabel('Final MSE')
            plt.title('Reconstruction Quality\n(Lower is Better)', fontweight='bold')
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar, mse in zip(bars, mse_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{mse:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Success Rate Analysis
    plt.subplot(3, 4, 8)
    success_rates = []
    impl_names = []
    
    for impl_name, results in stress_results.items():
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        success_rate = successful / total * 100
        
        success_rates.append(success_rate)
        impl_names.append(impl_name.title())
    
    if success_rates:
        bars = plt.bar(impl_names, success_rates, 
                      color=sns.color_palette("Set2", len(impl_names)), alpha=0.8)
        plt.ylabel('Success Rate (%)')
        plt.title('Stress Test Success Rate', fontweight='bold')
        plt.ylim(0, 105)
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, rate in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # 9. Performance Heatmap
    plt.subplot(3, 4, 9)
    
    # Create performance matrix
    metrics = ['Speed', 'Memory\nEfficiency', 'Scalability', 'Quality']
    implementations = []
    performance_data = []
    
    if autoencoder_results:
        for impl, results in autoencoder_results.items():
            if results:
                implementations.append(impl.title())
                
                # Normalize metrics (0-1 scale, higher is better)
                speed_score = 1.0 / (results['duration'] / 60)  # Normalize by minute
                memory_score = 1.0 / (results['memory_peak'] / 1000)  # Normalize by GB
                quality_score = 1.0 / (results['final_mse'] * 100)  # Invert MSE
                
                # Get scalability score from stress tests
                scalability_score = 0.5  # Default
                if impl in stress_results:
                    successful = sum(1 for r in stress_results[impl] if r['success'])
                    scalability_score = successful / len(stress_results[impl])
                
                performance_data.append([
                    min(speed_score, 1.0),
                    min(memory_score, 1.0),
                    scalability_score,
                    min(quality_score, 1.0)
                ])
    
    if performance_data:
        performance_matrix = np.array(performance_data).T
        
        sns.heatmap(performance_matrix, 
                    xticklabels=implementations,
                    yticklabels=metrics,
                    annot=True, 
                    cmap='RdYlGn',
                    vmin=0, vmax=1,
                    cbar_kws={'label': 'Performance Score'},
                    fmt='.2f')
        plt.title('Performance Summary Heatmap', fontweight='bold')
    
    # 10. Memory Delta Analysis
    plt.subplot(3, 4, 10)
    if autoencoder_results:
        implementations = []
        memory_deltas = []
        
        for impl, results in autoencoder_results.items():
            if results:
                implementations.append(impl.title())
                memory_deltas.append(results.get('memory_delta', 0))
        
        if implementations:
            bars = plt.bar(implementations, memory_deltas, 
                          color=sns.color_palette("magma", len(implementations)), alpha=0.8)
            plt.ylabel('Memory Delta (MB)')
            plt.title('Memory Growth During Training', fontweight='bold')
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar, delta in zip(bars, memory_deltas):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{delta:.1f}MB', ha='center', va='bottom', fontweight='bold')
    
    # 11. Loss Convergence Rate
    plt.subplot(3, 4, 11)
    if autoencoder_results:
        for impl, results in autoencoder_results.items():
            if results and 'losses' in results and len(results['losses']) > 1:
                losses = results['losses']
                # Calculate convergence rate (loss reduction per epoch)
                convergence_rates = []
                for i in range(1, len(losses)):
                    rate = (losses[i-1] - losses[i]) / losses[i-1] if losses[i-1] > 0 else 0
                    convergence_rates.append(rate)
                
                epochs = range(1, len(convergence_rates) + 1)
                plt.plot(epochs, convergence_rates, label=impl.title(), linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss Reduction Rate')
        plt.title('Convergence Rate Analysis', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 12. Summary Statistics
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    # Create summary text
    summary_text = "ENHANCED ANALYSIS SUMMARY\n"
    summary_text += "=" * 30 + "\n\n"
    
    if autoencoder_results:
        summary_text += "Autoencoder Benchmarks:\n"
        for impl, results in autoencoder_results.items():
            if results:
                summary_text += f"• {impl.title()}:\n"
                summary_text += f"  Time: {results['duration']:.1f}s\n"
                summary_text += f"  Loss: {results['final_loss']:.4f}\n"
                summary_text += f"  MSE: {results['final_mse']:.4f}\n"
                summary_text += f"  Memory: {results['memory_peak']:.0f}MB\n\n"
    
    summary_text += "Stress Test Results:\n"
    for impl_name, results in stress_results.items():
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        summary_text += f"• {impl_name.title()}: {successful}/{total} passed\n"
    
    summary_text += f"\nAnalysis completed:\n{datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the enhanced plot
    plot_path = REPORT_DIR / "enhanced_comprehensive_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved enhanced analysis plots to {plot_path}")
    
    plt.show()


def create_detailed_report(autoencoder_results, stress_results):
    """Create detailed analysis report with comprehensive metrics."""
    print("\n" + "="*70)
    print("CREATING DETAILED ANALYSIS REPORT")
    print("="*70)
    
    # Create comprehensive summary data
    summary_data = []
    
    # Autoencoder results
    if autoencoder_results:
        for impl, results in autoencoder_results.items():
            if results:
                summary_data.append({
                    'Implementation': impl.title(),
                    'Task': 'Autoencoder Training',
                    'Training Time (s)': f"{results['duration']:.2f}",
                    'Final Loss': f"{results['final_loss']:.6f}",
                    'Final MSE': f"{results.get('final_mse', 0):.6f}",
                    'Peak Memory (MB)': f"{results['memory_peak']:.1f}",
                    'Memory Delta (MB)': f"{results.get('memory_delta', 0):.1f}",
                    'Efficiency (Loss/s)': f"{(results['losses'][0] - results['final_loss']) / results['duration']:.6f}" if results['losses'] else 'N/A'
                })
    
    # Stress test summary
    for impl_name, results in stress_results.items():
        successful_tests = [r for r in results if r['success']]
        if successful_tests:
            avg_time = np.mean([r['duration'] for r in successful_tests])
            avg_memory = np.mean([r['memory_peak'] for r in successful_tests])
            max_size = max([r['size'] for r in successful_tests])
            success_rate = len(successful_tests) / len(results) * 100
            
            summary_data.append({
                'Implementation': impl_name.title(),
                'Task': 'Stress Test',
                'Avg Time (s)': f"{avg_time:.2f}",
                'Max Size Tested': str(max_size),
                'Success Rate (%)': f"{success_rate:.0f}",
                'Avg Memory (MB)': f"{avg_memory:.1f}",
                'Scalability Score': f"{success_rate/100:.2f}"
            })
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    print("\nDETAILED ANALYSIS SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    
    # Save results
    summary_path = REPORT_DIR / "enhanced_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Save comprehensive results
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'autoencoder_benchmarks': autoencoder_results,
        'stress_test_results': stress_results,
        'summary_table': summary_df.to_dict(),
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'python_version': sys.version,
            'tensorflow_available': TENSORFLOW_AVAILABLE
        }
    }
    
    results_path = REPORT_DIR / "enhanced_comprehensive_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(report_data, f)
    
    print(f"\n✓ Saved detailed summary to {summary_path}")
    print(f"✓ Saved comprehensive results to {results_path}")
    
    return summary_df, report_data


def main():
    """Main enhanced comparison function."""
    print("Starting Enhanced Comprehensive Analysis with Stress Testing...")
    
    # 1. Run comprehensive autoencoder benchmarking
    autoencoder_results = benchmark_autoencoder_implementations()
    
    # 2. Run stress tests
    stress_results = run_stress_tests()
    
    # 3. Create enhanced visualizations
    create_enhanced_visualizations(autoencoder_results, stress_results)
    
    # 4. Create detailed report
    summary_df, report_data = create_detailed_report(autoencoder_results, stress_results)
    
    # 5. Final summary
    print("\n" + "="*70)
    print("ENHANCED ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*70)
    
    print("Key Findings from Enhanced Analysis:")
    print("1. 🔬 Comprehensive autoencoder benchmarking completed")
    print("2. 📊 Stress testing reveals performance limits and scalability")
    print("3. 💾 Memory usage patterns thoroughly analyzed")
    print("4. 🎯 Training efficiency and convergence rates measured")
    print("5. 📈 Performance trade-offs quantified across implementations")
    
    if autoencoder_results:
        print("\nPerformance Highlights:")
        for impl, results in autoencoder_results.items():
            if results:
                print(f"• {impl.title()}: {results['duration']:.1f}s training, "
                      f"{results['final_mse']:.4f} MSE, {results['memory_peak']:.0f}MB peak")
    
    print(f"\nAll enhanced outputs saved to: {REPORT_DIR.absolute()}")
    print("Enhanced files generated:")
    print("- enhanced_comprehensive_analysis.png")
    print("- enhanced_analysis_summary.csv")
    print("- enhanced_comprehensive_results.pkl")
    
    return report_data


if __name__ == "__main__":
    main()