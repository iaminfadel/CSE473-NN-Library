#!/usr/bin/env python3
"""
Quick check script to see what's available for TensorFlow comparison
and run the comparison with existing results.
"""

import os
import sys

def check_files():
    """Check what files are available for the comparison."""
    print("TENSORFLOW COMPARISON - FILE CHECK")
    print("=" * 50)
    
    files_to_check = [
        'autoencoder_results_final.pkl',
        'autoencoder_results.pkl', 
        'xor_results.pkl',
        'tensorflow_comparison.py',
        'requirements.txt'
    ]
    
    available_files = []
    missing_files = []
    
    for file in files_to_check:
        if os.path.exists(file):
            available_files.append(file)
            size = os.path.getsize(file)
            print(f"✓ {file} ({size:,} bytes)")
        else:
            missing_files.append(file)
            print(f"✗ {file} (missing)")
    
    print(f"\nSummary:")
    print(f"Available: {len(available_files)} files")
    print(f"Missing: {len(missing_files)} files")
    
    return available_files, missing_files

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nDEPENDENCY CHECK")
    print("=" * 30)
    
    dependencies = [
        ('numpy', 'np'),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns'),
        ('pandas', 'pd'),
        ('tensorflow', 'tf'),
        ('pickle', 'pickle')
    ]
    
    available_deps = []
    missing_deps = []
    
    for dep_name, import_name in dependencies:
        try:
            if dep_name == 'tensorflow':
                import tensorflow as tf
                print(f"✓ {dep_name} (version {tf.__version__})")
            elif dep_name == 'matplotlib':
                import matplotlib.pyplot as plt
                print(f"✓ {dep_name}")
            elif dep_name == 'seaborn':
                import seaborn as sns
                print(f"✓ {dep_name}")
            elif dep_name == 'pandas':
                import pandas as pd
                print(f"✓ {dep_name}")
            elif dep_name == 'numpy':
                import numpy as np
                print(f"✓ {dep_name}")
            elif dep_name == 'pickle':
                import pickle
                print(f"✓ {dep_name}")
            available_deps.append(dep_name)
        except ImportError:
            print(f"✗ {dep_name} (not installed)")
            missing_deps.append(dep_name)
    
    return available_deps, missing_deps

def check_custom_library():
    """Check if custom neural network library is available."""
    print("\nCUSTOM LIBRARY CHECK")
    print("=" * 30)
    
    lib_modules = [
        'lib.layers',
        'lib.activations', 
        'lib.losses',
        'lib.network',
        'lib.optimizer'
    ]
    
    available_modules = []
    missing_modules = []
    
    for module in lib_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
            available_modules.append(module)
        except ImportError as e:
            print(f"✗ {module} ({e})")
            missing_modules.append(module)
    
    return available_modules, missing_modules

def run_comparison():
    """Run the TensorFlow comparison if everything is available."""
    print("\nRUNNING TENSORFLOW COMPARISON")
    print("=" * 40)
    
    try:
        from tensorflow_comparison import TensorFlowComparison
        
        comparison = TensorFlowComparison()
        results = comparison.run_full_comparison()
        
        print("\n" + "="*60)
        print("COMPARISON COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"Error running comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to check everything and run comparison."""
    print("TensorFlow Baseline Comparison - Setup Check")
    print("=" * 60)
    
    # Check files
    available_files, missing_files = check_files()
    
    # Check dependencies
    available_deps, missing_deps = check_dependencies()
    
    # Check custom library
    available_modules, missing_modules = check_custom_library()
    
    # Summary
    print("\nOVERALL STATUS")
    print("=" * 30)
    
    can_run = True
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        if 'tensorflow' in missing_deps:
            print("Install TensorFlow: pip install tensorflow")
        can_run = False
    
    if missing_modules:
        print(f"Missing custom library modules: {', '.join(missing_modules)}")
        can_run = False
    
    if 'autoencoder_results_final.pkl' not in available_files:
        print("Warning: autoencoder_results_final.pkl not found")
        print("Comparison will run with XOR only")
    
    if can_run:
        print("✓ Ready to run TensorFlow comparison!")
        
        response = input("\nRun comparison now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            success = run_comparison()
            if success:
                print("\nComparison completed! Check the generated plots and output.")
            else:
                print("\nComparison failed. Check the error messages above.")
        else:
            print("Comparison skipped. Run 'python tensorflow_comparison.py' when ready.")
    else:
        print("✗ Cannot run comparison. Please fix the issues above.")
    
    return can_run

if __name__ == "__main__":
    main()