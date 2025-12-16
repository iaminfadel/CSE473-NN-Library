#!/usr/bin/env python3
"""
Simple script to check if the report compiles and provide a summary.
"""

import os

def check_report_files():
    """Check if all required files exist for the report."""
    required_files = [
        'report/project_report.typ',
        'report/ASU_LOGO.png',
        'report/autoencoder_architecture.png',
        'report/autoencoder_training_curves.png',
        'report/autoencoder_reconstructions.png',
        'report/autoencoder_latent_space.png',
        'report/autoencoder_error_distribution.png',
        'report/xor_architecture.png',
        'report/xor_loss_curve.png',
        'report/xor_decision_boundary.png'
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    print("=== REPORT FILE STATUS ===")
    print(f"Total required files: {len(required_files)}")
    print(f"Existing files: {len(existing_files)}")
    print(f"Missing files: {len(missing_files)}")
    
    if missing_files:
        print("\nMISSING FILES:")
        for file_path in missing_files:
            print(f"  - {file_path}")
    else:
        print("\n✓ All required files are present!")
    
    print("\nEXISTING FILES:")
    for file_path in existing_files:
        print(f"  ✓ {file_path}")
    
    return len(missing_files) == 0

def get_report_summary():
    """Get a summary of the report content."""
    try:
        with open('report/project_report.typ', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count sections
        sections = content.count('= ')
        subsections = content.count('== ')
        subsubsections = content.count('=== ')
        
        # Count figures
        figures = content.count('#figure(')
        
        # Count equations
        equations = content.count('<eq:')
        
        # Estimate word count (rough)
        words = len(content.split())
        
        print("\n=== REPORT CONTENT SUMMARY ===")
        print(f"Sections: {sections}")
        print(f"Subsections: {subsections}")
        print(f"Sub-subsections: {subsubsections}")
        print(f"Figures: {figures}")
        print(f"Equations: {equations}")
        print(f"Estimated word count: {words}")
        
        # Check for key sections
        key_sections = [
            'Introduction',
            'Library Design and Architecture',
            'Gradient Checking Validation',
            'XOR Problem',
            'Autoencoder Implementation',
            'Conclusion'
        ]
        
        print("\nKEY SECTIONS:")
        for section in key_sections:
            if section in content:
                print(f"  ✓ {section}")
            else:
                print(f"  ✗ {section}")
        
    except Exception as e:
        print(f"Error reading report file: {e}")

def main():
    """Main function."""
    print("NEURAL NETWORK LIBRARY REPORT STATUS")
    print("=" * 50)
    
    files_ok = check_report_files()
    get_report_summary()
    
    print("\n" + "=" * 50)
    if files_ok:
        print("✓ REPORT IS READY FOR COMPILATION")
        print("\nTo compile the report:")
        print("1. Install Typst: https://typst.app/")
        print("2. Run: typst compile report/project_report.typ")
    else:
        print("✗ SOME FILES ARE MISSING")
        print("Please ensure all required files are present before compiling.")

if __name__ == "__main__":
    main()