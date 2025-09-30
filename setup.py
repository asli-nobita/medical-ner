#!/usr/bin/env python3
"""
Simple starter script for Medical NER Project
This script helps you get started without requiring all dependencies upfront.
"""

import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)


def main():
    """Main execution function."""
    print("🏥 Medical NER Project - Getting Started")
    print("=" * 50)
    
    # Check project structure
    print("\n📁 Checking project structure...")
    check_project_structure()
    
    # Setup directories
    print("\n🚀 Setting up directories...")
    setup_basic_directories()
    
    # Check for datasets
    print("\n📄 Dataset Options:")
    show_dataset_options()
    
    # Show next steps
    print("\n📋 Next Steps:")
    show_next_steps()


def check_project_structure():
    """Check if the project structure exists."""
    required_dirs = [
        'src/data_processing',
        'src/extraction', 
        'src/interface',
        'src/ml_models',
        'src/utils',
        'config'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path}")
            all_exist = False
    
    if all_exist:
        print("✅ Project structure is complete!")
    else:
        print("⚠️  Some directories are missing. Run the setup script.")


def setup_basic_directories():
    """Create basic directory structure."""
    directories = [
        'data/raw',
        'data/processed', 
        'data/sample',
        'outputs/cleaned_images',
        'outputs/ocr_results',
        'outputs/extraction_results',
        'outputs/hitl_results/corrections',
        'outputs/hitl_results/confirmed',
        'models/saved_models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  📁 Created: {directory}")
    
    print("✅ Basic directories created!")


def show_dataset_options():
    """Show available dataset options."""
    print("\n💡 How to specify your dataset:")
    print("\n1. 📁 Local Files:")
    print("   - Place your PDF/PNG/JPG files in: data/raw/")
    print("   - The system will automatically find them")
    
    print("\n2. 🌐 Kaggle Dataset:")
    print("   - Install: pip install kagglehub")
    print("   - The script will download automatically")
    
    print("\n3. 🎯 Custom Path:")
    print("   - You can specify any file path when prompted")
    
    # Check what's currently available
    raw_data_dir = "data/raw"
    if os.path.exists(raw_data_dir):
        files = []
        for ext in ['.pdf', '.png', '.jpg', '.jpeg']:
            for file in os.listdir(raw_data_dir):
                if file.lower().endswith(ext):
                    files.append(file)
        
        if files:
            print(f"\n📄 Found {len(files)} documents in data/raw/:")
            for i, file in enumerate(files, 1):
                print(f"   {i}. {file}")
        else:
            print(f"\n📭 No documents found in {raw_data_dir}")
            print("   💡 Place your medical documents there to get started")


def show_next_steps():
    """Show next steps for the user."""
    print("\n1. 📦 Install Dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n2. 📄 Add Your Documents:")
    print("   - Copy medical documents to data/raw/")
    print("   - Supported formats: PDF, PNG, JPG")
    
    print("\n3. 🚀 Run Full Pipeline:")
    print("   python main.py")
    
    print("\n4. 🔍 Or Run Individual Steps:")
    print("   - Image preprocessing")
    print("   - OCR text extraction")
    print("   - Rule-based extraction")
    print("   - Human-in-the-loop validation")
    print("   - Model training")
    
    print("\n📚 Documentation:")
    print("   Check README.md for detailed usage instructions")


if __name__ == "__main__":
    main()