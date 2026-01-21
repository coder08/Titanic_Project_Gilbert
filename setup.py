"""
Setup script for Titanic Survival Prediction System
Run this script to set up the project structure and verify everything is ready
"""

import os
import sys

def create_directory_structure():
    """Create the required directory structure"""
    print("Creating directory structure...")
    
    directories = [
        'model',
        'templates',
        'static'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created {directory}/ directory")
        else:
            print(f"✓ {directory}/ directory already exists")

def check_required_files():
    """Check if all required files exist"""
    print("\nChecking required files...")
    
    required_files = {
        'app.py': 'Flask application file',
        'requirements.txt': 'Python dependencies',
        'Titanic_hosted_webGUI_link.txt': 'Submission information',
        'README.md': 'Project documentation',
        'templates/index.html': 'Web interface',
        'model/model_building.ipynb': 'Model training notebook'
    }
    
    all_present = True
    for file, description in required_files.items():
        if os.path.exists(file):
            print(f"✓ {file} - {description}")
        else:
            print(f"✗ {file} - {description} [MISSING]")
            all_present = False
    
    return all_present

def check_model_files():
    """Check if model files exist"""
    print("\nChecking model files...")
    
    model_files = [
        'model/titanic_survival_model.pkl',
        'model/scaler.pkl',
        'model/le_sex.pkl',
        'model/le_embarked.pkl'
    ]
    
    all_present = True
    for file in model_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file} ({size:,} bytes)")
        else:
            print(f"✗ {file} [MISSING - Run model training first]")
            all_present = False
    
    return all_present

def verify_dependencies():
    """Verify required Python packages are installed"""
    print("\nVerifying dependencies...")
    
    required_packages = [
        'flask',
        'sklearn',
        'pandas',
        'numpy',
        'joblib'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            all_installed = False
    
    return all_installed

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Train the Model:")
    print("   - Open and run model/model_building.ipynb")
    print("   - Or run: python model/model_building.py")
    print("\n2. Test Locally:")
    print("   - Run: python app.py")
    print("   - Visit: http://127.0.0.1:5000")
    print("\n3. Deploy:")
    print("   - Push to GitHub")
    print("   - Deploy to Render.com, PythonAnywhere, or Streamlit Cloud")
    print("   - See DEPLOYMENT_GUIDE.md for detailed instructions")
    print("\n4. Submit:")
    print("   - Update Titanic_hosted_webGUI_link.txt with your details")
    print("   - Upload to Scorac.com before deadline")
    print("   - Deadline: Friday, February 5, 2026, 11:59 PM")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("="*60)
    print("TITANIC SURVIVAL PREDICTION SYSTEM - SETUP")
    print("="*60)
    
    # Create directories
    create_directory_structure()
    
    # Check files
    files_ok = check_required_files()
    
    # Check model files
    models_ok = check_model_files()
    
    # Verify dependencies
    deps_ok = verify_dependencies()
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    if files_ok and deps_ok:
        print("✓ All required files are present")
        print("✓ All dependencies are installed")
        
        if not models_ok:
            print("⚠ Model files missing - You need to train the model first")
        else:
            print("✓ Model files are present")
            print("\n🎉 Setup complete! You're ready to run the application.")
    else:
        print("\n⚠ Setup incomplete. Please address the missing items above.")
        if not deps_ok:
            print("\nInstall dependencies with: pip install -r requirements.txt")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()