#!/usr/bin/env python3
"""
Simple test script to verify environment setup
"""

def test_imports():
    """Test if all required packages can be imported"""
    
    packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'plotly',
        'sklearn',
        'scipy',
        'streamlit',
        'watchdog'
    ]
    
    print("🧪 Testing package imports...")
    
    failed = []
    for package in packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Please install missing packages:")
        print("conda activate netflixv1  # if using conda")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n🎉 All packages imported successfully!")
        return True

def test_project_files():
    """Test if required project files exist"""
    import os
    
    print("\n📁 Testing project files...")
    
    required_files = [
        'recomender_v1.py',
        'app.py',
        'requirements.txt',
        'environment.yml'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing.append(file)
    
    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    else:
        print("\n✅ All project files found!")
        return True

def main():
    print("🎬 Netflix Recommender - Environment Test")
    print("=" * 40)
    
    imports_ok = test_imports()
    files_ok = test_project_files()
    
    print("\n" + "=" * 40)
    if imports_ok and files_ok:
        print("🚀 Environment is ready!")
        print("Run: streamlit run app.py")
    else:
        print("🔧 Please fix the issues above before running the app")

if __name__ == "__main__":
    main()