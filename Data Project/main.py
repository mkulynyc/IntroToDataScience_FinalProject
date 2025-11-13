"""
Netflix Analysis Project - Main Orchestrator
============================================

This is the main entry point for the Netflix content analysis project.
It provides a menu-driven interface to run all analysis modules.

Author: Netflix Analysis Project
Date: October 2025
"""

import os
import sys
from typing import Optional

def print_banner():
    """Print project banner"""
    print("\n" + "="*80)
    print("🎬 NETFLIX CONTENT ANALYSIS PROJECT")
    print("="*80)
    print("Comprehensive analysis of Netflix movies and TV shows")
    print("Author: Netflix Analysis Project | Date: October 2025")
    print("="*80)

def print_menu():
    """Print main menu options"""
    print("\n📋 ANALYSIS MODULES:")
    print("-" * 50)
    print("1️⃣  Data Cleaning - Clean and prepare Netflix dataset")
    print("2️⃣  Data Analysis - Exploratory data analysis and insights")
    print("3️⃣  Visualizations - Generate static charts and plots")
    print("4️⃣  Interactive Dashboard - Launch Streamlit dashboard")
    print("5️⃣  Machine Learning - Content similarity and clustering")
    print("6️⃣  Network Analysis - Collaboration networks")
    print("7️⃣  Run All Modules - Execute complete analysis pipeline")
    print("8️⃣  Install Dependencies - Install required packages")
    print("0️⃣  Exit")
    print("-" * 50)

def check_file_exists(filename: str) -> bool:
    """Check if a file exists"""
    return os.path.exists(filename)

def run_module(module_name: str, description: str) -> bool:
    """
    Run a specific analysis module
    
    Args:
        module_name (str): Name of the Python module to run
        description (str): Description of the module
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n🚀 Running {description}...")
    print("-" * 60)
    
    if not check_file_exists(module_name):
        print(f"❌ Module not found: {module_name}")
        return False
    
    try:
        # Import and run the module
        if module_name == "1_clean_data.py":
            from importlib import import_module
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("clean_data", module_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'main'):
                module.main()
            
        elif module_name == "2_data_analysis.py":
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("data_analysis", module_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'main'):
                module.main()
                
        elif module_name == "3_visualization.py":
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("visualization", module_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'main'):
                module.main()
                
        elif module_name == "4_dashboard.py":
            print("🌐 Launching Streamlit dashboard...")
            print("📝 Note: Dashboard will open in your web browser")
            print("🔗 URL: http://localhost:8501")
            print("⏹️  Press Ctrl+C to stop the dashboard")
            
            os.system(f"streamlit run {module_name}")
            
        elif module_name == "5_machine_learning.py":
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("machine_learning", module_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'main'):
                module.main()
                
        elif module_name == "6_network_analysis.py":
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("network_analysis", module_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'main'):
                module.main()
        
        print(f"\n✅ {description} completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error running {description}: {e}")
        print("💡 Make sure all dependencies are installed (option 8)")
        return False

def run_all_modules():
    """Run all analysis modules in sequence"""
    print("\n🎯 RUNNING COMPLETE ANALYSIS PIPELINE")
    print("="*60)
    
    modules = [
        ("1_clean_data.py", "Data Cleaning"),
        ("2_data_analysis.py", "Data Analysis"),
        ("3_visualization.py", "Visualizations"),
        ("5_machine_learning.py", "Machine Learning"),
        ("6_network_analysis.py", "Network Analysis")
    ]
    
    results = []
    for module_name, description in modules:
        success = run_module(module_name, description)
        results.append((description, success))
        
        if not success:
            print(f"\n⚠️  Pipeline stopped due to error in {description}")
            break
        
        print("\n" + "="*40)
    
    # Summary
    print(f"\n📊 PIPELINE EXECUTION SUMMARY")
    print("-" * 40)
    for description, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{description:<20} {status}")
    
    print(f"\n🎉 Analysis pipeline completed!")
    print("💡 Run option 4 to launch the interactive dashboard")

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 INSTALLING DEPENDENCIES")
    print("-" * 40)
    
    packages = [
        "pandas",
        "numpy", 
        "matplotlib",
        "seaborn",
        "plotly",
        "streamlit",
        "scikit-learn",
        "networkx",
        "wordcloud"
    ]
    
    print("Installing packages:")
    for package in packages:
        print(f"  • {package}")
    
    try:
        import subprocess
        
        # Update pip first
        print("\n🔧 Updating pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install packages
        print("\n📥 Installing packages...")
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print(f"\n✅ All dependencies installed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        print("💡 Try running manually: pip install pandas numpy matplotlib seaborn plotly streamlit scikit-learn networkx wordcloud")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def show_project_status():
    """Show current project status"""
    print("\n📋 PROJECT STATUS")
    print("-" * 40)
    
    files_to_check = [
        ("netflix_titles.csv", "Original Netflix dataset"),
        ("1_clean_data.py", "Data cleaning module"),
        ("2_data_analysis.py", "Data analysis module"), 
        ("3_visualization.py", "Visualization module"),
        ("4_dashboard.py", "Interactive dashboard"),
        ("5_machine_learning.py", "Machine learning module"),
        ("6_network_analysis.py", "Network analysis module"),
        ("data/netflix_cleaned.csv", "Cleaned dataset"),
        ("visualizations/", "Visualization outputs"),
        ("models/", "ML models"),
        ("network_data/", "Network analysis outputs")
    ]
    
    for filename, description in files_to_check:
        exists = "✅" if check_file_exists(filename) else "❌"
        print(f"{exists} {description:<25} {filename}")
    
    print("\n💡 Tips:")
    print("  • Run option 1 first to clean the data")
    print("  • Install dependencies (option 8) if you see import errors")
    print("  • Check output folders after running analyses")

def main():
    """Main application loop"""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("\n🎯 Select an option (0-8): ").strip()
            
            if choice == "0":
                print("\n👋 Thank you for using Netflix Analysis Project!")
                print("🎬 Happy analyzing!")
                break
                
            elif choice == "1":
                run_module("1_clean_data.py", "Data Cleaning")
                
            elif choice == "2":
                run_module("2_data_analysis.py", "Data Analysis")
                
            elif choice == "3":
                run_module("3_visualization.py", "Static Visualizations")
                
            elif choice == "4":
                run_module("4_dashboard.py", "Interactive Dashboard")
                
            elif choice == "5":
                run_module("5_machine_learning.py", "Machine Learning Analysis")
                
            elif choice == "6":
                run_module("6_network_analysis.py", "Network Analysis")
                
            elif choice == "7":
                run_all_modules()
                
            elif choice == "8":
                install_dependencies()
                
            elif choice.lower() == "status":
                show_project_status()
                
            else:
                print("\n❌ Invalid option. Please select 0-8.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            
        # Pause before showing menu again
        input("\n⏸️  Press Enter to continue...")

if __name__ == "__main__":
    main()