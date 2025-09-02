#!/usr/bin/env python3
"""
IPL T20 Cricket Analysis Dashboard Launcher
This script launches the Streamlit application with proper error handling.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run: pip install -r requirements.txt")
            return False
    
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'matches.csv',
        'deliveries.csv', 
        'points_table.csv',
        'IPL - Winners.csv'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing data files: {', '.join(missing_files)}")
        print("📁 Please ensure all CSV files are in the current directory.")
        return False
    
    print("✅ All data files found!")
    return True

def main():
    """Main launcher function"""
    print("🏏 IPL T20 Cricket Analysis Dashboard")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return
    
    # Check data files
    print("📁 Checking data files...")
    if not check_data_files():
        return
    
    # Launch Streamlit app
    print("🚀 Launching Streamlit application...")
    print("🌐 The app will open in your default browser")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")

if __name__ == "__main__":
    main()
