"""
Deployment Helper Script for TF Binding Transformer
Automates the deployment process with checks and validations
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"[ERROR] Python 3.8+ required. Found: {version.major}.{version.minor}")
        return False
    print(f"[OK] Python version: {version.major}.{version.minor}")
    return True


def check_data_files():
    """Check if required data files exist"""
    data_dir = Path("R0315/iData")
    required_files = [
        "efastanotw12.txt",
        "expre12.tab", 
        "factordts.wtmx",
        "factorexpdts2.tab"
    ]
    
    missing = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing.append(file)
    
    if missing:
        print(f"[ERROR] Missing data files: {', '.join(missing)}")
        print(f"        Please ensure all files are in {data_dir}")
        return False
    
    print(f"[OK] All required data files found in {data_dir}")
    return True


def create_virtual_env():
    """Create virtual environment"""
    env_name = "transformer_env"
    
    if Path(env_name).exists():
        print(f"[INFO] Virtual environment '{env_name}' already exists")
        return env_name
    
    print(f"[INFO] Creating virtual environment '{env_name}'...")
    try:
        subprocess.run([sys.executable, "-m", "venv", env_name], check=True)
        print(f"[OK] Virtual environment created")
        return env_name
    except subprocess.CalledProcessError:
        print(f"[ERROR] Failed to create virtual environment")
        return None


def get_pip_command(env_name):
    """Get pip command for the virtual environment"""
    if os.name == 'nt':  # Windows
        return os.path.join(env_name, "Scripts", "pip")
    else:  # Linux/Mac
        return os.path.join(env_name, "bin", "pip")


def install_dependencies(env_name):
    """Install required dependencies"""
    pip_cmd = get_pip_command(env_name)
    
    print("[INFO] Installing dependencies...")
    
    # Upgrade pip first
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"], capture_output=True)
    
    # Install PyTorch
    print("[INFO] Installing PyTorch (this may take a few minutes)...")
    try:
        # Try GPU version first
        result = subprocess.run(
            [pip_cmd, "install", "torch", "torchvision", "torchaudio", 
             "--index-url", "https://download.pytorch.org/whl/cu118"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            # Fall back to CPU version
            print("[INFO] GPU version failed, installing CPU version...")
            subprocess.run(
                [pip_cmd, "install", "torch", "torchvision", "torchaudio",
                 "--index-url", "https://download.pytorch.org/whl/cpu"],
                check=True, capture_output=True
            )
            print("[OK] PyTorch installed (CPU version)")
        else:
            print("[OK] PyTorch installed (GPU version)")
            
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to install PyTorch")
        return False
    
    # Install other dependencies
    print("[INFO] Installing other dependencies...")
    try:
        subprocess.run(
            [pip_cmd, "install", "numpy", "pandas", "scikit-learn", 
             "matplotlib", "tqdm", "wandb"],
            check=True, capture_output=True
        )
        print("[OK] All dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to install dependencies")
        return False


def test_import(env_name):
    """Test if imports work"""
    if os.name == 'nt':  # Windows
        python_cmd = os.path.join(env_name, "Scripts", "python")
    else:  # Linux/Mac
        python_cmd = os.path.join(env_name, "bin", "python")
    
    print("[INFO] Testing imports...")
    
    test_script = '''
import torch
import numpy as np
import pandas as pd
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
'''
    
    try:
        result = subprocess.run(
            [python_cmd, "-c", test_script],
            capture_output=True, text=True, check=True
        )
        print("[OK] Import test passed")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("[ERROR] Import test failed")
        print(e.stderr)
        return False


def create_directories():
    """Create necessary directories"""
    dirs = ["checkpoints", "results", "logs"]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    print(f"[OK] Created directories: {', '.join(dirs)}")


def create_run_script(env_name):
    """Create convenient run scripts"""
    if os.name == 'nt':  # Windows
        python_cmd = os.path.join(env_name, "Scripts", "python")
        
        # Training script
        with open("train.bat", "w") as f:
            f.write(f'''@echo off
echo Starting transformer training...
{python_cmd} main.py --mode train --epochs 50 --batch_size 16
pause
''')
        
        # Prediction script  
        with open("predict.bat", "w") as f:
            f.write(f'''@echo off
echo Running predictions...
{python_cmd} main.py --mode predict --model_path checkpoints/best_model.pt
pause
''')
        
        print("[OK] Created train.bat and predict.bat")
        
    else:  # Linux/Mac
        python_cmd = os.path.join(env_name, "bin", "python")
        
        # Training script
        with open("train.sh", "w") as f:
            f.write(f'''#!/bin/bash
echo "Starting transformer training..."
{python_cmd} main.py --mode train --epochs 50 --batch_size 16
''')
        
        # Prediction script
        with open("predict.sh", "w") as f:
            f.write(f'''#!/bin/bash
echo "Running predictions..."
{python_cmd} main.py --mode predict --model_path checkpoints/best_model.pt
''')
        
        # Make scripts executable
        os.chmod("train.sh", 0o755)
        os.chmod("predict.sh", 0o755)
        
        print("[OK] Created train.sh and predict.sh")


def main():
    """Main deployment function"""
    print_header("TF Binding Transformer Deployment")
    
    # Step 1: Check Python version
    if not check_python_version():
        return
    
    # Step 2: Check data files
    if not check_data_files():
        print("\n[ACTION] Please copy your HMM data files to R0315/iData/")
        return
    
    # Step 3: Create virtual environment
    env_name = create_virtual_env()
    if not env_name:
        return
    
    # Step 4: Install dependencies
    if not install_dependencies(env_name):
        print("\n[ACTION] Try installing dependencies manually:")
        print(f"         1. Activate environment: {env_name}\\Scripts\\activate (Windows)")
        print(f"         2. Install PyTorch from https://pytorch.org")
        return
    
    # Step 5: Test imports
    if not test_import(env_name):
        return
    
    # Step 6: Create directories
    create_directories()
    
    # Step 7: Create run scripts
    create_run_script(env_name)
    
    # Success!
    print_header("Deployment Complete!")
    
    print("\n[NEXT STEPS]")
    print("1. Test data compatibility:")
    if os.name == 'nt':
        print(f"   {env_name}\\Scripts\\python simple_demo.py")
    else:
        print(f"   {env_name}/bin/python simple_demo.py")
    
    print("\n2. Start training:")
    if os.name == 'nt':
        print("   train.bat")
    else:
        print("   ./train.sh")
    
    print("\n3. Run predictions:")
    if os.name == 'nt':
        print("   predict.bat")
    else:
        print("   ./predict.sh")
    
    print("\n[INFO] Virtual environment:", env_name)
    print("[INFO] Checkpoints will be saved to: checkpoints/")
    print("[INFO] Results will be saved to: results/")
    
    print("\nDeployment successful! Your transformer is ready to use.")


if __name__ == "__main__":
    main()