import os
import subprocess
import sys

def setup_virtual_environment():
    # Create virtual environment
    subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
    
    # Determine the activation script and pip path based on OS
    if os.name == 'nt':  # Windows
        activate_script = os.path.join('venv', 'Scripts', 'activate')
        pip_path = os.path.join('venv', 'Scripts', 'pip')
    else:  # Unix-like
        activate_script = os.path.join('venv', 'bin', 'activate')
        pip_path = os.path.join('venv', 'bin', 'pip')
    
    # Install ninja inside the venv FIRST
    subprocess.run([pip_path, 'install', 'ninja'], check=True)
    # Now install required packages
    subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
    
    print("Virtual environment setup complete. Activate it using:")
    print(f"  {activate_script}")

if __name__ == '__main__':
    setup_virtual_environment() 