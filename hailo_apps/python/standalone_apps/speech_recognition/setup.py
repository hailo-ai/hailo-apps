import os
import subprocess
import sys
import argparse

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
VENV_DIR = os.path.join(ROOT_DIR, "whisper_env")
PYTHON_BIN = os.path.join(VENV_DIR, "bin", "python")
PIP_BIN = os.path.join(VENV_DIR, "bin", "pip")
APP_DIR = os.path.join(ROOT_DIR, "app")

def run_command(command, cwd=None):
    """Helper function to run shell commands."""
    subprocess.run(command, shell=True, cwd=cwd, check=True)

def is_raspberry_pi_5():
    """Checks if the host platform is a Raspberry Pi 5."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip().lower()
        return ("raspberry pi 5" in model)
    except FileNotFoundError:
        return False

def create_venv():
    """Creates a virtual environment if it doesn't exist."""
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment in {VENV_DIR}...")
        if is_raspberry_pi_5():  # Raspberry Pi 5 with hailo-all package installed. Reuse existing packages.
            run_command(f"python3 -m venv --system-site-packages {VENV_DIR}")
        else:  # x86 and other cases
            run_command(f"python3 -m venv {VENV_DIR}")
    else:
        print("Virtual environment already exists.")

    # Upgrade pip to the latest version
    print("\nUpgrading pip inside the virtual environment...")
    run_command(f"{PIP_BIN} install --upgrade pip")

def download_resources():
    try:
        run_command(f"python3 ./download_resources.py", cwd=APP_DIR)
        print("Downloading inference files.")
    except subprocess.CalledProcessError:
        print("Inference files download failed.")
    return

def install_requirements(develop_install=False):
    """Installs required Python packages inside the virtual environment."""

    requirements_inference_file = os.path.join(ROOT_DIR, "requirements_inference.txt")
    if os.path.exists(requirements_inference_file):
        print("\nInstalling dependencies from requirements_inference.txt...")
        run_command(f"{PIP_BIN} install -r {requirements_inference_file}")
    else:
        print("No requirements_inference.txt found, skipping package installation.")

    download_resources()

    

def main():
    """Main function to set up the environment."""

    create_venv()
    install_requirements()

    print("\n✅ Setup complete! To activate the environment, run:")
    print(f"source {VENV_DIR}/bin/activate\n")

if __name__ == "__main__":
    main()