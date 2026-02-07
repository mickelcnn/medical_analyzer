#!/usr/bin/env python3
"""
Cross-Platform Setup Script for Advanced Object Detection & Person Recognition
Works on Windows, Linux, and macOS
"""

import os
import sys
import platform
import subprocess
import urllib.request
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def print_step(step, total, text):
    """Print a step indicator"""
    print(f"\n[{step}/{total}] {text}")

def print_success(text):
    """Print success message"""
    print(f"✓ {text}")

def print_error(text):
    """Print error message"""
    print(f"✗ {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠ {text}")

def check_python_version():
    """Check if Python version is 3.7+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print_error(f"Python 3.7+ required. You have {version.major}.{version.minor}")
        sys.exit(1)
    print_success(f"Python {version.major}.{version.minor}.{version.micro} found")

def run_command(cmd, description="", check=True):
    """Run a shell command"""
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, check=check, 
                                   capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=check, 
                                   capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        if description:
            print_warning(f"{description} failed: {e}")
        return False

def install_pip_package(package, description=""):
    """Install a pip package"""
    print(f"  Installing {package}...")
    cmd = [sys.executable, "-m", "pip", "install", package]
    if run_command(cmd, f"Installing {package}"):
        print_success(f"{description or package} installed")
        return True
    else:
        print_warning(f"{description or package} installation failed")
        return False

def download_file(url, output_path):
    """Download a file from URL"""
    try:
        print(f"  Downloading {os.path.basename(output_path)}...")
        urllib.request.urlretrieve(url, output_path)
        print_success(f"Downloaded {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print_error(f"Download failed: {e}")
        return False

def create_directory(path):
    """Create a directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def main():
    print_header("Advanced Object Detection & Person Recognition Setup")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Step 1: Check Python version
    print_step(1, 7, "Checking Python version...")
    check_python_version()
    
    # Step 2: Upgrade pip
    print_step(2, 7, "Upgrading pip...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                "Upgrading pip", check=False)
    
    # Step 3: Install basic packages
    print_step(3, 7, "Installing basic packages...")
    install_pip_package("numpy", "NumPy")
    install_pip_package("opencv-python", "OpenCV")
    install_pip_package("opencv-contrib-python", "OpenCV Contrib")
    
    # Step 4: Install face recognition (optional)
    print_step(4, 7, "Installing face recognition library...")
    print("This provides 99.38% accuracy facial recognition")
    print("Note: May take 10-15 minutes on some systems...")
    
    # Try to install dlib and face_recognition
    is_windows = platform.system() == "Windows"
    
    if is_windows:
        print("\nWindows detected. Checking for Visual C++ Build Tools...")
        print("If installation fails, you have two options:")
        print("1. Install Visual Studio Build Tools")
        print("2. Use pre-built wheels (fallback)")
    
    # Try installing dlib
    dlib_installed = install_pip_package("dlib", "dlib (face detection library)")
    
    if dlib_installed:
        face_rec_installed = install_pip_package("face_recognition", 
                                                 "face_recognition library")
        if face_rec_installed:
            print_success("High-accuracy face recognition enabled!")
        else:
            print_warning("face_recognition failed. Will use OpenCV LBPH (good accuracy)")
    else:
        print_warning("dlib installation failed")
        
        if is_windows:
            print("\nTrying alternative installation method...")
            print("Attempting to install dlib-binary (pre-compiled)...")
            
            if install_pip_package("dlib-binary", "dlib-binary"):
                if install_pip_package("face_recognition", "face_recognition"):
                    print_success("Face recognition installed via pre-built wheel!")
                else:
                    print_warning("Will use OpenCV LBPH face recognizer instead")
            else:
                print_warning("Pre-built wheel also failed")
                print("To install manually later:")
                print("1. Install Visual Studio Build Tools")
                print("2. Run: pip install cmake")
                print("3. Run: pip install dlib")
                print("4. Run: pip install face_recognition")
        else:
            print("\nTo install manually later:")
            print("  pip install dlib")
            print("  pip install face_recognition")
        
        print_warning("System will use OpenCV LBPH (still good, ~85% accuracy)")
    
    # Step 5: Create directories
    print_step(5, 7, "Creating directories...")
    create_directory("models")
    create_directory("person_images")
    create_directory("screenshots")
    print_success("Created directories: models, person_images, screenshots")
    
    # Step 6: Download MobileNet SSD model
    print_step(6, 7, "Downloading MobileNet SSD model...")
    
    models_dir = Path("models")
    prototxt_path = models_dir / "MobileNetSSD_deploy.prototxt"
    caffemodel_path = models_dir / "MobileNetSSD_deploy.caffemodel"
    
    if not prototxt_path.exists():
        download_file(
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
            str(prototxt_path)
        )
    else:
        print_success("MobileNet prototxt already exists")
    
    if not caffemodel_path.exists():
        download_file(
            "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel",
            str(caffemodel_path)
        )
    else:
        print_success("MobileNet model already exists")
    
    # Step 7: Download YOLO model (optional)
    print_step(7, 7, "Downloading YOLO model...")
    
    yolo_weights = models_dir / "yolov3.weights"
    yolo_cfg = models_dir / "yolov3.cfg"
    yolo_names = models_dir / "coco.names"
    
    print("YOLO provides better accuracy but is larger (250MB)")
    response = input("Download YOLO model? [y/N]: ").strip().lower()
    
    if response in ['y', 'yes']:
        print("Downloading YOLO files (this may take a while)...")
        
        if not yolo_weights.exists():
            download_file(
                "https://pjreddie.com/media/files/yolov3.weights",
                str(yolo_weights)
            )
        else:
            print_success("YOLO weights already exist")
        
        if not yolo_cfg.exists():
            download_file(
                "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
                str(yolo_cfg)
            )
        else:
            print_success("YOLO config already exists")
        
        if not yolo_names.exists():
            download_file(
                "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
                str(yolo_names)
            )
        else:
            print_success("YOLO names already exist")
    else:
        print("Skipping YOLO download (you can download later)")
    
    # Final summary
    print_header("Setup Complete!")
    print("\nQUICK START GUIDE:")
    print("\n1. Add people you want to recognize:")
    print('   python advanced_detector.py --add-person "John Doe" photo1.jpg photo2.jpg')
    print("\n2. List all known people:")
    print("   python advanced_detector.py --list-people")
    print("\n3. Run the detector:")
    print("   python advanced_detector.py")
    print("\n4. Advanced options:")
    print("   python advanced_detector.py --model yolo              # Use YOLO (more accurate)")
    print("   python advanced_detector.py --confidence 0.7          # Higher confidence threshold")
    print("   python advanced_detector.py --face-confidence 60      # Face recognition threshold")
    print("   python advanced_detector.py --no-objects              # Face recognition only")
    print("\nTIPS:")
    print("- Put reference photos in person_images/ folder")
    print("- Use 2-3 clear photos per person for best results")
    print("- Press 'c' while running to capture unknown faces")
    print("- Press 's' to save screenshots")
    print("- Press 'q' to quit")
    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
