"""
Setup and installation verification script.
"""

import sys
import subprocess
import importlib.util


def check_python_version():
    """Check if Python version is sufficient."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ✗ Python {version.major}.{version.minor} detected")
        print("  ! Python 3.8 or higher required")
        return False
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("  ! CUDA not available - will use CPU (slower)")
            return False
    except ImportError:
        print("  ! PyTorch not installed yet")
        return False


def check_ffmpeg():
    """Check if FFmpeg is installed."""
    print("\nChecking FFmpeg...")
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  ✓ {version_line}")
            return True
        else:
            print("  ✗ FFmpeg not found")
            return False
    except FileNotFoundError:
        print("  ✗ FFmpeg not installed")
        print("  ! Please install FFmpeg: https://ffmpeg.org/download.html")
        return False


def install_dependencies():
    """Install Python dependencies."""
    print("\nInstalling Python dependencies...")
    try:
        subprocess.check_call([
            sys.executable,
            '-m',
            'pip',
            'install',
            '-r',
            'requirements.txt'
        ])
        print("  ✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error installing dependencies: {e}")
        return False


def verify_installation():
    """Verify all packages are installed correctly."""
    print("\nVerifying installation...")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'librosa': 'Librosa',
        'face_alignment': 'Face Alignment',
        'whisper': 'Whisper',
        'TTS': 'Coqui TTS',
    }
    
    all_ok = True
    for package, name in packages.items():
        spec = importlib.util.find_spec(package)
        if spec is not None:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} not found")
            all_ok = False
    
    return all_ok


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    import os
    
    dirs = [
        'data/train',
        'data/val',
        'checkpoints',
        'logs',
        'outputs'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✓ Created {dir_path}")


def main():
    print("="*60)
    print("Facial Lip-Sync System - Setup")
    print("="*60)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    check_cuda()
    ffmpeg_ok = check_ffmpeg()
    
    # Ask to install dependencies
    print("\n" + "="*60)
    response = input("Install Python dependencies? (y/n): ")
    
    if response.lower() == 'y':
        if not install_dependencies():
            print("\nInstallation failed. Please check errors above.")
            sys.exit(1)
        
        # Verify
        if verify_installation():
            print("\n  ✓ All packages verified successfully")
        else:
            print("\n  ! Some packages missing - please check errors")
    
    # Create directories
    create_directories()
    
    # Summary
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    print("✓ Python version OK")
    if ffmpeg_ok:
        print("✓ FFmpeg installed")
    else:
        print("! FFmpeg missing (required for video processing)")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Prepare your training data:")
    print("   python examples/prepare_data.py --input <videos> --output data/train")
    print("\n2. Train the model:")
    print("   python examples/train_model.py --data_dir data/train")
    print("\n3. Run inference:")
    print("   python examples/run_inference.py --image face.jpg --audio speech.wav")
    print("\nSee QUICKSTART.md for detailed instructions")
    print("="*60)


if __name__ == '__main__':
    main()
