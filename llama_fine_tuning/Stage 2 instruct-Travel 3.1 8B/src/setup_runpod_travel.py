#!/usr/bin/env python3
"""
ULTIMATE RunPod Setup Script for Llama-3-8B-Instruct Travel Fine-tuning
Optimized for travel domain expertise with catastrophic forgetting prevention
"""

import os
import sys
import subprocess
import logging
import shutil
import time
import json
import torch
from pathlib import Path

def setup_logging():
    """Setup comprehensive logging."""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/setup.log')
        ]
    )
    return logging.getLogger(__name__)

def run_command_with_retry(cmd, description="", logger=None, max_retries=3, ignore_errors=False):
    """Run a command with retry mechanism."""
    if logger:
        logger.info(f"🔧 {description}")
        logger.info(f"Running: {cmd}")
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=900)
            if logger:
                logger.info(f"✅ {description} completed successfully")
            return True, result.stdout
        except subprocess.TimeoutExpired:
            if logger:
                logger.warning(f"⏰ Attempt {attempt + 1} timed out, retrying...")
            time.sleep(10)
        except subprocess.CalledProcessError as e:
            if logger:
                logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e.stderr}")
            if attempt < max_retries - 1:
                time.sleep(15)
            else:
                if ignore_errors:
                    if logger:
                        logger.warning(f"❌ {description} failed after {max_retries} attempts, continuing...")
                    return True, e.stderr
                else:
                    if logger:
                        logger.error(f"❌ {description} failed after {max_retries} attempts")
                    return False, e.stderr
    
    return False, "Max retries exceeded"

def setup_travel_environment(logger):
    """Setup optimized environment for travel model fine-tuning."""
    logger.info("🌍 Setting up travel model fine-tuning environment...")
    
    # Detect workspace path
    workspace_candidates = ["/workspace", "/runpod-volume", "/storage", os.getcwd()]
    workspace_path = None
    
    for candidate in workspace_candidates:
        if os.path.exists(candidate):
            try:
                test_file = os.path.join(candidate, "test_write_access")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                workspace_path = candidate
                logger.info(f"✅ Workspace detected at {workspace_path}")
                break
            except:
                continue
    
    if not workspace_path:
        workspace_path = os.getcwd()
        logger.warning(f"⚠️ Using current directory: {workspace_path}")
    
    # Aggressive cleanup for travel model training
    logger.info("🧹 Comprehensive cleanup for travel model training...")
    cleanup_commands = [
        ("rm -rf /tmp/* 2>/dev/null || true", "Cleaning /tmp"),
        ("rm -rf /var/tmp/* 2>/dev/null || true", "Cleaning /var/tmp"),
        (f"rm -rf {workspace_path}/hf_cache/models--meta-llama--* 2>/dev/null || true", "Cleaning old model cache"),
        (f"rm -rf {workspace_path}/.cache/huggingface/hub/models--* 2>/dev/null || true", "Cleaning HF cache"),
        ("pip cache purge 2>/dev/null || true", "Cleaning pip cache"),
        ("python -c 'import torch; torch.cuda.empty_cache()' 2>/dev/null || true", "Clearing CUDA cache"),
    ]
    
    for cmd, desc in cleanup_commands:
        try:
            subprocess.run(cmd, shell=True, check=False, capture_output=True)
            logger.info(f"   ✅ {desc}")
        except:
            logger.warning(f"   ⚠️ {desc} failed")
    
    # Setup optimized cache directories for travel training
    cache_dirs = {
        'HF_HOME': os.path.join(workspace_path, 'travel_hf_cache'),
        'HF_DATASETS_CACHE': os.path.join(workspace_path, 'travel_hf_cache', 'datasets'),
        'TRANSFORMERS_CACHE': os.path.join(workspace_path, 'travel_hf_cache', 'transformers'),
        'HUGGINGFACE_HUB_CACHE': os.path.join(workspace_path, 'travel_hf_cache', 'hub'),
        'TMPDIR': os.path.join(workspace_path, 'travel_tmp'),
        'TEMP': os.path.join(workspace_path, 'travel_tmp'),
        'TMP': os.path.join(workspace_path, 'travel_tmp'),
        'TORCH_HOME': os.path.join(workspace_path, 'travel_torch_cache'),
        'CUDA_CACHE_PATH': os.path.join(workspace_path, 'travel_cuda_cache'),
        'WANDB_CACHE_DIR': os.path.join(workspace_path, 'travel_wandb_cache'),
        'WANDB_DATA_DIR': os.path.join(workspace_path, 'travel_wandb_data'),
    }
    
    # Create directories and set environment variables
    for env_var, path in cache_dirs.items():
        os.makedirs(path, exist_ok=True)
        os.environ[env_var] = path
        logger.info(f"   Set {env_var}={path}")
    
    # Create travel model specific directories
    travel_dirs = [
        'travel_models', 'travel_datasets', 'travel_checkpoints', 
        'travel_logs', 'travel_outputs', 'travel_plots', 'travel_results'
    ]
    
    for dir_name in travel_dirs:
        dir_path = os.path.join(workspace_path, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"   Created {dir_path}")
    
    # Create persistent environment script
    env_script_path = os.path.join(workspace_path, 'setup_travel_env.sh')
    with open(env_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Travel Model Fine-tuning Environment Setup\n")
        f.write("# Source this file: source setup_travel_env.sh\n\n")
        for env_var, path in cache_dirs.items():
            f.write(f"export {env_var}={path}\n")
        f.write("\n# Travel model specific settings\n")
        f.write("export PYTHONPATH=$PYTHONPATH:$(pwd)\n")
        f.write("export TOKENIZERS_PARALLELISM=false\n")
        f.write("export CUDA_LAUNCH_BLOCKING=1\n")
        f.write("export TORCH_USE_CUDA_DSA=1\n")
        f.write("\necho '🌍 Travel model fine-tuning environment ready!'\n")
    
    os.chmod(env_script_path, 0o755)
    logger.info(f"   📝 Created travel environment script: {env_script_path}")
    
    os.chdir(workspace_path)
    logger.info(f"   Changed to workspace: {workspace_path}")
    
    return workspace_path

def validate_gpu_setup(logger):
    """Validate GPU setup for travel model training."""
    logger.info("🔍 Validating GPU setup for travel model training...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! GPU required for Llama-3-8B training.")
        return False
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"✅ Found {gpu_count} GPU(s)")
    
    total_memory = 0
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        total_memory += gpu_memory
        logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Check if we have enough memory for Llama-3-8B
    required_memory = 16  # Minimum for 8B model with LoRA
    if total_memory < required_memory:
        logger.warning(f"⚠️ Low GPU memory: {total_memory:.1f} GB (recommended: {required_memory} GB+)")
        logger.info("   💡 Will use 4-bit quantization and gradient checkpointing")
    else:
        logger.info(f"✅ Sufficient GPU memory: {total_memory:.1f} GB")
    
    return True

def install_travel_packages(logger):
    """Install packages optimized for travel model training."""
    logger.info("📦 Installing travel model training packages...")
    
    # Upgrade pip first
    success, _ = run_command_with_retry(
        "python -m pip install --upgrade pip setuptools wheel",
        "Upgrading pip and tools",
        logger
    )
    
    if not success:
        logger.error("❌ Failed to upgrade pip")
        return False
    
    # Install PyTorch with CUDA support
    success, _ = run_command_with_retry(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir",
        "Installing PyTorch with CUDA 12.1",
        logger,
        max_retries=3
    )
    
    if not success:
        logger.warning("⚠️ CUDA PyTorch install failed, trying CPU version...")
        success, _ = run_command_with_retry(
            "pip install torch torchvision torchaudio --no-cache-dir",
            "Installing PyTorch (CPU fallback)",
            logger
        )
    
    # Install transformers and related packages
    transformers_packages = [
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "scipy>=1.11.0"
    ]
    
    for package in transformers_packages:
        success, _ = run_command_with_retry(
            f"pip install {package} --no-cache-dir",
            f"Installing {package}",
            logger
        )
        if not success:
            logger.error(f"❌ Failed to install {package}")
            return False
    
    # Install monitoring and utilities
    utility_packages = [
        "wandb", "tensorboard", "evaluate", "pandas", "numpy", 
        "matplotlib", "seaborn", "tqdm", "psutil", "jsonlines",
        "rich", "typer", "gpustat"
    ]
    
    for package in utility_packages:
        run_command_with_retry(
            f"pip install {package} --no-cache-dir",
            f"Installing {package}",
            logger,
            ignore_errors=True
        )
    
    logger.info("✅ Package installation completed")
    return True

def verify_travel_setup(logger):
    """Verify the travel model training setup."""
    logger.info("🔍 Verifying travel model training setup...")
    
    # Test imports
    test_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("peft", "PEFT"),
        ("bitsandbytes", "BitsAndBytes"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy")
    ]
    
    for module, name in test_imports:
        try:
            __import__(module)
            logger.info(f"   ✅ {name} imported successfully")
        except ImportError as e:
            logger.error(f"   ❌ {name} import failed: {e}")
            return False
    
    # Test CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            # Test tensor operations
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = torch.matmul(x, y)
            logger.info("   ✅ CUDA operations working")
        else:
            logger.warning("   ⚠️ CUDA not available")
    except Exception as e:
        logger.error(f"   ❌ CUDA test failed: {e}")
        return False
    
    # Test transformers model loading
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        logger.info("   ✅ Transformers model loading works")
    except Exception as e:
        logger.warning(f"   ⚠️ Transformers test failed: {e}")
    
    logger.info("✅ Travel model training setup verified!")
    return True

def main():
    """Main setup function."""
    logger = setup_logging()
    logger.info("🚀 Starting Ultimate Travel Model Fine-tuning Setup...")
    
    try:
        # Setup environment
        workspace_path = setup_travel_environment(logger)
        
        # Validate GPU
        if not validate_gpu_setup(logger):
            logger.error("❌ GPU validation failed")
            return False
        
        # Install packages
        if not install_travel_packages(logger):
            logger.error("❌ Package installation failed")
            return False
        
        # Verify setup
        if not verify_travel_setup(logger):
            logger.error("❌ Setup verification failed")
            return False
        
        logger.info("🎉 Travel model fine-tuning setup completed successfully!")
        logger.info(f"📁 Workspace: {workspace_path}")
        logger.info("🔥 Ready to fine-tune Llama-3-8B-Instruct for travel expertise!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 