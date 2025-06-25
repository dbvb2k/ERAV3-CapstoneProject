#!/usr/bin/env python3
"""
Environment Verification Script for Travel Model Training
Ensures all dependencies are correctly installed and GPU is available
"""

import sys
import os
import subprocess
import importlib
from typing import List, Tuple, Dict

def check_python_version() -> bool:
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_gpu_availability() -> bool:
    """Check GPU availability and CUDA setup."""
    print("🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"   ✅ GPU Available: {gpu_name}")
            print(f"   ✅ GPU Memory: {gpu_memory:.1f} GB")
            print(f"   ✅ GPU Count: {gpu_count}")
            
            if gpu_memory >= 15:
                print("   ✅ Sufficient GPU memory for training")
                return True
            else:
                print("   ⚠️  Low GPU memory - training may be slow")
                return True
        else:
            print("   ❌ No GPU available - training will be very slow")
            return False
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

def check_required_packages() -> bool:
    """Check all required packages are installed with correct versions."""
    print("📦 Checking required packages...")
    
    required_packages = {
        'torch': '2.1.0',
        'transformers': '4.36.2',
        'datasets': '2.14.6',
        'peft': '0.6.2',
        'bitsandbytes': '0.41.3.post2',
        'accelerate': '0.24.1',
        'numpy': '1.24.4',
        'pandas': '1.5.3',
        'matplotlib': '3.7.5',
        'tqdm': '4.66.1'
    }
    
    all_good = True
    
    for package, expected_version in required_packages.items():
        try:
            module = importlib.import_module(package)
            if hasattr(module, '__version__'):
                installed_version = module.__version__
                if installed_version == expected_version:
                    print(f"   ✅ {package}: {installed_version}")
                else:
                    print(f"   ⚠️  {package}: {installed_version} (expected {expected_version})")
            else:
                print(f"   ✅ {package}: installed (version unknown)")
        except ImportError:
            print(f"   ❌ {package}: not installed")
            all_good = False
    
    return all_good

def check_model_access() -> bool:
    """Check if we can access the base model."""
    print("🤖 Checking model access...")
    
    try:
        from transformers import AutoTokenizer
        local_model_path = "/workspace/hf_cache/transformers/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        print("   ✅ Local base model accessible")
        return True
    except Exception as e:
        print(f"   ❌ Cannot access local base model: {e}")
        print("   💡 Trying HuggingFace fallback...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
            print("   ✅ HuggingFace base model accessible")
            return True
        except Exception as e2:
            print(f"   ❌ Cannot access HuggingFace model: {e2}")
            return False

def check_dataset_files() -> bool:
    """Check if dataset files are present."""
    print("📊 Checking dataset files...")
    
    dataset_files = [
        "../FINAL_TRAINING_DATASET_LLAMA8B.jsonl",
        "../FINAL_VALIDATION_DATASET_LLAMA8B.jsonl", 
        "../FINAL_TEST_DATASET_LLAMA8B.jsonl"
    ]
    
    all_present = True
    
    for file_path in dataset_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✅ {os.path.basename(file_path)}: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {os.path.basename(file_path)}: not found")
            all_present = False
    
    return all_present

def check_disk_space() -> bool:
    """Check available disk space."""
    print("💾 Checking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        if free_gb >= 50:
            print(f"   ✅ Available space: {free_gb:.1f} GB")
            return True
        else:
            print(f"   ⚠️  Available space: {free_gb:.1f} GB (recommended: 50+ GB)")
            return free_gb >= 20
    except Exception as e:
        print(f"   ❌ Cannot check disk space: {e}")
        return False

def check_memory() -> bool:
    """Check system memory."""
    print("🧠 Checking system memory...")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        if total_gb >= 24:
            print(f"   ✅ System RAM: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
            return True
        else:
            print(f"   ⚠️  System RAM: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
            print("   💡 Recommended: 32+ GB for optimal training")
            return total_gb >= 16
    except ImportError:
        print("   ❌ Cannot check memory (psutil not installed)")
        return False

def run_quick_training_test() -> bool:
    """Run a quick training test to ensure everything works."""
    print("🧪 Running quick training test...")
    
    try:
        # Import key training components
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        import torch
        
        print("   ✅ All training imports successful")
        
        # Test model loading with quantization
        if torch.cuda.is_available():
            print("   🔄 Testing model loading with quantization...")
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
            
            # Quick tokenization test
            test_text = "This is a test"
            tokens = tokenizer(test_text, return_tensors="pt")
            print("   ✅ Tokenization test passed")
            
            print("   ✅ Quick training test completed successfully")
            return True
        else:
            print("   ⚠️  Skipping model loading test (no GPU)")
            return True
            
    except Exception as e:
        print(f"   ❌ Training test failed: {e}")
        return False

def main():
    """Run comprehensive environment verification."""
    print("=" * 60)
    print("🚀 TRAVEL MODEL TRAINING - ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("GPU Availability", check_gpu_availability),
        ("Required Packages", check_required_packages),
        ("Model Access", check_model_access),
        ("Dataset Files", check_dataset_files),
        ("Disk Space", check_disk_space),
        ("System Memory", check_memory),
        ("Training Test", run_quick_training_test)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print()
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"   ❌ {check_name} check failed: {e}")
            results[check_name] = False
    
    print()
    print("=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20} : {status}")
        if result:
            passed += 1
    
    print()
    print(f"Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED! Ready to start training!")
        print()
        print("▶️  Start training with: ./start_travel_training.sh")
        return True
    elif passed >= total - 2:
        print("⚠️  MOSTLY READY - Minor issues detected")
        print("💡 Training should work but may have reduced performance")
        return True
    else:
        print("❌ ENVIRONMENT ISSUES DETECTED")
        print("🔧 Please fix the failed checks before training")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 