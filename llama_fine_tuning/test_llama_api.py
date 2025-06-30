#!/usr/bin/env python3
"""
Test script to verify Llama API functionality and GPU support
"""

import requests
import json
import time
import sys
from pathlib import Path

def test_gpu_availability():
    """Test if GPU is available in the container"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA not available - running in CPU mode")
        return True
    except ImportError:
        print("PyTorch not installed")
        return False

def test_bitsandbytes_gpu():
    """Test bitsandbytes GPU support"""
    try:
        import bitsandbytes as bnb
        import torch
        
        if torch.cuda.is_available():
            # Test creating a quantized layer
            test_layer = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False)
            test_layer = test_layer.cuda()
            print("✅ BitsAndBytes GPU support is working")
            return True
        else:
            print("⚠️  CUDA not available, cannot test bitsandbytes GPU support")
            return False
    except Exception as e:
        print(f"❌ BitsAndBytes GPU support error: {e}")
        return False

def test_api_health(api_url="http://localhost:8080"):
    """Test API health endpoint"""
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API health check passed: {result}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API health check error: {e}")
        return False

def test_chat_endpoint(api_url="http://localhost:8080"):
    """Test chat completion endpoint"""
    try:
        payload = {
            "messages": [
                {"role": "user", "content": "Hello! Can you help me plan a trip to Paris?"}
            ],
            "max_length": 200,
            "temperature": 0.7
        }
        
        print("Testing chat completion endpoint...")
        response = requests.post(
            f"{api_url}/chat",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat completion test passed")
            print(f"Generated text: {result['generated_text'][:100]}...")
            return True
        else:
            print(f"❌ Chat completion test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat completion test error: {e}")
        return False

def test_generate_endpoint(api_url="http://localhost:8080"):
    """Test text generation endpoint"""
    try:
        payload = {
            "prompt": "Plan a 3-day trip to Tokyo:",
            "max_length": 150,
            "temperature": 0.7
        }
        
        print("Testing text generation endpoint...")
        response = requests.post(
            f"{api_url}/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Text generation test passed")
            print(f"Generated text: {result['generated_text'][:100]}...")
            return True
        else:
            print(f"❌ Text generation test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Text generation test error: {e}")
        return False

def test_model_info(api_url="http://localhost:8080"):
    """Test model info endpoint"""
    try:
        response = requests.get(f"{api_url}/model-info", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model info test passed: {result}")
            return True
        else:
            print(f"❌ Model info test failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info test error: {e}")
        return False

def main():
    print("=== Llama API Test ===")
    
    # Test GPU availability
    print("\n1. Testing GPU availability...")
    gpu_ok = test_gpu_availability()
    
    # Test bitsandbytes GPU support
    print("\n2. Testing bitsandbytes GPU support...")
    bnb_ok = test_bitsandbytes_gpu()
    
    # Wait a bit for API to start
    print("\n3. Waiting for API to start...")
    time.sleep(10)
    
    # Test API health
    print("\n4. Testing API health...")
    health_ok = test_api_health()
    
    # Test model info
    print("\n5. Testing model info...")
    model_info_ok = test_model_info()
    
    # Test chat endpoint
    print("\n6. Testing chat completion...")
    chat_ok = test_chat_endpoint()
    
    # Test generate endpoint
    print("\n7. Testing text generation...")
    generate_ok = test_generate_endpoint()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"GPU Available: {'✅' if gpu_ok else '❌'}")
    print(f"BitsAndBytes GPU: {'✅' if bnb_ok else '❌'}")
    print(f"API Health: {'✅' if health_ok else '❌'}")
    print(f"Model Info: {'✅' if model_info_ok else '❌'}")
    print(f"Chat Endpoint: {'✅' if chat_ok else '❌'}")
    print(f"Generate Endpoint: {'✅' if generate_ok else '❌'}")
    
    if all([gpu_ok, health_ok, model_info_ok, chat_ok, generate_ok]):
        print("\n🎉 All tests passed! Llama API is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 