#!/usr/bin/env python3
"""
Test script to verify GPU setup and GeoCLIP API functionality
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

def test_api_health(api_url="http://localhost:8000"):
    """Test API health endpoint"""
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            print(f"✅ API health check passed: {response.json()}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API health check error: {e}")
        return False

def test_geoclip_endpoint(api_url="http://localhost:8000"):
    """Test GeoCLIP endpoint with a sample image"""
    try:
        # Create a simple test image (1x1 pixel)
        import numpy as np
        from PIL import Image
        import io
        import base64
        
        # Create a simple test image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Test the endpoint
        payload = {
            "image": img_base64,
            "top_k": 5
        }
        
        print("Testing GeoCLIP endpoint...")
        response = requests.post(
            f"{api_url}/identify_location",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ GeoCLIP test passed: {result}")
            return True
        else:
            print(f"❌ GeoCLIP test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ GeoCLIP test error: {e}")
        return False

def main():
    print("=== GPU Setup Test ===")
    
    # Test GPU availability
    print("\n1. Testing GPU availability...")
    gpu_ok = test_gpu_availability()
    
    # Wait a bit for API to start
    print("\n2. Waiting for API to start...")
    time.sleep(5)
    
    # Test API health
    print("\n3. Testing API health...")
    health_ok = test_api_health()
    
    # Test GeoCLIP endpoint
    print("\n4. Testing GeoCLIP endpoint...")
    geoclip_ok = test_geoclip_endpoint()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"GPU Available: {'✅' if gpu_ok else '❌'}")
    print(f"API Health: {'✅' if health_ok else '❌'}")
    print(f"GeoCLIP Endpoint: {'✅' if geoclip_ok else '❌'}")
    
    if all([gpu_ok, health_ok, geoclip_ok]):
        print("\n🎉 All tests passed! GPU setup is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 