#!/usr/bin/env python3
"""
Client script to test the Llama API endpoints
"""

import requests
import json
import time

# API configuration
API_BASE_URL = "http://localhost:8080"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return data.get("model_loaded", False)
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n🔍 Testing model info endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {str(e)}")
        return False

def test_text_generation():
    """Test the text generation endpoint"""
    print("\n🔍 Testing text generation endpoint...")
    
    payload = {
        "prompt": "Tell me about the best places to visit in Tokyo for first-time travelers.",
        "max_length": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Text generation successful:")
            print(f"Generated text: {data['generated_text']}")
            print(f"Input length: {data['input_length']}")
            print(f"Generated length: {data['generated_length']}")
            return True
        else:
            print(f"❌ Text generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Text generation error: {str(e)}")
        return False

def test_chat_completion():
    """Test the chat completion endpoint"""
    print("\n🔍 Testing chat completion endpoint...")
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful travel assistant. Provide detailed and helpful travel advice."
            },
            {
                "role": "user",
                "content": "I'm planning a trip to Italy. What are the must-visit cities and what should I know about Italian culture?"
            }
        ],
        "max_length": 300,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat completion successful:")
            print(f"Generated text: {data['generated_text']}")
            print(f"Input length: {data['input_length']}")
            print(f"Generated length: {data['generated_length']}")
            return True
        else:
            print(f"❌ Chat completion failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Chat completion error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Llama API tests...")
    print(f"API Base URL: {API_BASE_URL}")
    
    # Wait a bit for the server to start
    print("⏳ Waiting for server to be ready...")
    time.sleep(5)
    
    # Test health endpoint
    model_loaded = test_health()
    
    if not model_loaded:
        print("❌ Model is not loaded. Please check the server logs.")
        return
    
    # Test model info
    test_model_info()
    
    # Test text generation
    test_text_generation()
    
    # Test chat completion
    test_chat_completion()
    
    print("\n🎉 All API tests completed!")

if __name__ == "__main__":
    main() 