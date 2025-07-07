#!/usr/bin/env python3
"""
Debug script to test local Llama API response
"""

import asyncio
import aiohttp
import json

async def test_api_response():
    """Test the local Llama API directly"""
    print("🔍 Testing Local Llama API Response...")
    
    # Test 1: Simple text generation
    print("\n1. Testing simple text generation...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "prompt": "Tell me about Tokyo in one sentence.",
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "num_return_sequences": 1
            }
            
            async with session.post(
                "http://localhost:8080/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                print(f"Status: {response.status}")
                print(f"Headers: {dict(response.headers)}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"Response: {json.dumps(result, indent=2)}")
                    
                    if result.get('generated_text'):
                        print(f"Generated text: '{result['generated_text']}'")
                        print(f"Text length: {len(result['generated_text'])}")
                    else:
                        print("❌ No generated_text in response")
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text}")
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    # Test 2: Chat completion
    print("\n2. Testing chat completion...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Tell me about Tokyo in one sentence."
                    }
                ],
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "num_return_sequences": 1
            }
            
            async with session.post(
                "http://localhost:8080/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                print(f"Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"Response: {json.dumps(result, indent=2)}")
                    
                    if result.get('generated_text'):
                        print(f"Generated text: '{result['generated_text']}'")
                        print(f"Text length: {len(result['generated_text'])}")
                    else:
                        print("❌ No generated_text in response")
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text}")
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    # Test 3: Health check
    print("\n3. Testing health endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8080/health") as response:
                print(f"Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"Health: {json.dumps(result, indent=2)}")
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text}")
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    # Test 4: Model info
    print("\n4. Testing model info...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8080/model-info") as response:
                print(f"Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"Model info: {json.dumps(result, indent=2)}")
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text}")
    except Exception as e:
        print(f"❌ Exception: {str(e)}")

async def main():
    """Main function"""
    print("🚀 Starting Local Llama API Debug Tests...")
    await test_api_response()
    print("\n🎉 Debug tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 