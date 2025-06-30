"""
Fallback LLM implementation that uses Local Llama API as primary and OpenRouter as fallback.
"""

import os
import asyncio
import aiohttp
import json
from typing import Any, List, Mapping, Optional, Dict
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenRouterLLM(LLM):
    """LangChain LLM implementation for OpenRouter API."""
    
    api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    model: str = "meta-llama/llama-3.1-8b-instruct"
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = 2000
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async call to OpenRouter API."""
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AI Travel Planner"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenRouter API call failed with status {response.status}: {error_text}")
                
                result = await response.json()
                if not result.get('choices') or not result['choices'][0].get('message', {}).get('content'):
                    raise ValueError("No generated text found in OpenRouter API result")
                
                return result['choices'][0]['message']['content']

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Sync call to OpenRouter API."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._acall(prompt, stop, run_manager, **kwargs))

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

class FallbackLLM(LLM):
    """LangChain LLM implementation with fallback from Local Llama to OpenRouter."""
    
    llama_api_url: str = Field(default_factory=lambda: os.getenv("LLAMA_API_URL", "http://localhost:8080"))
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_length: int = 2000
    use_chat_endpoint: bool = True
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llama_available = None
        self._openrouter_available = None
        self._test_availability()
    
    @property
    def _llm_type(self) -> str:
        return "fallback_llm"

    def _test_availability(self):
        """Test availability of both LLM services."""
        # Test Local Llama API
        try:
            import requests
            response = requests.get(f"{self.llama_api_url}/health", timeout=5)
            self._llama_available = response.status_code == 200
        except:
            self._llama_available = False
        
        # Test OpenRouter API key
        self._openrouter_available = bool(self.openrouter_api_key)
        
        print(f"🔍 LLM Availability Check:")
        print(f"   Local Llama API: {'✅ Available' if self._llama_available else '❌ Not available'}")
        print(f"   OpenRouter API: {'✅ Available' if self._openrouter_available else '❌ Not available'}")

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async call with fallback logic."""
        
        # Try Local Llama API first
        if self._llama_available:
            try:
                return await self._call_llama_api(prompt)
            except Exception as e:
                print(f"⚠️  Local Llama API failed: {str(e)}")
                print("🔄 Falling back to OpenRouter API...")
        
        # Fallback to OpenRouter API
        if self._openrouter_available:
            try:
                return await self._call_openrouter_api(prompt)
            except Exception as e:
                print(f"⚠️  OpenRouter API failed: {str(e)}")
        
        # If both fail, raise an error
        raise Exception("Both Local Llama API and OpenRouter API are unavailable")

    async def _call_llama_api(self, prompt: str) -> str:
        """Call Local Llama API."""
        headers = {"Content-Type": "application/json"}
        
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "messages": messages,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": 0.9,
            "do_sample": True,
            "num_return_sequences": 1
        }
        
        endpoint = f"{self.llama_api_url}/chat"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Local Llama API call failed with status {response.status}: {error_text}")
                
                result = await response.json()
                if not result.get('generated_text'):
                    raise ValueError("No generated text found in Local Llama API result")
                
                return result['generated_text']

    async def _call_openrouter_api(self, prompt: str) -> str:
        """Call OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AI Travel Planner"
        }
        
        payload = {
            "model": "meta-llama/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_length
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenRouter API call failed with status {response.status}: {error_text}")
                
                result = await response.json()
                if not result.get('choices') or not result['choices'][0].get('message', {}).get('content'):
                    raise ValueError("No generated text found in OpenRouter API result")
                
                return result['choices'][0]['message']['content']

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Sync call with fallback logic."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._acall(prompt, stop, run_manager, **kwargs))

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "llama_api_url": self.llama_api_url,
            "openrouter_available": self._openrouter_available,
            "temperature": self.temperature,
            "max_length": self.max_length
        } 