"""
Custom OpenRouter LLM implementation for LangChain.
"""

from typing import Any, List, Mapping, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import BaseModel, Field, ConfigDict
import aiohttp
import json

class OpenRouterLLM(LLM):
    """LangChain LLM implementation for OpenRouter."""
    
    api_key: str
    model: str = "meta-llama/llama-2-70b-chat"
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = 2000
    site_url: str = "http://localhost:8501"
    site_name: str = "AI Travel Planner"
    
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
        """Async call to OpenRouter's API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if stop:
            payload["stop"] = stop
            
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")
                
                result = await response.json()
                if not result.get('choices'):
                    raise ValueError("No response choices found in API result")
                
                return result['choices'][0]['message']['content']

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Sync call to OpenRouter's API."""
        import asyncio
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