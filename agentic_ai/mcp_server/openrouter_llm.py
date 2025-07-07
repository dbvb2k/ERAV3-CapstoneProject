"""
Custom Local Llama API LLM implementation for LangChain.
"""

from typing import Any, List, Mapping, Optional, Dict
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import BaseModel, Field, ConfigDict
import aiohttp
import json

class LocalLlamaLLM(LLM):
    """LangChain LLM implementation for Local Llama API."""
    
    api_base_url: str = "http://localhost:8080"
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_length: int = 2000
    use_chat_endpoint: bool = True  # Use chat endpoint for better conversation handling
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @property
    def _llm_type(self) -> str:
        return "local_llama"

    def _format_chat_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Convert a single prompt to chat format for the Llama API"""
        # If the prompt already contains chat markers, use it as is
        if "<|start_header_id|>" in prompt:
            return [{"role": "user", "content": prompt}]
        
        # Otherwise, treat it as a user message
        return [{"role": "user", "content": prompt}]

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async call to Local Llama API."""
        headers = {
            "Content-Type": "application/json"
        }
        
        # Always use chat endpoint for better compatibility with Llama 3
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "messages": messages,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": 0.9,
            "do_sample": True,
            "num_return_sequences": 1
        }
        
        endpoint = f"{self.api_base_url}/chat"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")
                
                result = await response.json()
                if not result.get('generated_text'):
                    raise ValueError("No generated text found in API result")
                
                return result['generated_text']

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Sync call to Local Llama API."""
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
            "api_base_url": self.api_base_url,
            "temperature": self.temperature,
            "max_length": self.max_length,
            "use_chat_endpoint": self.use_chat_endpoint
        } 