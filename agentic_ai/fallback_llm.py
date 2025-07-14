"""
Fallback LLM implementation that uses Local Llama API as primary and OpenRouter as fallback.
"""

import os
import asyncio
import aiohttp
import json
from typing import Any, List, Sequence, Mapping, Optional, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from pydantic import Field, ConfigDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FallbackLLM(BaseChatModel):
    """LangChain Chat Model implementation with fallback from Local Llama to OpenRouter."""
    
    llama_api_url: str = Field(default_factory=lambda: os.getenv("LLAMA_API_URL", "http://localhost:8080"))
    openrouter_api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_model: str = "meta-llama/llama-3-8b-instruct" # "google/gemini-2.0-flash-001" 
    site_url: str = Field(default_factory=lambda: os.getenv("SITE_URL", "http://localhost:8501"))
    site_name: str = Field(default_factory=lambda: os.getenv("SITE_NAME", "AI Travel Planner"))
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_length: int = 2000
    debug: bool = False
    
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

    async def _agenerate(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion asynchronously."""
        # Extract tools/functions
        tools = kwargs.get("tools", [])
        
        if self.debug:
            print(f"Tool calling enabled: {bool(tools)}")
            if tools:
                print(f"Number of tools: {len(tools)}")
                tool_names = [t.get("function", {}).get("name") for t in tools if "function" in t]
                print(f"Tool names: {tool_names}")
        
        # Try Local Llama API first
        if self._llama_available:
            try:
                return await self._call_llama_api_chat(messages, tools=tools)
            except Exception as e:
                print(f"⚠️ Local Llama API failed: {str(e)}")
                print("🔄 Falling back to OpenRouter API...")
        
        # Fallback to OpenRouter API
        if self._openrouter_available:
            try:
                return await self._call_openrouter_api_chat(messages, tools=tools)
            except Exception as e:
                print(f"⚠️ OpenRouter API failed: {str(e)}")
        
        # If both fail, raise an error
        raise Exception("Both Local Llama API and OpenRouter API are unavailable")

    async def _call_llama_api_chat(self, messages: Sequence[BaseMessage], tools=None) -> ChatResult:
        """Call Local Llama API and format response as ChatResult."""
        headers = {"Content-Type": "application/json"}
        
        # Convert LangChain messages to API format
        api_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                api_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            else:
                api_messages.append({"role": "assistant", "content": str(msg.content)})
        
        payload = {
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_length,
        }
        
        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        endpoint = f"{self.llama_api_url}/v1/chat/completions"
        
        if self.debug:
            print(f"Payload to Llama API: {json.dumps(payload, indent=2)}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Local Llama API call failed with status {response.status}: {error_text}")
                
                result = await response.json()
                
                # Create AI message from response
                message = result.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "")  # Get content or empty string
                tool_calls = message.get("tool_calls", [])
                additional_kwargs = {}

                # Handle tool calls if present
                if tool_calls:
                    additional_kwargs["tool_calls"] = tool_calls
                    if self.debug:
                        print(f"Tool calls detected: {json.dumps(tool_calls, indent=2)}")
                    
                    # LangChain requires content to be a string, not None
                    if content is None:
                        content = ""  # Ensure content is never None

                # Create generation
                generation = ChatGeneration(
                    message=AIMessage(content=content, additional_kwargs=additional_kwargs),
                    generation_info={"finish_reason": result.get("choices", [{}])[0].get("finish_reason")}
                )
                
                return ChatResult(generations=[generation])

    async def _call_openrouter_api_chat(self, messages: Sequence[BaseMessage], tools=None) -> ChatResult:
        """Call OpenRouter API and format response as ChatResult."""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name
        }
        
        # Convert LangChain messages to API format
        api_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                api_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            else:
                api_messages.append({"role": "assistant", "content": str(msg.content)})
        
        payload = {
            "model": self.openrouter_model,
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_length
        }
        
        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        if self.debug:
            print(f"Payload to OpenRouter API: {json.dumps(payload, indent=2)}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.openrouter_api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenRouter API call failed with status {response.status}: {error_text}")
                
                result = await response.json()
                
                # Create AI message from response
                message = result.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "")
                
                # Handle tool calls if present
                tool_calls = message.get("tool_calls", [])
                additional_kwargs = {}
                
                if tool_calls:
                    additional_kwargs["tool_calls"] = tool_calls
                    if self.debug:
                        print(f"Tool calls detected: {json.dumps(tool_calls, indent=2)}")
                
                # Create generation
                generation = ChatGeneration(
                    message=AIMessage(content=content, additional_kwargs=additional_kwargs),
                    generation_info={"finish_reason": result.get("choices", [{}])[0].get("finish_reason")}
                )
                
                return ChatResult(generations=[generation])

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion synchronously."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._agenerate(messages, stop, run_manager, **kwargs))

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "llama_api_url": self.llama_api_url,
            "openrouter_available": self._openrouter_available,
            "openrouter_model": self.openrouter_model,
            "temperature": self.temperature,
            "max_length": self.max_length
        }
    
    async def apredict(self, prompt: str) -> str:
        """Asynchronously generate a text completion from a single prompt string."""
        messages = [SystemMessage(content=prompt)]
        result = await self._agenerate(messages)
        content = result.generations[0].message.content
        return content if isinstance(content, str) else str(content)