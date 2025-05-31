"""
AI Models for Travel Advisory Bot - Python 3.13 Compatible Architecture
Prioritizes local models (Ollama, HuggingFace) over cloud APIs for better reliability
"""

import requests
import json
import asyncio
from typing import Dict, Optional, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Base classes to avoid circular imports
class InputType(Enum):
    TEXT = "text"
    IMAGE = "image" 
    PDF = "pdf"
    VIDEO = "video"
    AUDIO = "audio"

@dataclass
class ProcessedInput:
    """Processed input from various sources"""
    input_type: InputType
    content: str
    metadata: Dict[str, Any]

class AIModel(ABC):
    """Abstract base class for AI models"""
    
    @abstractmethod
    async def analyze_input(self, processed_input: ProcessedInput) -> Dict[str, Any]:
        """Analyze processed input and return insights"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available"""
        pass

# Python 3.13 Compatible Dependencies
TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False

# HuggingFace Transformers (Primary choice for local models)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    TORCH_AVAILABLE = True
except ImportError as e:
    pass

# Configuration - Using fallback to avoid import issues
class Config:
    OLLAMA_BASE_URL = "http://localhost:11434"
    DEVSTRAL_MODEL_NAME = "devstral"
    HUGGINGFACE_MODEL = "microsoft/DialoGPT-medium"
    HUGGINGFACE_TOKEN = None

class DevstralModel(AIModel):
    """
    Devstral model via Ollama - Primary AI backend
    Optimized for travel advisory and agentic reasoning
    """
    
    def __init__(self, model_name: str = "devstral"):
        self.model_name = model_name
        self.base_url = getattr(Config, 'OLLAMA_BASE_URL', "http://localhost:11434")
        self._is_available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Ollama and Devstral are running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = response.json().get('models', [])
            return any(model.get('name', '').startswith(self.model_name) for model in models)
        except:
            return False
    
    def is_available(self) -> bool:
        """Check if the model is available (implements abstract method)"""
        return self._is_available
    
    async def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate travel advisory response using Devstral"""
        if not self.is_available():
            return "Devstral model not available. Please ensure Ollama is running with Devstral model."
        
        try:
            system_prompt = """You are an expert AI travel advisor with comprehensive knowledge of:
- Global destinations, attractions, and hidden gems
- Cultural insights and local customs
- Travel logistics, visas, and requirements
- Budget planning and optimization
- Safety considerations and travel advisories
- Sustainable and responsible tourism

Provide detailed, personalized, and actionable travel recommendations. Always consider the traveler's preferences, budget, and constraints."""

            full_prompt = f"{system_prompt}\n\n"
            if context:
                full_prompt += f"Context: {context}\n\n"
            full_prompt += f"Request: {prompt}\n\nResponse:"
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,                "options": {
                    "temperature": 0.7,  # Slightly lower for more focused responses
                    "num_predict": 800,  # Reduced from 1024 for faster responses
                    "top_p": 0.9,
                    "num_ctx": 4096,     # Context window
                    "repeat_penalty": 1.1  # Avoid repetition
                }            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # Increased to 120 seconds for large 14GB model
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                return f"Error generating response: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "⏰ The AI model is taking longer than expected to respond. This is normal for the first query with large models like Devstral (14GB). Please try again or try a shorter request."
        except requests.exceptions.ConnectionError:
            return "🔌 Unable to connect to Ollama service. Please ensure Ollama is running with the Devstral model loaded."
        except Exception as e:
            return f"Error communicating with Devstral: {str(e)}"
    
    async def analyze_input(self, processed_input: ProcessedInput) -> Dict[str, Any]:
        """Analyze processed input using Devstral's travel expertise"""
        content_summary = processed_input.content[:500] + "..." if len(processed_input.content) > 500 else processed_input.content
        
        analysis_prompt = f"""
        Analyze this travel-related content and provide insights:
        
        Content Type: {processed_input.input_type.value}
        Content: {content_summary}
        Metadata: {processed_input.metadata}
        
        Please provide:
        1. Key travel insights
        2. Destination recommendations
        3. Budget considerations
        4. Safety and cultural tips
        5. Best time to visit suggestions
        
        Format as structured data with confidence scores.
        """
        
        response = await self.generate_response(analysis_prompt)
        return {
            "detailed_analysis": response,
            "model": f"devstral-{self.model_name}",
            "capabilities": ["travel_expert", "cultural_insights", "personalization"],
            "processing_time": "optimized_local"
        }

class HuggingFaceLocalModel(AIModel):
    """
    Local HuggingFace model for offline processing
    Fallback when cloud services are unavailable
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def is_available(self) -> bool:
        """Check if the local model is available"""
        return self.is_loaded and TRANSFORMERS_AVAILABLE
    
    def _load_model(self):
        """Load model locally"""
        try:
            print(f"Loading {self.model_name} locally...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.is_loaded = True
            print(f"✅ {self.model_name} loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {e}")
            self.is_loaded = False
    
    async def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response using local HuggingFace model"""
        if not self.is_available():
            return "Local HuggingFace model not available. Please ensure transformers library is installed."
        
        try:
            travel_context = "Travel Advisory Context: Provide helpful travel information and recommendations.\n\n"
            full_prompt = travel_context
            if context:
                full_prompt += f"Context: {context}\n"
            full_prompt += f"Query: {prompt}\nResponse:"
            
            inputs = self.tokenizer.encode(full_prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(full_prompt):].strip()
            
            return response if response else "Unable to generate response with local model."
            
        except Exception as e:
            return f"Error with local model: {str(e)}"
    
    async def analyze_input(self, processed_input: ProcessedInput) -> Dict[str, Any]:
        """Analyze input using local HuggingFace model"""
        prompt = f"Analyze this travel content: {processed_input.content[:300]}"
        response = await self.generate_response(prompt)
        
        return {
            "analysis": response,
            "model": f"local-{self.model_name}",
            "processing": "offline"
        }

class OllamaGenericModel(AIModel):
    """
    Generic Ollama model support for various models
    Supports Llama, Mistral, CodeLlama, and other Ollama models
    """
    
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        self.base_url = getattr(Config, 'OLLAMA_BASE_URL', "http://localhost:11434")
        self.available_models = self._get_available_models()
    
    def is_available(self) -> bool:
        """Check if the Ollama model is available"""
        return self.model_name in self.available_models
    
    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = response.json().get('models', [])
            return [model.get('name', '') for model in models]
        except:
            return []
    
    async def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response using any Ollama model"""
        if not self.is_available():
            return f"Model {self.model_name} not available. Available models: {', '.join(self.available_models)}"
        
        try:
            system_prompt = "You are a helpful AI assistant specializing in travel advice and planning."
            
            full_prompt = f"{system_prompt}\n\n"
            if context:
                full_prompt += f"Context: {context}\n\n"
            full_prompt += f"Request: {prompt}\n\nResponse:"
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 512
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=45
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Error with Ollama model: {str(e)}"
    
    async def analyze_input(self, processed_input: ProcessedInput) -> Dict[str, Any]:
        """Analyze input using generic Ollama model"""
        prompt = f"Analyze this travel-related content: {processed_input.content[:400]}"
        response = await self.generate_response(prompt)
        
        return {
            "analysis": response,
            "model": f"ollama-{self.model_name}",
            "capabilities": ["general_purpose"]
        }

class ModelFactory:
    """Factory for creating and managing AI models"""
    
    @staticmethod
    def create_model(model_type: str = "auto") -> AIModel:
        """Create the appropriate model based on type and availability"""
        
        if model_type == "devstral":
            return DevstralModel()
        elif model_type == "huggingface":
            return HuggingFaceLocalModel()
        elif model_type == "ollama":
            return OllamaGenericModel()
        elif model_type == "auto":
            return ModelFactory.get_best_available_model()
        else:
            # Fallback to best available
            return ModelFactory.get_best_available_model()
    
    @staticmethod
    def get_best_available_model() -> AIModel:
        """Get the best available model based on priority"""
        
        # Priority 1: Devstral (specialized for travel)
        devstral = DevstralModel()
        if devstral.is_available():
            return devstral
        
        # Priority 2: Other Ollama models
        ollama_models = ["llama2", "mistral", "codellama"]
        for model_name in ollama_models:
            model = OllamaGenericModel(model_name)
            if model.is_available():
                return model
        
        # Priority 3: Local HuggingFace (fallback)
        hf_model = HuggingFaceLocalModel()
        if hf_model.is_available():
            return hf_model
        
        # Fallback: Return DevStral anyway (will show error message if not available)
        return devstral
    
    @staticmethod
    def get_available_models() -> List[Dict[str, Any]]:
        """Get list of all available models with their status"""
        models = []
        
        # Check Devstral
        devstral = DevstralModel()
        models.append({
            "name": "Devstral",
            "type": "ollama",
            "available": devstral.is_available(),
            "specialization": "travel_expert"
        })
        
        # Check HuggingFace
        hf_model = HuggingFaceLocalModel()
        models.append({
            "name": "HuggingFace Local",
            "type": "local",
            "available": hf_model.is_available(),
            "specialization": "general"
        })
        
        # Check other Ollama models
        ollama_models = ["llama2", "mistral", "codellama"]
        for model_name in ollama_models:
            model = OllamaGenericModel(model_name)
            models.append({
                "name": model_name,
                "type": "ollama",
                "available": model.is_available(),
                "specialization": "general"
            })
        
        return models
