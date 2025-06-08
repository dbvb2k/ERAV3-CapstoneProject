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
import traceback
import time

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
    print("✅ Successfully imported transformers and torch")
except ImportError as e:
    print(f"❌ Failed to import dependencies: {str(e)}")
    pass

# Configuration - Using fallback to avoid import issues
class Config:
    OLLAMA_BASE_URL = "http://localhost:11434"
    DEVSTRAL_MODEL_NAME = "devstral"
    HUGGINGFACE_MODEL = "microsoft/phi-2"
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
    Optimized for CPU usage with Microsoft's Phi model
    """
    
    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_loaded = False
        
        if TRANSFORMERS_AVAILABLE:
            print("🔄 Attempting to load model...")
            self._load_model()
        else:
            print("❌ Transformers library not available")
    
    def is_available(self) -> bool:
        """Check if the local model is available"""
        return self.is_loaded and TRANSFORMERS_AVAILABLE
    
    def _load_model(self):
        """Load model locally"""
        try:
            print(f"📥 Loading {self.model_name} locally...")
            start_time = time.time()
            
            # Step 1: Load tokenizer
            print("1️⃣ Loading tokenizer...")
            tokenizer_start = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            print(f"✅ Tokenizer loaded ({time.time() - tokenizer_start:.2f}s)")
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("ℹ️ Set pad token to eos token")
            
            # Step 2: Load model with optimizations
            print("2️⃣ Loading model (this may take a few minutes)...")
            model_start = time.time()
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                    offload_folder="model_cache"  # Cache model parts to disk
                )
                print(f"✅ Model loaded ({time.time() - model_start:.2f}s)")
            except Exception as model_error:
                print(f"⚠️ First loading attempt failed: {str(model_error)}")
                print("🔄 Trying alternative loading method...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
            
            # Step 3: Create pipeline
            print("3️⃣ Creating text generation pipeline...")
            pipeline_start = time.time()
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    framework="pt",
                    model_kwargs={"low_cpu_mem_usage": True}
                )
                print(f"✅ Pipeline created ({time.time() - pipeline_start:.2f}s)")
            except Exception as pipeline_error:
                print(f"⚠️ First pipeline creation attempt failed: {str(pipeline_error)}")
                print("🔄 Trying alternative pipeline creation...")
                self.pipeline = pipeline(
                    task="text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device_map="auto"
                )
            
            total_time = time.time() - start_time
            self.is_loaded = True
            print(f"✅ {self.model_name} loaded successfully in {total_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {str(e)}")
            print(f"📋 Full error: {traceback.format_exc()}")
            self.is_loaded = False
    
    async def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response using local HuggingFace model"""
        if not self.is_available():
            return "Local HuggingFace model not available. Please ensure transformers library is installed."
        
        try:
            # Log the incoming request
            print("\n" + "="*50)
            print("📝 Received Query:", prompt)
            print("="*50)
            
            travel_context = """You are a knowledgeable travel advisor. Your goal is to provide helpful, accurate, and detailed travel information and recommendations.
Focus on:
- Destination insights and local attractions
- Cultural considerations and customs
- Travel logistics and planning
- Budget recommendations
- Safety tips
- Local transportation options
Please provide specific, actionable advice."""

            full_prompt = travel_context + "\n\n"
            if context:
                full_prompt += f"Context: {context}\n"
            full_prompt += f"Query: {prompt}\nResponse:"
            
            # Generate response using the pipeline with optimized settings
            print("🤔 Generating response...")
            start_time = time.time()
            
            outputs = self.pipeline(
                full_prompt,
                max_new_tokens=256,  # Reduced for faster responses
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,  # Enable KV-cache
                repetition_penalty=1.1,
                no_repeat_ngram_size=3  # Prevent repetition
            )
            
            # Extract and clean the response
            response = outputs[0]['generated_text']
            response = response[len(full_prompt):].strip()
            
            # Log the response and timing
            gen_time = time.time() - start_time
            print("\n" + "="*50)
            print(f"✨ Generated Response ({gen_time:.2f}s):")
            print(response)
            print("="*50 + "\n")
            
            return response if response else "Unable to generate response with local model."
            
        except Exception as e:
            error_msg = f"Error with local model: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    async def analyze_input(self, processed_input: ProcessedInput) -> Dict[str, Any]:
        """Analyze input using local HuggingFace model"""
        prompt = f"Analyze this travel content: {processed_input.content[:300]}"
        response = await self.generate_response(prompt)
        
        return {
            "analysis": response,
            "model": f"local-{self.model_name}",
            "processing": "offline",
            "capabilities": ["travel_expert", "local_processing"]
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
        
        # If a specific model type is requested (not "auto"), return that model
        if model_type != "auto":
            if model_type == "devstral":
                return DevstralModel()
            elif model_type == "huggingface":
                return HuggingFaceLocalModel()
            elif model_type == "ollama":
                return OllamaGenericModel()
        
        # Only use automatic selection if "auto" is specifically requested
        return ModelFactory.get_best_available_model()
    
    @staticmethod
    def get_best_available_model() -> AIModel:
        """Get the best available model based on priority"""
        
        # Priority 1: Local HuggingFace (Phi-2)
        hf_model = HuggingFaceLocalModel()
        if hf_model.is_available():
            return hf_model
        
        # Priority 2: Devstral
        devstral = DevstralModel()
        if devstral.is_available():
            return devstral
        
        # Priority 3: Other Ollama models
        ollama_models = ["llama2", "mistral", "codellama"]
        for model_name in ollama_models:
            model = OllamaGenericModel(model_name)
            if model.is_available():
                return model
        
        # Fallback: Return HuggingFace model anyway (will show proper error message)
        return hf_model
    
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
