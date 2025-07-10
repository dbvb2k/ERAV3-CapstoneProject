from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uvicorn
from datetime import datetime
import logging
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel, PeftConfig
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables from .env file
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
CORS_METHODS = os.getenv("CORS_METHODS", "*").split(",")
CORS_HEADERS = os.getenv("CORS_HEADERS", "*").split(",")
API_MODE = os.getenv("API_MODE", "local").lower() # 'local' for 4-bit, 'production' for 16-bit

# Check if we're running in a container or local environment
IS_CONTAINER = os.getenv("IS_CONTAINER", "false").lower() == "true"

# For container deployment, we might not need HF_TOKEN if using local models
if not HF_TOKEN and IS_CONTAINER:
    logger.warning("HF_TOKEN not set, but running in container mode. Will attempt to load local models.")
    HF_TOKEN = None
elif not HF_TOKEN:
    logger.error("HF_TOKEN environment variable is required but not set!")
    logger.error("Please set HF_TOKEN in your .env file or environment.")
    logger.error("Alternatively, set IS_CONTAINER=true to use local models without HF_TOKEN.")
    raise ValueError("HF_TOKEN environment variable is required. Please set it in your .env file or environment.")

logger.info("Initializing Llama API service...")

# Global variables for model and tokenizer
model = None
tokenizer = None
generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    # Startup
    logger.info("Starting up Llama API service...")
    success = load_model()
    if not success:
        logger.error("Failed to load model during startup")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Llama API service...")

app = FastAPI(
    title="Llama Travel Model API",
    description="API for inference using fine-tuned Llama 3 8B Instruct model for travel-related tasks",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=CORS_METHODS,
    allow_headers=CORS_HEADERS,
)

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    max_length: Optional[int] = Field(512, description="Maximum length of generated response")
    temperature: Optional[float] = Field(0.7, description="Temperature for generation")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter")
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
    num_return_sequences: Optional[int] = Field(1, description="Number of sequences to return")

class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for text generation")
    max_length: Optional[int] = Field(512, description="Maximum length of generated text")
    temperature: Optional[float] = Field(0.7, description="Temperature for generation")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter")
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
    num_return_sequences: Optional[int] = Field(1, description="Number of sequences to return")

class GenerationResponse(BaseModel):
    generated_text: str
    input_length: int
    generated_length: int
    model_name: str
    timestamp: str


def get_model_precision_and_kwargs(gpu_available, bnb_gpu_support):
    """Determines model loading kwargs based on API_MODE and hardware support."""
    model_kwargs = {
        "trust_remote_code": True
    }

    if API_MODE == "production":
        logger.info("Production mode enabled: loading model in 16-bit precision.")
        if gpu_available:
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
        else:
            logger.warning("Production mode selected, but no GPU available. Falling back to CPU.")
            model_kwargs["device_map"] = "cpu"
    
    # Default to local mode (4-bit quantization)
    else:
        logger.info("Local mode enabled: attempting to load model with 4-bit quantization.")
        if gpu_available and bnb_gpu_support:
            logger.info("Using 4-bit quantization with GPU support")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.bfloat16 # Still specify for compute
        elif gpu_available:
            logger.warning("4-bit not supported, falling back to 16-bit precision on GPU.")
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
        else:
            logger.info("No GPU available, using CPU mode.")
            model_kwargs["device_map"] = "cpu"
            
    return model_kwargs


def check_gpu_support():
    """Check if GPU and bitsandbytes GPU support are available"""
    gpu_available = torch.cuda.is_available()
    bnb_gpu_support = False
    
    try:
        import bitsandbytes as bnb
        # Try to create a small quantized tensor to test GPU support
        if gpu_available:
            test_tensor = torch.randn(10, 10).cuda()
            bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False).cuda()
            bnb_gpu_support = True
            logger.info("BitsAndBytes GPU support is available")
        else:
            logger.info("CUDA not available, will use CPU mode")
    except Exception as e:
        logger.warning(f"BitsAndBytes GPU support not available: {e}")
        logger.info("Will use alternative quantization or CPU mode")
    
    return gpu_available, bnb_gpu_support

def load_model():
    """Load the fine-tuned Llama model with LoRA adapter"""
    global model, tokenizer, generator
    
    try:
        logger.info("Loading model configuration...")
        
        # Load PEFT configuration
        peft_config = PeftConfig.from_pretrained("model/COMPLETE_TRAVEL_MODEL")
        logger.info(f"Loaded PEFT config: {peft_config.base_model_name_or_path}")
        
        # Check GPU and bitsandbytes support
        gpu_available, bnb_gpu_support = check_gpu_support()
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer_kwargs = {
            "trust_remote_code": True,
            "use_fast": False
        }
        
        # Add token only if provided
        if HF_TOKEN:
            tokenizer_kwargs["token"] = HF_TOKEN
        
        tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            **tokenizer_kwargs
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model with appropriate configuration
        logger.info("Loading base model...")
        model_kwargs = get_model_precision_and_kwargs(gpu_available, bnb_gpu_support)
        
        # Add token only if provided
        if HF_TOKEN:
            model_kwargs["token"] = HF_TOKEN
        
        # Fix the configuration file directly to avoid RoPE scaling issues
        logger.info("Fixing configuration file to resolve RoPE scaling issues...")
        
        import json
        import os
        
        # Try to find and fix the config.json file
        config_paths = [
            os.path.join(peft_config.base_model_name_or_path, "config.json"),
            "model/COMPLETE_TRAVEL_MODEL/config.json"
        ]
        
        config_fixed = False
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    logger.info(f"Found config file: {config_path}")
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    # Fix RoPE scaling if present
                    if 'rope_scaling' in config_data:
                        logger.info(f"Original RoPE scaling in config: {config_data['rope_scaling']}")
                        
                        if isinstance(config_data['rope_scaling'], dict):
                            if 'rope_type' in config_data['rope_scaling'] and 'factor' in config_data['rope_scaling']:
                                # Fix the format
                                config_data['rope_scaling'] = {
                                    'type': config_data['rope_scaling']['rope_type'],
                                    'factor': config_data['rope_scaling']['factor']
                                }
                                logger.info(f"Fixed RoPE scaling: {config_data['rope_scaling']}")
                            elif 'type' in config_data['rope_scaling'] and 'factor' in config_data['rope_scaling']:
                                # Already correct format, but ensure no extra fields
                                config_data['rope_scaling'] = {
                                    'type': config_data['rope_scaling']['type'],
                                    'factor': config_data['rope_scaling']['factor']
                                }
                                logger.info(f"Cleaned RoPE scaling: {config_data['rope_scaling']}")
                            else:
                                # Remove problematic rope_scaling
                                del config_data['rope_scaling']
                                logger.info("Removed problematic rope_scaling")
                        
                        # Write the fixed config back
                        with open(config_path, 'w') as f:
                            json.dump(config_data, f, indent=2)
                        
                        config_fixed = True
                        logger.info(f"Successfully fixed config file: {config_path}")
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to fix config file {config_path}: {e}")
                    continue
        
        if not config_fixed:
            logger.warning("Could not find or fix config file, will try alternative approach")
        
        try:
            # Try loading without custom config first (this worked in our local test)
            logger.info("Attempting to load model with determined configuration...")
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                **model_kwargs
            )
        except Exception as fallback_error:
            logger.warning(f"Failed to load with default config: {fallback_error}")
            logger.info("Attempting to load with ignore_mismatched_sizes...")
            
            # Try with ignore_mismatched_sizes to bypass configuration issues
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                ignore_mismatched_sizes=True,
                **model_kwargs
            )
        
        # Load LoRA adapter
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, "model/COMPLETE_TRAVEL_MODEL")
        
        # Create text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto" if gpu_available else "cpu",
            torch_dtype=torch.bfloat16 if gpu_available else torch.float32
        )
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Format chat messages into the Llama 3 chat format"""
    formatted_prompt = ""
    
    for message in messages:
        if message.role == "user":
            formatted_prompt += f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{message.content}<|eot_id|>"
        elif message.role == "assistant":
            formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{message.content}<|eot_id|>"
        elif message.role == "system":
            formatted_prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{message.content}<|eot_id|>"
    
    # Add assistant header for response
    formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return formatted_prompt

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI endpoint"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

@app.post("/chat", response_model=GenerationResponse)
async def chat_completion(request: ChatRequest):
    """
    Generate chat completion using the fine-tuned Llama model
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Format chat messages
        prompt = format_chat_prompt(request.messages)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_length = inputs.input_ids.shape[1]
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=input_length + request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                num_return_sequences=request.num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        generated_length = len(outputs[0]) - input_length
        
        return GenerationResponse(
            generated_text=generated_text.strip(),
            input_length=input_length,
            generated_length=generated_length,
            model_name="llama-3-8b-instruct-travel",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation error: {str(e)}"
        )

@app.post("/generate", response_model=GenerationResponse)
async def text_generation(request: TextGenerationRequest):
    """
    Generate text from a prompt using the fine-tuned Llama model
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_length = inputs.input_ids.shape[1]
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=input_length + request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                num_return_sequences=request.num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        generated_length = len(outputs[0]) - input_length
        
        return GenerationResponse(
            generated_text=generated_text.strip(),
            input_length=input_length,
            generated_length=generated_length,
            model_name="llama-3-8b-instruct-travel",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in text generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation error: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    precision = "Unknown"
    if API_MODE == 'production':
        precision = "16-bit (bfloat16)"
    else:
        # In local mode, check if quantization was actually applied
        if hasattr(model, 'quantization_method') and model.quantization_method == 'bitsandbytes':
            precision = "4-bit (quantized)"
        else:
            precision = "16-bit (fallback)"


    return {
        "model_name": "llama-3-8b-instruct-travel",
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "adapter_type": "LoRA",
        "parameters": "8B",
        "precision": precision,
        "api_mode": API_MODE,
        "device": str(next(model.parameters()).device),
        "loaded_at": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "llama_api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    ) 