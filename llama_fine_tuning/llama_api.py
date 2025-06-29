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
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

logger.info("Initializing Llama API service...")

app = FastAPI(
    title="Llama Travel Model API",
    description="API for inference using fine-tuned Llama 3 8B Instruct model for travel-related tasks",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Global variables for model and tokenizer
model = None
tokenizer = None
generator = None

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

def load_model():
    """Load the fine-tuned Llama model with LoRA adapter"""
    global model, tokenizer, generator
    
    try:
        logger.info("Loading model configuration...")
        
        # Load PEFT configuration
        peft_config = PeftConfig.from_pretrained("model/COMPLETE_TRAVEL_MODEL")
        logger.info(f"Loaded PEFT config: {peft_config.base_model_name_or_path}")
        
        # Load tokenizer with HF token if available
        logger.info("Loading tokenizer...")
        tokenizer_kwargs = {
            "trust_remote_code": True,
            "use_fast": False
        }
        
        # Add HF token if configured
        if settings.HF_TOKEN:
            tokenizer_kwargs["token"] = settings.HF_TOKEN
            logger.info("Using HF token from environment")
        else:
            logger.warning("No HF token found in environment. Make sure you have access to Llama models.")
        
        tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            **tokenizer_kwargs
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model with quantization for memory efficiency
        logger.info("Loading base model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16
        }
        
        # Add HF token if configured
        if settings.HF_TOKEN:
            model_kwargs["token"] = settings.HF_TOKEN
        
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
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
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
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

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting up Llama API service...")
    success = load_model()
    if not success:
        logger.error("Failed to load model during startup")

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
    
    return {
        "model_name": "llama-3-8b-instruct-travel",
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "adapter_type": "LoRA",
        "parameters": "8B",
        "quantization": "4-bit",
        "device": str(next(model.parameters()).device),
        "loaded_at": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "llama_api:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    ) 