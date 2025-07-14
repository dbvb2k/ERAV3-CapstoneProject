import os
import json
import traceback
import uuid
import time
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
    max_length: Optional[int] = Field(2048, description="Maximum length of generated response")
    temperature: Optional[float] = Field(0.1, description="Temperature for generation")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter")
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
    num_return_sequences: Optional[int] = Field(1, description="Number of sequences to return")
    tools: Optional[List[Dict[str, Any]]] = Field([], description="Tools/functions the model can call")
    tool_choice: Optional[str] = Field("auto", description="How tools should be chosen")
    
    
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


def get_model_precision_and_kwargs(gpu_available: bool, bnb_gpu_support: bool) -> dict:
    """
    Determines model loading kwargs based on API_MODE and hardware support.
    Returns a plain dict suitable for from_pretrained().
    """
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
            model_kwargs["torch_dtype"] = torch.float32
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
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif gpu_available:
            logger.warning("4-bit not supported, falling back to 16-bit precision on GPU.")
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
        else:
            logger.info("No GPU available, using CPU mode.")
            model_kwargs["device_map"] = "cpu"
            model_kwargs["torch_dtype"] = torch.float32

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
            os.path.join(peft_config.base_model_name_or_path if peft_config.base_model_name_or_path is not None else "", "config.json"),
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
        # from transformers import PreTrainedModel

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

def extract_tool_calls(text, tools):
    """Extract tool calls from generated text using a more robust generic approach"""
    import re
    import json
    import uuid
    
    # Add debug output with truncated text to avoid huge logs
    logger.info(f"Attempting to extract tool calls from text: {text[:200]}...")
    
    # Initialize the result
    processed_calls = []
    
    # APPROACH 1: Try to parse the entire object first
    try:
        # Clean up the text - remove any text before { or after }
        json_text = re.search(r'(\{[\s\S]*\})', text)
        if json_text:
            clean_text = json_text.group(1)
            data = json.loads(clean_text)
            
            # If we have a valid tool_calls array, process it
            if "tool_calls" in data and isinstance(data["tool_calls"], list):
                for i, call in enumerate(data["tool_calls"]):
                    if "type" in call and "function" in call and "name" in call["function"]:
                        processed_call = {
                            "index": i,
                            "id": call.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": json.dumps(call["function"]["arguments"]) 
                                    if isinstance(call["function"].get("arguments"), dict) 
                                    else call["function"].get("arguments", "{}")
                            }
                        }
                        processed_calls.append(processed_call)
                
                if processed_calls:
                    logger.info(f"Successfully parsed complete JSON with {len(processed_calls)} tool calls")
                    return processed_calls
    except Exception as e:
        logger.info(f"Complete JSON parsing failed: {e}")
    
    # APPROACH 2: Extract individual tool calls with a generic pattern
    # This pattern looks for all tool call structures in the text
    tool_pattern = r'"function"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[\s\S]*?(?:\}\s*\}\s*\}|\}\s*\}))'
    tool_matches = re.findall(tool_pattern, text)
    
    if tool_matches:
        logger.info(f"Found {len(tool_matches)} potential tool calls using generic pattern")
        
        for i, (name, args_json) in enumerate(tool_matches):
            try:
                # Clean up the arguments JSON
                # First, find the outermost balanced braces
                brace_count = 0
                end_pos = 0
                
                for pos, char in enumerate(args_json):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = pos + 1
                            break
                
                # Extract just the balanced JSON object
                if end_pos > 0:
                    args_json = args_json[:end_pos]
                
                # Remove trailing commas before closing braces (common JSON error)
                args_json = re.sub(r',\s*}', '}', args_json)
                
                # Try to parse the arguments
                args_dict = json.loads(args_json)
                
                # Create a properly formatted tool call
                processed_calls.append({
                    "index": i,
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args_dict)
                    }
                })
                logger.info(f"Successfully parsed tool call for {name}")
            except json.JSONDecodeError as e:
                logger.info(f"Failed to parse arguments for {name}: {e}")
                
                # Special handling for common patterns - last resort
                if name in [tool["function"]["name"] for tool in tools if "function" in tool]:
                    # Extract key patterns based on the tool's required parameters
                    tool_info = next((t for t in tools if "function" in t and t["function"]["name"] == name), None)
                    
                    if tool_info:
                        try:
                            # Try to extract required parameters as key-value pairs
                            params = {}
                            required_params = tool_info["function"]["parameters"].get("required", [])
                            
                            # For each required parameter, try to find it in the text
                            for param in required_params:
                                param_pattern = fr'"{param}"\s*:\s*(?:"([^"]+)"|(\d+))'
                                param_match = re.search(param_pattern, args_json)
                                if param_match:
                                    # Get either string or numeric value
                                    value = param_match.group(1) if param_match.group(1) is not None else param_match.group(2)
                                    params[param] = value if not value.isdigit() else int(value)
                            
                            # Only add if we found all required parameters
                            if all(param in params for param in required_params):
                                processed_calls.append({
                                    "index": i,
                                    "id": f"call_{uuid.uuid4().hex[:24]}",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(params)
                                    }
                                })
                                logger.info(f"Reconstructed basic parameters for {name}")
                        except Exception as ex:
                            logger.info(f"Failed to extract basic parameters for {name}: {ex}")
    
    # Return any tool calls we found, or None if we found none
    if processed_calls:
        logger.info(f"Successfully extracted {len(processed_calls)} tool calls")
        return processed_calls
    
    # APPROACH 3: Look for individual tools by name as a last resort
    if not processed_calls:
        for tool in tools:
            if "function" in tool:
                name = tool["function"]["name"]
                pattern = fr'"name"\s*:\s*"{re.escape(name)}"[\s\S]*?"arguments"\s*:\s*\{{([^{{}}]+)\}}'
                match = re.search(pattern, text)
                
                if match:
                    try:
                        # Clean and parse arguments
                        args_str = "{" + match.group(1) + "}"
                        args_str = re.sub(r',\s*}', '}', args_str)
                        args_dict = json.loads(args_str)
                        
                        processed_calls.append({
                            "index": len(processed_calls),
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args_dict)
                            }
                        })
                        logger.info(f"Extracted tool call for {name} using name-based pattern")
                    except json.JSONDecodeError:
                        logger.info(f"Failed to parse simple arguments for {name}")
    
    return processed_calls if processed_calls else None

## Format tool prompt and send it to the generate method in main function
def format_tool_prompt(messages, tools):
    """Generate response with tool calling"""
    try:
        system_msg = ""
        prompt = ""
        
        # Extract system message if present
        for msg in messages:
            if msg["role"] == "system":
                system_msg += msg["content"]
                break
        
        # Append tool definitions to the prompt to guide the model
        tools_description = "You have access to the following tools:\n"
        for tool in tools:
            if "function" in tool:
                func = tool["function"]
                tools_description += f"- {func['name']}: {func['description']}\n"
                
        # In the generate_tool_response function, replace the instruction variable with:
        instruction = """IMPORTANT: To use tools, you must provide a PROPER JSON response with tool_calls array.
            TOOL CALLING EXAMPLES:
            - FlightSearchTool: {"origin": "Mumbai", "destination": "Bangkok", "depart_date": "2025-08-20", "return_date": "2025-08-27", "adults": 2}
            - HotelSearchTool: {"location": "Bangkok", "check_in": "2025-08-20", "check_out": "2025-08-27", "occupancy": 2}
            - WeatherTool: {"location": "Bangkok"}
            - ItineraryPlannerTool: {"location": "Bangkok", "duration": 7}}
            YOUR RESPONSE MUST BE A LIST OF VALID JSON OBJECTS:
            {
            "tool_calls": [
                {
                "type": "function",
                "function": {
                    "name": "ItineraryPlannerTool",
                    "arguments": {
                    "location": "Bangkok", 
                    "duration": 7
                    }
                }
                },
                {
                "type": "function",
                "function": {
                    "name": "HotelSearchTool",
                    "arguments": {
                    "location": "Bangkok",
                    "check_in": "2026-01-20",
                    "check_out": "2026-01-27",
                    "occupancy": 4
                    }
                }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "WeatherTool",
                        "arguments": {
                            "location": "Bangkok"
                        }
                    }
                }
                
            ]
            }
            FINAL INSTRUCTIONS:
            - YOU MUST CALL "ItineraryPlannerTool" atleast once. This is MANDATORY. 
            - IF THE TOOLS ARE ALREADY CALLED, STRICTLY DO NOT CALL THEM AGAIN. They might be there in text as "<coroutine object flight_search at 0x000001B2C31C6820>". If you encounter this means this tool is already called, do not call it again.
            - MENTION ALL THE TOOL CALLS INSIDE SINGLE JSON ARRAY of "tool_calls"
            - DO NOT write any text before or after the JSON.
            - DO NOT use markdown code blocks.
            """
        # Build prompt with system message and tools and instructions first
        system_msg = system_msg + tools_description + instruction
        if system_msg:
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
        
        # Add other messages
        for msg in messages:
            if msg["role"] != "system":  # Skip system message as we already added it
                if msg["role"] == "user":
                    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
                elif msg["role"] == "assistant":
                    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['content']}<|eot_id|>"

        # Add assistant header for response
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n" 
        
        return prompt
        
    except Exception as e:
        logger.error(f"Error generating tool prompt: {str(e)}")
        logger.error(traceback.format_exc())
        # Return a basic error response
        return prompt
    
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
        openapi_url=app.openapi_url or "/openapi.json",
        title=app.title + " - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

## Add functional tool calling capability here
@app.post("/v1/chat/completions", response_model=Dict[str, Any])
async def chat_completion(request: ChatRequest):
    """
    Generate chat completion using the fine-tuned Llama model
    with support for function calling/tools
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
        
    #logger.info(f"Request JSON: {request.model_dump()}")
    
    try:
        # Format chat messages
        formatted_messages = []
        for msg in request.messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})
        
        # Generate response
        temperature = request.temperature if request.temperature is not None else 0.1
        
        # Handle tool calls if tools are provided
        use_tool_calling = request.tools is not None and len(request.tools) > 0
        
        logger.info(f"Chat request with tool calling: {use_tool_calling}")
        # if use_tool_calling:
        #     logger.info(f"Tools: {json.dumps(request.tools, indent=2)}")
            
        # Process with tools if provided
        if use_tool_calling:
            # Call the model with tools
            prompt = format_tool_prompt(
                formatted_messages, 
                request.tools
            )
        else:
            # Normal chat completion without tools
            #logger.info(f"Request JSON: {request.model_dump()}")
            prompt = format_chat_prompt(request.messages)
            
        ## Generate text using model
        with torch.no_grad():
            # Tokenize input for length calculation
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_length = inputs.input_ids.shape[1]
            
            # Move inputs to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            outputs = model.generate(
                **inputs,
                max_length=input_length + request.max_length,
                temperature=temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                num_return_sequences=request.num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        if use_tool_calling:
            tool_calls = extract_tool_calls(generated_text, request.tools)
        else:
            tool_calls = None
        
        # Build response safely
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "llama-3-8b-instruct-travel",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None if tool_calls else generated_text.strip()
                    },
                    "finish_reason": "tool_calls" if tool_calls else "stop"
                }
            ],
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": len(outputs[0]) - input_length,
                "total_tokens": len(outputs[0])
            }
        }

        # Add tool_calls safely if they exist
        if tool_calls:
            response["choices"][0]["message"]["tool_calls"] = tool_calls
            logger.info(f"Added {len(tool_calls)} tool calls to response")
        else:
            # Ensure tool_calls is not None - should be empty list or not present
            response["choices"][0]["message"]["tool_calls"] = []
            
        return response
                
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        logger.error(traceback.format_exc())
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