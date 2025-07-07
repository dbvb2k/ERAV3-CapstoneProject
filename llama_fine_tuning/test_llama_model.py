#!/usr/bin/env python3
"""
Test script for Llama model loading and inference
"""

import os
import sys
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if the model can be loaded successfully"""
    try:
        logger.info("Testing model loading...")
        
        # Check if model directory exists
        model_path = "model/COMPLETE_TRAVEL_MODEL"
        if not os.path.exists(model_path):
            logger.error(f"Model directory not found: {model_path}")
            return False
        
        # Load PEFT configuration
        logger.info("Loading PEFT configuration...")
        peft_config = PeftConfig.from_pretrained(model_path)
        logger.info(f"Base model: {peft_config.base_model_name_or_path}")
        
        # Test tokenizer loading
        logger.info("Testing tokenizer loading...")
        try:
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
            logger.info("Tokenizer loaded successfully")
            
            # Test basic tokenization
            test_text = "Hello, how are you?"
            tokens = tokenizer(test_text, return_tensors="pt")
            logger.info(f"Test tokenization successful: {tokens.input_ids.shape}")
            
        except Exception as e:
            logger.error(f"Tokenizer loading failed: {str(e)}")
            return False
        
        # Test base model loading (this will download if not present)
        logger.info("Testing base model loading...")
        try:
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
            logger.info("Base model loaded successfully")
            
            # Test LoRA adapter loading
            logger.info("Testing LoRA adapter loading...")
            model = PeftModel.from_pretrained(base_model, model_path)
            logger.info("LoRA adapter loaded successfully")
            
            # Test basic inference
            logger.info("Testing basic inference...")
            test_prompt = "Tell me about travel to Paris"
            inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Move inputs to the same device as the model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logger.info(f"Model device: {device}")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            logger.info(f"Generated text: {generated_text[:200]}...")
            
            logger.info("✅ All tests passed! Model is working correctly.")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    logger.info("Starting Llama model tests...")
    
    # Check HF token
    if not settings.HF_TOKEN:
        logger.warning("No HF_TOKEN found in environment. You may need to set it in your .env file.")
        logger.info("You can get your HF token from: https://huggingface.co/settings/tokens")
    
    success = test_model_loading()
    
    if success:
        logger.info("🎉 All tests passed! The model is ready for use.")
        sys.exit(0)
    else:
        logger.error("❌ Tests failed! Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 