#!/usr/bin/env python3
"""
Evaluate Llama 3 8B Travel Assistant Model Efficiency
"""

import os
import torch
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Llama 3 Travel Assistant Model')
    parser.add_argument('model_path', type=str, help='Path to the trained model folder')
    return parser.parse_args()

def load_model_and_tokenizer(model_path=None):
    """Load either the base model or fine-tuned model"""
    print("\nLoading model and tokenizer...")
    
    # Get HuggingFace token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is not set. Please set your HuggingFace token using:\n"
            "1. Create a .env file with: HF_TOKEN=your_token_here\n"
            "2. Or set it in your environment: export HF_TOKEN=your_token_here\n"
            "You can get your token from: https://huggingface.co/settings/tokens"
        )
    
    # Load tokenizer
    if model_path:
        print(f"Loading fine-tuned model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        print("Loading base model: NousResearch/Meta-Llama-3-8B-Instruct")
        # print("Loading base model: meta-llama/Meta-Llama-3.1-8B-Instruct")

        model_name = "NousResearch/Meta-Llama-3-8B-Instruct"
        #   model_name = "meta-llama/Llama-3.1-8B"  
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
    
    return model, tokenizer

def format_prompt(user_input):
    """Format the prompt in the correct format for Llama 3"""
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful travel assistant. You help users with travel planning, booking accommodations, finding restaurants, transportation, and providing travel information. Be friendly, informative, and helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def generate_response(model, tokenizer, prompt, max_length=2048):
    """Generate response and measure performance"""
    # Format the prompt
    formatted_prompt = format_prompt(prompt)
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Measure generation time
    start_time = time.time()
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            min_length=100,  # Ensure minimum response length
            max_new_tokens=1024  # Allow for longer new tokens
        )
    
    generation_time = time.time() - start_time
    
    # Decode and clean up response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    try:
        # Find the last assistant header
        assistant_start = full_response.rfind("<|start_header_id|>assistant<|end_header_id|>")
        if assistant_start != -1:
            # Get everything after the last assistant header
            response = full_response[assistant_start:].split("<|eot_id|>")[0]
            # Remove the assistant header
            response = response.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
        else:
            response = full_response
    except:
        response = full_response
    
    return response, generation_time

def evaluate_model(model_path=None):
    """Evaluate model performance on sample prompts"""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Sample prompts for evaluation
    test_prompts = [
        "What are the best places to visit in Paris?",
        "How do I plan a budget trip to Japan?",
        "What's the best time to visit Bali?",
        "Can you suggest a 3-day itinerary for New York City?",
        "What are some must-try local foods in Thailand?"
    ]
    
    print("\n=== Model Evaluation ===")
    print(f"Model: {'Fine-tuned' if model_path else 'Base'} Llama 3 8B")
    print(f"Device: {model.device}")
    print(f"Number of parameters: {model.num_parameters():,}")
    
    total_time = 0
    print("\nGenerating responses for test prompts...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        response, gen_time = generate_response(model, tokenizer, prompt)
        total_time += gen_time
        
        print(f"Generation time: {gen_time:.2f} seconds")
        print("\nResponse:")
        print("-" * 80)
        print(response)
        print("-" * 80)
    
    avg_time = total_time / len(test_prompts)
    print(f"\nAverage generation time: {avg_time:.2f} seconds")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

def main():
    """Main evaluation pipeline"""
    print("=== Llama 3 8B Travel Assistant Model Evaluation ===")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  No GPU detected - evaluation will be slow")
    
    try:
        # Evaluate base model
        print("\nEvaluating base model...")
        evaluate_model()
        
        # Evaluate fine-tuned model
        print(f"\nEvaluating fine-tuned model: {args.model_path}")
        evaluate_model(args.model_path)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main() 