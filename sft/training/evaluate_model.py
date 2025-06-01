#!/usr/bin/env python3
"""
Evaluate Llama 3 8B Travel Assistant Model Efficiency
"""

import os
import torch
import time
import argparse
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Llama 3 Travel Assistant Model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model folder (relative to models directory)')
    return parser.parse_args()

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint folder within the main checkpoint directory"""
    try:
        # First check if the directory exists
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        print(f"\nChecking checkpoint directory: {checkpoint_dir}")
        print("Directory contents:")
        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(item_path):
                print(f"Directory: {item}")
                print("  Contents:")
                try:
                    for subitem in os.listdir(item_path):
                        print(f"    {subitem}")
                except Exception as e:
                    print(f"    Error listing contents: {str(e)}")
            else:
                print(f"File: {item}")
        
        # List all items in the directory
        items = os.listdir(checkpoint_dir)
        
        # Look for checkpoint folders
        checkpoint_folders = [d for d in items if d.startswith('checkpoint-')]
        
        if not checkpoint_folders:
            # If no checkpoint folders found, check if this is a model directory itself
            if os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")) or \
               os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin")):
                print(f"Using model directory directly: {checkpoint_dir}")
                return checkpoint_dir
            raise ValueError(f"No checkpoint folders or model files found in {checkpoint_dir}")
        
        # Extract step numbers and find the latest
        latest_checkpoint = max(checkpoint_folders, key=lambda x: int(x.split('-')[1]))
        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        print(f"\nExamining latest checkpoint: {latest_checkpoint}")
        print("Checkpoint contents:")
        for item in os.listdir(latest_checkpoint_path):
            print(f"  {item}")
        
        # Check for various possible model file names
        possible_model_files = [
            "pytorch_model.bin",
            "adapter_model.bin",
            "model.safetensors",
            "adapter_model.safetensors",
            "training_args.bin",
            "optimizer.pt",
            "scheduler.pt"
        ]
        
        found_files = []
        for file in possible_model_files:
            if os.path.exists(os.path.join(latest_checkpoint_path, file)):
                found_files.append(file)
        
        if not found_files:
            raise ValueError(f"Checkpoint {latest_checkpoint} does not contain any model files")
        
        print(f"\nFound model files: {', '.join(found_files)}")
        return latest_checkpoint_path
        
    except Exception as e:
        print(f"Error finding checkpoint: {str(e)}")
        raise

def load_model_and_tokenizer(model_path=None):
    """Load either the base model or fine-tuned model"""
    
    USE_BASE_MODEL = True

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
    
    # Load tokenizer and model
    if model_path:
        # Ensure the model path is within the models directory
        full_model_path = os.path.join("models", model_path)
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Model directory not found: {full_model_path}")
        
        # Find the latest checkpoint
        latest_checkpoint = find_latest_checkpoint(full_model_path)
        print(f"\nLoading model from latest checkpoint: {latest_checkpoint}")
            
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=latest_checkpoint,
                max_seq_length=2048,
                dtype=torch.float16,
                load_in_4bit=True
            )
        except Exception as e:
            print(f"Error loading model with Unsloth: {str(e)}")
            print("Falling back to standard model loading...")
            tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)
            model = FastLanguageModel.from_pretrained(
                model_name=latest_checkpoint,
                max_seq_length=2048,
                dtype=torch.float16,
                load_in_4bit=True
            )
    else:
        if USE_BASE_MODEL:
            print("Loading Llama 3 8B Base model with Unsloth: meta-llama/Llama-3.1-8B")
            model_name = "meta-llama/Llama-3.1-8B"
        else:
            print("Loading Llama 3 8B Instruct model with Unsloth: NousResearch/Meta-Llama-3-8B-Instruct")
            model_name = "NousResearch/Meta-Llama-3-8B-Instruct"
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
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
    # Sample prompts for evaluation
    test_prompts = [
        "What are the best places to visit in Paris?",
        "How do I plan a budget trip to Japan?",
        "What's the best time to visit Bali?",
        "Can you suggest a 3-day itinerary for New York City?",
        "What are some must-try local foods in Thailand?"
    ]
    
    print("\n=== Base Model Evaluation ===")
    print("Model: Llama 3 8B Base")
    
    # Load base model and tokenizer
    base_model, base_tokenizer = load_model_and_tokenizer(None)
    print(f"Device: {base_model.device}")
    print(f"Number of parameters: {base_model.num_parameters():,}")
    
    total_time = 0
    print("\nGenerating responses for test prompts using base model...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        response, gen_time = generate_response(base_model, base_tokenizer, prompt)
        total_time += gen_time
        
        print(f"Generation time: {gen_time:.2f} seconds")
        print("\nResponse:")
        print("-" * 80)
        print(response)
        print("-" * 80)
    
    avg_time = total_time / len(test_prompts)
    print(f"\nAverage generation time for base model: {avg_time:.2f} seconds")
    
    # Memory usage for base model
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage (Base Model):")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    # Clear GPU memory
    del base_model
    del base_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # If model path is provided, evaluate the fine-tuned model
    if model_path:
        print("\n=== Fine-tuned Model Evaluation ===")
        print(f"Model: Fine-tuned Llama 3 8B from {model_path}")
        
        # Load fine-tuned model and tokenizer
        ft_model, ft_tokenizer = load_model_and_tokenizer(model_path)
        print(f"Device: {ft_model.device}")
        print(f"Number of parameters: {ft_model.num_parameters():,}")
        
        total_time = 0
        print("\nGenerating responses for test prompts using fine-tuned model...")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nPrompt {i}: {prompt}")
            response, gen_time = generate_response(ft_model, ft_tokenizer, prompt)
            total_time += gen_time
            
            print(f"Generation time: {gen_time:.2f} seconds")
            print("\nResponse:")
            print("-" * 80)
            print(response)
            print("-" * 80)
        
        avg_time = total_time / len(test_prompts)
        print(f"\nAverage generation time for fine-tuned model: {avg_time:.2f} seconds")
        
        # Memory usage for fine-tuned model
        if torch.cuda.is_available():
            print(f"\nGPU Memory Usage (Fine-tuned Model):")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        # Clear GPU memory
        del ft_model
        del ft_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main evaluation pipeline"""
    print("=== Llama 3 8B Travel Assistant Model Evaluation ===")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if models directory exists
    if not os.path.exists("models"):
        print("⚠️  Models directory not found. Creating it...")
        os.makedirs("models", exist_ok=True)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  No GPU detected - evaluation will be slow")
    
    try:
        # Evaluate both base and fine-tuned models
        evaluate_model(args.model_path)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main() 