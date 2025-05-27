#!/usr/bin/env python3
"""
Test script for the fine-tuned Llama 3 model
"""

import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
import os
import glob

def load_trained_model(model_dir):
    """Load the fine-tuned model"""
    print(f"Loading model from {model_dir}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def generate_response(model, tokenizer, user_input, max_length=512):
    """Generate a response for the given user input"""
    
    # Format the input for Llama 3 Instruct model
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful travel assistant. You help users with travel planning, booking accommodations, finding restaurants, transportation, and providing travel information. Be friendly, informative, and helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # Tokenize the input
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Create text streamer for real-time output
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=max_length,
            use_cache=True,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    assistant_start = response.find("<|start_header_id|>assistant<|end_header_id|>\n\n") + len("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assistant_response = response[assistant_start:].strip()
    
    # Clean up any remaining special tokens
    assistant_response = assistant_response.replace("<|eot_id|>", "").strip()
    
    return assistant_response

def interactive_chat(model, tokenizer):
    """Interactive chat with the travel assistant model"""
    print("\n=== Interactive Chat with Llama 3 Travel Assistant ===")
    print("Ask me about travel planning, hotels, restaurants, transportation, and more!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("User: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("Assistant: ", end="")
        response = generate_response(model, tokenizer, user_input)
        print()  # Add a newline after the response

def main():
    """Main function to test the model"""
    
    # Find the most recent travel assistant model directory
    model_dirs = glob.glob("./llama3_travel_assistant_*_merged")
    
    if not model_dirs:
        # If no merged model, look for inference model
        model_dirs = glob.glob("./llama3_travel_assistant_*_inference")
    
    if not model_dirs:
        # If no inference model, look for regular training output
        model_dirs = glob.glob("./llama3_travel_assistant_*")
        model_dirs = [d for d in model_dirs if not d.endswith(('_inference', '_merged'))]
    
    if not model_dirs:
        print("No trained model found!")
        print("Please run train_llama3.py first to train a model.")
        return
    
    # Use the most recent model
    latest_model = max(model_dirs, key=os.path.getctime)
    print(f"Using model: {latest_model}")
    
    try:
        # Load the model
        model, tokenizer = load_trained_model(latest_model)
        
        # Test with a few example inputs
        test_inputs = [
            "I need help booking a train from Cambridge to London.",
            "Can you help me find a restaurant in the city center?",
            "I'm looking for a hotel with parking facilities.",
        ]
        
        print("\n=== Testing with example inputs ===")
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\nTest {i}:")
            print(f"User: {test_input}")
            print("Assistant: ", end="")
            response = generate_response(model, tokenizer, test_input)
            print()
        
        # Start interactive chat
        interactive_chat(model, tokenizer)
        
    except Exception as e:
        print(f"Error loading or testing model: {e}")
        raise

if __name__ == "__main__":
    main() 