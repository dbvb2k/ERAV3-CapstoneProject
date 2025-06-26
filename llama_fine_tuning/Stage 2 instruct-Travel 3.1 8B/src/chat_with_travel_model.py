#!/usr/bin/env python3
"""
🚀 Interactive Travel Model Chat Interface
Load your fine-tuned travel model and chat with it!
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
import os

def load_travel_model():
    """Load the fine-tuned travel model"""
    print("🔄 Loading your fine-tuned travel model...")
    
    travel_model_path = "./COMPLETE_TRAVEL_MODEL"
    
    # Check if travel model exists
    if not os.path.exists(travel_model_path):
        print(f"❌ Travel model not found at {travel_model_path}")
        print("Make sure you're in the correct directory with COMPLETE_TRAVEL_MODEL/")
        return None, None
    
    try:
        # Load tokenizer directly from the travel model
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(travel_model_path)
        
        # Load the complete fine-tuned model directly
        print("🧠 Loading your fine-tuned travel model...")
        model = AutoModelForCausalLM.from_pretrained(
            travel_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Travel model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure COMPLETE_TRAVEL_MODEL/ directory exists")
        print("2. Check if you have enough GPU memory")
        print("3. Verify the model files are complete")
        return None, None

def generate_response(model, tokenizer, question, max_length=512):
    """Generate response from the travel model"""
    
    # Better prompt formatting for more specific responses
    chat_input = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{question.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    
    # Tokenize
    inputs = tokenizer(chat_input, return_tensors="pt", truncation=True, max_length=768)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response with better parameters for specificity
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.3,  # Lower temperature for more focused responses
            do_sample=True,
            top_p=0.8,        # More focused sampling
            top_k=50,         # Limit vocabulary choices
            repetition_penalty=1.2,  # Stronger penalty for repetition
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3  # Prevent repetitive patterns
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        response = response.replace("<|eot_id|>", "").strip()
    else:
        response = full_response[len(chat_input):].strip()
    
    return response

def main():
    """Main chat interface"""
    print("🌍 Welcome to Your Personal Travel Assistant!")
    print("=" * 50)
    print("Your fine-tuned Llama 3 8B model specialized for Indian travel")
    print("Ask me anything about travel, destinations, planning, etc.")
    print("Type 'quit', 'exit', or 'bye' to stop")
    print("=" * 50)
    
    # Load the model
    model, tokenizer = load_travel_model()
    if model is None:
        print("❌ Failed to load model. Exiting...")
        return
    
    print("\n🚀 Ready to help with your travel questions!")
    print("💡 Try asking about: destinations, itineraries, costs, bookings, cultural tips")
    
    while True:
        try:
            # Get user input
            print("\n" + "="*50)
            question = input("🧳 Your travel question: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'bye', 'q']:
                print("👋 Happy travels! Goodbye!")
                break
                
            if not question:
                print("Please ask a question!")
                continue
            
            # Generate response
            print("\n🤖 Travel Assistant:")
            print("-" * 30)
            response = generate_response(model, tokenizer, question)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Happy travels! Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main() 