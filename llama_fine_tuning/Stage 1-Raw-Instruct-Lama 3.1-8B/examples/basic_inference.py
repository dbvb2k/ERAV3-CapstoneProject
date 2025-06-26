#!/usr/bin/env python3
"""
Basic Inference Example for Llama Travel Model
==============================================

Simple example showing how to use the minimal travel model for inference.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_travel_model(model_path="./model"):
    """Load the travel model for inference"""
    print("🚀 Loading travel model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✅ Tokenizer loaded")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Meta-Llama-3-8B-Instruct',
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True
    )
    print("✅ Base model loaded")
    
    # Load fine-tuned adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    print("✅ Travel adapter loaded")
    
    return model, tokenizer

def generate_travel_advice(model, tokenizer, prompt, max_tokens=200, temperature=0.7):
    """Generate travel advice for a given prompt"""
    
    # Format prompt for Llama-3
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(formatted_prompt, "").strip()
    
    return response

def main():
    """Main function demonstrating basic usage"""
    
    # Example travel queries
    travel_queries = [
        "What are the best budget destinations in Southeast Asia?",
        "How should I prepare for a 2-week trip to Japan?",
        "What are some safety tips for solo female travelers in Europe?",
        "Best time to visit the Maldives for good weather and lower prices?",
        "How can I experience authentic local culture while traveling?"
    ]
    
    try:
        # Load model
        model, tokenizer = load_travel_model()
        
        print("\n🧳 Travel Assistant Ready! Let's answer some questions...\n")
        print("=" * 60)
        
        # Generate responses for each query
        for i, query in enumerate(travel_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 50)
            
            response = generate_travel_advice(model, tokenizer, query)
            print(f"🤖 Travel Assistant: {response}")
            
            print("\n" + "=" * 60)
        
        print("\n✅ Demo completed! Try your own queries by modifying the script.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Downloaded the minimal model to ./model/ directory")
        print("   2. Installed required packages: pip install transformers torch peft")
        print("   3. Have sufficient GPU memory (24GB+ recommended)")

if __name__ == "__main__":
    main() 