#!/usr/bin/env python3
"""
🔍 Test script to verify model identity and capabilities
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def test_model():
    """Test if the model is actually fine-tuned"""
    
    travel_model_path = "./COMPLETE_TRAVEL_MODEL"
    
    if not os.path.exists(travel_model_path):
        print("❌ COMPLETE_TRAVEL_MODEL not found")
        return
    
    print("🔍 Testing model identity...")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(travel_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        travel_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Test questions
    test_questions = [
        "Plan a trip to Kerala",
        "What to see in Copenhagen", 
        "Best time to visit Rajasthan",
        "Budget for Goa trip"
    ]
    
    print("\n🧪 Testing responses...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i}: {question} ---")
        
        # Format input
        chat_input = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        
        # Tokenize and generate
        inputs = tokenizer(chat_input, return_tensors="pt", truncation=True, max_length=768)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                top_p=0.8,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            response = response.replace("<|eot_id|>", "").strip()
        else:
            response = full_response[len(chat_input):].strip()
        
        print(f"Response: {response[:300]}...")
        
        # Check for travel-specific indicators
        travel_indicators = ['₹', 'rupees', 'INR', 'India', 'Indian', 'temple', 'monsoon', 'festival']
        found_indicators = [ind for ind in travel_indicators if ind.lower() in response.lower()]
        
        print(f"Travel indicators found: {found_indicators}")
        print("-" * 50)

if __name__ == "__main__":
    test_model() 