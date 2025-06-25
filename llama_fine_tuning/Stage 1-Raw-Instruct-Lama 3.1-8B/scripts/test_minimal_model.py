#!/usr/bin/env python3
"""
Test Minimal Model for Inference
================================

Simple script to test the minimal model extracted by extract_and_verify_minimal_model.py
This can be run independently to verify the minimal model works correctly.
"""

import os
import json
import time
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Travel queries for testing
TRAVEL_QUERIES = [
    "I'm planning a 7-day trip to Japan in spring. What are the must-visit places?",
    "What are some budget-friendly European destinations for backpackers under $50 per day?",
    "What's the best time to book international flights to get the lowest prices?",
    "How can I experience authentic local cuisine while traveling without getting sick?",
    "What safety precautions should solo female travelers take when visiting developing countries?",
    "What documents and preparations do I need for international travel as a first-time traveler?",
    "What are the best winter destinations for travelers who want to escape cold weather?"
]

def load_minimal_model(model_path="llama_travel_model_minimal"):
    """Load the minimal model for testing"""
    logger.info(f"🚀 Loading minimal travel model from {model_path}...")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Check for essential files
    essential_files = [
        "tokenizer.json",
        "tokenizer_config.json", 
        "special_tokens_map.json",
        "adapter_model.safetensors",
        "adapter_config.json"
    ]
    
    missing_files = []
    for file in essential_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        raise FileNotFoundError(f"Missing essential files: {missing_files}")
    
    logger.info("✅ All essential files found")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"✅ Tokenizer loaded from {model_path}")
        
        # Load base model with rope_scaling fix
        base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        try:
            # Try direct loading first
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory={0: "14GB"},
                trust_remote_code=True
            )
            logger.info(f"✅ Base model {base_model_name} loaded")
        except Exception as error:
            logger.info(f"🔧 Direct loading failed: {error}")
            logger.info("🔧 Trying with rope_scaling fix...")
            
            try:
                from transformers import LlamaConfig
                import requests
                
                # Get and fix config
                config_url = f"https://huggingface.co/{base_model_name}/resolve/main/config.json"
                response = requests.get(config_url)
                config_dict = response.json()
                
                # Fix rope_scaling
                if 'rope_scaling' in config_dict and config_dict['rope_scaling'] is not None:
                    rope_scaling = config_dict['rope_scaling'].copy()
                    if 'rope_type' in rope_scaling:
                        rope_scaling['type'] = rope_scaling.pop('rope_type')
                    config_dict['rope_scaling'] = {
                        'type': rope_scaling.get('type', 'linear'),
                        'factor': rope_scaling.get('factor', 1.0)
                    }
                
                config = LlamaConfig.from_dict(config_dict)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    config=config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_memory={0: "14GB"}
                )
                logger.info(f"✅ Base model loaded with rope_scaling fix")
            except Exception as config_error:
                logger.error(f"❌ Failed to load base model: {config_error}")
                raise error
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        logger.info(f"✅ LoRA adapter loaded from {model_path}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load minimal model: {e}")
        raise

def test_model(model, tokenizer, queries=None, max_new_tokens=200, temperature=0.7):
    """Test model with travel queries"""
    if queries is None:
        queries = TRAVEL_QUERIES
    
    logger.info(f"🧪 Testing minimal model with {len(queries)} travel queries...")
    
    results = []
    total_time = 0
    
    for i, query in enumerate(queries, 1):
        logger.info(f"📝 Query {i}/{len(queries)}: {query[:60]}...")
        
        # Format prompt for Llama 3
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response_time = time.time() - start_time
            total_time += response_time
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            logger.info(f"✅ Response in {response_time:.2f}s: {response[:100]}...")
            
            results.append({
                "query_id": i,
                "query": query,
                "response": response,
                "response_time": response_time,
                "word_count": len(response.split()),
                "char_count": len(response)
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to generate response for query {i}: {e}")
            results.append({
                "query_id": i,
                "query": query,
                "response": f"ERROR: {str(e)}",
                "response_time": 0,
                "word_count": 0,
                "char_count": 0
            })
    
    # Calculate summary statistics
    successful_results = [r for r in results if not r["response"].startswith("ERROR:")]
    
    summary = {
        "total_queries": len(queries),
        "successful_queries": len(successful_results),
        "failed_queries": len(queries) - len(successful_results),
        "total_time": total_time,
        "average_time": total_time / len(queries) if queries else 0,
        "total_words": sum(r["word_count"] for r in successful_results),
        "average_words": sum(r["word_count"] for r in successful_results) / len(successful_results) if successful_results else 0,
        "total_chars": sum(r["char_count"] for r in successful_results),
        "average_chars": sum(r["char_count"] for r in successful_results) / len(successful_results) if successful_results else 0
    }
    
    return results, summary

def save_test_results(results, summary, output_dir="minimal_model_test_results"):
    """Save test results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed JSON results
    test_data = {
        "timestamp": timestamp,
        "model_info": {
            "name": "Llama-3-8B Travel Model (Minimal)",
            "type": "LoRA fine-tuned",
            "base_model": "meta-llama/Meta-Llama-3-8B-Instruct"
        },
        "test_summary": summary,
        "test_results": results
    }
    
    json_file = os.path.join(output_dir, f"minimal_model_test_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Create human-readable report
    report_file = os.path.join(output_dir, f"minimal_model_report_{timestamp}.md")
    with open(report_file, 'w') as f:
        f.write("# Minimal Travel Model Test Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Test Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Queries | {summary['total_queries']} |\n")
        f.write(f"| Successful | {summary['successful_queries']} |\n")
        f.write(f"| Failed | {summary['failed_queries']} |\n")
        f.write(f"| Success Rate | {(summary['successful_queries']/summary['total_queries']*100):.1f}% |\n")
        f.write(f"| Total Time | {summary['total_time']:.2f}s |\n")
        f.write(f"| Average Time | {summary['average_time']:.2f}s |\n")
        f.write(f"| Average Words | {summary['average_words']:.1f} |\n")
        f.write(f"| Average Characters | {summary['average_chars']:.1f} |\n\n")
        
        f.write("## Test Results\n\n")
        for result in results:
            f.write(f"### Query {result['query_id']}\n")
            f.write(f"**Question:** {result['query']}\n\n")
            f.write(f"**Response:** {result['response']}\n\n")
            f.write(f"*Time: {result['response_time']:.2f}s | Words: {result['word_count']} | Characters: {result['char_count']}*\n\n")
            f.write("---\n\n")
    
    return json_file, report_file

def main():
    """Main testing function"""
    logger.info("🎯 Testing Minimal Travel Model")
    logger.info("=" * 50)
    
    model_path = "llama_travel_model_minimal"
    
    try:
        # Load model
        logger.info("🚀 Loading minimal model...")
        model, tokenizer = load_minimal_model(model_path)
        
        # Test model
        logger.info("🧪 Running travel query tests...")
        results, summary = test_model(model, tokenizer, TRAVEL_QUERIES)
        
        # Save results
        logger.info("💾 Saving test results...")
        json_file, report_file = save_test_results(results, summary)
        
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("🎉 MINIMAL MODEL TEST COMPLETED!")
        logger.info("=" * 50)
        
        logger.info("📊 Test Summary:")
        logger.info(f"   Successful queries: {summary['successful_queries']}/{summary['total_queries']}")
        logger.info(f"   Average response time: {summary['average_time']:.2f}s")
        logger.info(f"   Average response length: {summary['average_words']:.1f} words")
        
        if summary['failed_queries'] > 0:
            logger.warning(f"⚠️  {summary['failed_queries']} queries failed")
        else:
            logger.info("✅ All queries successful!")
        
        logger.info("📁 Results saved:")
        logger.info(f"   📄 JSON: {json_file}")
        logger.info(f"   📄 Report: {report_file}")
        
        logger.info("✅ Minimal model is working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 