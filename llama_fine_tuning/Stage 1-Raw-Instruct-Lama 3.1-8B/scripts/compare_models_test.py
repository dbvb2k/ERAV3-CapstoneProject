#!/usr/bin/env python3
"""
Comprehensive Model Comparison Test
Tests both base Llama 3.2 3B model and trained travel model with 15 travel queries
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

# 15 Travel queries for testing
TRAVEL_QUERIES = [
    "I'm planning a 7-day trip to Japan in spring. What are the must-visit places and experiences I should prioritize?",
    "What are some budget-friendly European destinations for backpackers under $50 per day?",
    "Where are the best family-friendly destinations in the US for traveling with young children?",
    "What's the most efficient way to travel between major cities in Europe by train?",
    "When is the best time to book international flights to get the lowest prices?",
    "How do I navigate public transportation in major Asian cities like Tokyo, Seoul, and Singapore?",
    "What are the pros and cons of staying in hostels vs hotels vs Airbnb for solo travelers?",
    "How can I experience authentic local cuisine while traveling without getting sick?",
    "What are the best strategies for vegetarian travelers in countries with meat-heavy cuisines?",
    "What safety precautions should solo female travelers take when visiting developing countries?",
    "What documents and preparations do I need for international travel, especially for first-time travelers?",
    "How should I handle money and payments while traveling internationally to avoid fees?",
    "What are the best winter destinations for travelers who want to escape cold weather?",
    "What gear and preparation do I need for trekking in the Himalayas or similar high-altitude destinations?",
    "How can I respectfully engage with local cultures and communities while traveling to avoid being just a tourist?"
]

def load_base_model():
    """Load the base Llama 3 8B model"""
    logger.info("🚀 Loading base Llama 3 8B model...")
    
    # Use Llama 3 8B model variants
    model_options = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct"
    ]
    
    for model_name in model_options:
        try:
            logger.info(f"🔄 Trying {model_name}...")
            
            # Load with rope_scaling fix
            from transformers import LlamaConfig
            
            try:
                # Try direct loading first (without attn_implementation)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_memory={0: "14GB"},  # More memory for 8B model
                    trust_remote_code=True
                )
                logger.info(f"✅ {model_name} loaded successfully!")
                return model, tokenizer
                
            except Exception as error:
                logger.info(f"🔧 Direct loading failed: {error}")
                logger.info(f"🔧 Trying with rope_scaling fix for {model_name}...")
                
                try:
                    # Load config and fix rope_scaling
                    config = LlamaConfig.from_pretrained(model_name)
                    if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
                        rope_scaling = config.rope_scaling.copy()
                        if 'rope_type' in rope_scaling:
                            rope_scaling['type'] = rope_scaling.pop('rope_type')
                        # Keep only required fields
                        config.rope_scaling = {
                            'type': rope_scaling.get('type', 'linear'),
                            'factor': rope_scaling.get('factor', 1.0)
                        }
                    
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        config=config,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        max_memory={0: "14GB"}
                    )
                    logger.info(f"✅ {model_name} loaded with rope_scaling fix!")
                    return model, tokenizer
                    
                except Exception as config_error:
                    logger.error(f"❌ Config fix failed: {config_error}")
                    raise error
                    
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            continue
    
    raise Exception("❌ Failed to load any base model variant")

def load_trained_model():
    """Load the trained model from backup"""
    logger.info("🚀 Loading trained model from backup...")
    
    try:
        # Load tokenizer from backup
        tokenizer = AutoTokenizer.from_pretrained("llama_travel_model_backup/model/")
        
        # Load base model (8B for the trained model) with rope_scaling fix
        base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        
        try:
            # Try direct loading first
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory={0: "14GB"},
                trust_remote_code=True
            )
        except Exception as error:
            logger.info(f"🔧 Direct loading failed for trained model base: {error}")
            logger.info("🔧 Trying with rope_scaling fix for trained model base...")
            
            try:
                # Load config manually and fix rope_scaling
                from transformers import LlamaConfig
                import json
                import requests
                
                # Get the config dict directly
                config_url = f"https://huggingface.co/{base_model_name}/resolve/main/config.json"
                response = requests.get(config_url)
                config_dict = response.json()
                
                # Fix rope_scaling in the dict before creating config
                if 'rope_scaling' in config_dict and config_dict['rope_scaling'] is not None:
                    rope_scaling = config_dict['rope_scaling'].copy()
                    if 'rope_type' in rope_scaling:
                        rope_scaling['type'] = rope_scaling.pop('rope_type')
                    # Keep only required fields
                    config_dict['rope_scaling'] = {
                        'type': rope_scaling.get('type', 'linear'),
                        'factor': rope_scaling.get('factor', 1.0)
                    }
                
                # Create config from fixed dict
                config = LlamaConfig.from_dict(config_dict)
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    config=config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_memory={0: "14GB"}
                )
            except Exception as config_error:
                logger.error(f"❌ Advanced config fix failed for trained model: {config_error}")
                
                # Last resort: try using the same model as base model
                logger.info("🔄 Trying to use same model as base model...")
                try:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        "meta-llama/Meta-Llama-3-8B-Instruct",
                        torch_dtype=torch.float16,
                        device_map="auto",
                        max_memory={0: "14GB"},
                        trust_remote_code=True
                    )
                except Exception as final_error:
                    logger.error(f"❌ Final fallback failed: {final_error}")
                    raise error
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, "llama_travel_model_backup/model/")
        
        logger.info("✅ Trained model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load trained model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def test_model(model, tokenizer, model_name, queries):
    """Test a model with given queries"""
    logger.info(f"🧪 Testing {model_name}...")
    
    results = []
    total_time = 0
    
    for i, query in enumerate(queries, 1):
        logger.info(f"📝 Query {i}/15: {query[:50]}...")
        
        # Format prompt
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
                    max_new_tokens=200,
                    temperature=0.7,
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
                "word_count": len(response.split())
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to generate response for query {i}: {e}")
            results.append({
                "query_id": i,
                "query": query,
                "response": f"ERROR: {str(e)}",
                "response_time": 0,
                "word_count": 0
            })
    
    avg_time = total_time / len(queries)
    total_words = sum(r["word_count"] for r in results)
    avg_words = total_words / len(queries)
    
    logger.info(f"📊 {model_name} Results:")
    logger.info(f"   Total time: {total_time:.2f}s")
    logger.info(f"   Average time per query: {avg_time:.2f}s")
    logger.info(f"   Total words: {total_words}")
    logger.info(f"   Average words per response: {avg_words:.1f}")
    
    return {
        "model_name": model_name,
        "results": results,
        "summary": {
            "total_time": total_time,
            "average_time": avg_time,
            "total_words": total_words,
            "average_words": avg_words,
            "successful_queries": len([r for r in results if not r["response"].startswith("ERROR:")])
        }
    }

def compare_responses(base_results, trained_results):
    """Compare responses between base and trained models"""
    logger.info("🔍 Comparing model responses...")
    
    comparisons = []
    
    for i in range(len(TRAVEL_QUERIES)):
        base_resp = base_results["results"][i]
        trained_resp = trained_results["results"][i]
        
        comparison = {
            "query_id": i + 1,
            "query": TRAVEL_QUERIES[i][:100] + "...",
            "base_model": {
                "response_length": len(base_resp["response"]),
                "word_count": base_resp["word_count"],
                "response_time": base_resp["response_time"],
                "preview": base_resp["response"][:150] + "..."
            },
            "trained_model": {
                "response_length": len(trained_resp["response"]),
                "word_count": trained_resp["word_count"],
                "response_time": trained_resp["response_time"],
                "preview": trained_resp["response"][:150] + "..."
            }
        }
        
        comparisons.append(comparison)
    
    return comparisons

def save_results(base_results, trained_results, comparisons):
    """Save all results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    os.makedirs("model_comparison_results", exist_ok=True)
    
    # Save comprehensive results
    all_results = {
        "timestamp": timestamp,
        "base_model_results": base_results,
        "trained_model_results": trained_results,
        "comparisons": comparisons,
        "summary": {
            "base_model": base_results["summary"],
            "trained_model": trained_results["summary"]
        }
    }
    
    results_file = f"model_comparison_results/comparison_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create readable report
    report_file = f"model_comparison_results/report_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Metric | Base Llama 3 8B | Trained Travel Model |\n")
        f.write("|--------|------------------|----------------------|\n")
        f.write(f"| Total Time | {base_results['summary']['total_time']:.2f}s | {trained_results['summary']['total_time']:.2f}s |\n")
        f.write(f"| Avg Time/Query | {base_results['summary']['average_time']:.2f}s | {trained_results['summary']['average_time']:.2f}s |\n")
        f.write(f"| Total Words | {base_results['summary']['total_words']} | {trained_results['summary']['total_words']} |\n")
        f.write(f"| Avg Words/Response | {base_results['summary']['average_words']:.1f} | {trained_results['summary']['average_words']:.1f} |\n")
        f.write(f"| Successful Queries | {base_results['summary']['successful_queries']}/15 | {trained_results['summary']['successful_queries']}/15 |\n\n")
        
        f.write("## Detailed Comparisons\n\n")
        for comp in comparisons:
            f.write(f"### Query {comp['query_id']}\n")
            f.write(f"**Question:** {comp['query']}\n\n")
            f.write(f"**Base Model Response:** {comp['base_model']['preview']}\n")
            f.write(f"*({comp['base_model']['word_count']} words, {comp['base_model']['response_time']:.2f}s)*\n\n")
            f.write(f"**Trained Model Response:** {comp['trained_model']['preview']}\n")
            f.write(f"*({comp['trained_model']['word_count']} words, {comp['trained_model']['response_time']:.2f}s)*\n\n")
            f.write("---\n\n")
    
    logger.info(f"📁 Results saved to:")
    logger.info(f"   - {results_file}")
    logger.info(f"   - {report_file}")
    
    return results_file, report_file

def main():
    """Main testing function"""
    logger.info("🎯 Starting comprehensive model comparison test...")
    logger.info("📝 Comparing Base Llama 3 8B vs Trained Travel Model")
    logger.info(f"📝 Testing {len(TRAVEL_QUERIES)} travel queries on both models")
    
    try:
        # Load base model
        base_model, base_tokenizer = load_base_model()
        
        # Load trained model
        trained_model, trained_tokenizer = load_trained_model()
        
        # Test base model
        logger.info("\n" + "="*60)
        base_results = test_model(base_model, base_tokenizer, "Base Llama 3 8B", TRAVEL_QUERIES)
        
        # Clear GPU memory
        del base_model
        torch.cuda.empty_cache()
        
        # Test trained model
        logger.info("\n" + "="*60)
        trained_results = test_model(trained_model, trained_tokenizer, "Trained Travel Model", TRAVEL_QUERIES)
        
        # Compare results
        logger.info("\n" + "="*60)
        comparisons = compare_responses(base_results, trained_results)
        
        # Save results
        results_file, report_file = save_results(base_results, trained_results, comparisons)
        
        # Print final summary
        logger.info("\n" + "="*60)
        logger.info("🎉 MODEL COMPARISON COMPLETE!")
        logger.info("="*60)
        logger.info("📊 SUMMARY:")
        logger.info(f"   Base Model Average Time: {base_results['summary']['average_time']:.2f}s")
        logger.info(f"   Trained Model Average Time: {trained_results['summary']['average_time']:.2f}s")
        logger.info(f"   Base Model Average Words: {base_results['summary']['average_words']:.1f}")
        logger.info(f"   Trained Model Average Words: {trained_results['summary']['average_words']:.1f}")
        logger.info(f"📁 Detailed results: {results_file}")
        logger.info(f"📄 Human-readable report: {report_file}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 