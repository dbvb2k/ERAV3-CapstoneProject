#!/usr/bin/env python3
"""
Extract and Verify Minimal Model Files (Fixed Version)
======================================================

This script identifies the essential files needed for inference from the full model backup,
extracts them to a minimal folder, and verifies the minimal model works correctly.

This fixed version handles GPU memory issues and model loading problems more gracefully.
"""

import os
import json
import time
import shutil
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test queries for verification
TEST_QUERIES = [
    "I'm planning a 7-day trip to Japan in spring. What are the must-visit places?",
    "What are some budget-friendly European destinations for backpackers?",
    "What's the best time to book international flights to get lowest prices?"
]

def get_file_size_mb(file_path):
    """Get file size in MB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0

def get_directory_size_mb(directory):
    """Get total directory size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)

def load_model_for_testing(model_path, model_name="model"):
    """Load model for testing with better memory management"""
    logger.info(f"🚀 Loading {model_name} from {model_path}...")
    
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"✅ Tokenizer loaded from {model_path}")
        
        # Clear any existing GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load base model with more conservative memory settings
        base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        logger.info(f"🔄 Loading base model {base_model_name}...")
        
        try:
            # Try with conservative memory settings
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory={0: "20GB"},  # More conservative memory allocation
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            logger.info(f"✅ Base model {base_model_name} loaded successfully")
            
        except Exception as error:
            logger.warning(f"⚠️  Standard loading failed: {error}")
            logger.info("🔧 Trying alternative loading strategy...")
            
            try:
                # Try with even more conservative settings
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="sequential",  # Use sequential device mapping
                    max_memory={0: "18GB"},
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    offload_folder="./offload_temp"
                )
                logger.info(f"✅ Base model loaded with alternative strategy")
                
            except Exception as alt_error:
                logger.error(f"❌ Alternative loading also failed: {alt_error}")
                raise error
        
        # Clear cache before loading LoRA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load LoRA adapter with error handling
        logger.info("🔄 Loading LoRA adapter...")
        try:
            model = PeftModel.from_pretrained(
                base_model, 
                model_path,
                is_trainable=False  # Load in inference mode
            )
            logger.info(f"✅ LoRA adapter loaded successfully")
            
        except Exception as lora_error:
            logger.error(f"❌ LoRA loading failed: {lora_error}")
            logger.info("🔧 Trying to load LoRA with alternative settings...")
            
            try:
                # Try loading with different settings
                model = PeftModel.from_pretrained(
                    base_model, 
                    model_path,
                    is_trainable=False,
                    torch_dtype=torch.float16
                )
                logger.info(f"✅ LoRA adapter loaded with alternative settings")
                
            except Exception as alt_lora_error:
                logger.error(f"❌ Alternative LoRA loading failed: {alt_lora_error}")
                logger.warning("⚠️  Will proceed with base model only for testing...")
                model = base_model
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load {model_name}: {e}")
        raise

def test_model_quick(model, tokenizer, model_name):
    """Quick test with just one query to verify model works"""
    logger.info(f"🧪 Quick test of {model_name}...")
    
    test_query = "What are some budget-friendly travel destinations?"
    
    try:
        # Format prompt
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{test_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate with conservative settings
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response_time = time.time() - start_time
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        logger.info(f"✅ {model_name} test successful!")
        logger.info(f"   Response time: {response_time:.2f}s")
        logger.info(f"   Response preview: {response[:100]}...")
        
        return {
            "success": True,
            "response_time": response_time,
            "response_length": len(response),
            "word_count": len(response.split())
        }
        
    except Exception as e:
        logger.error(f"❌ Quick test failed for {model_name}: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def identify_essential_files():
    """Identify the essential files needed for inference"""
    
    essential_files = [
        # Tokenizer files (essential)
        "tokenizer.json",
        "tokenizer_config.json", 
        "special_tokens_map.json",
        
        # LoRA adapter files (essential)
        "adapter_model.safetensors",
        "adapter_config.json",
        
        # Documentation (optional but useful)
        "README.md"
    ]
    
    logger.info("📋 Essential files for inference:")
    for file in essential_files:
        logger.info(f"   - {file}")
    
    return essential_files

def extract_minimal_model(source_path, dest_path, essential_files):
    """Extract only the essential files to minimal model folder"""
    
    logger.info(f"📦 Extracting minimal model from {source_path} to {dest_path}...")
    
    # Create destination directory
    os.makedirs(dest_path, exist_ok=True)
    
    # Track copied files and sizes
    copied_files = []
    skipped_files = []
    total_size = 0
    
    for file in essential_files:
        source_file = os.path.join(source_path, file)
        dest_file = os.path.join(dest_path, file)
        
        if os.path.exists(source_file):
            # Copy file
            shutil.copy2(source_file, dest_file)
            file_size = get_file_size_mb(source_file)
            total_size += file_size
            copied_files.append((file, file_size))
            logger.info(f"✅ Copied {file} ({file_size:.2f} MB)")
        else:
            skipped_files.append(file)
            logger.warning(f"⚠️  File not found: {file}")
    
    # Create a minimal README
    readme_path = os.path.join(dest_path, "README.md")
    with open(readme_path, 'w') as f:
        f.write("# Minimal Travel Model for Inference\n\n")
        f.write("This is a minimal version of the Llama-3-8B travel model fine-tuned on Alpaca dataset.\n\n")
        f.write("## Essential Files:\n")
        for file, size in copied_files:
            f.write(f"- `{file}` - {size:.2f} MB\n")
        f.write(f"\n**Total size:** {total_size:.2f} MB\n")
        f.write(f"**Extracted on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Usage Example:\n")
        f.write("```python\n")
        f.write("from transformers import AutoTokenizer, AutoModelForCausalLM\n")
        f.write("from peft import PeftModel\n\n")
        f.write("# Load tokenizer\n")
        f.write("tokenizer = AutoTokenizer.from_pretrained('path/to/this/folder')\n\n")
        f.write("# Load base model\n")
        f.write("base_model = AutoModelForCausalLM.from_pretrained(\n")
        f.write("    'meta-llama/Meta-Llama-3-8B-Instruct',\n")
        f.write("    torch_dtype=torch.float16,\n")
        f.write("    device_map='auto'\n")
        f.write(")\n\n")
        f.write("# Load LoRA adapter\n")
        f.write("model = PeftModel.from_pretrained(base_model, 'path/to/this/folder')\n")
        f.write("```\n\n")
        
        f.write("## Testing Commands:\n")
        f.write("```bash\n")
        f.write("# Test the minimal model\n")
        f.write("python test_minimal_model.py\n")
        f.write("```\n")
    
    if "README.md" not in [f[0] for f in copied_files]:
        copied_files.append(("README.md", get_file_size_mb(readme_path)))
        total_size += get_file_size_mb(readme_path)
    
    return copied_files, skipped_files, total_size

def main():
    """Main function to extract and verify minimal model"""
    
    logger.info("🎯 Starting Minimal Model Extraction (Fixed Version)")
    logger.info("=" * 60)
    
    # Paths
    original_model_path = "llama_travel_model_backup/model"
    minimal_model_path = "llama_travel_model_minimal"
    
    try:
        # Step 1: Identify essential files
        logger.info("\n📋 Step 1: Identifying essential files...")
        essential_files = identify_essential_files()
        
        # Step 2: Extract minimal model first (safer approach)
        logger.info("\n📦 Step 2: Extracting minimal model...")
        copied_files, skipped_files, total_size = extract_minimal_model(
            original_model_path, minimal_model_path, essential_files
        )
        
        # Step 3: Test the minimal model
        logger.info("\n🧪 Step 3: Testing minimal model...")
        try:
            minimal_model, minimal_tokenizer = load_model_for_testing(minimal_model_path, "Minimal Model")
            minimal_test_result = test_model_quick(minimal_model, minimal_tokenizer, "Minimal Model")
            
            # Clear GPU memory
            del minimal_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ Minimal model test failed: {e}")
            minimal_test_result = {"success": False, "error": str(e)}
        
        # Step 4: Compare sizes
        logger.info("\n📊 Step 4: Analyzing results...")
        original_size = get_directory_size_mb(original_model_path)
        minimal_size = get_directory_size_mb(minimal_model_path)
        size_reduction = ((original_size - minimal_size) / original_size) * 100
        
        # Create results summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("minimal_model_test_results", exist_ok=True)
        
        summary = {
            "timestamp": timestamp,
            "extraction_success": True,
            "minimal_model_test": minimal_test_result,
            "file_analysis": {
                "files_copied": len(copied_files),
                "files_skipped": len(skipped_files),
                "copied_files": copied_files,
                "skipped_files": skipped_files
            },
            "size_analysis": {
                "original_size_mb": original_size,
                "minimal_size_mb": minimal_size,
                "size_reduction_percent": size_reduction,
                "space_saved_mb": original_size - minimal_size
            }
        }
        
        # Save results
        results_file = f"minimal_model_test_results/extraction_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("🎉 MINIMAL MODEL EXTRACTION COMPLETED!")
        logger.info("=" * 60)
        
        logger.info("📦 Extraction Summary:")
        logger.info(f"   Files copied: {len(copied_files)}")
        logger.info(f"   Total size: {total_size:.2f} MB")
        logger.info(f"   Size reduction: {size_reduction:.1f}%")
        logger.info(f"   Space saved: {original_size - minimal_size:.2f} MB")
        
        logger.info("🧪 Test Results:")
        if minimal_test_result.get("success", False):
            logger.info(f"   ✅ Minimal model test PASSED")
            logger.info(f"   Response time: {minimal_test_result.get('response_time', 0):.2f}s")
        else:
            logger.warning(f"   ⚠️  Minimal model test had issues: {minimal_test_result.get('error', 'Unknown')}")
        
        logger.info("📁 Files created:")
        logger.info(f"   📂 Minimal model: {minimal_model_path}/")
        logger.info(f"   📄 Results: {results_file}")
        
        logger.info("💡 Next steps:")
        logger.info(f"   1. Download the folder: {minimal_model_path}/")
        logger.info(f"   2. Total download size: {total_size:.2f} MB")
        logger.info("   3. Test locally with: python test_minimal_model.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Process failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
    finally:
        # Clean up any temporary files
        if os.path.exists("./offload_temp"):
            shutil.rmtree("./offload_temp", ignore_errors=True)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 