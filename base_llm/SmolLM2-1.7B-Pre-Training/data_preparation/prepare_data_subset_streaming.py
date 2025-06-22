import os
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json

def setup_cache_environment(cache_dir):
    """Set all HuggingFace cache environment variables"""
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset using streaming")
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    # parser.add_argument("--subset_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to process")
    
    args = parser.parse_args()
    
    # Setup cache
    setup_cache_environment(args.cache_dir)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        cache_dir=args.cache_dir
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset in streaming mode
    print("Loading dataset in streaming mode...")
    dataset_stream = load_dataset(
        args.dataset_name,
        # name=args.subset_name,
        split="train",
        streaming=True  # This avoids downloading everything at once
    )
    
    # Process in batches
    batch_size = 1000
    tokenized_examples = []
    
    print("Processing dataset in batches...")
    for i, example in enumerate(tqdm(dataset_stream)):
        if args.max_samples and i >= args.max_samples:
            break
            
        # Tokenize single example
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=args.sequence_length,
            return_tensors=None
        )
        
        tokenized_examples.append({
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        })
        
        # Save batch when full
        if len(tokenized_examples) >= batch_size:
            save_batch(tokenized_examples, args.output_path, i // batch_size)
            tokenized_examples = []
    
    # Save final batch
    if tokenized_examples:
        save_batch(tokenized_examples, args.output_path, (i // batch_size) + 1)
    
    # Create final dataset
    combine_batches(args.output_path)
    print(f"Dataset processing complete. Saved to {args.output_path}")

def save_batch(examples, output_path, batch_num):
    """Save a batch of tokenized examples"""
    batch_dir = os.path.join(output_path, "batches")
    os.makedirs(batch_dir, exist_ok=True)
    
    batch_file = os.path.join(batch_dir, f"batch_{batch_num:04d}.json")
    with open(batch_file, 'w') as f:
        json.dump(examples, f)

def combine_batches(output_path):
    """Combine all batches into a single dataset"""
    batch_dir = os.path.join(output_path, "batches")
    all_examples = []
    
    # Load all batch files
    for batch_file in sorted(os.listdir(batch_dir)):
        if batch_file.endswith('.json'):
            with open(os.path.join(batch_dir, batch_file), 'r') as f:
                batch_examples = json.load(f)
                all_examples.extend(batch_examples)
    
    # Create dataset and save
    dataset = Dataset.from_list(all_examples)
    dataset.save_to_disk(output_path)
    
    # Clean up batch files
    import shutil
    shutil.rmtree(batch_dir)

if __name__ == "__main__":
    main()
    
"""
Usage (Worked)
# Use streaming version with limited samples first
python prepare_data_subset_streaming.py \
    --tokenizer_name "HuggingFaceTB/SmolLM2-1.7B" \
    --dataset_name "EleutherAI/SmolLM2-1.7B-stage-4-20B" \
    --output_path "/home/ubuntu/bigdata/Training/Day4/cosmopedia-v2-20B/cosmopedia-v2-20B-tokenized" \
    --sequence_length 2048 \
    --num_proc 16 \
    --cache_dir "/home/ubuntu/bigdata/.cache" \
    --max_samples 100000 
"""