import argparse
import os
from transformers import AutoTokenizer
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-tokenize a dataset for language model training.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Name or path of the tokenizer.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process.")
    parser.add_argument("--output_path", type=str, required=True, help="Path on EBS to save the tokenized dataset.")
    parser.add_argument("--sequence_length", type=int, default=2048, help="The sequence length for tokenization.")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of CPU cores to use for tokenization.")
    parser.add_argument("--cache_dir", type=str, default="/home/ubuntu/bigdata/.cache", help="Cache directory for datasets.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set cache directory to your EBS volume
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables to redirect HuggingFace cache
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    
    print(f"Using cache directory: {cache_dir}")
    print(f"Loading dataset '{args.dataset_name}'...")
    
    # Load the raw dataset with explicit cache_dir
    dataset = load_dataset(
        args.dataset_name, 
        split="train",
        cache_dir=cache_dir  # Explicitly set cache directory
    )

    print(f"Loading tokenizer '{args.tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, 
        use_fast=True,
        cache_dir=cache_dir  # Also cache tokenizer on EBS
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer `pad_token` set to `eos_token`.")

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=args.sequence_length)

    print(f"Tokenizing dataset with {args.num_proc} processes...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        cache_file_name=os.path.join(cache_dir, "tokenized_cache.arrow")  # Cache tokenized data on EBS
    )

    print(f"Saving tokenized dataset to '{args.output_path}'...")
    tokenized_dataset.save_to_disk(args.output_path)
    
    print(f"\n🎉 Dataset preparation complete!")
    print(f"Tokenized dataset saved at: {args.output_path}")
    print(f"Cache used: {cache_dir}")

if __name__ == "__main__":
    main()
    
    
"""
# Example (Worked)
python prepare_data.py \
    --tokenizer_name "HuggingFaceTB/SmolLM2-1.7B" \
    --dataset_name "HuggingFaceTB/cosmopedia-100k" \
    --output_path "/home/ubuntu/data/final_training/Day4/New_Approach/cosmopedia-100k/cosmopedia-100k-tokenized" \
    --sequence_length 2048 \
    --num_proc 16 # Use a good number of CPU cores
"""

""" 
Usage 2 (Worked)
python prepare_data.py \
    --tokenizer_name "HuggingFaceTB/SmolLM2-1.7B" \
    --dataset_name "PatrickHaller/cosmopedia-v2-1B" \
    --output_path "/home/ubuntu/bigdata/Training/Day4/cosmopedia-v2-1B/cosmopedia-v2-1B-tokenized" \
    --sequence_length 2048 \
    --num_proc 16 \
    --cache_dir "/home/ubuntu/bigdata/.cache"
"""

""" 
Usage 3 (Did not work)
python prepare_data.py \
    --tokenizer_name "HuggingFaceTB/SmolLM2-1.7B" \
    --dataset_name "EleutherAI/SmolLM2-1.7B-stage-4-20B" \
    --output_path "/home/ubuntu/bigdata/Training/Day4/cosmopedia-v2-20B/cosmopedia-v2-20B-tokenized" \
    --sequence_length 2048 \
    --num_proc 16 \
    --cache_dir "/home/ubuntu/bigdata/.cache"
"""