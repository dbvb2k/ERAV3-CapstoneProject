import os
import argparse

# CRITICAL: Set environment variables BEFORE importing datasets/transformers
def setup_cache_environment(cache_dir):
    """Set all HuggingFace cache environment variables"""
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set all possible HF cache environment variables
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir
    
    print(f"Set HF cache directories to: {cache_dir}")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE')}")

# NOW import after setting environment
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Prepare and tokenize dataset subset")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer to use")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name on HuggingFace Hub")
    # parser.add_argument("--subset_name", type=str, required=True, help="Subset/config name")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for tokenized dataset")
    parser.add_argument("--sequence_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--num_proc", type=int, default=16, help="Number of processes for tokenization")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory for HuggingFace")
    
    args = parser.parse_args()
    
    # Setup cache FIRST
    setup_cache_environment(args.cache_dir)
    
    print(f"Using cache directory: {args.cache_dir}")
    
    # Load tokenizer with explicit cache_dir
    print(f"Loading tokenizer '{args.tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        cache_dir=args.cache_dir
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    # Load dataset with explicit cache_dir
    print(f"Loading dataset '{args.dataset_name}'...")
    dataset = load_dataset(
        args.dataset_name,
        # name=args.subset_name,
        split="train",
        cache_dir=args.cache_dir,
        streaming=False  # Ensure we're not using streaming mode
    )
    
    print(f"Dataset loaded with {len(dataset):,} examples")
    
    # Tokenization function
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.sequence_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return outputs
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        cache_file_name=os.path.join(args.cache_dir, "tokenized_cache.arrow")
    )
    
    # Save tokenized dataset
    print(f"Saving tokenized dataset to {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    tokenized_dataset.save_to_disk(args.output_path)
    
    print(f"Tokenized dataset saved with {len(tokenized_dataset):,} examples")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Output saved to: {args.output_path}")

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
python prepare_data_subset2.py \
    --tokenizer_name "HuggingFaceTB/SmolLM2-1.7B" \
    --dataset_name "EleutherAI/SmolLM2-1.7B-stage-4-20B" \
    --output_path "/home/ubuntu/bigdata/Training/Day4/cosmopedia-v2-20B/cosmopedia-v2-20B-tokenized" \
    --sequence_length 2048 \
    --num_proc 16 \
    --cache_dir "/home/ubuntu/bigdata/.cache"
"""