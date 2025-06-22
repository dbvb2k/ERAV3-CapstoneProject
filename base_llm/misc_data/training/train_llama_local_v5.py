import argparse
import json
import math
import os
import shutil
import torch
import logging
from pathlib import Path
import datasets
import transformers
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, interleave_datasets, Dataset, DatasetDict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from huggingface_hub import login
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    get_scheduler,
    SchedulerType
)
import bitsandbytes as bnb # For 8-bit optimizer
torch.set_float32_matmul_precision('high')

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Llama-like model from scratch locally")
    
    # Model Configuration
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size.")
    parser.add_argument("--hidden_size", type=int, default=2048, help="Dimension of the hidden states.")
    parser.add_argument("--intermediate_size", type=int, default=5504, help="Dimension of the intermediate MLP layers.")
    parser.add_argument("--num_hidden_layers", type=int, default=24, help="Number of hidden layers in the Transformer encoder.")
    parser.add_argument("--num_attention_heads", type=int, default=16, help="Number of attention heads for each attention layer.")
    parser.add_argument("--num_key_value_heads", type=int, default=16, help="Number of key/value heads for GQA (set to num_attention_heads for MHA).")
    parser.add_argument("--max_position_embeddings", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="Standard deviation of the truncated_normal_initializer for initializing all weight matrices.")
    parser.add_argument("--rms_norm_eps", type=float, default=1e-6, help="Epsilon value for RMSNorm.")
    parser.add_argument("--tie_word_embeddings", type=bool, default=True, help="Whether to tie input and output embeddings.")
    parser.add_argument("--dataset_streaming", type=bool, default=False, help="Enable streaming for large datasets.") # Set to True for AWS
    parser.add_argument(
        "--rope_theta", 
        type=float, 
        default=10000.0, # Default Llama 1/2 value, Llama 3 uses higher
        help="RoPE base theta frequency."
    )
    parser.add_argument(
        "--rope_scaling_json", 
        type=str, 
        default=None,
        help="JSON string for RoPE scaling configuration dictionary (e.g., '{\"type\": \"linear\", \"factor\": 2.0}')"
    )
    
    # Dataset and Tokenizer
    # IMPORTANT: Modify these for your specific dataset
    parser.add_argument("--dataset_name", type=str, default=None, help="Hugging Face dataset name or path to a local dataset.") # Example small dataset
    parser.add_argument("--dataset_config_name", type=str, default=None, help="Specific dataset configuration (if any).")
    parser.add_argument("--text_column", type=str, default="text", help="The name of the column in the dataset containing the text.")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-3.2-1B", help="Tokenizer to use (e.g., from a pretrained Llama model).")
    parser.add_argument("--sequence_length", type=int, default=512, help="Sequence length for tokenization. Adjust based on VRAM.") # Start small

    # Training Hyperparameters
    parser.add_argument("--output_dir", type=str, default="./llama_1b_scratch_local_chkpt", help="Where to store the final model and checkpoints.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size (per device) for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Initial learning rate (after the potential warmup period).")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to apply.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform (will be overridden by max_train_steps).")
    parser.add_argument("--max_train_steps", type=int, default=200, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help="The scheduler type to use.")
    parser.add_argument("--num_warmup_steps", type=int, default=20, help="Number of steps for the warmup phase.")
    # parser.add_argument("--perform_raw_token_counts", action="store_true", help="If set, count raw tokens in the prepared dataset before training tokenization using the main tokenizer specified by --tokenizer_name.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    
    # Checkpointing and Generation
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X update steps.")
    parser.add_argument("--generation_steps", type=int, default=50, help="Generate a sample every X update steps.")
    parser.add_argument("--generation_prompt", type=str, default="Once upon a time,", help="Prompt for sample generation.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max new tokens for sample generation.")
    
    # Resume Training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")

    # Performance & Memory Optimization
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose bf16 if supported (RTX 30xx/40xx), else fp16.")
    parser.add_argument("--use_gradient_checkpointing", type=bool, default=True, help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--use_8bit_optimizer", type=bool, default=True, help="Use 8-bit AdamW optimizer from bitsandbytes.")
    parser.add_argument("--torch_compile", type=bool, default=False, help="Enable torch.compile for potential speedups (experimental, requires PyTorch 2.0+). Might add overhead for short runs.")
    
    ## Monitoring and Reporting
    parser.add_argument("--report_to", type=str, default="tensorboard", help="The integration to report results and logs to (tensorboard or wandb).")

    args = parser.parse_args()
    return args

from datasets import load_dataset, interleave_datasets, Dataset, DatasetDict

os.environ["HF_DATASETS_CACHE"] = "/mnt/ebs/hf_cache"

def prepare_combined_dataset_streaming(sample_sizes=None, save_path=None):
    """
    Streams and combines datasets. If `save_path` is given, saves to disk.
    Returns: streaming or standard Hugging Face dataset.
    """
    # Define the text field mapping (manually verified)
    text_field_map = {
        "EleutherAI/pile": "text",
        "Skylion007/openwebtext": "text",
        "allenai/c4": "text",
        "togethercomputer/RedPajama-Data-1T": "text",
        "ai4bharat/sangraha": "text",
        "zicsx/mC4-Hindi-Cleaned-3.0": "text",
        "PleIAs/common_corpus": "text"
    }

    if sample_sizes is None:
        sample_sizes = {
            "Skylion007/openwebtext": 30000,
            "allenai/c4": 30000,
            "togethercomputer/RedPajama-Data-1T": 30000,
            "ai4bharat/sangraha": 30000,
            "zicsx/mC4-Hindi-Cleaned-3.0": 30000,
            "PleIAs/common_corpus": 30000
        }

    datasets_streamed = []

    for name, limit in sample_sizes.items():
        kwargs = {"split": "train", "streaming": True, "trust_remote_code": True}  # Always stream and trust remote code for flexibility
        if name == "allenai/c4":
            kwargs["name"] = "en"
        if name == "togethercomputer/RedPajama-Data-1T":
            kwargs["name"] = "default"
        if name == "ai4bharat/sangraha":
            kwargs["data_dir"] = "verified/hin"

        try:
            logger.info(f"🌐 Streaming {name} (limit {limit})...")
            ds_stream = load_dataset(name, **kwargs)
            field = text_field_map.get(name, "text")
            ds_sample = ds_stream.take(limit)

            # Normalize to {"text": actual_text}
            ds_sample = ds_sample.map(lambda x: {"text": x.get(field, "")})
            ds_sample = ds_sample.filter(lambda x: isinstance(x["text"], str) and x["text"].strip() != "")
            
            datasets_streamed.append(ds_sample)
        except Exception as e:
            logger.info(f"⚠️ Failed to stream {name}: {e}")

    # Combine
    combined_iterable = interleave_datasets(datasets_streamed, stopping_strategy="all_exhausted")
    logger.info("✅ Combined streaming dataset ready.")

    if save_path:
        logger.info(f"💾 Saving combined dataset to disk at: {save_path}")
        os.makedirs(save_path, exist_ok=True)
        raw_data = list(combined_iterable)
        ds = Dataset.from_list(raw_data)
        ds.save_to_disk(save_path)
        logger.info("✅ Dataset saved to: ", save_path)
        return ds

    return combined_iterable

# Function to calculate raw dataset statistics (commented out for now)
# def calculate_raw_dataset_statistics(dataset_obj, tokenizer_instance, text_column_name, accelerator_instance, logger_instance):
#     if not accelerator_instance.is_main_process:
#         return

#     logger_instance.info(f"Starting raw dataset statistics calculation for column '{text_column_name}' using tokenizer '{tokenizer_instance.name_or_path}'...")
#     total_rows = 0
#     total_tokens = 0
    
#     is_iterable_dataset = isinstance(dataset_obj, datasets.IterableDataset)

#     if not is_iterable_dataset and hasattr(dataset_obj, '__len__'):
#         try:
#             total_rows = len(dataset_obj)
#             logger_instance.info(f"Dataset has a defined length: {total_rows} rows. Iterating for token count.")
#         except TypeError:
#             logger_instance.info("Dataset length not available directly, will count rows during iteration.")
#             total_rows = 0 
#     else:
#         logger_instance.info("Dataset is iterable or length is not efficiently available. Counting rows and tokens during iteration.")

#     # Use tqdm for progress if available
#     try:
#         from tqdm.auto import tqdm
#         pbar = tqdm(total=total_rows if total_rows > 0 and not is_iterable_dataset else None, desc="Counting tokens", unit=" examples", disable=not accelerator_instance.is_local_main_process)
#     except ImportError:
#         pbar = None
#         logger_instance.info("tqdm not available, progress will not be shown for token counting.")

#     processed_rows_for_tokens = 0
#     for example in dataset_obj: # This will re-iterate the dataset
#         text = example.get(text_column_name)
#         if isinstance(text, str) and text.strip():
#             token_ids = tokenizer_instance(text, truncation=False, padding=False)["input_ids"]
#             total_tokens += len(token_ids)
        
#         processed_rows_for_tokens += 1
#         if pbar:
#             pbar.update(1)
#         elif processed_rows_for_tokens % 100000 == 0: 
#             logger_instance.info(f"Processed {processed_rows_for_tokens} examples for token counting...")

#     if pbar:
#         pbar.close()

#     if is_iterable_dataset or total_rows == 0 : # If it was an iterable or len() failed, use the count from iteration
#         total_rows = processed_rows_for_tokens

#     logger_instance.info("--- Raw Dataset Statistics ---")
#     logger_instance.info(f"Total examples (rows) processed for counting: {total_rows}")
#     logger_instance.info(f"Total raw tokens (using '{tokenizer_instance.name_or_path}'): {total_tokens}")
#     logger_instance.info("-----------------------------")


def main():
    args = parse_args()
    login(os.getenv("HUGGINGFACE_TOKEN"))

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
        log_with=args.report_to, # or "wandb"
        project_dir=args.output_dir
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set seed before initializing model.
    set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            # Save args
            with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
    accelerator.wait_for_everyone()

    # --- 1. Model Configuration and Initialization ---
    logger.info("Initializing model from scratch...")
    rope_scaling_config = None
    if args.rope_scaling_json:
        try:
            rope_scaling_config = json.loads(args.rope_scaling_json)
            logger.info(f"Applying RoPE scaling configuration: {rope_scaling_config}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON string for --rope_scaling_json: {args.rope_scaling_json}. Error: {e}")
            # Decide how to handle: raise error, or proceed without RoPE scaling
            # For now, let's proceed without, but log a clear warning.
            logger.warning("Proceeding without RoPE scaling due to parsing error.")
            rope_scaling_config = None # Ensure it's None

    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_position_embeddings,
        initializer_range=args.initializer_range,
        rms_norm_eps=args.rms_norm_eps,
        use_cache=False, 
        tie_word_embeddings=args.tie_word_embeddings,
        hidden_act="silu",
        # --- Add/Modify these for RoPE ---
        rope_theta=args.rope_theta,
        rope_scaling=rope_scaling_config  # This will be None if not provided or if JSON parsing failed
    )
    model = LlamaForCausalLM(config)
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    # --- Gradient Checkpointing ---
    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    # --- 2. Tokenizer ---
    logger.info(f"Loading tokenizer: {args.tokenizer_name}")
    # For training from scratch, ensure the tokenizer's vocab size matches the model's.
    # If tokenizer vocab_size is different, model.resize_token_embeddings might be needed
    # AFTER loading the tokenizer, IF the tokenizer has a different vocab size than model_config.vocab_size.
    # However, LlamaConfig sets vocab_size, and we expect the tokenizer to match or be adjusted.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    
    # Llama specific tokenizer settings if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Common practice for Llama-like models
        config.pad_token_id = config.eos_token_id

    # Resize model embeddings if tokenizer vocab size is different (should match if configured correctly)
    if len(tokenizer) != model.config.vocab_size:
        logger.warning(
            f"Tokenizer vocab size ({len(tokenizer)}) does not match model config vocab size ({model.config.vocab_size}). "
            f"Resizing model token embeddings. This is not ideal for 'from scratch' if vocabularies differ significantly."
        )
        model.resize_token_embeddings(len(tokenizer))


    # --- 3. Dataset Loading and Preprocessing ---
    logger.info(f"Loading dataset: {args.dataset_name}")
    # For "stas/openwebtext-10k", it has a 'train' split. Adjust if your dataset has different split names.
    # For AWS, ensure dataset_name points to S3 URI or path on instance, and set dataset_streaming=True
    try:
        dataset_is_from_internal_preparation = not args.dataset_name
        effective_streaming = args.dataset_streaming or dataset_is_from_internal_preparation
        if args.dataset_name:
            logger.info(f"Loading dataset from --dataset_name: {args.dataset_name}")
            raw_datasets = load_dataset(
                args.dataset_name, 
                args.dataset_config_name, 
                streaming=args.dataset_streaming # User controls streaming for external datasets
            )
        else:
            logger.info("No --dataset_name provided. Using internal prepare_combined_dataset_streaming().")
            # Internal preparation ALWAYS results in a stream for tokenization, save_path is not used here
            raw_datasets = prepare_combined_dataset_streaming(sample_sizes=None) # Pass sample_sizes if configurable via args
            dataset_is_from_internal_preparation = True
            # If internal prep is used, it's inherently a stream for the next steps
            # args.dataset_streaming = True # Option 1: Force it for subsequent logic
            logger.info(f"Internal dataset preparation complete. Effective streaming: True")

        # Adapt split selection for streaming if necessary.
        # For streaming, you typically work with an iterable dataset directly.
        # If not streaming, or if streaming still returns a dict of iterables:
        split_to_use = "train" # Assuming 'train' is the target

        if effective_streaming:
            logger.info("Processing dataset as a stream.")
            if isinstance(raw_datasets, dict) and split_to_use in raw_datasets: 
                train_iterable_dataset_for_map = raw_datasets[split_to_use]
                # Need to get column names if this path is taken for an IterableDataset inside a dict
                try:
                    # Assuming raw_datasets[split_to_use] is an IterableDataset
                    first_example = next(iter(train_iterable_dataset_for_map)) 
                    all_original_columns = list(first_example.keys())
                    # Re-assign to ensure the iterator is fresh for the map operation
                    train_iterable_dataset_for_map = raw_datasets[split_to_use] 
                except StopIteration:
                    logger.error(f"Raw dataset split '{split_to_use}' is empty. Cannot proceed.")
                    return
                except AttributeError: # Not an iterable, or doesn't behave as expected
                    logger.error(f"Could not determine columns for raw_datasets['{split_to_use}']. Type: {type(raw_datasets[split_to_use])}")
                    return


            elif isinstance(raw_datasets, torch.utils.data.IterableDataset): 
                # This is the path taken if prepare_combined_dataset_streaming() was used.
                # datasets.IterableDataset can typically be iterated multiple times.
                train_iterable_dataset_for_map = raw_datasets 
                try:
                    first_example = next(iter(train_iterable_dataset_for_map))
                    all_original_columns = list(first_example.keys())
                    # No need to re-assign train_iterable_dataset_for_map if it's a datasets.IterableDataset,
                    # as it will create a new iterator for the .map() call.
                except StopIteration:
                    logger.error("Raw dataset (from internal preparation) is empty. Cannot proceed.")
                    return
            else:
                raise ValueError(f"Unsupported dataset type for streaming: {type(raw_datasets)}. Expected dict with IterableDataset or IterableDataset.")

            # Determine columns to remove: all original columns that are NOT args.text_column.
            # The select_text_column_fn will ensure args.text_column is correctly populated.
            columns_to_remove = [
                col for col in all_original_columns if col != args.text_column
            ]

            def select_text_column_fn(example):
                # This function ensures that the output dictionary (which updates the example)
                # contains args.text_column with the correct text value.
                # It prioritizes existing args.text_column, then 'text' (from internal prep), then empty.
                if args.text_column in example:
                    return {args.text_column: example[args.text_column]}
                elif "text" in example: # 'text' field from prepare_combined_dataset_streaming
                    return {args.text_column: example["text"]}
                else:
                    logger.warning(
                        f"Neither '{args.text_column}' nor 'text' found in example keys: {list(example.keys())}. "
                        f"Returning empty string for '{args.text_column}'."
                    )
                    return {args.text_column: ""}

            logger.info(f"Mapping dataset to select/create '{args.text_column}'. Original columns found: {all_original_columns}. Columns to remove: {columns_to_remove}.")

            tokenized_train_dataset = train_iterable_dataset_for_map.map(
                select_text_column_fn,
                remove_columns=columns_to_remove  # These are original columns to remove
            )

        else: # Not streaming (implies args.dataset_name was given and args.dataset_streaming was False)
            logger.info("Processing dataset as a map-style (non-streaming) dataset.")
            if not isinstance(raw_datasets, dict) or split_to_use not in raw_datasets:
                raise ValueError(f"Dataset '{args.dataset_name}' not found or does not contain '{split_to_use}' split for non-streaming mode.")
            train_dataset_not_iterable = raw_datasets[split_to_use]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Please ensure your dataset is correctly specified and accessible.")
        logger.error("For local testing, 'stas/openwebtext-10k' is a good small dataset.")
        logger.error("For larger datasets on cloud, you'll need to use streaming and point to the correct data source.")
        return

    # Tokenization (map function is fine for non-streamed, for streamed, use .map or on-the-fly)
    # The .map function might behave differently or have limitations with streaming.
    # Often, for streaming, you apply tokenization on-the-fly in the DataLoader or via dataset.map if supported efficiently.
    # Let's assume for now dataset.map works or you'll adapt to on-the-fly tokenization if needed for S3.

    # def tokenize_function(examples):
    #     return tokenizer(examples[args.text_column], truncation=True, max_length=args.sequence_length, return_special_tokens_mask=True)
    # For streaming, the dataset might already be iterable.
    if effective_streaming:
        logger.info("Using streaming dataset. Tokenization will be applied on-the-fly.")    
        # Now, tokenized_train_dataset should only contain examples with {args.text_column: "content..."}
        # Proceed with tokenization on this cleaned dataset.
        def minimalistic_streaming_tokenize_function(examples):
            # examples here will be a batch dict, e.g., {args.text_column: [list_of_strings]}
            return tokenizer(examples[args.text_column], truncation=True, max_length=args.sequence_length)
        
        logger.info(f"Applying tokenization on column '{args.text_column}'.")
        tokenized_train_dataset = tokenized_train_dataset.map(
            minimalistic_streaming_tokenize_function, 
            batched=True, 
            remove_columns=[args.text_column] # Remove the text column after tokenization
        )
    else: # Non-streaming path
        def non_streaming_tokenize_function(examples):
            return tokenizer(examples[args.text_column], truncation=True, max_length=args.sequence_length, return_special_tokens_mask=True)
        with accelerator.main_process_first():
            logger.info("Applying 'non_streaming_tokenize_function' for non-streaming dataset.")
            # train_dataset_not_iterable was assigned above
            tokenized_datasets_dict = train_dataset_not_iterable.map( # Apply to the specific split
                non_streaming_tokenize_function,
                batched=True,
                remove_columns=train_dataset_not_iterable.column_names,
                desc="Running tokenizer on dataset (non-streaming)",
            )
        tokenized_train_dataset = tokenized_datasets_dict # This is now the tokenized map-style dataset
        
    # We need to define a collate function for dynamic padding for LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # MLM=False for Causal LM

    # Create DataLoader
    # For IterableDataset, shuffle is typically False at DataLoader level, handled by dataset.
    train_dataloader = DataLoader(
        tokenized_train_dataset,
        shuffle=not effective_streaming, 
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size
    )
    
    num_update_steps_per_epoch = "N/A (streaming)" # Default for streaming / unknown length

    if not effective_streaming:
        # This block is ONLY for non-streaming, map-style datasets where len() is defined
        if len(tokenized_train_dataset) == 0: # Check length on the dataset itself before DataLoader
            logger.error("Tokenized train dataset is empty for non-streaming mode. Check data and tokenization.")
            return
        if len(train_dataloader) == 0: 
            logger.error("Train dataloader is empty for non-streaming mode. This should not happen if dataset wasn't empty.")
            return
            
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            # If max_train_steps isn't set, derive it from num_train_epochs for non-streaming datasets
            if args.num_train_epochs is None: # Should have a default, but good to be safe
                args.num_train_epochs = 1 
                logger.warning("num_train_epochs was None, defaulting to 1 for max_train_steps calculation.")
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            # If max_train_steps is set, derive num_train_epochs for logging/looping purposes
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        logger.info(f"Number of training steps per epoch (for non-streaming): {num_update_steps_per_epoch}")
    else: # This block is for effective_streaming == True
        if args.max_train_steps is None:
            logger.error("max_train_steps must be set when using a streaming dataset or internal preparation.")
            return 
        logger.info("Using streaming dataset. Training will proceed for args.max_train_steps.")
        # num_train_epochs is less relevant here, training is governed by max_train_steps

    # --- 4. Optimizer ---
    if args.use_8bit_optimizer:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        logger.info("Using 8-bit AdamW optimizer.")
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        logger.info("Using standard AdamW optimizer.")

    # --- 5. Learning Rate Scheduler ---
    # Learning Rate Scheduler (num_training_steps needs to be robust)
    # The multiplication by accelerator.num_processes is for multi-GPU.
    # It assumes max_train_steps is the total number of optimizer steps on each process.
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        # Ensure num_warmup_steps and num_training_steps are scaled by num_processes if they represent per-process steps
        # However, typically max_train_steps is the *global* number of optimizer steps.
        # Let's assume args.max_train_steps is global. Accelerator handles scaling if necessary for its internal logic.
        # The Hugging Face convention is that num_training_steps for the scheduler is the total number of optimizer steps.
        num_warmup_steps=args.num_warmup_steps, 
        num_training_steps=args.max_train_steps 
    )
    logger.info(f"Using {args.lr_scheduler_type} scheduler with {args.num_warmup_steps} warmup steps.")

    # --- Prepare with Accelerator ---
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # --- Potentially compile the model ---
    if args.torch_compile and hasattr(torch, "compile"):
        if accelerator.is_main_process:
            logger.info("Attempting to compile the model with torch.compile...")
        # Unwrap the model before compiling if it's a DDP model
        unwrapped_model = accelerator.unwrap_model(model)
        try:
            compiled_model = torch.compile(unwrapped_model, mode="reduce-overhead") # or "max-autotune" for longer runs
            # Re-assign the compiled model to the original model variable if using DDP/FSDP
            # For single GPU with Accelerate, this should be okay.
            # If model was wrapped by DDP by accelerator, you'd need to handle this more carefully.
            # accelerator.prepare will handle wrapping the model correctly.
            # However, the model passed to accelerator.prepare should ideally be the one we want to train.
            # It's often better to compile then prepare, or compile the unwrapped_model.
            # Let's assume accelerator handles the already prepared model correctly if we compile its core.
            # This part might need adjustment based on Accelerator's DDP/FSDP wrapping.
            # For single GPU, `model = accelerator.prepare(compiled_model)` would be more direct if compiling before prepare.
            # Since we prepared already, we are compiling the potentially DDP-wrapped model's module.
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
                 model.module = compiled_model
            else:
                 model = compiled_model # If not wrapped, replace directly
            logger.info("Model compiled successfully.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Proceeding without compilation.")
            # compiled_model = unwrapped_model # Fallback
            # model = accelerator.prepare(compiled_model, optimizer, train_dataloader, lr_scheduler)[0] # Re-prepare
    
    # --- 6. Training Loop ---
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    if not effective_streaming:
        logger.info(f"  Num examples = {len(tokenized_train_dataset)}") 
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
    else:
        logger.info("  Num examples = N/A (streaming dataset)")
        logger.info(f"  Num Epochs = N/A (streaming, will run for {args.max_train_steps} steps)")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    
    # If not already initialized, define completed_steps here
    if 'completed_steps' not in locals() and 'completed_steps' not in globals():
        completed_steps = 0

    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.resume_from_checkpoint)
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            logger.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            try:
                # Extract step number from checkpoint folder name (e.g., "checkpoint-500")
                resume_step_str = checkpoint_path.name.split("-")[-1]
                completed_steps = int(resume_step_str)
                logger.info(f"Resumed at optimizer step: {completed_steps}")
            except ValueError:
                logger.warning(
                    f"Could not parse step number from checkpoint folder name: {checkpoint_path.name}. "
                    "Optimizer and scheduler states are loaded. Progress bar might not be perfectly aligned initially."
                )
            except Exception as e:
                logger.error(f"An error occurred trying to parse step from checkpoint name: {e}")

        else:
            logger.warning(
                f"Resume checkpoint path specified but not found or not a directory: {args.resume_from_checkpoint}. "
                "Starting training from scratch."
            )
    else:
        logger.info("No checkpoint specified for resumption. Starting training from scratch.")

    # --- Training Loop ---
    # Update progress_bar initialization to account for resumed steps
    # The `total` for tqdm should be the total number of steps you intend to run *in total across all runs*.
    # If `args.max_train_steps` is the total desired steps, then initial should be `completed_steps`.
    
    logger.info("***** Running training *****")
    # ... (your existing logging for Num examples, Epochs, etc.)
    logger.info(f"  Initial optimizer steps completed (if resuming): {completed_steps}")
    logger.info(f"  Target total optimizer steps for this run: {args.max_train_steps}")


    progress_bar = tqdm(
        range(args.max_train_steps),  # Total steps for the progress bar
        initial=completed_steps,      # Start from this step
        disable=not accelerator.is_local_main_process
    )
    # Ensure progress bar visually starts at the right place if initial > 0
    # `progress_bar.update(0)` can sometimes help refresh display if `initial` doesn't immediately reflect.
    # Or, more directly for some tqdm versions:
    # progress_bar.n = completed_steps 
    # progress_bar.refresh()

    starting_epoch = 0 # Can be adapted for resuming from checkpoint


    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss_this_epoch = 0.0 # Initialize for the current epoch
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss_this_epoch += loss.detach().item() # Use .item() for scalar CPU float
                accelerator.backward(loss)
                
                # Gradient clipping (optional, but good practice)
                if accelerator.sync_gradients: # Ensure gradients are synced before clipping
                    accelerator.clip_grad_norm_(model.parameters(), 1.0) 

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                avg_epoch_loss = total_loss_this_epoch / (step + 1) 
                progress_bar.set_description(
                    f"Epoch {epoch+1} Step {completed_steps} LR: {lr_scheduler.get_last_lr()[0]:.2e} Loss: {avg_epoch_loss:.4f}"
                )


                if completed_steps % args.generation_steps == 0 and completed_steps > 0:
                    if accelerator.is_main_process:
                        logger.info(f"\n--- Generating sample at step {completed_steps} ---")
                        model.eval() # Already there
                        unwrapped_model = accelerator.unwrap_model(model)

                        # Tokenize the prompt and get attention_mask
                        prompt_encoding = tokenizer(
                            args.generation_prompt, 
                            return_tensors="pt", 
                            padding=True, # Pad if prompt is shorter than a model's internal minimum (usually not an issue for short prompts)
                            truncation=True, 
                            max_length=args.sequence_length # Ensure prompt isn't longer than model can handle
                        )
                        prompt_ids = prompt_encoding.input_ids.to(accelerator.device)
                        attention_mask = prompt_encoding.attention_mask.to(accelerator.device)

                        try:
                            with torch.no_grad():
                                generated_ids = unwrapped_model.generate(
                                    prompt_ids,
                                    attention_mask=attention_mask, # Pass attention_mask
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=True,
                                    top_k=10, # Reduced from 50 for potentially more coherent initial gibberish
                                    top_p=0.95,
                                    temperature=0.8,
                                    pad_token_id=tokenizer.eos_token_id # Explicitly set pad_token_id for open-end generation
                                )
                            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                            logger.info(f"Prompt: {args.generation_prompt}")
                            logger.info(f"Generated: {generated_text}\n")
                        except Exception as e:
                            logger.error(f"Error during generation: {e}")
                        model.train() # Already there

                if completed_steps % args.save_steps == 0 and completed_steps > 0:
                    output_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{completed_steps}")
                    accelerator.save_state(output_checkpoint_dir)
                    if accelerator.is_main_process:
                        # Save model config with the checkpoint
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained( # Save the unwrapped model
                           output_checkpoint_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                        )
                        # Save tokenizer
                        tokenizer.save_pretrained(output_checkpoint_dir)
                        logger.info(f"Saved checkpoint to {output_checkpoint_dir}")


            if completed_steps >= args.max_train_steps:
                break
        if completed_steps >= args.max_train_steps:
            break
    
    # --- 7. Save Final Model ---
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info(f"Saving final model to {args.output_dir}")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            tokenizer.save_pretrained(args.output_dir)
            # Save final training args
            with open(os.path.join(args.output_dir, "training_args_final.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
            
            # Clean up intermediate checkpoints if desired
            # for step_num in range(args.save_steps, completed_steps, args.save_steps):
            #     chkpt_dir = os.path.join(args.output_dir, f"checkpoint-{step_num}")
            #     if os.path.exists(chkpt_dir):
            #         logger.info(f"Cleaning up checkpoint: {chkpt_dir}")
            #         shutil.rmtree(chkpt_dir)

    accelerator.end_training()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()