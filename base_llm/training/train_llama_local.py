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

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
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

    # Dataset and Tokenizer
    # IMPORTANT: Modify these for your specific dataset
    parser.add_argument("--dataset_name", type=str, default="stas/openwebtext-10k", help="Hugging Face dataset name or path to a local dataset.") # Example small dataset
    parser.add_argument("--dataset_config_name", type=str, default=None, help="Specific dataset configuration (if any).")
    parser.add_argument("--text_column", type=str, default="text", help="The name of the column in the dataset containing the text.")
    parser.add_argument("--tokenizer_name", type=str, default="hf-internal-testing/llama-tokenizer", help="Tokenizer to use (e.g., from a pretrained Llama model).")
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
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    
    # Checkpointing and Generation
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X update steps.")
    parser.add_argument("--generation_steps", type=int, default=50, help="Generate a sample every X update steps.")
    parser.add_argument("--generation_prompt", type=str, default="Once upon a time,", help="Prompt for sample generation.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max new tokens for sample generation.")

    # Performance & Memory Optimization
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose bf16 if supported (RTX 30xx/40xx), else fp16.")
    parser.add_argument("--use_gradient_checkpointing", type=bool, default=True, help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--use_8bit_optimizer", type=bool, default=True, help="Use 8-bit AdamW optimizer from bitsandbytes.")
    parser.add_argument("--torch_compile", type=bool, default=False, help="Enable torch.compile for potential speedups (experimental, requires PyTorch 2.0+). Might add overhead for short runs.")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
        log_with="tensorboard", # or "wandb"
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
    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads, # For MHA, same as num_attention_heads
        max_position_embeddings=args.max_position_embeddings,
        initializer_range=args.initializer_range,
        rms_norm_eps=args.rms_norm_eps,
        use_cache=False, # Important for training
        tie_word_embeddings=args.tie_word_embeddings,
        hidden_act="silu" # Standard for Llama
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
    # For local development, use a small dataset. For actual training, use streaming=True for large datasets.
    # Example: raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, streaming=True)
    # For this local script, we'll assume a smaller, non-streaming dataset for simplicity.
    try:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        # For "stas/openwebtext-10k", it has a 'train' split. Adjust if your dataset has different split names.
        if "train" not in raw_datasets:
            # If no 'train' split, try to use the first available split or handle error
            available_splits = list(raw_datasets.keys())
            if not available_splits:
                raise ValueError("No splits found in the dataset.")
            logger.warning(f"'train' split not found. Using '{available_splits[0]}' split instead.")
            raw_datasets["train"] = raw_datasets[available_splits[0]]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Please ensure your dataset is correctly specified and accessible.")
        logger.error("For local testing, 'stas/openwebtext-10k' is a good small dataset.")
        logger.error("For larger datasets on cloud, you'll need to use streaming and point to the correct data source.")
        return


    def tokenize_function(examples):
        return tokenizer(examples[args.text_column], truncation=True, max_length=args.sequence_length, return_special_tokens_mask=True)

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names, # Remove original text column
            desc="Running tokenizer on dataset",
        )

    # We need to define a collate function for dynamic padding for LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # MLM=False for Causal LM

    # Create DataLoader
    train_dataset = tokenized_datasets["train"]
    # For testing, limit the size of the dataset if it's too large for quick iteration
    # if args.max_train_steps is not None and len(train_dataset) > args.max_train_steps * args.per_device_train_batch_size * args.gradient_accumulation_steps :
    #     train_dataset = train_dataset.select(range(args.max_train_steps * args.per_device_train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes))
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size
    )

    # --- 4. Optimizer ---
    if args.use_8bit_optimizer:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        logger.info("Using 8-bit AdamW optimizer.")
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        logger.info("Using standard AdamW optimizer.")

    # --- 5. Learning Rate Scheduler ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes, # Adjust warmup steps for grad accumulation
        num_training_steps=args.max_train_steps * accelerator.num_processes, # Adjust total steps for grad accumulation
    )

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
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0 # Can be adapted for resuming from checkpoint

    # Potentially load checkpoint
    # TODO: Implement checkpoint loading if needed for resuming

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float() # Accumulate loss for logging
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
                avg_loss = total_loss / (step +1) # Or over completed_steps if accumulated loss is reset
                progress_bar.set_description(f"Epoch {epoch+1} Step {completed_steps} LR: {lr_scheduler.get_last_lr()[0]:.2e} Loss: {avg_loss:.4f}")


                if completed_steps % args.generation_steps == 0 and completed_steps > 0:
                    if accelerator.is_main_process:
                        logger.info(f"\n--- Generating sample at step {completed_steps} ---")
                        model.eval()
                        unwrapped_model = accelerator.unwrap_model(model)
                        prompt_ids = tokenizer(args.generation_prompt, return_tensors="pt").input_ids.to(accelerator.device)
                        
                        try:
                            with torch.no_grad(): # Ensure no gradients are computed
                                generated_ids = unwrapped_model.generate(
                                    prompt_ids,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=True,
                                    top_k=10,
                                    top_p=0.95,
                                    temperature=0.8 # Add some creativity
                                )
                            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                            logger.info(f"Prompt: {args.generation_prompt}")
                            logger.info(f"Generated: {generated_text}\n")
                        except Exception as e:
                            logger.error(f"Error during generation: {e}")
                        model.train()

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