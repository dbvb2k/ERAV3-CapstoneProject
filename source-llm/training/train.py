# train.py
import argparse
import json
import os
import torch
import logging
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from torch.utils.data import DataLoader
from datasets import load_from_disk
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import bitsandbytes as bnb

# Set up logging
logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a SmolLM-1.7B model with Accelerate and DeepSpeed")
    
    # Model and Tokenizer
    parser.add_argument("--model_config_name", type=str, default="HuggingFaceTB/SmolLM2-1.7B", help="Model config name.")
    parser.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM2-1.7B", help="Tokenizer name.")
    parser.add_argument("--pretokenized_dataset_path", type=str, required=True, help="Path to the pre-tokenized dataset on disk.")
    
    # Training Hyperparameters
    parser.add_argument("--output_dir", type=str, default="./smollm_training_output", help="Output directory for checkpoints and final model.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Steps for gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Peak learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--max_train_steps", type=int, default=15000, help="Total number of training steps.") # e.g., ~5 epochs on 100k samples
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--num_warmup_steps", type=int, default=300, help="Number of warmup steps for the scheduler.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # Checkpointing and Logging
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Report logs to 'wandb' or 'tensorboard'.")

    # Optimization Flags (to "start limited" as you requested)
    parser.add_argument("--use_8bit_optimizer", action="store_true", help="Enable 8-bit AdamW optimizer.")
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for the model.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Initialize Accelerator ---
    # `accelerate` will handle DeepSpeed, DDP, device placement, etc. based on `accelerate config`
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=args.output_dir
    )

    # --- Setup Logging ---
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info(accelerator.state, main_process_only=False)

    set_seed(args.seed)

    # --- Load Pre-Tokenized Dataset ---
    with accelerator.main_process_first():
        # This will only be downloaded/processed once by the main process
        logger.info(f"Loading pre-tokenized dataset from {args.pretokenized_dataset_path}")
        train_dataset = load_from_disk(args.pretokenized_dataset_path)
    
    # --- Load Tokenizer (for DataCollator) and Model Config ---
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    config = AutoConfig.from_pretrained(args.model_config_name)
    config.use_cache = False # Essential for training
    config.gradient_checkpointing = True # Essential for memory saving
    
    logger.info("Initializing model with Flash Attention 2...")
    model = AutoModelForCausalLM.from_config(
        config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )

    # --- Data Loader ---
    # Data collator will handle dynamic padding and creating labels
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, # Can shuffle map-style datasets
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=8 # Now robust and will provide a massive speedup
    )

    # --- Optimizer ---
    # Create grouped parameters to exclude bias and LayerNorm from weight decay
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if args.use_8bit_optimizer:
        logger.info("Using 8-bit AdamW optimizer.")
        optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        logger.info("Using standard AdamW optimizer.")
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # --- Learning Rate Scheduler ---
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # --- Compile Model (if enabled) ---
    if args.torch_compile:
        logger.info("Compiling the model with torch.compile...")
        model = torch.compile(model)

    # --- Prepare with Accelerator ---
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # --- Training Loop ---
    logger.info("***** Starting Training *****")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1
            
            if accelerator.is_main_process:
                progress_bar.set_description(f"Step {completed_steps} Loss: {loss.detach().float().item():.4f}")
                
            if report_to_wandb := (args.report_to == "wandb"):
                accelerator.log({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=completed_steps)

            if completed_steps % args.save_steps == 0:
                output_dir = f"checkpoint-{completed_steps}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

        if completed_steps >= args.max_train_steps:
            break

    logger.info("***** Training Complete *****")
    
    # --- Save Final Model ---
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
    
    
""" 
Example Usage:
accelerate launch train.py \
    --model_config_name "HuggingFaceTB/SmolLM2-1.7B" \
    --tokenizer_name "HuggingFaceTB/SmolLM2-1.7B" \
    --pretokenized_dataset_path "/home/ubuntu/data/final_training/Day4/New_Approach/cosmopedia-100k/cosmopedia-100k-tokenized" \
    --output_dir "/home/ubuntu/data/final_training/Day4/New_Approach/cosmopedia-100k/smollm-1.7B-cosmo-test-run" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-5 \
    --max_train_steps 15000 \
    --num_warmup_steps 300 \
    --save_steps 1000 \
    --report_to "wandb" 
"""

# --use_8bit_optimizer False \
#     --torch_compile False