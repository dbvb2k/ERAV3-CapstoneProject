import argparse
import logging
import os
import time
import glob
import datetime
import signal
import sys
import gc
import shutil
from pathlib import Path
import torch
import torch.nn as nn
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

# Set up logging
logger = get_logger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle spot instance interruption signals"""
    global shutdown_requested
    print("Spot instance interruption detected. Saving checkpoint...")
    log_message("INTERRUPTION: Spot instance termination signal received")
    shutdown_requested = True

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
    parser.add_argument("--max_train_steps", type=int, default=15000, help="Total number of training steps.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--num_warmup_steps", type=int, default=300, help="Number of warmup steps for the scheduler.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # Checkpointing and Logging
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps.")
    parser.add_argument("--log_steps", type=int, default=50, help="Log progress every X steps.")
    parser.add_argument("--generation_steps", type=int, default=1000, help="Generate sample text every X steps.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Report logs to 'wandb' or 'tensorboard'.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")

    # Optimization Flags
    parser.add_argument("--use_8bit_optimizer", action="store_true", help="Enable 8-bit AdamW optimizer.")
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for the model.")
    parser.add_argument("--empty_cache_steps", type=int, default=50, help="Empty CUDA cache every N steps.")
    parser.add_argument("--max_memory_mb", type=int, default=20000, help="Maximum memory usage in MB per GPU.")
    
    return parser.parse_args()

def log_message(message, log_file=None):
    """Log messages to markdown file with timestamp"""
    if log_file is None:
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"**{timestamp}** - {message}\n\n")

def initialize_log(log_file, args):
    """Initialize the training log file"""
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("# SmolLM2 Training Log\n\n")
        f.write(f"**Training Started:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Configuration\n")
        f.write(f"- Model: {args.model_config_name}\n")
        f.write(f"- Dataset: {args.pretokenized_dataset_path}\n")
        f.write(f"- Batch Size per Device: {args.per_device_train_batch_size}\n")
        f.write(f"- Gradient Accumulation: {args.gradient_accumulation_steps}\n")
        f.write(f"- Learning Rate: {args.learning_rate}\n")
        f.write(f"- Max Train Steps: {args.max_train_steps:,}\n")
        f.write(f"- Warmup Steps: {args.num_warmup_steps}\n")
        f.write(f"- Log Interval: {args.log_steps} steps\n")
        f.write(f"- Checkpoint Interval: {args.save_steps} steps\n")
        f.write(f"- Generation Interval: {args.generation_steps} steps\n")
        f.write(f"- Output Directory: {args.output_dir}\n\n")
        f.write("## Training Progress\n\n")
        f.write("| Step | Loss | Learning Rate | GPU Memory | Generation Sample |\n")
        f.write("|------|------|---------------|------------|-------------------|\n")

def log_training_step(step, total_steps, loss, lr, gpu_memory, generation_sample, log_file, epoch=0):
    """Log training step details"""
    with open(log_file, "a", encoding="utf-8") as f:
        generation_text = generation_sample.replace('\n', ' ').replace('|', '\\|')[:50] + "..." if generation_sample else "-"
        f.write(f"| Step {step:,}/{total_steps:,} | Epoch {epoch} | Loss: {loss:.4f} | LR: {lr:.2e} | GPU: {gpu_memory:.1f}GB | {generation_text} |\n")

def log_checkpoint(step, loss, log_file):
    """Log checkpoint saves"""
    log_message(f"CHECKPOINT: Saved at step {step:,}, loss {loss:.4f}", log_file)

def generate_sample_text(model, tokenizer, accelerator, step):
    """Generate sample text during training"""
    model.eval()
    
    # Sample prompts for generation
    test_prompts = [
        "The weather today is very",
        "Machine learning is a field of",
        "In the future, technology will",
        "The most important thing in life is"
    ]
    
    prompt = test_prompts[step % len(test_prompts)]
    
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
            
            # Generate with controlled parameters
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Return only the new part
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            model.train()
            return f"{prompt} {generated_text}"
            
    except Exception as e:
        logger.warning(f"Generation failed at step {step}: {e}")
        model.train()
        return None

def aggressive_memory_cleanup():
    """Ultra-aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in the output directory"""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    return checkpoints[-1]

def cleanup_old_checkpoints(output_dir, keep_last=3):
    """Keep only the last N checkpoints to save disk space"""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if len(checkpoints) <= keep_last:
        return
    
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    
    # Remove old checkpoints
    for checkpoint_dir in checkpoints[:-keep_last]:
        try:
            shutil.rmtree(checkpoint_dir)
            logger.info(f"Removed old checkpoint: {checkpoint_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {checkpoint_dir}: {e}")

def log_system_stats(step, log_file):
    """Log system statistics"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_used = torch.cuda.memory_allocated(i) / 1e9
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            utilization = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
            
            log_message(f"GPU {i}: {memory_used:.1f}GB/{memory_total:.1f}GB ({utilization}% util)", log_file)


def main():
    args = parse_args()
    
    # CRITICAL FIX 1: Declare shutdown_requested in main scope
    global shutdown_requested
    shutdown_requested = False
    
    # Register signal handlers for spot instance interruptions
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set environment variables for memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # --- Initialize Accelerator ---
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=args.output_dir
    )

    # --- Setup Logging ---
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info(accelerator.state, main_process_only=False)

    # Create output directory and log file
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training_log.md")
    
    # Initialize log file
    if accelerator.is_main_process:
        initialize_log(log_file, args)
        log_message("Training started with Accelerate", log_file)

    set_seed(args.seed)

    # --- Load Pre-Tokenized Dataset ---
    with accelerator.main_process_first():
        logger.info(f"Loading pre-tokenized dataset from {args.pretokenized_dataset_path}")
        train_dataset = load_from_disk(args.pretokenized_dataset_path)
        logger.info(f"Dataset loaded with {len(train_dataset):,} samples")
    
    # --- Load Tokenizer and Model Config ---
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
        
    config = AutoConfig.from_pretrained(args.model_config_name)
    config.use_cache = False  # Essential for training
    config.gradient_checkpointing = True  # Essential for memory saving
    
    logger.info("Initializing model with Flash Attention 2...")
    model = AutoModelForCausalLM.from_config(
        config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # --- Data Loader ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=4,  # Reduced for memory
        pin_memory=False,  # Disabled for memory
        persistent_workers=True,
        drop_last=True,  # Ensure full batches
        prefetch_factor=1
    )

    # --- Optimizer ---
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if args.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
            logger.info("Using 8-bit AdamW optimizer")
        except ImportError:
            logger.warning("bitsandbytes not available, using standard AdamW")
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )

    # --- Learning Rate Scheduler ---
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # --- Prepare with Accelerator ---
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Apply torch.compile if requested
    if args.torch_compile:
        try:
            model = torch.compile(model)
            logger.info("Applied torch.compile to model")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # --- Resume from checkpoint ---
    starting_step = 0
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            # Extract step number from checkpoint path
            try:
                starting_step = int(args.resume_from_checkpoint.split('-')[-1])
                logger.info(f"Resumed from step {starting_step}")
                if accelerator.is_main_process:
                    log_message(f"RESUMED: From checkpoint at step {starting_step}", log_file)
            except:
                logger.warning("Could not extract step number from checkpoint path")
        else:
            logger.warning(f"Checkpoint not found: {args.resume_from_checkpoint}")
    else:
        # Check for latest checkpoint in output directory
        latest_checkpoint = find_latest_checkpoint(args.output_dir)
        if latest_checkpoint:
            logger.info(f"Found existing checkpoint: {latest_checkpoint}")
            response = input("Resume from latest checkpoint? (y/n): ").lower().strip()
            if response == 'y':
                accelerator.load_state(latest_checkpoint)
                try:
                    starting_step = int(latest_checkpoint.split('-')[-1])
                    logger.info(f"Resumed from step {starting_step}")
                    if accelerator.is_main_process:
                        log_message(f"RESUMED: From latest checkpoint at step {starting_step}", log_file)
                except:
                    logger.warning("Could not extract step number from checkpoint path")

    # Clear cache after preparation
    aggressive_memory_cleanup()

    # --- Training Loop ---
    logger.info("***** Starting Training *****")
    logger.info(f"  Num examples = {len(train_dataset):,}")
    logger.info(f"  Num Epochs = {args.max_train_steps // len(train_dataloader) + 1}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps:,}")
    logger.info(f"  Starting from step = {starting_step}")

    progress_bar = tqdm(range(starting_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = starting_step
    running_loss = 0.0
    
    # Skip batches if resuming
    dataloader_iter = iter(train_dataloader)
    if starting_step > 0:
        steps_to_skip = starting_step % len(train_dataloader)
        for _ in range(steps_to_skip):
            try:
                next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_dataloader)
                break

    model.train()
    
    try:
        while completed_steps < args.max_train_steps:
            if shutdown_requested:
                break
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_dataloader)
                batch = next(dataloader_iter)
            
            # Memory management
            if completed_steps % args.empty_cache_steps == 0 and completed_steps > starting_step:
                aggressive_memory_cleanup()
                
            outputs = model(**batch)
            loss = outputs.loss
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Only update progress on gradient sync
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                running_loss += loss.detach().float().item()
                
                # Logging
                if completed_steps % args.log_steps == 0:
                    avg_loss = running_loss / args.log_steps
                    current_lr = lr_scheduler.get_last_lr()[0]
                    
                    gpu_memory = 0
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1e9
                    
                    if accelerator.is_main_process:
                        progress_bar.set_description(f"Step {completed_steps} Loss: {avg_loss:.4f}")
                        
                    if args.report_to == "wandb" and accelerator.is_main_process:
                        accelerator.log({
                            "loss": avg_loss,
                            "learning_rate": current_lr,
                            "step": completed_steps
                        }, step=completed_steps)
                    
                    # Generate sample text occasionally
                    generation_sample = None
                    if completed_steps % args.generation_steps == 0:
                        generation_sample = generate_sample_text(model, tokenizer, accelerator, completed_steps)
                        if generation_sample and accelerator.is_main_process:
                            logger.info(f"Generation sample at step {completed_steps}: {generation_sample}")
                            log_message(f"GENERATION at step {completed_steps}: {generation_sample}", log_file)
                    
                    # Log to markdown file
                    if accelerator.is_main_process:
                        log_training_step(completed_steps, args.max_train_steps, avg_loss, current_lr, gpu_memory, generation_sample, log_file)
                        log_system_stats(completed_steps, log_file)
                    
                    running_loss = 0.0

                # Save checkpoint
                if completed_steps % args.save_steps == 0:
                    output_dir = f"checkpoint-{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    
                    accelerator.save_state(output_dir)
                    
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint at step {completed_steps}")
                        log_checkpoint(completed_steps, loss.detach().float().item(), log_file)
                        cleanup_old_checkpoints(args.output_dir)
                    
                    # Clear cache after saving
                    aggressive_memory_cleanup()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if accelerator.is_main_process:
            log_message("Training interrupted by user", log_file)
        shutdown_requested = True
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if accelerator.is_main_process:
            log_message(f"ERROR: Training failed: {e}", log_file)
        raise
    finally:
        # Always save final checkpoint
        if completed_steps > starting_step:
            final_output_dir = os.path.join(args.output_dir, f"checkpoint-{completed_steps}")
            accelerator.save_state(final_output_dir)
            
            if accelerator.is_main_process:
                logger.info(f"Saved final checkpoint at step {completed_steps}")
                log_message(f"FINAL: Training ended at step {completed_steps}", log_file)

    # Save final model
    if accelerator.is_main_process:
        logger.info("Saving final model...")
        final_model_dir = os.path.join(args.output_dir, "final_model")
        accelerator.unwrap_model(model).save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        log_message(f"Final model saved to {final_model_dir}", log_file)

    logger.info(f"Training completed! Total steps: {completed_steps}")

if __name__ == "__main__":
    main()

""" 
Test Usage (Worked)
accelerate launch train2.py \
    --model_config_name "HuggingFaceTB/SmolLM2-1.7B" \
    --tokenizer_name "HuggingFaceTB/SmolLM2-1.7B" \
    --pretokenized_dataset_path "/home/ubuntu/bigdata/Training/Day4/cosmopedia-v2-1B/cosmopedia-v2-1B-tokenized" \
    --output_dir "/home/ubuntu/bigdata/Training/Day4/cosmopedia-v2-1B/smollm-1.7B-cosmo-1B-test-run" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-5 \
    --max_train_steps 15000 \
    --num_warmup_steps 300 \
    --save_steps 1000 \
    --log_steps 50 \
    --generation_steps 1000 \
    --report_to "wandb" \
    --use_8bit_optimizer \
    --empty_cache_steps 50 \
    --resume_from_checkpoint "/home/ubuntu/bigdata/Training/Day4/cosmopedia-v2-1B/smollm-1.7B-cosmo-1B-test-run/checkpoint-1082"

# To resume from a specific checkpoint:
accelerate launch train.py \
    --resume_from_checkpoint "/path/to/checkpoint-5000" \
    [other arguments...]
"""

""" 
Production Usage (Working)
accelerate launch train3.py \
    --model_config_name "HuggingFaceTB/SmolLM2-1.7B" \
    --tokenizer_name "HuggingFaceTB/SmolLM2-1.7B" \
    --pretokenized_dataset_path "/home/ubuntu/bigdata/Training/Day4/cosmopedia-v2-1B/cosmopedia-v2-1B-tokenized" \
    --output_dir "/home/ubuntu/bigdata/Training/Day4/cosmopedia-v2-1B/smollm-1.7B-cosmo-1B-production" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --max_train_steps 50000 \
    --num_warmup_steps 1000 \
    --save_steps 2000 \
    --log_steps 500 \
    --generation_steps 2000 \
    --report_to "wandb" \
    --use_8bit_optimizer \
    --empty_cache_steps 100
"""