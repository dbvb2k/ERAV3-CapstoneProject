#!/usr/bin/env python3
"""
Meta-Llama-3-8B Instruction Fine-tuning with Alpaca Dataset
Optimized for RunPod GPU instances with comprehensive monitoring and validation.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Dataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import psutil
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# Setup comprehensive logging system
def setup_comprehensive_logging():
    """Setup comprehensive logging to files for training and inference."""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training_complete.log'),  # Complete training log
            logging.StreamHandler(sys.stdout)  # Console output
        ]
    )
    
    # Create separate loggers for different components
    training_logger = logging.getLogger('training')
    training_handler = logging.FileHandler('logs/training_metrics.log')
    training_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    training_logger.addHandler(training_handler)
    training_logger.setLevel(logging.INFO)
    
    inference_logger = logging.getLogger('inference')
    inference_handler = logging.FileHandler('logs/inference_results.log')
    inference_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    inference_logger.addHandler(inference_handler)
    inference_logger.setLevel(logging.INFO)
    
    system_logger = logging.getLogger('system')
    system_handler = logging.FileHandler('logs/system_monitoring.log')
    system_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    system_logger.addHandler(system_handler)
    system_logger.setLevel(logging.INFO)
    
    return training_logger, inference_logger, system_logger

# Initialize comprehensive logging
training_logger, inference_logger, system_logger = setup_comprehensive_logging()

# Setup main logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_complete.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SystemValidator:
    """Validates system requirements and setup."""
    
    @staticmethod
    def check_gpu():
        """Check GPU availability and memory."""
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA is not available. GPU required for training.")
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"✅ Found {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return gpu_count
    
    @staticmethod
    def check_memory():
        """Check system memory."""
        memory = psutil.virtual_memory()
        memory_gb = memory.total / 1e9
        logger.info(f"✅ System RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 16:
            logger.warning("⚠️  Low system memory. Consider reducing batch size.")
        
        return memory_gb
    
    @staticmethod
    def check_disk_space():
        """Check available disk space."""
        disk = psutil.disk_usage('.')
        free_gb = disk.free / 1e9
        logger.info(f"✅ Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 50:
            logger.warning("⚠️  Low disk space. Models and datasets require ~20-30 GB.")
        
        return free_gb
    
    @staticmethod
    def create_directories():
        """Create necessary directories."""
        dirs = ['outputs', 'logs', 'checkpoints', 'data', 'plots']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
        logger.info("✅ Created project directories")

class DatasetProcessor:
    """Handles dataset loading and preprocessing."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_alpaca_dataset(self) -> Dataset:
        """Load and validate Alpaca instruction dataset."""
        logger.info("📥 Loading Alpaca instruction dataset...")
        
        try:
            dataset = load_dataset("tatsu-lab/alpaca", split="train")
            logger.info(f"✅ Loaded {len(dataset)} instruction examples")
            
            # Sample a few examples for validation
            sample = dataset.select(range(min(3, len(dataset))))
            for i, example in enumerate(sample):
                logger.info(f"   Example {i+1}: {example['instruction'][:100]}...")
            
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise
    
    def format_instruction(self, example: Dict) -> Dict:
        """Format instruction examples into training format."""
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]
        
        # Create the prompt format
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        full_text = prompt + output
        
        return {"text": full_text}
    
    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        # Tokenize the full text
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )
        
        # Set labels (for causal language modeling, labels = input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for training."""
        logger.info("🔄 Formatting and tokenizing dataset...")
        
        # Format instructions
        formatted_dataset = dataset.map(
            self.format_instruction,
            remove_columns=dataset.column_names,
            desc="Formatting instructions"
        )
        
        # Tokenize
        tokenized_dataset = formatted_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=formatted_dataset.column_names,
            desc="Tokenizing"
        )
        
        logger.info(f"✅ Prepared {len(tokenized_dataset)} training examples")
        return tokenized_dataset

class ModelTrainer:
    """Handles model loading and training configuration."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer without quantization."""
        logger.info(f"🤖 Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model without quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Ensure model config matches tokenizer
        if hasattr(self.model.config, 'pad_token_id') and self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        logger.info("✅ Model and tokenizer loaded successfully")
        return self.model, self.tokenizer
    
    def setup_lora(self, r: int = 64, alpha: int = 16, dropout: float = 0.1):
        """Setup LoRA for parameter-efficient fine-tuning."""
        logger.info("🔧 Setting up LoRA configuration...")
        
        # Ensure model is in training mode and gradients are enabled
        self.model.train()
        
        # Enable gradient computation for the model
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze base model first
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Ensure LoRA parameters require gradients
        self.model.train()
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
        
        # Additional fix: Enable gradient checkpointing for PEFT model
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()
            logger.info("   ✅ Enabled input gradients for PEFT model")
        
        # Ensure the model is properly prepared for training
        if hasattr(self.model, 'prepare_inputs_for_generation'):
            # This ensures the model wrapper is properly set up
            pass
        
        # Verify gradient setup
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        if trainable_params == 0:
            raise RuntimeError("No trainable parameters found! LoRA setup failed.")
        
        logger.info(f"✅ LoRA setup complete:")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Percentage: {100 * trainable_params / total_params:.2f}%")
        
        # Verify some parameters actually require gradients
        grad_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
        logger.info(f"   Sample trainable parameters: {grad_params[:5]}")
        
        return self.model

class TrainingMonitor:
    """Monitors training progress and generates visualizations with comprehensive logging."""
    
    def __init__(self):
        self.training_history = []
        self.inference_history = []
        self.training_logger = logging.getLogger('training')
        self.inference_logger = logging.getLogger('inference')
        self.system_logger = logging.getLogger('system')
        
        # Initialize metrics file with headers
        self._initialize_metrics_file()
        
    def _initialize_metrics_file(self):
        """Initialize CSV file for structured metrics logging."""
        metrics_file = "logs/training_metrics.csv"
        if not os.path.exists(metrics_file):
            with open(metrics_file, 'w') as f:
                f.write("timestamp,step,epoch,loss,eval_loss,learning_rate,gpu_memory_gb,train_samples_per_second\n")
    
    def log_metrics(self, logs: Dict[str, Any]):
        """Log training metrics to both console and files."""
        # Add timestamp and system info
        logs['timestamp'] = time.time()
        logs['gpu_memory_gb'] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        self.training_history.append(logs.copy())
        
        # Display current metrics
        if 'loss' in logs:
            message = f"Step {logs.get('step', 'N/A')}: Loss = {logs['loss']:.6f}"
            if 'learning_rate' in logs:
                message += f", LR = {logs['learning_rate']:.2e}"
            if 'train_samples_per_second' in logs:
                message += f", Speed = {logs['train_samples_per_second']:.1f} samples/sec"
            
            logger.info(f"📊 {message}")
            self.training_logger.info(f"TRAINING_STEP: {message}")
        
        if 'eval_loss' in logs:
            eval_message = f"Evaluation Loss: {logs['eval_loss']:.6f}"
            logger.info(f"📊 {eval_message}")
            self.training_logger.info(f"EVALUATION: {eval_message}")
        
        # Write to CSV for structured analysis
        self._write_metrics_to_csv(logs)
        
        # Write detailed JSON log entry
        self._write_detailed_log(logs)
        
        # Log system resources
        self._log_system_resources()
    
    def _write_metrics_to_csv(self, logs: Dict[str, Any]):
        """Write metrics to CSV file for easy analysis."""
        metrics_file = "logs/training_metrics.csv"
        
        # Extract key metrics
        timestamp = logs.get('timestamp', time.time())
        step = logs.get('step', '')
        epoch = logs.get('epoch', '')
        loss = logs.get('loss', '')
        eval_loss = logs.get('eval_loss', '')
        learning_rate = logs.get('learning_rate', '')
        gpu_memory = logs.get('gpu_memory_gb', '')
        samples_per_sec = logs.get('train_samples_per_second', '')
        
        with open(metrics_file, 'a') as f:
            f.write(f"{timestamp},{step},{epoch},{loss},{eval_loss},{learning_rate},{gpu_memory},{samples_per_sec}\n")
    
    def _write_detailed_log(self, logs: Dict[str, Any]):
        """Write detailed JSON log entry."""
        log_entry = {
            'timestamp': time.time(),
            'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': logs.copy()
        }
        
        # Save to detailed JSON log
        json_log_file = "logs/training_detailed.jsonl"
        with open(json_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _log_system_resources(self):
        """Log current system resource usage."""
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1e9
            gpu_memory_cached = torch.cuda.memory_reserved() / 1e9
            gpu_memory_max = torch.cuda.max_memory_allocated() / 1e9
            
            gpu_info = f"GPU Memory - Used: {gpu_memory_used:.2f}GB, Cached: {gpu_memory_cached:.2f}GB, Max: {gpu_memory_max:.2f}GB"
            self.system_logger.info(gpu_info)
        
        # System memory
        memory = psutil.virtual_memory()
        ram_used = (memory.total - memory.available) / 1e9
        ram_total = memory.total / 1e9
        ram_percent = memory.percent
        
        ram_info = f"RAM - Used: {ram_used:.2f}GB/{ram_total:.2f}GB ({ram_percent:.1f}%)"
        self.system_logger.info(ram_info)
    
    def run_inference_test(self, model, tokenizer, step: int):
        """Run quick inference test during training with detailed logging."""
        logger.info(f"🧪 Running inference test at step {step}...")
        self.inference_logger.info(f"=== INFERENCE TEST AT STEP {step} ===")
        
        test_prompts = [
            "### Instruction:\nWhat is machine learning?\n\n### Response:\n",
            "### Instruction:\nWrite a Python function to add two numbers.\n\n### Response:\n",
            "### Instruction:\nExplain the water cycle.\n\n### Response:\n",
            "### Instruction:\nHow do you make a peanut butter sandwich?\n\n### Response:\n",
            "### Instruction:\nWhat are the benefits of exercise?\n\n### Response:\n"
        ]
        
        model.eval()
        results = []
        
        for i, prompt in enumerate(test_prompts):
            try:
                start_time = time.time()
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                inference_time = time.time() - start_time
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.replace(prompt, "").strip()
                
                # Log a short version to console
                short_response = response[:150] + "..." if len(response) > 150 else response
                logger.info(f"   Test {i+1}: {short_response}")
                
                # Log full details to inference log
                self.inference_logger.info(f"TEST {i+1} (Step {step}):")
                self.inference_logger.info(f"  Prompt: {prompt.strip()}")
                self.inference_logger.info(f"  Response: {response}")
                self.inference_logger.info(f"  Response Length: {len(response)} chars")
                self.inference_logger.info(f"  Inference Time: {inference_time:.3f}s")
                self.inference_logger.info(f"  Tokens Generated: {len(outputs[0]) - len(inputs['input_ids'][0])}")
                self.inference_logger.info("---")
                
                results.append({
                    "step": step,
                    "test_number": i + 1,
                    "prompt": prompt,
                    "response": response,
                    "response_length": len(response),
                    "inference_time_seconds": inference_time,
                    "tokens_generated": len(outputs[0]) - len(inputs['input_ids'][0]),
                    "timestamp": time.time(),
                    "datetime": time.strftime('%Y-%m-%d %H:%M:%S')
                })
                
            except Exception as e:
                error_msg = f"Test {i+1} failed: {e}"
                logger.warning(f"   {error_msg}")
                self.inference_logger.error(f"TEST {i+1} ERROR: {error_msg}")
                
                results.append({
                    "step": step,
                    "test_number": i + 1,
                    "prompt": prompt,
                    "response": f"ERROR: {e}",
                    "response_length": 0,
                    "inference_time_seconds": 0,
                    "tokens_generated": 0,
                    "timestamp": time.time(),
                    "datetime": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "error": str(e)
                })
        
        # Save results to multiple formats
        self.inference_history.extend(results)
        
        # Save to JSON file for this specific step
        inference_file = f"outputs/inference_step_{step}.json"
        os.makedirs("outputs", exist_ok=True)
        with open(inference_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save to comprehensive inference log file
        comprehensive_file = "logs/inference_comprehensive.jsonl"
        with open(comprehensive_file, 'a') as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        # Calculate and log statistics
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            avg_length = sum(r['response_length'] for r in valid_results) / len(valid_results)
            avg_time = sum(r['inference_time_seconds'] for r in valid_results) / len(valid_results)
            total_tokens = sum(r['tokens_generated'] for r in valid_results)
            
            stats_msg = f"Inference Stats - Avg Length: {avg_length:.1f} chars, Avg Time: {avg_time:.3f}s, Total Tokens: {total_tokens}"
            logger.info(f"   📈 {stats_msg}")
            self.inference_logger.info(f"STEP_STATS: {stats_msg}")
        
        model.train()  # Switch back to training mode
        logger.info(f"✅ Inference test completed, saved to {inference_file}")
        self.inference_logger.info(f"=== INFERENCE TEST COMPLETED ===\n")
        
        return results
    
    def plot_training_progress(self):
        """Generate training progress plots with inference tracking."""
        if not self.training_history:
            return
        
        df = pd.DataFrame(self.training_history)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress Monitoring', fontsize=16, fontweight='bold')
        
        # Training loss
        if 'loss' in df.columns:
            axes[0, 0].plot(df.index, df['loss'], 'b-', linewidth=2, alpha=0.8)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rate' in df.columns:
            axes[0, 1].plot(df.index, df['learning_rate'], 'g-', linewidth=2, alpha=0.8)
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('LR')
            axes[0, 1].grid(True, alpha=0.3)
        
        # GPU memory usage
        axes[0, 2].text(0.5, 0.5, f'GPU Memory:\n{torch.cuda.max_memory_allocated()/1e9:.1f} GB', 
                       ha='center', va='center', transform=axes[0, 2].transAxes, fontsize=12)
        axes[0, 2].set_title('Resource Usage')
        
        # Inference response length over time
        if self.inference_history:
            inf_df = pd.DataFrame(self.inference_history)
            if 'step' in inf_df.columns and 'response_length' in inf_df.columns:
                avg_lengths = inf_df.groupby('step')['response_length'].mean()
                axes[1, 0].plot(avg_lengths.index, avg_lengths.values, 'r-', marker='o', linewidth=2)
                axes[1, 0].set_title('Inference Response Length')
                axes[1, 0].set_xlabel('Training Step')
                axes[1, 0].set_ylabel('Avg Response Length')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Training summary
        summary_text = f"""Training Summary:
Total Steps: {len(df)}
Final Loss: {df['loss'].iloc[-1]:.4f}
Min Loss: {df['loss'].min():.4f}
Avg Loss: {df['loss'].mean():.4f}
Inference Tests: {len(set(r['step'] for r in self.inference_history))}"""
        
        axes[1, 1].text(0.1, 0.5, summary_text, ha='left', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=10, family='monospace')
        axes[1, 1].set_title('Training Statistics')
        axes[1, 1].axis('off')
        
        # Latest inference samples
        if self.inference_history:
            latest_step = max(r['step'] for r in self.inference_history)
            latest_samples = [r for r in self.inference_history if r['step'] == latest_step]
            
            sample_text = f"Latest Inference (Step {latest_step}):\n\n"
            for i, sample in enumerate(latest_samples[:2]):  # Show first 2
                prompt_short = sample['prompt'].split('\n')[0].replace('### Instruction:', '')
                response_short = sample['response'][:80] + "..." if len(sample['response']) > 80 else sample['response']
                sample_text += f"{i+1}. Q: {prompt_short}\n   A: {response_short}\n\n"
            
            axes[1, 2].text(0.05, 0.95, sample_text, ha='left', va='top', 
                           transform=axes[1, 2].transAxes, fontsize=8, family='monospace')
            axes[1, 2].set_title('Latest Inference Samples')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('plots/training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Training progress plot saved to plots/training_progress.png")

def run_inference_test(model, tokenizer, test_instructions: list):
    """Run inference tests to validate the trained model with comprehensive logging."""
    logger.info("🧪 Running final inference tests...")
    
    # Setup inference loggers
    inference_logger = logging.getLogger('inference')
    inference_logger.info("=== FINAL MODEL INFERENCE TESTS ===")
    
    model.eval()
    results = []
    
    for i, instruction in enumerate(test_instructions):
        logger.info(f"🔍 Running final test {i+1}/{len(test_instructions)}: {instruction[:50]}...")
        
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        start_time = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        inference_time = time.time() - start_time
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        # Calculate additional metrics
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
        
        result = {
            "test_number": i + 1,
            "instruction": instruction,
            "response": response,
            "response_length_chars": len(response),
            "response_length_words": len(response.split()),
            "tokens_generated": tokens_generated,
            "inference_time_seconds": inference_time,
            "tokens_per_second": tokens_per_second,
            "timestamp": time.time(),
            "datetime": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        results.append(result)
        
        # Log to console (shortened)
        short_response = response[:100] + "..." if len(response) > 100 else response
        logger.info(f"   Response: {short_response}")
        logger.info(f"   Stats: {len(response)} chars, {tokens_generated} tokens, {inference_time:.3f}s, {tokens_per_second:.1f} tok/s")
        
        # Log full details to file
        inference_logger.info(f"FINAL_TEST_{i+1}:")
        inference_logger.info(f"  Instruction: {instruction}")
        inference_logger.info(f"  Full Response: {response}")
        inference_logger.info(f"  Length: {len(response)} chars, {len(response.split())} words")
        inference_logger.info(f"  Tokens Generated: {tokens_generated}")
        inference_logger.info(f"  Inference Time: {inference_time:.3f}s")
        inference_logger.info(f"  Speed: {tokens_per_second:.1f} tokens/second")
        inference_logger.info("---")
    
    # Calculate overall statistics
    total_chars = sum(r['response_length_chars'] for r in results)
    total_words = sum(r['response_length_words'] for r in results)
    total_tokens = sum(r['tokens_generated'] for r in results)
    total_time = sum(r['inference_time_seconds'] for r in results)
    avg_chars = total_chars / len(results)
    avg_words = total_words / len(results)
    avg_tokens = total_tokens / len(results)
    avg_time = total_time / len(results)
    avg_speed = total_tokens / total_time if total_time > 0 else 0
    
    stats = {
        "total_tests": len(results),
        "total_characters": total_chars,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "total_time_seconds": total_time,
        "average_characters_per_response": avg_chars,
        "average_words_per_response": avg_words,
        "average_tokens_per_response": avg_tokens,
        "average_time_per_response": avg_time,
        "average_tokens_per_second": avg_speed,
        "timestamp": time.time(),
        "datetime": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Log summary statistics
    logger.info("📊 Final Inference Test Summary:")
    logger.info(f"   Total Tests: {len(results)}")
    logger.info(f"   Avg Response: {avg_chars:.1f} chars, {avg_words:.1f} words, {avg_tokens:.1f} tokens")
    logger.info(f"   Avg Time: {avg_time:.3f}s, Speed: {avg_speed:.1f} tokens/s")
    
    inference_logger.info("=== FINAL INFERENCE SUMMARY ===")
    inference_logger.info(f"Total Tests: {stats['total_tests']}")
    inference_logger.info(f"Total Characters: {stats['total_characters']}")
    inference_logger.info(f"Total Words: {stats['total_words']}")
    inference_logger.info(f"Total Tokens: {stats['total_tokens']}")
    inference_logger.info(f"Total Time: {stats['total_time_seconds']:.3f}s")
    inference_logger.info(f"Average Characters per Response: {stats['average_characters_per_response']:.1f}")
    inference_logger.info(f"Average Words per Response: {stats['average_words_per_response']:.1f}")
    inference_logger.info(f"Average Tokens per Response: {stats['average_tokens_per_response']:.1f}")
    inference_logger.info(f"Average Time per Response: {stats['average_time_per_response']:.3f}s")
    inference_logger.info(f"Average Speed: {stats['average_tokens_per_second']:.1f} tokens/second")
    inference_logger.info("=== FINAL INFERENCE COMPLETED ===")
    
    # Save comprehensive results
    final_results = {
        "test_results": results,
        "summary_statistics": stats,
        "model_info": {
            "test_type": "final_inference_validation",
            "num_instructions": len(test_instructions)
        }
    }
    
    # Save to multiple files
    with open('outputs/final_inference_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save to comprehensive log file
    with open('logs/final_inference_comprehensive.jsonl', 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
        f.write(json.dumps({"summary_statistics": stats}) + "\n")
    
    logger.info("💾 Final inference results saved to:")
    logger.info("   - outputs/final_inference_results.json")
    logger.info("   - logs/final_inference_comprehensive.jsonl")
    logger.info("   - logs/inference_results.log")
    
    return results

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the output directory."""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, item)):
            try:
                step_num = int(item.split("-")[1])
                checkpoints.append((step_num, os.path.join(output_dir, item)))
            except (IndexError, ValueError):
                continue
    
    if checkpoints:
        # Return the checkpoint with the highest step number
        latest_checkpoint = max(checkpoints, key=lambda x: x[0])
        logger.info(f"🔄 Found latest checkpoint: {latest_checkpoint[1]} (step {latest_checkpoint[0]})")
        return latest_checkpoint[1]
    
    return None

def validate_checkpoint(checkpoint_path: str) -> bool:
    """Validate that a checkpoint is complete and usable."""
    if not os.path.exists(checkpoint_path):
        return False
    
    required_files = [
        "adapter_model.safetensors",  # LoRA weights
        "adapter_config.json",       # LoRA config
        "trainer_state.json",        # Training state
        "training_args.bin"          # Training arguments
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(checkpoint_path, file)):
            logger.warning(f"⚠️  Checkpoint {checkpoint_path} missing {file}")
            return False
    
    logger.info(f"✅ Checkpoint {checkpoint_path} is valid")
    return True

def main():
    """Main training function with automatic checkpoint resumption."""
    parser = argparse.ArgumentParser(description="Fine-tune Meta-Llama-3-8B on Alpaca dataset")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B", help="Model name")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--resume-from-checkpoint", type=str, help="Specific checkpoint path to resume from")
    parser.add_argument("--auto-resume", action="store_true", default=True, help="Automatically resume from latest checkpoint")
    parser.add_argument("--run-comparison", action="store_true", default=False, help="Run model comparison against base models")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Meta-Llama-3-8B Instruction Fine-tuning")
    logger.info("=" * 60)
    
    # Check for automatic resumption
    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        if validate_checkpoint(args.resume_from_checkpoint):
            resume_from_checkpoint = args.resume_from_checkpoint
            logger.info(f"🔄 Resuming from specified checkpoint: {resume_from_checkpoint}")
        else:
            logger.error(f"❌ Invalid checkpoint: {args.resume_from_checkpoint}")
            return
    elif args.auto_resume:
        latest_checkpoint = find_latest_checkpoint(args.output_dir)
        if latest_checkpoint and validate_checkpoint(latest_checkpoint):
            resume_from_checkpoint = latest_checkpoint
            logger.info(f"🔄 Auto-resuming from latest checkpoint: {resume_from_checkpoint}")
        else:
            logger.info("🆕 No valid checkpoint found, starting fresh training")
    
    # Check for Hugging Face authentication
    logger.info("🔐 Checking Hugging Face authentication...")
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    if hf_token:
        logger.info("✅ HF_TOKEN found in environment variables")
        # Set the token for this session
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
    elif os.path.exists(os.path.expanduser('~/.cache/huggingface/token')):
        logger.info("✅ Hugging Face token file found")
    else:
        logger.error("❌ No Hugging Face token detected!")
        logger.error("🔧 To fix this:")
        logger.error("   1. Get access: https://huggingface.co/meta-llama/Meta-Llama-3-8B")
        logger.error("   2. Get token: https://huggingface.co/settings/tokens")
        logger.error("   3. Set token: export HF_TOKEN='your_token'")
        logger.error("   4. Or login: huggingface-cli login")
        raise RuntimeError("Hugging Face authentication required for Llama models")
    
    # Test model access
    try:
        logger.info("🦙 Testing Llama model access...")
        from transformers import AutoTokenizer
        test_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        logger.info("✅ Llama model access confirmed!")
    except Exception as e:
        logger.error(f"❌ Cannot access {args.model_name}: {e}")
        logger.error("💡 Make sure you have requested access to the model on Hugging Face")
        raise RuntimeError(f"Model access failed: {e}")
    
    # System validation
    validator = SystemValidator()
    validator.check_gpu()
    validator.check_memory()
    validator.check_disk_space()
    validator.create_directories()
    
    # Initialize components
    trainer = ModelTrainer(args.model_name)
    monitor = TrainingMonitor()
    
    try:
        # Load model and tokenizer
        model, tokenizer = trainer.load_model_and_tokenizer()
        
        # Setup LoRA
        model = trainer.setup_lora()
        
        # Prepare dataset
        processor = DatasetProcessor(tokenizer, args.max_length)
        dataset = processor.load_alpaca_dataset()
        
        # Split dataset
        train_size = int(0.95 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, train_size + eval_size))
        
        logger.info(f"📊 Dataset split: {train_size} training, {eval_size} evaluation examples")
        
        # Process datasets
        train_dataset = processor.prepare_dataset(train_dataset)
        eval_dataset = processor.prepare_dataset(eval_dataset)
        
        # Data collator with proper tensor handling
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            return_tensors="pt",
            label_pad_token_id=-100  # Ignore padding tokens in loss calculation
        )
        
        # Debug: Check model state before training
        logger.info("🔍 Pre-training model state check:")
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        total_count = sum(1 for p in model.parameters())
        logger.info(f"   Trainable parameters: {trainable_count}/{total_count}")
        
        # Verify model is in training mode
        model.train()
        logger.info(f"   Model training mode: {model.training}")
        
        # Test a small batch to verify gradient flow
        logger.info("🧪 Testing gradient flow with sample batch...")
        try:
            sample_batch = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=data_collator)))
            sample_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}
            
            # Forward pass
            outputs = model(**sample_batch)
            loss = outputs.loss
            
            # Check if loss requires grad
            logger.info(f"   Sample loss: {loss.item():.6f}")
            logger.info(f"   Loss requires grad: {loss.requires_grad}")
            
            # Test backward pass
            loss.backward()
            
            # Check if any gradients were computed
            grad_norms = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
            
            if grad_norms:
                logger.info(f"   ✅ Gradients computed successfully! Avg grad norm: {sum(grad_norms)/len(grad_norms):.6f}")
            else:
                logger.error("   ❌ No gradients computed! This will cause training failure.")
                raise RuntimeError("Gradient computation failed - no gradients found")
            
            # Clear gradients
            model.zero_grad()
            
        except Exception as e:
            logger.error(f"   ❌ Gradient flow test failed: {e}")
            raise RuntimeError(f"Model setup validation failed: {e}")
        
        # Training arguments with enhanced checkpointing
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_steps=args.warmup_steps,
            logging_steps=50,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=5,  # Keep only 5 most recent checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb for now
            dataloader_pin_memory=False,
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            logging_dir="./logs",
            resume_from_checkpoint=resume_from_checkpoint,  # Enable automatic resumption
        )
        
        # Custom trainer with monitoring and gradient preservation
        class MonitoredTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Store LoRA parameter names for gradient verification
                self.lora_param_names = [name for name, param in self.model.named_parameters() if 'lora_' in name]
                logger.info(f"   Trainer initialized with {len(self.lora_param_names)} LoRA parameters to monitor")
            
            def training_step(self, model, inputs):
                """Override training step to ensure gradients are preserved."""
                # Ensure LoRA parameters still require gradients before each step
                for name, param in model.named_parameters():
                    if 'lora_' in name and not param.requires_grad:
                        logger.warning(f"   Re-enabling gradient for {name}")
                        param.requires_grad = True
                
                # Call parent training step
                return super().training_step(model, inputs)
            
            def log(self, logs: Dict[str, float]) -> None:
                super().log(logs)
                monitor.log_metrics(logs)
                
                # Generate plots periodically
                if self.state.global_step % 200 == 0 and self.state.global_step > 0:
                    monitor.plot_training_progress()
                
                # Run inference tests at key intervals
                if self.state.global_step % 500 == 0 and self.state.global_step > 0:
                    try:
                        monitor.run_inference_test(self.model, tokenizer, self.state.global_step)
                    except Exception as e:
                        logger.warning(f"Inference test failed at step {self.state.global_step}: {e}")
                
                # Early quality check after initial steps
                if self.state.global_step == 100:
                    logger.info("🧪 Running early quality check...")
                    try:
                        results = monitor.run_inference_test(self.model, tokenizer, self.state.global_step)
                        # Check if model is producing reasonable outputs
                        avg_length = sum(r['response_length'] for r in results) / len(results)
                        if avg_length < 10:
                            logger.warning("⚠️  Model producing very short responses - check hyperparameters!")
                        elif avg_length > 300:
                            logger.warning("⚠️  Model producing very long responses - might be unstable!")
                        else:
                            logger.info("✅ Early quality check passed - model learning normally")
                    except Exception as e:
                        logger.warning(f"Early quality check failed: {e}")
        
        # Initialize trainer
        trainer_instance = MonitoredTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # CRITICAL: Re-ensure gradient requirements after trainer initialization
        logger.info("🔧 Re-verifying gradient setup after trainer initialization...")
        model.train()
        
        # Re-enable gradients for LoRA parameters (trainer might have reset them)
        lora_param_count = 0
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                lora_param_count += 1
        
        logger.info(f"   Re-enabled gradients for {lora_param_count} LoRA parameters")
        
        # Final verification before training
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        if trainable_count == 0:
            raise RuntimeError("CRITICAL: No trainable parameters after trainer initialization!")
        
        logger.info(f"   Final check: {trainable_count} trainable parameters confirmed")
        
        # Start training
        if resume_from_checkpoint:
            logger.info(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            logger.info("🏋️  Starting fresh training...")
            
        start_time = time.time()
        
        # Training with automatic checkpoint resumption
        trainer_instance.train(resume_from_checkpoint=resume_from_checkpoint)
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")
        
        # Final plots
        monitor.plot_training_progress()
        
        # Save final model
        logger.info("💾 Saving final model...")
        trainer_instance.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Run inference tests
        test_instructions = [
            "Explain the concept of machine learning in simple terms.",
            "Write a Python function to calculate the factorial of a number.",
            "What are the benefits of renewable energy?",
            "How does photosynthesis work?",
            "Describe the water cycle.",
            "What is the difference between AI and machine learning?",
            "Write a simple recipe for chocolate chip cookies."
        ]
        
        inference_results = run_inference_test(model, tokenizer, test_instructions)
        
        # Run model comparison if requested
        if hasattr(args, 'run_comparison') and args.run_comparison:
            logger.info("🔍 Running model comparison against base models...")
            try:
                import subprocess
                comparison_cmd = [
                    "python", "compare_models.py",
                    "--finetuned-path", args.output_dir,
                    "--custom-queries", "custom_queries.json",
                    "--output-file", f"model_comparison_final_{int(time.time())}.json"
                ]
                result = subprocess.run(comparison_cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
                
                if result.returncode == 0:
                    logger.info("✅ Model comparison completed successfully")
                    logger.info("📊 Check the comparison results to see your model's improvement!")
                else:
                    logger.warning(f"⚠️  Model comparison had issues: {result.stderr}")
                    
            except Exception as e:
                logger.warning(f"⚠️  Could not run model comparison: {e}")
                logger.info("💡 You can run it manually later with: python compare_models.py")
        
        # Generate final training summary
        final_summary = {
            "training_completed": True,
            "training_time_minutes": training_time / 60,
            "total_training_steps": trainer_instance.state.global_step,
            "final_loss": trainer_instance.state.log_history[-1].get('train_loss', 'N/A'),
            "best_eval_loss": min([log.get('eval_loss', float('inf')) for log in trainer_instance.state.log_history if 'eval_loss' in log], default='N/A'),
            "model_path": args.output_dir,
            "inference_tests_count": len(inference_results),
            "resumed_from_checkpoint": resume_from_checkpoint is not None,
            "checkpoint_path": resume_from_checkpoint if resume_from_checkpoint else "N/A",
            "completion_timestamp": time.time(),
            "completion_datetime": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save training summary
        with open('logs/training_summary.json', 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        # Log final summary
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info("📊 Final Training Summary:")
        logger.info(f"   Training Time: {training_time/60:.1f} minutes")
        logger.info(f"   Total Steps: {trainer_instance.state.global_step}")
        logger.info(f"   Final Loss: {final_summary['final_loss']}")
        logger.info(f"   Best Eval Loss: {final_summary['best_eval_loss']}")
        logger.info(f"   Inference Tests: {len(inference_results)}")
        if resume_from_checkpoint:
            logger.info(f"   Resumed from: {resume_from_checkpoint}")
        
        logger.info("📁 Check the following directories:")
        logger.info("   - outputs/: Model checkpoints and final model")
        logger.info("   - plots/: Training progress visualizations")
        logger.info("   - logs/: Comprehensive training and inference logs")
        logger.info("📝 Log Files Created:")
        logger.info("   - logs/training_complete.log: Complete training log")
        logger.info("   - logs/training_metrics.log: Training metrics only")
        logger.info("   - logs/training_metrics.csv: Structured metrics for analysis")
        logger.info("   - logs/training_detailed.jsonl: Detailed JSON training logs")
        logger.info("   - logs/inference_results.log: Inference test logs")
        logger.info("   - logs/inference_comprehensive.jsonl: Structured inference data")
        logger.info("   - logs/system_monitoring.log: System resource usage")
        logger.info("   - logs/training_summary.json: Final training summary")
        
        # Log final system stats
        system_logger = logging.getLogger('system')
        system_logger.info("=== TRAINING COMPLETED ===")
        system_logger.info(f"Total Training Time: {training_time/60:.1f} minutes")
        system_logger.info(f"Final GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        system_logger.info(f"Final System Memory: {psutil.virtual_memory().percent:.1f}% used")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        
        # Log the failure details
        error_details = {
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": time.time(),
            "datetime": time.strftime('%Y-%m-%d %H:%M:%S'),
            "latest_checkpoint": find_latest_checkpoint(args.output_dir)
        }
        
        with open('logs/training_error.json', 'w') as f:
            json.dump(error_details, f, indent=2)
        
        logger.error("💾 Error details saved to logs/training_error.json")
        if error_details["latest_checkpoint"]:
            logger.info(f"🔄 You can resume training with: --resume-from-checkpoint {error_details['latest_checkpoint']}")
        
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 