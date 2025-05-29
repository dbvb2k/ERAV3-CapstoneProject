#!/usr/bin/env python3
"""
Llama 3 8B Travel Assistant Fine-tuning with Unsloth Acceleration
"""

import os
import json
import torch
import argparse
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)

from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class InferenceCallback(TrainerCallback):
    """Callback to perform inference during training"""
    def __init__(self, tokenizer, inference_steps):
        self.tokenizer = tokenizer
        self.inference_steps = inference_steps
        self.test_prompts = [
            "What are the best places to visit in Paris?",
            "How do I plan a budget trip to Japan?",
            "What's the best time to visit Bali?"
        ]
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.inference_steps == 0:
            print(f"\n=== Inference at step {state.global_step} ===")
            for prompt in self.test_prompts:
                print(f"\nPrompt: {prompt}")
                response = self.generate_response(model, prompt)
                print(f"Response: {response}")
                print("-" * 80)
    
    def generate_response(self, model, prompt, max_length=2048):
        """Generate response for a given prompt"""
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful travel assistant. You help users with travel planning, booking accommodations, finding restaurants, transportation, and providing travel information. Be friendly, informative, and helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                min_length=100,
                max_new_tokens=1024
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            assistant_start = full_response.rfind("<|start_header_id|>assistant<|end_header_id|>")
            if assistant_start != -1:
                response = full_response[assistant_start:].split("<|eot_id|>")[0]
                response = response.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
            else:
                response = full_response
        except:
            response = full_response
            
        return response

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Llama 3 Travel Assistant Model with Unsloth')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs (default: 1)')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint folder to resume training from')
    parser.add_argument('--inference_steps', type=int, default=100, help='Perform inference every N steps (default: 100)')
    return parser.parse_args()

def load_and_prepare_data(train_path, val_path):
    """Load and prepare the training and validation datasets"""
    print("Loading datasets...")
    
    # Load training data
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Load validation data
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    def format_conversations(data):
        """Convert conversations to the format expected by the Instruct model"""
        formatted_data = []
        for item in data:
            conversations = item['conversations']
            
            # Format for Llama 3 Instruct model with proper system message
            text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful travel assistant. You help users with travel planning, booking accommodations, finding restaurants, transportation, and providing travel information. Be friendly, informative, and helpful.<|eot_id|>"
            
            for conv in conversations:
                if conv['from'] == 'user':
                    text += f"<|start_header_id|>user<|end_header_id|>\n\n{conv['value']}<|eot_id|>"
                elif conv['from'] == 'assistant':
                    text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{conv['value']}<|eot_id|>"
            
            formatted_data.append({"text": text})
        
        return formatted_data
    
    # Format the data
    train_formatted = format_conversations(train_data)
    val_formatted = format_conversations(val_data)
    
    # Create datasets
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint folder within the main checkpoint directory"""
    try:
        # First check if the directory exists
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        print(f"\nChecking checkpoint directory: {checkpoint_dir}")
        print("Directory contents:")
        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(item_path):
                print(f"Directory: {item}")
                print("  Contents:")
                try:
                    for subitem in os.listdir(item_path):
                        print(f"    {subitem}")
                except Exception as e:
                    print(f"    Error listing contents: {str(e)}")
            else:
                print(f"File: {item}")
        
        # List all items in the directory
        items = os.listdir(checkpoint_dir)
        
        # Look for checkpoint folders
        checkpoint_folders = [d for d in items if d.startswith('checkpoint-')]
        
        if not checkpoint_folders:
            # If no checkpoint folders found, check if this is a model directory itself
            if os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")) or \
               os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin")):
                print(f"Using model directory directly: {checkpoint_dir}")
                return checkpoint_dir
            raise ValueError(f"No checkpoint folders or model files found in {checkpoint_dir}")
        
        # Extract step numbers and find the latest
        latest_checkpoint = max(checkpoint_folders, key=lambda x: int(x.split('-')[1]))
        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        print(f"\nExamining latest checkpoint: {latest_checkpoint}")
        print("Checkpoint contents:")
        for item in os.listdir(latest_checkpoint_path):
            print(f"  {item}")
        
        # Check for various possible model file names
        possible_model_files = [
            "pytorch_model.bin",
            "adapter_model.bin",
            "model.safetensors",
            "adapter_model.safetensors",
            "training_args.bin",
            "optimizer.pt",
            "scheduler.pt"
        ]
        
        found_files = []
        for file in possible_model_files:
            if os.path.exists(os.path.join(latest_checkpoint_path, file)):
                found_files.append(file)
        
        if not found_files:
            raise ValueError(f"Checkpoint {latest_checkpoint} does not contain any model files")
        
        print(f"\nFound model files: {', '.join(found_files)}")
        return latest_checkpoint_path
        
    except Exception as e:
        print(f"Error finding checkpoint: {str(e)}")
        raise

def setup_model_and_tokenizer(checkpoint_path=None):
    """Setup the model and tokenizer using Unsloth"""
    print("Loading Llama 3 8B Instruct model with Unsloth...")
    
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will be very slow on CPU!")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    model_name = "NousResearch/Meta-Llama-3-8B-Instruct"
    
    # Get HuggingFace token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is not set. Please set your HuggingFace token using:\n"
            "1. Create a .env file with: HF_TOKEN=your_token_here\n"
            "2. Or set it in your environment: export HF_TOKEN=your_token_here\n"
            "You can get your token from: https://huggingface.co/settings/tokens"
        )
    
    try:
        # Initialize tokenizer first
        print("Initializing tokenizer...")
        
        # Handle checkpoint path
        if checkpoint_path:
            # Find the latest checkpoint folder
            latest_checkpoint = find_latest_checkpoint(checkpoint_path)
            
            print(f"Loading tokenizer from checkpoint: {latest_checkpoint}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    latest_checkpoint,
                    trust_remote_code=True,
                    padding_side="right"
                )
            except Exception as e:
                print(f"Warning: Could not load tokenizer from checkpoint, falling back to base model tokenizer: {str(e)}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    token=hf_token,
                    trust_remote_code=True,
                    padding_side="right"
                )
        else:
            print("Loading tokenizer from base model...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True,
                padding_side="right"
            )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize Unsloth model
        print("Initializing model...")
        if checkpoint_path:
            print(f"Loading model from checkpoint: {latest_checkpoint}")
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=latest_checkpoint,
                    max_seq_length=2048,
                    dtype=torch.float16,
                    load_in_4bit=True
                )
            except Exception as e:
                print(f"Error loading model from checkpoint: {str(e)}")
                print("Attempting to load base model and then load checkpoint weights...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=2048,
                    dtype=torch.float16,
                    load_in_4bit=True,
                    token=hf_token
                )
                # Try to load checkpoint weights
                try:
                    checkpoint_files = os.listdir(latest_checkpoint)
                    model_file = next((f for f in checkpoint_files if f.endswith('.bin') or f.endswith('.safetensors')), None)
                    if model_file:
                        print(f"Loading weights from: {model_file}")
                        model.load_state_dict(torch.load(os.path.join(latest_checkpoint, model_file)))
                    else:
                        print("No model weight files found in checkpoint")
                except Exception as e:
                    print(f"Error loading checkpoint weights: {str(e)}")
        else:
            print("Loading base model...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                dtype=torch.float16,
                load_in_4bit=True,
                token=hf_token
            )
        
        if not checkpoint_path:
            # Setup LoRA with Unsloth optimizations only for new training
            print("Setting up LoRA...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing=True,
                random_state=42
            )
        else:
            print(f"Resuming training from checkpoint: {latest_checkpoint}")
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        raise

def tokenize_function(examples, tokenizer):
    """Tokenize the examples"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=2048,
        return_tensors="pt"
    )

def train_model(model, tokenizer, train_dataset, val_dataset, num_epochs=1, inference_steps=100, checkpoint_path=None):
    """Train the model using Unsloth optimizations"""
    print("Starting training with Unsloth...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("models", f"llama3_travel_assistant_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=1
    )
    
    val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        num_proc=1
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Calculate total steps based on dataset size and batch size
    batch_size = 2  # per_device_train_batch_size
    gradient_accumulation_steps = 16
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = batch_size * gradient_accumulation_steps * num_gpus
    total_steps = (len(train_dataset) * num_epochs) // effective_batch_size
    
    # Get the starting step from checkpoint if provided
    starting_step = 0
    if checkpoint_path:
        latest_checkpoint = find_latest_checkpoint(checkpoint_path)
        checkpoint_name = os.path.basename(latest_checkpoint)
        if checkpoint_name.startswith('checkpoint-'):
            starting_step = int(checkpoint_name.split('-')[1])
            print(f"Resuming training from step {starting_step}")
            
            # Adjust total steps to account for remaining steps
            remaining_steps = total_steps - starting_step
            if remaining_steps <= 0:
                print("Warning: All steps for the specified epochs have already been completed in the checkpoint.")
                print("No additional training will be performed.")
                return None, latest_checkpoint
            
            print(f"Remaining steps to train: {remaining_steps}")
            total_steps = remaining_steps
    
    # Training arguments optimized for Unsloth
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=total_steps // 10,
        eval_steps=total_steps // 10,
        save_total_limit=2,
        warmup_steps=total_steps // 10,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        # Add resume training parameters
        resume_from_checkpoint=latest_checkpoint if checkpoint_path else None,
        ignore_data_skip=True,  # Important for resuming training
        max_steps=total_steps
    )
    
    # Create inference callback
    inference_callback = InferenceCallback(tokenizer, inference_steps)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[inference_callback]
    )
    
    try:
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save final model
        print("Saving model and tokenizer...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return trainer, output_dir
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def main():
    """Main training pipeline"""
    print("=== Llama 3 8B Travel Assistant (Unsloth Accelerated) ===")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # File paths
    train_path = "../data/final/train_llama_format.json"
    val_path = "../data/final/validation_llama_format.json"
    
    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation file not found: {val_path}")
    
    try:
        # Step 1: Load and prepare data
        train_dataset, val_dataset = load_and_prepare_data(train_path, val_path)
        
        # Step 2: Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(args.checkpoint)
        
        # Step 3: Train the model
        trainer, output_dir = train_model(
            model, 
            tokenizer, 
            train_dataset, 
            val_dataset, 
            args.epochs,
            args.inference_steps,
            args.checkpoint
        )
        
        print("\n=== Training Complete! ===")
        print(f"Model saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main() 