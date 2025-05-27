#!/usr/bin/env python3
"""
Llama 3 8B Travel Assistant Fine-tuning with Unsloth Acceleration
"""

import os
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)

from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

def setup_model_and_tokenizer():
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
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,  # Use 4-bit quantization
            token=hf_token
        )
        
        # Setup LoRA with Unsloth optimizations
        print("Setting up LoRA...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.0,  # Changed to 0.0 to avoid performance hit
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42
        )
        
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

def train_model(model, tokenizer, train_dataset, val_dataset):
    """Train the model using Unsloth optimizations"""
    print("Starting training with Unsloth...")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./llama3_travel_assistant_{timestamp}"
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
    
    # Training arguments optimized for Unsloth
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        max_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        save_total_limit=2,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="adamw_torch",
        remove_unused_columns=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
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
        model, tokenizer = setup_model_and_tokenizer()
        
        # Step 3: Train the model
        trainer, output_dir = train_model(model, tokenizer, train_dataset, val_dataset)
        
        print("\n=== Training Complete! ===")
        print(f"Model saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main() 