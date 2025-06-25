#!/usr/bin/env python3
"""
SIMPLE BULLETPROOF Travel Model Trainer
No complex dependencies, no package conflicts, just works!
"""

import os
import json
import torch
import logging
from datetime import datetime
from pathlib import Path
import random

# Core imports - these should ALWAYS work
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

class TravelQualityCallback(TrainerCallback):
    """Smart callback that compares fine-tuned model vs base model on travel queries"""
    
    def __init__(self, base_model_path, tokenizer, test_queries, min_improvement=0.3):
        self.base_model_path = base_model_path
        self.tokenizer = tokenizer
        self.test_queries = test_queries
        self.min_improvement = min_improvement
        self.base_model = None
        self.best_travel_score = 0
        self.improvement_checks = []
        
        # Create evaluation directory
        os.makedirs('travel_evaluation', exist_ok=True)
        
    def setup_base_model(self):
        """Load base model for comparison"""
        if self.base_model is None:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
    
    def evaluate_travel_quality(self, model, step):
        """Evaluate travel response quality vs base model"""
        self.setup_base_model()
        
        fine_tuned_scores = []
        base_scores = []
        
        print(f"\n🔍 Evaluating travel quality at step {step}...")
        
        for query in self.test_queries:
            # Test fine-tuned model
            ft_response = self.generate_response(model, query)
            ft_score = self.score_travel_response(ft_response, query)
            fine_tuned_scores.append(ft_score)
            
            # Test base model
            base_response = self.generate_response(self.base_model, query)
            base_score = self.score_travel_response(base_response, query)
            base_scores.append(base_score)
            
            print(f"Query: {query[:50]}...")
            print(f"  Base score: {base_score:.2f}, Fine-tuned score: {ft_score:.2f}")
        
        avg_ft_score = sum(fine_tuned_scores) / len(fine_tuned_scores)
        avg_base_score = sum(base_scores) / len(base_scores)
        improvement = avg_ft_score - avg_base_score
        improvement_pct = (improvement / avg_base_score) * 100 if avg_base_score > 0 else 0
        
        # Save detailed results
        results = {
            'step': step,
            'fine_tuned_avg': avg_ft_score,
            'base_avg': avg_base_score,
            'improvement': improvement,
            'improvement_percentage': improvement_pct,
            'individual_scores': list(zip(self.test_queries, base_scores, fine_tuned_scores))
        }
        
        with open(f'travel_evaluation/step_{step}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 TRAVEL QUALITY RESULTS (Step {step}):")
        print(f"   Base Model Average: {avg_base_score:.2f}")
        print(f"   Fine-tuned Average: {avg_ft_score:.2f}")
        print(f"   Improvement: +{improvement:.2f} ({improvement_pct:.1f}%)")
        
        self.improvement_checks.append(improvement_pct)
        
        # Check if we've achieved 100% improvement
        if improvement_pct >= 100:
            print(f"🎉 GOAL ACHIEVED! 100%+ improvement in travel responses!")
            return True
            
        # Check if we're making good progress
        if improvement_pct >= 50:
            print(f"🔥 Excellent progress! {improvement_pct:.1f}% improvement")
        elif improvement_pct >= 20:
            print(f"✅ Good progress! {improvement_pct:.1f}% improvement")
        else:
            print(f"⏳ Early stage: {improvement_pct:.1f}% improvement")
            
        return False
    
    def generate_response(self, model, query):
        """Generate response from model"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful travel expert specializing in Indian travel. Provide comprehensive, detailed travel advice with exact costs, booking details, and practical information.<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("assistant<|end_header_id|>")[-1].strip()
    
    def score_travel_response(self, response, query):
        """Score travel response quality (0-10)"""
        score = 0
        response_lower = response.lower()
        
        # Travel-specific keywords
        travel_keywords = ['cost', 'price', 'budget', 'booking', 'hotel', 'flight', 'visa', 
                          'transport', 'itinerary', 'day', 'travel', 'trip', 'tourism']
        indian_keywords = ['indian', 'india', 'rupee', '₹', 'delhi', 'mumbai', 'bangalore', 'chennai']
        
        # Scoring criteria
        if len(response) > 200:  # Comprehensive response
            score += 2
        if len(response) > 500:  # Very detailed
            score += 1
            
        # Travel keyword density
        travel_score = sum(1 for keyword in travel_keywords if keyword in response_lower)
        score += min(travel_score * 0.5, 3)  # Max 3 points
        
        # Indian context
        indian_score = sum(1 for keyword in indian_keywords if keyword in response_lower)
        score += min(indian_score * 0.5, 2)  # Max 2 points
        
        # Specific details (costs, numbers)
        if '₹' in response or 'rupee' in response_lower or any(char.isdigit() for char in response):
            score += 1
        
        # Practical information
        practical_words = ['book', 'website', 'contact', 'phone', 'email', 'address', 'time', 'duration']
        if any(word in response_lower for word in practical_words):
            score += 1
            
        return min(score, 10)  # Max score of 10
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Called after each evaluation"""
        # Evaluate travel quality every 500 steps
        if state.global_step > 0 and state.global_step % 500 == 0:
            goal_achieved = self.evaluate_travel_quality(model, state.global_step)
            
            # Stop training if we've achieved 100% improvement
            if goal_achieved:
                print("\n🏆 TRAINING GOAL ACHIEVED! Stopping early.")
                control.should_training_stop = True

class SimpleTravelTrainer:
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Simple logging setup"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_dataset(self, file_path):
        """Load JSONL dataset"""
        self.logger.info(f"Loading dataset from {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
                
        self.logger.info(f"Loaded {len(data)} examples")
        return data
    
    def format_prompt(self, instruction, output):
        """Simple chat format for Llama 3"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful travel expert specializing in Indian travel. Provide comprehensive, detailed travel advice with exact costs, booking details, and practical information.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

    def preprocess_data(self, examples, tokenizer, max_length=2048):
        """Preprocess data for training with improved compatibility"""
        if isinstance(examples, dict):
            # Single example
            text = self.format_prompt(examples['instruction'], examples['output'])
            texts = [text]
        else:
            # Multiple examples
            texts = []
            for example in examples:
                text = self.format_prompt(example['instruction'], example['output'])
                texts.append(text)
        
        # Tokenize with improved settings
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
            add_special_tokens=True
        )
        
        # Set labels = input_ids for causal LM (copy properly)
        if isinstance(tokenized["input_ids"][0], list):
            tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        else:
            tokenized["labels"] = tokenized["input_ids"][:]
        
        return tokenized

    def train(self, 
              train_data_path="./FINAL_TRAINING_DATASET_LLAMA8B.jsonl",
              val_data_path="./FINAL_VALIDATION_DATASET_LLAMA8B.jsonl",
              model_name="meta-llama/Meta-Llama-3-8B-Instruct",
              output_dir="./simple_travel_model",
              max_length=2048,
              num_epochs=3,
              learning_rate=2e-4):
        
        self.logger.info("🚀 Starting Simple Travel Training")
        
        # 1. Load tokenizer and model
        self.logger.info("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load model with compatibility fixes
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"  # Fix for attention compatibility
            )
        except Exception as e:
            self.logger.warning(f"Loading with fallback options: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        # 2. Setup LoRA
        self.logger.info("Setting up LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # 3. Load and preprocess data
        self.logger.info("Loading datasets...")
        train_data = self.load_dataset(train_data_path)
        val_data = self.load_dataset(val_data_path)
        
        # Convert to datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Preprocess with better error handling
        def preprocess_function(examples):
            return self.preprocess_data(examples, tokenizer, max_length)
        
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=False,
            remove_columns=train_dataset.column_names,
            desc="Processing training data"
        )
        
        val_dataset = val_dataset.map(
            preprocess_function,
            batched=False,
            remove_columns=val_dataset.column_names,
            desc="Processing validation data"
        )
        
        # 4. Training arguments
        self.logger.info("Setting up training...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # Training params
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            
            # Learning rate
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            
            # Evaluation and saving (space optimized)
            eval_strategy="steps",  # Fixed for newer transformers
            eval_steps=50,
            save_strategy="steps",
            save_steps=200,  # Less frequent saves to save space
            save_total_limit=2,  # Keep only 2 checkpoints
            load_best_model_at_end=True,
            
            # Logging
            logging_steps=10,
            
            # Memory optimization
            bf16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            
            # No external reporting
            report_to="none"
        )
        
        # 5. Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=None
        )
        
        # 6. Setup travel quality evaluation
        travel_test_queries = [
            "Plan a 5-day budget trip to Goa for Indian travelers with complete cost breakdown",
            "What are visa requirements and process for Indians traveling to Thailand?",
            "Best time to visit Rajasthan with detailed itinerary and accommodation suggestions",
            "Complete guide for Kerala backwaters trip from Delhi with transport and food costs",
            "Business travel guide for Dubai from India with hotel and meeting venue recommendations"
        ]
        
        travel_callback = TravelQualityCallback(
            base_model_path=model_name,
            tokenizer=tokenizer,
            test_queries=travel_test_queries,
            min_improvement=0.3
        )
        
        # 7. Create trainer with smart callbacks
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=5),  # More patience for travel learning
                travel_callback  # Smart travel quality monitoring
            ]
        )
        
        # 8. Train with intelligent monitoring!
        self.logger.info("🔥 Starting intelligent travel training...")
        self.logger.info("Will automatically compare vs base model every 500 steps")
        self.logger.info("Training will stop early when 100% improvement is achieved!")
        
        trainer.train()
        
        # 9. Save final model
        self.logger.info("💾 Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # 10. Final comprehensive evaluation
        self.logger.info("📊 Running final travel quality evaluation...")
        final_eval = travel_callback.evaluate_travel_quality(model, "FINAL")
        
        eval_results = trainer.evaluate()
        self.logger.info(f"Final eval loss: {eval_results['eval_loss']:.4f}")
        
        # 11. Save training summary
        summary = {
            "training_completed": True,
            "final_eval_loss": eval_results['eval_loss'],
            "travel_improvements": travel_callback.improvement_checks,
            "output_directory": output_dir,
            "base_model_preserved": True,
            "travel_expertise_achieved": len(travel_callback.improvement_checks) > 0 and max(travel_callback.improvement_checks) >= 100
        }
        
        with open(f"{output_dir}/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("✅ Training completed successfully!")
        self.logger.info(f"Model saved to: {output_dir}")
        
        if summary["travel_expertise_achieved"]:
            self.logger.info("🎉 GOAL ACHIEVED: 100%+ improvement in travel responses!")
        else:
            self.logger.info(f"📈 Best improvement achieved: {max(travel_callback.improvement_checks) if travel_callback.improvement_checks else 0:.1f}%")
        
        return output_dir

def main():
    """Main training function"""
    print("🚀 Simple Travel Model Trainer")
    print("=" * 50)
    
    # Use your exact working model path
    model_name = "/workspace/hf_cache/transformers/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    
    trainer = SimpleTravelTrainer()
    
    # Start training
    output_dir = trainer.train(
        train_data_path="./FINAL_TRAINING_DATASET_LLAMA8B.jsonl",
        val_data_path="./FINAL_VALIDATION_DATASET_LLAMA8B.jsonl",
        model_name=model_name,
        output_dir="./simple_travel_model",
        max_length=2048,
        num_epochs=3,
        learning_rate=2e-4
    )
    
    print(f"✅ Training complete! Model saved to: {output_dir}")

if __name__ == "__main__":
    main() 