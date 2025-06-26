#!/usr/bin/env python3
"""
Comprehensive Travel Model Testing Script
Tests the fine-tuned Llama-3-8B travel model against various queries
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

def setup_logging():
    """Setup logging for testing."""
    os.makedirs('travel_logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('travel_logs/model_testing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class TravelModelTester:
    """Comprehensive travel model testing."""
    
    def __init__(self, model_dir: str, base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_dir = model_dir
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        
        # Comprehensive test queries
        self.test_queries = [
            "Plan a 10-day budget trip to Japan for Indian vegetarian travelers with detailed costs",
            "Complete visa requirements for Indians traveling to Schengen countries",
            "Cultural etiquette for Indian business travelers in Germany",
            "Vegetarian food guide for Indian travelers in Europe",
            "Safety guide for solo female Indian travelers to Thailand",
            "Family-friendly Europe itinerary for Indian family with kids",
            "Business travel guide to Singapore for Indian professionals",
            "What to do if you lose passport abroad? Guide for Indians",
            "Luxury honeymoon destinations from India under ₹5 lakhs",
            "Best time to visit European countries for Indian travelers"
        ]
    
    def load_model(self):
        """Load the fine-tuned travel model."""
        console.print(f"[blue]Loading travel model from {self.model_dir}...[/blue]")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                load_in_4bit=True
            )
            
            # Load PEFT model
            self.model = PeftModel.from_pretrained(base_model, self.model_dir)
            self.model.eval()
            
            console.print("[green]✅ Travel model loaded successfully![/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {e}[/red]")
            logger.error(f"Model loading failed: {e}")
            return False
    
    def generate_response(self, query: str, max_tokens: int = 1024) -> str:
        """Generate response for a travel query."""
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert travel advisor specializing in Indian travelers. Provide comprehensive, detailed, and practical travel advice with specific costs, booking information, and cultural insights.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"ERROR: {str(e)}"
    
    def evaluate_response(self, query: str, response: str) -> Dict[str, Any]:
        """Evaluate response quality."""
        travel_keywords = ['cost', 'price', 'budget', 'visa', 'flight', 'hotel', 'travel']
        indian_keywords = ['indian', 'rupee', '₹', 'vegetarian', 'embassy']
        
        travel_count = sum(1 for kw in travel_keywords if kw.lower() in response.lower())
        indian_count = sum(1 for kw in indian_keywords if kw.lower() in response.lower())
        
        score = 0.0
        if len(response) > 200: score += 2.0
        score += min(travel_count * 0.3, 3.0)
        score += min(indian_count * 0.5, 2.0)
        if response.endswith(('.', '!', '?')): score += 1.0
        
        return {
            'query': query,
            'response': response,
            'length': len(response),
            'travel_keywords': travel_count,
            'indian_keywords': indian_count,
            'quality_score': min(score, 10.0)
        }
    
    def run_test(self) -> List[Dict[str, Any]]:
        """Run comprehensive testing."""
        console.print("[blue]🧪 Running travel model tests...[/blue]")
        
        results = []
        for i, query in enumerate(self.test_queries, 1):
            console.print(f"[cyan]Testing query {i}/{len(self.test_queries)}...[/cyan]")
            
            response = self.generate_response(query)
            metrics = self.evaluate_response(query, response)
            results.append(metrics)
            
            console.print(f"Quality Score: {metrics['quality_score']:.2f}")
        
        return results
    
    def display_results(self, results: List[Dict[str, Any]]):
        """Display test results."""
        avg_score = sum(r['quality_score'] for r in results) / len(results)
        avg_length = sum(r['length'] for r in results) / len(results)
        
        console.print(Panel.fit("🌍 TRAVEL MODEL TEST RESULTS", style="bold blue"))
        
        table = Table(title="📊 Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Queries", str(len(results)))
        table.add_row("Average Quality Score", f"{avg_score:.2f}/10.0")
        table.add_row("Average Response Length", f"{avg_length:.0f} chars")
        
        console.print(table)
        
        if avg_score >= 7.0:
            console.print("[bold green]🎉 Excellent! Model ready for travel advisory![/bold green]")
        elif avg_score >= 5.0:
            console.print("[bold yellow]👍 Good performance with room for improvement.[/bold yellow]")
        else:
            console.print("[bold red]⚠️ Model needs more training.[/bold red]")

def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test fine-tuned travel model")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    
    args = parser.parse_args()
    
    tester = TravelModelTester(args.model_dir, args.base_model)
    
    if not tester.load_model():
        return
    
    results = tester.run_test()
    tester.display_results(results)
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    console.print("[green]✅ Results saved to test_results.json[/green]")

if __name__ == "__main__":
    main() 