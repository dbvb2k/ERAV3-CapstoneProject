#!/usr/bin/env python3
"""
Travel Query Testing Script for Fine-tuned Llama Model
Tests the model on 15 diverse travel-related queries to evaluate performance
"""

import os
import sys
import json
import time
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TravelQueryTester:
    def __init__(self, model_path="outputs/final_model", max_length=512, temperature=0.7):
        """
        Initialize the travel query tester.
        
        Args:
            model_path: Path to the fine-tuned model
            max_length: Maximum length for generated responses
            temperature: Temperature for text generation
        """
        self.model_path = model_path
        self.max_length = max_length
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        
        # 15 diverse travel queries for testing
        self.travel_queries = [
            # Planning and Destinations
            {
                "id": 1,
                "category": "Destination Planning",
                "query": "I'm planning a 7-day trip to Japan in spring. What are the must-visit places and experiences I shouldn't miss?",
                "expected_topics": ["cherry blossoms", "Tokyo", "Kyoto", "temples", "food", "culture"]
            },
            {
                "id": 2,
                "category": "Budget Travel",
                "query": "What are some budget-friendly European destinations for backpackers under $50 per day?",
                "expected_topics": ["Eastern Europe", "hostels", "budget airlines", "cheap food", "free activities"]
            },
            {
                "id": 3,
                "category": "Family Travel",
                "query": "Where are the best family-friendly destinations in the US for traveling with young children?",
                "expected_topics": ["theme parks", "national parks", "family resorts", "kid-friendly activities"]
            },
            
            # Transportation and Logistics
            {
                "id": 4,
                "category": "Transportation",
                "query": "What's the most efficient way to travel between major cities in Europe by train?",
                "expected_topics": ["Eurail pass", "high-speed trains", "booking", "routes", "schedules"]
            },
            {
                "id": 5,
                "category": "Flight Booking",
                "query": "When is the best time to book international flights to get the lowest prices?",
                "expected_topics": ["advance booking", "price comparison", "seasonal trends", "flexible dates"]
            },
            {
                "id": 6,
                "category": "Local Transportation",
                "query": "How do I navigate public transportation in major Asian cities like Tokyo, Seoul, and Singapore?",
                "expected_topics": ["metro systems", "IC cards", "apps", "etiquette", "maps"]
            },
            
            # Accommodation and Food
            {
                "id": 7,
                "category": "Accommodation",
                "query": "What are the pros and cons of staying in hostels vs hotels vs Airbnb for solo travelers?",
                "expected_topics": ["safety", "cost", "social aspects", "privacy", "amenities"]
            },
            {
                "id": 8,
                "category": "Food and Culture",
                "query": "How can I experience authentic local cuisine while traveling without getting sick?",
                "expected_topics": ["street food", "food safety", "local restaurants", "cultural etiquette"]
            },
            {
                "id": 9,
                "category": "Dietary Restrictions",
                "query": "What are the best strategies for vegetarian travelers in countries with meat-heavy cuisines?",
                "expected_topics": ["vegetarian options", "language phrases", "apps", "local dishes"]
            },
            
            # Safety and Practical Tips
            {
                "id": 10,
                "category": "Travel Safety",
                "query": "What safety precautions should solo female travelers take when visiting developing countries?",
                "expected_topics": ["safety tips", "clothing", "accommodation", "communication", "emergency contacts"]
            },
            {
                "id": 11,
                "category": "Travel Documents",
                "query": "What documents and preparations do I need for international travel, especially for first-time travelers?",
                "expected_topics": ["passport", "visa", "insurance", "vaccinations", "copies", "emergency info"]
            },
            {
                "id": 12,
                "category": "Money and Banking",
                "query": "How should I handle money and payments while traveling internationally to avoid fees?",
                "expected_topics": ["travel cards", "ATMs", "currency exchange", "bank fees", "cash vs cards"]
            },
            
            # Seasonal and Activity-Based
            {
                "id": 13,
                "category": "Seasonal Travel",
                "query": "What are the best winter destinations for travelers who want to escape cold weather?",
                "expected_topics": ["tropical destinations", "southern hemisphere", "weather patterns", "activities"]
            },
            {
                "id": 14,
                "category": "Adventure Travel",
                "query": "What gear and preparation do I need for trekking in the Himalayas or similar high-altitude destinations?",
                "expected_topics": ["altitude sickness", "gear list", "training", "permits", "guides", "safety"]
            },
            {
                "id": 15,
                "category": "Cultural Experience",
                "query": "How can I respectfully engage with local cultures and communities while traveling to avoid being just a tourist?",
                "expected_topics": ["cultural sensitivity", "local interactions", "volunteering", "learning", "respect"]
            }
        ]
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            logger.info("✅ Model and tokenizer loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def generate_response(self, query, max_new_tokens=256):
        """Generate a response for a given query."""
        try:
            # Format the prompt (adjust based on your training format)
            prompt = f"### Human: {query}\n\n### Assistant: "
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - max_new_tokens
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            if "### Assistant: " in full_response:
                response = full_response.split("### Assistant: ", 1)[1].strip()
            else:
                response = full_response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Could not generate response - {str(e)}"
    
    def evaluate_response(self, query_data, response):
        """Simple evaluation of response quality."""
        response_lower = response.lower()
        expected_topics = query_data.get("expected_topics", [])
        
        # Count how many expected topics are mentioned
        topics_covered = sum(1 for topic in expected_topics if topic.lower() in response_lower)
        coverage_score = topics_covered / len(expected_topics) if expected_topics else 0
        
        # Basic quality metrics
        word_count = len(response.split())
        has_practical_info = any(word in response_lower for word in 
                               ['recommend', 'suggest', 'should', 'can', 'try', 'consider', 'important'])
        
        return {
            "topics_covered": topics_covered,
            "total_expected_topics": len(expected_topics),
            "coverage_score": coverage_score,
            "word_count": word_count,
            "has_practical_info": has_practical_info,
            "response_length": "short" if word_count < 50 else "medium" if word_count < 150 else "long"
        }
    
    def run_tests(self):
        """Run all travel query tests."""
        if not self.load_model():
            return False
        
        logger.info("🚀 Starting travel query testing...")
        results = []
        
        start_time = time.time()
        
        for i, query_data in enumerate(self.travel_queries, 1):
            logger.info(f"\n📝 Testing Query {i}/15: {query_data['category']}")
            logger.info(f"Query: {query_data['query'][:100]}...")
            
            # Generate response
            query_start = time.time()
            response = self.generate_response(query_data['query'])
            query_time = time.time() - query_start
            
            # Evaluate response
            evaluation = self.evaluate_response(query_data, response)
            
            # Store results
            result = {
                "query_id": query_data['id'],
                "category": query_data['category'],
                "query": query_data['query'],
                "response": response,
                "evaluation": evaluation,
                "generation_time": query_time
            }
            results.append(result)
            
            # Print summary
            logger.info(f"✅ Response generated in {query_time:.2f}s")
            logger.info(f"📊 Coverage: {evaluation['topics_covered']}/{evaluation['total_expected_topics']} topics")
            logger.info(f"📝 Length: {evaluation['word_count']} words ({evaluation['response_length']})")
            
            # Print first 200 chars of response
            preview = response[:200] + "..." if len(response) > 200 else response
            logger.info(f"💬 Response preview: {preview}")
        
        total_time = time.time() - start_time
        
        # Save results
        self.save_results(results, total_time)
        
        # Print summary
        self.print_summary(results, total_time)
        
        return True
    
    def save_results(self, results, total_time):
        """Save test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        os.makedirs("travel_test_results", exist_ok=True)
        
        # Save detailed JSON results
        json_file = f"travel_test_results/travel_queries_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_info": {
                    "timestamp": datetime.now().isoformat(),
                    "model_path": self.model_path,
                    "total_queries": len(results),
                    "total_time": total_time,
                    "average_time_per_query": total_time / len(results)
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        # Save human-readable report
        report_file = f"travel_test_results/travel_queries_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Travel Query Test Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** {self.model_path}\n")
            f.write(f"**Total Queries:** {len(results)}\n")
            f.write(f"**Total Time:** {total_time:.2f} seconds\n\n")
            
            for result in results:
                f.write(f"## Query {result['query_id']}: {result['category']}\n\n")
                f.write(f"**Question:** {result['query']}\n\n")
                f.write(f"**Response:**\n{result['response']}\n\n")
                f.write(f"**Evaluation:**\n")
                f.write(f"- Topics covered: {result['evaluation']['topics_covered']}/{result['evaluation']['total_expected_topics']}\n")
                f.write(f"- Coverage score: {result['evaluation']['coverage_score']:.2f}\n")
                f.write(f"- Word count: {result['evaluation']['word_count']}\n")
                f.write(f"- Generation time: {result['generation_time']:.2f}s\n\n")
                f.write("---\n\n")
        
        logger.info(f"📁 Results saved to:")
        logger.info(f"   - {json_file}")
        logger.info(f"   - {report_file}")
    
    def print_summary(self, results, total_time):
        """Print test summary statistics."""
        logger.info("\n" + "="*60)
        logger.info("🎯 TRAVEL QUERY TEST SUMMARY")
        logger.info("="*60)
        
        # Overall stats
        total_queries = len(results)
        avg_time = total_time / total_queries
        total_words = sum(r['evaluation']['word_count'] for r in results)
        avg_words = total_words / total_queries
        
        logger.info(f"📊 Overall Statistics:")
        logger.info(f"   Total Queries: {total_queries}")
        logger.info(f"   Total Time: {total_time:.2f} seconds")
        logger.info(f"   Average Time per Query: {avg_time:.2f} seconds")
        logger.info(f"   Total Words Generated: {total_words}")
        logger.info(f"   Average Words per Response: {avg_words:.1f}")
        
        # Coverage statistics
        coverage_scores = [r['evaluation']['coverage_score'] for r in results]
        avg_coverage = sum(coverage_scores) / len(coverage_scores)
        
        logger.info(f"\n📈 Content Quality:")
        logger.info(f"   Average Topic Coverage: {avg_coverage:.2f} ({avg_coverage*100:.1f}%)")
        logger.info(f"   Responses with Practical Info: {sum(1 for r in results if r['evaluation']['has_practical_info'])}/{total_queries}")
        
        # Category breakdown
        categories = {}
        for result in results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result['evaluation']['coverage_score'])
        
        logger.info(f"\n📋 Performance by Category:")
        for category, scores in categories.items():
            avg_score = sum(scores) / len(scores)
            logger.info(f"   {category}: {avg_score:.2f} average coverage")
        
        # Response length distribution
        length_dist = {}
        for result in results:
            length = result['evaluation']['response_length']
            length_dist[length] = length_dist.get(length, 0) + 1
        
        logger.info(f"\n📏 Response Length Distribution:")
        for length, count in length_dist.items():
            logger.info(f"   {length.title()}: {count} responses")
        
        logger.info("="*60)

def main():
    """Main function to run travel query tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test fine-tuned Llama model on travel queries")
    parser.add_argument("--model-path", default="outputs/final_model", 
                       help="Path to the fine-tuned model")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum input length")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                       help="Maximum new tokens to generate")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"❌ Model not found at {args.model_path}")
        logger.info("💡 Make sure you've completed training first!")
        logger.info("💡 Or specify the correct path with --model-path")
        return False
    
    # Create tester and run tests
    tester = TravelQueryTester(
        model_path=args.model_path,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    success = tester.run_tests()
    
    if success:
        logger.info("🎉 Travel query testing completed successfully!")
        logger.info("📁 Check the travel_test_results/ folder for detailed results")
        return True
    else:
        logger.error("❌ Travel query testing failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 