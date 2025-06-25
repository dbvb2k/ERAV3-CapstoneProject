#!/usr/bin/env python3
"""
Advanced Travel Chatbot
======================

Interactive chatbot with conversation memory and specialized travel functions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import datetime

class TravelChatbot:
    def __init__(self, model_path="./model"):
        """Initialize the travel chatbot"""
        print("🚀 Loading Travel Assistant...")
        
        # Load model components
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Meta-Llama-3-8B-Instruct',
            torch_dtype=torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        # Conversation state
        self.conversation_history = []
        self.user_preferences = {}
        self.session_start = datetime.datetime.now()
        
        print("✅ Travel Assistant ready! Type 'help' for commands.")
    
    def detect_intent(self, user_input):
        """Detect the type of travel query"""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['budget', 'cheap', 'affordable', 'money']):
            return 'budget_planning'
        elif any(word in user_input_lower for word in ['safety', 'safe', 'dangerous', 'security']):
            return 'safety_advice'
        elif any(word in user_input_lower for word in ['itinerary', 'plan', 'schedule', 'days']):
            return 'itinerary_planning'
        elif any(word in user_input_lower for word in ['food', 'restaurant', 'cuisine', 'eat']):
            return 'food_advice'
        elif any(word in user_input_lower for word in ['pack', 'luggage', 'bring', 'essentials']):
            return 'packing_advice'
        elif any(word in user_input_lower for word in ['visa', 'passport', 'documents', 'requirements']):
            return 'travel_documents'
        else:
            return 'general_travel'
    
    def format_prompt(self, user_input, intent):
        """Format prompt based on detected intent"""
        
        # Add context from conversation
        context = ""
        if self.conversation_history:
            recent_context = "\n".join(self.conversation_history[-4:])  # Last 2 exchanges
            context = f"Previous conversation:\n{recent_context}\n\n"
        
        # Add user preferences if available
        prefs = ""
        if self.user_preferences:
            prefs = f"User preferences: {json.dumps(self.user_preferences)}\n\n"
        
        # Intent-specific formatting
        if intent == 'budget_planning':
            prompt = f"{context}{prefs}User is asking about budget travel: {user_input}\nTravel Assistant (focus on money-saving tips):"
        elif intent == 'safety_advice':
            prompt = f"{context}{prefs}User is asking about travel safety: {user_input}\nTravel Assistant (focus on safety and security):"
        elif intent == 'itinerary_planning':
            prompt = f"{context}{prefs}User wants help with itinerary planning: {user_input}\nTravel Assistant (provide structured daily plans):"
        else:
            prompt = f"{context}{prefs}User: {user_input}\nTravel Assistant:"
        
        return prompt
    
    def generate_response(self, user_input, max_tokens=250, temperature=0.7):
        """Generate contextual response"""
        
        # Detect intent
        intent = self.detect_intent(user_input)
        
        # Format prompt with context
        prompt = self.format_prompt(user_input, intent)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        # Extract user preferences from the conversation
        self.extract_preferences(user_input)
        
        # Update conversation history
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Assistant: {response}")
        
        # Keep history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response, intent
    
    def extract_preferences(self, user_input):
        """Extract user preferences from conversation"""
        user_input_lower = user_input.lower()
        
        # Extract budget information
        if 'budget' in user_input_lower:
            import re
            budget_match = re.search(r'\$?(\d+)', user_input)
            if budget_match:
                self.user_preferences['budget'] = budget_match.group(1)
        
        # Extract travel style
        if any(word in user_input_lower for word in ['backpack', 'budget', 'hostel']):
            self.user_preferences['style'] = 'budget'
        elif any(word in user_input_lower for word in ['luxury', 'resort', 'premium']):
            self.user_preferences['style'] = 'luxury'
        
        # Extract travel type
        if 'solo' in user_input_lower:
            self.user_preferences['travel_type'] = 'solo'
        elif any(word in user_input_lower for word in ['family', 'kids', 'children']):
            self.user_preferences['travel_type'] = 'family'
        elif any(word in user_input_lower for word in ['couple', 'romantic']):
            self.user_preferences['travel_type'] = 'couple'
    
    def get_specialized_advice(self, query_type):
        """Provide specialized advice based on query type"""
        
        prompts = {
            'budget': "What are the best strategies for budget travel? Include accommodation, food, and transportation tips.",
            'safety': "What are essential safety tips for international travelers? Cover personal security, health, and emergency preparedness.",
            'solo': "What advice do you have for solo travelers? Include safety, social aspects, and planning tips.",
            'family': "What are the best practices for family travel with children? Include planning, activities, and logistics.",
            'food': "How can travelers experience local cuisine safely? Include recommendations for trying new foods and avoiding illness.",
            'documents': "What travel documents and preparations are essential for international travel? Include visas, insurance, and health requirements."
        }
        
        if query_type in prompts:
            response, _ = self.generate_response(prompts[query_type], temperature=0.6)
            return response
        else:
            return "I can help with budget, safety, solo travel, family travel, food, or document advice. What would you like to know?"
    
    def show_preferences(self):
        """Display current user preferences"""
        if self.user_preferences:
            print("\n📋 Your Travel Preferences:")
            for key, value in self.user_preferences.items():
                print(f"   {key.title()}: {value}")
        else:
            print("\n📋 No preferences detected yet. Keep chatting and I'll learn your style!")
    
    def show_help(self):
        """Display help information"""
        print("""
🧳 Travel Assistant Commands:
        
💬 Chat Commands:
   • Just type your travel question naturally
   • Ask about destinations, budgets, safety, planning, etc.
   
🎯 Quick Advice:
   • /budget    - Budget travel strategies
   • /safety    - Travel safety tips  
   • /solo      - Solo travel advice
   • /family    - Family travel tips
   • /food      - Food and cuisine advice
   • /documents - Travel documents guide
   
⚙️ Settings:
   • /prefs     - Show your travel preferences
   • /clear     - Clear conversation history
   • /help      - Show this help
   • /quit      - Exit the assistant
   
💡 Examples:
   • "I'm planning a 2-week trip to Japan with a $3000 budget"
   • "What are the safest destinations for solo female travelers?"
   • "How can I experience authentic food in Thailand?"
   • "Family-friendly activities in Barcelona?"
        """)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.user_preferences = {}
        print("🔄 Conversation history and preferences cleared!")
    
    def run(self):
        """Main chat loop"""
        print("""
🧳 Welcome to your AI Travel Assistant!
I'm specialized in travel advice and can help with:
• Destination planning • Budget travel • Safety tips
• Itineraries • Local culture • Food recommendations

Type 'help' for commands or just ask me anything about travel!
        """)
        
        while True:
            try:
                user_input = input("\n🗣️  You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    session_duration = datetime.datetime.now() - self.session_start
                    print(f"\n🧳 Thanks for chatting! Session duration: {session_duration}")
                    print("Happy travels! 🌎✈️")
                    break
                
                elif user_input.lower() in ['help', '/help']:
                    self.show_help()
                    continue
                
                elif user_input.lower() in ['clear', '/clear']:
                    self.clear_history()
                    continue
                
                elif user_input.lower() in ['prefs', '/prefs']:
                    self.show_preferences()
                    continue
                
                elif user_input.startswith('/'):
                    command = user_input[1:].lower()
                    response = self.get_specialized_advice(command)
                    print(f"\n🤖 Travel Assistant: {response}")
                    continue
                
                # Generate regular response
                print("\n🤔 Thinking...")
                response, intent = self.generate_response(user_input)
                
                # Add intent indicator
                intent_emojis = {
                    'budget_planning': '💰',
                    'safety_advice': '🛡️',
                    'itinerary_planning': '📅',
                    'food_advice': '🍴',
                    'packing_advice': '🎒',
                    'travel_documents': '📋',
                    'general_travel': '🧳'
                }
                
                emoji = intent_emojis.get(intent, '🧳')
                print(f"\n{emoji} Travel Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\n🧳 Thanks for using Travel Assistant! Safe travels! 🌎")
                break
            except Exception as e:
                print(f"\n❌ Sorry, I encountered an error: {e}")
                print("Let's try again!")

def main():
    """Main function"""
    try:
        chatbot = TravelChatbot()
        chatbot.run()
    except Exception as e:
        print(f"❌ Failed to start Travel Assistant: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Downloaded the model to ./model/ directory")
        print("   2. Installed dependencies: pip install transformers torch peft")
        print("   3. Sufficient GPU memory available")

if __name__ == "__main__":
    main() 