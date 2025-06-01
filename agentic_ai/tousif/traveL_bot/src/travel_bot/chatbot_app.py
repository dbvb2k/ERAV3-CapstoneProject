from typing import Dict, Any, Optional
import logging
import re
import time
from .models import AIModel, ModelFactory
from .processors import ProcessorFactory, ProcessedInput, InputType

def setup_logging():
    """Configure logging for the chatbot"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('chatbot.log')
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

class TravelBot:
    """Travel Advisory Chatbot powered by AI models"""
    
    def __init__(self, model_type: str = "huggingface"):
        print(f"🔄 Initializing TravelBot with {model_type} model...")
        start_time = time.time()
        
        self.model = ModelFactory.create_model(model_type)
        self.processor_factory = ProcessorFactory()
        self.logger = setup_logging()
        
        # Log model initialization
        if self.model.is_available():
            init_time = time.time() - start_time
            print(f"✅ Initialized with {model_type} model successfully (took {init_time:.2f}s)")
            self.logger.info(f"Model {model_type} initialized in {init_time:.2f}s")
        else:
            print(f"⚠️ {model_type} model initialization failed")
            self.logger.warning(f"{model_type} model initialization failed")
        
    async def process_message(self, message: str, context: Optional[str] = None) -> str:
        """Process user message and generate response"""
        try:
            # Log user input
            print(f"👤 User: {message}")
            self.logger.info(f"Processing user message: {message}")
            start_time = time.time()
            
            # Process the input
            processor = self.processor_factory.get_processor(InputType.TEXT)
            processed_input = await processor.process(message)
            
            # Extract trip duration if present
            num_days = self._extract_trip_duration(message)
            
            # Add duration to context if found
            if context:
                context += f"\nPlanning for {num_days} days."
            else:
                context = f"Planning for {num_days} days."
            
            # Generate response
            print("🤔 Generating response...")
            response = await self._generate_travel_response(processed_input, context)
            
            # Log completion time
            process_time = time.time() - start_time
            print(f"✅ Response generated in {process_time:.2f}s")
            print(f"🤖 Assistant: {response}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            print(f"❌ {error_msg}")
            self.logger.error(error_msg)
            return error_msg
    
    def _extract_trip_duration(self, message: str) -> int:
        """Extract the number of days from the user's message"""
        match = re.search(r'(\d+)\s*days?', message.lower())
        return int(match.group(1)) if match else 3  # Default to 3 days
    
    async def _generate_travel_response(self, processed_input: ProcessedInput, context: Optional[str] = None) -> str:
        """Generate travel-specific response"""
        try:
            start_time = time.time()
            
            # Generate response using the model
            response = await self.model.generate_response(processed_input.content, context)
            
            # Log the interaction with timing
            gen_time = time.time() - start_time
            self.logger.info(
                f"Response generated in {gen_time:.2f}s\n"
                f"Input: {processed_input.content}\n"
                f"Context: {context}\n"
                f"Response: {response}"
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    async def analyze_input(self, processed_input: ProcessedInput) -> Dict[str, Any]:
        """Analyze input for travel insights"""
        print(f"📊 Analyzing input: {processed_input.content[:100]}...")
        return await self.model.analyze_input(processed_input) 