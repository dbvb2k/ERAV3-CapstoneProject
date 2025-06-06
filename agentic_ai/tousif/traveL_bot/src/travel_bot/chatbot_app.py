from typing import Dict, Any, Optional
import logging
import re
import time
import asyncio
import nest_asyncio
from .models import AIModel, ModelFactory
from .processors import ProcessorFactory, ProcessedInput, InputType

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

def setup_logging():
    """Configure logging for the chatbot"""
    # Create a custom formatter for better visibility
    class VisibleFormatter(logging.Formatter):
        def format(self, record):
            if record.levelname == 'INFO':
                # Add visual separators for better readability
                if "Query:" in str(record.msg):
                    return "\n" + "="*50 + f"\n📝 {record.msg}\n" + "="*50
                elif "Response:" in str(record.msg):
                    return "\n" + "="*50 + f"\n✨ {record.msg}\n" + "="*50
            return super().format(record)

    # Configure root logger
    logging.basicConfig(level=logging.INFO)
    
    # Create console handler with better formatting
    console = logging.StreamHandler()
    console.setFormatter(VisibleFormatter('%(message)s'))
    
    # Get logger and update handlers
    logger = logging.getLogger(__name__)
    logger.handlers = []
    logger.addHandler(console)
    
    return logger

# Initialize logger
logger = setup_logging()

class TravelBot:
    """Travel Advisory Chatbot powered by AI models"""
    
    def __init__(self, model_type: str = "huggingface"):
        print(f"\n🔄 Initializing TravelBot with {model_type} model...")
        self.model = ModelFactory.create_model(model_type)
        self.processor_factory = ProcessorFactory()
        self.logger = logger
        
    async def process_message(self, message: str, context: Optional[str] = None) -> str:
        """Process user message and generate response"""
        try:
            # Log user input with clear formatting
            self.logger.info(f"Query: {message}")
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
            response = await self._generate_travel_response(processed_input, context)
            
            # Log completion time and response
            process_time = time.time() - start_time
            self.logger.info(f"Response generated in {process_time:.2f}s:\n{response}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
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