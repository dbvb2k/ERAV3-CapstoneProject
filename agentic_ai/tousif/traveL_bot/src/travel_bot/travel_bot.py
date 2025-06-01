from typing import List, Any, Optional
import asyncio
from .models import InputType, TravelPreferences, TravelSuggestion, Itinerary, ProcessedInput
from .processors import ProcessorFactory
from .agents.travel_agent import SmartTravelAgent
from .utils.file_utils import FileUtils

class TravelBot:
    """Main Travel Advisory Bot orchestrator"""
    
    def __init__(self, model_type: str = "devstral"):
        self.processor_factory = ProcessorFactory()
        self.travel_agent = SmartTravelAgent(model_type)
        self.file_utils = FileUtils()
    
    async def process_input(self, input_data: Any, input_type: InputType) -> ProcessedInput:
        """Process any type of input (text, image, pdf, video, audio)"""
        processor = self.processor_factory.get_processor(input_type)
        if not processor:
            raise ValueError(f"No processor available for input type: {input_type}")
        
        return await processor.process(input_data)
    
    async def get_travel_suggestions(self, 
                                   input_data: Any,
                                   input_type: InputType,
                                   preferences: Optional[TravelPreferences] = None) -> List[TravelSuggestion]:
        """Get travel suggestions based on input and preferences"""
        
        # Use default preferences if none provided
        if preferences is None:
            preferences = TravelPreferences()
        
        # Process the input
        processed_input = await self.process_input(input_data, input_type)
        
        # Generate suggestions
        suggestions = await self.travel_agent.create_suggestions(processed_input, preferences)
        
        return suggestions
    
    async def create_itinerary(self,
                             destination: str,
                             days: int,
                             preferences: Optional[TravelPreferences] = None) -> Itinerary:
        """Create detailed itinerary for a specific destination"""
        
        if preferences is None:
            preferences = TravelPreferences()
        
        itinerary = await self.travel_agent.create_itinerary(destination, days, preferences)
        
        return itinerary
    
    async def chat_with_bot(self, 
                           message: str,
                           context: Optional[str] = None) -> str:
        """Chat interface for general travel questions"""
        
        processed_input = await self.process_input(message, InputType.TEXT)
        
        # Use the AI model directly for general chat
        chat_prompt = f"""
        You are a helpful travel assistant. Answer the user's travel-related question based on the context provided.
        
        User Question: {message}
        Processed Content: {processed_input.content}
        Detected Entities: {processed_input.extracted_entities}
        
        Provide a helpful, informative response about travel, destinations, or trip planning.
        """
        
        response = await self.travel_agent.ai_model.generate_response(chat_prompt, context)
        return response
    
    def get_supported_file_types(self) -> dict:
        """Get supported file types for upload"""
        return {
            "Text": [".txt"],
            "PDF": [".pdf"],
            "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
            "Video": [".mp4", ".avi", ".mov", ".wmv"],
            "Audio": [".mp3", ".wav", ".ogg", ".m4a"]
        }
    
    async def analyze_travel_document(self, file_path: str) -> dict:
        """Analyze uploaded travel document and extract relevant information"""
        try:
            input_type = self.file_utils.detect_file_type(file_path)
            
            with open(file_path, 'rb') as file:
                processed_input = await self.process_input(file, input_type)
            
            analysis = await self.travel_agent.ai_model.analyze_input(processed_input)
            
            return {
                "file_type": input_type.value,
                "content": processed_input.content,
                "language": processed_input.language,
                "confidence": processed_input.confidence,
                "entities": processed_input.extracted_entities,
                "ai_analysis": analysis
            }
            
        except Exception as e:
            return {
                "error": f"Failed to analyze document: {str(e)}"
            }
    
    def get_travel_templates(self) -> dict:
        """Get travel planning templates"""
        return {
            "quick_getaway": {
                "name": "Quick Getaway (1-3 days)",
                "description": "Short trips for weekend breaks",
                "typical_duration": "1-3 days",
                "best_for": ["Weekend breaks", "City exploration", "Nearby destinations"]
            },
            "week_vacation": {
                "name": "Week Vacation (5-7 days)",
                "description": "Standard vacation length",
                "typical_duration": "5-7 days", 
                "best_for": ["International trips", "Multiple cities", "Relaxation"]
            },
            "extended_travel": {
                "name": "Extended Travel (2+ weeks)",
                "description": "Long-term travel experiences",
                "typical_duration": "2+ weeks",
                "best_for": ["Backpacking", "Multi-country tours", "Cultural immersion"]
            },
            "business_trip": {
                "name": "Business Trip",
                "description": "Work-related travel with leisure options",
                "typical_duration": "3-5 days",
                "best_for": ["Conference attendance", "Business meetings", "Corporate travel"]
            }
        }