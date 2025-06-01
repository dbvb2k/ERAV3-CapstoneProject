"""
Abstract base classes for Travel Advisory Bot
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from .data_models import ProcessedInput, InputType, TravelPreferences, TravelSuggestion, Itinerary

class AIModel(ABC):
    """Abstract base class for AI models"""
    
    @abstractmethod
    async def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response from AI model"""
        pass
    
    @abstractmethod
    async def analyze_input(self, processed_input: ProcessedInput) -> Dict[str, Any]:
        """Analyze processed input and extract travel-related information"""
        pass

class InputProcessor(ABC):
    """Abstract base class for input processors"""
    
    @abstractmethod
    async def process(self, input_data: Any) -> ProcessedInput:
        """Process input data and return structured format"""
        pass
    
    @abstractmethod
    def supports_input_type(self, input_type: InputType) -> bool:
        """Check if processor supports given input type"""
        pass

class TravelAgent(ABC):
    """Abstract base class for travel agents"""
    
    @abstractmethod
    async def create_suggestions(self, 
                               processed_input: ProcessedInput,
                               preferences: TravelPreferences) -> List[TravelSuggestion]:
        """Create travel suggestions based on input and preferences"""
        pass
    
    @abstractmethod
    async def create_itinerary(self,
                             destination: str,
                             days: int,
                             preferences: TravelPreferences) -> Itinerary:
        """Create detailed itinerary for specific destination"""
        pass
