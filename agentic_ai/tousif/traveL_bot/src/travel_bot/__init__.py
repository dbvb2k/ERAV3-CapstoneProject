"""
Travel Advisory Bot - Agentic AI Application for Travel Planning

This package provides a comprehensive travel advisory bot that can:
- Process multimodal inputs (text, images, PDFs, videos, audio)
- Generate personalized travel suggestions
- Create detailed itineraries
- Provide multilingual support
- Chat about travel-related questions
"""

from .travel_bot import TravelBot
from .models import (
    InputType, TravelPreferences, TravelSuggestion, 
    Itinerary, ProcessedInput, TravelPlanType
)

__version__ = "0.1.0"
__author__ = "Travel Bot Team"

__all__ = [
    "TravelBot",
    "InputType", 
    "TravelPreferences",
    "TravelSuggestion",
    "Itinerary",
    "ProcessedInput",
    "TravelPlanType"
]