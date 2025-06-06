"""
Models package for Travel Advisory Bot.

This package contains data models and abstract base classes used throughout the application.
Imports are organized to prevent circular dependencies.
"""

# Import data models
from .data_models import (
    InputType,
    TravelPlanType,
    TravelPreferences,
    ProcessedInput,
    TravelSuggestion,
    Itinerary
)

# Import abstract base classes
from .base_classes import (
    AIModel,
    InputProcessor,
    TravelAgent
)

# Define what gets exported when using "from travel_bot.models import *"
__all__ = [
    # Data models
    'InputType',
    'TravelPlanType', 
    'TravelPreferences',
    'ProcessedInput',
    'TravelSuggestion',
    'Itinerary',
    # Abstract base classes
    'AIModel',
    'InputProcessor',
    'TravelAgent'
]
