"""
Data models for Travel Advisory Bot
This file contains only data classes and enums, no abstract base classes
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class InputType(Enum):
    TEXT = "text"
    IMAGE = "image" 
    PDF = "pdf"
    VIDEO = "video"
    AUDIO = "audio"

class TravelPlanType(Enum):
    QUICK_GETAWAY = "quick_getaway"
    WEEKEND_TRIP = "weekend_trip"
    WEEK_VACATION = "week_vacation"
    EXTENDED_TRAVEL = "extended_travel"
    BUSINESS_TRIP = "business_trip"
    ADVENTURE = "adventure"
    CULTURAL = "cultural"
    RELAXATION = "relaxation"

@dataclass
class TravelPreferences:
    """User travel preferences"""
    budget_range: Optional[str] = None
    travel_style: Optional[str] = None
    interests: List[str] = None
    mobility_requirements: Optional[str] = None
    dietary_restrictions: List[str] = None
    accommodation_type: Optional[str] = None
    group_size: int = 1
    age_group: Optional[str] = None
    language_preference: str = "en"

@dataclass
class ProcessedInput:
    """Processed input from various sources"""
    input_type: InputType
    content: str
    metadata: Dict[str, Any]
    language: str
    confidence: float
    extracted_entities: Dict[str, Any]

@dataclass
class TravelSuggestion:
    """Travel suggestion with details"""
    destination: str
    description: str
    best_time_to_visit: str
    estimated_budget: str
    duration: str
    activities: List[str]
    accommodation_suggestions: List[str]
    transportation: List[str]
    local_tips: List[str]
    weather_info: Optional[str] = None
    safety_info: Optional[str] = None

@dataclass
class Itinerary:
    """Detailed travel itinerary"""
    destination: str
    total_days: int
    total_budget: Optional[str]
    daily_plans: List[Dict[str, Any]]
    accommodation_details: List[Dict[str, Any]]
    transportation_details: List[Dict[str, Any]]
    emergency_contacts: List[Dict[str, str]]
    packing_list: List[str]
    important_notes: List[str]
