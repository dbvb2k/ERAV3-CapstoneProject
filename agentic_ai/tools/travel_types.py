"""
Shared data types for the travel planning system.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

@dataclass
class TravelSuggestion:
    destination: str
    description: str
    best_time_to_visit: str
    estimated_budget: str
    duration: str
    activities: List[str]
    accommodation_suggestions: List[str]
    transportation: List[str]
    local_tips: List[str]
    weather_info: str
    safety_info: str
    flights: Union[str, List[Dict], None] = None  # Can be either a string description or list of flight dicts

@dataclass
class Itinerary:
    destination: str
    total_days: int
    total_cost: float
    flights: List[Dict[str, Any]] = field(default_factory=list)
    hotels: List[Dict[str, Any]] = field(default_factory=list)
    daily_plans: List[Dict[str, Any]] = field(default_factory=list)
    packing_list: List[str] = field(default_factory=list)
    important_notes: List[str] = field(default_factory=list)
    emergency_contacts: List[Dict[str, str]] = field(default_factory=list) 