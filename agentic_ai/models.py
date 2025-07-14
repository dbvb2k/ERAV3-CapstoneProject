from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class TravelRequest:
    origin: str
    destination: str
    start_date: str
    end_date: str
    num_travelers: int
    preferences: Dict[str, Any]
    budget: Optional[float] = None

class TravelPreferences(BaseModel):
    budget_range: Optional[str] = None
    travel_style: Optional[str] = None
    interests: Optional[List[str]] = None
    group_size: int = Field(default=1)
    language_preference: str = Field(default="en")
    dietary_restrictions: Optional[List[str]] = None
    accommodation_type: Optional[str] = None
