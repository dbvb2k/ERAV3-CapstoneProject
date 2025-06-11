from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from tools.travel_tools import ItineraryPlannerTool, FlightSearchTool, HotelSearchTool
from tools.travel_utils import logger
from tools.travel_types import TravelSuggestion, Itinerary
from pydantic import BaseModel, Field
import aiohttp
import json
import traceback
import re

@dataclass
class TravelRequest:
    origin: str
    destination: str
    start_date: datetime
    end_date: datetime
    num_travelers: int
    preferences: Dict[str, any]
    budget: Optional[float] = None

class TravelPreferences(BaseModel):
    budget_range: Optional[str] = None
    travel_style: Optional[str] = None
    interests: Optional[List[str]] = None
    group_size: int = Field(default=1)
    language_preference: str = Field(default="en")
    dietary_restrictions: Optional[List[str]] = None
    accommodation_type: Optional[str] = None

@dataclass
class ProcessedInput:
    content: str
    extracted_entities: Dict[str, Any]

class TravelAgent:
    def __init__(self, config: Dict = None):
        # Initialize with default config
        default_config = {
            'market': 'US',
            'language': 'en-US',
            'currency': 'USD',
            'budget_range': 'Budget',
            'num_travelers': 1,
            'children_ages': [],
            'cabin_class': 'CABIN_CLASS_ECONOMY',
            'tools': {}
        }
        
        # Update with provided config
        self.config = {**default_config, **(config or {})}
        
        # Initialize memory
        self.memory = {}
        
        # Initialize tools
        self.itinerary_planner = self.config.get('tools', {}).get('itinerary_planner')
        
        # Initialize flight and hotel search tools
        self.flight_search = self.config.get('tools', {}).get('flight_search') or FlightSearchTool(
            api_key=self.config.get('rapidapi_key')
        )
        self.hotel_search = self.config.get('tools', {}).get('hotel_search') or HotelSearchTool(
            api_key=self.config.get('rapidapi_key')
        )
    
    def update_config(self, preferences: TravelPreferences):
        """Update config with user preferences"""
        self.config.update({
            'language': preferences.language_preference,
            'num_travelers': preferences.group_size,
            'budget_range': preferences.budget_range,
            # Map budget range to cabin class
            'cabin_class': {
                'Budget': 'CABIN_CLASS_ECONOMY',
                'Moderate': 'CABIN_CLASS_PREMIUM_ECONOMY',
                'Luxury': 'CABIN_CLASS_BUSINESS'
            }.get(preferences.budget_range, 'CABIN_CLASS_ECONOMY')
        })
    
    async def plan_trip(self, travel_request: TravelRequest) -> Itinerary:
        """
        Main method to plan a complete trip based on the travel request.
        """
        flights = await self.search_flights(
            travel_request.origin,
            travel_request.destination,
            travel_request.start_date
        )
        print(f"Flights: {flights}")
        hotels = await self.search_hotels(
            travel_request.destination,
            travel_request.start_date,
            travel_request.end_date
        )
        print(f"Hotels: {hotels}")
        # Calculate trip duration in days
        duration = (travel_request.end_date - travel_request.start_date).days
        
        activities = await self.suggest_activities(
            travel_request.destination,
            [travel_request.start_date, travel_request.end_date],
            duration=duration,
            preferences=travel_request.preferences
        )
        
        total_cost = self._calculate_total_cost(flights, hotels, activities)
        
        return Itinerary(
            travel_request=travel_request,
            flights=flights or [],
            hotels=hotels or [],
            activities=activities or [],
            total_cost=total_cost
        )
    
    async def search_flights(self, origin: str, destination: str, date: datetime) -> List[Dict]:
        """
        Search for flights using the FlightSearchTool.
        """
        logger.log_info("Searching for flights", {
            "origin": origin,
            "destination": destination,
            "date": date.strftime('%Y-%m-%d')
        })
        
        try:
            flights = await self.flight_search.execute(origin, destination, date)
            logger.log_info("Found flights", {"count": len(flights), "flights": flights})
            return flights
        except Exception as e:
            logger.log_error(e, "TravelAgent.search_flights")
            # Return simulated data as fallback
            return [
                {
                    'date': date.strftime('%Y-%m-%d'),
                    'price': 800,
                    'airline': 'Global Airways',
                    'flight_number': 'GA123',
                    'departure': origin,
                    'arrival': destination
                }
            ]
    
    async def search_hotels(self, location: str, check_in: datetime, check_out: datetime) -> List[Dict]:
        """
        Search for hotels using the HotelSearchTool.
        """
        logger.log_info("Searching for hotels", {
            "location": location,
            "check_in": check_in.strftime('%Y-%m-%d'),
            "check_out": check_out.strftime('%Y-%m-%d')
        })
        
        try:
            hotels = await self.hotel_search.execute(location, check_in, check_out)
            logger.log_info("Found hotels", {"count": len(hotels), "hotels": hotels})
            return hotels
        except Exception as e:
            logger.log_error(e, "TravelAgent.search_hotels")
            # Return simulated data as fallback
            return [
                {
                    'name': 'Grand Hotel',
                    'rating': 4.5,
                    'price': '$200 per night',
                    'address': f'123 Main Street, {location}',
                    'amenities': ['Pool', 'Spa', 'Restaurant', 'Gym', 'Free WiFi']
                }
            ]
    
    async def get_location_info(self, location: str) -> Dict:
        """
        Get detailed information about a location.
        """
        # To be implemented using OpenStreetMap
        return {}
    
    async def suggest_activities(self, location: str, dates: List[datetime], duration: int = None, preferences: Dict = None) -> List[Dict]:
        """
        Suggest activities for the given location and dates.
        """
        try:
            if duration is None:
                duration = (dates[1] - dates[0]).days
            
            activities = await self.itinerary_planner.execute(
                location=location,
                duration=duration,
                preferences=preferences or {}
            )
            return activities
        except Exception as e:
            print(f"Error suggesting activities: {str(e)}")
            return []
    
    def save_to_memory(self, key: str, value: any):
        """
        Save information to agent's memory.
        """
        self.memory[key] = value
    
    def get_from_memory(self, key: str) -> any:
        """
        Retrieve information from agent's memory.
        """
        return self.memory.get(key)
    
    def _calculate_total_cost(self, flights: List[Dict], hotels: List[Dict], activities: List[Dict]) -> float:
        """
        Calculate the total cost of the trip.
        """
        flight_cost = sum(float(str(flight.get('price', '0')).replace('$', '').replace(',', '')) for flight in flights)
        hotel_cost = sum(float(str(hotel.get('price', '0')).replace('$', '').replace(',', '')) for hotel in hotels)
        activity_cost = sum(float(str(activity.get('price', '0')).replace('$', '').replace(',', '')) for activity in activities)
        
        return flight_cost + hotel_cost + activity_cost

class SmartTravelAgent(TravelAgent):
    """Intelligent travel agent that creates suggestions and itineraries"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        self.model_type = config.get('model_type', 'devstral') if config else 'devstral'
        logger.log_info("Initializing SmartTravelAgent", {
            "model_type": self.model_type,
            "config": config
        })
        
        # Initialize tools if provided in config
        if config and 'tools' in config:
            self.itinerary_planner = config['tools'].get('itinerary_planner')
        
        # Initialize default tool if not provided
        if not hasattr(self, 'itinerary_planner'):
            from tools.travel_tools import ItineraryPlannerTool
            self.itinerary_planner = ItineraryPlannerTool()
            logger.log_info("Using default ItineraryPlannerTool")
    
    def _validate_suggestion_data(self, data: Dict, duration: int) -> Dict:
        """Validate and clean suggestion data"""
        # Extract destination
        destination = data.get("destination", "Unknown")
        if destination == "Unknown" and data.get("name"):
            destination = f"{data['name']}, {data.get('country', 'Location Unknown')}"
        
        # Extract description
        description = data.get("description", "")
        if not description and data.get("Description"):
            description = data["Description"]
        
        # Parse duration
        try:
            duration_val = int(''.join(c for c in str(data.get("duration", duration)) if c.isdigit()) or str(duration))
        except (ValueError, TypeError):
            duration_val = duration
        
        # Extract or generate activities
        activities = data.get("activities", [])
        if isinstance(activities, str):
            activities = [activities]
        if not activities and data.get("activitySuggestions"):
            if isinstance(data["activitySuggestions"], list):
                activities = data["activitySuggestions"]
            elif isinstance(data["activitySuggestions"], dict):
                activities = [item["title"] for item in data["activitySuggestions"].get("options", [])]
        activities = activities[:5] or ["Local exploration", "Cultural activities", "Food experiences", "Nature exploration", "Local markets"]
        
        # Extract or generate accommodations
        accommodations = data.get("accommodation_suggestions", [])
        if isinstance(accommodations, str):
            accommodations = [accommodations]
        if not accommodations and data.get("accommodations"):
            if isinstance(data["accommodations"], list):
                accommodations = [acc.get("name", "Hotel") for acc in data["accommodations"]]
            elif isinstance(data["accommodations"], dict):
                accommodations = [data["accommodations"].get("name", "Hotel")]
        accommodations = accommodations[:3] or ["Recommended hotels", "Local guesthouses", "Budget options"]
        
        # Extract or generate transportation
        transportation = data.get("transportation", [])
        if isinstance(transportation, str):
            transportation = [transportation]
        if not transportation and data.get("transportInformation"):
            if isinstance(data["transportInformation"], dict):
                for mode, info in data["transportInformation"].items():
                    if isinstance(info, list):
                        transportation.extend([f"{mode}: {route.get('routeName', 'Local route')}" for route in info])
                    else:
                        transportation.append(f"{mode}: Local routes available")
        transportation = transportation[:2] or ["Public transportation", "Walking tours"]
        
        # Extract or generate local tips
        local_tips = data.get("local_tips", [])
        if isinstance(local_tips, str):
            local_tips = [local_tips]
        if not local_tips and data.get("localTips"):
            if isinstance(data["localTips"], list):
                local_tips = data["localTips"]
            elif isinstance(data["localTips"], dict):
                local_tips = [tip["text"] for tip in data["localTips"].get("tips", [])]
        local_tips = local_tips[:3] or ["Research local customs", "Learn basic phrases", "Follow local guidelines"]
        
        return {
            "destination": destination,
            "description": description,
            "best_time_to_visit": data.get("best_time_to_visit", "Check local seasons"),
            "estimated_budget": data.get("estimated_budget", "Varies based on preferences"),
            "duration": str(duration_val),
            "activities": activities,
            "accommodation_suggestions": accommodations,
            "transportation": transportation,
            "local_tips": local_tips,
            "weather_info": data.get("weather_info", "Check local weather conditions"),
            "safety_info": data.get("safety_info", "Follow standard travel precautions")
        }

    async def create_suggestions(self, processed_input: ProcessedInput, preferences: TravelPreferences) -> List[TravelSuggestion]:
        """Create travel suggestions based on input and preferences"""
        
        logger.log_info("Creating Travel Suggestions", {
            "input_text": processed_input.content,
            "extracted_entities": processed_input.extracted_entities,
            "preferences": preferences.__dict__
        })
        
        # Parse duration from input
        try:
            duration_str = processed_input.extracted_entities.get('duration', '5 days')
            duration = int(''.join(c for c in duration_str if c.isdigit()) or '5')
        except (ValueError, TypeError):
            duration = 5
            logger.log_warning(f"Failed to parse duration from '{duration_str}', using default: {duration}")
        
        # Generate suggestions using LLM
        suggestions = await self._generate_suggestions(processed_input, preferences, duration)
        
        # For each suggestion, try to get real flight data
        for suggestion in suggestions:
            try:
                # Extract origin from input text or use default
                origin = None
                if "origin" in processed_input.extracted_entities:
                    origin = processed_input.extracted_entities["origin"]
                else:
                    # Try to find origin in input text
                    origin_match = re.search(r"from\s+([^,]+),?\s*", processed_input.content.lower())
                    if origin_match:
                        origin = origin_match.group(1).strip()
                
                if origin:
                    # Get flight data
                    flight_date = datetime.now() + timedelta(days=30)  # Default to 30 days from now
                    flights = await self.search_flights(origin, suggestion.destination, flight_date)
                    if flights:
                        suggestion.flights = flights
            except Exception as e:
                logger.log_error(e, "SmartTravelAgent.create_suggestions - flight search")
                # Keep the text description if flight search fails
                pass
        
        return suggestions

    async def _generate_suggestions(self, processed_input: ProcessedInput, preferences: TravelPreferences, duration: int) -> List[TravelSuggestion]:
        """Generate travel suggestions using the itinerary planner"""
        
        # Create context from input and preferences
        context = self._build_context(processed_input, preferences)
        
        prompt = f"""Based on the following travel request and preferences, suggest 2 detailed travel destinations:

Travel Request: {processed_input.content}
Duration: {duration} days

Preferences:
- Budget Level: {preferences.budget_range}
- Travel Style: {preferences.travel_style}
- Interests: {', '.join(preferences.interests or [])}
- Group Size: {preferences.group_size}
- Language: {preferences.language_preference}
- Dietary Restrictions: {', '.join(preferences.dietary_restrictions or [])}
- Accommodation Type: {preferences.accommodation_type}

Requirements:
1. Each destination must be suitable for {preferences.travel_style} travelers
2. All suggestions must fit within {preferences.budget_range} budget level
3. Accommodate dietary restrictions: {', '.join(preferences.dietary_restrictions or ['None specified'])}
4. Consider group size of {preferences.group_size} people
5. Focus on destinations that match interests: {', '.join(preferences.interests or [])}
6. Suggest specific accommodations that match {preferences.accommodation_type} preference
7. Include specific prices and details for all suggestions
8. Each suggestion should be exactly {duration} days
9. Include REAL destination names and specific locations
10. Provide DETAILED activities with actual places to visit

Return exactly 2 destinations in a JSON array with detailed, specific information for each field."""
        
        logger.log_info("Generated Prompt", {"prompt": prompt})
        
        try:
            # Get suggestions from the AI model
            suggestions = await self.itinerary_planner.execute(
                location="multiple",
                duration=duration,
                preferences={"prompt": prompt, "context": context}
            )
            
            logger.log_info("Raw Suggestions", {"suggestions": suggestions})
            
            # Convert to TravelSuggestion objects
            travel_suggestions = []
            for data in suggestions:
                # Validate and clean the data
                validated_data = self._validate_suggestion_data(data, duration)
                logger.log_info("Validated Suggestion Data", {"data": validated_data})
                
                # Create suggestion object
                suggestion = TravelSuggestion(**validated_data)
                travel_suggestions.append(suggestion)
            
            return travel_suggestions[:2]  # Return exactly 2 suggestions
            
        except Exception as e:
            logger.log_error(e, "SmartTravelAgent._generate_suggestions")
            # Create two fallback suggestions in case of error
            fallback_suggestions = [
                self._create_fallback_suggestion(f"Popular Travel Destination {i+1}", preferences, duration)
                for i in range(2)
            ]
            logger.log_info("Using fallback suggestions", {"suggestions": fallback_suggestions})
            return fallback_suggestions
    
    def _create_fallback_suggestion(self, text: str, preferences: TravelPreferences, duration: int = 5) -> TravelSuggestion:
        """Create a fallback suggestion with preference-aware defaults"""
        
        # Ensure we have valid lists for all list fields
        activities = [
            f"{preferences.travel_style.title()} activities in the area",
            "Local cultural experiences",
            "Food experiences suitable for your dietary needs",
            "Nature and outdoor activities",
            "Local market exploration"
        ] if preferences.travel_style else [
            "Local cultural experiences",
            "Traditional food tasting",
            "Historical site visits",
            "Nature exploration",
            "Local market tours"
        ]
        
        accommodation_type = preferences.accommodation_type or "Hotel"
        accommodation_suggestions = [
            f"{accommodation_type} in central location",
            f"Budget-friendly {accommodation_type.lower()}",
            "Local guesthouses with good reviews"
        ]
        
        transportation = [
            "Public transportation with route guidance",
            "Walking tours in safe areas"
        ]
        
        dietary_restrictions = preferences.dietary_restrictions or []
        local_tips = [
            f"Find {'-'.join(dietary_restrictions or ['local'])} friendly restaurants",
            "Learn basic local phrases",
            "Research local customs and traditions"
        ]
        
        return TravelSuggestion(
            destination=text,
            description=f"A destination selected to match your {preferences.travel_style or 'preferred'} travel style and {preferences.budget_range or 'moderate'} budget.",
            best_time_to_visit="Please check seasonal information",
            estimated_budget=f"Within {preferences.budget_range or 'moderate'} range",
            duration=str(duration),
            activities=activities,
            accommodation_suggestions=accommodation_suggestions,
            transportation=transportation,
            local_tips=local_tips,
            weather_info="Research current weather patterns",
            safety_info="Follow standard travel safety guidelines"
        )
    
    async def create_itinerary(self,
                             destination: str,
                             days: int,
                             preferences: TravelPreferences) -> Itinerary:
        """Create detailed itinerary for specific destination"""
        
        prompt = f"""
        Create a detailed {days}-day itinerary for {destination} with the following preferences:
        
        - Budget: {preferences.budget_range or 'Moderate'}
        - Travel Style: {preferences.travel_style or 'Balanced'}
        - Interests: {', '.join(preferences.interests or [])}
        - Group Size: {preferences.group_size}
        - Dietary Restrictions: {', '.join(preferences.dietary_restrictions or [])}
        - Accommodation Type: {preferences.accommodation_type or 'Hotel'}
        """
        
        response = await self.itinerary_planner.execute(
            location=destination,
            duration=days,
            preferences={"prompt": prompt}
        )
        
        try:
            itinerary_data = json.loads(response)
            
            return Itinerary(
                travel_request=TravelRequest(
                    origin="",  # To be filled by caller
                    destination=destination,
                    start_date=datetime.now(),  # To be filled by caller
                    end_date=datetime.now(),  # To be filled by caller
                    num_travelers=preferences.group_size,
                    preferences=preferences.__dict__,
                    budget=None  # To be calculated
                ),
                flights=[],  # To be filled by caller
                hotels=[],  # To be filled by caller
                activities=[],  # Will be filled from daily plans
                total_cost=0,  # To be calculated
                destination=itinerary_data.get("destination", destination),
                total_days=itinerary_data.get("total_days", days),
                total_budget=itinerary_data.get("total_budget"),
                daily_plans=itinerary_data.get("daily_plans", []),
                accommodation_details=itinerary_data.get("accommodation_details", []),
                transportation_details=itinerary_data.get("transportation_details", []),
                emergency_contacts=itinerary_data.get("emergency_contacts", []),
                packing_list=itinerary_data.get("packing_list", []),
                important_notes=itinerary_data.get("important_notes", [])
            )
            
        except json.JSONDecodeError:
            # Fallback itinerary
            daily_plans = []
            for day in range(1, days + 1):
                daily_plans.append({
                    "day": day,
                    "date": f"Day {day}",
                    "morning": "Explore local attractions",
                    "afternoon": "Cultural activities or sightseeing",
                    "evening": "Dinner and relaxation",
                    "meals": ["Local breakfast", "Local lunch", "Local dinner"],
                    "estimated_cost": "50-100 USD"
                })
            
            return Itinerary(
                travel_request=TravelRequest(
                    origin="",
                    destination=destination,
                    start_date=datetime.now(),
                    end_date=datetime.now(),
                    num_travelers=preferences.group_size,
                    preferences=preferences.__dict__,
                    budget=None
                ),
                flights=[],
                hotels=[],
                activities=[],
                total_cost=0,
                destination=destination,
                total_days=days,
                total_budget="Varies based on preferences",
                daily_plans=daily_plans,
                accommodation_details=[{
                    "name": "Recommended accommodation",
                    "type": "Hotel",
                    "location": "City center",
                    "price_range": "100-200 USD/night",
                    "amenities": ["WiFi", "Breakfast", "AC"]
                }],
                transportation_details=[{
                    "type": "Local transport",
                    "details": "Use public transportation or walking",
                    "cost": "10-20 USD/day"
                }],
                emergency_contacts=[{
                    "service": "Emergency Services",
                    "number": "Check local emergency numbers"
                }],
                packing_list=["Comfortable shoes", "Weather-appropriate clothing", "Travel documents", "Camera", "Medications"],
                important_notes=["Check visa requirements", "Research local customs", "Keep copies of important documents"]
            )
    
    def _build_context(self, processed_input: ProcessedInput, preferences: TravelPreferences) -> str:
        """Build context string for AI model"""
        context = f"""
        You are a professional travel advisor with expertise in creating personalized travel experiences.
        You have access to comprehensive knowledge about destinations worldwide, including:
        - Cultural attractions and activities
        - Accommodation options across all budgets
        - Transportation methods and costs
        - Local customs and etiquette
        - Safety considerations
        - Weather patterns
        - Budget planning
        
        Always provide practical, actionable advice while being sensitive to cultural differences.
        Consider the user's language preference: {preferences.language_preference}
        """
        return context 