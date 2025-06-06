from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from tools.travel_tools import ItineraryPlannerTool
import aiohttp
import json

@dataclass
class TravelRequest:
    origin: str
    destination: str
    start_date: datetime
    end_date: datetime
    num_travelers: int
    preferences: Dict[str, any]
    budget: Optional[float] = None

@dataclass
class TravelPreferences:
    budget_range: Optional[str] = None
    travel_style: Optional[str] = None
    interests: Optional[List[str]] = None
    group_size: int = 1
    language_preference: str = "en"
    dietary_restrictions: Optional[List[str]] = None
    accommodation_type: Optional[str] = None

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

@dataclass
class ProcessedInput:
    content: str
    extracted_entities: Dict[str, Any]

@dataclass
class Itinerary:
    travel_request: TravelRequest
    flights: List[Dict]
    hotels: List[Dict]
    activities: List[Dict]
    total_cost: float
    created_at: datetime = datetime.now()
    destination: Optional[str] = None
    total_days: Optional[int] = None
    total_budget: Optional[str] = None
    daily_plans: Optional[List[Dict]] = None
    accommodation_details: Optional[List[Dict]] = None
    transportation_details: Optional[List[Dict]] = None
    emergency_contacts: Optional[List[Dict]] = None
    packing_list: Optional[List[str]] = None
    important_notes: Optional[List[str]] = None

class TravelAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.memory = {}  # Will be replaced with proper memory system
        
        # Initialize tools
        self.itinerary_planner = config.get('tools', {}).get('itinerary_planner') or ItineraryPlannerTool(
            model_id="microsoft/phi-2"
        )
        
    async def plan_trip(self, travel_request: TravelRequest) -> Itinerary:
        """
        Main method to plan a complete trip based on the travel request.
        """
        flights = await self.search_flights(
            travel_request.origin,
            travel_request.destination,
            travel_request.start_date
        )
        
        hotels = await self.search_hotels(
            travel_request.destination,
            travel_request.start_date,
            travel_request.end_date
        )
        
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
        Search for available flights.
        """
        # Implementation using RapidAPI
        async with aiohttp.ClientSession() as session:
            headers = {
                'X-RapidAPI-Key': self.config.get('rapid_api_key'),
                'X-RapidAPI-Host': 'skyscanner-api.p.rapidapi.com'
            }
            
            try:
                # First get place IDs
                places_url = "https://skyscanner-api.p.rapidapi.com/v3/geo/hierarchy/flights/en-US"
                async with session.get(places_url, headers=headers) as response:
                    places_data = await response.json()
                    
                # Search flights
                search_url = "https://skyscanner-api.p.rapidapi.com/v3e/flights/live/search/create"
                payload = {
                    "query": {
                        "market": "US",
                        "locale": "en-US",
                        "currency": "USD",
                        "queryLegs": [
                            {
                                "originPlaceId": f"{origin}-sky",
                                "destinationPlaceId": f"{destination}-sky",
                                "date": date.strftime('%Y-%m-%d')
                            }
                        ],
                        "cabinClass": "CABIN_CLASS_ECONOMY",
                        "adults": 1
                    }
                }
                
                async with session.post(search_url, headers=headers, json=payload) as response:
                    data = await response.json()
                    flights = []
                    
                    if data.get('content', {}).get('results', {}).get('itineraries'):
                        for itinerary in data['content']['results']['itineraries']:
                            price_info = itinerary.get('pricingOptions', [{}])[0]
                            flight_info = itinerary.get('legs', [{}])[0]
                            
                            flights.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'price': price_info.get('price', {}).get('amount'),
                                'airline': flight_info.get('carriers', [{}])[0].get('name'),
                                'flight_number': flight_info.get('segments', [{}])[0].get('flightNumber'),
                                'departure': origin,
                                'arrival': destination
                            })
                    
                    return flights
            except Exception as e:
                print(f"Error searching flights: {str(e)}")
                return []
    
    async def search_hotels(self, location: str, check_in: datetime, check_out: datetime) -> List[Dict]:
        """
        Search for available hotels.
        """
        # Implementation using RapidAPI Hotels.com
        async with aiohttp.ClientSession() as session:
            headers = {
                'X-RapidAPI-Key': self.config.get('rapid_api_key'),
                'X-RapidAPI-Host': 'hotels4.p.rapidapi.com'
            }
            
            try:
                # Step 1: Get location ID
                location_url = "https://hotels4.p.rapidapi.com/locations/v3/search"
                location_params = {
                    'q': location,
                    'locale': 'en_US',
                    'langid': '1033'
                }
                
                async with session.get(location_url, headers=headers, params=location_params) as response:
                    location_data = await response.json()
                    
                    if not location_data.get('suggestions', []):
                        return []
                    
                    # Get the first location ID
                    location_id = None
                    for suggestion in location_data['suggestions']:
                        if suggestion['group'] == 'CITY_GROUP':
                            location_id = suggestion['entities'][0]['destinationId']
                            break
                    
                    if not location_id:
                        return []
                    
                    # Step 2: Search for hotels
                    properties_url = "https://hotels4.p.rapidapi.com/properties/v2/list"
                    payload = {
                        "currency": "USD",
                        "eapid": 1,
                        "locale": "en_US",
                        "siteId": 300000001,
                        "destination": {
                            "regionId": location_id
                        },
                        "checkInDate": {
                            "day": check_in.day,
                            "month": check_in.month,
                            "year": check_in.year
                        },
                        "checkOutDate": {
                            "day": check_out.day,
                            "month": check_out.month,
                            "year": check_out.year
                        },
                        "rooms": [{"adults": 1}],
                        "resultsStartingIndex": 0,
                        "resultsSize": 10
                    }
                    
                    async with session.post(properties_url, headers=headers, json=payload) as response:
                        hotels_data = await response.json()
                        hotels = []
                        
                        if hotels_data.get('data', {}).get('propertySearch', {}).get('properties'):
                            for hotel in hotels_data['data']['propertySearch']['properties']:
                                hotels.append({
                                    'name': hotel.get('name'),
                                    'id': hotel.get('id'),
                                    'price': hotel.get('price', {}).get('displayMessages', [{}])[0].get('lineItems', [{}])[0].get('price', {}).get('formatted'),
                                    'rating': hotel.get('reviews', {}).get('score'),
                                    'address': hotel.get('location', {}).get('address', {}).get('addressLine'),
                                    'amenities': [amenity.get('text') for amenity in hotel.get('amenities', [])[:5]] if hotel.get('amenities') else []
                                })
                        
                        return hotels
            except Exception as e:
                print(f"Error searching hotels: {str(e)}")
                return []
    
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
    
    def __init__(self, model_type: str = "devstral"):
        super().__init__(config={})
        self.model_type = model_type
    
    async def create_suggestions(self, 
                               processed_input: ProcessedInput,
                               preferences: TravelPreferences) -> List[TravelSuggestion]:
        """Create travel suggestions based on input and preferences"""
        
        print("\n=== User Input ===")
        print(f"Input text: {processed_input.content}")
        print(f"Extracted entities: {processed_input.extracted_entities}")
        print(f"Preferences: {preferences}")
        
        # Parse duration from input
        try:
            duration_str = processed_input.extracted_entities.get('duration', '5 days')
            # Extract first number from the duration string
            duration = int(''.join(c for c in duration_str if c.isdigit()) or '5')
        except (ValueError, TypeError):
            duration = 5
        
        # Create context from input and preferences
        context = self._build_context(processed_input, preferences)
        
        prompt = f"""You are a travel expert. Create 3 detailed travel suggestions for the following request.

Input: {processed_input.content}
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

Return exactly 3 destinations in a JSON array with detailed, specific information for each field."""
        
        response = await self.itinerary_planner.execute(
            location="multiple",
            duration=duration,  # Pass the specific duration
            preferences={"prompt": prompt, "context": context}
        )
        
        try:
            suggestions = []
            
            # Handle both list and dict responses
            if isinstance(response, dict):
                response = [response]
            
            for data in response:
                if isinstance(data, dict):
                    # Validate and clean the data
                    destination = data.get("destination", "Unknown")
                    if destination == "Unknown" and data.get("name"):
                        destination = f"{data['name']}, {data.get('country', 'Location Unknown')}"
                    
                    description = data.get("description", "")
                    if not description and data.get("Description"):
                        description = data["Description"]
                    
                    # Parse duration to ensure it's a number
                    try:
                        duration_val = int(''.join(c for c in str(data.get("duration", duration)) if c.isdigit()) or str(duration))
                    except (ValueError, TypeError):
                        duration_val = duration
                    
                    # Create suggestion with validated data
                    suggestion = TravelSuggestion(
                        destination=destination,
                        description=description,
                        best_time_to_visit=data.get("best_time_to_visit", "Check local seasons"),
                        estimated_budget=data.get("estimated_budget", "Varies based on preferences"),
                        duration=str(duration_val),  # Use the parsed duration
                        activities=data.get("activities", [])[:5] or ["Local exploration", "Cultural activities", "Food experiences", "Nature exploration", "Local markets"],
                        accommodation_suggestions=data.get("accommodation_suggestions", [])[:3] or ["Recommended hotels", "Local guesthouses", "Budget options"],
                        transportation=data.get("transportation", [])[:2] or ["Public transportation", "Walking tours"],
                        local_tips=data.get("local_tips", [])[:3] or ["Research local customs", "Learn basic phrases", "Follow local guidelines"],
                        weather_info=data.get("weather_info", "Check local weather conditions"),
                        safety_info=data.get("safety_info", "Follow standard travel precautions")
                    )
                    suggestions.append(suggestion)
            
            print("\n=== Processed Suggestions ===")
            for i, sugg in enumerate(suggestions, 1):
                print(f"\nSuggestion {i}:")
                print(f"Destination: {sugg.destination}")
                print(f"Description: {sugg.description}")
                print(f"Best Time: {sugg.best_time_to_visit}")
                print(f"Budget: {sugg.estimated_budget}")
                print(f"Duration: {sugg.duration} days")
                print(f"Activities: {', '.join(sugg.activities)}")
                print(f"Accommodations: {', '.join(sugg.accommodation_suggestions)}")
                print(f"Transportation: {', '.join(sugg.transportation)}")
                print(f"Local Tips: {', '.join(sugg.local_tips)}")
                print(f"Weather: {sugg.weather_info}")
                print(f"Safety: {sugg.safety_info}")
            
            # Ensure we have exactly 3 suggestions
            while len(suggestions) < 3:
                suggestions.append(self._create_fallback_suggestion(
                    f"Alternative Destination {len(suggestions) + 1}",
                    preferences,
                    duration
                ))
            
            return suggestions[:3]  # Return exactly 3 suggestions
            
        except Exception as e:
            print(f"\nError processing suggestions: {str(e)}")
            return [self._create_fallback_suggestion("Error processing travel suggestions", preferences, duration)]
    
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