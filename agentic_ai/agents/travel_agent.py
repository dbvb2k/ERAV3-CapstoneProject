from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from ..tools.travel_tools import ItineraryPlannerTool
import aiohttp

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
class Itinerary:
    travel_request: TravelRequest
    flights: List[Dict]
    hotels: List[Dict]
    activities: List[Dict]
    total_cost: float
    created_at: datetime = datetime.now()

class TravelAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.memory = {}  # Will be replaced with proper memory system
        
        # Initialize tools
        self.itinerary_planner = config.get('tools', {}).get('itinerary_planner') or ItineraryPlannerTool(
            hf_api_key=config.get('huggingface_api_key'),
            model_id="google/flan-t5-large"
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
        activity_cost = sum(activity.get('estimated_cost', 0) for activity in activities)
        
        return flight_cost + hotel_cost + activity_cost 