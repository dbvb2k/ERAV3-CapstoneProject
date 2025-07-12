from typing import Dict, List, Optional
from datetime import datetime, timedelta
import aiohttp
import python_weather
import os
import json
import logging
from langchain.tools import Tool

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple tool functions - each performs a specific task

async def flight_search(origin: str, destination: str, date: str) -> List[Dict]:
    """
    Search for flights between origin and destination on a specific date.
    
    Args:
        origin: Origin city
        destination: Destination city
        date: Travel date in YYYY-MM-DD format
    
    Returns:
        List of flight options
    """
    api_key = os.getenv("RAPID_API_KEY")
    if not api_key:
        logger.warning("No RapidAPI key available for flight search")
        return [{"airline": "API key missing", "price": "N/A", "departure": origin, "arrival": destination}]
        
    try:
        # Convert date string to datetime if needed
        if isinstance(date, str):
            try:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
                date_str = date
            except ValueError:
                logger.warning(f"Invalid date format: {date}, using current date")
                date_obj = datetime.now()
                date_str = date_obj.strftime('%Y-%m-%d')
        else:
            date_str = date.strftime('%Y-%m-%d')
            
        # Function to convert city to SkyID format
        def convert_to_sky_id(city):
            clean_city = city.upper()
            for word in ['CITY', 'INTERNATIONAL', 'AIRPORT', ',', '.']:
                clean_city = clean_city.replace(word, '')
            sky_id = clean_city.strip()[:4]
            return sky_id.ljust(4, 'X')
            
        async with aiohttp.ClientSession() as session:
            headers = {
                'X-RapidAPI-Key': api_key,
                'X-RapidAPI-Host': 'skyscanner-api.p.rapidapi.com'
            }
            
            origin_id = convert_to_sky_id(origin)
            destination_id = convert_to_sky_id(destination)
            
            url = "https://skyscanner-api.p.rapidapi.com/v3e/browse/en-GB"
            params = {
                'originSkyId': origin_id,
                'destinationSkyId': destination_id,
                'date': date_str
            }
            
            logger.info(f"Searching flights from {origin} to {destination} on {date_str}")
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"Flight API returned status {response.status}")
                    return [{"airline": "API error", "price": "N/A", "departure": origin, "arrival": destination}]
                    
                data = await response.json()
                flights = []
                
                if data.get('itineraries', {}).get('results'):
                    for itinerary in data['itineraries']['results'][:5]:  # Limit to 5 options
                        if isinstance(itinerary, dict) and itinerary.get('pricingOptions'):
                            pricing = itinerary['pricingOptions'][0]
                            if isinstance(pricing, dict):
                                flights.append({
                                    'airline': pricing.get('agents', [{}])[0].get('name', 'Unknown Airline'),
                                    'departure': origin,
                                    'arrival': destination,
                                    'price': f"${pricing.get('price', {}).get('amount', 'N/A')}",
                                    'stops': itinerary.get('legs', [{}])[0].get('stopCount', 0)
                                })
                
                logger.info(f"Found {len(flights)} flights")
                return flights
                
    except Exception as e:
        logger.error(f"Flight API error: {str(e)}")
        return [{"airline": f"Error: {str(e)[:50]}", "price": "N/A", "departure": origin, "arrival": destination}]

async def hotel_search(location: str, check_in: str, check_out: str) -> List[Dict]:
    """
    Search for hotels in a location between check-in and check-out dates.
    
    Args:
        location: City or location name
        check_in: Check-in date in YYYY-MM-DD format
        check_out: Check-out date in YYYY-MM-DD format
    
    Returns:
        List of hotel options
    """
    api_key = os.getenv("RAPID_API_KEY")
    if not api_key:
        logger.warning("No RapidAPI key available for hotel search")
        return [{"name": "API key missing", "price": "N/A", "rating": "N/A"}]
        
    try:
        # Convert date strings to datetime objects if needed
        if isinstance(check_in, str):
            try:
                check_in_obj = datetime.strptime(check_in, '%Y-%m-%d')
            except ValueError:
                logger.warning(f"Invalid check-in date format: {check_in}, using current date")
                check_in_obj = datetime.now()
        else:
            check_in_obj = check_in
            
        if isinstance(check_out, str):
            try:
                check_out_obj = datetime.strptime(check_out, '%Y-%m-%d')
            except ValueError:
                logger.warning(f"Invalid check-out date format: {check_out}, using current date + 7")
                check_out_obj = datetime.now() + timedelta(days=7)
        else:
            check_out_obj = check_out
            
        async with aiohttp.ClientSession() as session:
            headers = {
                'X-RapidAPI-Key': api_key,
                'X-RapidAPI-Host': 'hotels4.p.rapidapi.com'
            }
            
            # Get location ID
            location_url = "https://hotels4.p.rapidapi.com/locations/v3/search"
            location_params = {
                'q': location,
                'locale': 'en_US',
                'langid': '1033'
            }
            
            logger.info(f"Searching hotels in {location}")
            async with session.get(location_url, headers=headers, params=location_params) as response:
                if response.status != 200:
                    logger.error(f"Hotel location API returned status {response.status}")
                    return [{"name": "API error", "price": "N/A", "rating": "N/A"}]
                    
                location_data = await response.json()
                if not location_data.get('suggestions', []):
                    logger.warning(f"No location found for: {location}")
                    return [{"name": f"Location not found: {location}", "price": "N/A", "rating": "N/A"}]
                
                # Get the first location ID
                location_id = None
                for suggestion in location_data['suggestions']:
                    if suggestion['group'] == 'CITY_GROUP':
                        if suggestion.get('entities'):
                            location_id = suggestion['entities'][0].get('destinationId')
                            break
                
                if not location_id:
                    logger.warning(f"Could not find location ID for: {location}")
                    return [{"name": f"Could not process location: {location}", "price": "N/A", "rating": "N/A"}]
                
                # Search for hotels
                properties_url = "https://hotels4.p.rapidapi.com/properties/v2/list"
                payload = {
                    "currency": "USD",
                    "eapid": 1,
                    "locale": "en_US",
                    "siteId": 300000001,
                    "destination": {"regionId": str(location_id)},
                    "checkInDate": {
                        "day": check_in_obj.day,
                        "month": check_in_obj.month,
                        "year": check_in_obj.year
                    },
                    "checkOutDate": {
                        "day": check_out_obj.day,
                        "month": check_out_obj.month,
                        "year": check_out_obj.year
                    },
                    "rooms": [{"adults": 1}],
                    "resultsStartingIndex": 0,
                    "resultsSize": 5  # Limit to 5 hotels
                }
                
                async with session.post(properties_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Hotel search API returned status {response.status}")
                        return [{"name": "API error", "price": "N/A", "rating": "N/A"}]
                        
                    hotels_data = await response.json()
                    hotels = []
                    
                    if hotels_data.get('data', {}).get('propertySearch', {}).get('properties'):
                        for hotel in hotels_data['data']['propertySearch']['properties']:
                            if isinstance(hotel, dict):
                                hotels.append({
                                    'name': hotel.get('name', 'Hotel Name Not Available'),
                                    'price': hotel.get('price', {}).get('formatted', 'Price Not Available'),
                                    'rating': hotel.get('reviews', {}).get('score', 'N/A'),
                                    'address': hotel.get('location', {}).get('address', {}).get('addressLine', 'Address Not Available'),
                                    'amenities': [amenity.get('text', '') for amenity in hotel.get('amenities', [])[:3] if isinstance(amenity, dict)]
                                })
                    
                    logger.info(f"Found {len(hotels)} hotels")
                    return hotels
                    
    except Exception as e:
        logger.error(f"Hotel API error: {str(e)}")
        return [{"name": f"Error: {str(e)[:50]}", "price": "N/A", "rating": "N/A"}]

async def get_weather(location: str, date: Optional[str] = None) -> Dict:
    """
    Get weather information for a location.
    
    Args:
        location: City or location name
        date: Date in YYYY-MM-DD format (optional, defaults to current date)
    
    Returns:
        Weather information dictionary
    """
    try:
        logger.info(f"Getting weather for {location}")
        async with python_weather.Client(unit=python_weather.METRIC) as client:
            weather = await client.get(location)
            
            return {
                'temperature': weather.temperature,
                'description': weather.description,
                'humidity': weather.humidity or 50  # Default if not available
            }
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
        return {
            'temperature': 'N/A',
            'description': 'Weather data unavailable',
            'humidity': 'N/A'
        }

async def plan_itinerary(location: str, duration: int, preferences: Dict = {}) -> Dict:
    """
    Generate a travel itinerary for a location.
    
    Args:
        location: Destination city or location
        duration: Trip duration in days
        preferences: Optional preferences dictionary
    
    Returns:
        Itinerary dictionary
    """
    # Use simpler itinerary format
    if preferences is None:
        preferences = {}
    
    try:
        logger.info(f"Planning itinerary for {location} ({duration} days)")
        
        # Create a simple itinerary
        itinerary = {
            "destination": location,
            "duration": duration,
            "daily_plans": [],
            "estimated_budget": preferences.get("budget_range", "Moderate") + " ($100-200 per day)",
            "best_time_to_visit": "Year-round with seasonal variations"
        }
        
        # Generate daily plans
        for day in range(1, min(duration + 1, 8)):  # Cap at 7 days
            daily_plan = {
                "day": day,
                "morning": f"Explore popular attractions in {location}",
                "afternoon": "Enjoy local cuisine for lunch and visit museums",
                "evening": "Experience local nightlife and dinner"
            }
            itinerary["daily_plans"].append(daily_plan)
            
        return itinerary
        
    except Exception as e:
        logger.error(f"Itinerary planning error: {str(e)}")
        return {
            "destination": location,
            "duration": duration,
            "error": f"Could not generate itinerary: {str(e)}",
            "daily_plans": []
        }

def get_langchain_tools():
    """
    Create LangChain tools using the simplified functions.
    
    Returns:
        List of LangChain Tool objects
    """
    return [
        Tool(
            name="FlightSearchTool",
            func=flight_search,
            description="""Search for flights between two cities on a specific date.
            Args:
                origin (str): Origin city
                destination (str): Destination city  
                date (str): Date in YYYY-MM-DD format
            Returns:
                List of flight options with airlines and prices
            """
        ),
        Tool(
            name="HotelSearchTool",
            func=hotel_search,
            description="""Search for hotels in a city between check-in and check-out dates.
            Args:
                location (str): City or location name
                check_in (str): Check-in date in YYYY-MM-DD format
                check_out (str): Check-out date in YYYY-MM-DD format
            Returns:
                List of hotel options with names, prices, and ratings
            """
        ),
        Tool(
            name="WeatherTool",
            func=get_weather,
            description="""Get weather information for a location.
            Args:
                location (str): City or location name
                date (str, optional): Date in YYYY-MM-DD format
            Returns:
                Weather information including temperature and description
            """
        ),
        Tool(
            name="ItineraryPlannerTool",
            func=plan_itinerary,
            description="""Generate a travel itinerary for a location.
            Args:
                location (str): Destination city or location
                duration (int): Trip duration in days
                preferences (dict, optional): Preferences dictionary
            Returns:
                Detailed itinerary with daily plans
            """
        )
    ]