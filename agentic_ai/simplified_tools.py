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

# Mapping of major Indian and International cities to their IATA codes
city_to_iata = {
    # Indian Cities
    "Mumbai": "BOM",
    "Delhi": "DEL",
    "Bengaluru": "BLR",
    "Chennai": "MAA",
    "Kolkata": "CCU",
    "Hyderabad": "HYD",
    "Pune": "PNQ",
    "Ahmedabad": "AMD",
    "Goa": "GOI",
    "Jaipur": "JAI",
    "Kochi": "COK",
    
    # International Cities
    "New York": "JFK",
    "London": "LHR",
    "Paris": "CDG",
    "Tokyo": "HND",
    "Dubai": "DXB",
    "Singapore": "SIN",
    "Hong Kong": "HKG",
    "Sydney": "SYD",
    "Los Angeles": "LAX",
    "San Francisco": "SFO",
    "Amsterdam": "AMS",
    "Frankfurt": "FRA",
    "Toronto": "YYZ",
    "Bangkok": "BKK",
    "Istanbul": "IST"
}

async def flight_search(origin: str, destination: str, depart_date: str, return_date: str = '', adults: int = 1) -> List[Dict]:
    """Search for flights between origin and destination on a specific date. 
        Args: 
            origin (str): Origin city (Mumbai, Delhi, Bengaluru, Chennai, Kolkata, Hyderabad, Pune, Ahmedabad, Goa, Jaipur, Kochi, New York, London, Paris, Tokyo, Dubai, Singapore, Hong Kong, Sydney, Los Angeles, San Francisco, Amsterdam, Frankfurt, Toronto, Bangkok, Istanbul) 
            destination (str): Destination city (Mumbai, Delhi, Bengaluru, Chennai, Kolkata, Hyderabad, Pune, Ahmedabad, Goa, Jaipur, Kochi, New York, London, Paris, Tokyo, Dubai, Singapore, Hong Kong, Sydney, Los Angeles, San Francisco, Amsterdam, Frankfurt, Toronto, Bangkok, Istanbul)
            depart_date (str): Departure date in YYYY-MM-DD format 
            return_date (str, optional): Return date in YYYY-MM-DD format 
            adults (int, optional): Number of adults. 
        Returns: 
            List[Dict]: List of flight options with details like departure/arrival times, airline, price, etc."""
    
    global city_to_iata
    api_key = os.getenv("RAPID_API_KEY")
    if not api_key:
        logger.warning("No RapidAPI key available for flight search")
        return [{"error": "API key missing"}]
    
    try:
        # Get IATA codes from city names
        origin_iata = city_to_iata.get(origin, "BOM")
        destination_iata = city_to_iata.get(destination, "DEL")
        
        logger.info(f"Searching flights from {origin} ({origin_iata}) to {destination} ({destination_iata}) on {depart_date}")
        
        # Set up API request
        url = "https://google-flights2.p.rapidapi.com/api/v1/searchFlights"
        
        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "google-flights2.p.rapidapi.com"
        }
        
        # Set up query parameters
        params = {
            "departure_id": origin_iata,
            "arrival_id": destination_iata,
            "outbound_date": depart_date,
            "travel_class": "ECONOMY",
            "adults": str(adults),
            "show_hidden": "1",
            "currency": "INR",
            "language_code": "en-US",
            "country_code": "IN"
        }
        
        # Add return date if provided
        if return_date:
            params["inbound_date"] = return_date
        
        # Make the API request
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"Flight API returned status {response.status}")
                    return [{"error": f"API error: {response.status}"}]
                
                data = await response.json()
                
                # Check if the API call was successful
                if not data.get("status", False):
                    logger.error(f"Flight API returned error: {data.get('message', 'Unknown error')}")
                    return [{"error": data.get("message", "Unknown API error")}]
                
                # Initialize flights list
                flights = []
                
                # Extract flight data from the topFlights array only
                top_flights = data.get("data", {}).get("itineraries", {}).get("topFlights", [])
                
                if not top_flights:
                    logger.info("No flights found")
                    return [{"message": "No flights found for the requested route and dates"}]
                
                # Process each flight
                for flight in top_flights:
                    flight_info = {
                        "departure_time": flight.get("departure_time"),
                        "arrival_time": flight.get("arrival_time"),
                        "duration": flight.get("duration", {}).get("text"),
                        "price": flight.get("price"),
                        "stops": flight.get("stops", 0),
                    }
                    
                    # Extract detailed flight information if available
                    if flight.get("flights") and len(flight.get("flights")) > 0:
                        flight_detail = flight["flights"][0]
                        flight_info["airline"] = flight_detail.get("airline", "Unknown")
                        flight_info["flight_number"] = flight_detail.get("flight_number", "Unknown")
                        
                        # Add departure airport details
                        dep_airport = flight_detail.get("departure_airport", {})
                        flight_info["departure_airport"] = dep_airport.get("airport_name")
                        flight_info["departure_code"] = dep_airport.get("airport_code")
                        
                        # Add arrival airport details
                        arr_airport = flight_detail.get("arrival_airport", {})
                        flight_info["arrival_airport"] = arr_airport.get("airport_name")
                        flight_info["arrival_code"] = arr_airport.get("airport_code")
                    
                    # Add any layover information if present
                    if flight.get("layovers"):
                        layovers = []
                        for layover in flight["layovers"]:
                            layovers.append({
                                "airport": layover.get("airport_name"),
                                "duration": layover.get("duration_label")
                            })
                        flight_info["layovers"] = layovers
                    
                    flights.append(flight_info)
                
                logger.info(f"Found {len(flights)} flights")
                return flights
                
    except Exception as e:
        logger.error(f"Flight API error: {str(e)}")
        return [{"error": f"Error: {str(e)}"}]


async def hotel_search(location: str, check_in: str, check_out: str, occupancy: int = 2) -> List[Dict]:
    """
    Search for hotels in a location between check-in and check-out dates.
    
    Args:
        location (str): City or location name
        check_in (str): Check-in date in YYYY-MM-DD format
        check_out (str): Check-out date in YYYY-MM-DD format
        occupancy (int, optional): Number of people staying. Defaults to 2.
    
    Returns:
        List[Dict]: List of hotel options with details like name, price, rating, etc.
    """
    api_key = os.getenv("RAPID_API_KEY")
    if not api_key:
        logger.warning("No RapidAPI key available for hotel search")
        return [{"error": "API key missing"}]
    
    try:
        # Format dates as comma-separated string
        dates = f"{check_in},{check_out}"
        
        logger.info(f"Searching hotels in {location} from {check_in} to {check_out} for {occupancy} people")
        
        # Set up API request
        url = "https://google-hotels-data.p.rapidapi.com/search"
        
        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "google-hotels-data.p.rapidapi.com"
        }
        
        # Set up query parameters
        params = {
            "query": location,
            "dates": dates,
            "occupancy": str(occupancy),
            "free_cancellation": "false",
            "accommodation": "hotels",
            "region": "in",
            "lang": "en",
            "currency": "INR"
        }
        
        # Make the API request
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"Hotel API returned status {response.status}")
                    return [{"error": f"API error: {response.status}"}]
                
                # Parse the response
                response_data = await response.json()
                body_content = response_data.get("body", "{}")
                
                # If body is a string, parse it as JSON
                if isinstance(body_content, str):
                    body_data = json.loads(body_content)
                else:
                    body_data = body_content
                
                # Initialize hotels list
                hotels = []
                
                # Extract hotel data from the organic list
                organic_hotels = body_data.get("organic", [])
                
                if not organic_hotels:
                    logger.info("No hotels found")
                    return [{"message": "No hotels found for the requested location and dates"}]
                
                # Process the top 4 hotels (or fewer if less than 4 are available)
                for hotel in organic_hotels[:4]:
                    hotel_info = {
                        "name": hotel.get("title", "Hotel Name Not Available"),
                        "price": hotel.get("price", "Price Not Available"),
                        "rating": hotel.get("rating", "Rating Not Available"),
                        "reviews_count": hotel.get("reviews_cnt", 0),
                        "link": hotel.get("link", "")
                    }
                    
                    # Add coordinates if available
                    if "coordinates" in hotel and len(hotel["coordinates"]) == 2:
                        hotel_info["latitude"] = hotel["coordinates"][0]
                        hotel_info["longitude"] = hotel["coordinates"][1]
                    
                    hotels.append(hotel_info)
                
                logger.info(f"Found {len(hotels)} hotels")
                return hotels
    
    except Exception as e:
        logger.error(f"Hotel API error: {str(e)}")
        return [{"error": f"Error: {str(e)}"}]


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
    from langchain.tools import StructuredTool
    
    # Create structured tools that properly handle multiple arguments
    return [
        StructuredTool.from_function(
            func=flight_search,
            name="FlightSearchTool",
            description="""Search for flights between origin and destination on a specific date. 
        Args: 
            origin (str): Origin city (Mumbai, Delhi, Bengaluru, Chennai, Kolkata, Hyderabad, Pune, Ahmedabad, Goa, Jaipur, Kochi, New York, London, Paris, Tokyo, Dubai, Singapore, Hong Kong, Sydney, Los Angeles, San Francisco, Amsterdam, Frankfurt, Toronto, Bangkok, Istanbul) 
            destination (str): Destination city (Mumbai, Delhi, Bengaluru, Chennai, Kolkata, Hyderabad, Pune, Ahmedabad, Goa, Jaipur, Kochi, New York, London, Paris, Tokyo, Dubai, Singapore, Hong Kong, Sydney, Los Angeles, San Francisco, Amsterdam, Frankfurt, Toronto, Bangkok, Istanbul)
            depart_date (str): Departure date in YYYY-MM-DD format 
            return_date (str, optional): Return date in YYYY-MM-DD format 
            adults (int, optional): Number of adults. 
        Returns: 
            List[Dict]: List of flight options with details like departure/arrival times, airline, price, etc."""
        ),
        StructuredTool.from_function(
            func=hotel_search,
            name="HotelSearchTool",
            description="""
    Search for hotels in a location between check-in and check-out dates.
    
    Args:
        location (str): City or location name
        check_in (str): Check-in date in YYYY-MM-DD format
        check_out (str): Check-out date in YYYY-MM-DD format
        occupancy (int, optional): Number of people staying. Defaults to 2.
    
    Returns:
        List[Dict]: List of hotel options with details like name, price, rating, etc.
    """
        ),
        StructuredTool.from_function(
            func=get_weather,
            name="WeatherTool",
            description="""Get weather information for a location.
    Args:
        location (str): City or location name
        date (str, optional): Date in YYYY-MM-DD format
    Returns:
        Weather information including temperature and description
    """
        ),
        StructuredTool.from_function(
            func=plan_itinerary,
            name="ItineraryPlannerTool",
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