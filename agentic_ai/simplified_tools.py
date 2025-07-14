from typing import Dict, List, Optional
from datetime import datetime, timedelta
import aiohttp
import python_weather
import os
import json
import logging
import difflib  # Add this at the top with other imports
from langchain.tools import StructuredTool
from tools_helper import *

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        # Enhanced city code matching
        origin_iata, origin_match_method, origin_matched_city = find_city_code(origin, city_to_iata)
        destination_iata, dest_match_method, dest_matched_city = find_city_code(destination, city_to_iata)
        
        logger.info(f"Origin: '{origin}' → '{origin_matched_city}' ({origin_iata}) [match method: {origin_match_method}]")
        logger.info(f"Destination: '{destination}' → '{dest_matched_city}' ({destination_iata}) [match method: {dest_match_method}]")
        
        logger.info(f"Searching flights from {origin_matched_city} ({origin_iata}) to {dest_matched_city} ({destination_iata}) on {depart_date}")

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

## TODO: Modify this to only require location and duration as parameters (to avoid preferences dict issue)
async def plan_itinerary(location: str, duration: int = 7) -> Dict:
    """
    Generate a travel itinerary for a location.
    
    Args:
        location: Destination city or location
        duration: Trip duration in days
    Returns:
        Itinerary dictionary
    """
    preferences = {
        "budget_range": "standard",
        "interests": ["sightseeing", "cultural activities", "nature"],
        "travel_style": "balanced"
    }
    
    try:
        logger.info(f"Planning itinerary for {location} ({duration} days)")
        
        # Get OpenRouter API key from environment
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            logger.warning("OpenRouter API key not found, using fallback itinerary")
            return _generate_fallback_itinerary(location, duration, preferences)
            
        # Prepare request to OpenRouter API
        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:8501"),
            "X-Title": os.getenv("SITE_NAME", "AI Travel Planner")
        }
        
        # Create prompt for detailed itinerary
        budget_level = preferences.get("budget_range", "moderate")
        interests = preferences.get("interests", ["sightseeing"])
        travel_style = preferences.get("travel_style", "balanced")
        
        # Template-driven version prompt
        prompt = f"""Create a detailed {duration}-day itinerary for {location}.

        Budget: {budget_level}
        Travel Style: {travel_style}
        Interests: {', '.join(interests) if isinstance(interests, list) else interests}

        Please include:
        1. A day-by-day breakdown with morning, afternoon, and evening activities
        2. Recommended attractions, restaurants and local experiences
        3. Estimated costs for major activities and overall daily budget
        4. Best time to visit and seasonal considerations
        5. Local transportation tips

        YOUR RESPONSE MUST BE A VALID JSON OBJECT EXACTLY MATCHING THIS FORMAT:

        ```json
        {{
        "destination": "{location}",
        "duration": {duration},
        "best_time_to_visit": "October to March (cool weather)",
        "estimated_budget": "₹8,000 - ₹10,000 per day",
        "daily_plans": [
            {{
            "day": 1,
            "morning": {{
                "activity": "Visit Popular Landmark",
                "time": "9:00 AM - 11:00 AM",
                "cost": "₹500"
            }},
            "afternoon": {{
                "activity": "Lunch and Shopping",
                "time": "12:00 PM - 3:00 PM",
                "cost": "₹1,000"
            }},
            "evening": {{
                "activity": "Dinner at Local Restaurant",
                "time": "7:00 PM - 9:00 PM",
                "cost": "₹800"
            }}
            }},
            {{
            "day": 2,
            "morning": {{
                "activity": "Another Attraction",
                "time": "9:00 AM - 11:00 AM",
                "cost": "₹400"
            }},
            "afternoon": {{
                "activity": "Visit Museum",
                "time": "12:00 PM - 3:00 PM",
                "cost": "₹600"
            }},
            "evening": {{
                "activity": "Cultural Show",
                "time": "7:00 PM - 9:00 PM",
                "cost": "₹1,200"
            }}
            }}
            // Add more days as needed to match the specified duration
        ],
        "transportation_tips": "Use metro for city travel, auto-rickshaws for short distances"
        }}
        """
        # Prepare OpenRouter API request
        payload = {
            "model": "meta-llama/llama-3-8b-instruct",
            "messages": [
                {"role": "system", "content": "You are a travel planning assistant that creates detailed itineraries."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 2000
        }
        
        # Make the API request to OpenRouter
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API call failed: {error_text}")
                    return _generate_fallback_itinerary(location, duration, preferences)
                
                result = await response.json()
                
                # Extract content from response
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                # Try to parse JSON response using our robust parsing approach
                itinerary_data = None
                
                try:
                    # Step 1: Try extracting from code blocks
                    import re
                    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)

                    if json_match:
                        content_to_parse = json_match.group(1)
                        logger.info("Found JSON in code block, attempting to parse")
                        
                        # Try the new advanced parser first
                        itinerary_data = advanced_json_repair(content_to_parse, location, duration)
                        if itinerary_data:
                            logger.info("Successfully parsed JSON with advanced repair")
                        else:
                            # Fall back to standard parsing
                            try:
                                itinerary_data = json.loads(content_to_parse)
                                logger.info("Successfully parsed JSON from code block")
                            except json.JSONDecodeError as e:
                                logger.info(f"Failed to parse JSON from code block: {str(e)}")
                                # Try repair function
                                itinerary_data = repair_and_parse_json(content_to_parse, location, duration, preferences)
                    elif content.strip().startswith("{") and content.strip().endswith("}"):
                        # Direct JSON without code blocks
                        logger.info("Content appears to be direct JSON, attempting to parse")
                        
                        # Try the new advanced parser first
                        itinerary_data = advanced_json_repair(content, location, duration)
                        if itinerary_data:
                            logger.info("Successfully parsed direct JSON with advanced repair")
                        else:
                            try:
                                itinerary_data = json.loads(content)
                                logger.info("Successfully parsed direct JSON")
                            except json.JSONDecodeError as e:
                                logger.info(f"Failed to parse direct JSON: {str(e)}")
                                # Try repair function
                                itinerary_data = repair_and_parse_json(content, location, duration, preferences)
                    else:
                        # Try the advanced parser on the whole content
                        itinerary_data = advanced_json_repair(content, location, duration)
                        if itinerary_data:
                            logger.info("Successfully parsed content with advanced repair")
                        else:
                            # Try to repair and parse the entire content
                            logger.info("Content does not appear to be JSON, attempting repair and extraction")
                            itinerary_data = repair_and_parse_json(content, location, duration, preferences)

                    # If we still couldn't parse JSON, generate a fallback itinerary
                    if not itinerary_data:
                        logger.warning("All JSON parsing attempts failed, using template-based fallback")
                        itinerary_data = _generate_fallback_itinerary(location, duration, preferences)

                    # Ensure expected keys exist
                    required_keys = ["destination", "duration", "daily_plans", "best_time_to_visit", "estimated_budget"]
                    for key in required_keys:
                        if key not in itinerary_data:
                            if key == "destination":
                                itinerary_data[key] = location
                            elif key == "duration":
                                itinerary_data[key] = duration
                            elif key == "daily_plans":
                                itinerary_data[key] = []
                            elif key == "best_time_to_visit":
                                itinerary_data[key] = "Year-round with seasonal variations"
                            elif key == "estimated_budget":
                                itinerary_data[key] = f"{preferences.get('budget_range', 'moderate').capitalize()} budget (estimate unavailable)"

                    # Normalize day plans structure to ensure consistency
                    for day_plan in itinerary_data.get("daily_plans", []):
                        # Ensure each period has the right structure
                        for period in ["morning", "afternoon", "evening"]:
                            if period not in day_plan:
                                day_plan[period] = f"Explore {location}"
                            elif isinstance(day_plan[period], dict):
                                # Ensure minimum required fields
                                if "activity" not in day_plan[period]:
                                    day_plan[period]["activity"] = f"Explore {location}"

                    return itinerary_data
                except Exception as e:
                    logger.error(f"Failed to parse itinerary JSON: {str(e)}")
                    return _extract_itinerary_from_text(content, location, duration, preferences)
                
    except Exception as e:
        logger.error(f"Itinerary planning error: {str(e)}")
        return _generate_fallback_itinerary(location, duration, preferences)


def _generate_fallback_itinerary(location: str, duration: int, preferences: Dict = {}) -> Dict:
    """Generate a simple fallback itinerary when API calls fail."""
    itinerary = {
        "destination": location,
        "duration": duration,
        "daily_plans": [],
        "estimated_budget": preferences.get("budget_range", "Moderate") + " ($100-200 per day)",
        "best_time_to_visit": "Year-round with seasonal variations"
    }
    
    # Generate daily plans
    for day in range(1, min(duration + 1, 8)):
        daily_plan = {
            "day": day,
            "morning": f"Explore popular attractions in {location}",
            "afternoon": "Enjoy local cuisine for lunch and visit museums",
            "evening": "Experience local nightlife and dinner"
        }
        itinerary["daily_plans"].append(daily_plan)
    
    return itinerary


def _extract_itinerary_from_text(text: str, location: str, duration: int, preferences: Dict = {}) -> Dict:
    """Extract structured itinerary data from text if JSON parsing fails."""
    itinerary = {
        "destination": location,
        "duration": duration,
        "daily_plans": [],
        "estimated_budget": preferences.get("budget_range", "Moderate") + " (estimate)",
        "best_time_to_visit": "Information unavailable"
    }
    
    try:
        # Try to extract best time to visit
        import re
        best_time_match = re.search(r"best time to visit:?\s*([^\.]+)", text, re.IGNORECASE)
        if best_time_match:
            itinerary["best_time_to_visit"] = best_time_match.group(1).strip()
        
        # Try to extract budget information
        budget_match = re.search(r"budget:?\s*([^\.]+)", text, re.IGNORECASE)
        if budget_match:
            itinerary["estimated_budget"] = budget_match.group(1).strip()
        
        # Try to extract daily plans
        day_patterns = [
            r"day\s+(\d+)[:\s]+([^D]+)",  # Day 1: content (until next Day)
            r"day\s+(\d+)\s*\n+([^D]+)",  # Day 1\n content (until next Day)
        ]
        
        days_found = False
        for pattern in day_patterns:
            day_matches = re.findall(pattern, text, re.IGNORECASE)
            if day_matches:
                days_found = True
                for day_num, day_content in day_matches:
                    day_plan = {"day": int(day_num.strip())}
                    
                    # Extract morning, afternoon, evening
                    morning_match = re.search(r"morning:?\s*([^A-Z]+)", day_content, re.IGNORECASE)
                    if morning_match:
                        day_plan["morning"] = morning_match.group(1).strip()
                    else:
                        day_plan["morning"] = f"Explore {location}"
                    
                    afternoon_match = re.search(r"afternoon:?\s*([^A-Z]+)", day_content, re.IGNORECASE)
                    if afternoon_match:
                        day_plan["afternoon"] = afternoon_match.group(1).strip()
                    else:
                        day_plan["afternoon"] = "Explore local attractions"
                    
                    evening_match = re.search(r"evening:?\s*([^A-Z]+)", day_content, re.IGNORECASE)
                    if evening_match:
                        day_plan["evening"] = evening_match.group(1).strip()
                    else:
                        day_plan["evening"] = "Enjoy local cuisine and nightlife"
                    
                    itinerary["daily_plans"].append(day_plan)
                
                break
        
        # If no days found, create simple daily plans
        if not days_found:
            for day in range(1, min(duration + 1, 8)):
                daily_plan = {
                    "day": day,
                    "morning": f"Explore popular attractions in {location}",
                    "afternoon": "Enjoy local cuisine for lunch and visit museums",
                    "evening": "Experience local nightlife and dinner"
                }
                itinerary["daily_plans"].append(daily_plan)
        
        return itinerary
        
    except Exception as e:
        logger.error(f"Error extracting itinerary from text: {str(e)}")
        return _generate_fallback_itinerary(location, duration, preferences)
    
    
def get_langchain_tools():
    """
    Create LangChain tools using the simplified functions.
    
    Returns:
        List of LangChain Tool objects
    """    
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
Returns:
    Detailed itinerary with daily plans
"""
        )
    ]