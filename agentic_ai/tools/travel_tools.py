from typing import Dict, List, Optional
from datetime import datetime
import aiohttp
import python_weather
from forex_python.converter import CurrencyRates
from overpass import API
from geopy.geocoders import Nominatim
from abc import ABC, abstractmethod
from transformers import pipeline, AutoTokenizer
try:
    from transformers import AutoModelForSeq2SeqGeneration
except ImportError:
    from transformers import AutoModelForCausalLM as AutoModelForSeq2SeqGeneration
import asyncio
import json
import os
import streamlit as st
import re
import traceback
from .travel_utils import logger

class BaseTravelTool(ABC):
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        
    @abstractmethod
    async def execute(self, *args, **kwargs):
        pass

class FlightSearchTool(BaseTravelTool):
    async def execute(self, origin: str, destination: str, date: datetime) -> List[Dict]:
        """
        Search for flights using Fly Scraper API through RapidAPI.
        """
        if not self.api_key:
            raise Exception("RapidAPI key not available for flight search")
            
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'X-RapidAPI-Key': self.api_key,
                    'X-RapidAPI-Host': 'skyscanner-api.p.rapidapi.com'
                }
                
                # Convert city names to SkyID format
                origin_id = self._convert_to_sky_id(origin)
                destination_id = self._convert_to_sky_id(destination)
                
                # Format date for API
                date_str = date.strftime('%Y-%m-%d')
                
                url = "https://skyscanner-api.p.rapidapi.com/v3e/browse/en-GB"
                params = {
                    'originSkyId': origin_id,
                    'destinationSkyId': destination_id,
                    'date': date_str
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"Flight API returned status {response.status}")
                        
                    data = await response.json()
                    flights = []
                    
                    if data.get('itineraries', {}).get('results'):
                        for itinerary in data['itineraries']['results']:
                            if isinstance(itinerary, dict) and itinerary.get('pricingOptions'):
                                # Get the first pricing option
                                pricing = itinerary['pricingOptions'][0]
                                if isinstance(pricing, dict):
                                    flights.append({
                                        'airline': pricing.get('agents', [{}])[0].get('name', 'Unknown Airline'),
                                        'flight_number': 'N/A',  # Not available in this API
                                        'departure': origin,
                                        'arrival': destination,
                                        'departure_time': 'N/A',  # Not available in this API
                                        'arrival_time': 'N/A',    # Not available in this API
                                        'price': f"${pricing.get('price', {}).get('amount', 'N/A')}",
                                        'duration': 'N/A',        # Not available in this API
                                        'stops': itinerary.get('legs', [{}])[0].get('stopCount', 0)
                                    })
                    
                    return flights
                    
        except Exception as e:
            print(f"Flight API error: {str(e)}")
            raise Exception(f"Unable to retrieve flight data: {str(e)}")
    
    def _convert_to_sky_id(self, city: str) -> str:
        """
        Convert city name to SkyID format.
        This is a simple implementation - in a production environment, 
        you would want to use a proper city-to-airport code mapping.
        """
        # Remove common words and special characters
        clean_city = city.upper()
        for word in ['CITY', 'INTERNATIONAL', 'AIRPORT', ',', '.']:
            clean_city = clean_city.replace(word, '')
        
        # Take first 4 letters, pad with 'X' if needed
        sky_id = clean_city.strip()[:4]
        sky_id = sky_id.ljust(4, 'X')
        
        return sky_id

class HotelSearchTool(BaseTravelTool):
    async def execute(self, location: str, check_in: datetime, check_out: datetime) -> List[Dict]:
        """
        Search for hotels using Hotels.com API through RapidAPI (free tier).
        """
        if not self.api_key:
            raise Exception("RapidAPI key not available for hotel search")
            
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'X-RapidAPI-Key': self.api_key,
                    'X-RapidAPI-Host': 'hotels4.p.rapidapi.com'
                }
                
                # Get location ID
                location_url = "https://hotels4.p.rapidapi.com/locations/v3/search"
                location_params = {
                    'q': location,
                    'locale': 'en_US',
                    'langid': '1033'
                }
                
                async with session.get(location_url, headers=headers, params=location_params) as response:
                    if response.status != 200:
                        raise Exception(f"Hotel location API returned status {response.status}")
                        
                    location_data = await response.json()
                    if not location_data.get('suggestions', []):
                        raise Exception(f"No location found for: {location}")
                    
                    # Get the first location ID
                    location_id = None
                    for suggestion in location_data['suggestions']:
                        if suggestion['group'] == 'CITY_GROUP':
                            if suggestion.get('entities'):
                                location_id = suggestion['entities'][0].get('destinationId')
                                break
                    
                    if not location_id:
                        raise Exception(f"Could not find location ID for: {location}")
                    
                    # Search for hotels with proper error handling
                    properties_url = "https://hotels4.p.rapidapi.com/properties/v2/list"
                    payload = {
                        "currency": "USD",
                        "eapid": 1,
                        "locale": "en_US",
                        "siteId": 300000001,
                        "destination": {"regionId": str(location_id)},
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
                        if response.status != 200:
                            raise Exception(f"Hotel search API returned status {response.status}")
                            
                        hotels_data = await response.json()
                        hotels = []
                        
                        if hotels_data.get('data', {}).get('propertySearch', {}).get('properties'):
                            for hotel in hotels_data['data']['propertySearch']['properties']:
                                if isinstance(hotel, dict):  # Ensure hotel is a valid dictionary
                                    hotels.append({
                                        'name': hotel.get('name', 'Hotel Name Not Available'),
                                        'id': str(hotel.get('id', '')),
                                        'price': hotel.get('price', {}).get('formatted', 'Price Not Available'),
                                        'rating': hotel.get('reviews', {}).get('score', 'N/A'),
                                        'address': hotel.get('location', {}).get('address', {}).get('addressLine', 'Address Not Available'),
                                        'amenities': [amenity.get('text', '') for amenity in hotel.get('amenities', [])[:5] if isinstance(amenity, dict)]
                                    })
                        
                        return hotels
                        
        except Exception as e:
            print(f"Hotel API error: {str(e)}")
            raise Exception(f"Unable to retrieve hotel data: {str(e)}")

class WeatherTool(BaseTravelTool):
    async def execute(self, location: str, date: datetime) -> Dict:
        """
        Get weather information using python_weather (free).
        """
        try:
            async with python_weather.Client(unit=python_weather.METRIC) as client:
                weather = await client.get(location)
                
                # Get the current weather  # Use first forecast as current
                return {
                    'temperature': weather.temperature,
                    'description': weather.description,
                    'humidity': weather.humidity or 50  # Default if not available
                }
        except Exception as e:
            print(f"Weather API error: {str(e)}")
            return {
                'temperature': 'N/A',
                'description': 'Weather data unavailable',
                'humidity': 'N/A'
            }

class CurrencyTool(BaseTravelTool):
    def __init__(self):
        super().__init__()
        self.c = CurrencyRates()
        
    async def execute(self, amount: float, from_currency: str, to_currency: str) -> Dict:
        """
        Convert currency using forex-python (free).
        """
        rate = self.c.get_rate(from_currency, to_currency)
        converted = self.c.convert(from_currency, to_currency, amount)
        
        return {
            'original_amount': amount,
            'converted_amount': converted,
            'rate': rate,
            'from': from_currency,
            'to': to_currency
        }

class LocationInfoTool(BaseTravelTool):
    def __init__(self):
        super().__init__()
        self.geolocator = Nominatim(user_agent="travel_planner")
        
    async def execute(self, location: str) -> Dict:
        """Get location information and points of interest."""
        try:
            # Get location coordinates
            loc = self.geolocator.geocode(location)
            if not loc:
                raise Exception(f"Location not found: {location}")
            
            # Simplified query for better reliability
            query = f"""
            [out:json];
            area[name="{location}"]->.searchArea;
            (
                node["tourism"="information"](area.searchArea);
                node["tourism"="attraction"](area.searchArea);
            );
            out body;
            """
            
            api = API()
            result = api.get(query, responseformat="json")
            
            tips = []
            if result and 'elements' in result:
                for element in result['elements'][:5]:  # Limit to 5 POIs
                    if 'tags' in element:
                        name = element['tags'].get('name', 'Point of Interest')
                        tips.append(f"Visit {name}")
            
            if not tips:
                raise Exception(f"No points of interest found for: {location}")
            
            return {"tips": tips}
        except Exception as e:
            print(f"Location API error: {str(e)}")
            raise Exception(f"Unable to retrieve location information: {str(e)}")

@st.cache_resource
def get_cached_pipeline(model_id: str = "microsoft/phi-2"):
    """Initialize and cache the model pipeline"""
    try:
        return pipeline("text-generation", model=model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error initializing model pipeline: {e}")
        return None

class ItineraryPlannerTool(BaseTravelTool):
    """Tool for planning itineraries using AI"""
    
    def __init__(self, openrouter_api_key: str = None, site_url: str = None, site_name: str = None):
        """Initialize the tool with OpenRouter API configuration"""
        self.api_key = openrouter_api_key
        self.site_url = site_url or "http://localhost:8501"
        self.site_name = site_name or "AI Travel Planner"
        self.model = "meta-llama/llama-3.3-8b-instruct:free"  # Using Llama 2 70B through OpenRouter

    async def execute(self, location: str, duration: int, preferences: Dict) -> List[Dict]:
        """Execute the planning tool"""
        try:
            # Create a more structured system prompt
            system_prompt = """You are a travel expert AI assistant. Generate exactly 2 travel suggestions in this JSON format:

[
    {
        "destination": "Bangkok, Thailand",
        "description": "A vibrant city known for its street food, temples, and nightlife. Perfect for budget travelers seeking cultural experiences.",
        "best_time_to_visit": "November to March during the dry season",
        "estimated_budget": "$50-100 per day",
        "duration": "5",
        "activities": [
            "Visit the Grand Palace and Wat Phra Kaew",
            "Explore Chatuchak Weekend Market",
            "Take a Thai cooking class",
            "Temple hop to Wat Arun and Wat Pho",
            "Evening street food tour in Chinatown"
        ],
        "accommodation_suggestions": [
            "Lub d Bangkok Hostel ($15-20/night)",
            "Hotel Buddy Lodge ($40-60/night)",
            "Anantara Riverside ($150-200/night)"
        ],
        "transportation": [
            "BTS Skytrain and MRT ($1-2 per trip)",
            "Tuk-tuk and taxi ($3-10 per ride)"
        ],
        "local_tips": [
            "Always negotiate prices at markets",
            "Carry temple-appropriate clothing",
            "Use metered taxis instead of tuk-tuks at night"
        ],
        "weather_info": "Tropical climate with temperatures between 25-35°C year-round",
        "safety_info": "Generally safe for tourists. Be careful of scams near major attractions."
    }
]"""
            
            # Prepare the prompt from preferences
            prompt = preferences.get('prompt', '')
            context = preferences.get('context', '')
            
            # Combine prompt and context
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            logger.log_info("Sending Request to LLM", {
                "system_prompt": system_prompt,
                "user_prompt": full_prompt,
                "location": location,
                "duration": duration
            })
            
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            logger.log_api_request("OpenRouter Chat Completions", payload)
            
            # Make API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.log_error(Exception(f"API call failed: {error_text}"), "OpenRouter API Call")
                        raise Exception(f"API call failed with status {response.status}: {error_text}")
                    
                    result = await response.json()
                    logger.log_api_response("OpenRouter Chat Completions", result)
                    
                    if not result.get('choices'):
                        raise ValueError("No response choices found in API result")
                    
                    response_text = result['choices'][0]['message']['content']
                    logger.log_info("Extracted Response Text", {"text": response_text})
                    
                    try:
                        suggestions = json.loads(response_text)
                        if isinstance(suggestions, list):
                            validated = self._validate_suggestions(suggestions)
                            logger.log_info("Successfully parsed JSON response", {"suggestions": validated})
                            return validated
                        elif isinstance(suggestions, dict):
                            validated = self._validate_suggestions([suggestions])
                            logger.log_info("Successfully parsed single suggestion", {"suggestions": validated})
                            return validated
                    except json.JSONDecodeError:
                        logger.log_warning("JSON parse failed, attempting structured text parse")
                        return self._parse_structured_text(response_text)
                
        except Exception as e:
            logger.log_error(e, "ItineraryPlannerTool.execute")
            raise Exception(f"Unable to generate travel suggestions: {str(e)}")
    
    def _validate_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Validate and fix suggestions to ensure they meet requirements"""
        validated_data = []
        for suggestion in suggestions:
            if isinstance(suggestion, dict):
                # Parse duration to ensure it's a number
                try:
                    duration_str = str(suggestion.get("duration", "5"))
                    duration_val = int(''.join(c for c in duration_str if c.isdigit()) or '5')
                except (ValueError, TypeError):
                    duration_val = 5

                # Ensure all required fields are present
                validated_suggestion = {
                    "destination": suggestion.get("destination", "Unknown Location"),
                    "description": suggestion.get("description", "A great destination matching your preferences."),
                    "best_time_to_visit": suggestion.get("best_time_to_visit", "Year-round"),
                    "estimated_budget": suggestion.get("estimated_budget", "Varies based on preferences"),
                    "duration": str(duration_val),
                    "activities": suggestion.get("activities", [
                        "Local sightseeing",
                        "Cultural experiences",
                        "Food tasting",
                        "Nature exploration",
                        "Local markets"
                    ])[:5],
                    "accommodation_suggestions": suggestion.get("accommodation_suggestions", [
                        "Local hotel",
                        "Budget guesthouse",
                        "Boutique hostel"
                    ])[:3],
                    "transportation": suggestion.get("transportation", [
                        "Public transport",
                        "Walking tours"
                    ])[:2],
                    "local_tips": suggestion.get("local_tips", [
                        "Research local customs",
                        "Learn basic phrases",
                        "Follow local guidelines"
                    ])[:3],
                    "weather_info": suggestion.get("weather_info", "Check local weather before booking"),
                    "safety_info": suggestion.get("safety_info", "Follow standard travel precautions")
                }
                validated_data.append(validated_suggestion)
        
        # Ensure exactly 2 suggestions
        while len(validated_data) < 2:
            validated_data.append({
                "destination": f"Alternative Destination {len(validated_data) + 1}",
                "description": "A great destination matching your preferences.",
                "best_time_to_visit": "Year-round",
                "estimated_budget": "Within your specified budget",
                "duration": "5",
                "activities": [
                    "Local sightseeing",
                    "Cultural experiences",
                    "Food tasting",
                    "Nature exploration",
                    "Local markets"
                ],
                "accommodation_suggestions": [
                    "Local hotel",
                    "Budget guesthouse",
                    "Boutique hostel"
                ],
                "transportation": [
                    "Public transport",
                    "Walking tours"
                ],
                "local_tips": [
                    "Research local customs",
                    "Learn basic phrases",
                    "Follow local guidelines"
                ],
                "weather_info": "Check local weather before booking",
                "safety_info": "Follow standard travel precautions"
            })
        
        return validated_data[:2]  # Return exactly 2 suggestions

    def _parse_structured_text(self, text: str) -> List[Dict]:
        """Parse structured text response"""
        suggestions = []
        current_suggestion = {}
        
        # Split by numbered sections or clear delimiters
        sections = re.split(r'(?:\d+[\)\.:]|Suggestion \d+:|\n\n+)', text)
        
        for section in sections:
            if not section.strip():
                continue
            
            lines = section.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try to match key-value pairs
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    
                    # Handle lists
                    if key in ['activities', 'accommodation_suggestions', 'transportation', 'local_tips']:
                        if '[' in value and ']' in value:
                            value = [v.strip().strip('"') for v in value.strip('[]').split(',')]
                        else:
                            value = [value]
                    
                    current_suggestion[key] = value
            
            if current_suggestion:
                suggestions.append(current_suggestion)
                current_suggestion = {}
        
        if suggestions:
            return self._validate_suggestions(suggestions)
        
        raise ValueError("Could not parse suggestions from text")