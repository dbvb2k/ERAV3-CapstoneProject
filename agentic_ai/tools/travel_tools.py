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

class BaseTravelTool(ABC):
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        
    @abstractmethod
    async def execute(self, *args, **kwargs):
        pass

class FlightSearchTool(BaseTravelTool):
    async def execute(self, origin: str, destination: str, date: datetime) -> List[Dict]:
        """
        Search for flights using Travelpayouts API through RapidAPI (free tier).
        """
        async with aiohttp.ClientSession() as session:
            headers = {
                'X-RapidAPI-Key': self.api_key,
                'X-RapidAPI-Host': 'travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com'
            }
            
            # Format date as YYYY-MM
            month = date.strftime('%Y-%m')
            
            url = f"https://travelpayouts-travelpayouts-flight-data-v1.p.rapidapi.com/v1/prices/calendar"
            params = {
                'calendar_type': 'departure_date',
                'destination': destination,
                'origin': origin,
                'month': month,
            }
            
            async with session.get(url, headers=headers, params=params) as response:
                data = await response.json()
                flights = []
                
                if data.get('success') and data.get('data'):
                    for date_str, flight_data in data['data'].items():
                        flights.append({
                            'date': date_str,
                            'price': flight_data.get('price'),
                            'airline': flight_data.get('airline'),
                            'flight_number': flight_data.get('flight_number'),
                            'departure': origin,
                            'arrival': destination
                        })
                
                return flights

class HotelSearchTool(BaseTravelTool):
    async def execute(self, location: str, check_in: datetime, check_out: datetime) -> List[Dict]:
        """
        Search for hotels using Hotels.com API through RapidAPI (free tier).
        """
        async with aiohttp.ClientSession() as session:
            headers = {
                'X-RapidAPI-Key': self.api_key,
                'X-RapidAPI-Host': 'hotels4.p.rapidapi.com'
            }
            
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

class WeatherTool(BaseTravelTool):
    async def execute(self, location: str, date: datetime) -> Dict:
        """
        Get weather information using python_weather (free).
        """
        client = python_weather.Client(unit=python_weather.METRIC)
        weather = await client.get(location)
        await client.close()
        
        return {
            'temperature': weather.current.temperature,
            'description': weather.current.description,
            'humidity': weather.current.humidity
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
        self.overpass_api = API()
        self.geocoder = Nominatim(user_agent="travel_agent")
        
    async def execute(self, location: str) -> Dict:
        """
        Get location information using OpenStreetMap (free).
        """
        # Get coordinates
        location_info = self.geocoder.geocode(location)
        
        # Get nearby places
        query = f"""
        [out:json];
        area[name="{location}"]->.searchArea;
        (
          node["tourism"](area.searchArea);
          way["tourism"](area.searchArea);
          node["amenity"="restaurant"](area.searchArea);
          node["leisure"](area.searchArea);
        );
        out body;
        """
        result = self.overpass_api.get(query)
        
        return {
            'name': location,
            'coordinates': {
                'lat': location_info.latitude,
                'lon': location_info.longitude
            },
            'places_of_interest': result.get('elements', [])
        }

@st.cache_resource
def get_cached_pipeline(model_id: str = "microsoft/phi-2"):
    """Initialize and cache the model pipeline"""
    try:
        return pipeline("text-generation", model=model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error initializing model pipeline: {e}")
        return None

class ItineraryPlannerTool(BaseTravelTool):
    def __init__(self, model_id: str = "microsoft/phi-2"):
        super().__init__()
        self.model_id = model_id
        self.pipe = get_cached_pipeline(model_id)

    async def execute(self, location: str, duration: int, preferences: Dict) -> List[Dict]:
        """
        Create an itinerary using AI model.
        """
        if preferences.get("prompt"):
            base_prompt = preferences["prompt"]
            context = preferences.get("context", "")
            
            # Create a more structured system prompt with example
            system_prompt = """You are a travel expert AI assistant. Generate exactly 3 travel suggestions in this JSON format:

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
]

IMPORTANT RULES:
1. Return EXACTLY 3 destinations in a JSON array
2. Follow the EXACT format shown above
3. Use SPECIFIC numbers, names, and prices
4. Duration MUST be a specific number
5. Include DETAILED activities with locations
6. Consider dietary restrictions and accessibility
7. Stay within the specified budget range
8. NO placeholder or generic content

Your response must be a valid JSON array containing exactly 3 complete destination objects."""

            # Combine system prompt with user prompt
            prompt = f"{system_prompt}\n\nUser Request:\n{base_prompt}"

        try:
            if not self.pipe:
                raise RuntimeError("Pipeline is not initialized")

            print("\n=== Generating Response with Model ===")
            
            # First try with standard parameters
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.pipe(
                    prompt,
                    max_length=2048,
                    max_new_tokens=1500,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    num_return_sequences=1,
                    truncation=True,
                    pad_token_id=self.pipe.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    length_penalty=1.0
                )[0]["generated_text"]
            )

            # Clean up the response
            if result.startswith(prompt):
                result = result[len(prompt):].strip()

            print("\n=== Raw Model Output ===")
            print(result)

            # If no output or invalid, try again with different parameters
            if not result or result.isspace():
                print("\n=== Retrying with adjusted parameters ===")
                result = await loop.run_in_executor(
                    None,
                    lambda: self.pipe(
                        prompt,
                        max_length=1024,  # Reduced length
                        max_new_tokens=800,  # Reduced tokens
                        do_sample=True,
                        temperature=0.8,  # Slightly higher temperature
                        top_p=0.95,
                        top_k=100,  # Increased top_k
                        num_return_sequences=1,
                        truncation=True,
                        pad_token_id=self.pipe.tokenizer.eos_token_id,
                        repetition_penalty=1.1  # Reduced repetition penalty
                    )[0]["generated_text"]
                )
                
                if result.startswith(prompt):
                    result = result[len(prompt):].strip()
                
                print("\n=== Second Attempt Output ===")
                print(result)

            # If still no valid output, use fallback
            if not result or result.isspace():
                print("\n=== Using fallback suggestions ===")
                return self._create_fallback_suggestions()

            # Try to extract and parse JSON
            try:
                # Find the first [ and last ]
                start_idx = result.find('[')
                end_idx = result.rfind(']')
                
                if start_idx == -1 or end_idx == -1:
                    print("\n=== No JSON array found, creating from text ===")
                    # Try to create JSON from the text response
                    lines = result.split('\n')
                    suggestions = []
                    current_suggestion = {}
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith('"destination"') or line.startswith('destination'):
                            if current_suggestion:
                                suggestions.append(current_suggestion)
                                current_suggestion = {}
                            current_suggestion['destination'] = line.split(':')[-1].strip().strip('"').strip()
                        elif ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip().strip('"')
                            value = value.strip().strip(',').strip().strip('"')
                            if key in ['activities', 'accommodation_suggestions', 'transportation', 'local_tips']:
                                value = [v.strip().strip('"') for v in value.strip('[]').split(',')]
                            current_suggestion[key] = value
                    
                    if current_suggestion:
                        suggestions.append(current_suggestion)
                    
                    if suggestions:
                        return self._validate_suggestions(suggestions)
                    raise ValueError("Could not parse suggestions from text")
                
                json_str = result[start_idx:end_idx + 1]
                
                # Clean up common formatting issues
                json_str = json_str.replace('\n', ' ')
                json_str = json_str.replace('```', '')
                json_str = json_str.replace('json', '')
                json_str = json_str.replace('JSON', '')
                
                # Try to parse the JSON
                parsed_data = json.loads(json_str)
                return self._validate_suggestions(parsed_data if isinstance(parsed_data, list) else [parsed_data])
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"\n=== Error parsing response: {str(e)} ===")
                return self._create_fallback_suggestions()
                
        except Exception as e:
            print(f"\nError in ItineraryPlannerTool: {str(e)}")
            return self._create_fallback_suggestions()
    
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
        
        # Ensure exactly 3 suggestions
        while len(validated_data) < 3:
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
        
        return validated_data[:3]

    def _create_fallback_suggestions(self) -> List[Dict]:
        """Create three fallback suggestions"""
        suggestions = []
        for i in range(3):
            suggestions.append({
                "destination": f"Suggested Destination {i + 1}",
                "description": "A carefully selected destination matching your preferences.",
                "best_time_to_visit": "Please check seasonal information",
                "estimated_budget": "Within your specified budget range",
                "duration": "5",  # Use a specific number
                "activities": [
                    "Local cultural experiences",
                    "Traditional food tasting",
                    "Historical site visits",
                    "Nature exploration",
                    "Local market tours"
                ],
                "accommodation_suggestions": [
                    "Comfortable hotels in city center",
                    "Local guesthouses with good reviews",
                    "Boutique hotels with local charm"
                ],
                "transportation": [
                    "Efficient public transportation",
                    "Walking tours in historic areas"
                ],
                "local_tips": [
                    "Learn basic local phrases",
                    "Respect local customs and traditions",
                    "Try local specialties at recommended restaurants"
                ],
                "weather_info": "Research current weather patterns",
                "safety_info": "Follow standard travel safety guidelines"
            })
        return suggestions