from typing import Dict, List, Optional
from datetime import datetime
import aiohttp
import python_weather
from forex_python.converter import CurrencyRates
from overpass import API
from geopy.geocoders import Nominatim
from abc import ABC, abstractmethod
from transformers import pipeline
import asyncio
import json
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import asyncio

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

class ItineraryPlannerTool(BaseTravelTool):
    def __init__(self, model_id: str = "microsoft/phi-2"):
        super().__init__()
        # self.api_key = hf_api_key
        self.model_id = model_id

        try:
            # `phi-2` is a public model, so token is optional
            self.pipe = pipeline("text-generation", model=self.model_id, trust_remote_code=True)
        except Exception as e:
            print(f"Error initializing model pipeline: {e}")
            self.pipe = None

    async def execute(self, location: str, duration: int, preferences: Dict) -> List[Dict]:
        prompt = f"""Create a {duration}-day travel itinerary for {location}.
Include specific attractions, activities, and timing for each day.

Trip Details:
- Location: {location}
- Duration: {duration} days
- Budget: ${preferences.get('max_price', 'flexible')}
- Number of travelers: {preferences.get('num_travelers', 1)}
- Interests: {', '.join(preferences.get('activities', []))}

Format the response as a daily schedule with morning, afternoon, and evening activities.
"""
        try:
            if not self.pipe:
                raise RuntimeError("Pipeline is not initialized")

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.pipe(prompt, max_length=1024, do_sample=True, temperature=0.7)[0]["generated_text"]
            )

            # Basic response parsing (same as before)
            activities = []
            current_day = 1
            current_activities = []

            for line in result.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if any(line.lower().startswith(x) for x in ['day', 'morning', 'afternoon', 'evening']):
                    if current_activities and current_day <= duration:
                        activities.append({
                            'day': current_day,
                            'activities': current_activities.copy(),
                            'estimated_cost': 100
                        })
                        current_day += 1
                        current_activities = []
                    current_activities = [line]
                else:
                    current_activities.append(line)

            if current_activities and current_day <= duration:
                activities.append({
                    'day': current_day,
                    'activities': current_activities,
                    'estimated_cost': 100
                })

            return activities

        except Exception as e:
            print(f"Error initializing model pipeline: {e}")
            self.pipe = None
            print(f"Error generating itinerary: {str(e)}")
            return [
                {
                    'day': i + 1,
                    'activities': [
                        f"Day {i + 1} in {location}:",
                        "Morning:",
                        "- Visit major attractions and landmarks",
                        "- Guided city tour",
                        "Afternoon:",
                        "- Local cuisine experience",
                        "- Cultural activities",
                        "Evening:",
                        "- Dinner at local restaurant",
                        "- Evening entertainment or relaxation"
                    ],
                    'estimated_cost': 100
                }
                for i in range(duration)
            ]