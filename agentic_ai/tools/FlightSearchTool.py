import aiohttp
from datetime import datetime
from typing import Dict, List
import dotenv, os
from abc import ABC, abstractmethod

dotenv.load_dotenv()
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
            print("No API key provided")
            return self._get_default_flights(origin, destination, date)
            
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'X-RapidAPI-Key': self.api_key,
                    'X-RapidAPI-Host': 'fly-scraper.p.rapidapi.com'
                }
                # sky_url = "https://fly-scraper.p.rapidapi.com/airports"
                # sky_params = {
                #     'location': origin
                # }
                # async with session.get(sky_url, headers=headers, params=sky_params) as response:
                #     if response.status != 200:
                #         print(f"Error: {response.status}")
                #         print(f"Error: {response.text}")
                # Convert city names to SkyID format
                origin_sky_id = await self._get_sky_id(origin)
                print(f"Origin Sky ID: {origin_sky_id}")
                destination_sky_id = await self._get_sky_id(destination)
                print(f"Destination Sky ID: {destination_sky_id}")
                url = "https://fly-scraper.p.rapidapi.com/flights/search-one-way"
                params = {
                    'originSkyId': origin_sky_id,
                    'destinationSkyId': destination_sky_id,
                    'date': date.strftime('%Y-%m-%d')
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        print(f"Error: {response.status}")
                        print(f"Error: {response.text}")
                        # return self._get_default_flights(origin, destination, date)
                    
                    data = await response.json()
                    flights = []
                    
                    if data.get('data', {}).get('flights'):
                        for flight_data in data['data']['flights']:
                            if isinstance(flight_data, dict):  # Ensure flight_data is a valid dictionary
                                flight = {
                                    'date': date.strftime('%Y-%m-%d'),
                                    'price': flight_data.get('price', {}).get('amount', 'Price Not Available'),
                                    'airline': flight_data.get('airline', {}).get('name', 'Airline Not Available'),
                                    'flight_number': str(flight_data.get('flightNumber', '')),
                                    'departure': origin,
                                    'arrival': destination,
                                    'departure_time': flight_data.get('departureTime', 'Time Not Available'),
                                    'arrival_time': flight_data.get('arrivalTime', 'Time Not Available'),
                                    'duration': flight_data.get('duration', 'Duration Not Available'),
                                    'stops': flight_data.get('stops', 0)
                                }
                                flights.append(flight)
                    print(f"Flights: {flights}")
                    return flights if flights else self._get_default_flights(origin, destination, date)
                    
        except Exception as e:
            print(f"Flight API error: {str(e)}")
            return self._get_default_flights(origin, destination, date)
        
    async def _get_sky_id(self, city: str) -> str:
        """Get SkyID for a city."""
        headers = {
            'X-RapidAPI-Key': self.api_key,
            'X-RapidAPI-Host': 'fly-scraper.p.rapidapi.com'
        }
        url = "https://fly-scraper.p.rapidapi.com/airports"
        params = {
            'location': city
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        print(f"Error: {response.status}")
                        return self._convert_to_sky_id(city)  # fallback to simple conversion
                    
                    data = await response.json()
                    print(f"API Response: {data}")
                    
                    if data.get('status') and data.get('data'):
                        # Get the first airport's skyId
                        first_airport = data['data'][0]
                        if first_airport.get('skyId'):
                            return first_airport['skyId']
                    
                    # If no valid airport code found, fall back to simple conversion
                    return self._convert_to_sky_id(city)
        except Exception as e:
            print(f"Error getting sky id: {str(e)}")
            return self._convert_to_sky_id(city)  # fallback to simple conversion

    def _get_default_flights(self, origin: str, destination: str, date: str) -> List[Dict]:
        """Get default flight suggestions when API fails."""
        return [
            {
                'airline': 'Major Airline 1',
                'flight_number': 'MA101',
                'departure': origin,
                'arrival': destination,
                'departure_time': '09:00 AM',
                'arrival_time': '11:00 AM',
                'price': '$300-400',
                'duration': '2h 00m',
                'stops': 0
            },
            {
                'airline': 'Major Airline 2',
                'flight_number': 'MA202',
                'departure': origin,
                'arrival': destination,
                'departure_time': '02:00 PM',
                'arrival_time': '04:00 PM',
                'price': '$250-350',
                'duration': '2h 00m',
                'stops': 0
            },
            {
                'airline': 'Budget Airline',
                'flight_number': 'BA303',
                'departure': origin,
                'arrival': destination,
                'departure_time': '07:00 PM',
                'arrival_time': '09:00 PM',
                'price': '$200-300',
                'duration': '2h 00m',
                'stops': 1
            }
        ]
    
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

if __name__ == "__main__":
    import asyncio
    api_key = os.getenv("RAPID_API_KEY")
    print(f"API key: {api_key}")
    flightSearchTool = FlightSearchTool(api_key)
    flights = asyncio.run(flightSearchTool.execute("Mumbai", "London", datetime.now()))
    print(flights)