from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict
import python_weather
import dotenv, os

dotenv.load_dotenv()

class BaseTravelTool(ABC):
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        
    @abstractmethod
    async def execute(self, *args, **kwargs):
        pass

class WeatherTool(BaseTravelTool):
    async def execute(self, location: str, date: datetime) -> Dict:
        """
        Get weather information using python_weather (free).
        """
        try:
            async with python_weather.Client(unit=python_weather.METRIC) as client:
                weather = await client.get(location)
                print(f"Weather: {weather}")
                
                # Get the current weather directly from the Forecast object
                return {
                    'temperature': weather.temperature,
                    'description': weather.description if hasattr(weather, 'description') else 'No description available',
                    'humidity': weather.humidity if hasattr(weather, 'humidity') else 50  # Default if not available
                }
        except Exception as e:
            print(f"Weather API error: {str(e)}")
            return {
                'temperature': 'N/A',
                'description': 'Weather data unavailable',
                'humidity': 'N/A'
            }
        
if __name__ == "__main__":
    import asyncio
    weatherTool = WeatherTool()
    weather = asyncio.run(weatherTool.execute("Amritsar", datetime.now()))
    print(f"Weather: {weather}")