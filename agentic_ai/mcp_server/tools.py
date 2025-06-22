"""
LangChain tools implementation for the travel planner.
"""

from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional
from tools.travel_tools import ItineraryPlannerTool
from tools.travel_utils import TravelUtils

class FlightSearchInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    origin: str = Field(..., description="Origin city for flight search")
    destination: str = Field(..., description="Destination city for flight search")
    date: str = Field(..., description="Date of travel in YYYY-MM-DD format")
    num_passengers: int = Field(1, description="Number of passengers")

class HotelSearchInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    city: str = Field(..., description="City to search hotels in")
    check_in: str = Field(..., description="Check-in date in YYYY-MM-DD format")
    check_out: str = Field(..., description="Check-out date in YYYY-MM-DD format")
    guests: int = Field(1, description="Number of guests")

class WeatherInfoInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    location: str = Field(..., description="Location to get weather information for")
    date: Optional[str] = Field(None, description="Date to get weather for (optional)")

def create_langchain_tools(travel_utils: TravelUtils, itinerary_planner: ItineraryPlannerTool) -> List[BaseTool]:
    """Create LangChain tools from our existing travel tools."""
    
    class FlightSearchTool(BaseTool):
        name = "flight_search"
        description = "Search for flights between cities"
        args_schema = FlightSearchInput
        
        def _run(self, origin: str, destination: str, date: str, num_passengers: int = 1) -> Dict[str, Any]:
            return travel_utils.search_flights(origin, destination, date, num_passengers)
            
        async def _arun(self, origin: str, destination: str, date: str, num_passengers: int = 1) -> Dict[str, Any]:
            return await travel_utils.search_flights_async(origin, destination, date, num_passengers)

    class HotelSearchTool(BaseTool):
        name = "hotel_search"
        description = "Search for hotels in a city"
        args_schema = HotelSearchInput
        
        def _run(self, city: str, check_in: str, check_out: str, guests: int = 1) -> Dict[str, Any]:
            return travel_utils.search_hotels(city, check_in, check_out, guests)
            
        async def _arun(self, city: str, check_in: str, check_out: str, guests: int = 1) -> Dict[str, Any]:
            return await travel_utils.search_hotels_async(city, check_in, check_out, guests)

    class WeatherInfoTool(BaseTool):
        name = "weather_info"
        description = "Get weather information for a location"
        args_schema = WeatherInfoInput
        
        def _run(self, location: str, date: Optional[str] = None) -> Dict[str, Any]:
            return travel_utils.get_weather_info(location, date)
            
        async def _arun(self, location: str, date: Optional[str] = None) -> Dict[str, Any]:
            return await travel_utils.get_weather_info_async(location, date)

    class ItineraryPlannerLangChainTool(BaseTool):
        name = "create_itinerary"
        description = "Create a detailed travel itinerary"
        
        def _run(self, **kwargs) -> Dict[str, Any]:
            return itinerary_planner.create_itinerary(**kwargs)
            
        async def _arun(self, **kwargs) -> Dict[str, Any]:
            return await itinerary_planner.create_itinerary_async(**kwargs)

    return [
        FlightSearchTool(),
        HotelSearchTool(),
        WeatherInfoTool(),
        ItineraryPlannerLangChainTool(),
    ]

def register_tools(mcp_server, travel_utils: TravelUtils, itinerary_planner: ItineraryPlannerTool):
    """Register all tools with the MCP server."""
    tools = create_langchain_tools(travel_utils, itinerary_planner)
    for tool in tools:
        mcp_server.register_tool(tool.name, tool)
    return tools 