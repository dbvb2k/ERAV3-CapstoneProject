from typing import Dict, List, Optional, Union
from datetime import datetime
from agents.travel_agent import TravelAgent, SmartTravelAgent, TravelRequest, Itinerary, TravelPreferences, ProcessedInput, TravelSuggestion

class TravelWorkflow:
    def __init__(self, travel_agent: Union[TravelAgent, SmartTravelAgent]):
        self.agent = travel_agent
        
    async def execute_planning_workflow(self, travel_request: TravelRequest) -> Itinerary:
        """
        Execute the complete travel planning workflow.
        """
        # Step 1: Validate the travel request
        self._validate_request(travel_request)
        
        # Step 2: Get destination information
        destination_info = await self.agent.get_location_info(travel_request.destination)
        
        # Step 3: Search for flights
        flights = await self.agent.search_flights(
            travel_request.origin,
            travel_request.destination,
            travel_request.start_date
        )
        
        # Step 4: Search for hotels
        hotels = await self.agent.search_hotels(
            travel_request.destination,
            travel_request.start_date,
            travel_request.end_date
        )
        
        # Step 5: Plan activities
        activities = await self.agent.suggest_activities(
            travel_request.destination,
            [travel_request.start_date, travel_request.end_date]
        )
        
        # Step 6: Create itinerary
        itinerary = Itinerary(
            travel_request=travel_request,
            flights=flights,
            hotels=hotels,
            activities=activities,
            total_cost=self._calculate_total_cost(flights, hotels, activities)
        )
        
        return itinerary
    
    async def execute_smart_planning_workflow(self, 
                                           processed_input: ProcessedInput,
                                           preferences: TravelPreferences) -> List[TravelSuggestion]:
        """
        Execute the smart travel planning workflow using AI-powered suggestions.
        """
        if not isinstance(self.agent, SmartTravelAgent):
            raise ValueError("Smart planning workflow requires SmartTravelAgent")
            
        # Get travel suggestions
        suggestions = await self.agent.create_suggestions(processed_input, preferences)
        
        # For each suggestion, create a detailed itinerary
        for suggestion in suggestions:
            try:
                # Extract duration in days from suggestion
                duration = int(suggestion.duration.split()[0])  # Assumes format like "5 days"
                
                # Create detailed itinerary
                itinerary = await self.agent.create_itinerary(
                    suggestion.destination,
                    duration,
                    preferences
                )
                
                # Add itinerary details to suggestion
                suggestion.detailed_itinerary = itinerary
                
            except Exception as e:
                print(f"Error creating itinerary for {suggestion.destination}: {e}")
                suggestion.detailed_itinerary = None
        
        return suggestions
    
    def _validate_request(self, request: TravelRequest):
        """
        Validate the travel request parameters.
        """
        if request.start_date >= request.end_date:
            raise ValueError("Start date must be before end date")
        
        if request.num_travelers < 1:
            raise ValueError("Number of travelers must be at least 1")
        
        if not request.origin or not request.destination:
            raise ValueError("Origin and destination must be specified")
    
    def _calculate_total_cost(self, flights: List[Dict], hotels: List[Dict], activities: List[Dict]) -> float:
        """
        Calculate the total cost of the trip.
        """
        flight_cost = sum(float(str(flight.get('price', '0')).replace('$', '').replace(',', '')) for flight in flights)
        hotel_cost = sum(float(str(hotel.get('price', '0')).replace('$', '').replace(',', '')) for hotel in hotels)
        activity_cost = sum(float(str(activity.get('price', '0')).replace('$', '').replace(',', '')) for activity in activities)
        
        return flight_cost + hotel_cost + activity_cost 