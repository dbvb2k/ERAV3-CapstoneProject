import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from agentic_ai.agents.travel_agent import TravelAgent, TravelRequest
from agentic_ai.workflows.travel_workflow import TravelWorkflow
from agentic_ai.tools.travel_tools import ItineraryPlannerTool

# Load environment variables
load_dotenv()

# Verify API keys are present
if not os.getenv('RAPID_API_KEY'):
    raise ValueError("RAPID_API_KEY not found in .env file")
# if not os.getenv('HUGGINGFACE_API_KEY'):
#     raise ValueError("HUGGINGFACE_API_KEY not found in .env file")

async def main():
    # Initialize the travel agent with configuration
    config = {
        'rapid_api_key': os.getenv('RAPID_API_KEY'),
        'huggingface_api_key': os.getenv('HUGGINGFACE_API_KEY'),
        'tools': {
            'itinerary_planner': ItineraryPlannerTool(
                model_id="microsoft/phi-2"
            )
        }
    }
    
    travel_agent = TravelAgent(config)
    workflow = TravelWorkflow(travel_agent)
    
    # Example travel request
    travel_request = TravelRequest(
        origin="NYC",  # Airport code for New York
        destination="LON",  # Airport code for London
        start_date=datetime.now() + timedelta(days=30),  # Trip in 30 days
        end_date=datetime.now() + timedelta(days=37),    # 7-day trip
        num_travelers=2,
        preferences={
            "max_price": 1000,
            "preferred_airlines": ["British Airways", "American Airlines"],
            "hotel_stars": 4,
            "activities": ["sightseeing", "museums", "food", "culture", "shopping"],
            "num_travelers": 2  # Added for activity planning
        },
        budget=2000.0
    )
    
    try:
        print("Starting travel planning...")
        print("Searching for flights and generating itinerary...")
        # Execute the travel planning workflow
        itinerary = await workflow.execute_planning_workflow(travel_request)
        
        # Print the itinerary details
        print("\n=== Travel Itinerary ===")
        print(f"Trip to {travel_request.destination} from {travel_request.origin}")
        print(f"Dates: {travel_request.start_date.date()} to {travel_request.end_date.date()}")
        
        if itinerary.flights:
            print("\nFlights:")
            for flight in itinerary.flights:
                print(f"- {flight['airline']} Flight {flight['flight_number']}")
                print(f"  Date: {flight['date']}")
                print(f"  Price: ${flight['price']}")
        else:
            print("\nNo flights found for the specified dates.")
        
        if itinerary.hotels:
            print("\nHotels:")
            for hotel in itinerary.hotels:
                print(f"- {hotel['name']}")
                print(f"  Address: {hotel['address']}")
                print(f"  Price: {hotel['price']}")
                print(f"  Rating: {hotel['rating']}")
                print(f"  Amenities: {', '.join(hotel['amenities'])}")
        else:
            print("\nNo hotels found for the specified dates.")
        
        if itinerary.activities:
            print("\nActivities:")
            for activity in itinerary.activities:
                print(f"\nDay {activity['day']}:")
                for act in activity['activities']:
                    if not act.startswith('Day'):  # Skip day headers as we're already printing them
                        print(f"  {act}")
        else:
            print("\nNo activities planned.")
        
        print(f"\nTotal Estimated Cost: ${itinerary.total_cost:.2f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 