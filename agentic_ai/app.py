import streamlit as st
import asyncio
from datetime import datetime, timedelta
import nest_asyncio
from agents.travel_agent import TravelPreferences, ProcessedInput, TravelRequest
from tools.travel_utils import TravelUtils
from tools.travel_tools import ItineraryPlannerTool, WeatherTool, LocationInfoTool, HotelSearchTool, FlightSearchTool
from mcp_server import MCPServer, register_tools
import os
from dotenv import load_dotenv
from typing import Dict, Any, List
import httpx
import json
import logging
import traceback

# Load environment variables
load_dotenv()

# Define constants
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8000")  # Default to localhost if not set
print(f"\n=== Using Agent URL: {AGENT_URL} ===")

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

# Enable nested asyncio for Streamlit
nest_asyncio.apply()

# Initialize the MCP server and tools
@st.cache_resource(show_spinner="Loading AI Travel Planner...")
def initialize_mcp_server():
    """Initialize and cache the MCP server with all tools"""
    try:
        # Get API configuration
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        rapidapi_key = os.getenv('RAPID_API_KEY')
        site_url = os.getenv('SITE_URL', 'http://localhost:8501')
        site_name = os.getenv('SITE_NAME', 'AI Travel Planner')
        
        if not openrouter_api_key:
            st.error("OpenRouter API key not found. Please set OPENROUTER_API_KEY in your environment variables.")
            st.stop()
        
        if not rapidapi_key:
            st.warning("RapidAPI key not found. Using simulated flight and hotel data.")
        
        # Initialize the planner tool
        planner_tool = ItineraryPlannerTool(
            openrouter_api_key=openrouter_api_key,
            site_url=site_url,
            site_name=site_name
        )
        
        # Initialize travel utils
        travel_utils = TravelUtils(rapidapi_key=rapidapi_key)
        
        # Initialize MCP server
        mcp_server = MCPServer()
        
        # Register tools and set up agent
        tools = register_tools(mcp_server, travel_utils, planner_tool)
        mcp_server.setup_agent(tools)
        
        # Start MCP server in background
        import threading
        server_thread = threading.Thread(target=mcp_server.run, daemon=True)
        server_thread.start()
        
        return mcp_server, travel_utils, planner_tool
    
    except Exception as e:
        st.error(f"Error initializing AI Travel Planner: {str(e)}")
        st.stop()

# Get or create the MCP server and tools
mcp_server, travel_utils, planner_tool = initialize_mcp_server()

# Title and description
st.title("🌍 AI Travel Planner")
st.markdown("""
This intelligent travel planner helps you create personalized travel itineraries and get destination suggestions 
based on your preferences. You can either get travel suggestions or create a detailed itinerary for a specific destination.
""")

# Sidebar for mode selection
mode = st.sidebar.radio(
    "Choose Planning Mode",
    ["Get Travel Suggestions", "Create Detailed Itinerary"]
)

# Common preferences input
with st.sidebar:
    st.subheader("Your Travel Preferences")
    budget_range = st.selectbox(
        "Budget Range",
        ["Budget", "Moderate", "Luxury"]
    )
    
    travel_style = st.selectbox(
        "Travel Style",
        list(TravelUtils.get_travel_style_descriptions().keys())
    )
    
    interests = st.multiselect(
        "Interests",
        ["Culture", "Nature", "Food", "Adventure", "Shopping", "History", "Art", "Nightlife"],
        default=["Culture", "Food"]
    )
    
    group_size = st.number_input("Number of Travelers", min_value=1, value=2)
    
    language = st.selectbox("Preferred Language", ["English", "Spanish", "French", "German", "Japanese"])
    
    dietary_restrictions = st.multiselect(
        "Dietary Restrictions",
        ["None", "Vegetarian", "Vegan", "Halal", "Kosher", "Gluten-free"],
        default=["None"]
    )
    
    accommodation_type = st.selectbox(
        "Preferred Accommodation",
        ["Hotel", "Hostel", "Resort", "Apartment", "Boutique Hotel"]
    )

# Create preferences object
preferences = TravelPreferences(
    budget_range=budget_range,
    travel_style=travel_style,
    interests=interests,
    group_size=group_size,
    language_preference=language.lower(),
    dietary_restrictions=[r for r in dietary_restrictions if r != "None"],
    accommodation_type=accommodation_type
)

def extract_travel_entities(text: str) -> Dict[str, Any]:
    """Extract travel-related entities from text"""
    # Simple keyword-based extraction
    travel_keywords = {
        'duration': [r'(\d+)\s*(day|days|week|weeks|month|months)', r'for\s+(\d+)\s*(day|days|week|weeks|month|months)'],
        'destinations': ['city', 'country', 'beach', 'mountain', 'hotel', 'resort'],
        'activities': ['hiking', 'sightseeing', 'museum', 'restaurant', 'shopping', 'adventure', 'cultural', 'food'],
        'budget_terms': ['budget', 'cheap', 'expensive', 'luxury', 'affordable', 'cost']
    }
    
    entities = {}
    text_lower = text.lower()
    
    # Extract duration using regex
    import re
    for pattern in travel_keywords['duration']:
        match = re.search(pattern, text_lower)
        if match:
            number = int(match.group(1))
            unit = match.group(2)
            if 'week' in unit:
                number *= 7
            elif 'month' in unit:
                number *= 30
            entities['duration'] = f"{number} days"
            break
    
    # Extract other entities
    for category, keywords in travel_keywords.items():
        if category != 'duration':  # Skip duration as it's handled above
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                entities[category] = found_keywords
    
    return entities

if mode == "Get Travel Suggestions":
    st.header("🔍 Get Travel Suggestions")
    
    # Input for travel preferences
    travel_input = st.text_area(
        "Describe your ideal trip",
        "I want to travel for about a week, interested in cultural experiences and good food."
    )
    
    if st.button("Get Suggestions"):
        with st.spinner("Generating travel suggestions..."):
            print("\n=== Starting travel suggestion process ===")
            # Create context with preferences
            context = {
                "preferences": preferences.model_dump(exclude_none=True),
                "mode": "suggestions"
            }
            
            # Define async function to make the request
            async def get_suggestions():
                """Get travel suggestions from the agent."""
                try:
                    print("\n=== Starting suggestion generation ===")
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"{AGENT_URL}/agent/execute",
                            json={"query": travel_input, "context": context},
                            timeout=30.0
                        )
                        print(f"\n=== Agent response received: {response.status_code} ===")
                        
                        if response.status_code == 200:
                            result = response.json()
                            if isinstance(result, dict) and "result" in result:
                                suggestions = result["result"].get("output", "")
                                print(f"\n=== Raw suggestions from agent: ===\n{suggestions}")
                                
                                # Convert the suggestions into a structured format
                                suggestions_list = []
                                if isinstance(suggestions, str):
                                    # Parse the text response into structured suggestions
                                    import re
                                    # Look for bullet points or numbered items
                                    suggestion_items = re.split(r'\n\s*[\*\•\-]\s*|\n\d+\.\s+', suggestions)
                                    suggestion_items = [s.strip() for s in suggestion_items if s.strip()]
                                    
                                    for item in suggestion_items[:2]:  # Limit to 2 suggestions
                                        if item:
                                            destination = item.split(" for ")[0] if " for " in item else item
                                            description = item.split(" for ")[1] if " for " in item else ""
                                            
                                            print(f"\n=== Processing suggestion for {destination} ===")
                                            
                                            # Create suggestion with additional information
                                            suggestion = {
                                                "destination": destination.replace('*', '').strip(),
                                                "description": description,
                                                "best_time_to_visit": await get_best_time(destination),
                                                "estimated_budget": await get_estimated_budget(destination),
                                                "duration": "7",  # Default to a week as per user request
                                                "weather": await get_weather(destination),
                                                "local_tips": await get_local_tips(destination),
                                                "hotels": await get_hotels(destination),
                                                "flights": await get_flights(destination)
                                            }
                                            suggestions_list.append(suggestion)
                                            print(f"\n=== Completed processing for {destination} ===")
                                    
                                return suggestions_list
                            return []
                        else:
                            st.error(f"Error from agent: {response.text}")
                            return []
                except Exception as e:
                    print(f"\n=== Error in get_suggestions: {str(e)} ===")
                    print(f"Error details: {traceback.format_exc()}")
                    st.error(f"An error occurred while generating suggestions: {str(e)}")
                    return []

            async def get_weather(destination: str) -> Dict:
                """Get weather information for the destination"""
                try:
                    print(f"\n=== Getting weather for {destination} ===")
                    weather_tool = WeatherTool()
                    weather_info = await weather_tool.execute(destination, datetime.now())
                    print(f"Weather info: {weather_info}")
                    return weather_info
                except Exception as e:
                    print(f"Error getting weather: {str(e)}")
                    return {"error": str(e)}

            async def get_local_tips(destination: str) -> List[str]:
                """Get local tips for the destination"""
                try:
                    print(f"\n=== Getting local tips for {destination} ===")
                    location_tool = LocationInfoTool()
                    location_info = await location_tool.execute(destination)
                    print(f"Local tips: {location_info.get('tips', [])}")
                    return location_info.get("tips", [])
                except Exception as e:
                    print(f"Error getting local tips: {str(e)}")
                    return []

            async def get_hotels(destination: str) -> List[Dict]:
                """Get hotel suggestions for the destination"""
                try:
                    print(f"\n=== Getting hotels for {destination} ===")
                    hotel_tool = HotelSearchTool(api_key=os.getenv("RAPIDAPI_KEY"))
                    check_in = datetime.now()
                    check_out = datetime.now()  # Add 7 days in production
                    hotels = await hotel_tool.execute(destination, check_in, check_out)
                    print(f"Found {len(hotels)} hotels")
                    return hotels[:3]  # Limit to top 3 hotels
                except Exception as e:
                    print(f"Error getting hotels: {str(e)}")
                    return []

            async def get_flights(destination: str) -> List[Dict]:
                """Get flight suggestions for the destination"""
                try:
                    print(f"\n=== Getting flights for {destination} ===")
                    flight_tool = FlightSearchTool(api_key=os.getenv("RAPIDAPI_KEY"))
                    origin = "New York"  # Default origin
                    flights = await flight_tool.execute(origin, destination, datetime.now())
                    print(f"Found {len(flights)} flights")
                    return flights[:3]  # Limit to top 3 flights
                except Exception as e:
                    print(f"Error getting flights: {str(e)}")
                    return []

            async def get_best_time(destination: str) -> str:
                """Get best time to visit using ItineraryPlanner"""
                try:
                    print(f"\n=== Getting best time to visit for {destination} ===")
                    planner = ItineraryPlannerTool(
                        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                        site_url=os.getenv("SITE_URL", "http://localhost:8501"),
                        site_name=os.getenv("SITE_NAME", "AI Travel Planner")
                    )
                    suggestions = await planner.execute(destination, 7, {"focus": "best time"})
                    best_time = suggestions[0].get("best_time_to_visit", "Contact travel agent for details") if suggestions and len(suggestions) > 0 else "Contact travel agent for details"
                    print(f"Best time to visit: {best_time}")
                    return best_time
                except Exception as e:
                    print(f"Error getting best time: {str(e)}")
                    return "Contact travel agent for details"

            async def get_estimated_budget(destination: str) -> str:
                """Get estimated budget using ItineraryPlanner"""
                try:
                    print(f"\n=== Getting estimated budget for {destination} ===")
                    planner = ItineraryPlannerTool(
                        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                        site_url=os.getenv("SITE_URL", "http://localhost:8501"),
                        site_name=os.getenv("SITE_NAME", "AI Travel Planner")
                    )
                    suggestions = await planner.execute(destination, 7, {"focus": "budget"})
                    budget = suggestions[0].get("estimated_budget", "Varies by season") if suggestions and len(suggestions) > 0 else "Varies by season"
                    print(f"Estimated budget: {budget}")
                    return budget
                except Exception as e:
                    print(f"Error getting estimated budget: {str(e)}")
                    return "Varies by season"

            # Run the async function
            suggestions = asyncio.run(get_suggestions())
            print(f"\n=== Got {len(suggestions)} suggestions ===")
            
            # Display suggestions
            if suggestions:
                st.subheader("Travel Suggestions")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"\n=== Displaying suggestion {i} ===")
                    print(f"Type: {type(suggestion)}")
                    print(f"Content: {json.dumps(suggestion, indent=2)}")
                    
                    try:
                        st.write(f"## Suggestion {i}: {suggestion['destination']}")
                        
                        # Basic Information
                        st.write(f"**Description:** {suggestion['description']}")
                        
                        # Create three columns for key info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Best Time:** {suggestion['best_time_to_visit']}")
                        with col2:
                            st.write(f"**Budget:** {suggestion['estimated_budget']}")
                        with col3:
                            st.write(f"**Duration:** {suggestion['duration']} days")
                        
                        # Daily Itinerary
                        st.subheader("�� Suggested Daily Itinerary")
                        days = {
                            1: {
                                "morning": "Explore iconic landmarks and main attractions",
                                "afternoon": "Visit local museums and cultural sites",
                                "evening": "Dinner at a local restaurant",
                                "activities": ["City orientation tour", "Museum visits", "Local dining"]
                            },
                            2: {
                                "morning": "Local market and shopping areas",
                                "afternoon": "Park or nature area visit",
                                "evening": "Entertainment district exploration",
                                "activities": ["Shopping", "Nature walks", "Nightlife"]
                            },
                            3: {
                                "morning": "Historical district tour",
                                "afternoon": "Art galleries or cultural centers",
                                "evening": "Cultural performance or show",
                                "activities": ["Historical sites", "Art appreciation", "Cultural shows"]
                            },
                            4: {
                                "morning": "Local food tour or cooking class",
                                "afternoon": "Neighborhood exploration",
                                "evening": "Sunset viewpoint visit",
                                "activities": ["Food experiences", "Local life", "Photography"]
                            },
                            5: {
                                "morning": "Adventure or outdoor activity",
                                "afternoon": "Relaxation time",
                                "evening": "Local entertainment",
                                "activities": ["Active pursuits", "Spa/relaxation", "Entertainment"]
                            },
                            6: {
                                "morning": "Special interest activities",
                                "afternoon": "Shopping for souvenirs",
                                "evening": "Farewell dinner",
                                "activities": ["Personal interests", "Shopping", "Fine dining"]
                            },
                            7: {
                                "morning": "Final sightseeing",
                                "afternoon": "Last-minute activities",
                                "evening": "Departure preparation",
                                "activities": ["Last visits", "Packing", "Travel"]
                            }
                        }
                        
                        itinerary_tab = st.tabs([f"Day {day}" for day in range(1, 8)])
                        for day, tab in enumerate(itinerary_tab, 1):
                            with tab:
                                schedule = days[day]
                                st.write("**Morning:** " + schedule["morning"])
                                st.write("**Afternoon:** " + schedule["afternoon"])
                                st.write("**Evening:** " + schedule["evening"])
                                st.write("**Suggested Activities:**")
                                for activity in schedule["activities"]:
                                    st.write(f"- {activity}")
                        
                        # Weather Information with fallback
                        st.subheader("🌤️ Weather Information")
                        weather = suggestion.get('weather', {})
                        if isinstance(weather, dict) and 'error' not in weather:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Temperature", f"{weather.get('temperature', 'N/A')}°C")
                            with col2:
                                st.metric("Humidity", f"{weather.get('humidity', 'N/A')}%")
                            with col3:
                                st.write(f"**Conditions:** {weather.get('description', 'N/A')}")
                        else:
                            st.info("Weather data currently unavailable. Please check local weather services.")
                        
                        # Local Tips with fallback
                        st.subheader("💡 Local Tips")
                        if suggestion.get('local_tips'):
                            for tip in suggestion['local_tips']:
                                st.write(f"- {tip}")
                        else:
                            default_tips = [
                                "Research local customs and etiquette",
                                "Keep important documents secure",
                                "Learn basic local phrases",
                                "Have emergency contacts handy",
                                "Check travel advisories before departure"
                            ]
                            for tip in default_tips:
                                st.write(f"- {tip}")
                        
                        # Accommodation with fallback
                        st.subheader("🏨 Recommended Accommodations")
                        if suggestion.get('hotels'):
                            for hotel in suggestion['hotels']:
                                st.write(f"**{hotel.get('name', 'Hotel')}**")
                                st.write(f"- Price: {hotel.get('price', 'N/A')}")
                                st.write(f"- Rating: {hotel.get('rating', 'N/A')}/10")
                                st.write(f"- Address: {hotel.get('address', 'N/A')}")
                                if hotel.get('amenities'):
                                    st.write("- Amenities:")
                                    for amenity in hotel['amenities']:
                                        st.write(f"  • {amenity}")
                        else:
                            st.info("Search for accommodations on popular booking platforms:")
                            st.write("- Hotels.com")
                            st.write("- Booking.com")
                            st.write("- Airbnb")
                            st.write("- Local hotel websites")
                        
                        # Transportation with fallback
                        st.subheader("🚗 Transportation Options")
                        transport_options = {
                            "Public Transport": ["Subway/Metro", "Buses", "Light Rail"],
                            "Private Options": ["Taxis", "Ride-sharing services", "Car rentals"],
                            "Tour Services": ["Hop-on-hop-off buses", "Guided tours", "Private drivers"]
                        }
                        for category, options in transport_options.items():
                            st.write(f"**{category}:**")
                            for option in options:
                                st.write(f"- {option}")
                        
                        # Packing List
                        st.subheader("🎒 Suggested Packing List")
                        packing_categories = {
                            "Essential Documents": [
                                "Passport/ID",
                                "Travel insurance",
                                "Booking confirmations",
                                "Emergency contacts"
                            ],
                            "Clothing": [
                                "Weather-appropriate attire",
                                "Comfortable walking shoes",
                                "Formal wear for nice restaurants",
                                "Swimming gear if applicable"
                            ],
                            "Electronics": [
                                "Phone and charger",
                                "Camera",
                                "Power adapter",
                                "Portable power bank"
                            ],
                            "Health & Safety": [
                                "Medications",
                                "First aid kit",
                                "Face masks",
                                "Hand sanitizer"
                            ]
                        }
                        packing_tabs = st.tabs(list(packing_categories.keys()))
                        for tab, (category, items) in zip(packing_tabs, packing_categories.items()):
                            with tab:
                                for item in items:
                                    st.write(f"- {item}")
                        
                        # Safety Information
                        st.subheader("🛡️ Safety Information")
                        st.write("**General Safety Tips:**")
                        safety_tips = [
                            "Keep valuables in hotel safe",
                            "Be aware of your surroundings",
                            "Use reputable transportation",
                            "Keep emergency numbers handy",
                            "Follow local health guidelines"
                        ]
                        for tip in safety_tips:
                            st.write(f"- {tip}")
                        
                        # Emergency Contacts
                        st.subheader("☎️ Emergency Contacts")
                        st.write("- Local Emergency: 911 (US)")
                        st.write("- Tourist Police: Check local numbers")
                        st.write("- Nearest Embassy/Consulate: Research before travel")
                        st.write("- Travel Insurance 24/7 Helpline: Add your provider's number")
                        
                        st.divider()  # Add a visual separator between suggestions
                    
                    except Exception as e:
                        st.error(f"Error displaying suggestion {i}: {str(e)}")
                        print(f"Error details: {traceback.format_exc()}")
            else:
                st.error("No suggestions were generated. Please try again with different preferences.")

else:  # Create Detailed Itinerary
    st.header("📝 Create Detailed Itinerary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        destination = st.text_input("Destination", "Paris, France")
        start_date = st.date_input(
            "Start Date",
            datetime.now() + timedelta(days=30)
        )
    
    with col2:
        duration = st.number_input("Duration (days)", min_value=1, max_value=30, value=7)
        origin = st.text_input("Origin City (for flights)", "New York, USA")
    
    if st.button("Create Itinerary"):
        with st.spinner("Creating your personalized itinerary..."):
            try:
                # Create travel request
                end_date = datetime.combine(start_date, datetime.min.time()) + timedelta(days=duration)
                travel_request = TravelRequest(
                    origin=origin,
                    destination=destination,
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=end_date,
                    num_travelers=preferences.group_size,
                    preferences=preferences.__dict__,
                    budget=None
                )
                
                # Create context with request details
                context = {
                    "travel_request": travel_request.__dict__,
                    "mode": "itinerary"
                }
                
                # Use MCP server's agent to create itinerary
                response = asyncio.run(
                    mcp_server.app.post("/agent/execute", json={
                        "query": f"Create a detailed itinerary for a trip from {origin} to {destination}",
                        "context": context
                    })
                )
                
                itinerary = response.json()["result"]["output"]
                
                # Display itinerary
                st.subheader(f"📍 {itinerary.destination}")
                st.write(f"**Duration:** {itinerary.total_days} days")
                st.write(f"**Total Budget:** ${itinerary.total_cost:.2f}")
                
                # Display flights
                if itinerary.flights:
                    with st.expander("✈️ Flight Details"):
                        for flight in itinerary.flights:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**{flight['airline']}** - Flight {flight.get('flight_number', 'N/A')}")
                                st.write(f"From: {flight['departure']} To: {flight['arrival']}")
                            with col2:
                                st.write(f"Date: {flight['date']}")
                                st.write(f"Price: ${flight['price']}")
                                if 'departure_time' in flight:
                                    st.write(f"Departure: {flight['departure_time']}")
                                if 'arrival_time' in flight:
                                    st.write(f"Arrival: {flight['arrival_time']}")
                                if 'duration' in flight:
                                    st.write(f"Duration: {flight['duration']}")
                                if 'stops' in flight:
                                    st.write(f"Stops: {flight['stops']}")
                            st.divider()
                
                # Display hotels
                if itinerary.hotels:
                    print(f"Itinerary hotels: {itinerary.hotels}")
                    with st.expander("🏨 Hotel Options"):
                        for hotel in itinerary.hotels:
                            st.write(f"**{hotel['name']}**")
                            st.write(f"Rating: {hotel['rating']}/10")
                            st.write(f"Price: {hotel['price']}")
                            st.write(f"Address: {hotel['address']}")
                            st.write("Amenities:", ", ".join(hotel['amenities']))
                            st.divider()
                
                # Display daily plans
                st.subheader("📅 Daily Schedule")
                for plan in itinerary.daily_plans:
                    with st.expander(f"Day {plan['day']} - {plan['date']}"):
                        st.write("**Morning:**", plan['morning'])
                        st.write("**Afternoon:**", plan['afternoon'])
                        st.write("**Evening:**", plan['evening'])
                        st.write("**Meals:**", ", ".join(plan['meals']))
                        st.write(f"**Estimated Cost:** {plan['estimated_cost']}")
                
                # Display additional information
                col3, col4 = st.columns(2)
                
                with col3:
                    with st.expander("🎒 Packing List"):
                        for item in itinerary.packing_list:
                            st.write(f"- {item}")
                
                with col4:
                    with st.expander("ℹ️ Important Notes"):
                        for note in itinerary.important_notes:
                            st.write(f"- {note}")
                
                # Display emergency contacts
                with st.expander("🆘 Emergency Contacts"):
                    for contact in itinerary.emergency_contacts:
                        st.write(f"**{contact['service']}:** {contact['number']}")
                
                # Get and display region-specific tips
                tips = TravelUtils.get_travel_tips_by_region(destination)
                with st.expander("💡 Local Tips"):
                    for tip in tips:
                        st.write(f"- {tip}")

            except Exception as e:
                st.error(f"An error occurred while creating the itinerary: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Powered by AI Travel Planner • Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True) 