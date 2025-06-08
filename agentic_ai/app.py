import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

import asyncio
from datetime import datetime, timedelta
import nest_asyncio
from agents.travel_agent import SmartTravelAgent, TravelPreferences, ProcessedInput, TravelRequest
from tools.travel_utils import TravelUtils
from workflows.travel_workflow import TravelWorkflow
from tools.travel_tools import ItineraryPlannerTool
import json
import traceback
import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Enable nested asyncio for Streamlit
nest_asyncio.apply()

# Pre-initialize the model pipeline
# pipeline = get_cached_pipeline()

# Initialize the agent and workflow only once using session state
@st.cache_resource(show_spinner="Loading AI Travel Planner...")
def get_agent_and_workflow():
    """Initialize and cache the agent and workflow with all tools"""
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
        
        # Initialize the planner tool with OpenRouter configuration
        planner_tool = ItineraryPlannerTool(
            openrouter_api_key=openrouter_api_key,
            site_url=site_url,
            site_name=site_name
        )
        
        # Create agent configuration
        agent_config = {
            'model_type': 'openrouter',
            'tools': {
                'itinerary_planner': planner_tool
            },
            'market': 'US',
            'language': 'en-US',
            'currency': 'USD',
            'rapidapi_key': rapidapi_key
        }
        
        # Initialize the agent with the configured tool
        agent = SmartTravelAgent(config=agent_config)
        workflow = TravelWorkflow(agent)
        
        return agent, workflow
    except Exception as e:
        st.error(f"Error initializing AI Travel Planner: {str(e)}")
        st.stop()

# Get or create the agent and workflow
agent, workflow = get_agent_and_workflow()

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
            try:
                # Extract entities from input
                extracted_entities = extract_travel_entities(travel_input)
                
                # Create processed input
                processed_input = ProcessedInput(
                    content=travel_input,
                    extracted_entities=extracted_entities
                )
                
                # Get suggestions
                suggestions = asyncio.run(
                    workflow.execute_smart_planning_workflow(processed_input, preferences)
                )
                
                # Debug print
                print(f"\nNumber of suggestions received: {len(suggestions) if suggestions else 0}")
                
                # Display suggestions
                if suggestions:
                    st.success(f"Found {len(suggestions)} travel suggestions for you!")
                    
                    for i, suggestion in enumerate(suggestions, 1):
                        # Debug print
                        print(f"\nDisplaying suggestion {i}:")
                        print(f"Type: {type(suggestion)}")
                        print(f"Content: {suggestion.__dict__}")
                        
                        with st.expander(f"Suggestion {i}: {suggestion.destination}"):
                            st.write(f"**Description:** {suggestion.description}")
                            st.write(f"**Best Time to Visit:** {suggestion.best_time_to_visit}")
                            st.write(f"**Estimated Budget:** {suggestion.estimated_budget}")
                            st.write(f"**Recommended Duration:** {suggestion.duration} days")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Top Activities:**")
                                activities = suggestion.activities if isinstance(suggestion.activities, list) else []
                                for activity in activities:
                                    st.write(f"- {activity}")
                            
                            with col2:
                                st.write("**Accommodation Options:**")
                                accommodations = suggestion.accommodation_suggestions if isinstance(suggestion.accommodation_suggestions, list) else []
                                for acc in accommodations:
                                    st.write(f"- {acc}")
                            
                            st.write("**Transportation Options:**")
                            transportation = suggestion.transportation if isinstance(suggestion.transportation, list) else []
                            for transport in transportation:
                                st.write(f"- {transport}")
                            
                            # Add flight information display
                            st.write("**Flight Options:**")
                            if isinstance(suggestion.flights, str):
                                st.write(f"- {suggestion.flights}")
                            elif isinstance(suggestion.flights, list):
                                for flight in suggestion.flights:
                                    st.write(f"- {flight['airline']} Flight {flight['flight_number']}: {flight['departure']} → {flight['arrival']}")
                                    st.write(f"  Date: {flight['date']}, Price: ${flight['price']}")
                            
                            st.write("**Local Tips:**")
                            tips = suggestion.local_tips if isinstance(suggestion.local_tips, list) else []
                            for tip in tips:
                                st.write(f"- {tip}")
                            
                            col3, col4 = st.columns(2)
                            with col3:
                                st.write(f"**Weather Info:** {suggestion.weather_info}")
                            with col4:
                                st.write(f"**Safety Info:** {suggestion.safety_info}")
                else:
                    st.error("No suggestions were generated. Please try again with different preferences.")
                    
            except Exception as e:
                st.error(f"An error occurred while generating suggestions: {str(e)}")
                print(f"Error details: {traceback.format_exc()}")

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
            
            # Get itinerary
            itinerary = asyncio.run(workflow.execute_planning_workflow(travel_request))
            
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Powered by AI Travel Planner • Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True) 