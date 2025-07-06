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
import requests
from PIL import Image

# Load environment variables
load_dotenv()

# Define constants
REQUEST_TIMEOUT = 300
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

# Initialize session state for conversation management
if 'conversation_session_id' not in st.session_state:
    st.session_state.conversation_session_id = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "Get Travel Suggestions"
# Initialize session state for image upload and location
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'location_info' not in st.session_state:
    st.session_state.location_info = None

# Initialize the MCP server and tools
@st.cache_resource(show_spinner="Loading AI Travel Planner...")
def initialize_mcp_server():
    """Initialize the MCP server and tools."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get configuration from environment variables
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        rapidapi_key = os.getenv('RAPID_API_KEY')
        llama_api_url = os.getenv('LLAMA_API_URL', 'http://localhost:8080')
        site_url = os.getenv('SITE_URL', 'http://localhost:8501')
        site_name = os.getenv('SITE_NAME', 'AI Travel Planner')
        
        # Check for at least one LLM service
        if not openrouter_api_key and not llama_api_url:
            st.error("No LLM service configured. Please set either OPENROUTER_API_KEY or ensure LLAMA_API_URL is accessible.")
            st.stop()
        
        if not rapidapi_key:
            st.warning("RapidAPI key not found. Flight and hotel data will be unavailable.")
        
        # Initialize the planner tool with fallback support
        planner_tool = ItineraryPlannerTool(
            api_base_url=llama_api_url
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
based on your preferences. You can ask follow-up questions to refine your travel plans!
""")

# Add conversation management buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🔄 New Conversation"):
        st.session_state.conversation_session_id = None
        st.session_state.conversation_history = []
        st.session_state.uploaded_image = None
        st.session_state.location_info = None
        st.rerun()
with col2:
    if st.button("🗑️ Clear History"):
        st.session_state.conversation_history = []
        st.rerun()
with col3:
    if st.session_state.conversation_session_id:
        st.info(f"Session: {st.session_state.conversation_session_id[:8]}...")

# Add image upload section
st.markdown("---")
st.subheader("📸 Upload Location Image (Optional)")

# Define GeoCLIP API URL
GEOCLIP_API_URL = os.getenv('GEOCLIP_API_URL', 'http://localhost:8090/predict')

# Create two columns for image upload and display
img_col1, img_col2 = st.columns([1, 1])

with img_col1:
    uploaded_file = st.file_uploader(
        "Choose an image to identify location", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a location to automatically identify the destination"
    )
    
    if uploaded_file:
        # Store the uploaded image
        st.session_state.uploaded_image = uploaded_file
        
        # Process the image with GeoCLIP API
        if st.button("🔍 Identify Location"):
            try:
                with st.spinner("Identifying location from image..."):
                    # Prepare the file for API request
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "image/jpeg"
                        )
                    }
                    
                    # Make request to GeoCLIP API
                    response = requests.post(GEOCLIP_API_URL, files=files, timeout=REQUEST_TIMEOUT)
                    
                    if response.status_code == 200:
                        predictions = response.json()
                        if predictions and len(predictions) > 0:
                            st.session_state.location_info = predictions[0]
                            st.success(f"✅ Location identified: {st.session_state.location_info['location_name']}")
                            
                            # Display location details
                            with st.expander("📍 Location Details"):
                                st.write(f"**Location:** {st.session_state.location_info['location_name']}")
                                st.write(f"**Coordinates:** {st.session_state.location_info['latitude']}, {st.session_state.location_info['longitude']}")
                                if 'confidence' in st.session_state.location_info:
                                    st.write(f"**Confidence:** {st.session_state.location_info['confidence']:.2f}")
                        else:
                            st.warning("⚠️ No location found in the image. Please try a different image.")
                    else:
                        st.error(f"❌ Error from GeoCLIP API: {response.text}")
                        
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to GeoCLIP API. Please ensure the service is running.")
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")

with img_col2:
    # Display uploaded image and location info
    if st.session_state.uploaded_image:
        st.write("**Uploaded Image:**")
        image = Image.open(st.session_state.uploaded_image).convert("RGB")
        
        # Resize image to fit 300x300 container while maintaining aspect ratio
        def resize_image_with_aspect_ratio(img, target_size=(300, 200)):
            """Resize image to fit target size while maintaining aspect ratio"""
            # Get original dimensions
            original_width, original_height = img.size
            
            # Calculate aspect ratios
            target_aspect = target_size[0] / target_size[1]
            original_aspect = original_width / original_height
            
            if original_aspect > target_aspect:
                # Image is wider than target, fit to width
                new_width = target_size[0]
                new_height = int(target_size[0] / original_aspect)
            else:
                # Image is taller than target, fit to height
                new_height = target_size[1]
                new_width = int(target_size[1] * original_aspect)
            
            # Resize image
            resized_image = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a new image with target size and white background
            final_image = Image.new('RGB', target_size, (255, 255, 255))
            
            # Calculate position to center the resized image
            x_offset = (target_size[0] - new_width) // 2
            y_offset = (target_size[1] - new_height) // 2
            
            # Paste the resized image onto the background
            final_image.paste(resized_image, (x_offset, y_offset))
            
            return final_image
        
        # Resize the image
        display_image = resize_image_with_aspect_ratio(image, (300, 200))
        
        # Display the image in a container
        st.image(display_image, caption="Uploaded Image", use_container_width=False, width=300)
        
        if st.session_state.location_info:
            st.write("**Identified Location:**")
            st.info(f"📍 {st.session_state.location_info['location_name']}")
            
            # Add a button to use this location
            if st.button("🎯 Use This Location"):
                st.success(f"Location '{st.session_state.location_info['location_name']}' will be used in your travel planning!")
        else:
            st.write("**Status:** No location identified yet")
    else:
        st.write("**No image uploaded**")
        st.info("Upload an image to automatically identify the location for your travel planning.")

# Add export functionality
if st.session_state.conversation_history:
    col4, col5 = st.columns([1, 1])
    with col4:
        if st.button("📄 Export Conversation"):
            # Create export data
            export_data = {
                "session_id": st.session_state.conversation_session_id,
                "export_date": datetime.now().isoformat(),
                "conversation_history": st.session_state.conversation_history,
                "user_preferences": preferences.model_dump(exclude_none=True)
            }
            
            # Create downloadable JSON
            import json
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"travel_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col5:
        if st.button("📋 Copy Summary"):
            # Create a text summary
            summary_lines = ["# Travel Planning Conversation Summary\n"]
            summary_lines.append(f"**Session ID:** {st.session_state.conversation_session_id}\n")
            summary_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for msg in st.session_state.conversation_history:
                role = "User" if msg["role"] == "user" else "AI Assistant"
                summary_lines.append(f"**{role}:** {msg['content']}\n")
            
            summary_text = "\n".join(summary_lines)
            st.text_area("Conversation Summary", summary_text, height=200)

# Display conversation history
if st.session_state.conversation_history:
    st.subheader("💬 Conversation History")
    conversation_container = st.container()
    
    with conversation_container:
        for i, message in enumerate(st.session_state.conversation_history):
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI:** {message['content']}")
            st.divider()

# Sidebar for mode selection
mode = st.sidebar.radio(
    "Choose Planning Mode",
    ["Get Travel Suggestions", "Create Detailed Itinerary"]
)

# Add intelligent mode detection
def detect_request_type(user_input: str) -> str:
    """Detect if the user is asking for suggestions or a specific itinerary"""
    input_lower = user_input.lower()
    
    # Keywords that indicate a specific itinerary request
    itinerary_keywords = [
        'itinerary', 'schedule', 'plan', 'day 1', 'day 2', 'day 3', 'morning', 'afternoon', 'evening',
        'create itinerary', 'make itinerary', 'detailed itinerary', 'travel plan'
    ]
    
    # Keywords that indicate a suggestion request
    suggestion_keywords = [
        'suggest', 'recommend', 'where should', 'what should', 'ideas', 'options', 'alternatives'
    ]
    
    # Check for itinerary keywords
    for keyword in itinerary_keywords:
        if keyword in input_lower:
            return "itinerary"
    
    # Check for suggestion keywords
    for keyword in suggestion_keywords:
        if keyword in input_lower:
            return "suggestions"
    
    # Default to suggestions if unclear
    return "suggestions"

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
            
            # Add location information if available from uploaded image
            if st.session_state.location_info:
                context["location_information"] = {
                    "location_name": st.session_state.location_info['location_name'],
                    "latitude": st.session_state.location_info['latitude'],
                    "longitude": st.session_state.location_info['longitude'],
                    "confidence": st.session_state.location_info.get('confidence', 0.0)
                }
                # Pre-populate the travel input with the identified location
                if not travel_input or travel_input.strip() == "I want to travel for about a week, interested in cultural experiences and good food.":
                    travel_input = f"I want to travel to {st.session_state.location_info['location_name']} for about a week, interested in cultural experiences and good food."
                else:
                    # If user has already entered something, enhance it with location info
                    if st.session_state.location_info['location_name'].lower() not in travel_input.lower():
                        travel_input = f"I want to travel to {st.session_state.location_info['location_name']}. {travel_input}"
            
            # Define async function to make the request
            async def get_suggestions():
                """Get travel suggestions from the agent."""
                try:
                    print("\n=== Starting suggestion generation ===")
                    
                    # Detect if this is actually an itinerary request
                    request_type = detect_request_type(travel_input)
                    print(f"\n=== Detected request type: {request_type} ===")
                    
                    async with httpx.AsyncClient() as client:
                        # Prepare request with conversation state
                        request_data = {
                            "query": travel_input, 
                            "context": context,
                            "session_id": st.session_state.conversation_session_id,
                            "conversation_history": st.session_state.conversation_history
                        }
                        
                        print(f"\n=== Sending request to agent ===")
                        print(f"Query: {travel_input}")
                        print(f"Context keys: {list(context.keys())}")
                        
                        response = await client.post(
                            f"{AGENT_URL}/agent/execute",
                            json=request_data,
                            timeout=60.0  # Increased timeout to 60 seconds
                        )
                        print(f"\n=== Agent response received: {response.status_code} ===")
                        
                        if response.status_code == 200:
                            result = response.json()
                            print(f"\n=== Full result: {result} ===")
                            
                            if isinstance(result, dict) and "result" in result:
                                suggestions = result["result"].get("output", "")
                                print(f"\n=== Raw suggestions from agent: ===\n{suggestions}")
                                
                                # Update session state
                                if "session_id" in result:
                                    st.session_state.conversation_session_id = result["session_id"]
                                if "conversation_history" in result:
                                    st.session_state.conversation_history = result["conversation_history"]
                                
                                # Check if the response contains itinerary content
                                if request_type == "itinerary" or "day 1" in suggestions.lower() or "morning:" in suggestions.lower():
                                    # This is an itinerary response
                                    return {"type": "itinerary", "content": suggestions}
                                else:
                                    # This is a suggestions response
                                    return {"type": "suggestions", "content": suggestions}
                            else:
                                print(f"\n=== Unexpected result structure: {result} ===")
                                return {"type": "error", "content": "Unexpected response structure"}
                        else:
                            error_text = response.text
                            print(f"\n=== Agent error: {response.status_code} - {error_text} ===")
                            st.error(f"Error from agent: {error_text}")
                            return {"type": "error", "content": error_text}
                except httpx.TimeoutException:
                    error_msg = "Request timed out. The server is taking too long to respond. Please try again."
                    print(f"\n=== Timeout error: {error_msg} ===")
                    st.error(error_msg)
                    return {"type": "error", "content": error_msg}
                except httpx.ConnectError:
                    error_msg = "Cannot connect to the travel agent server. Please ensure the MCP server is running."
                    print(f"\n=== Connection error: {error_msg} ===")
                    st.error(error_msg)
                    return {"type": "error", "content": error_msg}
                except Exception as e:
                    print(f"\n=== Error in get_suggestions: {str(e)} ===")
                    print(f"Error details: {traceback.format_exc()}")
                    st.error(f"An error occurred while generating suggestions: {str(e)}")
                    return {"type": "error", "content": str(e)}

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
                    return {"status": "unavailable", "message": "Weather data unavailable"}

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
                    return ["Local tips unavailable"]

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
                    return [{"name": "Hotel data unavailable", "price": "N/A", "rating": "N/A", "address": "N/A", "amenities": ["Data unavailable"]}]

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
                    return [{"airline": "Flight data unavailable", "flight_number": "N/A", "departure": "N/A", "arrival": "N/A", "departure_time": "N/A", "arrival_time": "N/A", "price": "N/A", "duration": "N/A", "stops": "N/A"}]

            async def get_best_time(destination: str) -> str:
                """Get best time to visit using ItineraryPlanner"""
                try:
                    print(f"\n=== Getting best time to visit for {destination} ===")
                    planner = ItineraryPlannerTool(
                        # openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                        # site_url=os.getenv("SITE_URL", "http://localhost:8501"),
                        # site_name=os.getenv("SITE_NAME", "AI Travel Planner")
                        api_base_url=llama_api_url
                    )
                    suggestions = await planner.execute(destination, 7, {"focus": "best time"})
                    best_time = suggestions[0].get("best_time_to_visit", "Contact travel agent for details") if suggestions and len(suggestions) > 0 else "Contact travel agent for details"
                    print(f"Best time to visit: {best_time}")
                    return best_time
                except Exception as e:
                    print(f"Error getting best time: {str(e)}")
                    return "Best time data unavailable"

            async def get_estimated_budget(destination: str) -> str:
                """Get estimated budget using ItineraryPlanner"""
                try:
                    print(f"\n=== Getting estimated budget for {destination} ===")
                    planner = ItineraryPlannerTool(
                        # openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                        # site_url=os.getenv("SITE_URL", "http://localhost:8501"),
                        # site_name=os.getenv("SITE_NAME", "AI Travel Planner")
                        api_base_url=llama_api_url
                    )
                    suggestions = await planner.execute(destination, 7, {"focus": "budget"})
                    budget = suggestions[0].get("estimated_budget", "Varies by season") if suggestions and len(suggestions) > 0 else "Varies by season"
                    print(f"Estimated budget: {budget}")
                    return budget
                except Exception as e:
                    print(f"Error getting estimated budget: {str(e)}")
                    return "Budget data unavailable"

            # Run the async function
            result = asyncio.run(get_suggestions())
            print(f"\n=== Got result: {result} ===")
            
            # Display results based on type
            if result["type"] == "itinerary":
                # Display itinerary response
                st.subheader("📝 Your Travel Itinerary")
                st.markdown(result["content"])
                
                # Add some helpful information
                st.info("💡 **Tip**: This itinerary was generated based on your preferences. You can modify the details or ask for specific changes.")
                
            elif result["type"] == "suggestions":
                # Parse and display suggestions
                suggestions = result["content"]
                
                # First, try to display the raw response to see what we're working with
                st.subheader("Travel Suggestions")
                st.markdown("**Raw Response:**")
                st.markdown(suggestions)
                
                # Convert the suggestions into a structured format
                async def process_suggestions(suggestions_text):
                    """Process suggestions and add additional information"""
                    suggestions_list = []
                    if isinstance(suggestions_text, str):
                        # Try multiple parsing strategies
                        import re
                        
                        # Strategy 1: Look for bullet points or numbered items
                        suggestion_items = re.split(r'\n\s*[\*\•\-]\s*|\n\d+\.\s+', suggestions_text)
                        suggestion_items = [s.strip() for s in suggestion_items if s.strip()]
                        
                        # Strategy 2: If no bullet points found, look for location names
                        if not suggestion_items or len(suggestion_items) < 2:
                            # Look for common location patterns
                            location_patterns = [
                                r'([A-Z][a-z]+(?:[,\s]+[A-Z][a-z]+)*)',
                                r'([A-Z][a-z]+(?:[,\s]+[A-Z][a-z]+)*\s+for\s+.+)',
                                r'Location:\s*([A-Z][a-z]+(?:[,\s]+[A-Z][a-z]+)*)'
                            ]
                            
                            for pattern in location_patterns:
                                matches = re.findall(pattern, suggestions_text)
                                if matches:
                                    suggestion_items = matches[:2]  # Take first 2 matches
                                    break
                        
                        # Strategy 3: If still no items, create a single suggestion from the text
                        if not suggestion_items:
                            # Extract location from context if available
                            location_name = "Unknown Location"
                            if st.session_state.location_info:
                                location_name = st.session_state.location_info['location_name']
                            
                            suggestion_items = [location_name]
                        
                        print(f"\n=== Found {len(suggestion_items)} suggestion items ===")
                        
                        for i, item in enumerate(suggestion_items[:2]):  # Limit to 2 suggestions
                            if item:
                                # Clean up the destination name
                                destination = item.split(" for ")[0] if " for " in item else item
                                description = item.split(" for ")[1] if " for " in item else f"Great destination for travel experiences"
                                
                                # Remove common prefixes and clean up
                                destination = re.sub(r'^[\*\•\-]\s*', '', destination)
                                destination = re.sub(r'^\d+\.\s*', '', destination)
                                destination = destination.strip()
                                
                                print(f"\n=== Processing suggestion {i+1} for {destination} ===")
                                
                                # Create suggestion with additional information
                                suggestion = {
                                    "destination": destination,
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
                
                # Process suggestions
                suggestions_list = asyncio.run(process_suggestions(suggestions))
                
                # Display structured suggestions if available
                if suggestions_list:
                    st.markdown("---")
                    st.subheader("Structured Travel Information")
                    for i, suggestion in enumerate(suggestions_list, 1):
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
                                best_time = suggestion['best_time_to_visit']
                                if best_time == "Best time data unavailable":
                                    st.warning("Best time data unavailable")
                                else:
                                    st.write(f"**Best Time:** {best_time}")
                            with col2:
                                budget = suggestion['estimated_budget']
                                if budget == "Budget data unavailable":
                                    st.warning("Budget data unavailable")
                                else:
                                    st.write(f"**Budget:** {budget}")
                            with col3:
                                st.write(f"**Duration:** {suggestion['duration']} days")
                            
                            # Display additional information in expandable sections
                            with st.expander("🌤️ Weather Information"):
                                if isinstance(suggestion['weather'], dict):
                                    if suggestion['weather'].get('status') == 'unavailable':
                                        st.warning("Weather data unavailable")
                                    else:
                                        st.write(f"**Temperature:** {suggestion['weather'].get('temperature', 'N/A')}°C")
                                        st.write(f"**Conditions:** {suggestion['weather'].get('description', 'N/A')}")
                                        st.write(f"**Humidity:** {suggestion['weather'].get('humidity', 'N/A')}%")
                                else:
                                    st.warning("Weather data unavailable")
                            
                            with st.expander("💡 Local Tips"):
                                if suggestion['local_tips'] and suggestion['local_tips'][0] == "Local tips unavailable":
                                    st.warning("Local tips unavailable")
                                else:
                                    for tip in suggestion['local_tips']:
                                        st.write(f"• {tip}")
                            
                            with st.expander("🏨 Hotel Options"):
                                if suggestion['hotels'] and suggestion['hotels'][0].get('name') == "Hotel data unavailable":
                                    st.warning("Hotel data unavailable")
                                else:
                                    for hotel in suggestion['hotels']:
                                        st.write(f"**{hotel['name']}** - {hotel['price']} ({hotel['rating']})")
                                        st.write(f"*{hotel['address']}*")
                                        st.write("Amenities: " + ", ".join(hotel['amenities']))
                                        st.divider()
                            
                            with st.expander("✈️ Flight Options"):
                                if suggestion['flights'] and suggestion['flights'][0].get('airline') == "Flight data unavailable":
                                    st.warning("Flight data unavailable")
                                else:
                                    for flight in suggestion['flights']:
                                        st.write(f"**{flight['airline']}** - Flight {flight.get('flight_number', 'N/A')}")
                                        st.write(f"**{flight['departure']} → {flight['arrival']}**")
                                        st.write(f"**{flight['departure_time']} - {flight['arrival_time']}** ({flight['duration']})")
                                        st.write(f"**Price:** {flight['price']} | **Stops:** {flight['stops']}")
                                        st.divider()
                            
                            st.divider()  # Add a visual separator between suggestions
                            
                        except Exception as e:
                            st.error(f"Error displaying suggestion {i}: {str(e)}")
                            print(f"Error details: {traceback.format_exc()}")
                else:
                    st.warning("Could not parse structured suggestions from the response.")
                    
            elif result["type"] == "error":
                st.error(f"An error occurred: {result['content']}")
            else:
                st.error("Unexpected response type received.")

else:  # Create Detailed Itinerary
    st.header("📝 Create Detailed Itinerary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pre-populate destination if location info is available
        default_destination = st.session_state.location_info['location_name'] if st.session_state.location_info else "Paris, France"
        destination = st.text_input("Destination", default_destination)
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
                
                # Add location information if available from uploaded image
                if st.session_state.location_info:
                    context["location_information"] = {
                        "location_name": st.session_state.location_info['location_name'],
                        "latitude": st.session_state.location_info['latitude'],
                        "longitude": st.session_state.location_info['longitude'],
                        "confidence": st.session_state.location_info.get('confidence', 0.0)
                    }
                
                # Use MCP server's agent to create itinerary
                async def create_itinerary():
                    async with httpx.AsyncClient() as client:
                        # Create the query with location information if available
                        query = f"Create a detailed itinerary for a trip from {origin} to {destination}"
                        
                        # Enhance query with location information if available
                        if st.session_state.location_info:
                            query += f" (Location identified from image: {st.session_state.location_info['location_name']} at coordinates {st.session_state.location_info['latitude']}, {st.session_state.location_info['longitude']})"
                        
                        request_data = {
                            "query": query,
                            "context": context,
                            "session_id": st.session_state.conversation_session_id,
                            "conversation_history": st.session_state.conversation_history
                        }
                        response = await client.post(
                            f"{AGENT_URL}/agent/execute",
                            json=request_data,
                            timeout=60.0  # Increased timeout to 60 seconds
                        )
                        return response.json()
                
                result = asyncio.run(create_itinerary())
                
                if result.get("status") == "success":
                    # Update session state
                    if "session_id" in result:
                        st.session_state.conversation_session_id = result["session_id"]
                    if "conversation_history" in result:
                        st.session_state.conversation_history = result["conversation_history"]
                    
                    itinerary_response = result["result"].get("output", "")
                    
                    # Display itinerary
                    st.subheader("📝 Your Travel Itinerary")
                    st.markdown(itinerary_response)
                    
                    # Add some helpful information
                    st.info("💡 **Tip**: You can ask follow-up questions below to modify or get more details about your itinerary.")
                else:
                    st.error("Failed to create itinerary")

            except Exception as e:
                st.error(f"An error occurred while creating the itinerary: {str(e)}")

# Add chat interface for follow-up questions
st.markdown("---")
st.subheader("💬 Ask Follow-up Questions")

# Show conversation context if available
if st.session_state.conversation_history:
    with st.expander("📋 Conversation Context"):
        st.write("**Recent conversation summary:**")
        recent_messages = st.session_state.conversation_history[-4:]  # Show last 4 messages
        for msg in recent_messages:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            st.write(f"{role_icon} **{msg['role'].title()}:** {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")

# Example follow-up questions
with st.expander("💡 Example Follow-up Questions"):
    st.write("Try asking questions like:")
    example_questions = [
        "Can you add more details about the restaurants?",
        "What about transportation options?",
        "Can you suggest alternative activities?",
        "What's the weather like during that time?",
        "Can you modify the itinerary for a different budget?",
        "What are the best photo spots?",
        "Can you add more cultural activities?",
        "What about safety considerations?"
    ]
    for question in example_questions:
        st.write(f"• {question}")

# Chat input
follow_up_question = st.text_input(
    "Ask a follow-up question about your travel plans...",
    placeholder="e.g., 'Can you add more details about the restaurants?' or 'What about transportation options?'",
    key="follow_up_input"
)

if st.button("Send Follow-up", key="send_follow_up"):
    if follow_up_question.strip():
        with st.spinner("Processing your follow-up question..."):
            try:
                # Create context with current preferences
                context = {
                    "preferences": preferences.model_dump(exclude_none=True),
                    "mode": "follow_up"
                }
                
                # Add location information if available from uploaded image
                if st.session_state.location_info:
                    context["location_information"] = {
                        "location_name": st.session_state.location_info['location_name'],
                        "latitude": st.session_state.location_info['latitude'],
                        "longitude": st.session_state.location_info['longitude'],
                        "confidence": st.session_state.location_info.get('confidence', 0.0)
                    }
                
                # Prepare request with conversation state
                request_data = {
                    "query": follow_up_question,
                    "context": context,
                    "session_id": st.session_state.conversation_session_id,
                    "conversation_history": st.session_state.conversation_history
                }
                
                # Send follow-up question
                async def send_follow_up():
                    async with httpx.AsyncClient() as client:
                        try:
                            response = await client.post(
                                f"{AGENT_URL}/agent/execute",
                                json=request_data,
                                timeout=60.0  # Increased timeout to 60 seconds
                            )
                            return response.json()
                        except httpx.TimeoutException:
                            return {"status": "error", "error": "Request timed out. Please try again."}
                        except httpx.ConnectError:
                            return {"status": "error", "error": "Cannot connect to the travel agent server."}
                        except Exception as e:
                            return {"status": "error", "error": f"An error occurred: {str(e)}"}
                
                result = asyncio.run(send_follow_up())
                
                if result.get("status") == "success":
                    # Update session state
                    if "session_id" in result:
                        st.session_state.conversation_session_id = result["session_id"]
                    if "conversation_history" in result:
                        st.session_state.conversation_history = result["conversation_history"]
                    
                    # Display the response
                    response_content = result["result"].get("output", "")
                    st.markdown("**AI Response:**")
                    st.markdown(response_content)
                    
                    # Clear the input
                    st.session_state.follow_up_input = ""
                    st.rerun()
                else:
                    st.error("Failed to process follow-up question")
                    
            except Exception as e:
                st.error(f"Error processing follow-up: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Powered by AI Travel Planner • Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True) 