import os
import tempfile
import asyncio
import nest_asyncio
import requests
import streamlit as st
from PIL import Image
import json
import logging
from datetime import datetime
import pathlib
import re
import codecs

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = pathlib.Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"llm_prompts_{timestamp}.log"
    
    # Configure logging with UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Create a custom formatter that handles Unicode
    class UnicodeFormatter(logging.Formatter):
        def format(self, record):
            # Convert any non-ASCII characters to their Unicode escape sequences
            if isinstance(record.msg, str):
                record.msg = record.msg.encode('ascii', 'backslashreplace').decode('ascii')
            return super().format(record)
    
    # Apply the custom formatter to all handlers
    for handler in logging.getLogger().handlers:
        handler.setFormatter(UnicodeFormatter('%(asctime)s - %(message)s'))
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Configure Streamlit page
st.set_page_config(
    page_title="Location Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuration
API_URL = "http://localhost:8000/predict"  # GeoCLIP API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama API endpoint
MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"  # Ollama model name

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "location_info" not in st.session_state:
    st.session_state.location_info = None

def extract_trip_duration(prompt):
    """Extract the number of days from the user's prompt."""
    # Look for patterns like "3 days", "5 days", "7 days", etc.
    match = re.search(r'(\d+)\s*days?', prompt.lower())
    if match:
        return int(match.group(1))
    return 3  # Default to 3 days if not specified

def generate_dynamic_days(start_date, num_days):
    """Generate a list of dates for the itinerary."""
    from datetime import datetime, timedelta
    start = datetime.strptime(start_date, "%B %d, %Y")
    dates = []
    for i in range(num_days):
        current_date = start + timedelta(days=i)
        dates.append(current_date.strftime("%B %d, %Y"))
    return dates

def generate_response(prompt, location_info):
    # Extract number of days from user prompt
    num_days = extract_trip_duration(prompt)
    start_date = "July 25, 2025"  # This could also be made dynamic if needed
    dates = generate_dynamic_days(start_date, num_days)
    
    # Create a more specific and directive prompt template
    context = f"""You are a travel planning expert. Create a detailed {num_days}-day itinerary for {location_info['location_name']} starting from {start_date}. The traveler will be departing from Bangalore, India and requires economy flight tickets.

IMPORTANT CONSTRAINTS:
- Trip duration must be exactly {num_days} days ({dates[0]} to {dates[-1]})
- Do not include any activities beyond day {num_days}
- Focus on the most important attractions that can be realistically visited in {num_days} days
- Ensure all timings are realistic and account for travel time between locations

LOCATION: {location_info['location_name']}
COORDINATES: {location_info['latitude']}, {location_info['longitude']}
DEPARTURE: Bangalore, India
TRAVEL CLASS: Economy
DURATION: {num_days} days ({dates[0]} to {dates[-1]})

Based on this information, provide a detailed travel plan that includes:

FLIGHT INFORMATION:
- Specific airlines operating between Bangalore and {location_info['location_name']}
- Flight durations and layovers
- Economy class ticket prices in INR
- Best booking recommendations
- Note: Include only flights that arrive on {dates[0]} and depart on {dates[-1]}

"""

    # Dynamically generate day sections
    for i, date in enumerate(dates, 1):
        context += f"""
DAY {i} ({date}):
Morning (8:00 AM - 12:00 PM): [Specific activities with exact timings]
Afternoon (12:00 PM - 5:00 PM): [Specific activities with exact timings]
Evening (5:00 PM - 10:00 PM): [Specific activities with exact timings]
"""

    context += f"""
TRANSPORTATION:
- International Flights: List specific airlines, routes, and economy class prices from Bangalore
- Local Transport: Include specific options like metro, buses, or taxis with costs
- Airport Transfer: Options and costs from airport to hotel
- Local Sightseeing: Pre-booked taxi service details and costs

ACCOMMODATION:
- List 3 specific hotels with their ratings, features, and price ranges
- Include both budget and luxury options
- Specify distance from main attractions
- Include booking recommendations
- Note: Book for exactly {num_days-1} nights ({dates[0]} to {dates[-1]})

DINING:
- List specific restaurants for each meal with their specialties
- Include local cuisine recommendations
- Provide price ranges for each option
- Include vegetarian/Indian food options

ATTRACTIONS:
- List specific must-visit places with brief descriptions
- Include entry fees and best visiting times
- Mention any special events or festivals
- Provide pre-booking recommendations
- Note: Focus on the most important attractions that can be realistically visited in {num_days} days

BUDGET BREAKDOWN (in INR):
- Flights: [Specific amount]
- Accommodation ({num_days-1} nights): [Specific amount]
- Local Transport: [Specific amount]
- Food: [Specific amount]
- Attractions: [Specific amount]
- Miscellaneous: [Specific amount]
- Total Estimated Cost: [Total amount]

TRAVEL TIPS:
- Best time to visit
- Local customs and etiquette
- Safety recommendations
- What to pack
- Visa requirements for Indian citizens
- Currency exchange recommendations
- Local SIM card and internet options

Remember to:
1. Be specific with names of places, restaurants, and attractions
2. Include realistic prices in INR where applicable
3. Consider local customs and best practices
4. Format the response in a clear, structured way
5. Ensure the itinerary is exactly {num_days} days (no more, no less)
6. Include specific information about flights from Bangalore
7. Account for realistic travel times between locations"""

    try:
        # Log the prompt with timestamp
        logger.info(f"Location: {location_info['location_name']}\nPrompt:\n{context}\n{'='*80}")

        # Prepare the request to Ollama API
        payload = {
            "model": MODEL_NAME,
            "prompt": context,
            "stream": False,
            "options": {
                "temperature": 0.7,  # Reduced for more focused responses
                "top_p": 0.95,
                "repeat_penalty": 1.3,
                "stop": ["Remember to:", "Human:", "Assistant:"]
            }
        }

        # Make request to Ollama API
        response = requests.post(OLLAMA_API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            # Log the response
            logger.info(f"Response:\n{generated_text}\n{'='*80}")
            
            return format_itinerary_response(generated_text)
        else:
            error_msg = f"Error from Ollama API: {response.text}"
            logger.error(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error while generating a response: {str(e)}"
        logger.error(error_msg)
        return error_msg

def format_itinerary_response(response):
    """Format the response as a structured itinerary."""
    # Add markdown formatting for better readability
    formatted_response = "### Your Travel Plan\n\n"
    
    # Split the response into sections
    sections = response.split("\n\n")
    current_section = ""
    
    for section in sections:
        if section.strip():
            # Check if it's a main section
            if section.strip().endswith(":"):
                current_section = section.strip()
                formatted_response += f"#### {current_section}\n\n"
            # Check if it's a day section
            elif "DAY" in section.strip():
                formatted_response += f"##### {section.strip()}\n\n"
            # Check if it's a time section
            elif "Morning" in section or "Afternoon" in section or "Evening" in section:
                formatted_response += f"**{section.strip()}**\n\n"
            # Check if it's a bullet point
            elif section.strip().startswith("-"):
                formatted_response += f"{section.strip()}\n\n"
            else:
                formatted_response += f"{section.strip()}\n\n"
    
    return formatted_response

def format_conversation_history():
    if not st.session_state.messages:
        return "No previous conversation."
    
    formatted_history = ""
    for msg in st.session_state.messages[-4:]:  # Only include last 4 messages for context
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n"
    return formatted_history

# Main UI
st.title("🤖 Location Chatbot")

# Create two columns
col1, col2 = st.columns(2)

# Left column for image upload and display
with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=False)
        
        # Get location predictions if not already done
        if st.session_state.location_info is None:
            try:
                with st.spinner("Identifying location..."):
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "image/jpeg"
                        )
                    }
                    
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        predictions = response.json()
                        st.session_state.location_info = predictions[0]
                        st.success(f"Location identified: {st.session_state.location_info['location_name']}")
                    else:
                        st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

# Right column for chat interface
with col2:
    st.subheader("Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about this place?"):
        if st.session_state.location_info is None:
            st.warning("Please upload an image first!")
        else:
            try:
                with st.spinner("Planning your trip..."):
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Generate and display response
                    response = generate_response(prompt, st.session_state.location_info)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Display the new messages
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    
            except Exception as e:
                st.error(f"Error processing request: {str(e)}") 