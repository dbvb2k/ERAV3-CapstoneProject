import streamlit as st
import asyncio
import os
from typing import Optional
import sys
import traceback
import nest_asyncio

# Enable nested event loops
nest_asyncio.apply()

# Add the project root and src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
src_dir = os.path.join(project_root, 'src')

# Ensure the paths exist before adding them
if os.path.exists(project_root):
    sys.path.insert(0, project_root)
if os.path.exists(src_dir):
    sys.path.insert(0, src_dir)

try:
    from travel_bot.travel_bot import TravelBot
    from travel_bot.models import InputType, TravelPreferences
    from config.settings import Config
except ImportError as e:
    st.error(f"""Failed to import required modules. Please check your installation.
Error: {str(e)}
Current paths:
- Project root: {project_root}
- Src directory: {src_dir}
""")
    st.stop()

# Create event loop for async operations
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

class StreamlitTravelApp:
    """Streamlit interface for the Travel Advisory Bot"""
    
    def __init__(self):
        self.travel_bot = None
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="AI Travel Advisory Bot",
            page_icon="✈️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'travel_bot' not in st.session_state:
            st.session_state.travel_bot = None
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'current_suggestions' not in st.session_state:
            st.session_state.current_suggestions = []
        if 'current_itinerary' not in st.session_state:
            st.session_state.current_itinerary = None
    
    def initialize_travel_bot(self, model_type: str = "huggingface"):
        """Initialize travel bot with selected model"""
        try:
            if st.session_state.travel_bot is None:
                with st.spinner("Initializing AI model..."):
                    st.session_state.travel_bot = TravelBot(model_type)
                    
                    # Check if model is properly loaded
                    if not st.session_state.travel_bot.model.is_available():
                        st.error("""
                        ❌ Failed to initialize the AI model. This could be due to:
                        1. Missing dependencies (try: pip install -e .)
                        2. Memory constraints
                        3. Model download issues
                        
                        Please check the terminal output for detailed error messages.
                        """)
                        return None
                    else:
                        st.success("✅ AI model initialized successfully!")
                        
            return st.session_state.travel_bot
        except Exception as e:
            st.error(f"""
            ❌ Error initializing Travel Bot: {str(e)}
            
            Please try:
            1. Refreshing the page
            2. Checking your internet connection
            3. Running: pip install -e .
            """)
            return None
    
    def render_sidebar(self):
        """Render sidebar with configuration options"""
        st.sidebar.header("🔧 Configuration")
          # Model selection
        model_options = ["devstral", "huggingface", "ollama"]
        selected_model = st.sidebar.selectbox(
            "Select AI Model",
            model_options,
            index=0,
            help="Choose the AI model for generating responses",
            key="ai_model_selectbox"
        )
        
        # Configuration validation
        config_status = Config.validate_config()
        
        st.sidebar.subheader("📋 Model Status")
        for model, status in config_status.items():
            icon = "✅" if status else "❌"
            st.sidebar.write(f"{icon} {model.replace('_', ' ').title()}")
          # Language preference
        st.sidebar.subheader("🌍 Language Preference")
        language = st.sidebar.selectbox(
            "Select Language",
            Config.SUPPORTED_LANGUAGES,
            index=0,
            key="language_selectbox"
        )
        
        return selected_model, language
    
    def render_travel_preferences(self, context: str = "default") -> TravelPreferences:
        """Render travel preferences form"""
        st.subheader("🎯 Travel Preferences")
        
        col1, col2 = st.columns(2)
        with col1:
            budget_range = st.selectbox(
                "Budget Range",
                ["Budget (Under $50/day)", "Moderate ($50-150/day)", 
                 "Luxury ($150-300/day)", "Ultra-luxury ($300+/day)"],
                index=1,
                key=f"budget_range_selectbox_{context}"
            )
            
            travel_style = st.selectbox(
                "Travel Style",
                ["Adventure", "Cultural", "Relaxation", "Business", 
                 "Family", "Romantic", "Budget", "Luxury"],                index=0,
                key=f"travel_style_selectbox_{context}"
            )
            
            group_size = st.number_input(
                "Group Size",
                min_value=1,
                max_value=20,
                value=2,
                step=1,
                key=f"group_size_number_input_{context}"
            )
        
        with col2:
            interests = st.multiselect(
                "Interests",
                ["History", "Food", "Nature", "Art", "Sports", "Shopping", 
                 "Nightlife", "Architecture", "Museums", "Beaches", "Mountains"],
                default=["Food", "Nature"],
                key=f"interests_multiselect_{context}"
            )
            
            accommodation_type = st.selectbox(
                "Preferred Accommodation",
                ["Hotel", "Airbnb", "Hostel", "Resort", "Boutique Hotel", "Any"],
                index=0,                key=f"accommodation_type_selectbox_{context}"
            )
            
            dietary_restrictions = st.multiselect(
                "Dietary Restrictions",
                ["Vegetarian", "Vegan", "Gluten-free", "Halal", "Kosher", "No restrictions"],
                default=["No restrictions"],
                key=f"dietary_restrictions_multiselect_{context}"
            )
        
        return TravelPreferences(
            budget_range=budget_range,
            travel_style=travel_style.lower(),
            interests=interests,
            group_size=group_size,
            accommodation_type=accommodation_type,
            dietary_restrictions=dietary_restrictions,
            language_preference="en"  # Using the selected language from sidebar
        )
    
    def render_file_upload(self):
        """Render file upload interface"""
        st.subheader("📁 Upload Travel Documents")
        
        uploaded_files = st.file_uploader(
            "Upload images, PDFs, videos, or audio files",
            type=['jpg', 'jpeg', 'png', 'pdf', 'mp4', 'avi', 'mp3', 'wav'],
            accept_multiple_files=True,
            help="Upload travel documents, photos, or media for analysis"
        )
        
        return uploaded_files
    
    def process_uploaded_file(self, uploaded_file, travel_bot):
        """Process a single uploaded file"""
        try:
            # Determine input type based on file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                input_type = InputType.IMAGE
            elif file_extension == 'pdf':
                input_type = InputType.PDF
            elif file_extension in ['mp4', 'avi', 'mov', 'wmv']:
                input_type = InputType.VIDEO
            elif file_extension in ['mp3', 'wav', 'ogg', 'm4a']:
                input_type = InputType.AUDIO
            else:
                return {"error": f"Unsupported file type: {file_extension}"}
            
            # Process the file
            processed_input = asyncio.run(travel_bot.process_input(uploaded_file, input_type))
            
            return {
                "filename": uploaded_file.name,
                "type": input_type.value,
                "content": processed_input.content,
                "language": processed_input.language,
                "confidence": processed_input.confidence,
                "entities": processed_input.extracted_entities
            }
        
        except Exception as e:
            return {"error": f"Error processing {uploaded_file.name}: {str(e)}"}
    
    def render_chat_interface(self, travel_bot):
        """Render chat interface"""
        st.subheader("💬 Chat with Travel Assistant")
        
        # Display conversation history
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about travel..."):
            # Add user message to history
            st.session_state.conversation_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = asyncio.run(travel_bot.chat_with_bot(prompt))
                        st.write(response)
                        
                        # Add assistant response to history
                        st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.conversation_history.append({"role": "assistant", "content": error_msg})
    
    def render_suggestions_display(self, suggestions):
        """Display travel suggestions"""
        if not suggestions:
            return
        
        st.subheader("🎯 Travel Suggestions")
        
        for i, suggestion in enumerate(suggestions):
            with st.expander(f"Option {i+1}: {suggestion.destination}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Description:**")
                    st.write(suggestion.description)
                    
                    st.write("**Best Time to Visit:**")
                    st.write(suggestion.best_time_to_visit)
                    
                    st.write("**Estimated Budget:**")
                    st.write(suggestion.estimated_budget)
                    
                    st.write("**Recommended Duration:**")
                    st.write(suggestion.duration)
                
                with col2:
                    st.write("**Top Activities:**")
                    for activity in suggestion.activities[:5]:
                        st.write(f"• {activity}")
                    
                    st.write("**Accommodation Options:**")
                    for acc in suggestion.accommodation_suggestions[:3]:
                        st.write(f"• {acc}")
                    
                    st.write("**Transportation:**")
                    for transport in suggestion.transportation[:3]:
                        st.write(f"• {transport}")
                
                if st.button(f"Create Itinerary for {suggestion.destination}", key=f"itinerary_{i}"):
                    st.session_state.selected_destination = suggestion.destination
                    st.rerun()
    
    def render_itinerary_form(self):
        """Render itinerary creation form"""
        st.subheader("📅 Create Detailed Itinerary")
        
        col1, col2 = st.columns(2)
        
        with col1:            destination = st.text_input(
                "Destination",
                value=getattr(st.session_state, 'selected_destination', ''),
                placeholder="Enter destination city/country",
                key="itinerary_destination_text_input"
            )
        
        with col2:
            days = st.number_input(
                "Number of Days",
                min_value=1,
                max_value=30,
                value=7,
                step=1,
                key="itinerary_days_number_input"
            )
        
        return destination, days
    
    def render_itinerary_display(self, itinerary):
        """Display detailed itinerary"""
        if not itinerary:
            return
        
        st.subheader(f"📋 {itinerary.destination} - {itinerary.total_days} Day Itinerary")
        
        # Overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Duration", f"{itinerary.total_days} days")
        
        with col2:
            st.metric("Estimated Budget", itinerary.total_budget or "Not specified")
        
        with col3:
            st.metric("Activities", len([plan for plan in itinerary.daily_plans]))
        
        # Daily plans
        st.subheader("📆 Daily Schedule")
        
        for plan in itinerary.daily_plans:
            with st.expander(f"Day {plan.get('day', 1)} - {plan.get('date', '')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Morning:**", plan.get('morning', 'Free time'))
                    st.write("**Afternoon:**", plan.get('afternoon', 'Free time'))
                    st.write("**Evening:**", plan.get('evening', 'Free time'))
                
                with col2:
                    st.write("**Meals:**")
                    for meal in plan.get('meals', []):
                        st.write(f"• {meal}")
                    st.write("**Estimated Cost:**", plan.get('estimated_cost', 'Not specified'))
        
        # Additional information
        if itinerary.packing_list:
            st.subheader("🎒 Packing List")
            cols = st.columns(3)
            for i, item in enumerate(itinerary.packing_list):
                with cols[i % 3]:
                    st.write(f"• {item}")
        
        if itinerary.important_notes:
            st.subheader("⚠️ Important Notes")
            for note in itinerary.important_notes:
                st.info(note)
    
    def run(self):
        """Main application runner"""
        # Header
        st.title("✈️ AI Travel Advisory Bot")
        st.markdown("*Your intelligent travel companion for personalized trip planning*")
        
        # Sidebar
        selected_model, language = self.render_sidebar()
        
        # Initialize travel bot
        travel_bot = self.initialize_travel_bot(selected_model)
        
        if not travel_bot:
            st.error("Failed to initialize the travel bot. Please check your configuration.")
            return
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔍 Get Suggestions", "📅 Create Itinerary", "💬 Chat"])
        
        with tab1:
            st.markdown("""
            ## Welcome to Your AI Travel Assistant! 🌍
            
            This intelligent travel bot can help you:
            - **Analyze** your travel documents, photos, and media
            - **Generate** personalized travel suggestions
            - **Create** detailed itineraries
            - **Chat** about any travel-related questions
            
            ### Supported Input Types:
            - 📝 **Text**: Describe your travel preferences
            - 🖼️ **Images**: Upload photos for destination inspiration
            - 📄 **PDFs**: Analyze travel documents or brochures
            - 🎥 **Videos**: Extract travel information from video content
            - 🎵 **Audio**: Voice descriptions of your travel plans
            
            ### Get Started:
            1. Choose your AI model in the sidebar
            2. Set your travel preferences
            3. Upload files or use the chat interface
            4. Get personalized suggestions and itineraries!
            """)
              # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Supported Languages", len(Config.SUPPORTED_LANGUAGES))
            with col2:
                st.metric("File Types", "5+")
            with col3:
                st.metric("AI Models", "3")
            with col4:
                st.metric("Max File Size", f"{Config.MAX_FILE_SIZE_MB}MB")
        
        with tab2:
            st.header("🔍 Get Travel Suggestions")
            
            # Travel preferences
            preferences = self.render_travel_preferences("suggestions")
            
            # File upload
            uploaded_files = self.render_file_upload()
              # Text input
            text_input = st.text_area(
                "Describe your travel preferences",
                placeholder="I want to visit a warm destination with beautiful beaches and good food...",
                key="travel_suggestions_text_area"
            )
            
            if st.button("Get Suggestions", type="primary"):
                if text_input or uploaded_files:
                    with st.spinner("Analyzing your preferences and generating suggestions..."):
                        try:
                            suggestions = []
                            
                            # Process text input
                            if text_input:
                                text_suggestions = asyncio.run(
                                    travel_bot.get_travel_suggestions(text_input, InputType.TEXT, preferences)
                                )
                                suggestions.extend(text_suggestions)
                            
                            # Process uploaded files
                            for uploaded_file in uploaded_files or []:
                                file_result = self.process_uploaded_file(uploaded_file, travel_bot)
                                if "error" not in file_result:
                                    st.success(f"Processed: {file_result['filename']}")
                                else:
                                    st.error(file_result["error"])
                            
                            st.session_state.current_suggestions = suggestions
                            
                        except Exception as e:
                            st.error(f"Error generating suggestions: {str(e)}")
                            st.error(traceback.format_exc())
                else:
                    st.warning("Please provide either text input or upload files.")
              # Display suggestions
            if st.session_state.current_suggestions:
                self.render_suggestions_display(st.session_state.current_suggestions)
        
        with tab3:
            st.header("📅 Create Detailed Itinerary")
            
            # Itinerary form
            destination, days = self.render_itinerary_form()
            preferences = self.render_travel_preferences("itinerary")
            
            if st.button("Create Itinerary", type="primary"):
                if destination:
                    with st.spinner(f"Creating {days}-day itinerary for {destination}..."):
                        try:
                            itinerary = asyncio.run(
                                travel_bot.create_itinerary(destination, days, preferences)
                            )
                            st.session_state.current_itinerary = itinerary
                            st.success(f"Itinerary created for {destination}!")
                            
                        except Exception as e:
                            st.error(f"Error creating itinerary: {str(e)}")
                            st.error(traceback.format_exc())
                else:
                    st.warning("Please enter a destination.")
            
            # Display itinerary
            if st.session_state.current_itinerary:
                self.render_itinerary_display(st.session_state.current_itinerary)
        
        with tab4:
            self.render_chat_interface(travel_bot)

def main():
    """Main function to run the Streamlit app"""
    app = StreamlitTravelApp()
    app.run()

if __name__ == "__main__":
    main()