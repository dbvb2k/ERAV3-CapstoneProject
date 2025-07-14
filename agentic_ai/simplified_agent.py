import asyncio
import os
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import traceback
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.prompts import HumanMessagePromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from simplified_tools import get_langchain_tools
from fallback_llm import FallbackLLM  # Import the FallbackLLM

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define request models
class AgentRequest(BaseModel):
    query: str
    context: Dict = {}
    session_id: Optional[str] = None
    conversation_history: List = Field(default_factory=list)

class ConversationSession:
    """Manages conversation state for a user session"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True,
            memory_key="chat_history",
            input_key="input"
        )
        self.context = {}
        
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        if role == "user":
            self.memory.chat_memory.add_user_message(content)
        elif role == "assistant":
            self.memory.chat_memory.add_ai_message(content)
        self.last_activity = datetime.now()
        
    def get_messages(self) -> List[Dict[str, str]]:
        """Get conversation history as a list of dicts"""
        messages = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        return messages

class SimplifiedAgent:
    """
    Simplified agent that uses LangChain tools to handle travel queries.
    """
    
    def __init__(self):
        self.app = FastAPI(title="Travel Planner API")
        self.tools = get_langchain_tools()
        self.agent_executor = None
        self.conversation_sessions = {}  # session_id -> ConversationSession
        self.travel_dates =[]
        
        # Set up routes and agent
        self.setup_routes()
        self.setup_agent()

    def cleanup_expired_sessions(self, max_age_hours=24):
        """Remove sessions that are older than max_age_hours"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.conversation_sessions.items():
            age = (current_time - session.last_activity).total_seconds() / 3600
            if age > max_age_hours:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.conversation_sessions[session_id]
            
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """Get existing session or create a new one"""
        if session_id not in self.conversation_sessions:
            self.conversation_sessions[session_id] = ConversationSession(session_id)
        return self.conversation_sessions[session_id]
        
    def _format_context_for_prompt(self, context):
        """Format context information for the prompt template"""
        if not context:
            return "No additional context provided."
        
        formatted_context = ""
        
        # Handle location information specifically
        if "location_information" in context:
            loc_info = context["location_information"]
            formatted_context += f"LOCATION FROM IMAGE: {loc_info.get('location_name', 'Unknown')}\n"
            formatted_context += f"Coordinates: {loc_info.get('latitude', 'N/A')}, {loc_info.get('longitude', 'N/A')}\n"
            if 'confidence' in loc_info:
                formatted_context += f"Confidence: {loc_info['confidence']:.2f}\n"
            formatted_context += "\n"
        
        # Handle preferences
        if "preferences" in context:
            prefs = context["preferences"]
            formatted_context += "USER PREFERENCES:\n"
            for key, value in prefs.items():
                if value:  # Only include non-empty values
                    formatted_context += f"- {key}: {value}\n"
            formatted_context += "\n"
        
        # Handle travel request
        if "travel_request" in context:
            travel_req = context["travel_request"]
            formatted_context += "TRAVEL REQUEST:\n"
            formatted_context += f"- Origin: {travel_req.get('origin', 'N/A')}\n"
            formatted_context += f"- Destination: {travel_req.get('destination', 'N/A')}\n"
            formatted_context += f"- Duration: {travel_req.get('start_date', 'N/A')} to {travel_req.get('end_date', 'N/A')}\n"
            formatted_context += f"- Travelers: {travel_req.get('num_travelers', 'N/A')}\n"
            formatted_context += "\n"
        
        # Handle mode
        if "mode" in context:
            formatted_context += f"MODE: {context['mode']}\n\n"
        
        # Add any other context
        for key, value in context.items():
            if key not in ["location_information", "preferences", "travel_request", "mode"]:
                formatted_context += f"{key}: {value}\n"
        
        return formatted_context.strip()
    
    def _preprocess_query_for_tool_use(self, query):
        """Add hints to queries that should trigger tool use."""
        travel_keywords = ["flight", "hotel", "book", "price", "cost", "itinerary", "weather"]
        date_patterns = [r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', r'January|February|March|April|May|June|July|August|September|October|November|December']
        
        # Check if this is a travel query that should use tools
        should_use_tools = any(keyword in query.lower() for keyword in travel_keywords) or \
                        any(re.search(pattern, query, re.IGNORECASE) for pattern in date_patterns)
        
        if should_use_tools:
            # Add tool usage hint
            enhanced_query = f"[IMPORTANT: Use appropriate tools to get REAL flight and hotel data] {query}"
            return enhanced_query
        
        return query
    
    def _extract_travel_dates(self, query):
        """Extract travel dates from query text."""
        # Look for date patterns
        date_pattern = r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})'
        matches = re.findall(date_pattern, query, re.IGNORECASE)
        
        dates = []
        for day, month, year in matches:
            try:
                date_obj = datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
                dates.append(date_obj)
            except ValueError:
                continue
        self.travel_dates = dates if dates else []
        return dates if dates else []

    # async def _maybe_await(self, obj):
    #     """Helper to await an object if it's a coroutine, otherwise return it as is."""
    #     if asyncio.iscoroutine(obj):
    #         try:
    #             return await obj
    #         except Exception as e:
    #             logger.error(f"Error awaiting coroutine: {str(e)}")
    #             return {"error": f"Error executing tool: {str(e)}"}
    #     return obj
    
    async def deep_await_coroutines(self, obj):
        """Recursively await any coroutines in a nested structure."""
        if asyncio.iscoroutine(obj):
            return await obj
        
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                result[key] = await self.deep_await_coroutines(value)
            return result
        elif isinstance(obj, list):
            return [await self.deep_await_coroutines(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple([await self.deep_await_coroutines(item) for item in obj])
        else:
            return obj
    
    
    async def format_tool_results_for_response(self, query: str, tool_calls: list) -> str:
        """Create a formatted string with tool results for the agent to use in its final response."""
        
        # Extract actual real data from all tool calls
        flight_data = []
        hotel_data = []
        weather_data = None
        itinerary_data = None
        
        for tc in tool_calls:
            if tc["tool"] == "FlightSearchTool" and isinstance(tc["output"], list) and tc["output"]:
                flight_data = tc["output"]
            elif tc["tool"] == "HotelSearchTool" and isinstance(tc["output"], list) and tc["output"]:
                hotel_data = tc["output"]
            elif tc["tool"] == "WeatherTool" and isinstance(tc["output"], dict):
                weather_data = tc["output"]
            elif tc["tool"] == "ItineraryPlannerTool" and isinstance(tc["output"], dict):
                itinerary_data = tc["output"]
        
        # Format flight data
        flight_section = ""
        if flight_data:
            flight_section = "**Flight Options:**\n\n"
            for i, flight in enumerate(flight_data[:3], 1):
                flight_section += f"{i}. **{flight.get('airline', 'Unknown')} {flight.get('flight_number', '')}**\n"
                flight_section += f"   - Departure: {flight.get('departure_time', 'N/A')}\n"
                flight_section += f"   - Arrival: {flight.get('arrival_time', 'N/A')}\n" 
                flight_section += f"   - Duration: {flight.get('duration', 'N/A')}\n"
                flight_section += f"   - Price: ₹{flight.get('price', 'N/A')}\n\n"
        
        # Format hotel data
        hotel_section = ""
        if hotel_data:
            hotel_section = "**Hotel Options:**\n\n"
            for i, hotel in enumerate(hotel_data[:3], 1):
                hotel_section += f"{i}. **{hotel.get('name', 'Unknown')}**\n"
                hotel_section += f"   - Price: {hotel.get('price', 'N/A')} per night\n"
                hotel_section += f"   - Rating: {hotel.get('rating', 'N/A')}/5 ({hotel.get('reviews_count', 0)} reviews)\n"
                if hotel.get('link'):
                    hotel_section += f"   - Website: {hotel.get('link')}\n\n"
                else:
                    hotel_section += "\n"
        
        # Format weather data
        weather_section = ""
        if weather_data:
            weather_section = "**Current Weather:**\n\n"
            weather_section += f"- Temperature: {weather_data.get('temperature', 'N/A')}°C\n"
            weather_section += f"- Conditions: {weather_data.get('description', 'N/A')}\n"
            weather_section += f"- Humidity: {weather_data.get('humidity', 'N/A')}%\n\n"
        
        # Format itinerary data
        itinerary_section = ""
        if itinerary_data:
            itinerary_section = "**Recommended Itinerary:**\n\n"
            
            # Add best time and budget info if available
            if "best_time_to_visit" in itinerary_data:
                itinerary_section += f"- Best time to visit: {itinerary_data.get('best_time_to_visit', 'Year-round')}\n"
            if "estimated_budget" in itinerary_data:
                itinerary_section += f"- Estimated budget: {itinerary_data.get('estimated_budget', 'Varies by traveler')}\n\n"
            
            # Add daily plans if available
            if "daily_plans" in itinerary_data and itinerary_data["daily_plans"]:
                for day in itinerary_data["daily_plans"]:
                    day_num = day.get("day", "")
                    itinerary_section += f"**Day {day_num}:**\n"
                    
                    # Format morning activities
                    if "morning" in day:
                        morning_activity = day["morning"]
                        # Check if it's a dictionary or string
                        if isinstance(morning_activity, dict):
                            activity = morning_activity.get('activity', 'No activity specified')
                            time = morning_activity.get('time', '')
                            cost = morning_activity.get('cost', '')
                            itinerary_section += f"- Morning: {activity}"
                            if time:
                                itinerary_section += f" ({time})"
                            if cost:
                                itinerary_section += f" - {cost}"
                            itinerary_section += "\n"
                        else:
                            # Handle as string
                            itinerary_section += f"- Morning: {morning_activity}\n"
                    
                    # Format afternoon activities
                    if "afternoon" in day:
                        afternoon_activity = day["afternoon"]
                        # Check if it's a dictionary or string
                        if isinstance(afternoon_activity, dict):
                            activity = afternoon_activity.get('activity', 'No activity specified')
                            time = afternoon_activity.get('time', '')
                            cost = afternoon_activity.get('cost', '')
                            itinerary_section += f"- Afternoon: {activity}"
                            if time:
                                itinerary_section += f" ({time})"
                            if cost:
                                itinerary_section += f" - {cost}"
                            itinerary_section += "\n"
                        else:
                            # Handle as string
                            itinerary_section += f"- Afternoon: {afternoon_activity}\n"
                    
                    # Format evening activities
                    if "evening" in day:
                        evening_activity = day["evening"]
                        # Check if it's a dictionary or string
                        if isinstance(evening_activity, dict):
                            activity = evening_activity.get('activity', 'No activity specified')
                            time = evening_activity.get('time', '')
                            cost = evening_activity.get('cost', '')
                            itinerary_section += f"- Evening: {activity}"
                            if time:
                                itinerary_section += f" ({time})"
                            if cost:
                                itinerary_section += f" - {cost}"
                            itinerary_section += "\n"
                        else:
                            # Handle as string
                            itinerary_section += f"- Evening: {evening_activity}\n"
                    
                    itinerary_section += "\n"
        
        # Determine destination and dates from the tool calls
        destination = None
        from_date = None
        to_date = None
        
        # Extract destination from tools
        if itinerary_data and "destination" in itinerary_data:
            destination = itinerary_data["destination"]
        elif tool_calls:
            for tc in tool_calls:
                if tc["tool"] == "HotelSearchTool" and isinstance(tc["input"], dict):
                    destination = tc["input"].get("location")
                    break
                elif tc["tool"] == "WeatherTool" and isinstance(tc["input"], str):
                    destination = tc["input"]
                    break
        
        # Extract dates from tools or class attribute
        if hasattr(self, 'travel_dates') and self.travel_dates and len(self.travel_dates) >= 2:
            from_date = self.travel_dates[0]
            to_date = self.travel_dates[1]
        else:
            for tc in tool_calls:
                if tc["tool"] == "HotelSearchTool" and isinstance(tc["input"], dict):
                    from_date = tc["input"].get("check_in")
                    to_date = tc["input"].get("check_out")
                    break
                elif tc["tool"] == "FlightSearchTool" and isinstance(tc["input"], dict):
                    from_date = tc["input"].get("depart_date")
                    to_date = tc["input"].get("return_date")
                    break
        
        # Create the prompt for the final response
        prompt = f"""
        You are a travel itinerary summarizer whose role is to only generate a well-structured itinerary from the data given below.
        User Query: "{query}"

        We have collected REAL travel data that you MUST use in your response. 
        Do not invent or fabricate any travel information. Do not add additional days in your itinerary.  
        Use ONLY the data provided below:

        {flight_section}
        {hotel_section}
        {weather_section}
        {itinerary_section}

        Based ONLY on this real data, create a comprehensive travel response that:
        1. Acknowledges the user's request
        2. Includes ALL available flight options using ONLY the data provided above
        3. Includes ALL available hotel options using ONLY the data provided above  
        4. Includes the current weather information if available
        5. Provides a detailed daily itinerary from the **Recommended Itinerary**
        6. Creates a cohesive travel plan that incorporates ALL tool data together
        7. Avoid using "Shopping and Shopping" or similar incoherent phrases
        8. Uses only the information provided in the tool data, not generate any additional text.

        Destination: {destination or 'Not specified'}
        Travel period: {from_date or 'Not specified'} to {to_date or 'Not specified'}
        
        INSTRUCTIONS:
        - Format your response in a clear, helpful way. Do not mention this prompt or that you're using "real data" - simply present the information as your recommendations.
        - Repeat the EXACT SAME Itinerary as given by the **Recommended Itinerary:**. DO NOT ADD ADDITIONAL DAYS, limit it to the day activities mentioned in **Recommended Itinerary:**
        """

        # Call LLM to generate final response
        llm = FallbackLLM(temperature=0.1, max_length=2000)
        enhanced_response = await llm.apredict(prompt)
        return enhanced_response
    
    
    def setup_agent(self):
        """Set up the agent with tools using FallbackLLM."""
        # Initialize FallbackLLM with debug mode
        llm = FallbackLLM(
            temperature=0.5,
            max_length=2000,
            debug=False  # Enable debug output
        )
        
        # Create more aggressive system prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a travel planning assistant with access to real-time tools.

            CRITICAL TOOL INSTRUCTIONS:
            YOU MUST use tools to answer travel queries. DO NOT fabricate information.
            
            MANDATORY TOOL USAGE INSTRUCTIONS:
            1. When asked about travel between locations, you MUST call FlightSearchTool with origin, destination and date.
            2. When asked about accommodations, you MUST call HotelSearchTool with location, check-in and check-out dates.
            3. When asked about weather, you MUST call WeatherTool with location.
            4. ALWAYS call tools FIRST before generating responses about flights, hotels, or weather.
            
            CRITICAL RESPONSE INSTRUCTIONS:
            1. After receiving tool outputs, you MUST incorporate the actual data from the tools in your response.
            2. For flights: Include specific flight numbers, prices, times, and airlines from the tool response.
            3. For hotels: Include actual hotel names, prices, ratings, and locations from the tool response.
            4. NEVER reference tool outputs in JSON format in your final answer.
            5. Format the information in a readable, user-friendly way.
            6. Always create a detailed itinerary based on REAL tool outputs, not fabricated information.
            
            IMPORTANT: Never fake tool calls by writing "I have called..." - You must actually call the tools through the API.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("Context: {context}\n\nUser Query: {input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Import and create the agent with the correct approach
        from langchain.agents import AgentExecutor, create_openai_tools_agent
        
        # Create the agent properly as a tools agent
        agent = create_openai_tools_agent(llm, self.tools, prompt)
        
        # Create the agent executor with more aggressive tool usage
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=2,
            early_stopping_method="force",
            return_intermediate_steps=True,
        )
        
        logger.info("Agent initialized with FallbackLLM and tools")
        
    def setup_routes(self):
        @self.app.post("/agent/execute")
        async def execute_agent(request: AgentRequest):
            """Execute the agent with a user query."""
            if not self.agent_executor:
                raise HTTPException(status_code=500, detail={"error": "Agent not initialized"})
            
            try:
                # Get or create session
                session_id = request.session_id or str(uuid.uuid4())
                session = self.get_or_create_session(session_id)
                
                self.cleanup_expired_sessions()
                
                # Add tool-use hint to the query
                enhanced_query = self._preprocess_query_for_tool_use(request.query)
                
                # Add user message to conversation history
                session.add_message("user", request.query)
                
                # Extract travel dates if available
                if hasattr(self, '_extract_travel_dates') and callable(self._extract_travel_dates):
                    self.travel_dates = self._extract_travel_dates(request.query)
                    
                # Prepare input for the agent
                agent_input = {
                    "input": f"[IMPORTANT: Use appropriate tools to get REAL Flight, Hotel, Itinerary and Weather Data] {request.query}",
                    "context": self._format_context_for_prompt(session.context),
                    "chat_history": session.memory.chat_memory.messages
                }
                
                # Add travel dates if extracted
                if hasattr(self, 'travel_dates') and self.travel_dates:
                    agent_input["travel_dates"] = self.travel_dates
                
                # Execute the agent
                logger.info(f"Executing agent with query: {request.query}")
                logger.info(f"Agent input: {json.dumps(agent_input, default=str)}")
                
                # Add debug for agent execution
                print("==== EXECUTING AGENT WITH TOOLS ====")
                for tool in self.tools:
                    print(f"Available tool: {tool.name} - {tool.description.split('.')[0]}")
                print("=================================")

                result = await self.agent_executor.ainvoke(agent_input)
                processed_result = await self.deep_await_coroutines(result)
                
                # Extract and process tool calls
                tool_calls = []
                for i, step in enumerate(processed_result.get('intermediate_steps', [])):
                    if len(step) >= 2:
                        action, output = step
                        tool_calls.append({
                            "tool": action.tool,
                            "input": action.tool_input,
                            "output": output
                        })
                
                # Process the result to ensure real data is used in the final response
                if tool_calls:
                    # Generate enhanced response with real tool data
                    enhanced_response = await self.format_tool_results_for_response(
                        request.query,
                        tool_calls
                    )
                    
                    # Replace the original output
                    processed_result["output"] = enhanced_response
                    
                    # Add to conversation history
                    session.add_message("assistant", enhanced_response)
                else:
                    # If no tool calls were made, use the original response
                    session.add_message("assistant", processed_result.get("output", ""))
                
                # Debug the result
                print(f"==== AGENT RESULT ====")
                print(f"Intermediate steps: {len(processed_result.get('intermediate_steps', []))}")
                for i, step in enumerate(processed_result.get('intermediate_steps', [])):
                    logger.info(f"Intermediate steps: {len(processed_result['intermediate_steps'])}")
                    if len(step) >= 2:
                        action, output = step
                        print(f"Step {i+1}: Tool {action.tool} called with inputs {action.tool_input}")
                        print(f"Output: {output}")
                print("======================")
                
                return {
                    "status": "success", 
                    "result": processed_result,
                    "tool_calls": tool_calls,
                    "session_id": session_id,
                    "conversation_history": session.get_messages()
                }
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in execute_agent: {error_msg}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail={"error": error_msg})

        @self.app.get("/conversation/{session_id}")
        async def get_conversation(session_id: str):
            """Get conversation history for a session"""
            if session_id not in self.conversation_sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = self.conversation_sessions[session_id]
            return {
                "session_id": session_id,
                "conversation_history": session.get_messages(),
                "context": session.context,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat()
            }
    
    def run(self, host="0.0.0.0", port=8000):
        """Run the FastAPI server."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

# Create and run the agent when this file is executed directly
if __name__ == "__main__":
    agent = SimplifiedAgent()
    agent.run()