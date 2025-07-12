"""
MCP Server implementation using FastAPI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from .fallback_llm import FallbackLLM
import uvicorn
import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import json
import uuid
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Enable nested asyncio
nest_asyncio.apply()

class ToolRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class AgentRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
        
    def model_dump(self, **kwargs):
        """Custom serialization to handle nested Pydantic models."""
        data = super().model_dump(**kwargs)
        if "context" in data and isinstance(data["context"], dict):
            for key, value in data["context"].items():
                if hasattr(value, "model_dump"):
                    data["context"][key] = value.model_dump()
        return data

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
        self.preferences = {}
        
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
        
    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if session has expired"""
        return datetime.now() - self.last_activity > timedelta(hours=max_age_hours)

class MCPServer:
    def __init__(self):
        self.app = FastAPI(title="Travel Planner MCP Server")
        self.tools = {}
        self.agent_executor = None
        self.conversation_sessions = {}  # session_id -> ConversationSession
        self.setup_routes()
        
    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """Get existing session or create a new one"""
        if session_id not in self.conversation_sessions:
            self.conversation_sessions[session_id] = ConversationSession(session_id)
        return self.conversation_sessions[session_id]
        
    def cleanup_expired_sessions(self):
        """Remove expired conversation sessions"""
        expired_sessions = [
            session_id for session_id, session in self.conversation_sessions.items()
            if session.is_expired()
        ]
        for session_id in expired_sessions:
            del self.conversation_sessions[session_id]
    
    def _preprocess_query_for_tool_use(self, query):
        """Add hints to queries that should trigger tool use."""
        import re
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
        import re
        from datetime import datetime
        
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
        
        return dates if dates else None
        
        
    def setup_routes(self):
        @self.app.post("/invoke_tool")
        async def invoke_tool(request: ToolRequest):
            if request.tool_name not in self.tools:
                raise HTTPException(status_code=404, detail=f"Tool {request.tool_name} not found")
            
            tool = self.tools[request.tool_name]
            try:
                result = await tool.arun(**request.parameters)
                return {"status": "success", "result": result}
            except Exception as e:
                error_msg = str(e)
                if hasattr(e, 'detail'):
                    error_msg = e.detail
                raise HTTPException(status_code=500, detail={"error": error_msg})

        @self.app.post("/agent/execute")
        async def execute_agent(request: AgentRequest):
            if not self.agent_executor:
                raise HTTPException(status_code=500, detail={"error": "Agent not initialized"})
            
            try:
                # Clean up expired sessions
                self.cleanup_expired_sessions()
                
                # Get or create session
                session_id = request.session_id or str(uuid.uuid4())
                session = self.get_or_create_session(session_id)
                
                # Add tool-use hint to the query
                enhanced_query = self._preprocess_query_for_tool_use(request.query)
                
                # Add user message to conversation history
                session.add_message("user", request.query)
                
                # Update session context with new information
                if request.context:
                    session.context.update(request.context)
                
                # Prepare input with conversation history
                memory_variables = session.memory.load_memory_variables({})
                chat_history = memory_variables.get("chat_history", [])
                
                
                # Create input with conversation context
                agent_input = {
                    "input": enhanced_query,
                    "context": self._format_context_for_prompt(session.context),
                    "chat_history": chat_history
                }
                
                travel_dates = self._extract_travel_dates(request.query)
                if travel_dates and "context" in agent_input:
                    agent_input["travel_dates"] = [d.strftime("%Y-%m-%d") for d in travel_dates]
                    if isinstance(agent_input["context"], str):
                        agent_input["context"] += f"\nTravel dates: {', '.join(agent_input['travel_dates'])}"
                
                print(agent_input)
                
                result = await self.agent_executor.ainvoke(agent_input)        
                # Extract tool usage information
                tool_calls = []
                if "intermediate_steps" in result:
                    for step in result["intermediate_steps"]:
                        if len(step) >= 2:
                            action, action_output = step
                            tool_calls.append({
                                "tool": action.tool,
                                "input": action.tool_input,
                                "output": action_output
                            })
                
                # Add assistant response to conversation history
                if "output" in result:
                    session.add_message("assistant", result["output"])
                
                return {
                    "status": "success", 
                    "result": result,
                    "tool_calls": tool_calls,  # Include tool call information
                    "session_id": session_id,
                    "conversation_history": session.get_messages()
                }
            except Exception as e:
                error_msg = str(e)
                if hasattr(e, 'detail'):
                    error_msg = e.detail
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
            
        @self.app.delete("/conversation/{session_id}")
        async def delete_conversation(session_id: str):
            """Delete a conversation session"""
            if session_id in self.conversation_sessions:
                del self.conversation_sessions[session_id]
                return {"status": "success", "message": "Session deleted"}
            else:
                raise HTTPException(status_code=404, detail="Session not found")

    def register_tool(self, tool_name: str, tool):
        """Register a new tool with the MCP server."""
        self.tools[tool_name] = tool

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

    def setup_agent(self, tools):
        """Set up the LangChain agent with Fallback LLM (Local Llama + OpenRouter)."""
        # Initialize Fallback LLM
        llm = FallbackLLM(
            temperature=0.2,
            max_length=2000,
            debug = True
        )
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a travel planning assistant with access to real-time tools.

            MANDATORY TOOL USAGE INSTRUCTIONS:
            1. For ANY travel planning request, you MUST use tools to get real information.
            2. NEVER fabricate flight or hotel details - ONLY use the data from tools.
            3. You MUST call FlightSearchTool when asked about travel between locations.
            4. You MUST call HotelSearchTool when information about accommodations is needed.
            5. You MUST call WeatherTool for weather conditions.
            6. You MUST call LocationInfoTool to get real information about places.

            When a user asks about travel plans or itineraries:
            1. FIRST use relevant tools to collect accurate data
            2. THEN generate your response using ONLY that data

            TOOL CALLING PROCESS:
            1. IDENTIFY which tools are needed for the query
            2. CALL each needed tool with accurate parameters
            3. WAIT for tool results 
            4. USE tool results in your final answer
            5. CITE the specific tools you used

            DO NOT PRETEND to use tools by writing "ToolNameTool Response:" - actually call them!
            
            CONTEXT INFORMATION:
            1. If location information is provided from an uploaded image, use it as the primary destination for suggestions and
            itineraries. The location information includes:
                - location_name: The identified location from the image
                - latitude/longitude: Geographic coordinates
                - confidence: Confidence score of the identification
            2. If location information is not provided from the uploaded image, then check the user query for their preferred destination.
        
            RESPONSE FORMATS:
            1. For DESTINATION SUGGESTIONS (when user asks for general travel ideas):
                If location information is not available, return EXACTLY 2 suggestions in this format:
                * [Destination Name] for [brief description of culture and food highlights]

                Example:
                * Tokyo, Japan for its blend of modern technology and traditional culture, featuring world-class sushi and ramen
                * Barcelona, Spain for its stunning Gaudi architecture, vibrant tapas scene, and Mediterranean charm

            2. For SPECIFIC ITINERARIES (when user asks for a detailed plan for a specific destination):
                Return a detailed day-by-day itinerary in this format:

                Day 1:
                - Morning: [specific activity]
                - Afternoon: [specific activity]
                - Evening: [specific activity]

                Day 2:
                - Morning: [specific activity]
                - Afternoon: [specific activity]
                - Evening: [specific activity]

                [Continue for requested number of days]

            3. For GENERAL TRAVEL ADVICE:
                Provide helpful, structured advice with clear sections and bullet points.

            4. For FOLLOW-UP QUESTIONS:
                Use the conversation history to provide contextual responses. If the user asks about something mentioned earlier, reference that context.

            IMPORTANT:
            - Always match the response format to the user's request type
            - For itineraries, include specific activities, attractions, and timing
            - For suggestions, focus on destination highlights and unique experiences
            - For follow-ups, maintain context from previous messages
            - Keep responses focused and structured
            - Do not engage in open-ended conversation
            - If location information is provided, prioritize that location in your responses
            - Do not mention any references to information available, or tool usage. You have to only generate itinerary plan, no additional text is required.

            USER QUERY:"""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("Context: {context}\n\nUser Query: {input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create the agent
        agent = OpenAIFunctionsAgent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        # Create the agent executor with memory
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,  # Increased from 3
            early_stopping_method="force",  # Ensure completion even if tools fail
            return_intermediate_steps=True,  # Important: Return the tool calls
            agent_executor_kwargs={
                "force_tool_use": True,  # Strongly encourage tool usage
                "enforce_response_schemas": True  # Ensure proper response formats
            }
        )
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the MCP server."""
        uvicorn.run(self.app, host=host, port=port)
    
    def run_async(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the MCP server asynchronously."""
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        
        # Use asyncio to run the server
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

if __name__ == "__main__":
    server = MCPServer()
    server.run() 