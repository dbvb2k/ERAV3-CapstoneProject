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
        
        return dates if dates else None
        
    def setup_agent(self):
        """Set up the agent with tools using FallbackLLM."""
        # Initialize FallbackLLM with debug mode
        llm = FallbackLLM(
            temperature=0.2,
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
            max_iterations=5,
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
                
                # Extract and add travel dates if present
                travel_dates = self._extract_travel_dates(request.query)
                if travel_dates and "context" in agent_input:
                    agent_input["travel_dates"] = [d.strftime("%Y-%m-%d") for d in travel_dates]
                    if isinstance(agent_input["context"], str):
                        agent_input["context"] += f"\nTravel dates: {', '.join(agent_input['travel_dates'])}"
                
                # Debug information
                logger.info(f"Executing agent with query: {request.query}")
                logger.info(f"Agent input: {json.dumps(agent_input, default=str)}")
                
                # Add debug for agent execution
                print("==== EXECUTING AGENT WITH TOOLS ====")
                for tool in self.tools:
                    print(f"Available tool: {tool.name} - {tool.description.split('.')[0]}")
                print("=================================")

                result = await self.agent_executor.ainvoke(agent_input)

                # Debug the result
                print(f"==== AGENT RESULT ====")
                print(f"Intermediate steps: {len(result.get('intermediate_steps', []))}")
                tool_calls = []
                for i, step in enumerate(result.get('intermediate_steps', [])):
                    logger.info(f"Intermediate steps: {len(result['intermediate_steps'])}")
                    if len(step) >= 2:
                        action, output = step
                        print(f"Step {i+1}: Tool {action.tool} called with inputs {action.tool_input}")
                        print(f"Output: {output}")
                        tool_calls.append({
                                "tool": action.tool,
                                "input": action.tool_input,
                                "output": output
                            })
                print("======================")                            
                
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