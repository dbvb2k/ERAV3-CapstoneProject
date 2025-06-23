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
from .openrouter_llm import OpenRouterLLM
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
                
                # Add user message to conversation history
                session.add_message("user", request.query)
                
                # Update session context with new information
                if request.context:
                    session.context.update(request.context)
                
                # Prepare input with conversation history
                memory_variables = session.memory.load_memory_variables({})
                chat_history = memory_variables.get("chat_history", [])
                
                # Convert request to dict with proper serialization
                request_dict = request.model_dump()
                
                # Create input with conversation context
                agent_input = {
                    "input": request_dict["query"],
                    "context": session.context,
                    "chat_history": chat_history
                }
                
                result = await self.agent_executor.ainvoke(agent_input)
                
                # Add assistant response to conversation history
                if "output" in result:
                    session.add_message("assistant", result["output"])
                
                return {
                    "status": "success", 
                    "result": result,
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

    def setup_agent(self, tools):
        """Set up the LangChain agent with OpenRouter LLM."""
        # Initialize OpenRouter LLM
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
        llm = OpenRouterLLM(
            api_key=openrouter_api_key,
            model="meta-llama/llama-3.3-8b-instruct:free",
            # model="mistralai/mistral-7b-instruct",  # Using a smaller, cheaper model
            temperature=0.7,
            max_tokens=2000,
            site_url=os.getenv('SITE_URL', 'http://localhost:8501'),
            site_name=os.getenv('SITE_NAME', 'AI Travel Planner')
        )
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful AI travel assistant that helps plan trips and create itineraries.

RESPONSE FORMATS:

1. For DESTINATION SUGGESTIONS (when user asks for general travel ideas):
   Return EXACTLY 2 suggestions in this format:
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
- Do not engage in open-ended conversation"""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
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
            tools=tools,            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
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