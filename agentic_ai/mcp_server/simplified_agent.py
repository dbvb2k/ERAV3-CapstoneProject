import os
from typing import Dict, List, Any
import uuid
from datetime import datetime
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from simplified_tools import get_langchain_tools

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define request models
class AgentRequest(BaseModel):
    query: str
    context: Dict = {}
    session_id: str = None
    conversation_history: List = []

class SimplifiedAgent:
    """
    Simplified agent that uses LangChain tools to handle travel queries.
    """
    
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()
        self.agent_executor = None
        self.tools = get_langchain_tools()
        self.sessions = {}
        
        # Set up the agent
        self.setup_agent()
        
    def setup_agent(self):
        """Set up the agent with tools."""
        from langchain.chat_models import ChatOpenAI
        
        # Use OpenAI API (you can replace with your custom LLM client)
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model_name="openai/gpt-3.5-turbo", # Can be replaced with your model
            temperature=0.2,
        )
        
        # Create a stronger prompt that enforces tool usage
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a travel planning assistant with access to real-time tools.

TOOL USAGE REQUIREMENTS:
1. ALWAYS use FlightSearchTool when asked about flights between cities
2. ALWAYS use HotelSearchTool when asked about accommodations
3. ALWAYS use WeatherTool when asked about weather conditions
4. ALWAYS use ItineraryPlannerTool to create travel itineraries

DO NOT make up information about flights, hotels, or weather - ONLY use the tools to get real data.
When generating itineraries, include SPECIFIC information from the other tools.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent with the tools
        agent = create_openai_tools_agent(llm, self.tools, prompt)
        
        # Create the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,
            return_intermediate_steps=True
        )
        
        logger.info("Agent initialized with tools")
    
    def setup_routes(self):
        @self.app.post("/agent/execute")
        async def execute_agent(request: AgentRequest):
            """Execute the agent with a user query."""
            if not self.agent_executor:
                raise HTTPException(status_code=500, detail={"error": "Agent not initialized"})
            
            try:
                # Get or create session
                session_id = request.session_id or str(uuid.uuid4())
                if session_id not in self.sessions:
                    self.sessions[session_id] = {
                        "memory": ConversationBufferMemory(return_messages=True),
                        "context": {},
                        "history": [],
                        "created_at": datetime.now()
                    }
                session = self.sessions[session_id]
                
                # Update session context
                if request.context:
                    session["context"].update(request.context)
                
                # Add user message to history
                session["history"].append({"role": "user", "content": request.query})
                
                # Add travel tool hints to query
                enhanced_query = request.query
                if "flight" in request.query.lower() or "travel" in request.query.lower():
                    enhanced_query = "[Use FlightSearchTool] " + enhanced_query
                if "hotel" in request.query.lower() or "stay" in request.query.lower():
                    enhanced_query = "[Use HotelSearchTool] " + enhanced_query
                if "weather" in request.query.lower():
                    enhanced_query = "[Use WeatherTool] " + enhanced_query
                if "itinerary" in request.query.lower() or "plan" in request.query.lower():
                    enhanced_query = "[Use ItineraryPlannerTool] " + enhanced_query
                
                # Create the input for the agent
                agent_input = {
                    "input": enhanced_query,
                    "chat_history": session["memory"].chat_memory.messages,
                    "context": str(session["context"])
                }
                
                # Execute the agent
                logger.info(f"Executing agent with query: {request.query}")
                result = await self.agent_executor.ainvoke(agent_input)
                
                # Extract tool calls
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
                
                # Add assistant response to history
                if "output" in result:
                    session["history"].append({"role": "assistant", "content": result["output"]})
                    session["memory"].chat_memory.add_user_message(request.query)
                    session["memory"].chat_memory.add_ai_message(result["output"])
                
                return {
                    "status": "success",
                    "result": result,
                    "tool_calls": tool_calls,
                    "session_id": session_id,
                    "conversation_history": session["history"]
                }
                
            except Exception as e:
                logger.error(f"Error in agent execution: {str(e)}")
                raise HTTPException(status_code=500, detail={"error": str(e)})
    
    def run(self, host="0.0.0.0", port=8000):
        """Run the FastAPI server."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

# Create and run the agent when this file is executed directly
if __name__ == "__main__":
    agent = SimplifiedAgent()
    agent.run()