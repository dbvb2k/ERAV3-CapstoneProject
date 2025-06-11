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
from .openrouter_llm import OpenRouterLLM
import uvicorn
import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import json

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

class MCPServer:
    def __init__(self):
        self.app = FastAPI(title="Travel Planner MCP Server")
        self.tools = {}
        self.agent_executor = None
        self.setup_routes()
        
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
                # Convert request to dict with proper serialization
                request_dict = request.model_dump()
                result = await self.agent_executor.ainvoke({
                    "input": request_dict["query"],
                    "context": request_dict["context"] or {}
                })
                return {"status": "success", "result": result}
            except Exception as e:
                error_msg = str(e)
                if hasattr(e, 'detail'):
                    error_msg = e.detail
                raise HTTPException(status_code=500, detail={"error": error_msg})

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
When suggesting destinations, ALWAYS return EXACTLY 2 suggestions in this format:
* [Destination Name] for [brief description of culture and food highlights]

For example:
* Tokyo, Japan for its blend of modern technology and traditional culture, featuring world-class sushi and ramen
* Barcelona, Spain for its stunning Gaudi architecture, vibrant tapas scene, and Mediterranean charm

Keep responses focused and structured. Do not engage in open-ended conversation."""),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create the agent
        agent = OpenAIFunctionsAgent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        # Create the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the MCP server."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    server = MCPServer()
    server.run() 