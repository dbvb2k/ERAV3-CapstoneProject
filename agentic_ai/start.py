"""
Start script for the AI Travel Planner application.
"""
import os
import subprocess
import threading
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def start_streamlit():
    """Start the Streamlit application."""
    print("🌐 Starting Streamlit app...")
    subprocess.run(["streamlit", "run", "app.py"])

def start_agent():
    """Start the simplified agent server."""
    print("🤖 Starting AI Agent...")
    subprocess.run(["python", "simplified_agent.py"])

if __name__ == "__main__":
    print("🚀 Starting AI Travel Planner...")
    
    # Display configuration
    llama_api_url = os.getenv('LLAMA_API_URL', 'http://localhost:8080')
    site_url = os.getenv('SITE_URL', 'http://localhost:8501')
    site_name = os.getenv('SITE_NAME', 'AI Travel Planner')
    
    print(f"🔗 Llama API URL: {llama_api_url}")
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    if openrouter_api_key:
        print("✅ OpenRouter API key configured (fallback)")
    
    rapidapi_key = os.getenv('RAPID_API_KEY')
    if rapidapi_key:
        print("✅ RapidAPI key configured")
    else:
        print("❌ RapidAPI key not found. Flight and hotel data will be unavailable.")
    
    print(f"🌐 Site URL: {site_url}")
    print(f"📝 Site Name: {site_name}")
    
    # Start the agent server in a separate thread
    agent_thread = threading.Thread(target=start_agent)
    agent_thread.daemon = True
    agent_thread.start()
    
    # Wait a moment for the agent to start
    time.sleep(2)
    
    # Start Streamlit app in the main thread
    print("✅ Started Streamlit app on http://localhost:8501")
    print("🤖 Using Local Llama API with OpenRouter fallback")
    print("\nPress Ctrl+C to stop the application")
    
    start_streamlit()