"""
Start script for the AI Travel Planner.
This script starts the Streamlit app with local Llama API integration and OpenRouter fallback.
"""

import subprocess
import sys
import os
import time
from dotenv import load_dotenv

def check_environment():
    """Check if required environment variables are set."""
    load_dotenv()
    
    # Check for Llama API URL
    llama_api_url = os.getenv('LLAMA_API_URL', 'http://localhost:8080')
    print(f"🔗 Llama API URL: {llama_api_url}")
    
    # Check for OpenRouter API key (optional fallback)
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    if openrouter_api_key:
        print("✅ OpenRouter API key configured (fallback)")
    else:
        print("⚠️  OpenRouter API key not set - will use Local Llama API only")
        print("   Get a free API key from: https://openrouter.ai/ for fallback support")
    
    # Check for RapidAPI key (optional but recommended)
    rapid_api_key = os.getenv('RAPID_API_KEY')
    if not rapid_api_key:
        print("⚠️  RAPID_API_KEY not set - flight and hotel search features will be limited")
        print("   Get a free API key from: https://rapidapi.com/")
    else:
        print("✅ RapidAPI key configured")
    
    # Check for site configuration
    site_url = os.getenv('SITE_URL', 'http://localhost:8501')
    site_name = os.getenv('SITE_NAME', 'AI Travel Planner')
    print(f"🌐 Site URL: {site_url}")
    print(f"📝 Site Name: {site_name}")
    
    # Check if at least one LLM service is available
    if not openrouter_api_key and llama_api_url == 'http://localhost:8080':
        print("⚠️  Warning: No LLM service configured!")
        print("   Please set either OPENROUTER_API_KEY or ensure Local Llama API is running")

def main():
    """Main function to start the services."""
    print("🚀 Starting AI Travel Planner...")
    
    # Check environment
    check_environment()
    
    try:
        # Start Streamlit app
        streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
        streamlit_process = subprocess.Popen(streamlit_cmd)
        
        print("✅ Started Streamlit app on http://localhost:8501")
        print("🤖 Using Local Llama API with OpenRouter fallback")
        print("\nPress Ctrl+C to stop the application")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping application...")
        streamlit_process.terminate()
        print("✅ Application stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 