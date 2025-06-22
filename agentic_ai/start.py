"""
Start script for the AI Travel Planner.
This script starts both the MCP server and Streamlit app.
"""

import subprocess
import sys
import os
import time
from dotenv import load_dotenv

def check_environment():
    """Check if OpenRouter API key is set."""
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("Please set it before running the application")
        print("You can get an API key from https://openrouter.ai/")
        sys.exit(1)

def main():
    """Main function to start the services."""
    # Load environment variables
    load_dotenv()
    
    # Check environment
    check_environment()
    
    print("🚀 Starting AI Travel Planner...")
    
    try:
        # Start Streamlit app
        streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
        streamlit_process = subprocess.Popen(streamlit_cmd)
        
        print("✅ Started Streamlit app on http://localhost:8501")
        print("\nPress Ctrl+C to stop all services")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping services...")
        streamlit_process.terminate()
        print("✅ All services stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 