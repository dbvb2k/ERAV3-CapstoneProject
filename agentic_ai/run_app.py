"""
Main entry point for the AI Travel Planner application.
"""

import os
import sys
import asyncio
import nest_asyncio
import streamlit.web.bootstrap as bootstrap
import warnings

def setup_environment():
    """Set up the environment for the application."""
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Set environment variables
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    
    # Initialize multiprocessing method for Windows
    if sys.platform == "win32":
        import multiprocessing
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    # Configure PyTorch
    try:
        import torch
        import torch.backends.cudnn as cudnn
        # Disable the PyTorch class path warning
        torch._C._log_api_usage_once = lambda *args, **kwargs: None
        # Configure CUDA if available
        if torch.cuda.is_available():
            cudnn.benchmark = True
    except ImportError:
        pass  # PyTorch not installed
    
    # Set Streamlit configuration
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'

def run_streamlit():
    """Run the Streamlit application."""
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Add the current directory to Python path
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    # Initialize event loop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Apply nest_asyncio
    try:
        nest_asyncio.apply()
    except Exception:
        pass
    
    # Run the Streamlit app with all required arguments
    script_path = os.path.join(script_dir, "app.py")
    bootstrap.run(
        main_script_path=script_path,
        is_hello=False,
        args=[],
        flag_options={
            "server.port": 8501,
            "server.headless": True
        }
    )

if __name__ == "__main__":
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("Please set it before running the application")
        print("You can get an API key from https://openrouter.ai/")
        sys.exit(1)
    
    try:
        # Set up environment
        setup_environment()
        
        # Run the application
        run_streamlit()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1) 