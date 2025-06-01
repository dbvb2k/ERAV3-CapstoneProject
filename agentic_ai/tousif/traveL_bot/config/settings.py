import os
from dotenv import load_dotenv
from typing import Dict, List

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the Travel Advisory Bot"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    
    # Model Configuration
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "microsoft/phi-2")
    HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "microsoft/phi-2")
    
    # Application Settings
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    SUPPORTED_LANGUAGES = os.getenv("SUPPORTED_LANGUAGES", "en,es,fr,de,it,pt,zh,ja,ko,ar,hi,ru").split(",")
    DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
    
    # File Upload Settings
    UPLOAD_FOLDER = "data/uploads"
    ALLOWED_EXTENSIONS = {
        'pdf': ['.pdf'],
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
        'video': ['.mp4', '.avi', '.mov', '.wmv', '.flv'],
        'audio': ['.mp3', '.wav', '.ogg', '.m4a', '.flac']
    }
    
    # Travel Planning Settings
    MAX_ITINERARY_DAYS = 30
    DEFAULT_CURRENCY = "USD"
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate configuration settings"""
        return {
            "openai_configured": bool(cls.OPENAI_API_KEY),
            "huggingface_configured": bool(cls.HUGGINGFACE_API_KEY),
            "upload_folder_exists": os.path.exists(cls.UPLOAD_FOLDER)
        }
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available AI models"""
        models = []
        if cls.OPENAI_API_KEY:
            models.extend(["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
        models.append("huggingface/" + cls.HUGGINGFACE_MODEL)
        return models
