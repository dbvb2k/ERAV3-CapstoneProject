import os
from typing import Dict, Any
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "Place Identifier API"
    API_VERSION: str = "1.0.0"
    
    # Model Settings
    DEFAULT_NUM_PREDICTIONS: int = 3
    MIN_CONFIDENCE_THRESHOLD: float = 0.0
    
    # Geocoding Settings
    GEOCODER_USER_AGENT: str = "place_identifier_api"
    GEOCODER_TIMEOUT: int = 10
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Export settings
__all__ = ["settings"] 