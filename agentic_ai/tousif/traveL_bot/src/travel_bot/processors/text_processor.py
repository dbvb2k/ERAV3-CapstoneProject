import asyncio
from typing import Any, Dict

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    detect = None

try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False
    Translator = None

from ..models import InputProcessor, ProcessedInput, InputType

class BaseProcessor(InputProcessor):
    """Base processor with common functionality"""
    
    def __init__(self):
        if GOOGLETRANS_AVAILABLE:
            self.translator = Translator()
        else:
            self.translator = None
            print("Warning: Google Translate not available. Translation features will be limited.")
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        if not LANGDETECT_AVAILABLE:
            return "en"  # Default to English if langdetect not available
        try:
            return detect(text)
        except:
            return "en"  # Default to English
    
    def translate_text(self, text: str, target_lang: str = "en") -> str:
        """Translate text to target language"""
        if not GOOGLETRANS_AVAILABLE or self.translator is None:
            return text  # Return original text if translator not available
        try:
            if self.detect_language(text) != target_lang:
                translated = self.translator.translate(text, dest=target_lang)
                return translated.text
            return text
        except:
            return text  # Return original if translation fails
    
    def extract_travel_entities(self, text: str) -> Dict[str, Any]:
        """Extract travel-related entities from text"""
        # Simple keyword-based extraction (can be enhanced with NER models)
        travel_keywords = {
            'destinations': ['city', 'country', 'beach', 'mountain', 'hotel', 'resort'],
            'activities': ['hiking', 'sightseeing', 'museum', 'restaurant', 'shopping', 'adventure'],
            'time_expressions': ['days', 'weeks', 'months', 'weekend', 'vacation'],
            'budget_terms': ['budget', 'cheap', 'expensive', 'luxury', 'affordable', 'cost']
        }
        
        entities = {}
        text_lower = text.lower()
        
        for category, keywords in travel_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                entities[category] = found_keywords
        
        return entities

class TextProcessor(BaseProcessor):
    """Process text input"""
    
    def supports_input_type(self, input_type: InputType) -> bool:
        return input_type == InputType.TEXT
    
    async def process(self, input_data: str) -> ProcessedInput:
        """Process text input"""
        language = self.detect_language(input_data)
        translated_text = self.translate_text(input_data, "en")
        entities = self.extract_travel_entities(translated_text)
        
        return ProcessedInput(
            input_type=InputType.TEXT,
            content=translated_text,
            metadata={"original_text": input_data, "detected_language": language},
            language=language,
            confidence=0.9,
            extracted_entities=entities
        )
