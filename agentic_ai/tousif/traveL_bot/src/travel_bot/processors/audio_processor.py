try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

import io
import tempfile
import os
from typing import Any
from ..models import InputProcessor, ProcessedInput, InputType
from .text_processor import BaseProcessor

class AudioProcessor(BaseProcessor):
    """Process audio files using Whisper for speech-to-text"""
    
    def __init__(self):
        super().__init__()
        if not WHISPER_AVAILABLE:
            print("Warning: Whisper not available. Audio processing will be limited.")
            self.model_loaded = False
            return
            
        try:
            # Load Whisper model (using base model for speed)
            self.model = whisper.load_model("base")
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load Whisper model: {e}")
            self.model_loaded = False
    
    def supports_input_type(self, input_type: InputType) -> bool:
        return input_type == InputType.AUDIO
    
    async def process(self, input_data: Any) -> ProcessedInput:
        """Process audio file and convert to text"""
        try:
            if not self.model_loaded:
                return ProcessedInput(
                    input_type=InputType.AUDIO,
                    content="Audio processing unavailable (Whisper model not loaded)",
                    metadata={"error": "Whisper model not available"},
                    language="en",
                    confidence=0.0,
                    extracted_entities={}
                )
            
            # Create temporary file for audio processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                if hasattr(input_data, 'read'):
                    temp_file.write(input_data.read())
                else:
                    temp_file.write(input_data)
                temp_file_path = temp_file.name
            
            try:
                # Transcribe audio using Whisper
                result = self.model.transcribe(temp_file_path)
                transcribed_text = result["text"]
                detected_language = result.get("language", "en")
                
                # Clean up temporary file
                os.unlink(temp_file_path)
                
                # Translate if needed
                translated_text = self.translate_text(transcribed_text, "en")
                
                # Extract travel entities
                entities = self.extract_travel_entities(translated_text)
                
                return ProcessedInput(
                    input_type=InputType.AUDIO,
                    content=translated_text,
                    metadata={
                        "original_text": transcribed_text,
                        "detected_language": detected_language,
                        "transcription_confidence": result.get("confidence", 0.0)
                    },
                    language=detected_language,
                    confidence=0.8,
                    extracted_entities=entities
                )
                
            except Exception as e:
                # Clean up temporary file on error
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                raise e
                
        except Exception as e:
            return ProcessedInput(
                input_type=InputType.AUDIO,
                content=f"Error processing audio: {str(e)}",
                metadata={"error": str(e)},
                language="en",
                confidence=0.0,
                extracted_entities={}
            )