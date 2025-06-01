try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    BlipProcessor = None
    BlipForConditionalGeneration = None

import io
from typing import Any
from ..models import InputProcessor, ProcessedInput, InputType
from .text_processor import BaseProcessor

class ImageProcessor(BaseProcessor):
    """Process image files using BLIP model for image captioning"""
    
    def __init__(self):
        super().__init__()
        if not PIL_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            print("Warning: PIL or Transformers not available. Image processing will be limited.")
            self.model_loaded = False
            return
            
        try:
            # Load BLIP model for image captioning
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load BLIP model: {e}")
            self.model_loaded = False
    
    def supports_input_type(self, input_type: InputType) -> bool:
        return input_type == InputType.IMAGE
    
    async def process(self, input_data: Any) -> ProcessedInput:
        """Process image and generate description"""
        try:
            # Handle different input formats
            if hasattr(input_data, 'read'):
                image = Image.open(input_data)
            elif isinstance(input_data, bytes):
                image = Image.open(io.BytesIO(input_data))
            else:
                image = Image.open(input_data)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            description = ""
            if self.model_loaded:
                # Generate image caption using BLIP
                inputs = self.processor(image, return_tensors="pt")
                
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_length=50)
                    description = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            else:
                description = "Image uploaded (caption generation unavailable)"
            
            # Add travel-specific prompts for better context
            travel_context = f"Travel image: {description}. This appears to be related to travel, tourism, or destinations."
            
            # Extract entities from description
            entities = self.extract_travel_entities(description)
            
            return ProcessedInput(
                input_type=InputType.IMAGE,
                content=travel_context,
                metadata={
                    "image_description": description,
                    "image_size": image.size,
                    "image_mode": image.mode
                },
                language="en",
                confidence=0.7 if self.model_loaded else 0.3,
                extracted_entities=entities
            )
            
        except Exception as e:
            return ProcessedInput(
                input_type=InputType.IMAGE,
                content=f"Error processing image: {str(e)}",
                metadata={"error": str(e)},
                language="en",
                confidence=0.0,
                extracted_entities={}
            )
