from .text_processor import TextProcessor
from .pdf_processor import PDFProcessor
from .image_processor import ImageProcessor
from .video_processor import VideoProcessor
from .audio_processor import AudioProcessor
from ..models import InputType, InputProcessor

class ProcessorFactory:
    """Factory for creating input processors"""
    
    def __init__(self):
        self.processors = {
            InputType.TEXT: TextProcessor(),
            InputType.PDF: PDFProcessor(),
            InputType.IMAGE: ImageProcessor(),
            InputType.VIDEO: VideoProcessor(),
            InputType.AUDIO: AudioProcessor()
        }
    
    def get_processor(self, input_type: InputType) -> InputProcessor:
        """Get appropriate processor for input type"""
        return self.processors.get(input_type)
    
    def get_all_processors(self):
        """Get all available processors"""
        return self.processors