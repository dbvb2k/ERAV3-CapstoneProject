import PyPDF2
import io
from typing import Any
from ..models import InputProcessor, ProcessedInput, InputType
from .text_processor import BaseProcessor

class PDFProcessor(BaseProcessor):
    """Process PDF files"""
    
    def supports_input_type(self, input_type: InputType) -> bool:
        return input_type == InputType.PDF
    
    async def process(self, input_data: Any) -> ProcessedInput:
        """Process PDF file and extract text"""
        try:
            # Handle file-like object or bytes
            if hasattr(input_data, 'read'):
                pdf_file = input_data
            else:
                pdf_file = io.BytesIO(input_data)
            
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = ""
            
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            if not text_content.strip():
                text_content = "No text could be extracted from the PDF"
            
            # Process extracted text
            language = self.detect_language(text_content)
            translated_text = self.translate_text(text_content, "en")
            entities = self.extract_travel_entities(translated_text)
            
            return ProcessedInput(
                input_type=InputType.PDF,
                content=translated_text,
                metadata={
                    "original_text": text_content,
                    "num_pages": len(pdf_reader.pages),
                    "detected_language": language
                },
                language=language,
                confidence=0.8,
                extracted_entities=entities
            )
            
        except Exception as e:
            return ProcessedInput(
                input_type=InputType.PDF,
                content=f"Error processing PDF: {str(e)}",
                metadata={"error": str(e)},
                language="en",
                confidence=0.0,
                extracted_entities={}
            )
