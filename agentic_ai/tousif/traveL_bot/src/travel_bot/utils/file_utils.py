import os
import mimetypes
from pathlib import Path
from typing import Optional
from travel_bot.models import InputType
from config.settings import Config

class FileUtils:
    """Utility functions for file handling"""
    
    def __init__(self):
        self.allowed_extensions = Config.ALLOWED_EXTENSIONS
        self.max_file_size = Config.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
    
    def detect_file_type(self, file_path: str) -> Optional[InputType]:
        """Detect input type based on file extension"""
        file_extension = Path(file_path).suffix.lower()
        
        for input_type, extensions in self.allowed_extensions.items():
            if file_extension in extensions:
                return InputType(input_type)
        
        return None
    
    def is_file_allowed(self, filename: str) -> bool:
        """Check if file type is allowed"""
        file_extension = Path(filename).suffix.lower()
        
        for extensions in self.allowed_extensions.values():
            if file_extension in extensions:
                return True
        
        return False
    
    def is_file_size_valid(self, file_size: int) -> bool:
        """Check if file size is within limits"""
        return file_size <= self.max_file_size
    
    def get_file_info(self, file_path: str) -> dict:
        """Get comprehensive file information"""
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        file_stat = os.stat(file_path)
        file_path_obj = Path(file_path)
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        
        return {
            "filename": file_path_obj.name,
            "extension": file_path_obj.suffix.lower(),
            "size_bytes": file_stat.st_size,
            "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
            "mime_type": mime_type,
            "input_type": self.detect_file_type(file_path),
            "is_allowed": self.is_file_allowed(file_path_obj.name),
            "is_size_valid": self.is_file_size_valid(file_stat.st_size)
        }
    
    def create_upload_directory(self) -> str:
        """Create upload directory if it doesn't exist"""
        upload_dir = Config.UPLOAD_FOLDER
        os.makedirs(upload_dir, exist_ok=True)
        return upload_dir
    
    def save_uploaded_file(self, uploaded_file, filename: str) -> str:
        """Save uploaded file to upload directory"""
        upload_dir = self.create_upload_directory()
        file_path = os.path.join(upload_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    def cleanup_temp_files(self, file_paths: list):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {file_path}: {e}")
    
    def get_supported_formats_display(self) -> str:
        """Get human-readable supported formats"""
        format_strings = []
        for file_type, extensions in self.allowed_extensions.items():
            ext_list = ", ".join(extensions)
            format_strings.append(f"{file_type.title()}: {ext_list}")
        
        return " | ".join(format_strings)