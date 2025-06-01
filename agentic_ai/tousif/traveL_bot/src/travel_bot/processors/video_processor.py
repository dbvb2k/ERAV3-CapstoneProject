import cv2
import numpy as np
from typing import Any, List
from ..models import InputProcessor, ProcessedInput, InputType
from .text_processor import BaseProcessor
from .image_processor import ImageProcessor

class VideoProcessor(BaseProcessor):
    """Process video files by extracting keyframes and analyzing them"""
    
    def __init__(self):
        super().__init__()
        self.image_processor = ImageProcessor()
    
    def supports_input_type(self, input_type: InputType) -> bool:
        return input_type == InputType.VIDEO
    
    def extract_keyframes(self, video_path: str, max_frames: int = 5) -> List[np.ndarray]:
        """Extract keyframes from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Extract frames at regular intervals
            frame_interval = max(1, total_frames // max_frames)
            
            for i in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                if len(frames) >= max_frames:
                    break
            
            cap.release()
            return frames, {"duration": duration, "fps": fps, "total_frames": total_frames}
            
        except Exception as e:
            print(f"Error extracting keyframes: {e}")
            return [], {"error": str(e)}
    
    async def process(self, input_data: Any) -> ProcessedInput:
        """Process video by analyzing keyframes"""
        try:
            # Save uploaded video temporarily if it's not a file path
            if hasattr(input_data, 'name'):
                video_path = input_data.name
            else:
                # For uploaded files, we'd need to save temporarily
                video_path = "temp_video.mp4"  # This would need proper temp file handling
            
            # Extract keyframes
            keyframes, video_metadata = self.extract_keyframes(video_path)
            
            if not keyframes:
                return ProcessedInput(
                    input_type=InputType.VIDEO,
                    content="Could not extract frames from video",
                    metadata=video_metadata,
                    language="en",
                    confidence=0.0,
                    extracted_entities={}
                )
            
            # Analyze each keyframe
            frame_descriptions = []
            all_entities = {}
            
            for i, frame in enumerate(keyframes):
                try:
                    # Convert frame to PIL Image for processing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    from PIL import Image
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Process frame as image
                    frame_result = await self.image_processor.process(pil_image)
                    frame_descriptions.append(frame_result.content)
                    
                    # Merge entities
                    for key, values in frame_result.extracted_entities.items():
                        if key not in all_entities:
                            all_entities[key] = []
                        all_entities[key].extend(values)
                    
                except Exception as e:
                    frame_descriptions.append(f"Error processing frame {i}: {str(e)}")
            
            # Combine all frame descriptions
            combined_description = f"Video analysis with {len(keyframes)} keyframes: " + " | ".join(frame_descriptions)
            
            return ProcessedInput(
                input_type=InputType.VIDEO,
                content=combined_description,
                metadata={
                    "video_info": video_metadata,
                    "num_keyframes": len(keyframes),
                    "frame_descriptions": frame_descriptions
                },
                language="en",
                confidence=0.6,
                extracted_entities=all_entities
            )
            
        except Exception as e:
            return ProcessedInput(
                input_type=InputType.VIDEO,
                content=f"Error processing video: {str(e)}",
                metadata={"error": str(e)},
                language="en",
                confidence=0.0,
                extracted_entities={}
            )
