# Image Upload and GeoCLIP API Integration

## Overview
The AI Travel Planner now includes an image upload feature that allows users to identify locations from images using the GeoCLIP API. This feature automatically detects travel destinations from uploaded photos and integrates the location information into the travel planning process.

## Features

### 🖼️ Image Upload
- **Supported Formats**: JPG, JPEG, PNG
- **File Size**: Handled by Streamlit's file uploader
- **Location**: Separate section below conversation management buttons

### 🔍 Location Identification
- **API Integration**: Uses GeoCLIP API for location detection
- **Configurable Endpoint**: Set via `GEOCLIP_API_URL` environment variable
- **Response Data**: Location name, coordinates, and confidence score

### 🎯 Location Integration
- **Session State**: Location information stored in session state
- **Auto-population**: Destination fields pre-filled with identified location
- **Context Enhancement**: Location information added to LLM prompts as "location_information"

## Configuration

### Environment Variables
```bash
# GeoCLIP API endpoint (default: http://localhost:8000/predict)
GEOCLIP_API_URL=http://localhost:8000/predict
```

### Dependencies
The following packages have been added to `requirements.txt`:
```
opencv-python>=4.8.0
```

## Usage

### 1. Upload Image
1. Navigate to the "📸 Upload Location Image (Optional)" section
2. Click "Choose an image to identify location"
3. Select an image file (JPG, JPEG, or PNG)

### 2. Identify Location
1. After uploading an image, click "🔍 Identify Location"
2. The system will send the image to the GeoCLIP API
3. Location information will be displayed if found

### 3. Use Location in Planning
1. Click "🎯 Use This Location" to confirm
2. The location will be automatically used in:
   - Travel suggestions (pre-populates the input field)
   - Detailed itinerary creation (pre-fills destination field)
   - Follow-up questions (includes location context)

## API Integration

### GeoCLIP API Request
```python
files = {
    "file": (
        uploaded_file.name,
        uploaded_file.getvalue(),
        uploaded_file.type or "image/jpeg"
    )
}
response = requests.post(GEOCLIP_API_URL, files=files, timeout=30)
```

### GeoCLIP API Response
```json
[
    {
        "location_name": "Paris, France",
        "latitude": 48.8566,
        "longitude": 2.3522,
        "confidence": 0.95
    }
]
```

### Location Information in LLM Context
```python
context["location_information"] = {
    "location_name": "Paris, France",
    "latitude": 48.8566,
    "longitude": 2.3522,
    "confidence": 0.95
}
```

## Session State Management

### New Variables Added
```python
# Image upload state
st.session_state.uploaded_image = None
st.session_state.location_info = None
```

### State Clearing
- **New Conversation**: Clears both image and location data
- **Clear History**: Keeps image and location data intact
- **Session Reset**: All data cleared when starting fresh

## Error Handling

### Connection Errors
- **API Unavailable**: Shows error message with instructions
- **Timeout**: 30-second timeout for API requests
- **Network Issues**: Graceful error handling with user feedback

### Processing Errors
- **Invalid Images**: Handled by Streamlit's file uploader
- **No Location Found**: Warning message with suggestions
- **API Errors**: Detailed error messages from API response

## Testing

### Test Script
Run the integration test:
```bash
python test_geoclip_integration.py
```

### Manual Testing
1. Start the GeoCLIP API service
2. Upload an image of a known location
3. Verify location identification
4. Test integration with travel planning features

## Benefits

### User Experience
- **Visual Input**: Users can upload photos instead of typing location names
- **Automatic Detection**: No need to know exact location names
- **Context Enhancement**: Better travel suggestions with location context

### Technical Benefits
- **Modular Design**: Separate from main travel planning logic
- **Configurable**: Easy to change API endpoints
- **Robust**: Comprehensive error handling
- **Session-aware**: Maintains state across interactions

## Troubleshooting

### Common Issues

1. **GeoCLIP API Not Running**
   - Error: "Cannot connect to GeoCLIP API"
   - Solution: Start the GeoCLIP API service

2. **No Location Found**
   - Error: "No location found in the image"
   - Solution: Try a different image with clearer landmarks

3. **Image Upload Issues**
   - Error: File format not supported
   - Solution: Use JPG, JPEG, or PNG format

4. **API Timeout**
   - Error: Request timeout
   - Solution: Check API service performance

### Debug Steps
1. Run the test script: `python test_geoclip_integration.py`
2. Check environment variables: `echo $GEOCLIP_API_URL`
3. Verify API service: `curl http://localhost:8000/health`
4. Test with sample image

## Future Enhancements

### Potential Improvements
- **Multiple Location Support**: Handle images with multiple locations
- **Confidence Thresholds**: Filter results by confidence score
- **Image Preprocessing**: Enhance image quality before API call
- **Caching**: Cache results for repeated images
- **Batch Processing**: Handle multiple images at once

### Integration Opportunities
- **Map Integration**: Display location on interactive map
- **Weather Integration**: Get weather data for identified location
- **Photo Gallery**: Store and manage uploaded images
- **Social Features**: Share identified locations with others 