# Location-Based Travel Planning System

A comprehensive system that combines image-based location identification with an AI-powered travel planning chatbot. The system can identify locations from images and generate detailed travel itineraries based on user preferences.

## Features

### Location Identification API
- Image-based location identification using [GeoCLIP](https://github.com/VicenteVivan/geo-clip) model
- Multiple location predictions with confidence scores
- Detailed address information using reverse geocoding
- RESTful API with Swagger documentation
- Health check endpoint
- Configurable settings via environment variables
- Comprehensive error handling and logging

### Travel Planning Chatbot
- Dynamic itinerary generation based on user preferences
- Support for flexible trip durations (3+ days)
- Detailed daily schedules with specific timings
- Flight information from specified departure city
- Hotel recommendations with pricing
- Local transportation options
- Restaurant recommendations
- Budget breakdown in local currency
- Travel tips and local customs information
- Comprehensive logging of all interactions
- Powered by Microsoft's Phi-2 model for efficient CPU-based inference

## Prerequisites

- Python 3.12+
- Hugging Face Transformers library
- No GPU required - optimized for CPU usage with Microsoft's Phi-2 model

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd place-identifier-api
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and modify as needed:
```bash
cp .env.example .env
```

## Environment Variables

Required environment variables in your `.env` file:

```env
# API Keys (Optional)
OPENAI_API_KEY=your_openai_key  # Optional for GPT models
HUGGINGFACE_API_KEY=your_huggingface_key  # Optional for some HF models
GOOGLE_PLACES_API_KEY=your_google_key  # Optional for enhanced location data
WEATHER_API_KEY=your_weather_key  # Optional for weather information

# Model Configuration
DEFAULT_LLM_MODEL=microsoft/phi-2  # Default model
HUGGINGFACE_MODEL=microsoft/phi-2  # Can be changed to other HF models

# Application Settings
MAX_FILE_SIZE_MB=50
SUPPORTED_LANGUAGES=en,es,fr,de,it,pt,zh,ja,ko,ar,hi,ru
DEFAULT_LANGUAGE=en
```

## Usage

1. Start the application:
```bash
python -m travel_bot
```

2. The chatbot will automatically use the Microsoft Phi-2 model for generating responses.

3. Send messages to get travel recommendations and itineraries.

## Model Information

The system uses Microsoft's Phi-2 model, which offers several advantages:
- Efficient CPU-based inference
- No GPU required
- Good performance for travel-related tasks
- Lightweight and fast responses
- Easy to deploy and use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the test scripts for usage examples