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

## Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended for better performance)
- Ollama server running locally with llama3.1:8b-instruct-q4_K_M model

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

5. Install and start Ollama server:
```bash
# Follow Ollama installation instructions for your platform
# Pull the required model
ollama pull llama3.1:8b-instruct-q4_K_M
```

## Usage

1. Start the Location Identification API:
```bash
python api.py
```

2. Start the Travel Planning Chatbot:
```bash
streamlit run chatbot_app.py
```

3. Access the applications:
- Location API Documentation: http://localhost:8000/docs
- Travel Planning Chatbot: http://localhost:8501

4. Using the Chatbot:
- Upload an image of a location
- Specify your travel preferences (e.g., "Plan a trip for 5 days from Bangalore")
- Receive a detailed travel itinerary

## API Endpoints

### POST /predict
Predicts locations from an uploaded image.

**Parameters:**
- `file`: Image file (required)
- `num_predictions`: Number of predictions to return (default: 3)
- `min_confidence`: Minimum confidence score threshold (optional)

**Response:**
```json
[
  {
    "prediction_number": 1,
    "location_name": "Full address",
    "latitude": 0.0,
    "longitude": 0.0,
    "confidence_score": 0.0
  }
]
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "model_loaded": true
}
```

## Configuration

The application can be configured using environment variables or a `.env` file. See `.env.example` for available options.

### Key Configuration Options
- `API_URL`: Location API endpoint (default: http://localhost:8000/predict)
- `OLLAMA_API_URL`: Ollama API endpoint (default: http://localhost:11434/api/generate)
- `MODEL_NAME`: Ollama model name (default: llama3.1:8b-instruct-q4_K_M)

## Development

### Running Tests
```bash
pytest
```

### Code Style and Quality
This project uses automated tools to maintain code quality and consistency:

1. **Code Formatting with Black**
   ```bash
   black .
   ```
   - Automatically formats Python code
   - Enforces consistent code style
   - No configuration needed - it's opinionated and consistent

2. **Code Linting with Flake8**
   ```bash
   flake8
   ```
   - Checks for programming errors
   - Enforces Python style guide (PEP 8)
   - Identifies potential bugs and code complexity issues

Before submitting a pull request, please ensure your code passes both Black formatting and Flake8 linting checks.

### Logging
- All chatbot interactions are logged in the `logs` directory
- Log files are created with timestamps
- Logs include both prompts and responses
- Unicode characters are properly handled

## Acknowledgments

This project uses the [GeoCLIP](https://github.com/VicenteVivan/geo-clip) framework for image-based location identification. GeoCLIP is a CLIP-inspired model that aligns images with geographical locations, achieving state-of-the-art results on geo-localization tasks. The framework is based on the paper "GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization" by Vivanco, Vicente and Nayak, Gaurav Kumar and Shah, Mubarak, published at NeurIPS 2023.

If you use this project in your research, please cite the original GeoCLIP paper:
```bibtex
@inproceedings{geoclip,
  title={GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization},
  author={Vivanco, Vicente and Nayak, Gaurav Kumar and Shah, Mubarak},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## License

[Your License]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 