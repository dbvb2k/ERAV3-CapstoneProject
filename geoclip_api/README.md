# Place Identifier API

A REST API service that identifies locations from images using the GeoCLIP model. The service provides location predictions with confidence scores and detailed address information.

## Features

- Image-based location identification
- Multiple location predictions with confidence scores
- Detailed address information using reverse geocoding
- RESTful API with Swagger documentation
- Health check endpoint
- Configurable settings via environment variables
- Comprehensive error handling and logging

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for better performance)

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

## Usage

1. Start the API server:
```bash
python api.py
```

2. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

3. Health check endpoint:
```bash
curl http://localhost:8000/health
```

4. Make predictions:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/image.jpg" \
     -F "num_predictions=3" \
     -F "min_confidence=50"
```

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

## Development

### Running Tests
```bash
pytest
```

### Code Style
```bash
black .
flake8
```

## License

[Your License]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 