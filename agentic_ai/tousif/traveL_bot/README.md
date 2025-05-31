# Travel Advisory Bot

An AI-powered travel advisory bot using Devstral for personalized travel recommendations and itinerary planning.

## Quick Start

1. **Check Setup**: Double-click `check_devstral.bat`
2. **Start Application**: Double-click `start_devstral.bat`
3. **Open Browser**: Navigate to `http://localhost:8501`

## Prerequisites

- Python 3.8+
- Ollama with Devstral model
- Required packages (auto-installed)

## Features

- 🎯 AI-powered travel suggestions
- 📅 Detailed itinerary planning
- 💬 Interactive chat interface
- 📁 Multi-file support (PDF, images, audio, video)
- 🔒 Complete privacy (local processing)

## Configuration

Edit `.env` file for customization:
```env
OLLAMA_MODEL=devstral
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_AI_MODEL=ollama
```

## Troubleshooting

1. Ensure Ollama is running: `ollama serve`
2. Verify Devstral is installed: `ollama list`
3. Check Python dependencies are installed

## Support

For issues, check the troubleshooting section or run the diagnostic batch file.