# AI Travel Planner - ERAV3 Capstone Project

An intelligent travel planning application that uses AI to create personalized travel itineraries and provide destination suggestions based on user preferences.

## 🌟 Features

- **AI-Powered Travel Suggestions**: Get personalized destination recommendations
- **Detailed Itinerary Creation**: Generate comprehensive day-by-day travel plans
- **Real-time Weather Information**: Check weather conditions for destinations
- **Flight Search**: Find available flights using Skyscanner API
- **Hotel Search**: Discover accommodation options using Hotels.com API
- **Currency Conversion**: Get real-time exchange rates
- **Interactive Maps**: Visualize destinations with interactive maps
- **Conversation History**: Maintain context across planning sessions
- **Export Functionality**: Save and share your travel plans

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ERAV3-CapstoneProject/agentic_ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the `agentic_ai` directory with your API keys:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   RAPID_API_KEY=your_rapidapi_key_here
   SITE_URL=http://localhost:8501
   SITE_NAME=AI Travel Planner
   ```

4. **Run the application**
   ```bash
   python run_app.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

## 🔑 Required API Keys

### 1. OpenRouter API Key (Required)

**What it's used for:**
- AI-powered travel suggestions and itinerary generation
- Natural language processing for user queries
- Intelligent conversation handling

**How to get it:**
1. Visit [OpenRouter.ai](https://openrouter.ai/)
2. Sign up for a free account
3. Navigate to your dashboard
4. Generate a new API key
5. Copy the key and add it to your `.env` file

**Cost:** Free tier available with generous limits

**Environment Variable:** `OPENROUTER_API_KEY`

---

### 2. RapidAPI Key (Optional but Recommended)

**What it's used for:**
- Flight search via Skyscanner API
- Hotel search via Hotels.com API
- Real-time travel data

**How to get it:**
1. Visit [RapidAPI.com](https://rapidapi.com/)
2. Create a free account
3. Subscribe to the following APIs:
   - **Skyscanner API** (for flight search)
   - **Hotels.com API** (for hotel search)
   - **Fly scrapper API** (for hotels, flight search)
4. Get your RapidAPI key from your dashboard
5. Add it to your `.env` file

**Cost:** Free tier available for most APIs

**Environment Variable:** `RAPID_API_KEY`

---

### 3. Optional Environment Variables

**SITE_URL** (Optional)
- Default: `http://localhost:8501`
- Used for generating shareable links
- Set to your deployment URL in production

**SITE_NAME** (Optional)
- Default: `AI Travel Planner`
- Used for branding in generated content

## 📁 Project Structure

```
agentic_ai/
├── app.py                 # Main Streamlit application
├── run_app.py            # Application runner
├── start.py              # Alternative startup script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .env                 # Environment variables (create this)
├── agents/              # AI agent implementations
├── tools/               # Travel utility tools
│   ├── travel_tools.py  # Core travel tools
│   ├── travel_utils.py  # Utility functions
│   ├── WeatherTool.py   # Weather information
│   ├── FlightSearchTool.py # Flight search
│   └── CurrencyTool.py  # Currency conversion
├── mcp_server/          # Model Context Protocol server
├── workflows/           # Workflow definitions
└── logs/               # Application logs
```

## 🛠️ Configuration

### Environment Variables Setup

Create a `.env` file in the `agentic_ai` directory:

```env
# Required API Keys
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_key_here
RAPID_API_KEY=your_rapidapi_key_here

# Optional Configuration
SITE_URL=http://localhost:8501
SITE_NAME=AI Travel Planner
```

### API Configuration Details

#### OpenRouter API
- **Base URL:** `https://openrouter.ai/api/v1/chat/completions`
- **Model Used:** `meta-llama/llama-3.3-8b-instruct:free`
- **Authentication:** Bearer token in Authorization header

#### RapidAPI Services
- **Skyscanner API Host:** `skyscanner-api.p.rapidapi.com`
- **Hotels.com API Host:** `hotels4.p.rapidapi.com`
- **Authentication:** X-RapidAPI-Key header

## 🚀 Running the Application

### Method 1: Using run_app.py (Recommended)
```bash
python run_app.py
```

### Method 2: Using start.py
```bash
python start.py
```

### Method 3: Direct Streamlit
```bash
streamlit run app.py
```

### Method 4: Using the batch file (Windows)
```bash
start_app.bat
```

## 🔧 Troubleshooting

### Common Issues

1. **"OpenRouter API key not found"**
   - Ensure `OPENROUTER_API_KEY` is set in your `.env` file
   - Verify the key is valid and active

2. **"RapidAPI key not found"**
   - This is a warning, not an error
   - Flight and hotel search will be unavailable
   - Other features will work normally

3. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

4. **Port already in use**
   - Change the port in `run_app.py` or use a different port
   - Kill existing processes using the port

### Logs

Application logs are stored in the `logs/` directory with timestamps. Check these files for detailed error information.

## 📊 API Usage and Limits

### OpenRouter API
- **Free Tier:** 10,000 requests/month
- **Rate Limit:** 10 requests/second
- **Model:** Llama 3.3 8B (free tier)

### RapidAPI
- **Skyscanner:** 100 requests/month (free tier)
- **Hotels.com:** 500 requests/month (free tier)
- **Rate Limits:** Vary by API

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure and rotate them regularly
- Monitor your API usage to avoid unexpected charges
- Use environment variables for all sensitive configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the ERAV3 Capstone Project.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs in the `logs/` directory
3. Ensure all API keys are properly configured
4. Verify all dependencies are installed

## 🔄 Updates

To update the application:
1. Pull the latest changes: `git pull`
2. Update dependencies: `pip install -r requirements.txt`
3. Restart the application

---

**Note:** This application requires internet connectivity for API calls and real-time data retrieval. 