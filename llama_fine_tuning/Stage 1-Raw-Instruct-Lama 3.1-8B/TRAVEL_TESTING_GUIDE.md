# Travel Query Testing Guide

## Overview

After training your Llama-3-8B model, you can test its performance on travel-related queries using the comprehensive testing scripts provided. This helps evaluate how well your fine-tuned model handles real-world travel questions.

## 🧪 Testing Scripts

### 1. `quick_model_test.py` - Basic Model Verification
**Purpose**: Quick test to verify the model loads and can generate responses  
**Time**: ~30 seconds  
**Use case**: Sanity check after training completes

```bash
/workspace/runpod_env.sh python quick_model_test.py
```

### 2. `test_travel_queries.py` - Comprehensive Testing
**Purpose**: Tests model on 15 diverse travel queries across multiple categories  
**Time**: ~5-10 minutes  
**Use case**: Detailed evaluation of model performance

```bash
/workspace/runpod_env.sh python test_travel_queries.py
```

### 3. `test_travel_model.sh` - Automated Wrapper
**Purpose**: Combines environment setup with testing  
**Time**: ~5-10 minutes  
**Use case**: One-click testing solution

```bash
chmod +x test_travel_model.sh
./test_travel_model.sh
```

## 📋 Test Categories

The comprehensive test covers 15 queries across these categories:

### Planning & Destinations
1. **Destination Planning** - Japan spring trip planning
2. **Budget Travel** - European backpacking under $50/day
3. **Family Travel** - US family-friendly destinations

### Transportation & Logistics  
4. **Transportation** - European train travel
5. **Flight Booking** - Best time to book international flights
6. **Local Transportation** - Asian city public transport

### Accommodation & Food
7. **Accommodation** - Hostels vs hotels vs Airbnb
8. **Food & Culture** - Authentic cuisine without getting sick
9. **Dietary Restrictions** - Vegetarian travel strategies

### Safety & Practical Tips
10. **Travel Safety** - Solo female traveler safety
11. **Travel Documents** - International travel preparations
12. **Money & Banking** - International payment strategies

### Seasonal & Activity-Based
13. **Seasonal Travel** - Winter warm weather destinations
14. **Adventure Travel** - Himalayan trekking preparation
15. **Cultural Experience** - Respectful cultural engagement

## 📊 Evaluation Metrics

### Automatic Evaluation
- **Topic Coverage**: How many expected topics are mentioned
- **Response Length**: Word count and categorization (short/medium/long)
- **Practical Information**: Presence of actionable advice
- **Generation Time**: Speed of response generation

### Manual Evaluation (from reports)
- **Accuracy**: Factual correctness of information
- **Helpfulness**: Practical value for travelers
- **Completeness**: Coverage of important aspects
- **Clarity**: Readability and organization

## 📁 Output Files

### JSON Results (`travel_queries_results_TIMESTAMP.json`)
```json
{
  "test_info": {
    "timestamp": "2025-01-20T10:30:00",
    "model_path": "outputs/final_model",
    "total_queries": 15,
    "total_time": 120.5,
    "average_time_per_query": 8.03
  },
  "results": [
    {
      "query_id": 1,
      "category": "Destination Planning",
      "query": "I'm planning a 7-day trip to Japan...",
      "response": "For a 7-day spring trip to Japan...",
      "evaluation": {
        "topics_covered": 4,
        "total_expected_topics": 6,
        "coverage_score": 0.67,
        "word_count": 185
      }
    }
  ]
}
```

### Markdown Report (`travel_queries_report_TIMESTAMP.md`)
Human-readable report with:
- Full questions and responses
- Evaluation scores for each query
- Easy-to-read formatting for manual review

## 🎯 Usage Examples

### Basic Testing
```bash
# Quick verification
/workspace/runpod_env.sh python quick_model_test.py

# Full testing
/workspace/runpod_env.sh python test_travel_queries.py
```

### Custom Parameters
```bash
# Higher temperature for more creative responses
/workspace/runpod_env.sh python test_travel_queries.py --temperature 0.9

# Longer responses
/workspace/runpod_env.sh python test_travel_queries.py --max-new-tokens 400

# Different model path
/workspace/runpod_env.sh python test_travel_queries.py --model-path outputs/checkpoint-5000
```

### With HuggingFace Token
```bash
# Set token and test
export HF_TOKEN="your_token_here"
./test_travel_model.sh

# Or pass token as argument
./test_travel_model.sh your_token_here
```

## 📈 Interpreting Results

### Good Performance Indicators
- **Coverage Score > 0.6**: Model mentions most expected topics
- **Word Count 100-300**: Detailed but not overly verbose responses
- **Practical Information**: Responses contain actionable advice
- **Fast Generation**: < 10 seconds per response

### Areas for Improvement
- **Low Coverage Score < 0.4**: Model missing key topics
- **Very Short Responses < 50 words**: Insufficient detail
- **Generic Responses**: Lack of specific, practical advice
- **Slow Generation > 15 seconds**: Potential memory issues

### Sample Good Response
```
Query: "What are some budget-friendly European destinations?"

Good Response:
"For budget-friendly European travel, consider Eastern European 
destinations like Prague, Budapest, and Krakow where you can easily 
stay under $50/day. Look for hostels ($15-25/night), eat at local 
markets and street food vendors, use budget airlines like Ryanair 
for transport, and take advantage of free walking tours and public 
parks for entertainment..."

Coverage: 4/5 topics ✅
Word Count: 120 words ✅
Practical Info: Yes ✅
```

## 🔧 Troubleshooting

### Model Not Found
```bash
❌ Model not found at outputs/final_model
💡 Make sure training completed successfully
💡 Check that outputs/final_model/ directory exists
```
**Solution**: Complete training first or specify correct model path

### GPU Memory Issues
```bash
❌ CUDA out of memory
```
**Solutions**:
- Reduce `--max-length` parameter
- Use `torch_dtype=torch.float16`
- Close other GPU processes

### Environment Variables Missing
```bash
❌ HF_HOME not set
```
**Solution**: Use wrapper scripts or source environment:
```bash
source /workspace/setup_env.sh
```

### Slow Generation
**Possible Causes**:
- Large model size
- CPU-only inference
- High temperature/long sequences

**Solutions**:
- Ensure GPU is being used
- Reduce `max_new_tokens`
- Lower temperature

## 💡 Tips for Better Testing

1. **Run Multiple Tests**: Test with different temperatures to see response variety
2. **Compare with Base Model**: Test the original Llama model for comparison
3. **Manual Review**: Read the markdown reports to assess quality beyond metrics
4. **Category Analysis**: Look at performance differences across travel categories
5. **Response Length**: Adjust `max_new_tokens` based on desired response length

## 🎯 Next Steps After Testing

### If Results Are Good (Coverage > 0.6, helpful responses)
- Deploy the model for production use
- Create a travel chatbot interface
- Test on additional domain-specific queries

### If Results Need Improvement (Coverage < 0.4, generic responses)
- Review training data quality
- Increase training epochs
- Adjust learning rate
- Add more travel-specific training data

## 📞 Support

If you encounter issues:
1. Check the error messages in the terminal
2. Review the troubleshooting section above
3. Ensure all environment variables are set
4. Verify the model training completed successfully

The testing scripts provide comprehensive evaluation to help you understand your model's travel domain performance! 🧳✈️ 