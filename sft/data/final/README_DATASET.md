# Dataset Files Reference

This directory contains the processed travel assistant dataset in multiple formats.

## 📊 Dataset Overview

- **Total Examples**: 83,598 (after deduplication and quality filtering)
- **Quality Score**: 6.22/10 average (97.1% excellent quality)
- **Real Data Ratio**: 99.8% (optimal for LLaMA 3 training)
- **India Focus**: Cultural preferences, vegetarian/halal options, family travel

## 📁 File Descriptions

### Training Files ⭐ **USE THESE FOR TRAINING**
- **`train_llama_format.json`** (31MB) - Training dataset in LLaMA conversation format
  - 75,238 examples for training
  - Optimized for LLaMA 3 chat templates
  - **Primary file for training**

- **`validation_llama_format.json`** (3.5MB) - Validation dataset in LLaMA conversation format
  - 8,360 examples for validation
  - Same format as training data
  - **Primary file for validation**

### Reference File 📋
- **`complete_travel_dataset.json`** (32MB) - Complete merged dataset
  - Contains all 83,598 examples combined
  - Useful for analysis and reference
  - **Not needed for training** (just for reference)

## 🚀 Quick Training Usage

```bash
# Ultra-fast training with Unsloth
python training/train_llama3b_unsloth.py \
  --train_data data/final/train_llama_format.json \
  --val_data data/final/validation_llama_format.json
```

## 📈 Dataset Sources

| Source | Examples | Percentage | Description |
|--------|----------|------------|-------------|
| **MultiWOZ** | 75,564 | 90.4% | Multi-domain travel dialogues |
| **MTOP** | 4,458 | 5.3% | Travel intent parsing |
| **Synthetic** | 165 | 0.2% | India-centric generated data |
| **Enhanced ATIS** | 12 | 0.0% | Flight booking queries |

## 🎯 Data Format

Each example follows the LLaMA conversation format:
```json
{
  "conversations": [
    {
      "from": "human", 
      "value": "User query about travel"
    },
    {
      "from": "gpt", 
      "value": "AI assistant response"
    }
  ]
}
```

## 📊 Quality Metrics

- **Length Optimization**: 10-1024 chars instructions, 20-2048 chars outputs
- **Travel Domain Focus**: 100% travel-related content
- **Cultural Relevance**: India-centric preferences and scenarios
- **Duplicate Removal**: Intelligent deduplication based on instruction similarity 