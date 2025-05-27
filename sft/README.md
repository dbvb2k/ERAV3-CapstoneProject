# 🚀 Ultra-Fast LLaMA 3 Instruct India Travel Assistant

Train LLaMA 3 8B Instruct for India-centric travel assistance with **2x faster training** using Unsloth!

- **Dataset**: 83,598 high-quality examples
- **Speed**: 2x faster than standard training
- **Memory**: 50% less GPU memory usage
- **Quality**: 97.1% excellent examples

## ⚡ Ultra-Fast Training with Unsloth

### 🔧 Step 1: Setup Environment

#### Option A: Google Colab (Recommended for Beginners)
```bash
# Install Unsloth in Colab
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
```

#### Option B: Local/Server Setup
```bash
# Clone this repository
git clone https://github.com/yourusername/llama3-india-travel-assistant
cd llama3-india-travel-assistant

# Install dependencies
pip install -r training/requirements.txt
```

### 🤗 Step 2: Get Hugging Face Access

1. **Create Hugging Face Account**: https://huggingface.co/join
2. **Get LLaMA 3 Access**: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
3. **Generate Token**: https://huggingface.co/settings/tokens
4. **Login**: `huggingface-cli login` or set `HF_TOKEN`

```bash
# Login to Hugging Face (required for LLaMA 3)
huggingface-cli login
# Enter your token when prompted
```

### 🚀 Step 3: Start Ultra-Fast Training

```bash
# One-command training (2-4 hours instead of 8-12!)
python training/train_llama3b_unsloth.py \
  --model_name "unsloth/llama-3-8b-Instruct-bnb-4bit" \
  --train_data data/final/train_llama_format.json \
  --val_data data/final/validation_llama_format.json \
  --output_dir ./travel-assistant-unsloth
```

### 🧪 Step 4: Test Your Model

```bash
# Interactive chat mode
python training/inference_unsloth.py \
  --model_path ./travel-assistant-unsloth/final \
  --interactive

# Test with sample queries
python training/inference_unsloth.py \
  --model_path ./travel-assistant-unsloth/final \
  --test_samples
```

## 🖥️ GPU Requirements & Configurations

| GPU Model | VRAM | Training Time | Batch Size | Status |
|-----------|------|---------------|------------|--------|
| **RTX 4090** | 24GB | 2-3 hours | 4 | ✅ Optimal |
| **RTX 4080** | 16GB | 3-4 hours | 2 | ✅ Great |
| **RTX 3090** | 24GB | 3-4 hours | 4 | ✅ Great |
| **RTX 3080** | 10GB | 4-5 hours | 2 | ✅ Good |
| **V100** | 16GB | 3-4 hours | 2 | ✅ Good |
| **T4** | 16GB | 5-6 hours | 1 | ✅ Budget |
| **RTX 3070** | 8GB | 6-8 hours | 1 | ⚠️ Tight |

### 🎛️ GPU Memory Optimization

```bash
# For 8GB GPUs (tight memory)
python training/train_llama3b_unsloth.py \
  --model_name "unsloth/llama-3-8b-Instruct-bnb-4bit" \
  --max_seq_length 1024

# For 16GB+ GPUs (optimal)
python training/train_llama3b_unsloth.py \
  --model_name "unsloth/llama-3-8b-Instruct-bnb-4bit" \
  --max_seq_length 2048
```

## 📊 Dataset Details

- **Size**: 83,598 examples (75,238 train / 8,360 validation)
- **Quality**: 97.1% excellent quality, 6.22/10 average score  
- **Sources**: MultiWOZ (90.4%), MTOP (5.3%), Synthetic (0.2%)
- **Focus**: India-centric travel planning and assistance

## 🎯 Training Performance Comparison

| Method | Training Time | GPU Memory | Quality |
|--------|---------------|------------|---------|
| **Unsloth** | 2-4 hours | 6-12GB | 🏆 Best |
| Standard LoRA | 8-12 hours | 12-20GB | Good |
| Full Fine-tuning | 24+ hours | 40GB+ | Good |

## 💡 Model Capabilities

After training, your model will excel at:
- ✈️ **Flight Booking**: India-specific airlines, routes, pricing
- 🏨 **Hotels**: Budget to luxury, vegetarian-friendly options
- 🏛️ **Attractions**: Cultural sites, festivals, local experiences  
- 🍛 **Dining**: Vegetarian/halal restaurants, regional cuisines
- 👨‍👩‍👧‍👦 **Family Travel**: Kid-friendly destinations, safety tips
- 💰 **Budget Planning**: Cost estimates, money-saving tips
- 📋 **Documentation**: Visa requirements, travel documents

## 🔄 Advanced Training Options

### Custom Models
```bash
# Use different LLaMA 3 variants
python training/train_llama3b_unsloth.py \
  --model_name "unsloth/llama-3-8b-bnb-4bit"  # Base model

python training/train_llama3b_unsloth.py \
  --model_name "unsloth/llama-3-70b-Instruct-bnb-4bit"  # Larger model
```

### Extended Training
```bash
# Train for more epochs (better quality)
python training/train_llama3b_unsloth.py \
  --model_name "unsloth/llama-3-8b-Instruct-bnb-4bit" \
  --num_train_epochs 3
```

## 📁 Repository Structure

```
📦 llama3-india-travel-assistant/
├── 📁 data/final/
│   ├── 📄 train_llama_format.json (31MB)      # Training dataset
│   └── 📄 validation_llama_format.json (3.5MB) # Validation dataset
├── 📁 training/
│   ├── 🚀 train_llama3b_unsloth.py            # Ultra-fast training script
│   ├── ⚡ inference_unsloth.py               # Fast inference script
│   └── 📄 requirements.txt                   # Optimized dependencies
└── 📄 README.md                              # This guide
```

## ❓ Troubleshooting

### Common Issues & Solutions

**🚨 CUDA Out of Memory**
```bash
# Reduce batch size and sequence length
python training/train_llama3b_unsloth.py \
  --per_device_train_batch_size 1 \
  --max_seq_length 1024
```

**🚨 Hugging Face Access Denied**
```bash
# Make sure you have LLaMA 3 access and are logged in
huggingface-cli login
# Request access: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

**🚨 Slow Training**
```bash
# Make sure you're using Unsloth optimized models
--model_name "unsloth/llama-3-8b-Instruct-bnb-4bit"  # ✅ Fast
--model_name "meta-llama/Meta-Llama-3-8B-Instruct"   # ❌ Slow
```

## 📈 Expected Results

After training, test with these India-specific queries:

```
🧪 "Plan a 5-day Kerala backwater trip for ₹30,000"
🧪 "Best vegetarian restaurants in Rajasthan"
🧪 "How to travel from Delhi to Ladakh by road?"
🧪 "Visa requirements for Indians visiting Thailand"
🧪 "Family-friendly hotels in Goa under ₹5,000/night"
```

Your model should provide detailed, culturally-aware responses with:
- **Local insights** and cultural preferences
- **Budget considerations** in Indian Rupees
- **Vegetarian/halal** dining options
- **Family-friendly** recommendations
- **Practical advice** for Indian travelers

## 🎉 Success! You've Built an AI Travel Assistant

Your trained model is now ready to help with:
- Personalized India travel planning
- Cultural recommendations  
- Budget-friendly options
- Family travel advice
- International travel for Indians

## 📄 License

MIT License - Feel free to use for research and commercial purposes.

---

**⭐ Star this repo if it helped you build an amazing travel assistant! ⭐** 