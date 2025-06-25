# 🏋️ Training Guide

This guide explains how to train your own Llama-3-8B travel model using LoRA fine-tuning.

## 🎯 Overview

We use **LoRA (Low-Rank Adaptation)** to fine-tune Llama-3-8B on the Alpaca dataset. This approach:
- ✅ **Reduces memory requirements** from ~30GB to ~25GB
- ✅ **Faster training** compared to full fine-tuning
- ✅ **Smaller model files** (~640MB vs ~16GB)
- ✅ **Easy to merge/unmerge** with base model

## 🖥️ Hardware Requirements

### Minimum Requirements
- **GPU**: RTX A5000 (24GB VRAM)
- **RAM**: 32GB system RAM
- **Storage**: 50GB+ available space
- **Internet**: Stable connection for model download

### Recommended Requirements
- **GPU**: RTX A6000, A100 (40GB+ VRAM)
- **RAM**: 64GB+ system RAM
- **Storage**: 100GB+ NVMe SSD
- **Internet**: High-speed connection

## 🚀 Training on RunPod (Recommended)

### Step 1: Setup RunPod Instance

1. **Go to RunPod.io** and create an account
2. **Select GPU Instance**:
   - GPU: RTX A5000 or better
   - Template: PyTorch 2.0+
   - Storage: 50GB+ network volume
3. **Start Instance** and note the connection details

### Step 2: Upload Training Files

Upload these files to your RunPod workspace (`/workspace/`):

```bash
scripts/train_llama.py
scripts/setup_runpod.py  
scripts/requirements.txt
```

### Step 3: Environment Setup

```bash
# Run the setup script
python setup_runpod.py

# Source environment variables
source /workspace/setup_env.sh

# Set your Hugging Face token
export HF_TOKEN="your_huggingface_token_here"
```

### Step 4: Start Training

```bash
# Basic training command
python train_llama.py --model-name meta-llama/Meta-Llama-3-8B

# With custom parameters
python train_llama.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --num-epochs 3 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --max-length 512
```

### Step 5: Monitor Training

The training script provides comprehensive logging:

```bash
# Watch training progress
tail -f logs/training_complete.log

# Monitor system resources
tail -f logs/system_monitoring.log

# Check training metrics
cat logs/training_metrics.csv
```

## 📊 Training Parameters

### Default Settings
```python
# Model settings
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset = "alpaca"  # 52,002 instruction examples

# LoRA settings
lora_r = 16          # Rank of adaptation
lora_alpha = 32      # LoRA scaling parameter
lora_dropout = 0.1   # LoRA dropout

# Training settings
num_epochs = 3
batch_size = 2
learning_rate = 2e-4
max_length = 512
warmup_steps = 100
```

### Advanced Configuration
```python
# Memory optimization
gradient_checkpointing = True
fp16 = True
dataloader_num_workers = 4

# Evaluation
eval_strategy = "steps"
eval_steps = 500
save_steps = 500
logging_steps = 10
```

## 🔧 Local Training (Advanced)

### Requirements
- CUDA-compatible GPU (24GB+ VRAM)
- Python 3.8+
- PyTorch 2.0+

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/llama-travel-model
cd llama-travel-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r scripts/requirements.txt

# Set Hugging Face token
export HF_TOKEN="your_token_here"
```

### Training
```bash
# Run training
python scripts/train_llama.py --model-name meta-llama/Meta-Llama-3-8B
```

## 📈 Training Process

### Phase 1: Setup (5-10 minutes)
- Download base model (~16GB)
- Load and preprocess Alpaca dataset
- Initialize LoRA adapters
- Setup training environment

### Phase 2: Training (6-8 hours)
- **Steps**: ~9,261 training steps
- **Checkpoints**: Saved every 500 steps
- **Evaluation**: Every 500 steps
- **Logging**: Comprehensive metrics

### Phase 3: Completion (5 minutes)
- Save final model
- Generate training summary
- Create inference tests
- Export minimal model

## 📋 Training Outputs

After training completes, you'll have:

```
outputs/
├── final_model/              # Complete trained model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── ...
├── checkpoint-9000/          # Final checkpoint
├── checkpoint-8500/          # Previous checkpoints
└── ...

logs/
├── training_complete.log     # Full training log
├── training_metrics.csv     # Structured metrics
├── training_summary.json    # Final summary
└── inference_results.log    # Quality tests

plots/
└── training_progress.png    # Training curves
```

## 🧪 Quality Verification

The training script automatically runs quality tests:

1. **Loss Convergence**: Monitors training/validation loss
2. **Inference Tests**: Generates responses to travel queries
3. **Performance Metrics**: Tracks response quality
4. **Memory Usage**: Monitors GPU/CPU utilization

### Expected Results
- **Final Loss**: ~1.13
- **Training Time**: 6-8 hours
- **Model Size**: ~640MB (LoRA adapter)
- **Inference Speed**: ~5-6 seconds per query

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train_llama.py --batch-size 1

# Reduce sequence length
python train_llama.py --max-length 256
```

**2. Slow Training**
```bash
# Enable gradient checkpointing (memory for speed trade-off)
python train_llama.py --gradient-checkpointing

# Reduce evaluation frequency
python train_llama.py --eval-steps 1000
```

**3. Model Loading Issues**
```bash
# Check Hugging Face token
huggingface-cli whoami

# Re-login if needed
huggingface-cli login
```

**4. Package Conflicts**
```bash
# Clean reinstall
pip uninstall transformers peft datasets
pip install -r scripts/requirements.txt
```

## 💡 Tips for Better Results

### 1. Data Quality
- Use high-quality instruction datasets
- Filter out low-quality examples
- Balance different types of travel queries

### 2. Hyperparameter Tuning
- **Lower learning rate** (1e-4) for stable training
- **Higher LoRA rank** (32) for complex tasks
- **More epochs** (5) for better convergence

### 3. Evaluation Strategy
- Test on diverse travel scenarios
- Monitor both training metrics and actual responses
- Use held-out validation set

### 4. Resource Optimization
- Use mixed precision training (fp16)
- Enable gradient checkpointing
- Optimize batch size for your GPU

## 📚 Further Reading

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca)
- [Llama-3 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## 🤝 Need Help?

- **Issues**: [GitHub Issues](https://github.com/yourusername/llama-travel-model/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/llama-travel-model/discussions)
- **RunPod Support**: [RunPod Discord](https://discord.gg/runpod) 