# 🧳 Llama-3-8B Travel Assistant

A fine-tuned Llama-3-8B model specialized in travel advice, trained on the Alpaca dataset using LoRA (Low-Rank Adaptation) for efficient fine-tuning.

## 🎯 Overview

This repository contains everything you need to:
- **Train** your own travel-specialized Llama model on RunPod
- **Use** the pre-trained minimal model for travel advice
- **Extract** essential files for deployment
- **Test** model performance on travel queries

## 🚀 Quick Start

### Option 1: Use Pre-trained Model

Download the minimal model and start generating travel advice immediately:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load tokenizer from minimal model
tokenizer = AutoTokenizer.from_pretrained('./model')

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3-8B-Instruct',
    torch_dtype=torch.float16,
    device_map='auto'
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, './model')

# Generate travel advice
prompt = "What are the best budget destinations in Southeast Asia?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Option 2: Train Your Own Model

Follow the [RunPod Setup Guide](docs/RUNPOD_SETUP.md) to train from scratch.

## 📁 Repository Structure

```
llama-travel-model/
├── README.md                 # This file
├── model/                    # Pre-trained minimal model (download separately)
│   ├── tokenizer.json
│   ├── adapter_model.safetensors
│   └── ...
├── scripts/                  # Training and testing scripts
│   ├── train_llama.py       # Main training script
│   ├── setup_runpod.py      # Environment setup
│   ├── test_minimal_model.py # Model testing
│   └── requirements.txt     # Dependencies
├── docs/                     # Documentation
│   ├── RUNPOD_SETUP.md     # RunPod training guide
│   ├── TRAINING_GUIDE.md   # Training instructions
│   └── USAGE_EXAMPLES.md   # Code examples
└── examples/                 # Usage examples
    ├── basic_inference.py   # Simple usage
    └── travel_chatbot.py    # Advanced chatbot
```

## 💾 Model Details

- **Base Model**: Meta-Llama-3-8B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Dataset**: Alpaca dataset (52,002 examples)
- **Specialization**: Travel advice and recommendations
- **Model Size**: ~649 MB (minimal version)
- **Hardware**: Trained on RTX A5000 (25GB VRAM)

## 🔧 Installation

### Requirements
- Python 3.8+
- CUDA-compatible GPU (24GB+ VRAM recommended)
- 650 MB disk space for minimal model

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/llama-travel-model
cd llama-travel-model

# Install dependencies
pip install -r scripts/requirements.txt

# Download the minimal model (see releases)
# Place in ./model/ directory
```

## 🏋️ Training

### RunPod Training (Recommended)

1. **Setup RunPod Instance**
   - GPU: RTX A5000 or better (24GB+ VRAM)
   - Storage: 50GB+ network volume
   - Template: PyTorch 2.0+

2. **Upload Files**
   ```bash
   # Upload these files to RunPod workspace:
   scripts/train_llama.py
   scripts/setup_runpod.py
   scripts/requirements.txt
   ```

3. **Run Training**
   ```bash
   python scripts/setup_runpod.py
   source /workspace/setup_env.sh
   python scripts/train_llama.py --model-name meta-llama/Meta-Llama-3-8B
   ```

4. **Extract Minimal Model**
   ```bash
   python scripts/extract_and_verify_minimal_model_fixed.py
   ```

See [detailed RunPod guide](docs/RUNPOD_SETUP.md) for complete instructions.

## 🧪 Testing

### Test Pre-trained Model
```bash
python scripts/test_minimal_model.py
```

### Test Custom Queries
```python
from scripts.test_minimal_model import load_minimal_model, test_model

# Load model
model, tokenizer = load_minimal_model("./model")

# Test custom queries
queries = [
    "Best time to visit Japan for cherry blossoms?",
    "Budget backpacking route through Europe?",
    "Safety tips for solo female travelers in India?"
]

results, summary = test_model(model, tokenizer, queries)
print(f"Average response time: {summary['average_time']:.2f}s")
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Training Time | ~7 hours |
| Final Loss | 1.132 |
| Model Size | 649 MB |
| Inference Speed | ~5.6s per query |
| Success Rate | 100% |

## 🌟 Features

- **Travel-Specialized**: Fine-tuned specifically for travel advice
- **Efficient**: LoRA adaptation reduces model size by 93%
- **Fast Inference**: Optimized for quick responses
- **Comprehensive**: Covers destinations, budgets, safety, culture
- **Easy Deployment**: Minimal model with simple setup

## 📝 Example Outputs

**Query**: "What are some budget-friendly destinations in Southeast Asia?"

**Response**: "Here are some excellent budget-friendly destinations in Southeast Asia:

1. **Vietnam** - Incredible street food, affordable accommodation ($5-15/night), and rich history
2. **Thailand** - Beautiful beaches, friendly locals, and great value for money
3. **Cambodia** - Home to Angkor Wat, very affordable living costs
4. **Laos** - Peaceful atmosphere, stunning nature, extremely budget-friendly
5. **Indonesia** - Diverse islands, cheap local food, and affordable transport

Daily budget: $15-30 including accommodation, food, and local transport."

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI** for the Llama-3-8B base model
- **Hugging Face** for the transformers library
- **Microsoft** for the LoRA implementation (PEFT)
- **Stanford** for the Alpaca dataset
- **RunPod** for GPU infrastructure

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/llama-travel-model/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/llama-travel-model/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/llama-travel-model/wiki)

## 🔗 Related Projects

- [Llama-3 by Meta](https://github.com/meta-llama/llama3)
- [PEFT by Hugging Face](https://github.com/huggingface/peft)
- [Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca)

---

⭐ **If this project helped you, please give it a star!** ⭐ 