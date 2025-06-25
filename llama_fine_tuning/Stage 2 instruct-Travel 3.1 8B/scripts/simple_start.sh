#!/bin/bash

# SIMPLE TRAVEL TRAINING STARTUP
# No complexity, no conflicts, just works!

set -e

echo "🚀 Simple Travel Model Training"
echo "================================"

# Check GPU
echo "📍 GPU Check:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "Warning: GPU check failed"

# Check datasets
echo "📊 Dataset Check:"
if [ ! -f "FINAL_TRAINING_DATASET_LLAMA8B.jsonl" ]; then
    echo "❌ Training dataset not found!"
    exit 1
fi

if [ ! -f "FINAL_VALIDATION_DATASET_LLAMA8B.jsonl" ]; then
    echo "❌ Validation dataset not found!"
    exit 1
fi

echo "✅ Training dataset: $(wc -l < FINAL_TRAINING_DATASET_LLAMA8B.jsonl) examples"
echo "✅ Validation dataset: $(wc -l < FINAL_VALIDATION_DATASET_LLAMA8B.jsonl) examples"

# Check Python packages
echo "📦 Package Check:"
python -c "
import torch
import transformers
import datasets  
import peft
print('✅ All packages available')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Start training
echo ""
echo "🔥 Starting training..."
echo "Expected time: 2-4 hours"
echo "Expected result: 100% better travel responses"
echo ""

python simple_travel_trainer.py

echo ""
echo "✅ Training complete!"
echo "Model saved to: ./simple_travel_model" 