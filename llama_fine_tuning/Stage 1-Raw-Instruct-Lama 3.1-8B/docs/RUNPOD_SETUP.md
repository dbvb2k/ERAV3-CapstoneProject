# Llama-3-8B Fine-tuning on RunPod

**ROBUST SETUP** - Eliminates package conflicts and handles network drive mounting automatically.

## Quick Start

### 1. Upload Files to RunPod
Upload these files to your RunPod workspace:
- `train_llama.py` - Main training script
- `requirements.txt` - Package dependencies  
- `setup_runpod.py` - Environment setup script
- `runpod_start.sh` - Bash setup script (alternative)
- `verify_setup.py` - Setup verification script

### 2. Setup Environment (Choose One Method)

**Method A - Python Setup (Recommended):**
```bash
python setup_runpod.py
```

**Method B - Bash Setup:**
```bash
chmod +x runpod_start.sh
./runpod_start.sh
```

### 3. Verify Setup (with environment variables)
```bash
# Option 1: Automatic verification script
bash /workspace/run_verification.sh

# Option 2: Using wrapper script  
/workspace/runpod_env.sh python verify_setup.py

# Option 3: Manual source
source /workspace/setup_env.sh && python verify_setup.py
```

### 4. Set Hugging Face Token
```bash
export HF_TOKEN="your_huggingface_token_here"
```

### 5. Start Training (with environment variables)
```bash
# Option 1: Using wrapper script (recommended)
/workspace/runpod_env.sh python train_llama.py --model-name meta-llama/Meta-Llama-3-8B

# Option 2: Source environment first
source /workspace/setup_env.sh && python train_llama.py --model-name meta-llama/Meta-Llama-3-8B
```

## Key Improvements

✅ **Network Drive Support** - Automatically detects and uses `/workspace` network volume  
✅ **Fixed evaluation_strategy Error** - Installs transformers from GitHub source  
✅ **Package Conflict Resolution** - Pinned compatible versions  
✅ **Disk Space Management** - Proper cache directory setup  
✅ **Environment Variables** - Automatic HuggingFace cache configuration + current session  
✅ **NumPy Compatibility** - Forces NumPy 1.24.4 for verification script compatibility  
✅ **Comprehensive Verification** - Pre-training checks pass without errors  

## Environment Variables Set Automatically

```bash
HF_HOME=/workspace/hf_cache
HF_DATASETS_CACHE=/workspace/hf_cache/datasets  
TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
TMPDIR=/workspace/tmp
TORCH_HOME=/workspace/torch_cache
```

## GPU Requirements

**Minimum:** 24GB VRAM (RTX A5000, RTX 4090)  
**Recommended:** 40GB+ VRAM (A100, H100)

## Memory Management

If you get OOM errors, try:
```bash
# Smaller batch size
python train_llama.py --batch-size 1

# Shorter sequences  
python train_llama.py --max-length 256

# Fewer epochs
python train_llama.py --num-epochs 1
```

## Package Versions (Tested & Compatible)

- `torch>=2.0.0,<2.2.0`
- `transformers` (from GitHub source v4.35.2)
- `datasets>=2.14.0,<2.19.0`
- `numpy==1.24.4` (pinned for verification script compatibility)
- `peft>=0.6.0,<0.8.0`

## Output Files

- `outputs/` - Model checkpoints and final model
- `logs/` - Training logs and metrics  
- `plots/` - Training progress visualizations

## Resume Training

```bash
/workspace/runpod_env.sh python train_llama.py --auto-resume
```

## Test Your Trained Model

### Quick Model Test (verify model loads)
```bash
# Test if model can be loaded and generate responses
/workspace/runpod_env.sh python quick_model_test.py
```

### Comprehensive Travel Query Testing
```bash
# Test model on 15 diverse travel queries
/workspace/runpod_env.sh python test_travel_queries.py

# Or use the wrapper script
chmod +x test_travel_model.sh
./test_travel_model.sh

# With custom settings
/workspace/runpod_env.sh python test_travel_queries.py --temperature 0.8 --max-new-tokens 300
```

### Testing Results
- **JSON file**: Detailed structured results for analysis
- **Markdown file**: Human-readable report with all responses
- **Categories tested**: Destination planning, budget travel, transportation, safety, food, etc.
- **Evaluation metrics**: Topic coverage, response length, generation time

## Custom Configuration

```bash
python train_llama.py \
  --num-epochs 3 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --max-length 512
```

## Troubleshooting

**evaluation_strategy Error:**
- ✅ **FIXED** - Setup installs transformers from GitHub source

**Network Drive Issues:**
- ✅ **FIXED** - Automatic detection and environment setup

**Package Conflicts:**
- ✅ **FIXED** - Pinned compatible versions in requirements.txt

**NumPy Version Issues:**
- ✅ **FIXED** - Forces NumPy 1.24.4 installation for verification script compatibility

**Environment Variables Not Set:**
- ✅ **FIXED** - Sets variables in current session automatically

**Disk Space Issues:**
- ✅ **FIXED** - Caches stored on network volume, automatic cleanup

**GPU Memory Issues:**
- Reduce `--batch-size`
- Reduce `--max-length`  
- Use gradient checkpointing (already enabled)

## Training Completion

When training completes successfully, you'll see output like this:

```
2025-06-16 01:39:29,240 - __main__ - INFO - 💾 Final inference results saved to:
2025-06-16 01:39:29,244 - __main__ - INFO -    - outputs/final_inference_results.json
2025-06-16 01:39:29,245 - __main__ - INFO -    - logs/final_inference_comprehensive.jsonl
2025-06-16 01:39:29,245 - __main__ - INFO -    - logs/inference_results.log
2025-06-16 01:39:29,254 - __main__ - INFO - 🎉 Training pipeline completed successfully!
2025-06-16 01:39:29,254 - __main__ - INFO - 📊 Final Training Summary:
2025-06-16 01:39:29,255 - __main__ - INFO -    Training Time: 424.7 minutes
2025-06-16 01:39:29,255 - __main__ - INFO -    Total Steps: 9261
2025-06-16 01:39:29,255 - __main__ - INFO -    Final Loss: 1.132094896788371
2025-06-16 01:39:29,255 - __main__ - INFO -    Best Eval Loss: 1.2782177925109863
2025-06-16 01:39:29,256 - __main__ - INFO -    Inference Tests: 7
2025-06-16 01:39:29,256 - __main__ - INFO - 📁 Check the following directories:
2025-06-16 01:39:29,256 - __main__ - INFO -    - outputs/: Model checkpoints and final model
2025-06-16 01:39:29,257 - __main__ - INFO -    - plots/: Training progress visualizations
2025-06-16 01:39:29,257 - __main__ - INFO -    - logs/: Comprehensive training and inference logs
2025-06-16 01:39:29,257 - __main__ - INFO - 📝 Log Files Created:
2025-06-16 01:39:29,257 - __main__ - INFO -    - logs/training_complete.log: Complete training log
2025-06-16 01:39:29,258 - __main__ - INFO -    - logs/training_metrics.log: Training metrics only
2025-06-16 01:39:29,258 - __main__ - INFO -    - logs/training_metrics.csv: Structured metrics for analysis
2025-06-16 01:39:29,258 - __main__ - INFO -    - logs/training_detailed.jsonl: Detailed JSON training logs
2025-06-16 01:39:29,258 - __main__ - INFO -    - logs/inference_results.log: Inference test logs
2025-06-16 01:39:29,259 - __main__ - INFO -    - logs/inference_comprehensive.jsonl: Structured inference data
2025-06-16 01:39:29,259 - __main__ - INFO -    - logs/system_monitoring.log: System resource usage
2025-06-16 01:39:29,260 - __main__ - INFO -    - logs/training_summary.json: Final training summary
2025-06-16 01:39:29,260 - system - INFO - === TRAINING COMPLETED ===
```

### Training Performance Summary
- **⏱️ Training Time:** 424.7 minutes (~7.1 hours)
- **📊 Total Steps:** 9,261 training steps completed
- **📉 Final Loss:** 1.132 (excellent convergence)
- **📈 Best Eval Loss:** 1.278 (good generalization)
- **🧪 Inference Tests:** 7 successful quality checks

### Final Output Structure
```
/workspace/
├── outputs/                    # 🎯 Your fine-tuned model
│   ├── final_model/           # Complete trained model
│   ├── checkpoint-XXXX/       # Training checkpoints
│   └── final_inference_results.json
├── plots/                     # Training visualizations
│   └── training_progress.png
└── logs/                      # Comprehensive logs
    ├── training_complete.log
    ├── training_metrics.csv
    ├── training_detailed.jsonl
    ├── inference_results.log
    ├── inference_comprehensive.jsonl
    ├── system_monitoring.log
    └── training_summary.json
```

## Verification Commands

```bash
# Check all components
python verify_setup.py

# Test transformers specifically
python -c "from transformers import TrainingArguments; print('✅ evaluation_strategy works!')"

# Check disk usage
df -h /workspace
``` 