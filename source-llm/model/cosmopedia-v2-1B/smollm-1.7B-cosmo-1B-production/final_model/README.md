---
license: mit
base_model: HuggingFaceTB/SmolLM2-1.7B
tags:
- text-generation
- causal-lm
- pytorch
- smollm2
- cosmopedia
- pre-trained
language:
- en
pipeline_tag: text-generation
---

# SmolLM2-1.7B pre-trained on Cosmopedia-v2

This model is a pre-trained version of [HuggingFaceTB/SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) on the Cosmopedia-v2 dataset.

## Model Details

- **Base Model**: HuggingFaceTB/SmolLM2-1.7B (1.7B parameters)
- **pre-trained on**: Cosmopedia-v2 dataset (1B tokens)
- **Training Steps**: 30,000
- **Final Loss**: 3.7547
- **Training Date**: 2025-06-22

## Training Configuration

```python
- Batch Size per Device: 1
- Gradient Accumulation Steps: 16
- Learning Rate: 2e-5
- Sequence Length: 2048
- Optimizer: 8-bit AdamW
- Mixed Precision: bf16
```

## Dataset

The model was trained on Cosmopedia-v2, a high-quality synthetic dataset containing educational content across various topics.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("saish-shetty/SmolLM2-1.7B-pre-trained")
model = AutoModelForCausalLM.from_pretrained(
    "saish-shetty/SmolLM2-1.7B-pre-trained",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate text
prompt = "Machine learning is a field of"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Performance

The model shows improved performance on text generation tasks compared to the base model, with better coherence and domain knowledge from the Cosmopedia-v2 training.

## Training Infrastructure

- **GPUs**: 4x NVIDIA A10G (24GB each)
- **Framework**: Transformers + DeepSpeed ZeRO Stage 2
- **Distributed Training**: Accelerate
- **Memory Optimization**: 8-bit optimizer, gradient checkpointing

## Limitations

- The model inherits limitations from the base SmolLM2-1.7B model
- Training was focused on educational content from Cosmopedia-v2
- May not perform optimally on tasks outside the training domain

## Citation

If you use this model, please cite:

```bibtex
@misc{smollm2-cosmopedia-finetune,
  title={SmolLM2-1.7B pre-trained on Cosmopedia-v2},
  author={Saish Shetty},
  year={2025},
  url={https://huggingface.co/saish-shetty/SmolLM2-1.7B-pre-trained}
}
```

## License

This model is released under the MIT license, following the base model's licensing.
