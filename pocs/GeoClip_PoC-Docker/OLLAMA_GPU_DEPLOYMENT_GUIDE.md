# GPU-Optimized Ollama Deployment Guide

## Overview
This guide explains how to deploy the Ollama service on GPU-enabled infrastructure (g4dn.2xlarge) for significantly faster inference times.

## Files Created

1. **`Dockerfile-GPU.ollama`** - GPU-optimized Docker image with CUDA support
2. **`copilot/ollama/manifest.yml`** - Updated Copilot manifest for GPU service
3. **`OLLAMA_GPU_DEPLOYMENT_GUIDE.md`** - This deployment guide

## Key Optimizations

### Dockerfile-GPU.ollama
- **NVIDIA CUDA Base Image**: Uses `nvidia/cuda:11.8-devel-ubuntu20.04`
- **GPU Layer Optimization**: Configures `OLLAMA_GPU_LAYERS=35` for better performance
- **GPU Detection**: Automatically detects and configures GPU environment
- **Memory Management**: Optimized for GPU memory usage
- **Enhanced Startup Script**: Includes GPU verification and optimization

### copilot/ollama/manifest.yml
- **GPU Configuration**: Specifies g4dn.2xlarge instance type
- **Resource Allocation**: 8 vCPU, 32GB RAM optimized for GPU workloads
- **EC2 Launch Type**: Required for GPU support
- **Environment Variables**: GPU-specific optimizations
- **Extended Health Check**: Longer start period for GPU model loading

## Deployment Steps

### 1. Prerequisites
- AWS ECS cluster with GPU-enabled EC2 instances (g4dn.2xlarge)
- NVIDIA drivers and Docker runtime installed on EC2 instances
- Copilot CLI installed
- AWS credentials configured

### 2. Initialize GPU Environment
```bash
# Create or update environment with GPU support
copilot env init --name test1-gpu --profile default --app geoclip
```

### 3. Deploy GPU-Optimized Ollama Service
```bash
# Deploy the GPU Ollama service
copilot svc deploy --name ollama --env test1-gpu
```

### 4. Update Chatbot Configuration
Update the chatbot manifest to use the GPU Ollama:
```yaml
# In copilot/chatbot/manifest.yml
environments:
  test1-gpu:
    variables:
      OLLAMA_API_URL: http://ollama.test1-gpu.geoclip.local:11434/api/generate
```

### 5. Deploy Updated Chatbot
```bash
# Deploy chatbot with GPU Ollama configuration
copilot svc deploy --name chatbot --env test1-gpu
```

## AWS Infrastructure Requirements

### ECS Cluster Configuration
- **Launch Type**: EC2 (required for GPU support)
- **Instance Type**: g4dn.2xlarge (8 vCPU, 32 GB RAM, 1 GPU)
- **AMI**: Amazon ECS-Optimized AMI with GPU support
- **Docker Runtime**: nvidia-docker2

### g4dn.2xlarge Specifications
- **vCPUs**: 8
- **Memory**: 32 GB
- **GPU**: 1x NVIDIA T4 (16 GB GPU memory)
- **Network**: Up to 25 Gbps
- **Storage**: NVMe SSD

## Performance Improvements

### Expected Performance Gains
- **Inference Speed**: 3-5x faster than CPU-only
- **Model Loading**: Faster model initialization
- **Concurrent Requests**: Better handling of multiple requests
- **Memory Efficiency**: Optimized GPU memory usage

### GPU Memory Optimization
- **Model Layers**: 35 GPU layers for optimal performance
- **Memory Allocation**: Automatic GPU memory management
- **Cache Management**: Efficient model caching

## Monitoring and Troubleshooting

### Health Check
```bash
# Check GPU Ollama health
copilot svc exec --name ollama --env test1-gpu --command "curl http://localhost:11434/api/tags"
```

### GPU Monitoring
```bash
# Check GPU usage
copilot svc exec --name ollama --env test1-gpu --command "nvidia-smi"
```

### Model Status
```bash
# Check loaded models
copilot svc exec --name ollama --env test1-gpu --command "ollama list"
```

### Logs
```bash
# Monitor GPU Ollama logs
copilot svc logs --name ollama --env test1-gpu --follow
```

## Cost Considerations

### g4dn.2xlarge Pricing (us-east-1)
- **On-Demand**: ~$0.752/hour
- **Spot**: ~$0.225/hour (70% savings)
- **Reserved**: ~$0.376/hour (1-year)

### Cost Optimization
- Use Spot instances for non-production workloads
- Implement auto-scaling based on demand
- Consider reserved instances for production
- Monitor GPU utilization for right-sizing

## Migration from CPU to GPU

### 1. Backup Current Configuration
```bash
# Export current service configuration
copilot svc show --name ollama --env test1 --json > ollama-backup.json
```

### 2. Deploy GPU Service
```bash
# Deploy new GPU service
copilot svc deploy --name ollama --env test1-gpu
```

### 3. Update Service Discovery
```bash
# Update chatbot to use GPU Ollama
copilot svc deploy --name chatbot --env test1-gpu
```

### 4. Test and Validate
```bash
# Test GPU Ollama performance
curl -X POST http://ollama-url:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "tinyllama", "prompt": "Hello, how are you?"}'
```

## Troubleshooting

### Common Issues
1. **GPU Not Available**: Ensure EC2 instances have GPU support
2. **CUDA Errors**: Check NVIDIA drivers and Docker runtime
3. **Memory Issues**: Monitor GPU memory usage
4. **Model Loading**: Check model download and initialization

### Debug Commands
```bash
# Check GPU availability
nvidia-smi

# Check Ollama GPU support
ollama list

# Monitor GPU memory
watch -n 1 nvidia-smi

# Check container GPU access
docker run --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Performance Tuning
```bash
# Adjust GPU layers for different models
export OLLAMA_GPU_LAYERS=35  # For larger models
export OLLAMA_GPU_LAYERS=20  # For smaller models

# Monitor GPU utilization
nvidia-smi dmon -s pucvmet -d 1
```

## Best Practices

1. **Resource Allocation**: Allocate sufficient CPU and memory for GPU workloads
2. **Model Selection**: Choose appropriate models for GPU memory
3. **Monitoring**: Set up CloudWatch alarms for GPU utilization
4. **Scaling**: Use auto-scaling based on GPU utilization metrics
5. **Backup**: Maintain CPU fallback for critical workloads

## Environment Variables

### GPU Optimization Variables
```bash
CUDA_VISIBLE_DEVICES=0          # Specify GPU device
OLLAMA_GPU_LAYERS=35           # Number of GPU layers
OLLAMA_HOST=0.0.0.0            # Bind to all interfaces
OLLAMA_ORIGINS=*               # Allow all origins
```

### Performance Tuning
```bash
# For different model sizes
OLLAMA_GPU_LAYERS=35  # Large models (7B+)
OLLAMA_GPU_LAYERS=20  # Medium models (3B-7B)
OLLAMA_GPU_LAYERS=10  # Small models (<3B)
```

## Conclusion

The GPU-optimized Ollama deployment provides significant performance improvements:

- **3-5x faster inference times**
- **Better resource utilization**
- **Improved user experience**
- **Scalable architecture**

Follow this guide to deploy the GPU-optimized version and monitor the performance improvements. The g4dn.2xlarge instance provides an excellent balance of performance and cost for Ollama workloads. 