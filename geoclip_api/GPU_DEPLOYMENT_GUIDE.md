# GPU-Optimized API Deployment Guide

## Overview
This guide explains how to deploy the GeoCLIP API service on GPU-enabled infrastructure for significantly faster inference times.

## Files Created

1. **`Dockerfile-GPU.api`** - GPU-optimized Docker image with CUDA support
2. **`api-gpu.py`** - GPU-optimized API code with memory management
3. **`copilot/api-gpu/manifest.yml`** - Copilot manifest for GPU service
4. **`GPU_DEPLOYMENT_GUIDE.md`** - This deployment guide

## Key Optimizations

### Dockerfile-GPU.api
- **NVIDIA CUDA Base Image**: Uses `nvidia/cuda:11.8-devel-ubuntu20.04`
- **PyTorch with CUDA**: Installs PyTorch with CUDA 11.8 support
- **GPU Optimization Libraries**: Includes `accelerate`, `optimum`, `auto-gptq`, `bitsandbytes`
- **Memory Management**: Configures CUDA memory allocation settings
- **Startup Script**: Includes GPU verification and optimization

### api-gpu.py
- **GPU Detection**: Automatically detects and uses available GPUs
- **Memory Management**: Clears GPU cache before/after inference
- **Memory Fraction**: Limits GPU memory usage to 80% to prevent OOM
- **Enhanced Health Check**: Includes GPU information and memory stats
- **Single Worker**: Optimized for GPU usage with single worker

## Deployment Steps

### 1. Prerequisites
- AWS ECS cluster with GPU-enabled EC2 instances
- NVIDIA drivers and Docker runtime installed on EC2 instances
- Copilot CLI installed

### 2. Initialize GPU Service
```bash
# Initialize the GPU API service
copilot svc init --name api-gpu --svc-type "Backend Service" --dockerfile "./Dockerfile-GPU.api" --port 8000
```

### 3. Update Environment for GPU Support
```bash
# Create or update environment with GPU support
copilot env init --name test1-gpu --profile default --app geoclip
```

### 4. Deploy GPU Service
```bash
# Deploy the GPU API service
copilot svc deploy --name api-gpu --env test1-gpu
```

### 5. Update Chatbot Configuration
Update the chatbot manifest to use the GPU API:
```yaml
# In copilot/chatbot/manifest.yml
environments:
  test1-gpu:
    variables:
      API_URL: http://api-gpu.test1-gpu.geoclip.local:8000/predict
      OLLAMA_API_URL: http://ollama.test1-gpu.geoclip.local:11434/api/generate
```

### 6. Deploy Updated Chatbot
```bash
# Deploy chatbot with GPU API configuration
copilot svc deploy --name chatbot --env test1-gpu
```

## AWS Infrastructure Requirements

### ECS Cluster Configuration
- **Launch Type**: EC2 (required for GPU support)
- **Instance Type**: GPU-enabled instances (g4dn.xlarge, g5.xlarge, etc.)
- **AMI**: Amazon ECS-Optimized AMI with GPU support
- **Docker Runtime**: nvidia-docker2

### Instance Recommendations
- **g4dn.xlarge**: 4 vCPU, 16 GB RAM, 1 GPU (NVIDIA T4)
- **g5.xlarge**: 4 vCPU, 16 GB RAM, 1 GPU (NVIDIA A10G)
- **g4dn.2xlarge**: 8 vCPU, 32 GB RAM, 1 GPU (NVIDIA T4)

## Performance Improvements

### Expected Performance Gains
- **Inference Speed**: 5-10x faster than CPU-only
- **Throughput**: Higher concurrent request handling
- **Latency**: Reduced response times from 10-30 seconds to 2-5 seconds

### Memory Optimization
- **GPU Memory**: 8-16 GB recommended
- **System Memory**: 16-32 GB for optimal performance
- **Cache Management**: Automatic GPU cache clearing

## Monitoring and Troubleshooting

### Health Check
```bash
# Check GPU API health
copilot svc exec --name api-gpu --env test1-gpu --command "curl http://localhost:8000/health"
```

### GPU Monitoring
```bash
# Check GPU usage
copilot svc exec --name api-gpu --env test1-gpu --command "nvidia-smi"
```

### Logs
```bash
# Monitor GPU API logs
copilot svc logs --name api-gpu --env test1-gpu --follow
```

## Cost Considerations

### GPU Instance Pricing (us-east-1)
- **g4dn.xlarge**: ~$0.526/hour
- **g5.xlarge**: ~$1.006/hour
- **g4dn.2xlarge**: ~$0.752/hour

### Cost Optimization
- Use Spot instances for non-production workloads
- Implement auto-scaling based on demand
- Consider reserved instances for production

## Migration from CPU to GPU

### 1. Backup Current Configuration
```bash
# Export current service configuration
copilot svc show --name api --env test1 --json > api-backup.json
```

### 2. Deploy GPU Service
```bash
# Deploy new GPU service
copilot svc deploy --name api-gpu --env test1-gpu
```

### 3. Update Service Discovery
```bash
# Update chatbot to use GPU API
copilot svc deploy --name chatbot --env test1-gpu
```

### 4. Test and Validate
```bash
# Test GPU API performance
curl -X POST http://api-gpu-url:8000/predict -F "file=@test-image.jpg"
```

## Troubleshooting

### Common Issues
1. **GPU Not Available**: Ensure EC2 instances have GPU support
2. **CUDA Errors**: Check NVIDIA drivers and Docker runtime
3. **Memory Issues**: Adjust `PYTORCH_CUDA_ALLOC_CONF` settings
4. **Performance Issues**: Monitor GPU utilization and memory usage

### Debug Commands
```bash
# Check GPU availability
nvidia-smi

# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU memory
watch -n 1 nvidia-smi

# Check container GPU access
docker run --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

## Best Practices

1. **Resource Allocation**: Allocate sufficient CPU and memory for GPU workloads
2. **Memory Management**: Implement proper GPU memory cleanup
3. **Monitoring**: Set up CloudWatch alarms for GPU utilization
4. **Scaling**: Use auto-scaling based on GPU utilization metrics
5. **Backup**: Maintain CPU fallback for critical workloads

## Conclusion

The GPU-optimized deployment provides significant performance improvements for the GeoCLIP API. The key benefits include:

- **5-10x faster inference times**
- **Better resource utilization**
- **Improved user experience**
- **Scalable architecture**

Follow this guide to deploy the GPU-optimized version and monitor the performance improvements. 