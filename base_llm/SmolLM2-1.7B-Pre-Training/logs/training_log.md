# SmolLM2 Training Log

**Training Started:** 2025-06-21 16:21:13

## Configuration
- Model: HuggingFaceTB/SmolLM2-1.7B
- Dataset: /home/ubuntu/bigdata/Training/Day4/cosmopedia-v2-1B/cosmopedia-v2-1B-tokenized
- Batch Size per Device: 1
- Gradient Accumulation: 16
- Learning Rate: 2e-05
- Max Train Steps: 50,000
- Warmup Steps: 1000
- Log Interval: 500 steps
- Checkpoint Interval: 2000 steps
- Generation Interval: 2000 steps
- Output Directory: /home/ubuntu/bigdata/Training/Day4/cosmopedia-v2-1B/smollm-1.7B-cosmo-1B-production

## Training Progress

| Step | Loss | Learning Rate | GPU Memory | Generation Sample |
|------|------|---------------|------------|-------------------|
**2025-06-21 16:21:13** - Training started with Accelerate

| Step 500/50,000 | Epoch 0 | Loss: 7.8503 | LR: 2.00e-05 | GPU: 6.1GB | - |
**2025-06-21 16:37:29** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 16:37:29** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 16:37:29** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 16:37:29** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 1,000/50,000 | Epoch 0 | Loss: 6.1849 | LR: 1.98e-05 | GPU: 6.1GB | - |
**2025-06-21 16:52:34** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 16:52:34** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 16:52:34** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 16:52:34** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 1,500/50,000 | Epoch 0 | Loss: 5.7781 | LR: 1.95e-05 | GPU: 6.1GB | - |
**2025-06-21 17:07:39** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 17:07:39** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 17:07:39** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 17:07:39** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-21 17:22:45** - GENERATION at step 2000: The weather today is very , and cultural characters, often, and historical narratives of the world. They have had been a critical aspect of the concept of the importance of the lens

| Step 2,000/50,000 | Epoch 0 | Loss: 5.5012 | LR: 1.90e-05 | GPU: 6.1GB | The weather today is very , and cultural character... |
**2025-06-21 17:22:45** - GPU 0: 6.1GB/23.7GB (70% util)

**2025-06-21 17:22:45** - GPU 1: 0.0GB/23.7GB (70% util)

**2025-06-21 17:22:45** - GPU 2: 0.0GB/23.7GB (70% util)

**2025-06-21 17:22:45** - GPU 3: 0.0GB/23.7GB (70% util)

**2025-06-21 17:22:53** - CHECKPOINT: Saved at step 2,000, loss 5.2628

| Step 2,500/50,000 | Epoch 0 | Loss: 5.3206 | LR: 1.84e-05 | GPU: 6.1GB | - |
**2025-06-21 17:37:56** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 17:37:56** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 17:37:56** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 17:37:56** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 3,000/50,000 | Epoch 0 | Loss: 5.2509 | LR: 1.76e-05 | GPU: 6.1GB | - |
**2025-06-21 17:52:59** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 17:52:59** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 17:52:59** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 17:52:59** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 3,500/50,000 | Epoch 0 | Loss: 5.0630 | LR: 1.67e-05 | GPU: 6.1GB | - |
**2025-06-21 18:08:03** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 18:08:03** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 18:08:03** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 18:08:03** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-21 18:23:07** - GENERATION at step 4000: The weather today is very a powerful aspect of people who. You have that you have to do with a beautiful and family, but it's just a big and a new way

| Step 4,000/50,000 | Epoch 0 | Loss: 4.9602 | LR: 1.57e-05 | GPU: 6.1GB | The weather today is very a powerful aspect of peo... |
**2025-06-21 18:23:07** - GPU 0: 6.1GB/23.7GB (70% util)

**2025-06-21 18:23:07** - GPU 1: 0.0GB/23.7GB (70% util)

**2025-06-21 18:23:07** - GPU 2: 0.0GB/23.7GB (70% util)

**2025-06-21 18:23:07** - GPU 3: 0.0GB/23.7GB (70% util)

**2025-06-21 18:23:16** - CHECKPOINT: Saved at step 4,000, loss 5.3045

| Step 4,500/50,000 | Epoch 0 | Loss: 4.9255 | LR: 1.46e-05 | GPU: 6.2GB | - |
**2025-06-21 18:38:20** - GPU 0: 6.2GB/23.7GB (100% util)

**2025-06-21 18:38:20** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 18:38:20** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 18:38:20** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 5,000/50,000 | Epoch 0 | Loss: 4.7725 | LR: 1.35e-05 | GPU: 6.1GB | - |
**2025-06-21 18:53:23** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 18:53:23** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 18:53:23** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 18:53:23** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 5,500/50,000 | Epoch 0 | Loss: 4.7407 | LR: 1.22e-05 | GPU: 6.1GB | - |
**2025-06-21 19:08:29** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 19:08:29** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 19:08:29** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 19:08:29** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-21 19:23:32** - GENERATION at step 6000: The weather today is very a crucial question that involves into the world of mathematics, including humans, and other stakeholders. This process involves using the language, data analysis, and analysis

| Step 6,000/50,000 | Epoch 0 | Loss: 4.7152 | LR: 1.10e-05 | GPU: 6.1GB | The weather today is very a crucial question that ... |
**2025-06-21 19:23:32** - GPU 0: 6.1GB/23.7GB (70% util)

**2025-06-21 19:23:32** - GPU 1: 0.0GB/23.7GB (69% util)

**2025-06-21 19:23:32** - GPU 2: 0.0GB/23.7GB (70% util)

**2025-06-21 19:23:32** - GPU 3: 0.0GB/23.7GB (70% util)

**2025-06-21 19:23:41** - CHECKPOINT: Saved at step 6,000, loss 3.4940

| Step 6,500/50,000 | Epoch 0 | Loss: 4.6269 | LR: 9.68e-06 | GPU: 6.1GB | - |
**2025-06-21 19:38:46** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 19:38:46** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 19:38:46** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 19:38:46** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 7,000/50,000 | Epoch 0 | Loss: 4.5694 | LR: 8.40e-06 | GPU: 6.1GB | - |
**2025-06-21 19:53:48** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 19:53:48** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 19:53:48** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 19:53:48** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 7,500/50,000 | Epoch 0 | Loss: 4.5104 | LR: 7.15e-06 | GPU: 6.1GB | - |
**2025-06-21 20:08:54** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 20:08:54** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 20:08:54** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 20:08:54** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-21 20:24:00** - GENERATION at step 8000: The weather today is very a well of a community project. A concept that allows people to communicate and work effectively, leading to the potential impact of an individual's mental health.

| Step 8,000/50,000 | Epoch 0 | Loss: 4.5203 | LR: 5.95e-06 | GPU: 6.2GB | The weather today is very a well of a community pr... |
**2025-06-21 20:24:00** - GPU 0: 6.2GB/23.7GB (70% util)

**2025-06-21 20:24:00** - GPU 1: 0.0GB/23.7GB (69% util)

**2025-06-21 20:24:00** - GPU 2: 0.0GB/23.7GB (69% util)

**2025-06-21 20:24:00** - GPU 3: 0.0GB/23.7GB (70% util)

**2025-06-21 20:24:09** - CHECKPOINT: Saved at step 8,000, loss 4.0437

| Step 8,500/50,000 | Epoch 0 | Loss: 4.4695 | LR: 4.82e-06 | GPU: 6.1GB | - |
**2025-06-21 20:39:15** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 20:39:15** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 20:39:15** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 20:39:15** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 9,000/50,000 | Epoch 0 | Loss: 4.4569 | LR: 3.77e-06 | GPU: 6.1GB | - |
**2025-06-21 20:54:20** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 20:54:20** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 20:54:20** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 20:54:20** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 9,500/50,000 | Epoch 0 | Loss: 4.4486 | LR: 2.82e-06 | GPU: 6.1GB | - |
**2025-06-21 21:09:26** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 21:09:26** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 21:09:26** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 21:09:26** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-21 21:24:31** - GENERATION at step 10000: The weather today is very a powerful aspect of art-based that has gained a fascinating way to make a unique and beautiful country. This genre of art has been a popular language

| Step 10,000/50,000 | Epoch 0 | Loss: 4.4154 | LR: 1.99e-06 | GPU: 6.1GB | The weather today is very a powerful aspect of art... |
**2025-06-21 21:24:31** - GPU 0: 6.1GB/23.7GB (70% util)

**2025-06-21 21:24:31** - GPU 1: 0.0GB/23.7GB (69% util)

**2025-06-21 21:24:31** - GPU 2: 0.0GB/23.7GB (70% util)

**2025-06-21 21:24:31** - GPU 3: 0.0GB/23.7GB (70% util)

**2025-06-21 21:24:40** - CHECKPOINT: Saved at step 10,000, loss 5.1218

| Step 10,500/50,000 | Epoch 0 | Loss: 4.3977 | LR: 1.29e-06 | GPU: 6.1GB | - |
**2025-06-21 21:39:47** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 21:39:47** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 21:39:47** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 21:39:47** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 11,000/50,000 | Epoch 0 | Loss: 4.3568 | LR: 7.31e-07 | GPU: 6.2GB | - |
**2025-06-21 21:54:51** - GPU 0: 6.2GB/23.7GB (100% util)

**2025-06-21 21:54:51** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 21:54:51** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 21:54:51** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 11,500/50,000 | Epoch 0 | Loss: 4.3245 | LR: 3.27e-07 | GPU: 6.0GB | - |
**2025-06-21 22:09:54** - GPU 0: 6.0GB/23.7GB (100% util)

**2025-06-21 22:09:54** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 22:09:54** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 22:09:54** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-21 22:24:57** - GENERATION at step 12000: The weather today is very a crucial concept that involves in a critical period, including the law of the legal system. This section will explore the significance of this field, including its

| Step 12,000/50,000 | Epoch 0 | Loss: 4.3306 | LR: 8.21e-08 | GPU: 6.1GB | The weather today is very a crucial concept that i... |
**2025-06-21 22:24:57** - GPU 0: 6.1GB/23.7GB (69% util)

**2025-06-21 22:24:57** - GPU 1: 0.0GB/23.7GB (69% util)

**2025-06-21 22:24:57** - GPU 2: 0.0GB/23.7GB (69% util)

**2025-06-21 22:24:57** - GPU 3: 0.0GB/23.7GB (70% util)

**2025-06-21 22:25:06** - CHECKPOINT: Saved at step 12,000, loss 4.6746

| Step 12,500/50,000 | Epoch 0 | Loss: 4.3886 | LR: 0.00e+00 | GPU: 6.1GB | - |
**2025-06-21 22:40:10** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 22:40:10** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 22:40:10** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 22:40:10** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 13,000/50,000 | Epoch 0 | Loss: 4.3293 | LR: 8.21e-08 | GPU: 6.1GB | - |
**2025-06-21 22:55:14** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 22:55:14** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 22:55:14** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 22:55:14** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 13,500/50,000 | Epoch 0 | Loss: 4.3531 | LR: 3.27e-07 | GPU: 6.1GB | - |
**2025-06-21 23:10:18** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 23:10:18** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 23:10:18** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 23:10:18** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-21 23:25:22** - GENERATION at step 14000: The weather today is very a critical concept that is a crucial aspect of technology and culture, particularly in the context of education. This course unit will delve into the concept of music

| Step 14,000/50,000 | Epoch 0 | Loss: 4.3190 | LR: 7.31e-07 | GPU: 6.1GB | The weather today is very a critical concept that ... |
**2025-06-21 23:25:22** - GPU 0: 6.1GB/23.7GB (69% util)

**2025-06-21 23:25:22** - GPU 1: 0.0GB/23.7GB (70% util)

**2025-06-21 23:25:22** - GPU 2: 0.0GB/23.7GB (70% util)

**2025-06-21 23:25:22** - GPU 3: 0.0GB/23.7GB (70% util)

**2025-06-21 23:25:31** - CHECKPOINT: Saved at step 14,000, loss 4.2570

| Step 14,500/50,000 | Epoch 0 | Loss: 4.3156 | LR: 1.29e-06 | GPU: 6.1GB | - |
**2025-06-21 23:40:38** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 23:40:38** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 23:40:38** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 23:40:38** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 15,000/50,000 | Epoch 0 | Loss: 4.3567 | LR: 1.99e-06 | GPU: 6.1GB | - |
**2025-06-21 23:55:45** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-21 23:55:45** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-21 23:55:45** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-21 23:55:45** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 15,500/50,000 | Epoch 0 | Loss: 4.3016 | LR: 2.82e-06 | GPU: 6.1GB | - |
**2025-06-22 00:10:50** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 00:10:50** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 00:10:50** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 00:10:50** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-22 00:25:56** - GENERATION at step 16000: The weather today is very a small tool in your daily of digital books. It's like having a few, but instead of using the right tool, you can use your computer

| Step 16,000/50,000 | Epoch 0 | Loss: 4.3221 | LR: 3.77e-06 | GPU: 6.1GB | The weather today is very a small tool in your dai... |
**2025-06-22 00:25:56** - GPU 0: 6.1GB/23.7GB (69% util)

**2025-06-22 00:25:56** - GPU 1: 0.0GB/23.7GB (70% util)

**2025-06-22 00:25:56** - GPU 2: 0.0GB/23.7GB (69% util)

**2025-06-22 00:25:56** - GPU 3: 0.0GB/23.7GB (70% util)

**2025-06-22 00:26:05** - CHECKPOINT: Saved at step 16,000, loss 3.9602

| Step 16,500/50,000 | Epoch 0 | Loss: 4.3530 | LR: 4.82e-06 | GPU: 6.1GB | - |
**2025-06-22 00:41:11** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 00:41:11** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 00:41:11** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 00:41:11** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 17,000/50,000 | Epoch 0 | Loss: 4.3509 | LR: 5.95e-06 | GPU: 6.1GB | - |
**2025-06-22 00:56:17** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 00:56:17** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 00:56:17** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 00:56:17** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 17,500/50,000 | Epoch 0 | Loss: 4.2952 | LR: 7.15e-06 | GPU: 6.1GB | - |
**2025-06-22 01:11:21** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 01:11:21** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 01:11:21** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 01:11:21** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-22 01:26:26** - GENERATION at step 18000: The weather today is very a critical aspect of self and development that involves a single and effective aspect. This field of life has long been a popular and diverse experience, allowing individuals

| Step 18,000/50,000 | Epoch 0 | Loss: 4.3690 | LR: 8.40e-06 | GPU: 6.1GB | The weather today is very a critical aspect of sel... |
**2025-06-22 01:26:26** - GPU 0: 6.1GB/23.7GB (69% util)

**2025-06-22 01:26:26** - GPU 1: 0.0GB/23.7GB (69% util)

**2025-06-22 01:26:26** - GPU 2: 0.0GB/23.7GB (69% util)

**2025-06-22 01:26:26** - GPU 3: 0.0GB/23.7GB (70% util)

**2025-06-22 01:26:35** - CHECKPOINT: Saved at step 18,000, loss 4.3448

| Step 18,500/50,000 | Epoch 0 | Loss: 4.2890 | LR: 9.68e-06 | GPU: 6.1GB | - |
**2025-06-22 01:41:42** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 01:41:42** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 01:41:42** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 01:41:42** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 19,000/50,000 | Epoch 0 | Loss: 4.3087 | LR: 1.10e-05 | GPU: 6.1GB | - |
**2025-06-22 01:56:46** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 01:56:46** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 01:56:46** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 01:56:46** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 19,500/50,000 | Epoch 0 | Loss: 4.2962 | LR: 1.22e-05 | GPU: 6.1GB | - |
**2025-06-22 02:11:48** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 02:11:48** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 02:11:48** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 02:11:48** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-22 02:26:54** - GENERATION at step 20000: The weather today is very a type-friendly technique that deals with computers to keep us engaged, keep track of the world around them. One such tool is the "ditors

| Step 20,000/50,000 | Epoch 0 | Loss: 4.2597 | LR: 1.35e-05 | GPU: 6.1GB | The weather today is very a type-friendly techniqu... |
**2025-06-22 02:26:54** - GPU 0: 6.1GB/23.7GB (69% util)

**2025-06-22 02:26:54** - GPU 1: 0.0GB/23.7GB (70% util)

**2025-06-22 02:26:54** - GPU 2: 0.0GB/23.7GB (69% util)

**2025-06-22 02:26:54** - GPU 3: 0.0GB/23.7GB (70% util)

**2025-06-22 02:27:03** - CHECKPOINT: Saved at step 20,000, loss 4.0342

| Step 20,500/50,000 | Epoch 0 | Loss: 4.2683 | LR: 1.46e-05 | GPU: 6.2GB | - |
**2025-06-22 02:42:09** - GPU 0: 6.2GB/23.7GB (100% util)

**2025-06-22 02:42:09** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 02:42:09** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 02:42:09** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 21,000/50,000 | Epoch 0 | Loss: 4.2512 | LR: 1.57e-05 | GPU: 6.1GB | - |
**2025-06-22 02:57:13** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 02:57:13** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 02:57:13** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 02:57:13** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 21,500/50,000 | Epoch 0 | Loss: 4.2517 | LR: 1.67e-05 | GPU: 6.1GB | - |
**2025-06-22 03:12:16** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 03:12:16** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 03:12:16** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 03:12:16** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-22 03:27:19** - GENERATION at step 22000: The weather today is very the world for children and family. It is an important aspect that can help us understand and understand the world around us. Today, we will explore what

| Step 22,000/50,000 | Epoch 0 | Loss: 4.2321 | LR: 1.76e-05 | GPU: 6.1GB | The weather today is very the world for children a... |
**2025-06-22 03:27:19** - GPU 0: 6.1GB/23.7GB (69% util)

**2025-06-22 03:27:19** - GPU 1: 0.0GB/23.7GB (69% util)

**2025-06-22 03:27:19** - GPU 2: 0.0GB/23.7GB (69% util)

**2025-06-22 03:27:19** - GPU 3: 0.0GB/23.7GB (70% util)

**2025-06-22 03:27:28** - CHECKPOINT: Saved at step 22,000, loss 3.8393

| Step 22,500/50,000 | Epoch 0 | Loss: 4.2269 | LR: 1.84e-05 | GPU: 6.1GB | - |
**2025-06-22 03:42:36** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 03:42:36** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 03:42:36** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 03:42:36** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 23,000/50,000 | Epoch 0 | Loss: 4.2159 | LR: 1.90e-05 | GPU: 6.1GB | - |
**2025-06-22 03:57:40** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 03:57:40** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 03:57:40** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 03:57:40** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 23,500/50,000 | Epoch 0 | Loss: 4.1050 | LR: 1.95e-05 | GPU: 6.1GB | - |
**2025-06-22 04:12:44** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 04:12:44** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 04:12:44** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 04:12:44** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-22 04:27:50** - GENERATION at step 24000: The weather today is very a type of study that allows us to create their own images and stories. These characters often use their emotions, thoughts, and emotions, making them a

| Step 24,000/50,000 | Epoch 0 | Loss: 4.1012 | LR: 1.98e-05 | GPU: 6.1GB | The weather today is very a type of study that all... |
**2025-06-22 04:27:50** - GPU 0: 6.1GB/23.7GB (69% util)

**2025-06-22 04:27:50** - GPU 1: 0.0GB/23.7GB (70% util)

**2025-06-22 04:27:50** - GPU 2: 0.0GB/23.7GB (70% util)

**2025-06-22 04:27:50** - GPU 3: 0.0GB/23.7GB (71% util)

**2025-06-22 04:27:59** - CHECKPOINT: Saved at step 24,000, loss 3.9619

| Step 24,500/50,000 | Epoch 0 | Loss: 4.0592 | LR: 2.00e-05 | GPU: 6.1GB | - |
**2025-06-22 04:43:04** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 04:43:04** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 04:43:04** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 04:43:04** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 25,000/50,000 | Epoch 0 | Loss: 4.0911 | LR: 2.00e-05 | GPU: 6.1GB | - |
**2025-06-22 04:58:11** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 04:58:11** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 04:58:11** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 04:58:11** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 25,500/50,000 | Epoch 0 | Loss: 4.0646 | LR: 1.98e-05 | GPU: 6.1GB | - |
**2025-06-22 05:13:14** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 05:13:14** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 05:13:14** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 05:13:14** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-22 05:28:19** - GENERATION at step 26000: The weather today is very a natural term that affects how the physical is becoming a complex and high system. It involves the importance of any individual, such as the physical space,

| Step 26,000/50,000 | Epoch 0 | Loss: 3.9852 | LR: 1.95e-05 | GPU: 6.1GB | The weather today is very a natural term that affe... |
**2025-06-22 05:28:19** - GPU 0: 6.1GB/23.7GB (70% util)

**2025-06-22 05:28:19** - GPU 1: 0.0GB/23.7GB (70% util)

**2025-06-22 05:28:19** - GPU 2: 0.0GB/23.7GB (70% util)

**2025-06-22 05:28:19** - GPU 3: 0.0GB/23.7GB (71% util)

**2025-06-22 05:28:27** - CHECKPOINT: Saved at step 26,000, loss 3.7547

| Step 26,500/50,000 | Epoch 0 | Loss: 4.0014 | LR: 1.90e-05 | GPU: 6.1GB | - |
**2025-06-22 05:43:36** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 05:43:36** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 05:43:36** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 05:43:36** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 27,000/50,000 | Epoch 0 | Loss: 3.9675 | LR: 1.84e-05 | GPU: 6.1GB | - |
**2025-06-22 05:58:39** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 05:58:39** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 05:58:39** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 05:58:39** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 27,500/50,000 | Epoch 0 | Loss: 3.9442 | LR: 1.76e-05 | GPU: 6.1GB | - |
**2025-06-22 06:13:42** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 06:13:42** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 06:13:42** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 06:13:42** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-22 06:28:44** - GENERATION at step 28000: The weather today is very an important tool in business, but did you know that there are also many ways to get better and more people who need them? That's right!

| Step 28,000/50,000 | Epoch 0 | Loss: 3.8837 | LR: 1.67e-05 | GPU: 6.2GB | The weather today is very an important tool in bus... |
**2025-06-22 06:28:44** - GPU 0: 6.2GB/23.7GB (69% util)

**2025-06-22 06:28:44** - GPU 1: 0.0GB/23.7GB (70% util)

**2025-06-22 06:28:44** - GPU 2: 0.0GB/23.7GB (69% util)

**2025-06-22 06:28:44** - GPU 3: 0.0GB/23.7GB (70% util)

**2025-06-22 06:28:53** - CHECKPOINT: Saved at step 28,000, loss 4.6750

| Step 28,500/50,000 | Epoch 0 | Loss: 3.8894 | LR: 1.57e-05 | GPU: 6.1GB | - |
**2025-06-22 06:43:54** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 06:43:54** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 06:43:54** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 06:43:54** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 29,000/50,000 | Epoch 0 | Loss: 3.8714 | LR: 1.46e-05 | GPU: 6.1GB | - |
**2025-06-22 06:58:58** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 06:58:58** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 06:58:58** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 06:58:58** - GPU 3: 0.0GB/23.7GB (100% util)

| Step 29,500/50,000 | Epoch 0 | Loss: 3.8255 | LR: 1.35e-05 | GPU: 6.1GB | - |
**2025-06-22 07:13:57** - GPU 0: 6.1GB/23.7GB (100% util)

**2025-06-22 07:13:57** - GPU 1: 0.0GB/23.7GB (100% util)

**2025-06-22 07:13:57** - GPU 2: 0.0GB/23.7GB (100% util)

**2025-06-22 07:13:57** - GPU 3: 0.0GB/23.7GB (100% util)

**2025-06-22 07:28:55** - GENERATION at step 30000: The weather today is very a powerful task for many people who have been recognized for centuries in their daily lives and place. Today, we're going to learn about some important things

| Step 30,000/50,000 | Epoch 0 | Loss: 3.8318 | LR: 1.22e-05 | GPU: 6.1GB | The weather today is very a powerful task for many... |
**2025-06-22 07:28:55** - GPU 0: 6.1GB/23.7GB (69% util)

**2025-06-22 07:28:55** - GPU 1: 0.0GB/23.7GB (70% util)

**2025-06-22 07:28:55** - GPU 2: 0.0GB/23.7GB (69% util)

**2025-06-22 07:28:55** - GPU 3: 0.0GB/23.7GB (70% util)

**2025-06-22 07:29:04** - CHECKPOINT: Saved at step 30,000, loss 4.4253

**2025-06-22 07:29:35** - FINAL: Training ended at step 30011

**2025-06-22 07:29:41** - Final model saved to /home/ubuntu/bigdata/Training/Day4/cosmopedia-v2-1B/smollm-1.7B-cosmo-1B-production/final_model

