# Problem 04 Questions

## Multiple Choice

1. **In reference point generation for multi-scale pyramids, what does the stride parameter represent?**
   - A) Learning rate stride
   - B) Downsampling factor from original image to feature map
   - C) Temporal stride in video
   - D) Batch processing stride
   - Answer: B

2. **If eval_spatial_size=(512, 512) and feat_strides=[8, 16, 32], what is the total number of anchor points across all levels?**
   - A) 512*512
   - B) 64*64 + 32*32 + 16*16
   - C) 4096 + 1024 + 256 = 5376
   - D) 8 + 16 + 32 = 56
   - Answer: C

3. **Why are coordinates normalized to [0, 1] for reference point generation?**
   - A) To fit in GPU memory
   - B) To make model size-agnostic and enable batch processing
   - C) To reduce computational cost
   - D) To apply data augmentation
   - Answer: B

4. **In log-sigmoid space, what coordinate range do valid anchors typically fall in?**
   - A) [0, 1]
   - B) [-1, 1]
   - C) [-10, 10]
   - D) Depends on eps value
   - Answer: C

5. **What is the purpose of the valid_mask in reference point generation?**
   - A) To store which levels are used
   - B) To mark valid points (in [eps, 1-eps]) to avoid unstable gradients at boundaries
   - C) To speed up computation
   - D) To apply dropout
   - Answer: B

## Answer Key
1.B 2.C 3.B 4.C 5.B
