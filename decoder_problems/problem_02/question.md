# Problem 02 Questions

## Multiple Choice

1. **What is the primary purpose of translating GT to distribution bins?**
   - A) To reduce memory usage
   - B) To create soft labels for training distribution-based regression heads
   - C) To speed up inference
   - D) To apply data augmentation
   - Answer: B

2. **In the translate_gt function, what do weight_left and weight_right represent?**
   - A) Loss weights for left and right halves of the image
   - B) Interpolation weights for soft labeling across two adjacent bins
   - C) Channel weights for convolutional filters
   - D) Learning rates for left and right predictions
   - Answer: B

3. **For a valid bin translation, what should weight_left + weight_right equal?**
   - A) 0.0
   - B) 0.5
   - C) 1.0
   - D) 2.0
   - Answer: C

4. **How many bins does a weighting function with reg_max=15 create?**
   - A) 15
   - B) 16
   - C) 17
   - D) 30
   - Answer: B

5. **If a GT value exactly matches a bin position from the weighting function, what should happen?**
   - A) weight_left = 0.0, weight_right = 1.0
   - B) weight_left = 1.0, weight_right = 0.0
   - C) weight_left = weight_right = 0.5
   - D) Both weights become 0
   - Answer: B

## Answer Key
1.B 2.B 3.C 4.B 5.B
