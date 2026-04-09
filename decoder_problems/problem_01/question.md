# Problem 01 Questions

## Multiple Choice

1. **What is the primary purpose of the weighting function in D-FINE?**
   - A) To increase the learning rate during training
   - B) To assign higher importance to central bins in distribution refinement
   - C) To normalize batch statistics
   - D) To apply dropout to predictions
   - Answer: B

2. **If reg_max=15 and we compute weighting_function(), how many output bins should we get?**
   - A) 14
   - B) 15
   - C) 16
   - D) 17
   - Answer: C

3. **In the weighting function formula w(i) = 1/(1 + up * |i - center|), what does increasing the `up` parameter do?**
   - A) Makes edge weights larger
   - B) Makes edge weights smaller (steeper decay)
   - C) Has no effect on weights
   - D) Flips the distribution
   - Answer: B

4. **For a weighting function with reg_max=9 and up=2.0, which bin index has the maximum weight?**
   - A) 0
   - B) 4
   - C) 4.5 (center)
   - D) 9
   - Answer: C

5. **What is the mathematical relationship between w(i) and w(reg_max - i) in the weighting function?**
   - A) w(i) > w(reg_max - i)
   - B) w(i) < w(reg_max - i)
   - C) w(i) = w(reg_max - i) (symmetric)
   - D) No consistent relationship
   - Answer: C

## Answer Key
1.B 2.C 3.B 4.C 5.C
