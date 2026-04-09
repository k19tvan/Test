# Problem 03 Questions

## Multiple Choice

1. **What is the mathematical definition of the inverse sigmoid (logit) function?**
   - A) $1 / (1 + e^{-x})$
   - B) $\log(x / (1-x))$
   - C) $e^x / (1 + e^x)$
   - D) $\sin^{-1}(x)$
   - Answer: B

2. **Why is clamping necessary in the inverse sigmoid implementation?**
   - A) To improve computational speed
   - B) To prevent log(0) and division by zero errors
   - C) To reduce memory usage
   - D) To normalize outputs
   - Answer: B

3. **What should inverse_sigmoid(0.5) return?**
   - A) -1
   - B) 0
   - C) 0.5
   - D) 1
   - Answer: B

4. **If x is clamped to [eps, 1-eps] where eps=1e-5, what is the approximate range of inverse_sigmoid(x)?**
   - A) [-1, 1]
   - B) [0, 1]
   - C) [-10.6, 10.6]
   - D) [-100, 100]
   - Answer: C

5. **What is the relationship between sigmoid and inverse_sigmoid?**
   - A) sigmoid(inverse_sigmoid(x)) = x for x in (0,1)
   - B) They are independent functions
   - C) inverse_sigmoid always returns 0
   - D) sigmoid is faster than inverse_sigmoid
   - Answer: A

## Answer Key
1.B 2.B 3.B 4.C 5.A
