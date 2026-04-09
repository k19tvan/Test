# Problem 10 - LQE (Location Quality Estimator)

## Description
Predict location quality scores for each prediction by analyzing the corner distribution statistics. Takes top-k probabilities and mean, feeds to MLP.

## Formula
$$\text{score} = \text{score} + \text{MLP}([\text{topk\_probs}; \text{mean\_prob}])$$

Where topk_probs are the k largest softmax probabilities from the distribution.

## Input/Output
- scores: (B, L, num_classes)
- pred_corners: (B, L, 4, reg_max+1)
- output: refined scores (B, L, num_classes)
