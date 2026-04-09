# Problem 09 Theory - Gate Mechanism

Gating enables the model to learn optimal mixing of two feature streams:
$$\lambda_1(x_1, x_2) = \sigma(W_1[x_1; x_2])$$
$$\lambda_2(x_1, x_2) = \sigma(W_2[x_1; x_2])$$
$$\text{output} = \lambda_1 x_1 + \lambda_2 x_2$$

Benefits: Adaptive mixing based on content, learnable importance weights, smooth gradients.
