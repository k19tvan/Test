# Problem 14 - Full TransformerDecoder (Capstone)

## Description
Complete iterative refinement decoder. Stacks TransformerDecoderLayers and updates reference points across iterations. This is the entire decoder in one module - the capstone of the learning path.

## Core Loop
For each layer i=0 to num_layers-1:
1. Pass target through decoder layer i
2. Predict bboxes from refined features
3. Update reference points for next layer
4. Accumulate predictions for all layers

## Input/Output
- target: (B, num_queries, D)
- ref_points_unact: (B, num_queries, 4)
- memory: (B, HW, D)
- spatial_shapes: list
- Returns: Stacked predictions across all layers

Key: Iteratively refines predictions, each layer sees progressively better estimates.
