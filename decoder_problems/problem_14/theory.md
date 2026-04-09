# Problem 14 Theory

Coarse-to-fine refinement via iteration:
- Layer 0: Coarse predictions from encoder
- Layer 1: Refine bboxes based on layer 0
- ...
- Layer 5: Final fine-grained predictions

Each layer sees:
- Initial features from layer 0
- Accumulated corrections from previous layers
- Updated reference points

This enables progressive refinement where each layer focuses on local refinement around current best estimate.
