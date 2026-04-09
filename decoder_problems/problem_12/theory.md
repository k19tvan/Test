# Problem 12 Theory

Multi-scale attention:
- Level 0: Fine details, many points
- Level N: Coarse semantics, fewer points
- Unified offset normalization ensures consistent sampling across scales

Offset normalization:
$$\text{sampling\_location} = \text{ref\_pt} + \frac{\text{offset}}{\text{spatial\_shape}}$$
