"""
Model packages live under `tessgen.models.<name>`.

Each model subpackage owns:
- core model code
- LightningModule wrapper
- training / tuning / reporting entrypoints
"""

__all__ = ["surrogate", "edge", "node_diffusion"]

