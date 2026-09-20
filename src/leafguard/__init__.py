"""LeafGuard: crop leaf disease classification.

Modules:
    data     -- dataset loading & transforms (torchvision ImageFolder-based)
    model    -- model definitions (baseline CNN + transfer-learning heads)
    train    -- training loop, meant to be run on Kaggle (GPU)
    predict  -- lightweight inference helpers, safe to run on CPU/laptop
"""

__version__ = "0.1.0"
