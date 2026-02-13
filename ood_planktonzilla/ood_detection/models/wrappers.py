"""Model wrappers for consistent interface and DataParallel compatibility."""

import torch.nn as nn


class TorchModel(nn.Module):
    """
    Wrapper for HuggingFace models to ensure consistent tensor outputs.
    
    This wrapper is critical for DataParallel compatibility, as it ensures
    the model returns a Tensor instead of a dictionary or custom object,
    which allows DataParallel to properly gather results across GPUs.
    
    Args:
        hf_model: A HuggingFace model or similar model that may return
                  objects with a 'logits' attribute.
    """
    
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        """
        Forward pass that extracts logits if present.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor: Logits or raw output
        """
        out = self.model(x)
        # CRITICAL FOR DATA PARALLEL: 
        # Must return a Tensor, not a Dictionary/Object, so DP can gather results.
        if hasattr(out, "logits"):
            return out.logits
        return out
