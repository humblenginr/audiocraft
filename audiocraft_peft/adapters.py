import torch
import torch.nn as nn
import typing as tp
import math

class AdapterLayer(nn.Module):
    """
    Adapter layer for parameter-efficient fine-tuning
    
    Implements a bottleneck architecture that projects to a lower dimension,
    applies non-linearity, then projects back to the original dimension.
    
    Args:
        in_dim (int): Input dimension
        bottleneck_dim (int): Bottleneck dimension (smaller than in_dim)
        dropout (float): Dropout probability
        init_scale (float): Initial scale for the adapter output (smaller means closer to identity)
    """
    def __init__(self, in_dim: int, bottleneck_dim: int, dropout: float = 0.1, init_scale: float = 0.1):
        super().__init__()
        self.down = nn.Linear(in_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(in_dim)
        
        # Initialize to near-zero to start with near-identity
        with torch.no_grad():
            nn.init.normal_(self.up.weight, std=init_scale)
            nn.init.zeros_(self.up.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        x = residual + x
        x = self.norm(x)
        return x


class TransformerAdapterWrapper(nn.Module):
    """
    Wrapper for a transformer layer that adds an adapter after the layer.
    
    Args:
        transformer_layer: Original transformer layer to wrap
        adapter_dim (int): Adapter bottleneck dimension
        dropout (float): Dropout probability for the adapter
    """
    def __init__(self, transformer_layer: nn.Module, adapter_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer = transformer_layer
        
        # Determine the model dimension from the transformer layer
        if hasattr(transformer_layer, 'd_model'):
            model_dim = transformer_layer.d_model
        else:
            # Fallback to attempting to read from self_attn module
            model_dim = transformer_layer.self_attn.embed_dim
        
        self.adapter = AdapterLayer(model_dim, adapter_dim, dropout)
        
    def forward(self, x: torch.Tensor, *args, **kwargs):
        # Pass through the original layer
        x = self.layer(x, *args, **kwargs)
        # Then through the adapter
        return self.adapter(x)


class LoraLayer(nn.Module):
    """
    Implementation of Low-Rank Adaptation (LoRA) for efficient fine-tuning.
    
    Applies a low-rank update to a pre-existing linear layer.
    
    Args:
        original_layer (nn.Linear): Original linear layer to adapt
        rank (int): Rank of the update matrices
        alpha (float): Scaling factor for the update
        dropout (float): Dropout probability for the update path
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        self.original_layer = original_layer
        
        # Store original parameters
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Scaling factor
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA components
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize low-rank matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze the original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass
        original_output = self.original_layer(x)
        
        # LoRA forward pass
        lora_output = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        
        # Return combined output
        return original_output + (lora_output * self.scaling) 