# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Extended Language Model that adds additional transformer layers 
on top of a pre-trained MusicGen model.
"""

import typing as tp
import logging
import torch
from torch import nn

from .lm import LMModel, LMOutput, ConditionTensors, CFGConditions, State
from ..modules.transformer import StreamingTransformer, create_norm_fn
from ..modules.conditioners import ConditioningAttributes

logger = logging.getLogger(__name__)


class ExtendedLMModel(LMModel):
    """Transformer-based language model that extends a pre-trained MusicGen model
    with additional transformer layers, while keeping the original model frozen.

    Args:
        pretrained_model (LMModel): Pre-trained MusicGen LM model to extend
        num_additional_layers (int): Number of additional transformer layers to add
        **kwargs: Additional parameters for the transformer layers
    """
    def __init__(self, pretrained_model: LMModel, num_additional_layers: int = 12, **kwargs):
        # Don't call parent's __init__ as we'll initialize from the pretrained model
        nn.Module.__init__(self)
        
        # Copy attributes from the pretrained model
        self.cfg_coef = pretrained_model.cfg_coef
        self.cfg_dropout = pretrained_model.cfg_dropout
        self.att_dropout = pretrained_model.att_dropout
        self.condition_provider = pretrained_model.condition_provider
        self.fuser = pretrained_model.fuser
        self.card = pretrained_model.card
        self.n_q = pretrained_model.n_q
        self.dim = pretrained_model.dim
        self.pattern_provider = pretrained_model.pattern_provider
        self.two_step_cfg = pretrained_model.two_step_cfg
        self.emb = pretrained_model.emb
        self.attn_mask_per_stage = getattr(pretrained_model, 'attn_mask_per_stage', {})  # Copy attention masks if they exist
        
        # Keep a reference to the pretrained transformer
        self.pretrained_transformer = pretrained_model.transformer
        
        # Create a new transformer with additional layers
        # We use the same configuration as the pretrained model
        # First, we need to extract the parameters by looking at the first layer of the pretrained transformer
        first_layer = pretrained_model.transformer.layers[0]

        # Extract parameters from the first layer as these aren't directly accessible from the transformer
        # Create a dictionary of default parameters
        default_params = {}

        # Extract parameters from the first layer
        default_params['num_heads'] = first_layer.self_attn.num_heads
        default_params['dim_feedforward'] = first_layer.linear1.out_features  # Dimension of the hidden layer in FF
        default_params['dropout'] = first_layer.dropout.p if hasattr(first_layer, 'dropout') else 0.0
        default_params['bias_ff'] = first_layer.linear1.bias is not None
        default_params['bias_attn'] = first_layer.self_attn.in_proj_bias is not None if hasattr(first_layer.self_attn, 'in_proj_bias') else True

        # Get norm info
        norm_type = type(first_layer.norm1).__name__.lower()
        if "layer" in norm_type:
            default_params['norm'] = "layer_norm"
        elif "rmsnorm" in norm_type:
            default_params['norm'] = "rms_norm"
        else:
            default_params['norm'] = norm_type

        # Check for norm_first
        default_params['norm_first'] = hasattr(first_layer, 'norm_first') and first_layer.norm_first

        # Check if cross-attention is used
        default_params['cross_attention'] = hasattr(first_layer, 'cross_attention') and first_layer.cross_attention is not None

        # Check for other important attributes
        default_params['custom'] = getattr(first_layer.self_attn, 'custom', False)
        default_params['memory_efficient'] = getattr(first_layer.self_attn, 'memory_efficient', False)
        default_params['attention_as_float32'] = getattr(first_layer.self_attn, 'attention_as_float32', False)
        default_params['causal'] = getattr(first_layer.self_attn, 'causal', False)

        # Only include parameters that aren't in kwargs to avoid conflicts
        for key in list(kwargs.keys()):
            if key in default_params:
                default_params.pop(key)

        # Get the norm value for later use if needed
        norm = default_params.get('norm', 'layer_norm')

        # Now initialize the additional transformer with the extracted parameters
        self.additional_transformer = StreamingTransformer(
            d_model=self.dim,
            num_layers=num_additional_layers,
            **default_params,
            **kwargs
        )
        
        # Copy the output normalization layer if it exists
        self.out_norm = None
        if pretrained_model.out_norm is not None:
            self.out_norm = create_norm_fn(norm, self.dim)
            self.out_norm.load_state_dict(pretrained_model.out_norm.state_dict())
        
        # Copy the output projection layers
        self.linears = pretrained_model.linears
        
        # Freeze the pretrained model parameters
        self._freeze_pretrained_parameters()
        
        # Initialize the streaming state
        self._is_streaming = False
        self._streaming_state = None
        self._fsdp = None

    def _freeze_pretrained_parameters(self):
        """Freeze all parameters of the pretrained model."""
        # Freeze embeddings
        for param in self.emb.parameters():
            param.requires_grad = False
            
        # Freeze pretrained transformer
        for param in self.pretrained_transformer.parameters():
            param.requires_grad = False
            
        # Freeze output projection layers
        for param in self.linears.parameters():
            param.requires_grad = False
            
        # Freeze output norm if it exists
        if self.out_norm is not None:
            for param in self.out_norm.parameters():
                param.requires_grad = False
                
        # Freeze condition provider
        for param in self.condition_provider.parameters():
            param.requires_grad = False
            
        # Freeze condition fuser
        for param in self.fuser.parameters():
            param.requires_grad = False
            
        logger.info("Frozen all parameters of the pretrained model.")

    def forward(self, sequence: torch.Tensor,
                conditions: tp.List[ConditioningAttributes],
                condition_tensors: tp.Optional[ConditionTensors] = None,
                stage: int = -1) -> torch.Tensor:
        """Apply the extended language model on sequence and conditions.
        First applies the pretrained (frozen) model up to its transformer,
        then applies the additional transformer layers, and finally
        applies the output projection.

        Args:
            sequence (torch.Tensor): Indices of the codes to model.
            conditions (list of ConditioningAttributes): Conditions to use when modeling
                the given codes.
            condition_tensors (dict[str, ConditionType], optional): Pre-computed conditioning
                tensors.
            stage (int): The codebook level being predicted.
        
        Returns:
            torch.Tensor: Logits.
        """
        B, K, S = sequence.shape
        assert K == self.num_codebooks, "Sequence shape must match the specified number of codebooks"
        
        # Compute input embeddings (same as original model)
        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)])
        
        if condition_tensors is None:
            assert not self._is_streaming, "Conditions tensors should be precomputed when streaming."
            # Apply dropout modules
            conditions = self.cfg_dropout(conditions)
            conditions = self.att_dropout(conditions)
            tokenized = self.condition_provider.tokenize(conditions)
            # Encode conditions and fuse
            condition_tensors = self.condition_provider(tokenized)
        else:
            assert not conditions, "Shouldn't pass both conditions and condition_tensors."

        # Fuse input with conditions
        input_, cross_attention_input = self.fuser(input_, condition_tensors)

        # Apply pretrained transformer (frozen)
        pretrained_out = self.pretrained_transformer(
            input_, 
            cross_attention_src=cross_attention_input,
            src_mask=(self.attn_mask_per_stage[stage] if stage >= 0 else None)
        )
        
        # Apply additional transformer layers (trainable)
        out = self.additional_transformer(
            pretrained_out,
            cross_attention_src=cross_attention_input,
            src_mask=(self.attn_mask_per_stage[stage] if stage >= 0 else None)
        )
        
        if self.out_norm:
            out = self.out_norm(out)
            
        # Apply output projections
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)  # [B, K, S, card]

        # Remove the prefix from the model outputs
        if len(self.fuser.fuse2cond['prepend']) > 0:
            logits = logits[:, :, -S:]

        return logits  # [B, K, S, card]
        
    def streaming(self):
        """Context manager to activate streaming mode."""
        class StreamingContext:
            def __init__(self, model):
                self.model = model
                
            def __enter__(self):
                self.model._is_streaming = True
                self.model._streaming_state = {
                    'pretrained': None,
                    'additional': None,
                    'fuser': None, 
                    'condition_provider': None
                }
                return self.model
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.model._is_streaming = False
                self.model._streaming_state = None
                # Update to handle both clear_cache and the older reset_cache methods
                for module in [self.model.pretrained_transformer, self.model.additional_transformer, 
                              self.model.fuser, self.model.condition_provider]:
                    try:
                        if hasattr(module, 'clear_cache'):
                            module.clear_cache()
                        elif hasattr(module, 'reset_cache'):
                            module.reset_cache()
                    except Exception as e:
                        print(f"Warning: Could not clear cache for {type(module).__name__}: {e}")
                
        return StreamingContext(self)
    
    def get_streaming_state(self) -> State:
        """Get the streaming state for the model."""
        assert self._is_streaming, "Only call this method when in streaming mode."
        if self._streaming_state['pretrained'] is None:
            try:
                self._streaming_state['pretrained'] = self.pretrained_transformer.get_streaming_state()
            except Exception as e:
                print(f"Warning: Could not get streaming state for pretrained_transformer: {e}")
                self._streaming_state['pretrained'] = {}
        if self._streaming_state['additional'] is None:
            try:
                self._streaming_state['additional'] = self.additional_transformer.get_streaming_state()
            except Exception as e:
                print(f"Warning: Could not get streaming state for additional_transformer: {e}")
                self._streaming_state['additional'] = {}
        if self._streaming_state['fuser'] is None:
            try:
                self._streaming_state['fuser'] = self.fuser.get_streaming_state()
            except Exception as e:
                print(f"Warning: Could not get streaming state for fuser: {e}")
                self._streaming_state['fuser'] = {}
        if self._streaming_state['condition_provider'] is None:
            try:
                self._streaming_state['condition_provider'] = self.condition_provider.get_streaming_state()
            except Exception as e:
                print(f"Warning: Could not get streaming state for condition_provider: {e}")
                self._streaming_state['condition_provider'] = {}
        return self._streaming_state
    
    def set_streaming_state(self, state: State):
        """Set the streaming state for the model."""
        assert self._is_streaming, "Only call this method when in streaming mode."
        self._streaming_state = state
        try:
            if 'pretrained' in state and state['pretrained'] is not None:
                self.pretrained_transformer.set_streaming_state(state['pretrained'])
        except Exception as e:
            print(f"Warning: Could not set streaming state for pretrained_transformer: {e}")
        
        try:
            if 'additional' in state and state['additional'] is not None:
                self.additional_transformer.set_streaming_state(state['additional'])
        except Exception as e:
            print(f"Warning: Could not set streaming state for additional_transformer: {e}")
        
        try:
            if 'fuser' in state and state['fuser'] is not None:
                self.fuser.set_streaming_state(state['fuser'])
        except Exception as e:
            print(f"Warning: Could not set streaming state for fuser: {e}")
        
        try:
            if 'condition_provider' in state and state['condition_provider'] is not None:
                self.condition_provider.set_streaming_state(state['condition_provider'])
        except Exception as e:
            print(f"Warning: Could not set streaming state for condition_provider: {e}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str, num_additional_layers: int = 12, device=None, **kwargs):
        """Create an ExtendedLMModel from a pretrained MusicGen model.
        
        Args:
            pretrained_model_name (str): Name of the pretrained model to load
            num_additional_layers (int): Number of additional transformer layers
            device: Device to load the model on
            **kwargs: Additional arguments for the additional transformer
            
        Returns:
            ExtendedLMModel: Extended LM model
        """
        from .musicgen import MusicGen
        
        # Load the pretrained MusicGen model
        musicgen = MusicGen.get_pretrained(pretrained_model_name, device=device)
        
        # Create the extended model
        extended_model = cls(musicgen.lm, num_additional_layers, **kwargs)
        
        return extended_model 