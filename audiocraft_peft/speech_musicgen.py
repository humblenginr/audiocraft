import typing as tp
import torch
import torch.nn as nn
from copy import deepcopy

import sys
import os
# Add parent directory to path to import from audiocraft
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiocraft.models.musicgen import MusicGen
from audiocraft.models.lm import LMModel
from audiocraft.models.encodec import CompressionModel

from audiocraft_peft.adapters import AdapterLayer, TransformerAdapterWrapper


class SpeechMusicGenLM(LMModel):
    """Extended LMModel with adapter layers for speech fine-tuning.
    
    This class wraps the original LMModel from MusicGen and adds adapter layers
    while freezing the original model parameters.
    
    Args:
        base_lm (LMModel): Original language model to adapt
        adapter_dim (int): Dimension of the adapter bottleneck
        adapter_dropout (float): Dropout rate for adapters
        add_adapters_to (list): List of indices where to add adapters (e.g. [-1, -2] for last two layers)
        freeze_embeddings (bool): Whether to freeze the token embeddings
        add_speech_prefix_embedding (bool): Whether to add a special speech prefix embedding
    """
    
    def __init__(
        self,
        base_lm: LMModel,
        adapter_dim: int = 64,
        adapter_dropout: float = 0.1,
        add_adapters_to: tp.List[int] = [-1, -2, -3, -4],
        freeze_embeddings: bool = True, 
        add_speech_prefix_embedding: bool = False
    ):
        # Initialize with same parameters as the base model
        super().__init__(
            pattern_provider=base_lm.pattern_provider,
            condition_provider=base_lm.condition_provider,
            fuser=base_lm.fuser,
            n_q=base_lm.n_q,
            card=base_lm.card,
            dim=base_lm.dim,
        )
        
        # Copy the base model attributes that we need
        self.cfg_coef = base_lm.cfg_coef
        self.cfg_dropout = base_lm.cfg_dropout
        self.att_dropout = base_lm.att_dropout
        self.two_step_cfg = base_lm.two_step_cfg
        
        # Copy the token embeddings
        self.emb = base_lm.emb
        
        # Freeze embeddings if requested
        if freeze_embeddings:
            for param in self.emb.parameters():
                param.requires_grad = False
        
        # Create a copy of the transformer with adapters
        # We use deepcopy to ensure we don't modify the original model
        self.transformer = deepcopy(base_lm.transformer)
        
        # Freeze the transformer parameters
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # Add adapters to specified layers
        for i in add_adapters_to:
            layer_idx = i if i >= 0 else len(self.transformer.layers) + i
            if 0 <= layer_idx < len(self.transformer.layers):
                # Replace the layer with a wrapped version
                original_layer = self.transformer.layers[layer_idx]
                self.transformer.layers[layer_idx] = TransformerAdapterWrapper(
                    original_layer, adapter_dim, adapter_dropout
                )
                
        # Copy the output layers and freeze them
        self.out_norm = base_lm.out_norm
        self.linears = base_lm.linears
        
        for param in self.linears.parameters():
            param.requires_grad = False
            
        if self.out_norm is not None:
            for param in self.out_norm.parameters():
                param.requires_grad = False
                
        # Add an optional speech token embeddings to differentiate speech from music
        self.add_speech_prefix_embedding = add_speech_prefix_embedding
        if add_speech_prefix_embedding:
            # Add a special speech token embedding
            self.speech_embedding = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)
            
    def forward(self, sequence: torch.Tensor, 
                conditions: tp.List, 
                condition_tensors: tp.Optional[tp.Dict[str, tp.Any]] = None,
                stage: int = -1) -> torch.Tensor:
        """Forward pass with optional speech token prefix"""
        
        # Regular embedding
        B, K, S = sequence.shape
        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)])
        
        # Add speech embedding if requested - a special embedding 
        # that gets prepended to the input to signal this is speech
        if hasattr(self, 'add_speech_prefix_embedding') and self.add_speech_prefix_embedding:
            # Add a learnable "speech" embedding at the beginning
            speech_emb = self.speech_embedding.expand(B, -1, -1)
            input_ = torch.cat([speech_emb, input_], dim=1)
            
        # Process conditions
        if condition_tensors is None:
            assert not self._is_streaming, "Conditions tensors should be precomputed when streaming."
            # apply dropout modules
            conditions = self.cfg_dropout(conditions)
            conditions = self.att_dropout(conditions)
            tokenized = self.condition_provider.tokenize(conditions)
            condition_tensors = self.condition_provider(tokenized)
            
        # Apply the model
        outputs = self.fuser(input_, condition_tensors, self.transformer)
        
        # Remove the speech embedding token if we added it
        if hasattr(self, 'add_speech_prefix_embedding') and self.add_speech_prefix_embedding:
            outputs = outputs[:, 1:, :]
            
        if self.out_norm is not None:
            outputs = self.out_norm(outputs)
            
        # Get logits from outputs
        logits = torch.stack([linear(outputs) for linear in self.linears], dim=1)
        
        return logits


class SpeechMusicGen(MusicGen):
    """MusicGen model adapted for speech generation using parameter-efficient fine-tuning.
    
    This class extends MusicGen to use a customized LM with adapter modules,
    allowing fine-tuning for speech while preserving music generation capabilities.
    
    Args:
        base_musicgen (MusicGen): Original MusicGen model to adapt
        adapter_dim (int): Dimension of the adapter bottleneck
        adapter_dropout (float): Dropout rate for adapters
        add_adapters_to (list): List of indices where to add adapters (e.g. [-1, -2] for last two layers)
        freeze_embeddings (bool): Whether to freeze the token embeddings
        add_speech_prefix_embedding (bool): Whether to add a special speech prefix embedding
    """
    
    def __init__(
        self,
        base_musicgen: MusicGen,
        adapter_dim: int = 64,
        adapter_dropout: float = 0.1,
        add_adapters_to: tp.List[int] = [-1, -2, -3, -4],
        freeze_embeddings: bool = True,
        add_speech_prefix_embedding: bool = False
    ):
        # Extract the components from the base model
        name = f"{base_musicgen.name}_speech_adapted"
        compression_model = base_musicgen.compression_model
        
        # Create the adapted LM
        adapted_lm = SpeechMusicGenLM(
            base_musicgen.lm,
            adapter_dim=adapter_dim,
            adapter_dropout=adapter_dropout,
            add_adapters_to=add_adapters_to,
            freeze_embeddings=freeze_embeddings,
            add_speech_prefix_embedding=add_speech_prefix_embedding
        )
        
        # Initialize with the adapted components
        super().__init__(
            name=name,
            compression_model=compression_model,
            lm=adapted_lm,
            max_duration=base_musicgen.max_duration
        )
        
        # Copy generation parameters
        self.duration = base_musicgen.duration
        self.generation_params = base_musicgen.generation_params
        self.extend_stride = base_musicgen.extend_stride
    
    @staticmethod
    def from_pretrained(
        model_id: str = 'facebook/musicgen-small',
        adapter_dim: int = 64,
        adapter_dropout: float = 0.1,
        add_adapters_to: tp.List[int] = [-1, -2, -3, -4],
        freeze_embeddings: bool = True,
        add_speech_prefix_embedding: bool = False,
        device=None
    ) -> 'SpeechMusicGen':
        """Creates a SpeechMusicGen model from a pretrained MusicGen model.
        
        Args:
            model_id (str): HuggingFace model ID or local path
            adapter_dim (int): Dimension of adapter layers
            adapter_dropout (float): Dropout rate for adapters
            add_adapters_to (list): List of indices where to add adapters
            freeze_embeddings (bool): Whether to freeze token embeddings
            add_speech_prefix_embedding (bool): Whether to add speech prefix embedding
            device (str): Device to load the model on
            
        Returns:
            SpeechMusicGen: The adapted model
        """
        # First load the base model
        base_model = MusicGen.get_pretrained(model_id, device=device)
        
        # Create the adapted model
        return SpeechMusicGen(
            base_model,
            adapter_dim=adapter_dim,
            adapter_dropout=adapter_dropout,
            add_adapters_to=add_adapters_to,
            freeze_embeddings=freeze_embeddings,
            add_speech_prefix_embedding=add_speech_prefix_embedding
        )
    
    def count_parameters(self, only_trainable: bool = True) -> tp.Dict[str, int]:
        """Count the number of parameters in the model.
        
        Args:
            only_trainable (bool): If True, only count trainable parameters
            
        Returns:
            dict: Dictionary with parameter counts for different components
        """
        def _count_params(module, only_trainable: bool = True):
            return sum(p.numel() for p in module.parameters() 
                      if not only_trainable or p.requires_grad)
        
        counts = {
            'total': _count_params(self, only_trainable),
            'lm_total': _count_params(self.lm, only_trainable),
            'compression_model': _count_params(self.compression_model, only_trainable),
        }
        
        # Add counts for specific components if we have access to them
        if hasattr(self.lm, 'transformer'):
            counts['transformer'] = _count_params(self.lm.transformer, only_trainable)
        
        if hasattr(self.lm, 'emb'):
            counts['embeddings'] = _count_params(self.lm.emb, only_trainable)
            
        if hasattr(self.lm, 'linears'):
            counts['output_projections'] = _count_params(self.lm.linears, only_trainable)
            
        # Calculate the parameter efficiency
        total_params = _count_params(self, False)  # All parameters
        trainable_params = _count_params(self, True)  # Only trainable
        
        counts['trainable_percentage'] = (trainable_params / total_params) * 100 if total_params > 0 else 0
        
        return counts 