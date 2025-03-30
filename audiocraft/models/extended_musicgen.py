# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Extended MusicGen model that adds additional transformer layers 
on top of a pre-trained MusicGen model.
"""

import typing as tp

import torch

from .musicgen import MusicGen, MelodyType
from .extended_lm import ExtendedLMModel


class ExtendedMusicGen(MusicGen):
    """Extended MusicGen model that adds additional transformer layers on top of 
    a pre-trained MusicGen model while keeping the original model parameters frozen.

    Args:
        name (str): Name of the model.
        compression_model: Compression model from the original MusicGen.
        extended_lm (ExtendedLMModel): Extended language model.
        max_duration (float, optional): Maximum duration the model can produce.
    """
    @staticmethod
    def get_pretrained(name: str = 'facebook/musicgen-melody', device=None, 
                       num_additional_layers: int = 12, **kwargs):
        """Return pretrained ExtendedMusicGen model with additional transformer layers.
        
        Args:
            name (str): Name of the pretrained MusicGen model to extend.
            device: Device to load the model on.
            num_additional_layers (int): Number of additional transformer layers to add.
            **kwargs: Additional arguments for the additional transformer layers.
            
        Returns:
            ExtendedMusicGen: Extended MusicGen model
        """
        # First get the pretrained MusicGen model
        musicgen = MusicGen.get_pretrained(name, device=device)
        
        # Create the extended LM model
        extended_lm = ExtendedLMModel(musicgen.lm, num_additional_layers, **kwargs)
        
        # Create the extended MusicGen model
        return ExtendedMusicGen(name, musicgen.compression_model, extended_lm, 
                               max_duration=musicgen.max_duration)
    
    def generate(self, descriptions: tp.List[str], 
                progress: bool = False, return_tokens: bool = False) -> tp.Union[torch.Tensor, 
                                                                              tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate audio samples conditioned on text.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process.
            return_tokens (bool, optional): Return tokens along with generated audio.
            
        Returns:
            torch.Tensor or tuple(torch.Tensor, torch.Tensor): Generated audio and optionally tokens.
        """
        return super().generate(descriptions, progress=progress, return_tokens=return_tokens)
        
    def generate_with_chroma(self, descriptions: tp.List[str], melody_wavs: MelodyType,
                             melody_sample_rate: int, progress: bool = False,
                             return_tokens: bool = False) -> tp.Union[torch.Tensor,
                                                                      tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate audio samples conditioned on text and melody.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            melody_wavs (torch.Tensor or list of Tensor): Batch of waveforms used as melody conditioning.
            melody_sample_rate (int): Sample rate of the melody waveforms.
            progress (bool, optional): Flag to display progress of the generation process.
            return_tokens (bool, optional): Return tokens along with generated audio.
            
        Returns:
            torch.Tensor or tuple(torch.Tensor, torch.Tensor): Generated audio and optionally tokens.
        """
        return super().generate_with_chroma(descriptions, melody_wavs, melody_sample_rate, 
                                          progress=progress, return_tokens=return_tokens) 