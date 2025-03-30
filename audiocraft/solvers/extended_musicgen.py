# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import omegaconf
import torch
import typing as tp

from .musicgen import MusicGenSolver
from ..models.extended_lm import ExtendedLMModel
from .compression import CompressionSolver
from . import builders as solver_builders
from ..models import builders as model_builders


class ExtendedMusicGenSolver(MusicGenSolver):
    """Solver for ExtendedMusicGen training task.
    This solver extends MusicGenSolver to train an ExtendedLMModel where
    the original MusicGen model parameters are frozen and only the
    added transformer layers are trained.
    """
    
    def build_model(self) -> None:
        """Instantiate models and optimizer.
        Override the base method to load a pretrained model and extend it.
        """
        # Load the compression model first (similar to parent method)
        self.compression_model = CompressionSolver.wrapped_model_from_checkpoint(
            self.cfg, self.cfg.compression_model_checkpoint, device=self.device)
        
        # Verify that we have matching configuration between LM and compression model
        assert self.cfg.transformer_lm.card == self.compression_model.cardinality, (
            "Cardinalities of the LM and compression model don't match: ",
            f"LM cardinality is {self.cfg.transformer_lm.card} vs ",
            f"compression model cardinality is {self.compression_model.cardinality}"
        )
        assert self.cfg.transformer_lm.n_q == self.compression_model.num_codebooks, (
            "Numbers of codebooks of the LM and compression models don't match: ",
            f"LM number of codebooks is {self.cfg.transformer_lm.n_q} vs ",
            f"compression model numer of codebooks is {self.compression_model.num_codebooks}"
        )
        
        self.logger.info("Compression model has %d codebooks with %d cardinality, and a framerate of %d",
                         self.compression_model.num_codebooks, self.compression_model.cardinality,
                         self.compression_model.frame_rate)
        
        # We need to temporarily set lm_model to transformer_lm to get the base model
        original_lm_model = self.cfg.lm_model
        tmp_cfg = omegaconf.OmegaConf.create(omegaconf.OmegaConf.to_container(self.cfg, resolve=True))
        tmp_cfg.lm_model = "transformer_lm"
        
        # Load the pretrained model from checkpoint
        self.logger.info("Loading pretrained model from %s", self.cfg.pretrained_model_checkpoint)
        
        base_model = model_builders.get_lm_model(tmp_cfg)
        
        # Restore the original lm_model value
        self.cfg.lm_model = original_lm_model
        
        # Create the extended model by adding new layers on top of the frozen pretrained model
        self.model = ExtendedLMModel(
            base_model, 
            num_additional_layers=self.cfg.extended_model.num_additional_layers,
            **self.cfg.extended_model.transformer_args
        )
        self.model = self.model.to(self.device)
        
        # Initialize optimization
        self.initialize_optimization()
        
    def log_training_setup(self) -> None:
        """Log the training setup, including which parameters are frozen vs. trainable."""
        super().log_training_setup()
        
        # Count and log frozen vs trainable parameters
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        frozen_params = total_params - trainable_params
        self.logger.info(f"Model has {total_params:,} total parameters")
        self.logger.info(f"  - {trainable_params:,} trainable parameters ({trainable_params/total_params:.2%})")
        self.logger.info(f"  - {frozen_params:,} frozen parameters ({frozen_params/total_params:.2%})")
        
        # Log info about the new transformer layers
        self.logger.info(f"Added {self.cfg.extended_model.num_additional_layers} trainable transformer layers") 