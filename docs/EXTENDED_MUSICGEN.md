# Extended MusicGen

This document explains how to use the Extended MusicGen model, which builds on top of a pretrained MusicGen model by adding additional transformer layers while keeping the original model parameters frozen.

## Overview

The Extended MusicGen model adds a stack of new transformer layers on top of a frozen pretrained MusicGen model. This approach allows for:

1. Fine-tuning on new data while preserving the knowledge in the original model
2. Reducing computational requirements compared to full model fine-tuning
3. Avoiding catastrophic forgetting of the pre-trained model's capabilities

## Usage

### Training an Extended MusicGen Model

To train an Extended MusicGen model, you'll need to:

1. Have a pretrained MusicGen checkpoint
2. Configure the training using the provided configuration file
3. Run the training command

Example command:

```bash
python -m audiocraft.train --config=solver/extended_musicgen pretrained_model_checkpoint=/path/to/musicgen_checkpoint.pt dora.dir=/path/for/experiment/logs
```

The important parameters to set are:

- `pretrained_model_checkpoint`: Path to the pretrained MusicGen model checkpoint
- `extended_model.num_additional_layers`: Number of additional transformer layers to add (default: 12)

You can also customize other parameters like the learning rate, transformer architecture, etc. See the configuration file at `configs/solver/extended_musicgen.yaml` for all available options.

### Using a Trained Extended MusicGen Model

After training, you can use your extended model just like the regular MusicGen model:

```python
import torch
from audiocraft.models.extended_musicgen import ExtendedMusicGen

# Load the model
model = ExtendedMusicGen.get_pretrained("/path/to/your/extended_model_checkpoint.pt")

# Generate audio
wav = model.generate(
    descriptions=["happy rock music with electric guitar and drums"],
    progress=True,
)

# Save audio
import torchaudio
torchaudio.save("generated_audio.wav", wav[0].cpu(), sample_rate=model.sample_rate)
```

### Evaluation

You can evaluate your extended model using the same evaluation tools as regular MusicGen:

```bash
python -m audiocraft.evaluate --config=solver/extended_musicgen sig=your_extended_model_signature
```

## Implementation Details

The Extended MusicGen implementation consists of three main components:

1. `ExtendedLMModel`: Extends the base `LMModel` by adding additional transformer layers while freezing the original model parameters
2. `ExtendedMusicGen`: A wrapper class that uses the extended LM model
3. `ExtendedMusicGenSolver`: A solver for training the extended model

When training the model:
1. The original model parameters are loaded and frozen
2. New transformer layers are initialized and added on top
3. Only the parameters of the new layers are updated during training

This approach allows for efficient adaptation of the pretrained model to new domains or tasks. 