# Speech-MusicGen: Parameter-Efficient Fine-Tuning for AudioCraft

This extension to Meta's AudioCraft provides parameter-efficient fine-tuning capabilities for adapting MusicGen to speech generation while preserving its music generation capabilities.

## Features

- **Parameter-Efficient Fine-Tuning**: Uses adapters to fine-tune a small subset of parameters, typically reducing trainable parameters by >99%.
- **Preservation of Music Generation**: Keeps the original MusicGen capability intact while adding speech generation.
- **Modular Design**: Easily configurable adapter layers for different levels of capacity.
- **Easy Integration**: Built on top of the existing AudioCraft framework with minimal modifications.

## Installation

1. First, set up AudioCraft according to its installation instructions
2. Copy this `audiocraft_peft` directory to your project

## Usage

### Fine-tuning on Speech Data

```bash
python audiocraft_peft/train.py \
    --train_dir /path/to/speech/dataset \
    --val_dir /path/to/speech/validation \
    --output_dir ./speech_musicgen_checkpoints \
    --model_id facebook/musicgen-small \
    --batch_size 8 \
    --epochs 5 \
    --lr 1e-4 \
    --adapter_dim 64 \
    --adapter_layers -1,-2,-3,-4 \
    --freeze_embeddings \
    --add_speech_prefix
```

### Generating with the Fine-tuned Model

```bash
python audiocraft_peft/generate.py \
    --checkpoint_path ./speech_musicgen_checkpoints/best_model.pt \
    --base_model_id facebook/musicgen-small \
    --speech_prompt "A female narrator explaining quantum physics in a calm, clear voice" \
    --music_prompt "A jazzy piano piece with upbeat rhythm and melodic solos" \
    --output_dir ./generated_samples \
    --duration 10.0
```

## Speech Dataset Format

The speech dataset should be organized as follows:

```
speech_dataset/
├── sample1.wav  # Audio file
├── sample1.json  # Optional metadata with 'description' field
├── sample2.wav
├── sample2.json
...
```

Each JSON file should contain at least:
```json
{
  "description": "A detailed description of the speech content"
}
```

## Implementation Details

The parameter-efficient fine-tuning is implemented using adapter layers that:

1. Add small bottleneck adapters after selected transformer layers
2. Freeze the original model weights
3. Train only the adapter parameters (~0.1-1% of total parameters)
4. Optionally add a learnable speech-specific prefix embedding

## Citation

If you use this work, please cite the original AudioCraft paper:

```
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
}
``` 