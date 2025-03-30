import argparse
import os
import sys
import json
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiocraft.data.audio_dataset import AudioDataset, AudioMeta
from audiocraft.modules.conditioners import ConditioningAttributes
from audiocraft_peft.speech_musicgen import SpeechMusicGen

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_audio_metadata(data_dir: str, sample_rate: int = 32000):
    """Load audio metadata from a directory"""
    audio_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                filepath = os.path.join(root, file)
                # Get corresponding JSON file if exists
                json_path = os.path.splitext(filepath)[0] + '.json'
                if os.path.exists(json_path):
                    audio_paths.append((filepath, json_path))
                else:
                    audio_paths.append((filepath, None))
    
    metadata = []
    for audio_path, json_path in audio_paths:
        # You would typically read duration from audio file
        # For simplicity, we'll use a placeholder here
        # In a real implementation, use librosa or soundfile to get duration
        duration = 30.0  # placeholder
        
        meta = AudioMeta(
            path=audio_path,
            duration=duration,
            sample_rate=sample_rate,
        )
        metadata.append(meta)
    
    return metadata


def create_dataset(data_dir: str, sample_rate: int = 32000, segment_duration: float = 30.0):
    """Create an audio dataset from a directory"""
    metadata = load_audio_metadata(data_dir, sample_rate)
    if not metadata:
        raise ValueError(f"No audio files found in {data_dir}")
    
    dataset = AudioDataset(
        metadata,
        segment_duration=segment_duration,
        sample_rate=sample_rate,
        channels=1,  # MusicGen works with mono
        return_info=True,  # Return metadata along with audio
    )
    
    return dataset


def get_attributes_from_info(info, is_speech=True):
    """Convert info to conditioning attributes"""
    # Get description from JSON file if it exists
    description = None
    json_path = os.path.splitext(info.meta.path)[0] + '.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                metadata = json.load(f)
                description = metadata.get('description', '')
            except json.JSONDecodeError:
                pass
    
    # If no description found, use a default one
    if not description:
        if is_speech:
            description = "Speech audio"
        else:
            description = "Music audio"
    
    # Create conditioning attributes
    attributes = ConditioningAttributes(text={'description': description})
    
    return attributes


def train_epoch(model, dataloader, optimizer, device, epoch, is_speech=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (audio, info) in enumerate(tqdm(dataloader)):
        audio = audio.to(device)
        
        # Get audio tokens from compression model
        with torch.no_grad():
            tokens, scale = model.compression_model.encode(audio)
        
        # Get conditioning attributes
        attributes = [get_attributes_from_info(inf, is_speech) for inf in info]
        
        # Shift tokens to create targets (autoregressive prediction)
        inputs = tokens[:, :, :-1]
        targets = tokens[:, :, 1:]
        
        # Create a padding mask (all tokens are valid in this case)
        padding_mask = torch.ones_like(inputs, dtype=torch.bool)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model.lm.compute_predictions(inputs, attributes, keep_only_valid_steps=True)
        logits = outputs.logits
        
        # Compute loss
        loss = 0
        for k in range(logits.shape[1]):  # Loop over codebooks
            loss += F.cross_entropy(
                logits[:, k].reshape(-1, logits.shape[-1]), 
                targets[:, k].reshape(-1),
                ignore_index=model.lm.card  # Special padding token
            )
        loss = loss / logits.shape[1]  # Average over codebooks
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = create_dataset(
        args.train_dir, 
        sample_rate=args.sample_rate,
        segment_duration=args.segment_duration,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collater,
    )
    
    if args.val_dir:
        val_dataset = create_dataset(
            args.val_dir,
            sample_rate=args.sample_rate,
            segment_duration=args.segment_duration,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=val_dataset.collater,
        )
    else:
        val_loader = None
    
    # Create model
    logger.info(f"Loading pretrained model: {args.model_id}")
    model = SpeechMusicGen.from_pretrained(
        args.model_id,
        adapter_dim=args.adapter_dim,
        adapter_dropout=args.adapter_dropout,
        add_adapters_to=args.adapter_layers,
        freeze_embeddings=args.freeze_embeddings,
        add_speech_prefix_embedding=args.add_speech_prefix,
        device=device,
    )
    
    # Log parameter count
    param_counts = model.count_parameters(only_trainable=True)
    logger.info(f"Model parameter counts: {param_counts}")
    logger.info(f"Trainable parameters: {param_counts['total']:,} ({param_counts['trainable_percentage']:.2f}%)")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch+1, is_speech=True)
        logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for audio, info in tqdm(val_loader):
                    audio = audio.to(device)
                    tokens, scale = model.compression_model.encode(audio)
                    attributes = [get_attributes_from_info(inf, is_speech=True) for inf in info]
                    
                    inputs = tokens[:, :, :-1]
                    targets = tokens[:, :, 1:]
                    padding_mask = torch.ones_like(inputs, dtype=torch.bool)
                    
                    outputs = model.lm.compute_predictions(inputs, attributes, keep_only_valid_steps=True)
                    logits = outputs.logits
                    
                    batch_loss = 0
                    for k in range(logits.shape[1]):
                        batch_loss += F.cross_entropy(
                            logits[:, k].reshape(-1, logits.shape[-1]),
                            targets[:, k].reshape(-1),
                            ignore_index=model.lm.card
                        )
                    batch_loss = batch_loss / logits.shape[1]
                    val_loss += batch_loss.item()
            
            val_loss /= len(val_loader)
            logger.info(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(args.output_dir, 'best_model.pt'))
                logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
        }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MusicGen on speech data")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory with training audio files")
    parser.add_argument("--val_dir", type=str, default=None, help="Directory with validation audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--model_id", type=str, default="facebook/musicgen-small", help="Pretrained model ID")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of adapter layers")
    parser.add_argument("--adapter_dropout", type=float, default=0.1, help="Dropout rate for adapters")
    parser.add_argument("--adapter_layers", type=str, default="-1,-2,-3,-4", 
                        help="Comma-separated list of layer indices to add adapters to")
    parser.add_argument("--freeze_embeddings", action="store_true", help="Freeze token embeddings")
    parser.add_argument("--add_speech_prefix", action="store_true", help="Add speech prefix embedding")
    parser.add_argument("--sample_rate", type=int, default=32000, help="Audio sample rate")
    parser.add_argument("--segment_duration", type=float, default=30.0, help="Audio segment duration")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Convert adapter_layers from string to list of integers
    args.adapter_layers = [int(x) for x in args.adapter_layers.split(",")]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run main function
    main(args) 