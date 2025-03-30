import argparse
import os
import sys
import torch
import torchaudio
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiocraft.modules.conditioners import ConditioningAttributes
from audiocraft_peft.speech_musicgen import SpeechMusicGen

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_finetuned_model(checkpoint_path, base_model_id, device):
    """Load a fine-tuned model from a checkpoint"""
    # First create a base model with the same configuration
    model = SpeechMusicGen.from_pretrained(
        model_id=base_model_id,
        adapter_dim=64,  # Must match the value used during training
        adapter_dropout=0.1,
        add_adapters_to=[-1, -2, -3, -4],
        freeze_embeddings=True,
        add_speech_prefix_embedding=True,
        device=device,
    )
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded fine-tuned model from {checkpoint_path}")
    return model


def generate_speech(model, text_prompt, output_path, duration=10.0, device="cuda"):
    """Generate speech using the fine-tuned model"""
    model.to(device)
    model.eval()
    
    # Create conditioning attributes
    attributes = [ConditioningAttributes(text={'description': text_prompt})]
    
    # Generate audio
    logger.info(f"Generating speech with prompt: '{text_prompt}'")
    with torch.no_grad():
        audio = model.generate(attributes, progress=True)
    
    # Save audio
    torchaudio.save(output_path, audio.cpu(), sample_rate=model.sample_rate)
    logger.info(f"Saved generated speech to {output_path}")
    
    return audio


def generate_music(model, text_prompt, output_path, duration=10.0, device="cuda"):
    """Generate music using the fine-tuned model (to verify music generation is preserved)"""
    model.to(device)
    model.eval()
    
    # Create conditioning attributes
    attributes = [ConditioningAttributes(text={'description': text_prompt})]
    
    # Generate audio
    logger.info(f"Generating music with prompt: '{text_prompt}'")
    with torch.no_grad():
        audio = model.generate(attributes, progress=True)
    
    # Save audio
    torchaudio.save(output_path, audio.cpu(), sample_rate=model.sample_rate)
    logger.info(f"Saved generated music to {output_path}")
    
    return audio


def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    if args.checkpoint_path:
        model = load_finetuned_model(args.checkpoint_path, args.base_model_id, device)
    else:
        # If no checkpoint provided, use a pretrained model
        model = SpeechMusicGen.from_pretrained(
            model_id=args.base_model_id,
            adapter_dim=64,
            adapter_dropout=0.1,
            add_adapters_to=[-1, -2, -3, -4],
            freeze_embeddings=True,
            add_speech_prefix_embedding=True,
            device=device,
        )
    
    # Set generation parameters
    model.set_generation_params(
        duration=args.duration,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate speech
    speech_output_path = os.path.join(args.output_dir, f"speech_{Path(args.speech_prompt).stem}.wav")
    speech_audio = generate_speech(
        model,
        args.speech_prompt,
        speech_output_path,
        duration=args.duration,
        device=device,
    )
    
    # Generate music (to verify music generation is preserved)
    if args.music_prompt:
        music_output_path = os.path.join(args.output_dir, f"music_{Path(args.music_prompt).stem}.wav")
        music_audio = generate_music(
            model,
            args.music_prompt,
            music_output_path,
            duration=args.duration,
            device=device,
        )
    
    logger.info("Generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio with a fine-tuned MusicGen model")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a checkpoint file")
    parser.add_argument("--base_model_id", type=str, default="facebook/musicgen-small", help="Base model ID")
    parser.add_argument("--speech_prompt", type=str, required=True, help="Text prompt for speech generation")
    parser.add_argument("--music_prompt", type=str, default=None, help="Text prompt for music generation")
    parser.add_argument("--output_dir", type=str, default="./generated", help="Output directory")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of generated audio")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=250, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.0, help="Top-p sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    main(args) 