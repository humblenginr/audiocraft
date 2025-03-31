import torch
from audiocraft.models.extended_musicgen import ExtendedMusicGen
from audiocraft.data.audio import audio_write

def load_checkpoint_and_state(checkpoint_path, device):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Get the model state from the checkpoint
    if 'best_state' in checkpoint:
        state_dict = checkpoint['best_state']
        if 'model' in state_dict:
            state_dict = state_dict['model']
    else:
        state_dict = checkpoint
    return state_dict

# Load our extended model
# First, specify the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create the extended model with 12 additional layers (or however many you used in training)
model = ExtendedMusicGen.get_pretrained(
    name='facebook/musicgen-small',  # Base model
    device=device,
    num_additional_layers=12  # Should match what you used during training
)

# Load your trained state
checkpoint_path = '/tmp/audiocraft_root/xps/e8adc017/checkpoint.th'  # Update this with the actual path
try:
    state_dict = load_checkpoint_and_state(checkpoint_path, device)
    # Load the state dict into the model
    model.lm.load_state_dict(state_dict)
    print(f"Successfully loaded checkpoint from {checkpoint_path}")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    print("Continuing with pretrained weights only...")

# Set to evaluation mode
model.lm.eval()

# Double-check that all model components are on the correct device
for name, module in model.lm.named_modules():
    for param_name, param in module.named_parameters():
        if param.device != device:
            print(f"Warning: {name}.{param_name} is on {param.device}, not {device}")
            param.data = param.data.to(device)

# Set generation parameters
model.set_generation_params(
    duration=10,
    temperature=1.0,
    top_k=250,
    top_p=0.0,
    use_sampling=True,
)

# Test prompts
descriptions = [
    "This is an electronic song sending positive vibes.",
    "This is a pop song with a catchy melody.",
    "This is a rock song with a heavy guitar riff.",
    "This is a jazz song with a smooth saxophone solo.",
    "This is a country song with a simple acoustic guitar.",
    "This is a hip-hop song with a catchy beat.",
    "This is a classical song with a beautiful melody.",
]

# Generate audio
print("Generating audio...")
with torch.no_grad():  # Ensure no gradients are computed during inference
    wav = model.generate(descriptions, progress=True)

# Save the outputs
print("Saving audio files...")
for idx, one_wav in enumerate(wav):
    audio_write(
        f'generated_music_{idx}', 
        one_wav.cpu(), 
        model.sample_rate, 
        strategy="loudness", 
        loudness_compressor=True
    )
    print(f"Saved generated_music_{idx}.wav")