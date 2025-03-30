#!/usr/bin/env python3

import argparse
import os
import json
import logging
import shutil
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

try:
    import librosa
    import soundfile as sf
except ImportError:
    logger.error("Required packages not found. Please install librosa and soundfile:")
    logger.error("pip install librosa soundfile")
    sys.exit(1)

DATASETS = {
    "ljspeech": {
        "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "description": "A female narrator with a clear, professional voice reading passages with a neutral American accent"
    },
    "vctk_sample": {
        "url": "https://datashare.ed.ac.uk/download/DS_10283_3443.zip",
        "description": "A {gender} speaker with {accent} accent speaking clearly with natural intonation"
    }
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_archive(file_path, extract_path):
    logger.info(f"Extracting {file_path} to {extract_path}")
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif file_path.endswith('.tar.bz2'):
        with tarfile.open(file_path, 'r:bz2') as tar_ref:
            tar_ref.extractall(extract_path)
    else:
        raise ValueError(f"Unsupported archive format: {file_path}")

def process_ljspeech(extract_dir, output_dir, sample_rate=32000, max_samples=None):
    """Process LJSpeech dataset"""
    logger.info("Processing LJSpeech dataset")
    
    wavs_dir = os.path.join(extract_dir, "LJSpeech-1.1", "wavs")
    metadata_file = os.path.join(extract_dir, "LJSpeech-1.1", "metadata.csv")
    
    # Read metadata
    metadata = {}
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                file_id, _, text = parts
                metadata[file_id] = text
    
    # Process audio files
    wav_files = list(Path(wavs_dir).glob("*.wav"))
    if max_samples and max_samples > 0:
        wav_files = random.sample(wav_files, min(max_samples, len(wav_files)))
    
    for wav_file in tqdm(wav_files, desc="Processing audio files"):
        file_id = wav_file.stem
        
        # Load and resample audio
        y, sr = librosa.load(str(wav_file), sr=sample_rate, mono=True)
        
        # Get duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Skip files shorter than 1 second or longer than 30 seconds
        if duration < 1.0 or duration > 30.0:
            continue
        
        # Output paths
        output_wav = os.path.join(output_dir, f"{file_id}.wav")
        output_json = os.path.join(output_dir, f"{file_id}.json")
        
        # Save resampled audio
        sf.write(output_wav, y, sample_rate)
        
        # Create JSON description
        text = metadata.get(file_id, "")
        description = DATASETS["ljspeech"]["description"]
        if text:
            description += f" saying: {text}"
            
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump({"description": description}, f, ensure_ascii=False)

def process_vctk(extract_dir, output_dir, sample_rate=32000, max_samples=None):
    """Process VCTK dataset"""
    logger.info("Processing VCTK dataset")
    
    # VCTK structure might vary by version, adjust if needed
    wav_dirs = list(Path(extract_dir).glob("**/wav48"))
    if not wav_dirs:
        wav_dirs = list(Path(extract_dir).glob("**/wav48_silence_trimmed"))
    
    if not wav_dirs:
        logger.error("Could not find wav directories in VCTK dataset")
        return
    
    wav_dir = wav_dirs[0]
    
    # Create speaker mapping for gender and accent info
    speaker_info = {}
    speaker_info_file = Path(extract_dir).glob("**/speaker-info.txt")
    speaker_info_file = list(speaker_info_file)
    
    if speaker_info_file:
        with open(speaker_info_file[0], 'r', encoding='utf-8') as f:
            # Skip header
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    speaker_id = parts[0]
                    gender = parts[2].lower()
                    accent = parts[3].lower()
                    speaker_info[speaker_id] = {"gender": gender, "accent": accent}
    
    # Find all wav files
    all_wav_files = list(Path(wav_dir).glob("**/*.wav"))
    if max_samples and max_samples > 0:
        all_wav_files = random.sample(all_wav_files, min(max_samples, len(all_wav_files)))
    
    # Process audio files
    for wav_file in tqdm(all_wav_files, desc="Processing VCTK files"):
        # Extract speaker ID and utterance ID
        speaker_id = wav_file.parent.name
        file_id = wav_file.stem
        
        # Load and resample audio
        try:
            y, sr = librosa.load(str(wav_file), sr=sample_rate, mono=True)
        except Exception as e:
            logger.warning(f"Error loading {wav_file}: {e}")
            continue
        
        # Get duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Skip files shorter than 1 second or longer than 30 seconds
        if duration < 1.0 or duration > 30.0:
            continue
        
        # Output paths
        output_wav = os.path.join(output_dir, f"vctk_{speaker_id}_{file_id}.wav")
        output_json = os.path.join(output_dir, f"vctk_{speaker_id}_{file_id}.json")
        
        # Save resampled audio
        sf.write(output_wav, y, sample_rate)
        
        # Create JSON description
        speaker_data = speaker_info.get(speaker_id, {})
        gender = speaker_data.get("gender", "unknown")
        accent = speaker_data.get("accent", "unknown")
        
        description = DATASETS["vctk_sample"]["description"]
        description = description.format(gender=gender, accent=accent)
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump({"description": description}, f, ensure_ascii=False)

def prepare_dataset(dataset_name, output_dir, download_dir, sample_rate=32000, max_samples=None, val_split=0.1):
    """Prepare a dataset for training"""
    # Create directories
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Check if dataset is supported
    if dataset_name not in DATASETS:
        logger.error(f"Dataset {dataset_name} not supported. Available datasets: {', '.join(DATASETS.keys())}")
        return
    
    dataset_info = DATASETS[dataset_name]
    url = dataset_info["url"]
    
    # Download dataset if not already downloaded
    archive_path = os.path.join(download_dir, url.split('/')[-1])
    if not os.path.exists(archive_path):
        logger.info(f"Downloading {dataset_name} dataset from {url}")
        download_url(url, archive_path)
    
    # Extract archive
    extract_path = os.path.join(download_dir, dataset_name + "_extracted")
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        extract_archive(archive_path, extract_path)
    
    # Process dataset to a temporary directory
    temp_dir = os.path.join(download_dir, dataset_name + "_processed")
    os.makedirs(temp_dir, exist_ok=True)
    
    if dataset_name == "ljspeech":
        process_ljspeech(extract_path, temp_dir, sample_rate, max_samples)
    elif dataset_name == "vctk_sample":
        process_vctk(extract_path, temp_dir, sample_rate, max_samples)
    
    # Split into train and validation sets
    all_files = [(f, f.replace('.wav', '.json')) for f in Path(temp_dir).glob("*.wav")]
    random.shuffle(all_files)
    
    val_count = max(1, int(len(all_files) * val_split))
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]
    
    logger.info(f"Splitting into {len(train_files)} training and {len(val_files)} validation samples")
    
    # Copy files to train and validation directories
    for wav_file, json_file in tqdm(train_files, desc="Copying training files"):
        shutil.copy(wav_file, os.path.join(train_dir, os.path.basename(wav_file)))
        shutil.copy(json_file, os.path.join(train_dir, os.path.basename(json_file)))
    
    for wav_file, json_file in tqdm(val_files, desc="Copying validation files"):
        shutil.copy(wav_file, os.path.join(val_dir, os.path.basename(wav_file)))
        shutil.copy(json_file, os.path.join(val_dir, os.path.basename(json_file)))
    
    logger.info(f"Dataset preparation complete: {len(train_files)} training samples, {len(val_files)} validation samples")

def main():
    parser = argparse.ArgumentParser(description="Prepare speech datasets for AudioCraft fine-tuning")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS.keys(),
                      help=f"Dataset to prepare: {', '.join(DATASETS.keys())}")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for prepared dataset")
    parser.add_argument("--download_dir", type=str, default="./downloads",
                      help="Directory to store downloaded files")
    parser.add_argument("--sample_rate", type=int, default=32000,
                      help="Sample rate for audio files (default: 32000)")
    parser.add_argument("--max_samples", type=int, default=None,
                      help="Maximum number of samples to process (for testing)")
    parser.add_argument("--val_split", type=float, default=0.1,
                      help="Fraction of data to use for validation (default: 0.1)")
    
    args = parser.parse_args()
    
    prepare_dataset(
        args.dataset,
        args.output_dir,
        args.download_dir,
        args.sample_rate,
        args.max_samples,
        args.val_split
    )

if __name__ == "__main__":
    main() 