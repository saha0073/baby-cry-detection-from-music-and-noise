import librosa
import numpy as np
from scipy import signal
import os
import soundfile as sf

def load_audio(file_path):
    """
    Load audio file and convert to mono if stereo
    Returns: audio array and sample rate
    """
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None, mono=False)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    
    return audio, sr

def normalize_audio(audio):
    """
    Normalize audio to have maximum absolute value of 1
    """
    return librosa.util.normalize(audio)

def apply_noise_reduction(audio, sr, noise_reduction_strength=0.1):
    """
    Apply basic noise reduction using spectral subtraction
    """
    # Compute the spectrogram
    D = librosa.stft(audio)
    
    # Estimate noise from the first few frames
    noise_estimate = np.mean(np.abs(D[:, :5]), axis=1, keepdims=True)
    
    # Subtract noise estimate from the spectrogram
    D_clean = D - (noise_reduction_strength * noise_estimate)
    
    # Convert back to time domain
    audio_clean = librosa.istft(D_clean)
    
    return audio_clean

def preprocess_audio_file(file_path, apply_noise_reduction_flag=True):
    """
    Complete preprocessing pipeline for a single audio file
    """
    # Load and convert to mono
    audio, sr = load_audio(file_path)
    
    # Normalize
    audio_normalized = normalize_audio(audio)
    
    # Apply noise reduction if requested
    if apply_noise_reduction_flag:
        audio_processed = apply_noise_reduction(audio_normalized, sr)
    else:
        audio_processed = audio_normalized
    
    return audio_processed, sr

def process_directory(input_dir, output_dir=None):
    """
    Process all .ogg files in a directory
    """
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each .ogg file
    for filename in os.listdir(input_dir):
        if filename.endswith('.ogg'):
            input_path = os.path.join(input_dir, filename)
            
            # Process the audio
            audio_processed, sr = preprocess_audio_file(input_path)
            
            # Save processed audio if output directory is specified
            if output_dir:
                output_path = os.path.join(output_dir, f'processed_{filename}')
                # Save as WAV file
                sf.write(output_path.replace('.ogg', '.wav'), audio_processed, sr)
            
            print(f"Processed: {filename}") 