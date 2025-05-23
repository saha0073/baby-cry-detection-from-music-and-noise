import numpy as np
import librosa
from scipy import signal
import os
import soundfile as sf
import json
import scipy.stats

def extract_zcr(audio, frame_length=2048, hop_length=512):
    """
    Extract Zero Crossing Rate
    Returns: ZCR values for each frame
    """
    return librosa.feature.zero_crossing_rate(audio, 
                                            frame_length=frame_length, 
                                            hop_length=hop_length)[0]

def extract_rmse(audio, frame_length=2048, hop_length=512):
    """
    Extract Root Mean Square Energy
    Returns: RMSE values for each frame
    """
    return librosa.feature.rms(y=audio, 
                             frame_length=frame_length, 
                             hop_length=hop_length)[0]

def extract_spectral_centroid(audio, sr, frame_length=2048, hop_length=512):
    """
    Extract Spectral Centroid
    Returns: Spectral centroid values for each frame
    """
    return librosa.feature.spectral_centroid(y=audio, 
                                           sr=sr,
                                           n_fft=frame_length, 
                                           hop_length=hop_length)[0]

def extract_spectral_bandwidth(audio, sr, frame_length=2048, hop_length=512):
    """
    Extract Spectral Bandwidth
    Returns: Spectral bandwidth values for each frame
    """
    return librosa.feature.spectral_bandwidth(y=audio, 
                                            sr=sr,
                                            n_fft=frame_length, 
                                            hop_length=hop_length)[0]

def extract_spectral_rolloff(audio, sr, frame_length=2048, hop_length=512):
    """
    Extract Spectral Rolloff
    Returns: Spectral rolloff values for each frame
    """
    return librosa.feature.spectral_rolloff(y=audio, 
                                          sr=sr,
                                          n_fft=frame_length, 
                                          hop_length=hop_length)[0]

def extract_mfcc(audio, sr, n_mfcc=13, frame_length=2048, hop_length=512):
    """
    Extract MFCCs (Mel-frequency cepstral coefficients)
    Returns: MFCC values for each frame
    """
    return librosa.feature.mfcc(y=audio, 
                               sr=sr,
                               n_mfcc=n_mfcc,
                               n_fft=frame_length, 
                               hop_length=hop_length)

def extract_spectral_flatness(audio, frame_length=2048, hop_length=512):
    """
    Extract Spectral Flatness
    Returns: Spectral flatness values for each frame
    """
    return librosa.feature.spectral_flatness(y=audio, 
                                           n_fft=frame_length, 
                                           hop_length=hop_length)[0]

def extract_pitch(audio, sr, frame_length=2048, hop_length=512):
    """
    Extract Pitch (F0) using PYIN
    Returns: F0 values for each frame
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                fmin=80, 
                                                fmax=1000, 
                                                sr=sr,
                                                frame_length=frame_length,
                                                hop_length=hop_length)
    return f0

def extract_harmonic_noise_ratio(audio, sr, frame_length=2048, hop_length=512):
    """
    Extract Harmonic-to-Noise Ratio using HPSS
    Returns: HNR values for each frame
    """
    # Compute spectrogram
    D = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    
    # Harmonic-percussive separation
    D_harmonic, D_percussive = librosa.decompose.hpss(D)
    
    # Calculate energy ratios
    harmonic_energy = np.sum(np.abs(D_harmonic), axis=0)
    percussive_energy = np.sum(np.abs(D_percussive), axis=0)
    
    # Calculate HNR
    hnr = harmonic_energy / (percussive_energy + 1e-10)
    return hnr

def compute_energy_ratio(audio, sr, frame_length=2048, hop_length=512, cry_freq_min=250, cry_freq_max=600):
    """
    Compute energy ratio in cry frequency bands with improved analysis
    """
    # Compute spectrogram with better parameters
    D = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    
    # Define sub-bands for cry analysis
    sub_bands = {
        'low': (250, 400),
        'mid': (400, 600),
        'high': (600, 1000)
    }
    
    energy_ratios = {}
    
    # Calculate total energy with proper normalization
    total_energy = np.sum(D, axis=0)
    total_energy = np.maximum(total_energy, 1e-10)  # Prevent division by zero
    
    # Calculate energy in each sub-band
    for band_name, (fmin, fmax) in sub_bands.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_energy = np.sum(D[band_mask, :], axis=0)
        energy_ratios[f'energy_ratio_{band_name}'] = float(np.mean(band_energy / total_energy))
    
    # Calculate main cry band energy ratio with improved normalization
    cry_mask = (freqs >= cry_freq_min) & (freqs <= cry_freq_max)
    cry_energy = np.sum(D[cry_mask, :], axis=0)
    energy_ratios['energy_ratio_main'] = float(np.mean(cry_energy / total_energy))
    
    # Calculate energy distribution
    energy_ratios['energy_ratio_distribution'] = float(np.std(cry_energy / total_energy))
    
    # Find peak frequency in cry band
    peak_freq_idx = np.argmax(np.mean(D[cry_mask, :], axis=1))
    energy_ratios['energy_ratio_peak_freq'] = float(freqs[cry_mask][peak_freq_idx])
    
    # Calculate energy ratio in time domain with improved normalization
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = np.mean(rms)
    rms_max = np.max(rms)
    energy_ratios['time_domain'] = float(rms_mean / (rms_max + 1e-10))
    
    # Calculate energy ratio in mel domain with improved normalization
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)
    mel_energy = np.sum(mel_spec, axis=0)
    mel_mean = np.mean(mel_energy)
    mel_max = np.max(mel_energy)
    energy_ratios['mel_domain'] = float(mel_mean / (mel_max + 1e-10))
    
    # Calculate energy ratio in bark domain with improved normalization
    bark_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mels=24)
    bark_energy = np.sum(bark_spec, axis=0)
    bark_mean = np.mean(bark_energy)
    bark_max = np.max(bark_energy)
    energy_ratios['bark_domain'] = float(bark_mean / (bark_max + 1e-10))
    
    # Add spectral features with improved normalization
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    energy_ratios['spectral_centroid'] = float(np.mean(spectral_centroid))
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    energy_ratios['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    energy_ratios['spectral_rolloff'] = float(np.mean(spectral_rolloff))
    
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)
    energy_ratios['spectral_contrast'] = float(np.mean(spectral_contrast))
    
    spectral_flatness = librosa.feature.spectral_flatness(y=audio, n_fft=frame_length, hop_length=hop_length)[0]
    energy_ratios['spectral_flatness'] = float(np.mean(spectral_flatness))
    
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    energy_ratios['zcr'] = float(np.mean(zcr))
    
    # Calculate MFCCs with improved normalization
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    energy_ratios['mfcc_mean'] = float(np.mean(mfcc_mean))
    energy_ratios['mfcc_std'] = float(np.mean(mfcc_std))
    
    # Calculate delta MFCCs with improved normalization
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
    delta_mfcc_std = np.std(delta_mfcc, axis=1)
    energy_ratios['delta_mfcc_mean'] = float(np.mean(delta_mfcc_mean))
    energy_ratios['delta_mfcc_std'] = float(np.mean(delta_mfcc_std))
    
    # Calculate delta-delta MFCCs with improved normalization
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)
    delta2_mfcc_std = np.std(delta2_mfcc, axis=1)
    energy_ratios['delta2_mfcc_mean'] = float(np.mean(delta2_mfcc_mean))
    energy_ratios['delta2_mfcc_std'] = float(np.mean(delta2_mfcc_std))
    
    return energy_ratios

def compute_pattern_features(audio, sr, frame_length=2048, hop_length=512):
    """
    Compute pattern features with improved rhythm analysis
    """
    # Calculate amplitude envelope with improved parameters
    amplitude_envelope = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find peaks with improved parameters
    peaks, properties = signal.find_peaks(amplitude_envelope, 
                                        height=np.mean(amplitude_envelope),
                                        prominence=0.1,
                                        distance=frame_length//hop_length)
    
    if len(peaks) < 2:
        return {
            'rhythm_regularity': 0.0,
            'duration': 0.0,
            'amplitude_modulation': 0.0,
            'peak_prominence': 0.0,
            'rhythm_stability': 0.0,
            'pattern_score': 0.0
        }
    
    # Calculate time between peaks
    peak_intervals = np.diff(peaks) * hop_length / sr
    
    # Basic features with improved calculations
    rhythm_regularity = 1 - np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-10)
    duration = len(audio) / sr
    amplitude_modulation = np.std(amplitude_envelope) / (np.mean(amplitude_envelope) + 1e-10)
    
    # New features
    peak_prominence = np.mean(properties['prominences']) if 'prominences' in properties else 0.0
    
    # Rhythm stability score with improved calculation
    interval_std = np.std(peak_intervals)
    interval_mean = np.mean(peak_intervals)
    rhythm_stability = 1.0 / (1.0 + interval_std / (interval_mean + 1e-10))
    
    # Pattern score (looking for short-long-short pattern)
    if len(peak_intervals) >= 3:
        pattern_scores = []
        for i in range(len(peak_intervals) - 2):
            short1 = peak_intervals[i]
            long1 = peak_intervals[i + 1]
            short2 = peak_intervals[i + 2]
            if short1 < long1 and short2 < long1:
                pattern_scores.append(1.0)
            else:
                pattern_scores.append(0.0)
        pattern_score = np.mean(pattern_scores) if pattern_scores else 0.0
    else:
        pattern_score = 0.0
    
    # Add new pattern features
    # Calculate peak-to-peak ratio
    peak_ratios = []
    for i in range(len(peaks) - 1):
        ratio = amplitude_envelope[peaks[i+1]] / (amplitude_envelope[peaks[i]] + 1e-10)
        peak_ratios.append(ratio)
    peak_ratio_mean = np.mean(peak_ratios) if peak_ratios else 0.0
    
    # Calculate rhythm complexity
    rhythm_complexity = np.std(peak_ratios) if peak_ratios else 0.0
    
    return {
        'rhythm_regularity': float(rhythm_regularity),
        'duration': float(duration),
        'amplitude_modulation': float(amplitude_modulation),
        'peak_prominence': float(peak_prominence),
        'rhythm_stability': float(rhythm_stability),
        'pattern_score': float(pattern_score),
        'peak_ratio_mean': float(peak_ratio_mean),
        'rhythm_complexity': float(rhythm_complexity)
    }

def compute_music_noise_ratios(audio, sr, frame_length=2048, hop_length=512):
    """
    Compute music and noise ratios with improved separation
    """
    # Compute spectrogram with improved parameters
    D = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length, window='hann')
    
    # Harmonic-percussive separation with improved parameters
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=3.0)
    
    # Convert back to time domain
    audio_harmonic = librosa.istft(D_harmonic)
    audio_percussive = librosa.istft(D_percussive)
    
    # Calculate energy ratios with improved normalization
    total_energy = np.sum(audio ** 2)
    total_energy = np.maximum(total_energy, 1e-10)  # Prevent division by zero
    
    harmonic_energy = np.sum(audio_harmonic ** 2)
    percussive_energy = np.sum(audio_percussive ** 2)
    
    # Basic ratios
    music_ratio = harmonic_energy / total_energy
    noise_ratio = percussive_energy / total_energy
    
    # Additional features
    # Spectral flux (for detecting sudden changes)
    spectral_flux = np.mean(np.diff(np.abs(D), axis=1))
    
    # Spectral contrast (for music vs cry distinction)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)
    spectral_contrast_mean = np.mean(spectral_contrast)
    
    # Harmonicity measure with improved calculation
    harmonicity = np.mean(np.abs(D_harmonic)) / (np.mean(np.abs(D)) + 1e-10)
    
    # Percussiveness measure with improved calculation
    percussiveness = np.mean(np.abs(D_percussive)) / (np.mean(np.abs(D)) + 1e-10)
    
    # Add new features
    # Calculate spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    spectral_centroid_mean = np.mean(spectral_centroid)
    
    # Calculate spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    
    # Calculate spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    spectral_rolloff_mean = np.mean(spectral_rolloff)
    
    return {
        'music_ratio': float(music_ratio),
        'noise_ratio': float(noise_ratio),
        'spectral_flux': float(spectral_flux),
        'spectral_contrast': float(spectral_contrast_mean),
        'harmonicity': float(harmonicity),
        'percussiveness': float(percussiveness),
        'spectral_centroid_mean': float(spectral_centroid_mean),
        'spectral_bandwidth_mean': float(spectral_bandwidth_mean),
        'spectral_rolloff_mean': float(spectral_rolloff_mean)
    }

def extract_all_features(audio, sr, frame_length=2048, hop_length=512):
    """
    Extract all features from the audio signal with improved analysis
    """
    # Initialize features dictionary
    features = {}
    
    # Extract basic features
    features['zcr'] = extract_zcr(audio, frame_length, hop_length)
    features['rmse'] = extract_rmse(audio, frame_length, hop_length)
    features['spectral_centroid'] = extract_spectral_centroid(audio, sr, frame_length, hop_length)
    features['spectral_bandwidth'] = extract_spectral_bandwidth(audio, sr, frame_length, hop_length)
    features['spectral_rolloff'] = extract_spectral_rolloff(audio, sr, frame_length, hop_length)
    features['mfcc'] = extract_mfcc(audio, sr, frame_length=frame_length, hop_length=hop_length)
    features['spectral_flatness'] = extract_spectral_flatness(audio, frame_length, hop_length)
    features['pitch'] = extract_pitch(audio, sr, frame_length, hop_length)
    features['pitch'] = np.nan_to_num(features['pitch'], nan=0.0)
    features['hnr'] = extract_harmonic_noise_ratio(audio, sr, frame_length, hop_length)
    
    # Extract improved features
    features.update(compute_energy_ratio(audio, sr, frame_length, hop_length))
    features.update(compute_pattern_features(audio, sr, frame_length, hop_length))
    features.update(compute_music_noise_ratios(audio, sr, frame_length, hop_length))
    
    # Calculate statistical features
    statistical_features = {}
    
    # Calculate statistical features for non-MFCC features
    for feature_name in ['zcr', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 
                        'spectral_rolloff', 'spectral_flatness', 'pitch', 'hnr']:
        feature_values = features[feature_name]
        
        # Handle NaN values in pitch
        if feature_name == 'pitch':
            feature_values = np.nan_to_num(feature_values, nan=0.0)
        
        # Basic statistics
        statistical_features[f'{feature_name}_mean'] = np.mean(feature_values)
        statistical_features[f'{feature_name}_std'] = np.std(feature_values)
        statistical_features[f'{feature_name}_max'] = np.max(feature_values)
        statistical_features[f'{feature_name}_min'] = np.min(feature_values)
        
        # Additional statistics with NaN handling
        # Only calculate kurtosis and skewness if we have enough non-zero values
        non_zero_mask = feature_values != 0
        if np.sum(non_zero_mask) > 3:  # Need at least 4 non-zero values for kurtosis
            statistical_features[f'{feature_name}_kurtosis'] = float(scipy.stats.kurtosis(feature_values[non_zero_mask]))
            statistical_features[f'{feature_name}_skewness'] = float(scipy.stats.skew(feature_values[non_zero_mask]))
        else:
            statistical_features[f'{feature_name}_kurtosis'] = 0.0
            statistical_features[f'{feature_name}_skewness'] = 0.0
        
        # Percentiles and range statistics
        statistical_features[f'{feature_name}_q25'] = float(np.percentile(feature_values, 25))
        statistical_features[f'{feature_name}_q75'] = float(np.percentile(feature_values, 75))
        statistical_features[f'{feature_name}_range'] = float(np.ptp(feature_values))
        statistical_features[f'{feature_name}_iqr'] = float(np.percentile(feature_values, 75) - np.percentile(feature_values, 25))
    
    # Calculate statistical features for MFCCs
    mfcc_values = features['mfcc']
    for i in range(mfcc_values.shape[0]):
        statistical_features[f'mfcc_{i}_mean'] = float(np.mean(mfcc_values[i]))
        statistical_features[f'mfcc_{i}_std'] = float(np.std(mfcc_values[i]))
        statistical_features[f'mfcc_{i}_max'] = float(np.max(mfcc_values[i]))
        statistical_features[f'mfcc_{i}_min'] = float(np.min(mfcc_values[i]))
        
        # Handle kurtosis and skewness for MFCCs
        non_zero_mask = mfcc_values[i] != 0
        if np.sum(non_zero_mask) > 3:
            statistical_features[f'mfcc_{i}_kurtosis'] = float(scipy.stats.kurtosis(mfcc_values[i][non_zero_mask]))
            statistical_features[f'mfcc_{i}_skewness'] = float(scipy.stats.skew(mfcc_values[i][non_zero_mask]))
        else:
            statistical_features[f'mfcc_{i}_kurtosis'] = 0.0
            statistical_features[f'mfcc_{i}_skewness'] = 0.0
    
    # Add delta and delta-delta MFCCs
    delta_mfcc = librosa.feature.delta(features['mfcc'])
    delta2_mfcc = librosa.feature.delta(features['mfcc'], order=2)
    
    for i in range(delta_mfcc.shape[0]):
        statistical_features[f'delta_mfcc_{i}_mean'] = float(np.mean(delta_mfcc[i]))
        statistical_features[f'delta_mfcc_{i}_std'] = float(np.std(delta_mfcc[i]))
        statistical_features[f'delta2_mfcc_{i}_mean'] = float(np.mean(delta2_mfcc[i]))
        statistical_features[f'delta2_mfcc_{i}_std'] = float(np.std(delta2_mfcc[i]))
    
    # Combine all features
    features.update(statistical_features)
    
    # Convert all numpy types to native Python types
    features = convert_to_native(features)
    
    return features

def get_feature_names():
    """
    Get list of all feature names
    """
    base_features = ['zcr', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 
                    'spectral_rolloff', 'spectral_flatness', 'pitch', 'hnr']
    feature_names = []
    
    # Add base features with their statistics
    for feature in base_features:
        feature_names.extend([
            f'{feature}_mean',
            f'{feature}_std',
            f'{feature}_max',
            f'{feature}_min',
            f'{feature}_kurtosis',
            f'{feature}_skewness',
            f'{feature}_q25',
            f'{feature}_q75',
            f'{feature}_range',
            f'{feature}_iqr'
        ])
    
    # Add MFCC features
    for i in range(13):  # 13 MFCC coefficients
        feature_names.extend([
            f'mfcc_{i}_mean',
            f'mfcc_{i}_std',
            f'mfcc_{i}_max',
            f'mfcc_{i}_min',
            f'mfcc_{i}_kurtosis',
            f'mfcc_{i}_skewness',
            f'delta_mfcc_{i}_mean',
            f'delta_mfcc_{i}_std',
            f'delta2_mfcc_{i}_mean',
            f'delta2_mfcc_{i}_std'
        ])
    
    # Add new feature names
    feature_names.extend([
        'energy_ratio_main',
        'energy_ratio_low',
        'energy_ratio_mid',
        'energy_ratio_high',
        'energy_ratio_distribution',
        'energy_ratio_peak_freq',
        'rhythm_regularity',
        'duration',
        'amplitude_modulation',
        'peak_prominence',
        'rhythm_stability',
        'pattern_score',
        'music_ratio',
        'noise_ratio',
        'spectral_flux',
        'spectral_contrast',
        'harmonicity',
        'percussiveness'
    ])
    
    return feature_names

def convert_to_native(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int_)):
        return int(obj)
    else:
        return obj

def process_and_save_features(processed_audio_dir='processed_audio', output_dir='extracted_features'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for subdir in os.listdir(processed_audio_dir):
        subdir_path = os.path.join(processed_audio_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        features_list = []
        for fname in os.listdir(subdir_path):
            if not fname.endswith('.wav'):
                continue
            file_path = os.path.join(subdir_path, fname)
            try:
                audio, sr = librosa.load(file_path, sr=None, mono=True)
                features = extract_all_features(audio, sr)
                features_native = convert_to_native(features)
                features_list.append({
                    'filename': fname,
                    'features': features_native
                })
                print(f"Extracted features for {fname}")
            except Exception as e:
                print(f"Error processing {fname}: {e}")
        # Save features for this subdir
        output_file = os.path.join(output_dir, f"{subdir}_features.json")
        with open(output_file, 'w') as f:
            json.dump(features_list, f, indent=2)
        print(f"Saved features for {subdir} to {output_file}")

if __name__ == "__main__":
    process_and_save_features()