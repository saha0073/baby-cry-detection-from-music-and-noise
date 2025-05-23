import numpy as np
import librosa
from scipy import signal
from scipy.fft import fft, fftfreq
import json
import pprint

class CryDetector:
    def __init__(self, sr=22050):
        self.sr = sr
        # Cry frequency range (Hz)
        self.cry_freq_min = 250
        self.cry_freq_max = 600
        
        # Load thresholds from training data
        self.load_thresholds()
    
    def load_thresholds(self):
        """
        Load thresholds from training data
        """
        try:
            # Load features from all category files
            all_features = []
            feature_files = [
                'extracted_features/Cry-NoNoise-Music_features.json',
                'extracted_features/NoCry-NoNoise-Music_features.json',
                'extracted_features/Cry-NoNoise-NoMusic_features.json',
                'extracted_features/Cry-Noise-NoMusic_features.json',
                'extracted_features/NoCry-Noise-NoMusic_features.json'
            ]
            
            for file_path in feature_files:
                try:
                    with open(file_path, 'r') as f:
                        features = json.load(f)
                        all_features.extend(features)
                except FileNotFoundError:
                    print(f"Warning: {file_path} not found, skipping...")
                    continue
            
            if not all_features:
                raise FileNotFoundError("No feature files found")
            
            # Calculate thresholds from combined training data
            self.calculate_thresholds(all_features)
        except FileNotFoundError:
            print("Warning: No training data found. Using default thresholds.")
            self.set_default_thresholds()
    
    def set_default_thresholds(self):
        """
        Set default thresholds when no training data is available
        """
        self.thresholds = {
            'energy_ratio': 0.08,         # Keep same
            'rhythm_regularity': -0.15,   # Keep same
            'duration_min': 0.1,
            'duration_max': 12.0,
            'amplitude_modulation': 0.2,   # Keep same
            'noise_ratio': 0.15,          # Stricter on noise
            'music_ratio': 0.8            # Stricter on music
        }
    
    def calculate_thresholds(self, features):
        """
        Calculate thresholds from training data
        """
        # Extract features for cry and non-cry samples
        cry_features = []
        non_cry_features = []
        
        for feature_dict in features:
            if isinstance(feature_dict, dict) and 'features' in feature_dict:
                features = feature_dict['features']
                if feature_dict.get('is_cry', False):
                    cry_features.append(features)
                else:
                    non_cry_features.append(features)
        
        if not cry_features or not non_cry_features:
            print("Warning: Insufficient data for threshold calculation. Using default thresholds.")
            self.set_default_thresholds()
            return
        
        # Calculate thresholds based on feature distributions
        self.thresholds = {
            'energy_ratio': np.percentile([f.get('energy_ratio_main', 0) for f in cry_features], 25),
            'rhythm_regularity': np.percentile([f.get('rhythm_regularity', 0) for f in cry_features], 25),
            'duration_min': 0.1,  # Keep fixed
            'duration_max': 12.0,  # Keep fixed
            'amplitude_modulation': np.percentile([f.get('amplitude_modulation', 0) for f in cry_features], 25),
            'noise_ratio': np.percentile([f.get('noise_ratio', 0) for f in non_cry_features], 75),
            'music_ratio': np.percentile([f.get('music_ratio', 0) for f in non_cry_features], 75)
        }
        
        print("Calculated thresholds from training data:")
        for k, v in self.thresholds.items():
            print(f"{k}: {v:.3f}")
    
    def frequency_analysis(self, audio):
        """
        Analyze frequency content in the cry range (250-600 Hz)
        """
        # Compute spectrogram with better parameters
        D = librosa.stft(audio, n_fft=2048, hop_length=512)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Find indices for cry frequency range
        cry_freq_mask = (freqs >= self.cry_freq_min) & (freqs <= self.cry_freq_max)
        
        # Calculate energy in cry frequency band with improved normalization
        cry_energy = np.sum(np.abs(D[cry_freq_mask, :]), axis=0)
        total_energy = np.sum(np.abs(D), axis=0)
        
        # Calculate energy ratio with better handling of small values
        energy_ratio = np.zeros_like(cry_energy)
        mask = total_energy > 1e-10
        energy_ratio[mask] = cry_energy[mask] / total_energy[mask]
        
        # Calculate additional energy metrics
        cry_energy_mean = np.mean(cry_energy)
        total_energy_mean = np.mean(total_energy)
        
        # Calculate energy in sub-bands
        low_mask = (freqs >= 100) & (freqs < 250)
        mid_mask = (freqs >= 250) & (freqs <= 600)
        high_mask = (freqs > 600) & (freqs <= 1000)
        
        low_energy = np.sum(np.abs(D[low_mask, :]), axis=0)
        mid_energy = np.sum(np.abs(D[mid_mask, :]), axis=0)
        high_energy = np.sum(np.abs(D[high_mask, :]), axis=0)
        
        # Calculate sub-band ratios
        low_ratio = np.zeros_like(low_energy)
        mid_ratio = np.zeros_like(mid_energy)
        high_ratio = np.zeros_like(high_energy)
        
        mask = total_energy > 1e-10
        low_ratio[mask] = low_energy[mask] / total_energy[mask]
        mid_ratio[mask] = mid_energy[mask] / total_energy[mask]
        high_ratio[mask] = high_energy[mask] / total_energy[mask]
        
        return {
            'energy_ratio': np.mean(energy_ratio),
            'energy_std': np.std(energy_ratio),
            'cry_energy': cry_energy_mean,
            'total_energy': total_energy_mean,
            'energy_ratio_low': np.mean(low_ratio),
            'energy_ratio_mid': np.mean(mid_ratio),
            'energy_ratio_high': np.mean(high_ratio),
            'energy_ratio_main': np.mean(mid_ratio),  # Main cry band
            'energy_ratio_distribution': np.std([np.mean(low_ratio), np.mean(mid_ratio), np.mean(high_ratio)]),
            'energy_ratio_peak_freq': freqs[np.argmax(np.mean(np.abs(D), axis=1))] / (self.sr/2)
        }
    
    def pattern_recognition(self, audio):
        """
        Analyze cry patterns (rhythm, duration, amplitude modulation)
        """
        # Calculate amplitude envelope
        amplitude_envelope = librosa.feature.rms(y=audio)[0]
        
        # Detect peaks in amplitude envelope
        peaks, _ = signal.find_peaks(amplitude_envelope, height=np.mean(amplitude_envelope))
        
        if len(peaks) < 2:
            return {
                'rhythm_regularity': 0,
                'duration': 0,
                'amplitude_modulation': 0
            }
        
        # Calculate time between peaks
        peak_intervals = np.diff(peaks) / self.sr
        
        # Calculate rhythm regularity
        rhythm_regularity = 1 - np.std(peak_intervals) / np.mean(peak_intervals)
        
        # Calculate duration
        duration = len(audio) / self.sr
        
        # Calculate amplitude modulation
        amplitude_modulation = np.std(amplitude_envelope) / np.mean(amplitude_envelope)
        
        return {
            'rhythm_regularity': rhythm_regularity,
            'duration': duration,
            'amplitude_modulation': amplitude_modulation
        }
    
    def music_noise_separation(self, audio):
        """
        Separate music and noise components
        """
        # Compute spectrogram
        D = librosa.stft(audio)
        
        # Harmonic-percussive separation
        D_harmonic, D_percussive = librosa.decompose.hpss(D)
        
        # Convert back to time domain
        audio_harmonic = librosa.istft(D_harmonic)
        audio_percussive = librosa.istft(D_percussive)
        
        # Calculate energy ratios
        total_energy = np.sum(audio ** 2)
        harmonic_energy = np.sum(audio_harmonic ** 2)
        percussive_energy = np.sum(audio_percussive ** 2)
        
        return {
            'music_ratio': harmonic_energy / (total_energy + 1e-10),
            'noise_ratio': percussive_energy / (total_energy + 1e-10)
        }
    
    def detect_cry(self, audio):
        """
        Main cry detection function
        """
        # Perform frequency analysis
        freq_features = self.frequency_analysis(audio)
        
        # Perform pattern recognition
        pattern_features = self.pattern_recognition(audio)
        
        # Perform music/noise separation
        separation_features = self.music_noise_separation(audio)
        
        # Combine all features
        features = {**freq_features, **pattern_features, **separation_features}
        
        # Make decision based on thresholds
        is_cry = (
            features['energy_ratio'] > self.thresholds['energy_ratio'] and
            features['rhythm_regularity'] > self.thresholds['rhythm_regularity'] and
            self.thresholds['duration_min'] < features['duration'] < self.thresholds['duration_max'] and
            features['amplitude_modulation'] > self.thresholds['amplitude_modulation'] and
            features['noise_ratio'] < self.thresholds['noise_ratio'] and
            features['music_ratio'] < self.thresholds['music_ratio']
        )
        
        return {
            'is_cry': bool(is_cry),
            'confidence': self.calculate_confidence(features),
            'features': features
        }
    
    def calculate_confidence(self, features):
        """
        Calculate confidence score for the detection
        """
        # Normalize each feature to [0, 1] range
        normalized_features = {
            'energy_ratio': min(features['energy_ratio'] / self.thresholds['energy_ratio'], 1),
            'rhythm_regularity': min(features['rhythm_regularity'] / self.thresholds['rhythm_regularity'], 1),
            'amplitude_modulation': min(features['amplitude_modulation'] / self.thresholds['amplitude_modulation'], 1),
            'noise_ratio': 1 - min(features['noise_ratio'] / self.thresholds['noise_ratio'], 1),
            'music_ratio': 1 - min(features['music_ratio'] / self.thresholds['music_ratio'], 1)
        }
        
        # Calculate weighted average with adjusted weights
        weights = {
            'energy_ratio': 0.35,      # Increased weight for energy ratio
            'rhythm_regularity': 0.25,  # Increased weight for rhythm
            'amplitude_modulation': 0.25,  # Increased weight for modulation
            'noise_ratio': 0.075,      # Decreased weight for noise
            'music_ratio': 0.075       # Decreased weight for music
        }
        
        confidence = sum(normalized_features[k] * weights[k] for k in weights)
        return confidence

    def detect_cry_from_features(self, features):
        """
        Improved cry detection from features with consistent thresholds and proper scoring
        Adds detailed logging for all files, especially misclassified ones.
        """
        # Extract relevant features
        energy_ratio_main = features.get('energy_ratio_main', 0)
        amplitude_modulation = features.get('amplitude_modulation', 0)
        rhythm_regularity = features.get('rhythm_regularity', 0)
        music_ratio = features.get('music_ratio', 0)
        noise_ratio = features.get('noise_ratio', 0)
        duration = features.get('duration', 0)
        file_name = features.get('file_name', 'unknown')
        expected_label = features.get('expected_label', None)

        # Use thresholds from training data
        energy_threshold = self.thresholds['energy_ratio']
        music_threshold = self.thresholds['music_ratio']
        duration_min = self.thresholds['duration_min']
        duration_max = self.thresholds['duration_max']
        rhythm_threshold = self.thresholds['rhythm_regularity']
        amplitude_threshold = self.thresholds['amplitude_modulation']
        noise_threshold = self.thresholds['noise_ratio']

        # Calculate individual scores
        scores = {
            'energy_score': energy_ratio_main > energy_threshold,
            'rhythm_score': rhythm_regularity > rhythm_threshold,
            'duration_score': duration_min < duration < duration_max,
            'amplitude_score': amplitude_modulation > amplitude_threshold,
            'noise_score': noise_ratio < noise_threshold,
            'music_score': music_ratio < music_threshold
        }

        # Main rule
        is_cry = all([
            scores['energy_score'],
            scores['rhythm_score'],
            scores['duration_score'],
            scores['amplitude_score'],
            scores['noise_score'],
            scores['music_score']
        ])

        # Strong cry override: very high amplitude_modulation or (high amplitude_modulation and high rhythm)
        if amplitude_modulation > 1.0 or (amplitude_modulation > 0.8 and rhythm_regularity > 0.5):
            is_cry = True
            scores['strong_cry_override'] = True

        # Loosen rhythm rule for very high amplitude_modulation
        if amplitude_modulation > 1.0 and not scores['rhythm_score']:
            is_cry = True
            scores['loosened_rhythm_for_high_amp'] = True

        # Weak cry block: very low amplitude_modulation and low energy
        if amplitude_modulation < 0.3 and energy_ratio_main < 0.15:
            is_cry = False
            scores['weak_cry_block'] = True

        # Moderate cry block: moderate amplitude_modulation and low energy (raise energy threshold to 0.5)
        if amplitude_modulation < 0.5 and energy_ratio_main < 0.5:
            is_cry = False
            scores['moderate_cry_block'] = True

        # High rhythm rescue: only allow if amplitude_modulation > 0.2
        if scores.get('moderate_cry_block', False) and rhythm_regularity > 0.9 and noise_ratio < 0.02 and amplitude_modulation > 0.2:
            is_cry = True
            scores['high_rhythm_rescue'] = True

        # Special case for music+cry (tightened: require higher amplitude_modulation)
        if music_ratio > 0.85 and amplitude_modulation > 0.7 and rhythm_regularity > 0.2:
            is_cry = True
            scores['music_cry_special'] = True

        # Penalty for noise
        if noise_ratio > 0.03 and energy_ratio_main < 0.12:
            is_cry = False
            scores['noise_penalty'] = True

        # Low rhythm/noise rescue: if moderate_cry_block is True, but energy is moderate, noise is low, and duration is long, allow cry (stricter: amplitude_modulation > 0.3)
        if scores.get('moderate_cry_block', False) and energy_ratio_main > 0.2 and noise_ratio < 0.02 and duration > 2.0 and amplitude_modulation > 0.3:
            is_cry = True
            scores['low_rhythm_noise_rescue'] = True

        # Final block for non-cries: if amplitude_modulation < 0.5, energy_ratio_main < 0.5, and rhythm_regularity < 0.7, block cry
        if amplitude_modulation < 0.5 and energy_ratio_main < 0.5 and rhythm_regularity < 0.7:
            is_cry = False
            scores['final_noncry_block'] = True

        # Final cry rescue: if both blocks would block, but energy is moderate, noise is low, and duration is long, allow cry
        if scores.get('moderate_cry_block', False) and scores.get('final_noncry_block', False) and energy_ratio_main > 0.2 and noise_ratio < 0.02 and duration > 2.0:
            is_cry = True
            scores['final_cry_rescue'] = True

        # New rule: Block potential false positives in NoCry-NoNoise-Music
        # If we have high energy and rhythm but low music ratio, and moderate amplitude,
        # this might be a false positive in NoCry-NoNoise-Music
        if (energy_ratio_main > 0.4 and 
            rhythm_regularity > 0.6 and 
            music_ratio < 0.6 and 
            0.4 < amplitude_modulation < 0.6):
            is_cry = False
            scores['music_noise_false_positive_block'] = True

        # Calculate confidence
        base_confidence = sum(scores.values()) / len(scores)
        
        # Adjust confidence based on special cases
        if scores.get('strong_cry_override', False):
            confidence = max(base_confidence, 0.9)
        elif scores.get('loosened_rhythm_for_high_amp', False):
            confidence = max(base_confidence, 0.85)
        elif scores.get('music_cry_special', False):
            confidence = max(base_confidence, 0.8)
        elif scores.get('noise_penalty', False):
            confidence = min(base_confidence, 0.2)
        elif scores.get('weak_cry_block', False):
            confidence = min(base_confidence, 0.1)
        elif scores.get('moderate_cry_block', False) and not scores.get('high_rhythm_rescue', False):
            confidence = min(base_confidence, 0.2)
        elif scores.get('music_noise_false_positive_block', False):
            confidence = min(base_confidence, 0.15)
        else:
            confidence = base_confidence

        # Add detailed logging
        features['decision_logic'] = {
            'energy_ratio_main > threshold': scores['energy_score'],
            'amplitude_modulation > threshold': scores['amplitude_score'],
            'rhythm_regularity > threshold': scores['rhythm_score'],
            'music_ratio < threshold': scores['music_score'],
            'noise_ratio < threshold': scores['noise_score'],
            'duration in range': scores['duration_score'],
            'strong_cry_override': scores.get('strong_cry_override', False),
            'loosened_rhythm_for_high_amp': scores.get('loosened_rhythm_for_high_amp', False),
            'music_cry_special_case': scores.get('music_cry_special', False),
            'noise_penalty': scores.get('noise_penalty', False),
            'weak_cry_block': scores.get('weak_cry_block', False),
            'moderate_cry_block': scores.get('moderate_cry_block', False),
            'high_rhythm_rescue': scores.get('high_rhythm_rescue', False),
            'low_rhythm_noise_rescue': scores.get('low_rhythm_noise_rescue', False),
            'final_noncry_block': scores.get('final_noncry_block', False),
            'final_cry_rescue': scores.get('final_cry_rescue', False),
            'music_noise_false_positive_block': scores.get('music_noise_false_positive_block', False)
        }

        # Print detailed logs for all files, highlight misclassifications if expected_label is provided
        predicted_label = 'Cry' if is_cry else 'No Cry'
        if expected_label is not None:
            correct = (expected_label == predicted_label)
            if not correct:
                print(f"\nMisclassified file: {file_name}")
                print(f"Expected: {expected_label}, Predicted: {predicted_label}")
                print(f"Confidence: {confidence:.2f}")
                print("Feature values:")
                pprint.pprint({k: features.get(k, None) for k in ['energy_ratio_main', 'amplitude_modulation', 'rhythm_regularity', 'music_ratio', 'noise_ratio', 'duration']})
                print("Decision logic:")
                pprint.pprint(features['decision_logic'])
                print("Individual Scores:")
                pprint.pprint(scores)
        else:
            # Print for all files (optional, comment out if too verbose)
            print(f"\nFile: {file_name}")
            print(f"Predicted: {predicted_label}, Confidence: {confidence:.2f}")
            print("Feature values:")
            pprint.pprint({k: features.get(k, None) for k in ['energy_ratio_main', 'amplitude_modulation', 'rhythm_regularity', 'music_ratio', 'noise_ratio', 'duration']})
            print("Decision logic:")
            pprint.pprint(features['decision_logic'])
            print("Individual Scores:")
            pprint.pprint(scores)

        return {
            'is_cry': bool(is_cry),
            'confidence': confidence,
            'features': features,
            'scores': scores
        } 