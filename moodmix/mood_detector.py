import numpy as np
from typing import Tuple, Dict
import librosa

class MoodDetector:
    def __init__(self):
        self.valence_features = ['spectral_rolloff', 'tempo', 'spectral_centroid']
        self.arousal_features = ['rms_energy', 'zero_crossing_rate', 'spectral_contrast']
        
    def extract_audio_features(self, audio_path: str) -> Dict[str, float]:
        """Extract relevant audio features for mood detection."""
        y, sr = librosa.load(audio_path)
        
        features = {}
        # Valence features
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features['tempo'] = librosa.beat.tempo(y=y, sr=sr)[0]
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Arousal features
        features['rms_energy'] = np.mean(librosa.feature.rms(y=y))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y=y))
        features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        
        return features

    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize feature values to [0,1] range."""
        # Pre-defined normalization ranges based on typical values
        ranges = {
            'spectral_rolloff': (500, 12000),
            'tempo': (50, 200),
            'spectral_centroid': (500, 10000),
            'rms_energy': (0.01, 0.5),
            'zero_crossing_rate': (0.01, 0.3),
            'spectral_contrast': (-50, 50)
        }
        
        normalized = {}
        for feature, value in features.items():
            min_val, max_val = ranges[feature]
            normalized[feature] = np.clip((value - min_val) / (max_val - min_val), 0, 1)
        
        return normalized

    def calculate_mood_coordinates(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Calculate valence and arousal coordinates from audio features."""
        normalized = self.normalize_features(features)
        
        # Calculate valence (x-axis) - emotional positivity
        valence = np.mean([normalized[f] for f in self.valence_features])
        
        # Calculate arousal (y-axis) - energy level
        arousal = np.mean([normalized[f] for f in self.arousal_features])
        
        return valence, arousal

    def detect_mood(self, audio_path: str) -> Dict[str, float]:
        """Detect mood from audio file using valence-arousal model."""
        features = self.extract_audio_features(audio_path)
        valence, arousal = self.calculate_mood_coordinates(features)
        
        # Map coordinates to mood categories
        moods = {
            'happy': max(0, valence * arousal),
            'sad': max(0, (1-valence) * (1-arousal)),
            'angry': max(0, (1-valence) * arousal),
            'relaxed': max(0, valence * (1-arousal))
        }
        
        # Normalize mood scores to sum to 1
        total = sum(moods.values())
        if total > 0:
            moods = {k: v/total for k, v in moods.items()}
            
        return {
            'valence': valence,
            'arousal': arousal,
            'moods': moods
        }