import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

class MoodClassifier:
    """
    Classifies music mood based on audio features using advanced signal processing.
    Analyzes tempo, energy, valence and other characteristics to determine emotional qualities.
    """

    MOOD_CATEGORIES = {
        'happy': {'energy': 0.7, 'tempo': 120, 'valence': 0.8},
        'sad': {'energy': 0.3, 'tempo': 80, 'valence': 0.2}, 
        'energetic': {'energy': 0.9, 'tempo': 140, 'valence': 0.6},
        'calm': {'energy': 0.2, 'tempo': 90, 'valence': 0.5},
        'dark': {'energy': 0.4, 'tempo': 100, 'valence': 0.1}
    }

    def __init__(self):
        self.scaler = StandardScaler()

    def extract_features(self, audio_path: str) -> Dict[str, float]:
        """Extract relevant audio features from the track."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path)

            # Extract features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            mfcc = librosa.feature.mfcc(y=y, sr=sr)

            # Calculate aggregate features
            energy = np.mean(librosa.feature.rms(y=y)[0])
            valence = np.mean(spectral_centroids) / sr
            
            return {
                'energy': energy,
                'tempo': tempo,
                'valence': valence,
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                'mfcc_mean': np.mean(mfcc)
            }
        except Exception as e:
            raise RuntimeError(f'Error extracting features: {str(e)}')

    def classify_mood(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Determine the mood based on extracted features."""
        
        # Calculate distances to each mood category
        distances = {}
        for mood, target in self.MOOD_CATEGORIES.items():
            dist = (
                (features['energy'] - target['energy']) ** 2 +
                (features['tempo'] / 180 - target['tempo'] / 180) ** 2 +
                (features['valence'] - target['valence']) ** 2
            ) ** 0.5
            distances[mood] = dist

        # Find closest mood
        predicted_mood = min(distances.items(), key=lambda x: x[1])[0]
        confidence = 1 / (1 + min(distances.values()))

        return predicted_mood, confidence

    def analyze_track(self, audio_path: str) -> Dict[str, any]:
        """Complete mood analysis of an audio track."""
        
        features = self.extract_features(audio_path)
        mood, confidence = self.classify_mood(features)

        return {
            'mood': mood,
            'confidence': confidence,
            'features': features
        }

    def get_mood_description(self, mood: str) -> str:
        """Returns a text description of the mood's musical characteristics."""
        
        descriptions = {
            'happy': 'Upbeat and bright with high energy and positive valence',
            'sad': 'Slow tempo with low energy and negative valence',
            'energetic': 'Fast-paced with very high energy and moderate valence',
            'calm': 'Gentle and soothing with low energy and neutral valence',
            'dark': 'Moderate tempo with low valence and moderate energy'
        }
        return descriptions.get(mood, 'Unknown mood characteristics')