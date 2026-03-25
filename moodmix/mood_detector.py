import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class EmotionDetector:
    def __init__(self, model_path='moodmix/emotion_model.h5'):
        self.model = load_model(model_path)

    def detect_emotion(self, audio_data):
        """
        Detects the dominant emotion in the given audio data.
        
        Args:
            audio_data (np.ndarray): The audio data to analyze.
        
        Returns:
            str: The detected emotion ('happy', 'sad', 'angry', 'calm', 'fearful', 'surprised', 'neutral').
        """
        # Preprocess the audio data
        audio_data = self.preprocess_audio(audio_data)
        
        # Make a prediction using the emotion detection model
        prediction = self.model.predict(audio_data)
        emotion_index = np.argmax(prediction[0])
        
        # Map the emotion index to a label
        emotions = ['happy', 'sad', 'angry', 'calm', 'fearful', 'surprised', 'neutral']
        return emotions[emotion_index]
    
    def preprocess_audio(self, audio_data):
        """
        Preprocesses the audio data for emotion detection.
        
        Args:
            audio_data (np.ndarray): The audio data to preprocess.
        
        Returns:
            np.ndarray: The preprocessed audio data.
        """
        # Perform any necessary preprocessing steps, such as normalization, resampling, or feature extraction
        audio_data = np.expand_dims(audio_data, axis=0)
        return audio_data
