import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from statistics import mode

class MoodDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_model = load_model('models/emotion_model.h5')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.mood_buffer = []
        self.buffer_size = 10
    
    def preprocess_face(self, face_img):
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float') / 255.0
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        return face_img
    
    def detect_mood(self, frame=None):
        if frame is None:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if not ret:
                return None
            cap.release()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return 'neutral'
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            processed_face = self.preprocess_face(roi_gray)
            emotion_pred = self.emotion_model.predict(processed_face)[0]
            emotion_label = self.emotions[np.argmax(emotion_pred)]
            
            self.mood_buffer.append(emotion_label)
            if len(self.mood_buffer) > self.buffer_size:
                self.mood_buffer.pop(0)
        
        # Return most common emotion in buffer for stability
        return mode(self.mood_buffer) if self.mood_buffer else 'neutral'
    
    def get_mood_intensity(self):
        """Returns intensity level of current mood (0-1)"""
        if not self.mood_buffer:
            return 0.5
        
        current_mood = mode(self.mood_buffer)
        intensity = self.mood_buffer.count(current_mood) / len(self.mood_buffer)
        return intensity
    
    def get_mood_valence(self, mood):
        """Maps detected mood to valence score (-1 to 1)"""
        valence_map = {
            'happy': 0.8,
            'surprise': 0.4,
            'neutral': 0.0,
            'sad': -0.6,
            'fear': -0.7,
            'disgust': -0.8,
            'angry': -0.9
        }
        return valence_map.get(mood, 0.0)

    def get_continuous_mood_vector(self):
        """Returns continuous mood parameters for music generation"""
        current_mood = self.detect_mood()
        intensity = self.get_mood_intensity()
        valence = self.get_mood_valence(current_mood)
        
        return {
            'mood': current_mood,
            'intensity': intensity,
            'valence': valence
        }