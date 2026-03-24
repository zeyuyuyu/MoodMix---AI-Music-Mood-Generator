import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class MoodDetector:
    def __init__(self):
        # Load pre-trained face detection cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Load pre-trained emotion detection model
        self.emotion_model = load_model('models/emotion_model.h5')
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def detect_mood_from_camera(self):
        cap = cv2.VideoCapture(0)
        emotions_buffer = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = img_to_array(face_roi)
                face_roi = np.expand_dims(face_roi, axis=0)
                face_roi = face_roi / 255.0

                # Predict emotion
                predictions = self.emotion_model.predict(face_roi)[0]
                emotion_idx = np.argmax(predictions)
                emotion = self.emotions[emotion_idx]
                confidence = predictions[emotion_idx]

                # Add to buffer for smoothing
                emotions_buffer.append(emotion)
                if len(emotions_buffer) > 30:  # Keep last 30 frames
                    emotions_buffer.pop(0)

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display detected emotion
                dominant_emotion = max(set(emotions_buffer), key=emotions_buffer.count)
                cv2.putText(frame, f'{dominant_emotion} ({confidence:.2f})',
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                           (0, 255, 0), 2)

            cv2.imshow('Mood Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        # Return the most frequent emotion in buffer
        if emotions_buffer:
            return max(set(emotions_buffer), key=emotions_buffer.count)
        return 'Neutral'

    def get_mood_mapping(self, emotion):
        """Map detected emotion to musical characteristics"""
        mood_mappings = {
            'Happy': {'tempo': 'fast', 'mode': 'major', 'energy': 'high'},
            'Sad': {'tempo': 'slow', 'mode': 'minor', 'energy': 'low'},
            'Angry': {'tempo': 'fast', 'mode': 'minor', 'energy': 'high'},
            'Neutral': {'tempo': 'medium', 'mode': 'major', 'energy': 'medium'},
            'Surprise': {'tempo': 'fast', 'mode': 'major', 'energy': 'high'},
            'Fear': {'tempo': 'medium', 'mode': 'minor', 'energy': 'medium'},
            'Disgust': {'tempo': 'slow', 'mode': 'minor', 'energy': 'medium'}
        }
        return mood_mappings.get(emotion, mood_mappings['Neutral'])

if __name__ == '__main__':
    detector = MoodDetector()
    detected_mood = detector.detect_mood_from_camera()
    print(f'Detected mood: {detected_mood}')