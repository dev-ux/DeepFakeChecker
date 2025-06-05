import tensorflow as tf
import numpy as np
import cv2
import tensorflow_hub as hub

class DeepfakeDetector:
    def __init__(self):
        self.model = self._build_model()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def _build_model(self):
        """Créer le modèle de détection de deepfake avec Xception pré-entraîné"""
        # Charger le modèle Xception pré-entraîné sur FaceForensics++
        model_url = "https://tfhub.dev/deepmind/deepfake-detection/1"
        model = tf.keras.Sequential([
            hub.KerasLayer(model_url, input_shape=(299, 299, 3))
        ])
        
        return model
    
    def preprocess_frame(self, frame):
        """Prétraitement d'une frame pour le modèle Xception"""
        # Redimensionner la frame
        frame = cv2.resize(frame, (299, 299))
        # Conversion en RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalisation
        frame = frame / 255.0
        # Ajouter une dimension pour le batch
        frame = np.expand_dims(frame, axis=0)
        return frame
    
    def predict(self, frame):
        """Prédire si la frame est un deepfake"""
        # Prétraitement de la frame
        processed_frame = self.preprocess_frame(frame)
        # Prédiction
        prediction = self.model.predict(processed_frame)
        # Le modèle retourne une probabilité entre 0 et 1
        # Plus la valeur est proche de 1, plus c'est probablement un deepfake
        return prediction[0][0]
    
    def detect_faces(self, frame):
        """Détection des visages dans une frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    
    def preprocess_frame(self, frame):
        """Prétraitement d'une frame"""
        # Redimensionner la frame
        frame = cv2.resize(frame, (224, 224))
        # Conversion en RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalisation
        frame = frame / 255.0
        return frame
    
    def predict(self, frame):
        """Prédire si la frame est un deepfake"""
        # Prétraitement de la frame
        processed_frame = self.preprocess_frame(frame)
        # Ajouter une dimension pour le batch
        processed_frame = np.expand_dims(processed_frame, axis=0)
        # Prédiction
        prediction = self.model.predict(processed_frame)
        return prediction[0][0]
