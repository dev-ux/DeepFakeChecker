from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import tensorflow as tf
import cv2
import numpy as np
from typing import Optional
import os
from pydantic import BaseModel
import requests
import io
from PIL import Image
from models.deepfake_detector import DeepfakeDetector

app = FastAPI(title="Deepfake Verification Platform")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)

class VerificationResult(BaseModel):
    is_deepfake: bool
    confidence: float
    message: str

@app.get("/proxy/{url:path}")
async def proxy_video(url: str):
    """Proxy pour les vidéos externes"""
    try:
        # Nettoyer l'URL
        url = url.replace('/proxy/', '')  # Enlever le préfixe /proxy/
        url = url.strip('/')  # Enlever les slashes en trop
        
        # Ajouter le protocole si nécessaire
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        
        # Faire la requête avec un timeout
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Erreur lors du téléchargement de la vidéo")
            
        # Récupérer les headers de la réponse originale
        headers = {
            name: value for name, value in response.headers.items()
            if name.lower() in ['content-type', 'content-length', 'content-disposition']
        }
        
        # Retourner le contenu avec les bons headers
        return StreamingResponse(
            io.BytesIO(response.content),
            headers=headers,
            media_type='video/mp4'
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors du téléchargement: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur inattendue: {str(e)}")

def load_model():
    """Initialiser le détecteur de deepfake"""
    try:
        detector = DeepfakeDetector()
        return detector
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def preprocess_video(video_path: str) -> np.ndarray:
    """Prétraiter la vidéo pour l'analyse"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Prendre une frame tous les 10 frames
        if frame_count % 10 == 0:
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    return np.array(frames)

@app.post("/verify/", response_model=VerificationResult)
async def verify_video(request_data: dict = Body(...)):
    """Endpoint pour vérifier une vidéo"""
    try:
        video_url = request_data.get("video_url")
        if not video_url:
            raise HTTPException(status_code=422, detail="video_url est requis")

        # Télécharger la vidéo depuis l'URL avec une gestion robuste
        try:
            # Utiliser stream=True pour gérer les gros fichiers
            response = requests.get(video_url, stream=True, timeout=30)
            response.raise_for_status()  # Lever une exception pour les codes d'erreur HTTP
            
            # Sauvegarder temporairement la vidéo
            temp_video_path = "temp_video.mp4"
            with open(temp_video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Erreur lors du téléchargement de la vidéo: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur inattendue: {str(e)}")
            
        # Prétraiter la vidéo
        frames = preprocess_video(temp_video_path)
        
        # Charger le modèle
        model = load_model()
        
        # Faire la prédiction
        predictions = model.predict(frames)
        average_prediction = np.mean(predictions)
        
        # Interpréter le résultat
        is_deepfake = average_prediction > 0.5
        confidence = average_prediction if is_deepfake else 1 - average_prediction
        
        # Nettoyer le fichier temporaire
        os.remove(temp_video_path)
        
        return VerificationResult(
            is_deepfake=is_deepfake,
            confidence=float(confidence),
            message="Analyse terminée avec succès"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        # Initialiser le détecteur
        detector = load_model()
        
        # Analyser chaque frame
        predictions = []
        for frame in frames:
            # Détecter les visages
            faces = detector.detect_faces(frame)
            if len(faces) > 0:
                # Prendre le premier visage détecté
                (x, y, w, h) = faces[0]
                face = frame[y:y+h, x:x+w]
                
                # Faire la prédiction
                prediction = detector.predict(face)
                predictions.append(prediction)
        
        # Calculer la moyenne des prédictions
        if len(predictions) > 0:
            average_prediction = np.mean(predictions)
            is_deepfake = average_prediction > 0.5
            confidence = average_prediction if is_deepfake else 1 - average_prediction
        else:
            is_deepfake = False
            confidence = 0.0
            raise HTTPException(status_code=400, detail="Aucun visage détecté dans la vidéo")
        
        # Nettoyer le fichier temporaire
        os.remove(temp_video_path)
        
        return VerificationResult(
            is_deepfake=is_deepfake,
            confidence=float(confidence),
            message="Analyse terminée avec succès"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Deepfake Verification Platform"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")
