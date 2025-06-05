from fastapi import FastAPI, HTTPException
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Video Proxy")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/proxy/{url:path}")
async def proxy_video(url: str):
    """Proxy pour les vidéos externes"""
    try:
        # Ajouter le protocole si nécessaire
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
            
        # Faire la requête avec un timeout
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Erreur lors du téléchargement de la vidéo")
            
        # Retourner le contenu avec les bons headers
        return response.content
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors du téléchargement: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur inattendue: {str(e)}")
