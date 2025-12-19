"""
main.py

Punto de entrada principal de la aplicación.

Este archivo inicializa la aplicación FastAPI, carga los modelos necesarios
durante el arranque y registra los endpoints disponibles.

El objetivo es:
- Cargar una sola vez los modelos de Machine Learning en memoria.
- Inicializar el servicio de inferencia.
- Exponer la API para su consumo externo.

Este archivo no contiene lógica de negocio ni de Machine Learning.

Montar el servidor (por ahora local) utilizando el comando
python -m uvicorn app.main:app --reload
desde la carpeta contenedora de la caerpeta app 
"""

from fastapi import FastAPI

from app.core.wav2vec.loader import load_wav2vec
from app.core.classifier.loader import load_classifier
from app.services.inference_service import InferenceService
from app.utils.config import DEVICE, SAMPLE_RATE
from app.api.routes.predict import router as predict_router


app = FastAPI(
    title="Audio Fraud Detection API (POC)",
    description="API para detección de fraude en audio utilizando Wav2Vec2 + SVM",
    version="0.1.0",
)


#  carga de modelos

@app.on_event("startup")
def startup_event():
    """
    Evento de arranque de la aplicación.

    Carga en memoria los modelos necesarios para inferencia y
    los deja disponibles a través del estado de la aplicación.
    """

    # Carga de Wav2Vec2
    feature_extractor, wav2vec_model = load_wav2vec()

    # Carga del clasificador
    classifier_model = load_classifier()

    # Inicialización del servicio de inferencia
    inference_service = InferenceService(
        feature_extractor=feature_extractor,
        wav2vec_model=wav2vec_model,
        classifier_model=classifier_model,
        device=DEVICE,
        sampling_rate=SAMPLE_RATE,
    )

    # Almacenamiento en el estado global de la app
    app.state.inference_service = inference_service

@app.get("/")
def saludo():
    return {"message": "Deepfake detector utilizando wav2vec"}

app.include_router(predict_router, prefix="/api")
