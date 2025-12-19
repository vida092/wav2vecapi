"""
services/inference_service.py

Este módulo implementa el flujo completo de inferencia del sistema, conectando
los distintos componentes del core de Machine Learning.

A partir de un archivo de audio previamente cargado y validado, el servicio:
1. Convierte la señal de audio en un embedding utilizando Wav2Vec2 XLS-R.
2. Aplica el clasificador entrenado (pipeline SVM).
3. Devuelve el resultado de la predicción en un formato estructurado.

Este módulo actúa como una capa de orquestación entre:
- La extracción de embeddings (Wav2Vec2).
- El modelo de clasificación (SVM).
- La capa de API (FastAPI).

No contiene lógica específica de HTTP ni detalles de infraestructura.
Su propósito es desacoplar la lógica de inferencia del framework web, permitiendo
que el pipeline pueda ser reutilizado, probado o extendido fácilmente.
"""

import numpy as np

from app.core.wav2vec.embedder import extract_embedding
from app.core.classifier.svm import predict, predict_proba


class InferenceService:
    """
    Servicio de inferencia que encapsula el pipeline completo de predicción.

    Esta clase mantiene referencias a los modelos cargados en memoria y expone
    un método único para realizar inferencia a partir de audio crudo.
    """

    def __init__(
        self,
        feature_extractor,
        wav2vec_model,
        classifier_model,
        device: str,
        sampling_rate: int = 16_000,
    ):
        """
        Inicializa el servicio de inferencia.

        Parameters
        ----------
        feature_extractor : Wav2Vec2FeatureExtractor
            Feature extractor de Wav2Vec2 previamente cargado.

        wav2vec_model : Wav2Vec2Model
            Modelo Wav2Vec2 XLS-R en modo de evaluación.

        classifier_model : sklearn.pipeline.Pipeline
            Pipeline de clasificación entrenado (scaler + SVM).

        device : str
            Dispositivo de cómputo ("cpu" o "cuda").

        sampling_rate : int, optional
            Frecuencia de muestreo esperada del audio.
        """

        self.feature_extractor = feature_extractor
        self.wav2vec_model = wav2vec_model
        self.classifier_model = classifier_model
        self.device = device
        self.sampling_rate = sampling_rate

    def run(self, audio: np.ndarray) -> dict:
        """
        Ejecuta el pipeline completo de inferencia sobre una señal de audio.

        Parameters
        ----------
        audio : np.ndarray
            Señal de audio monofónica, normalizada y muestreada a la frecuencia
            esperada por el modelo.

        Returns
        -------
        result : dict
            Diccionario con el resultado de la inferencia, incluyendo la clase
            predicha y las probabilidades asociadas.
        """

        # 1. Extracción del embedding
        embedding = extract_embedding(
            audio=audio,
            feature_extractor=self.feature_extractor,
            model=self.wav2vec_model,
            device=self.device,
            sampling_rate=self.sampling_rate,
        )

        # 2. Predicción
        label = predict(embedding, self.classifier_model)

        probabilities = None
        try:
            probabilities = predict_proba(embedding, self.classifier_model)
            probabilities = probabilities.tolist()
        except RuntimeError:
            pass

        # 3. Resultado estructurado
        result = {
            "label": label,
            "probabilities": probabilities,
        }

        return result
