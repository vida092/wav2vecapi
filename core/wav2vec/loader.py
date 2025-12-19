"""
wav2vec/loader.py

Este módulo se encarga de la carga y configuración del modelo Wav2Vec2 XLS-R
preentrenado, el cual es utilizado como extractor de representaciones (embeddings)
a partir de señales de audio.

El modelo cargado en este archivo no se entrena ni se ajusta nuevamente; se utiliza
exclusivamente en modo de inferencia para transformar audios crudos (por ejemplo,
archivos WAV o FLAC) en vectores numéricos de dimensión fija (2024).

La responsabilidad de este módulo es:
- Cargar una única instancia del feature extractor y del modelo Wav2Vec2.
- Configurar el dispositivo de cómputo (CPU o GPU).
- Exponer estos objetos para ser reutilizados por otros componentes del sistema,
  evitando cargas repetidas y reduciendo la latencia en entornos de producción.

Este módulo no realiza:
- Lectura o preprocesamiento de archivos de audio.
- Extracción directa de embeddings.
- Clasificación o toma de decisiones.

Su único propósito es centralizar y encapsular la inicialización del modelo
Wav2Vec2 dentro de la arquitectura de la API.
"""

import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


# Identificador del modelo preentrenado utilizado en este POC
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"


# Selección del dispositivo de cómputo
# yo lo probé con CUDA pero en un servidor en la nube puede ser cpu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_wav2vec():
    """
    Carga el feature extractor y el modelo Wav2Vec2 XLS-R en memoria.

    El modelo se establece en modo de evaluación (eval) y se transfiere
    al dispositivo de cómputo disponible.

    Returns
    -------
    feature_extractor : Wav2Vec2FeatureExtractor
        Objeto encargado de preparar la señal de audio para el modelo.

    model : Wav2Vec2Model
        Modelo Wav2Vec2 XLS-R preentrenado, listo para inferencia.
    """

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    return feature_extractor, model
