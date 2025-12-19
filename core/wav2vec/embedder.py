"""
wav2vec/embedder.py

Este módulo se encarga de transformar señales de audio en embeddings numéricos
utilizando el modelo Wav2Vec2 XLS-R previamente cargado.

A partir de una señal de audio monofónica y normalizada, el módulo:
1. Prepara la entrada mediante el feature extractor de Wav2Vec2.
2. Ejecuta el modelo en modo de inferencia.
3. Agrega las representaciones temporales producidas por la red neuronal
   para obtener un vector de dimensión fija.

El embedding resultante representa el contenido acústico del audio en un
espacio vectorial donde señales similares tienden a ubicarse más cerca.
Este vector es utilizado posteriormente por el clasificador (SVM) para
realizar la detección o discriminación correspondiente.

Este módulo no realiza:
- Carga del modelo Wav2Vec2.
- Lectura directa de archivos de audio.
- Clasificación o evaluación del resultado.

Su responsabilidad es exclusivamente la extracción de embeddings consistentes
con aquellos utilizados durante la fase de entrenamiento del modelo.
"""

import torch
import numpy as np


def extract_embedding(
    audio: np.ndarray,
    feature_extractor,
    model,
    device: str,
    sampling_rate: int = 16_000,
):
    """
    Extrae un embedding de dimensión fija a partir de una señal de audio.

    Parameters
    ----------
    audio : np.ndarray
        Señal de audio monofónica, normalizada, representada como un arreglo
        unidimensional de NumPy.

    feature_extractor : Wav2Vec2FeatureExtractor
        Feature extractor de Wav2Vec2 previamente cargado.

    model : Wav2Vec2Model
        Modelo Wav2Vec2 XLS-R en modo de evaluación.

    device : str
        Dispositivo de cómputo ("cpu" o "cuda").

    sampling_rate : int, optional
        Frecuencia de muestreo del audio (por defecto 16 kHz).

    Returns
    -------
    embedding : np.ndarray
        Vector unidimensional que representa el embedding del audio.
    """

    # Preparación de la entrada para Wav2Vec2
    inputs = feature_extractor(
        audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    # Envío de los tensores al dispositivo correspondiente
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

        # Representaciones latentes por frame
        hidden_states = outputs.last_hidden_state

        # Pooling temporal (mean pooling)
        embedding = hidden_states.mean(dim=1).squeeze()

    return embedding.cpu().numpy()
