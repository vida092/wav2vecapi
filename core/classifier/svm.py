"""
classifier/svm.py

Este módulo define las funciones de inferencia asociadas al clasificador basado
en SVM utilizado en el sistema.

A partir de un embedding numérico previamente extraído, este módulo:
- Ejecuta la predicción de clase.
- Obtiene probabilidades asociadas a cada clase cuando el modelo lo permite.

El clasificador utilizado corresponde a un pipeline de scikit-learn que incluye
tanto el preprocesamiento como el modelo final, por lo que este módulo asume que
el embedding de entrada es directamente compatible con dicho pipeline.

Este módulo no realiza:
- Carga del modelo desde disco.
- Extracción de embeddings.
- Validaciones de entrada a nivel de API.

Su responsabilidad es ofrecer una interfaz simple y clara para realizar
predicciones a partir de embeddings.
"""

import numpy as np


def predict(embedding: np.ndarray, model):
    """
    Predice la clase asociada a un embedding de audio.

    Parameters
    ----------
    embedding : np.ndarray
        Vector unidimensional que representa el embedding del audio.

    model : sklearn.pipeline.Pipeline
        Pipeline de clasificación entrenado (scaler + SVM).

    Returns
    -------
    label : int
        Etiqueta predicha por el clasificador.
    """

    embedding = embedding.reshape(1, -1)
    label = model.predict(embedding)[0]
    return int(label)


def predict_proba(embedding: np.ndarray, model):
    """
    Obtiene la probabilidad asociada a cada clase para un embedding de audio.

    Parameters
    ----------
    embedding : np.ndarray
        Vector unidimensional que representa el embedding del audio.

    model : sklearn.pipeline.Pipeline
        Pipeline de clasificación entrenado (scaler + SVM).

    Returns
    -------
    probabilities : np.ndarray
        Arreglo con las probabilidades asociadas a cada clase.
    """

    embedding = embedding.reshape(1, -1)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "El modelo cargado no soporta predicción de probabilidades."
        )

    probabilities = model.predict_proba(embedding)[0]
    return probabilities
