"""
classifier/loader.py

Este módulo se encarga de la carga del clasificador entrenado utilizado en la
etapa final de inferencia del sistema.

El clasificador se encuentra almacenado como un artefacto serializado en formato
Joblib y corresponde a un pipeline de scikit-learn que incluye tanto el
preprocesamiento (escalamiento) como el modelo de clasificación (SVM con kernel RBF).

La responsabilidad de este módulo es:
- Cargar una única instancia del pipeline de clasificación en memoria.
- Exponer el modelo listo para inferencia.
- Asegurar que el mismo pipeline utilizado durante el entrenamiento sea empleado
  durante la predicción, garantizando consistencia en los resultados.

Este módulo no realiza:
- Entrenamiento o ajuste del modelo.
- Evaluación de métricas.
- Extracción de embeddings.
- Lógica de negocio o de API.

Su único propósito es encapsular la inicialización del clasificador dentro de la
arquitectura de la aplicación.
"""

import joblib
from app.utils.config import SVM_MODEL_PATH


def load_classifier():
    """
    Carga el pipeline de clasificación entrenado desde disco.

    El pipeline incluye internamente los pasos necesarios de preprocesamiento
    (por ejemplo, StandardScaler) y el clasificador final (SVM con kernel RBF),
    por lo que no es necesario aplicar transformaciones adicionales a los
    embeddings antes de la predicción.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Pipeline de scikit-learn listo para realizar inferencia.
    """

    model = joblib.load(SVM_MODEL_PATH)
    return model
