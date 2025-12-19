"""
utils/config.py

Este módulo centraliza la configuración global de la aplicación.

Aquí se definen:
- Rutas a recursos importantes del proyecto (modelos entrenados).
- Parámetros compartidos entre distintos componentes.
- Configuraciones relacionadas con el entorno de ejecución.

El objetivo de este archivo es evitar valores hardcodeados distribuidos
en múltiples módulos y facilitar la modificación y mantenimiento del sistema.

Este archivo no contiene:
- Lógica de negocio.
- Código específico de Machine Learning.
- Dependencias con FastAPI u otros frameworks web.
"""

from pathlib import Path
import torch


# Rutas del proyecto

# Directorio base del proyecto (app/)
BASE_DIR = Path(__file__).resolve().parents[1]

# Directorio donde se almacenan los modelos entrenados
MODEL_DIR = BASE_DIR / "models"

# Ruta al pipeline SVM entrenado
SVM_MODEL_PATH = MODEL_DIR / "svm_rbf_pipeline.joblib"


# Configuración de audio y modelos

# Frecuencia de muestreo esperada por Wav2Vec2
SAMPLE_RATE = 16_000

# Dispositivo de cómputo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
