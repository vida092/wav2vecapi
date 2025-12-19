"""
utils/audio.py

Este módulo proporciona utilidades para la carga y preprocesamiento de archivos
de audio utilizados por el sistema de inferencia.

Su responsabilidad es:
- Leer archivos de audio en formatos soportados (WAV y FLAC).
- Convertir la señal a formato monofónico.
- Ajustar la frecuencia de muestreo al valor esperado por el modelo.
- Normalizar la señal para garantizar estabilidad numérica.

El audio resultante es entregado como un arreglo de NumPy compatible con el
pipeline de extracción de embeddings basado en Wav2Vec2.

Este módulo no realiza:
- Extracción de embeddings.
- Clasificación.
- Validaciones a nivel de API (por ejemplo, tamaño máximo de archivo).

Su propósito es servir como capa de preparación de datos entre la entrada del
usuario y el core de Machine Learning.
"""

from pathlib import Path
import numpy as np
import librosa


SUPPORTED_EXTENSIONS = {".wav", ".flac"}


def load_audio(
    file_path: str | Path,
    target_sr: int = 16_000,
) -> np.ndarray:
    """
    Carga un archivo de audio desde disco y lo prepara para inferencia.

    El audio se convierte a mono, se remuestrea a la frecuencia objetivo y se
    normaliza para asegurar valores en el rango [-1, 1].

    Parameters
    ----------
    file_path : str or Path
        Ruta al archivo de audio (WAV o FLAC).

    target_sr : int, optional
        Frecuencia de muestreo objetivo (por defecto 16 kHz).

    Returns
    -------
    audio : np.ndarray
        Señal de audio monofónica, normalizada y lista para ser procesada por
        Wav2Vec2.

    Raises
    ------
    ValueError
        Si el archivo no existe o el formato no es soportado.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise ValueError(f"El archivo de audio no existe: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Formato de audio no soportado: {file_path.suffix}. "
            f"Formatos permitidos: {SUPPORTED_EXTENSIONS}"
        )

    # Carga del audio
    audio, sr = librosa.load(
        file_path,
        sr=target_sr,
        mono=True,
    )

    # Normalización (seguridad numérica)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio
