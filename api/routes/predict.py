"""
api/routes/predict.py

Este módulo define el endpoint de predicción de la API.

El endpoint permite recibir un archivo de audio (WAV o FLAC), procesarlo a través
del pipeline de inferencia y devolver el resultado de la clasificación en formato
JSON.

Este módulo se encarga únicamente de:
- Validar la entrada del usuario.
- Delegar la inferencia al servicio correspondiente.
- Construir la respuesta HTTP.

No contiene lógica de Machine Learning ni procesamiento avanzado de audio.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from tempfile import NamedTemporaryFile
from pathlib import Path

from app.utils.audio import load_audio

tmp_path = None
router = APIRouter()


@router.post("/predict")
async def predict_audio(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Ejecuta la predicción sobre un archivo de audio.

    Parameters
    ----------
    file : UploadFile
        Archivo de audio en formato WAV o FLAC.

    Returns
    -------
    dict
        Resultado de la inferencia, incluyendo la clase predicha y las
        probabilidades asociadas.
    """

    # Validación básica del tipo de archivo
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se recibió ningún archivo.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".wav", ".flac"}:
        raise HTTPException(
            status_code=400,
            detail="Formato no soportado. Use archivos WAV o FLAC.",
        )

    # Guardado temporal del archivo
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        audio = load_audio(tmp_path)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar el archivo de audio: {str(e)}",
        )

    # Obtención del servicio de inferencia
    inference_service = request.app.state.inference_service

    # Ejecución del pipeline de inferencia
    try:
        result = inference_service.run(audio)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la inferencia: {str(e)}",
        )

    return result
