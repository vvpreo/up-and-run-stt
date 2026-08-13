"""
Утилиты для загрузки и обработки аудио.
Предоставляет функции для загрузки аудиофайлов различных форматов
и их преобразования в формат, требуемый Whisper (16kHz mono float32).
"""

import io
import logging
import subprocess
import tempfile
from typing import Tuple

import numpy as np
import soundfile as sf
import librosa

from src.config import SAMPLE_RATE

logger = logging.getLogger(__name__)


def load_audio_from_file(audio_content: bytes) -> np.ndarray:
    """
    Загружает аудио из байтов и преобразует в numpy array.

    Поддерживает различные форматы аудио (wav, flac, ogg, mp3 и др.)
    и автоматически преобразует в формат, требуемый Whisper:
    - Частота дискретизации: 16kHz
    - Каналы: моно
    - Тип данных: float32

    Args:
        audio_content: Байты аудиофайла.

    Returns:
        np.ndarray: Аудиоданные в формате float32 с частотой 16kHz.

    Raises:
        Exception: Если аудио не удалось загрузить ни одним из методов.
    """
    audio_buffer = io.BytesIO(audio_content)

    try:
        # Try soundfile first (faster; wav/flac/ogg/opus/mp3 via libsndfile)
        audio_data, sample_rate = sf.read(audio_buffer)
        logger.debug(f"Audio loaded with soundfile: {sample_rate}Hz")
    except Exception as e:
        # Fallback to ffmpeg (m4a/aac, webm, wma и всё остальное).
        # Декодирует сразу в 16kHz mono float32 — ресемплинг не нужен.
        logger.debug(f"soundfile failed ({e}), falling back to ffmpeg")
        audio_data = _decode_with_ffmpeg(audio_content)
        sample_rate = SAMPLE_RATE

    # Free the BytesIO buffer immediately — no longer needed
    audio_buffer.close()
    del audio_buffer

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        stereo = audio_data
        audio_data = np.mean(stereo, axis=1)
        del stereo
        logger.debug("Converted stereo to mono")

    # Resample to 16kHz if needed (Whisper requirement)
    if sample_rate != SAMPLE_RATE:
        logger.debug(f"Resampling from {sample_rate}Hz to {SAMPLE_RATE}Hz")
        original = audio_data
        audio_data = librosa.resample(
            original,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE,
        )
        del original

    # Ensure float32 (avoid copy if already float32)
    if audio_data.dtype != np.float32:
        return audio_data.astype(np.float32)
    return audio_data


def _decode_with_ffmpeg(audio_content: bytes) -> np.ndarray:
    """
    Декодирует аудио через ffmpeg (временный файл -> raw float32 16kHz mono).

    Используется как фолбэк для форматов, которые не умеет libsndfile
    (m4a/aac, webm, wma, ...). Требует бинарь `ffmpeg` (есть в образе).
    Вход подаётся через временный файл, а не stdin: MP4/M4A с `moov atom`
    в конце файла из неперематываемого pipe не декодируется.

    Raises:
        RuntimeError: Если ffmpeg не смог декодировать данные.
    """
    with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
        tmp.write(audio_content)
        tmp.flush()
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-i", tmp.name,
            "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        stderr = proc.stderr.decode(errors="replace").strip().splitlines()
        detail = stderr[-1] if stderr else "unknown error"
        raise RuntimeError(f"ffmpeg failed to decode audio: {detail}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def get_audio_duration(audio_data: np.ndarray) -> float:
    """
    Вычисляет длительность аудио в секундах.

    Args:
        audio_data: Аудиоданные в формате numpy array (16kHz).

    Returns:
        Длительность аудио в секундах.
    """
    return len(audio_data) / SAMPLE_RATE


def load_audio_from_path(file_path: str) -> np.ndarray:
    """
    Загружает аудио из файла по пути.

    Args:
        file_path: Путь к аудиофайлу.

    Returns:
        np.ndarray: Аудиоданные в формате float32 с частотой 16kHz.
    """
    with open(file_path, "rb") as f:
        audio_content = f.read()
    return load_audio_from_file(audio_content)


def validate_audio(audio_data: np.ndarray) -> Tuple[bool, str]:
    """
    Валидирует аудиоданные.

    Проверяет:
    - Наличие данных
    - Допустимый диапазон значений
    - Минимальную длительность

    Args:
        audio_data: Аудиоданные для проверки.

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if audio_data is None or len(audio_data) == 0:
        return False, "Audio data is empty"

    duration = get_audio_duration(audio_data)
    if duration < 0.1:
        return False, f"Audio too short: {duration:.2f}s (minimum 0.1s)"

    if duration > 3600:
        return False, f"Audio too long: {duration:.2f}s (maximum 3600s)"

    # Check for valid audio range
    if np.max(np.abs(audio_data)) > 1.0:
        logger.warning("Audio values exceed [-1, 1] range, normalizing")

    return True, ""


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Нормализует аудиоданные.

    Приводит значения к диапазону [-1, 1] если они выходят за пределы.

    Args:
        audio_data: Аудиоданные для нормализации.

    Returns:
        np.ndarray: Нормализованные аудиоданные.
    """
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        return audio_data / max_val
    return audio_data
