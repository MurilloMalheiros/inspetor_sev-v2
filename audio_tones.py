"""Gera tons WAV em memoria para o mixer do Pygame."""

from __future__ import annotations

import io
import math
import wave
from array import array
from functools import lru_cache


@lru_cache(maxsize=64)
def tone_wav(frequency: int, duration: int, sample_rate: int = 22050) -> bytes:
    """Cria um tom mono com pequenas rampas para evitar estalos."""

    sample_count = max(1, int(sample_rate * duration / 1000))
    fade_count = min(int(sample_rate * 0.012), sample_count // 2)
    pcm = array("h")
    for index in range(sample_count):
        envelope = 1.0
        if fade_count and index < fade_count:
            envelope = index / fade_count
        elif fade_count and index >= sample_count - fade_count:
            envelope = (sample_count - index - 1) / fade_count
        sample = math.sin(2.0 * math.pi * frequency * index / sample_rate)
        pcm.append(int(sample * envelope * 0.70 * 32767))

    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()


@lru_cache(maxsize=4)
def siren_wav(duration: int = 700, sample_rate: int = 22050) -> bytes:
    """Cria uma sirene alternada que pode ser repetida sem pausas."""

    half_samples = max(1, int(sample_rate * duration / 2000))
    pcm = array("h")
    for frequency in (1900, 2400):
        for index in range(half_samples):
            fade_count = min(int(sample_rate * 0.012), half_samples // 2)
            envelope = 1.0
            if fade_count and index < fade_count:
                envelope = index / fade_count
            elif fade_count and index >= half_samples - fade_count:
                envelope = (half_samples - index - 1) / fade_count
            sample = math.sin(2.0 * math.pi * frequency * index / sample_rate)
            pcm.append(int(sample * envelope * 0.70 * 32767))

    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()
