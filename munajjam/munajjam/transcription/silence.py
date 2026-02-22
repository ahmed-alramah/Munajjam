"""
Silence detection utilities for audio processing.


Provides both pydub (accurate) and librosa (fast) implementations.
Use the fast implementation for long files (>5 minutes).
"""


from pathlib import Path
from typing import Any




def detect_silences(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
    use_fast: bool = True,
) -> list[tuple[int, int]]:
    """
    Detect silent portions in an audio file.


    Args:
        audio_path: Path to the audio file
        min_silence_len: Minimum silence length in milliseconds
        silence_thresh: Silence threshold in dB
        use_fast: Use fast librosa-based detection (recommended for long files)


    Returns:
        List of (start_ms, end_ms) tuples for silent portions
    """
    if use_fast:
        try:
            return _detect_silences_fast(audio_path, min_silence_len, silence_thresh)
        except:  # noqa: E722 - Bare except to catch numpy/scipy C-extension errors
            pass  # Fallback to pydub


    return _detect_silences_pydub(audio_path, min_silence_len, silence_thresh)




def _detect_silences_pydub(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
) -> list[tuple[int, int]]:
    """Pydub-based silence detection (slower but reliable)."""
    from pydub import AudioSegment, silence


    audio = AudioSegment.from_wav(str(audio_path))
    silences = silence.detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )


    return [(s[0], s[1]) for s in silences]




def _detect_silences_fast(
    audio_path: str | Path,
    min_silence_len: int = 300,
    silence_thresh: int = -30,
) -> list[tuple[int, int]]:
    """
    Fast silence detection using librosa + numpy.


    ~10-50x faster than pydub for long files.
    """
    # Import inside try block to catch numpy/scipy version conflicts
    try:
        import librosa
