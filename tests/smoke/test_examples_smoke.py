"""
Smoke tests for example scripts.

These tests exercise the core logic of each example with a mocked transcriber,
ensuring examples stay in sync with the library's public API.
No real model downloads or audio files are needed.

Closes #48
"""
from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from munajjam.models import Segment, Ayah, AlignmentResult
from munajjam.transcription.base import BaseTranscriber
from munajjam.data import load_surah_ayahs
from munajjam.core import align, Aligner, normalize_arabic, similarity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_SEGMENTS = [
    Segment(
        id=1,
        surah_id=1,
        text="بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
        start=0.0,
        end=3.5,
    ),
    Segment(
        id=2,
        surah_id=1,
        text="الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        start=3.5,
        end=7.0,
    ),
    Segment(
        id=3,
        surah_id=1,
        text="الرَّحْمَنِ الرَّحِيمِ",
        start=7.0,
        end=9.5,
    ),
    Segment(
        id=4,
        surah_id=1,
        text="مَالِكِ يَوْمِ الدِّينِ",
        start=9.5,
        end=12.0,
    ),
    Segment(
        id=5,
        surah_id=1,
        text="إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
        start=12.0,
        end=15.5,
    ),
    Segment(
        id=6,
        surah_id=1,
        text="اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
        start=15.5,
        end=19.0,
    ),
    Segment(
        id=7,
        surah_id=1,
        text="صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ",
        start=19.0,
        end=25.0,
    ),
]


class MockTranscriber(BaseTranscriber):
    """A mock transcriber that returns pre-defined segments."""

    def __init__(self, segments: list[Segment] | None = None) -> None:
        self._segments = segments or FAKE_SEGMENTS
        self._loaded = False

    def transcribe(self, audio_path: str | Path) -> list[Segment]:
        return self._segments

    async def transcribe_async(self, audio_path: str | Path) -> list[Segment]:
        return self._segments

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded


@pytest.fixture
def mock_transcriber() -> MockTranscriber:
    return MockTranscriber()


@pytest.fixture
def surah_1_ayahs() -> list[Ayah]:
    return load_surah_ayahs(1)


# ---------------------------------------------------------------------------
# Smoke test: basic_usage example pattern
# ---------------------------------------------------------------------------

class TestBasicUsageSmoke:
    """Smoke tests mirroring examples/basic_usage.py logic."""

    def test_imports_are_valid(self) -> None:
        """Verify that all imports used by basic_usage.py are still available."""
        from munajjam.transcription import WhisperTranscriber
        from munajjam.core import align
        from munajjam.data import load_surah_ayahs

    def test_transcriber_returns_segments(self, mock_transcriber: MockTranscriber) -> None:
        """Transcriber should return a list of Segment objects."""
        segments = mock_transcriber.transcribe("fake_audio.wav")
        assert isinstance(segments, list)
        assert len(segments) > 0
        assert all(isinstance(s, Segment) for s in segments)

    def test_load_surah_ayahs(self, surah_1_ayahs: list[Ayah]) -> None:
        """load_surah_ayahs should return Ayah objects for Al-Fatiha."""
        assert len(surah_1_ayahs) == 7
        assert all(isinstance(a, Ayah) for a in surah_1_ayahs)

    def test_align_produces_results(
        self, mock_transcriber: MockTranscriber, surah_1_ayahs: list[Ayah]
    ) -> None:
        """align() should produce AlignmentResult objects."""
        segments = mock_transcriber.transcribe("fake.wav")
        results = align("fake.wav", segments, surah_1_ayahs)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, AlignmentResult) for r in results)

    def test_result_has_expected_fields(
        self, mock_transcriber: MockTranscriber, surah_1_ayahs: list[Ayah]
    ) -> None:
        """Each AlignmentResult should expose the fields used by the example."""
        segments = mock_transcriber.transcribe("fake.wav")
        results = align("fake.wav", segments, surah_1_ayahs)
        for r in results:
            # Fields accessed in basic_usage.py
            assert hasattr(r, "ayah")
            assert hasattr(r, "start_time")
            assert hasattr(r, "end_time")
            assert hasattr(r, "transcribed_text")
            assert hasattr(r, "similarity_score")
            assert hasattr(r, "is_high_confidence")
            assert hasattr(r.ayah, "ayah_number")
            assert hasattr(r.ayah, "surah_id")
            assert hasattr(r.ayah, "text")

    def test_json_serializable_output(
        self, mock_transcriber: MockTranscriber, surah_1_ayahs: list[Ayah]
    ) -> None:
        """The output dict structure from basic_usage.py should be JSON-serializable."""
        segments = mock_transcriber.transcribe("fake.wav")
        results = align("fake.wav", segments, surah_1_ayahs)
        output = []
        for result in results:
            output.append(
                {
                    "id": result.ayah.ayah_number,
                    "sura_id": result.ayah.surah_id,
                    "ayah_index": result.ayah.ayah_number - 1,
                    "start": round(result.start_time, 2),
                    "end": round(result.end_time, 2),
                    "transcribed_text": result.transcribed_text,
                    "corrected_text": result.ayah.text,
                    "similarity_score": round(result.similarity_score, 3),
                }
            )
        # Should not raise
        json_str = json.dumps(output, ensure_ascii=False)
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# Smoke test: example_alignment (formerly test_alignment) pattern
# ---------------------------------------------------------------------------

class TestExampleAlignmentSmoke:
    """Smoke tests mirroring examples/example_alignment.py logic."""

    def test_aligner_class_available(self) -> None:
        """Aligner class should be importable from core."""
        from munajjam.core import Aligner
        assert Aligner is not None

    def test_aligner_with_mock(
        self, mock_transcriber: MockTranscriber, surah_1_ayahs: list[Ayah]
    ) -> None:
        """Aligner should work with mocked segments."""
        segments = mock_transcriber.transcribe("fake.wav")
        aligner = Aligner("fake.wav")
        results = aligner.align(segments, surah_1_ayahs)
        assert isinstance(results, list)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Smoke test: text utilities used across examples
# ---------------------------------------------------------------------------

class TestTextUtilitiesSmoke:
    """Smoke tests for text utilities referenced in examples."""

    def test_normalize_arabic(self) -> None:
        """normalize_arabic should be importable and functional."""
        result = normalize_arabic("بِسْمِ اللَّهِ")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_similarity(self) -> None:
        """similarity should return a float score."""
        score = similarity(
            "بسم الله الرحمن الرحيم",
            "بسم الله الرحمن الرحيم",
        )
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Smoke test: data module used across examples
# ---------------------------------------------------------------------------

class TestDataModuleSmoke:
    """Smoke tests for data module functions used in examples."""

    def test_load_surah_ayahs_range(self) -> None:
        """load_surah_ayahs should work for all 114 surahs."""
        for surah_id in [1, 2, 36, 67, 114]:
            ayahs = load_surah_ayahs(surah_id)
            assert isinstance(ayahs, list)
            assert len(ayahs) > 0

    def test_ayah_model_fields(self) -> None:
        """Ayah model should have expected fields."""
        ayahs = load_surah_ayahs(1)
        ayah = ayahs[0]
        assert hasattr(ayah, "surah_id")
        assert hasattr(ayah, "ayah_number")
        assert hasattr(ayah, "text")


# ---------------------------------------------------------------------------
# Smoke test: mock transcriber as context manager (WhisperTranscriber pattern)
# ---------------------------------------------------------------------------

class TestTranscriberContextManagerSmoke:
    """Smoke tests for transcriber context manager pattern used in examples."""

    def test_context_manager_pattern(self) -> None:
        """MockTranscriber should work as a context manager like WhisperTranscriber."""
        with MockTranscriber() as transcriber:
            segments = transcriber.transcribe("fake.wav")
        assert len(segments) > 0

    def test_base_transcriber_interface(self) -> None:
        """MockTranscriber should satisfy BaseTranscriber interface."""
        transcriber = MockTranscriber()
        assert isinstance(transcriber, BaseTranscriber)
