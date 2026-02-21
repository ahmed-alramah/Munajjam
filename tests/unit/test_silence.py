"""
Unit tests for silence detection.
"""

import pytest
from unittest.mock import patch, MagicMock
from munajjam.transcription.silence import detect_silences, detect_non_silent_chunks


class TestDetectSilences:
    """Test silence detection functions."""

    def test_detect_silences_file_not_found(self):
        """Test silence detection with non-existent file."""
        with pytest.raises(Exception):
            detect_silences("nonexistent_file.wav")

    def test_silence_tuple_format(self, sample_silences):
        """Test that silences are in (start_ms, end_ms) format."""
        for start, end in sample_silences:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start < end


class TestAdaptiveSilenceDetection:
    """Tests for adaptive silence detection retry logic."""

    def _make_chunks(self, n: int) -> list[tuple[int, int]]:
        """Helper: create n dummy non-silent chunks."""
        return [(i * 1000, i * 1000 + 500) for i in range(n)]

    def test_adaptive_disabled_by_default(self):
        """Non-adaptive call must not trigger retries regardless of chunk count."""
        few_chunks = self._make_chunks(1)

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=few_chunks,
        ) as mock_raw:
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=False,
                expected_chunks=10,
            )

        # Only one call – no retries
        assert mock_raw.call_count == 1
        assert result == few_chunks

    def test_adaptive_no_retry_when_enough_chunks(self):
        """When chunk count already meets the threshold, no retry should occur."""
        enough_chunks = self._make_chunks(8)  # 8 >= 0.5 * 10 = 5

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=enough_chunks,
        ) as mock_raw:
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=True,
                expected_chunks=10,
                min_chunks_ratio=0.5,
            )

        assert mock_raw.call_count == 1
        assert result == enough_chunks

    def test_adaptive_retries_when_too_few_chunks(self):
        """When initial detection finds too few chunks, retries with relaxed thresholds."""
        few_chunks = self._make_chunks(2)   # 2 < 0.5 * 10 = 5  → retry
        enough_chunks = self._make_chunks(6)  # 6 >= 5            → stop

        call_results = [few_chunks, enough_chunks]
        call_index = {"i": 0}

        def side_effect(*args, **kwargs):
            result = call_results[min(call_index["i"], len(call_results) - 1)]
            call_index["i"] += 1
            return result

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            side_effect=side_effect,
        ) as mock_raw:
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=True,
                expected_chunks=10,
                min_chunks_ratio=0.5,
            )

        # First call (initial) + at least one retry
        assert mock_raw.call_count >= 2
        assert result == enough_chunks

    def test_adaptive_retry_uses_relaxed_thresholds(self):
        """Retry calls must use a lower (more negative) silence_thresh."""
        few_chunks = self._make_chunks(1)

        recorded_thresholds: list[int] = []

        def side_effect(audio_path, min_silence_len, silence_thresh, use_fast):
            recorded_thresholds.append(silence_thresh)
            return few_chunks  # Always return few chunks to exhaust all retries

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            side_effect=side_effect,
        ):
            detect_non_silent_chunks(
                "dummy.wav",
                silence_thresh=-30,
                adaptive=True,
                expected_chunks=10,
                min_chunks_ratio=0.5,
            )

        # First threshold is the original; subsequent thresholds must be lower
        assert recorded_thresholds[0] == -30
        for i in range(1, len(recorded_thresholds)):
            assert recorded_thresholds[i] < recorded_thresholds[i - 1], (
                f"Retry {i} threshold {recorded_thresholds[i]} is not lower than "
                f"previous {recorded_thresholds[i - 1]}"
            )

    def test_adaptive_retry_uses_shorter_min_silence_len(self):
        """Retry calls must use a shorter min_silence_len."""
        few_chunks = self._make_chunks(1)

        recorded_lens: list[int] = []

        def side_effect(audio_path, min_silence_len, silence_thresh, use_fast):
            recorded_lens.append(min_silence_len)
            return few_chunks

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            side_effect=side_effect,
        ):
            detect_non_silent_chunks(
                "dummy.wav",
                min_silence_len=300,
                adaptive=True,
                expected_chunks=10,
                min_chunks_ratio=0.5,
            )

        assert recorded_lens[0] == 300
        for i in range(1, len(recorded_lens)):
            assert recorded_lens[i] < recorded_lens[i - 1], (
                f"Retry {i} min_silence_len {recorded_lens[i]} is not shorter than "
                f"previous {recorded_lens[i - 1]}"
            )

    def test_adaptive_min_silence_len_never_below_50ms(self):
        """min_silence_len must never drop below 50 ms during retries."""
        few_chunks = self._make_chunks(1)

        recorded_lens: list[int] = []

        def side_effect(audio_path, min_silence_len, silence_thresh, use_fast):
            recorded_lens.append(min_silence_len)
            return few_chunks

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            side_effect=side_effect,
        ):
            detect_non_silent_chunks(
                "dummy.wav",
                min_silence_len=100,  # Very short initial value
                adaptive=True,
                expected_chunks=10,
                min_chunks_ratio=0.5,
            )

        for length in recorded_lens:
            assert length >= 50, f"min_silence_len {length} dropped below 50 ms"

    def test_adaptive_exhausts_all_retries_gracefully(self):
        """When all retries fail to find enough chunks, return the last result."""
        few_chunks = self._make_chunks(1)

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=few_chunks,
        ) as mock_raw:
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=True,
                expected_chunks=100,
                min_chunks_ratio=0.9,  # Very strict: need 90 chunks
            )

        # Should have tried initial + 4 retry levels = 5 total calls
        assert mock_raw.call_count == 5
        # Returns whatever was found (graceful degradation)
        assert result == few_chunks

    def test_adaptive_ignored_when_expected_chunks_none(self):
        """When expected_chunks is None, adaptive mode is effectively disabled."""
        few_chunks = self._make_chunks(1)

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=few_chunks,
        ) as mock_raw:
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=True,
                expected_chunks=None,
            )

        assert mock_raw.call_count == 1
        assert result == few_chunks

    def test_adaptive_ignored_when_expected_chunks_zero(self):
        """When expected_chunks is 0, adaptive mode is effectively disabled."""
        few_chunks = self._make_chunks(1)

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=few_chunks,
        ) as mock_raw:
            result = detect_non_silent_chunks(
                "dummy.wav",
                adaptive=True,
                expected_chunks=0,
            )

        assert mock_raw.call_count == 1
        assert result == few_chunks

    def test_adaptive_backwards_compatible_signature(self):
        """Calling detect_non_silent_chunks without new args must behave as before."""
        chunks = self._make_chunks(5)

        with patch(
            "munajjam.transcription.silence._detect_non_silent_chunks_raw",
            return_value=chunks,
        ) as mock_raw:
            result = detect_non_silent_chunks("dummy.wav")

        assert mock_raw.call_count == 1
        assert result == chunks
