"""STT 엔진 유닛 테스트 — Whisper 모델을 목킹하여 외부 의존성 없이 실행"""

from unittest.mock import MagicMock, patch

import pytest

from pipeline.stt_engine import _split_by_words, transcribe


# ─── 테스트용 가짜 객체 ────────────────────────────────────────────────────────

class FakeWord:
    def __init__(self, word: str, start: float, end: float):
        self.word = word
        self.start = start
        self.end = end


class FakeSegment:
    def __init__(self, start: float, end: float, text: str, words=None):
        self.start = start
        self.end = end
        self.text = text
        self.words = words or []


class FakeInfo:
    def __init__(self, language: str = "en", duration: float = 10.0):
        self.language = language
        self.duration = duration


# ─── _split_by_words 테스트 ───────────────────────────────────────────────────

class TestSplitByWords:
    def test_no_words_uses_segment_directly(self):
        seg = FakeSegment(0.0, 2.0, "hello world", words=[])
        result = _split_by_words([seg])
        assert len(result) == 1
        assert result[0] == {"start": 0.0, "end": 2.0, "text": "hello world"}

    def test_splits_at_max_words(self):
        # MAX_WORDS=10 → 11개 단어는 2청크로 분할
        words = [FakeWord(f"word{i}", i * 0.5, (i + 1) * 0.5) for i in range(11)]
        seg = FakeSegment(0.0, 5.5, "...", words=words)
        result = _split_by_words([seg])
        assert len(result) == 2
        assert "word0" in result[0]["text"]
        assert "word10" in result[1]["text"]

    def test_splits_at_max_chars(self):
        # MAX_CHARS=60 → 61자 단어는 첫 청크에서 즉시 분할
        long_word = "a" * 61
        words = [FakeWord(long_word, 0.0, 1.0), FakeWord("short", 1.0, 2.0)]
        seg = FakeSegment(0.0, 2.0, "...", words=words)
        result = _split_by_words([seg])
        assert len(result) == 2

    def test_multiple_segments_preserved(self):
        segs = [
            FakeSegment(0.0, 1.0, "seg1", [FakeWord("seg1", 0.0, 1.0)]),
            FakeSegment(1.0, 2.0, "seg2", [FakeWord("seg2", 1.0, 2.0)]),
        ]
        result = _split_by_words(segs)
        assert len(result) == 2

    def test_result_has_required_keys(self):
        seg = FakeSegment(0.0, 1.0, "hi", [FakeWord("hi", 0.0, 1.0)])
        result = _split_by_words([seg])
        assert {"start", "end", "text"} <= set(result[0].keys())

    def test_empty_input(self):
        assert _split_by_words([]) == []


# ─── transcribe progress_callback 테스트 ─────────────────────────────────────

class TestTranscribeProgressCallback:
    def test_callback_is_called(self):
        """[버그 수정 검증] progress_callback이 실제로 호출되어야 합니다."""
        words = [FakeWord("hello", 0.0, 1.0), FakeWord("world", 1.0, 2.0)]
        fake_segs = [FakeSegment(0.0, 2.0, "hello world", words)]
        info = FakeInfo(language="en", duration=10.0)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(fake_segs), info)

        called_with = []
        with patch("pipeline.stt_engine._get_model", return_value=mock_model):
            transcribe("dummy.wav", progress_callback=called_with.append)

        assert len(called_with) > 0, "progress_callback이 한 번도 호출되지 않았습니다"
        assert called_with[-1] == 100, f"마지막 콜백 값은 100이어야 합니다. 실제: {called_with[-1]}"

    def test_callback_values_in_range(self):
        """콜백으로 전달되는 모든 값이 0~100 범위여야 합니다."""
        words = [FakeWord(f"w{i}", float(i), float(i + 1)) for i in range(5)]
        fake_segs = [FakeSegment(float(i), float(i + 1), f"w{i}", [words[i]]) for i in range(5)]
        info = FakeInfo(duration=5.0)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(fake_segs), info)

        values = []
        with patch("pipeline.stt_engine._get_model", return_value=mock_model):
            transcribe("dummy.wav", progress_callback=values.append)

        for v in values:
            assert 0 <= v <= 100, f"진행률 {v}이 유효 범위(0~100) 밖입니다"

    def test_no_callback_does_not_raise(self):
        """progress_callback=None이어도 오류 없이 동작해야 합니다."""
        fake_segs = [FakeSegment(0.0, 1.0, "hi", [FakeWord("hi", 0.0, 1.0)])]
        info = FakeInfo()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(fake_segs), info)

        with patch("pipeline.stt_engine._get_model", return_value=mock_model):
            segments, lang = transcribe("dummy.wav")

        assert len(segments) > 0

    def test_zero_duration_no_div_zero(self):
        """duration=0일 때 ZeroDivisionError 없이 동작해야 합니다."""
        fake_segs = [FakeSegment(0.0, 1.0, "hi", [FakeWord("hi", 0.0, 1.0)])]
        info = FakeInfo(duration=0.0)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(fake_segs), info)

        values = []
        with patch("pipeline.stt_engine._get_model", return_value=mock_model):
            transcribe("dummy.wav", progress_callback=values.append)

        # duration=0이면 중간 콜백 없이 최종 100만 호출
        assert values == [100]

    def test_returns_segments_and_language(self):
        fake_segs = [FakeSegment(0.0, 2.0, "test", [FakeWord("test", 0.0, 2.0)])]
        info = FakeInfo(language="ja", duration=2.0)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(fake_segs), info)

        with patch("pipeline.stt_engine._get_model", return_value=mock_model):
            segments, lang = transcribe("dummy.wav")

        assert isinstance(segments, list)
        assert lang == "ja"

    def test_progress_monotonically_increases(self):
        """진행률은 단조 증가해야 합니다 (세그먼트가 시간순으로 처리되므로)."""
        fake_segs = [FakeSegment(float(i * 2), float((i + 1) * 2), f"seg{i}", [FakeWord(f"w{i}", float(i * 2), float((i + 1) * 2))]) for i in range(5)]
        info = FakeInfo(duration=10.0)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(fake_segs), info)

        values = []
        with patch("pipeline.stt_engine._get_model", return_value=mock_model):
            transcribe("dummy.wav", progress_callback=values.append)

        intermediate = values[:-1]  # 마지막 100 제외
        for a, b in zip(intermediate, intermediate[1:]):
            assert a <= b, f"진행률이 감소했습니다: {a} → {b}"
