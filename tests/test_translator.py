"""번역 엔진 유닛 테스트 — LLM API를 목킹하여 외부 의존성 없이 실행"""

import inspect
from unittest.mock import patch

import pytest

from pipeline.translator import (
    OllamaTranslator,
    OpenAITranslator,
    TranslatorProvider,
    get_translator,
    translate_with_context,
)


# ─── TranslatorProvider 추상 클래스 ───────────────────────────────────────────

class TestTranslatorProvider:
    def test_is_abstract(self):
        assert inspect.isabstract(TranslatorProvider)

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            TranslatorProvider()


# ─── get_translator 팩토리 ────────────────────────────────────────────────────

class TestGetTranslator:
    def test_default_returns_ollama(self, monkeypatch):
        monkeypatch.delenv("TRANSLATOR_PROVIDER", raising=False)
        with patch.object(OllamaTranslator, "__init__", return_value=None):
            t = get_translator()
        assert isinstance(t, OllamaTranslator)

    def test_openai_provider(self, monkeypatch):
        monkeypatch.setenv("TRANSLATOR_PROVIDER", "openai")
        with patch.object(OpenAITranslator, "__init__", return_value=None):
            t = get_translator()
        assert isinstance(t, OpenAITranslator)

    def test_unknown_falls_back_to_ollama(self, monkeypatch):
        monkeypatch.setenv("TRANSLATOR_PROVIDER", "unknown_value")
        with patch.object(OllamaTranslator, "__init__", return_value=None):
            t = get_translator()
        assert isinstance(t, OllamaTranslator)


# ─── OllamaTranslator.translate ───────────────────────────────────────────────

class TestOllamaTranslatorTranslate:
    @pytest.fixture
    def translator(self, monkeypatch):
        monkeypatch.setattr(OllamaTranslator, "__init__", lambda self: None)
        t = OllamaTranslator()
        return t

    def _set_chat_response(self, monkeypatch, response: str):
        monkeypatch.setattr(OllamaTranslator, "_chat", lambda self, s, u: response)

    def test_returns_same_length_as_segments(self, translator, monkeypatch, sample_segments):
        self._set_chat_response(monkeypatch, "\n".join(f"번역{i}" for i in range(len(sample_segments))))
        result = translator.translate(sample_segments, "ctx", "ko")
        assert len(result) == len(sample_segments)

    def test_pads_when_response_too_short(self, translator, monkeypatch, sample_segments):
        """번역 결과가 세그먼트보다 적으면 빈 문자열로 패딩합니다."""
        self._set_chat_response(monkeypatch, "한 줄만")
        result = translator.translate(sample_segments, "", "ko")
        assert len(result) == len(sample_segments)
        assert result[0] == "한 줄만"
        assert result[1] == ""
        assert result[2] == ""

    def test_strips_numbering_prefix(self, translator, monkeypatch, sample_segments):
        """'1. 텍스트' 형식의 번호가 제거되어야 합니다."""
        numbered = "\n".join(f"{i + 1}. 번역결과{i + 1}" for i in range(len(sample_segments)))
        self._set_chat_response(monkeypatch, numbered)
        result = translator.translate(sample_segments, "", "ko")
        for r in result:
            assert not r.startswith(tuple("123456789")), f"번호가 제거되지 않음: {r!r}"

    def test_truncates_when_response_too_long(self, translator, monkeypatch, sample_segments):
        """번역 결과가 세그먼트보다 많으면 세그먼트 수로 자릅니다."""
        extra = "\n".join(f"번역{i}" for i in range(len(sample_segments) + 5))
        self._set_chat_response(monkeypatch, extra)
        result = translator.translate(sample_segments, "", "ko")
        assert len(result) == len(sample_segments)


# ─── translate_with_context ───────────────────────────────────────────────────

class TestTranslateWithContext:
    def _setup_mocks(self, monkeypatch, translations=None):
        monkeypatch.setattr(OllamaTranslator, "__init__", lambda self: None)
        monkeypatch.setattr(OllamaTranslator, "analyze_context", lambda self, t: "SUMMARY: test\nKEYWORDS: test")
        if translations is None:
            monkeypatch.setattr(OllamaTranslator, "translate", lambda self, s, c, l: [f"번역{i}" for i in range(len(s))])
        else:
            monkeypatch.setattr(OllamaTranslator, "translate", lambda self, s, c, l: translations[:len(s)])
        monkeypatch.setenv("TRANSLATOR_PROVIDER", "ollama")

    def test_returns_same_length(self, monkeypatch, sample_segments):
        self._setup_mocks(monkeypatch)
        result = translate_with_context(sample_segments, target_lang="ko")
        assert len(result) == len(sample_segments)

    def test_progress_callback_is_called(self, monkeypatch, sample_segments):
        self._setup_mocks(monkeypatch)
        called = []
        translate_with_context(sample_segments, target_lang="ko", progress_callback=called.append)
        assert len(called) > 0

    def test_progress_values_in_range(self, monkeypatch, sample_segments):
        self._setup_mocks(monkeypatch)
        values = []
        translate_with_context(sample_segments, target_lang="ko", progress_callback=values.append)
        for v in values:
            assert 0 <= v <= 100

    def test_chunk_split_for_long_video(self, monkeypatch):
        """300초 초과 영상은 청크 단위로 나눠 번역합니다."""
        # 600초 = 2청크
        long_segments = [
            {"start": i * 10.0, "end": (i + 1) * 10.0, "text": f"seg{i}"}
            for i in range(60)
        ]
        translate_calls = []

        def mock_translate(self, segs, ctx, lang):
            translate_calls.append(len(segs))
            return [f"번역{i}" for i in range(len(segs))]

        monkeypatch.setattr(OllamaTranslator, "__init__", lambda self: None)
        monkeypatch.setattr(OllamaTranslator, "analyze_context", lambda self, t: "ctx")
        monkeypatch.setattr(OllamaTranslator, "translate", mock_translate)
        monkeypatch.setenv("TRANSLATOR_PROVIDER", "ollama")

        result = translate_with_context(long_segments, target_lang="ko")

        assert len(translate_calls) == 2, f"600초 영상은 2회 번역 호출이어야 합니다. 실제: {len(translate_calls)}"
        assert len(result) == len(long_segments)

    def test_short_video_single_translate_call(self, monkeypatch, sample_segments):
        """300초 이하 영상은 translate를 1회만 호출합니다."""
        calls = []

        def mock_translate(self, segs, ctx, lang):
            calls.append(1)
            return [f"번역{i}" for i in range(len(segs))]

        monkeypatch.setattr(OllamaTranslator, "__init__", lambda self: None)
        monkeypatch.setattr(OllamaTranslator, "analyze_context", lambda self, t: "ctx")
        monkeypatch.setattr(OllamaTranslator, "translate", mock_translate)
        monkeypatch.setenv("TRANSLATOR_PROVIDER", "ollama")

        translate_with_context(sample_segments, target_lang="ko")
        assert len(calls) == 1
