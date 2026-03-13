"""E2E 파이프라인 통합 테스트

전체 파이프라인 (오디오 추출 → STT → 번역 → SRT 생성)을 검증하고
처리 시간을 측정합니다.

STT와 번역은 목킹하여 FFmpeg만 실제로 실행합니다.
실제 Whisper + Ollama가 필요한 테스트는 @pytest.mark.integration으로 표시합니다.

실행 방법:
    pytest tests/test_pipeline_e2e.py -v -s
    pytest tests/test_pipeline_e2e.py -v -s -m integration   # 실제 모델 포함
"""

import os
import re
import time
from unittest.mock import MagicMock, patch

import pytest

from pipeline.audio_extractor import extract_audio
from pipeline.srt_builder import build_all
from pipeline.stt_engine import transcribe
from pipeline.translator import OllamaTranslator, translate_with_context

SRT_TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}")


# ─── 가짜 Whisper 객체 ────────────────────────────────────────────────────────

class _Word:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class _Seg:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text
        self.words = [_Word(w, start + i * 0.1, start + (i + 1) * 0.1)
                      for i, w in enumerate(text.split())]


class _Info:
    def __init__(self, language="en", duration=30.0):
        self.language = language
        self.duration = duration


def _fake_segments(duration: float, interval: float = 5.0):
    segs, t, i = [], 0.0, 0
    while t < duration:
        end = min(t + interval, duration)
        segs.append(_Seg(t, end, f"This is segment {i} of the test video content"))
        t, i = end, i + 1
    return segs


def _fake_translate(self, segs, ctx, lang):
    return [f"번역된 세그먼트 {i}" for i in range(len(segs))]


# ─── 파이프라인 실행 헬퍼 ─────────────────────────────────────────────────────

def run_pipeline(video_path: str, results_dir: str, job_id: str, mock_duration: float) -> dict:
    """목킹된 STT/번역으로 전체 파이프라인을 실행하고 결과 및 타이밍을 반환합니다."""
    fake_segs = _fake_segments(mock_duration)
    info = _Info(language="en", duration=mock_duration)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter(fake_segs), info)

    with (
        patch("pipeline.stt_engine._get_model", return_value=mock_model),
        patch.object(OllamaTranslator, "__init__", return_value=None),
        patch.object(OllamaTranslator, "analyze_context", return_value="SUMMARY: test"),
        patch.object(OllamaTranslator, "translate", _fake_translate),
    ):
        t0 = time.perf_counter()
        audio_path = extract_audio(video_path)
        t_audio = time.perf_counter() - t0

        try:
            progress_calls = []

            t1 = time.perf_counter()
            segments, detected_lang = transcribe(audio_path, progress_callback=progress_calls.append)
            t_stt = time.perf_counter() - t1

            t2 = time.perf_counter()
            translations = translate_with_context(segments, target_lang="ko")
            t_trans = time.perf_counter() - t2

            t3 = time.perf_counter()
            paths = build_all(segments, translations, results_dir, job_id)
            t_srt = time.perf_counter() - t3

        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    return {
        "segments": segments,
        "detected_lang": detected_lang,
        "translations": translations,
        "paths": paths,
        "progress_calls": progress_calls,
        "timing": {
            "video_duration_sec": mock_duration,
            "audio_extraction_sec": round(t_audio, 3),
            "stt_sec": round(t_stt, 3),
            "translation_sec": round(t_trans, 3),
            "srt_build_sec": round(t_srt, 3),
            "total_sec": round(t_audio + t_stt + t_trans + t_srt, 3),
        },
    }


# ─── E2E 파이프라인 검증 ──────────────────────────────────────────────────────

class TestPipelineE2E:
    def test_30s_produces_three_srt_files(self, test_video_30s, results_dir):
        result = run_pipeline(test_video_30s, results_dir, "e2e-30s-files", 30.0)
        paths = result["paths"]
        assert set(paths.keys()) == {"original", "translated", "dual"}
        for key, path in paths.items():
            assert os.path.exists(path), f"{key} SRT 파일 없음"
            assert os.path.getsize(path) > 0, f"{key} SRT 파일 비어있음"

    def test_60s_produces_three_srt_files(self, test_video_60s, results_dir):
        result = run_pipeline(test_video_60s, results_dir, "e2e-60s-files", 60.0)
        for key, path in result["paths"].items():
            assert os.path.exists(path), f"{key} SRT 파일 없음"

    def test_srt_valid_timestamp_format(self, test_video_30s, results_dir):
        result = run_pipeline(test_video_30s, results_dir, "e2e-timestamp", 30.0)
        for key, path in result["paths"].items():
            content = open(path, encoding="utf-8").read()
            timestamps = SRT_TIMESTAMP_RE.findall(content)
            assert len(timestamps) > 0, f"{key} SRT에 유효한 타임스탬프가 없습니다"

    def test_segment_translation_count_match(self, test_video_30s, results_dir):
        result = run_pipeline(test_video_30s, results_dir, "e2e-count", 30.0)
        assert len(result["segments"]) == len(result["translations"])

    def test_progress_callback_invoked(self, test_video_30s, results_dir):
        """[버그 수정 E2E 검증] STT progress_callback이 전체 파이프라인에서 호출됩니다."""
        result = run_pipeline(test_video_30s, results_dir, "e2e-progress", 30.0)
        assert len(result["progress_calls"]) > 0, "STT progress_callback이 한 번도 호출되지 않았습니다"
        assert result["progress_calls"][-1] == 100

    def test_dual_srt_contains_both_languages(self, test_video_30s, results_dir):
        result = run_pipeline(test_video_30s, results_dir, "e2e-dual-content", 30.0)
        content = open(result["paths"]["dual"], encoding="utf-8").read()
        assert "segment" in content.lower()   # 원본(영문)
        assert "번역된" in content             # 번역(한글)

    def test_original_srt_no_korean(self, test_video_30s, results_dir):
        result = run_pipeline(test_video_30s, results_dir, "e2e-lang-sep", 30.0)
        content = open(result["paths"]["original"], encoding="utf-8").read()
        assert "번역된" not in content

    def test_translated_srt_no_english_segments(self, test_video_30s, results_dir):
        result = run_pipeline(test_video_30s, results_dir, "e2e-trans-lang", 30.0)
        content = open(result["paths"]["translated"], encoding="utf-8").read()
        assert "This is segment" not in content

    def test_srt_sequential_index(self, test_video_30s, results_dir):
        """SRT 파일의 인덱스가 1부터 순차적으로 증가해야 합니다."""
        result = run_pipeline(test_video_30s, results_dir, "e2e-index", 30.0)
        content = open(result["paths"]["original"], encoding="utf-8").read()
        blocks = [b.strip() for b in content.strip().split("\n\n") if b.strip()]
        for i, block in enumerate(blocks, start=1):
            first_line = block.split("\n")[0]
            assert first_line == str(i), f"블록 인덱스 오류: {first_line!r} ≠ {i}"


# ─── 처리 시간 벤치마크 ───────────────────────────────────────────────────────

class TestPipelineTiming:
    """오디오 추출 실시간 성능을 측정합니다. STT/번역은 목킹."""

    def _print_timing(self, capsys, label: str, timing: dict):
        with capsys.disabled():
            print(f"\n{'─' * 50}")
            print(f"[처리 시간 벤치마크] {label}")
            print(f"  영상 길이:     {timing['video_duration_sec']}초")
            print(f"  오디오 추출:   {timing['audio_extraction_sec']}s")
            print(f"  STT (목킹):    {timing['stt_sec']}s")
            print(f"  번역 (목킹):   {timing['translation_sec']}s")
            print(f"  SRT 생성:      {timing['srt_build_sec']}s")
            print(f"  총 소요 시간:  {timing['total_sec']}s")
            print(f"{'─' * 50}")

    def test_timing_30s_video(self, test_video_30s, results_dir, capsys):
        result = run_pipeline(test_video_30s, results_dir, "timing-30s", 30.0)
        self._print_timing(capsys, "30초 영상", result["timing"])
        # FFmpeg 오디오 추출은 5초 미만이어야 함
        assert result["timing"]["audio_extraction_sec"] < 5.0, \
            f"오디오 추출이 너무 느립니다: {result['timing']['audio_extraction_sec']}s"
        # SRT 빌드는 0.5초 미만이어야 함 (순수 파일 I/O)
        assert result["timing"]["srt_build_sec"] < 0.5

    def test_timing_60s_video(self, test_video_60s, results_dir, capsys):
        result = run_pipeline(test_video_60s, results_dir, "timing-60s", 60.0)
        self._print_timing(capsys, "60초 영상", result["timing"])
        assert result["timing"]["audio_extraction_sec"] < 10.0, \
            f"오디오 추출이 너무 느립니다: {result['timing']['audio_extraction_sec']}s"


# ─── 실제 모델 통합 테스트 (선택) ────────────────────────────────────────────

@pytest.mark.integration
class TestPipelineWithRealModels:
    """실제 Whisper 모델과 Ollama를 사용하는 통합 테스트.

    실행하려면:
        pytest tests/test_pipeline_e2e.py -m integration -v -s
    """

    def test_real_stt_30s(self, test_video_30s, results_dir, capsys):
        """실제 Whisper 모델로 30초 영상을 STT합니다."""
        audio_path = extract_audio(test_video_30s)
        try:
            t0 = time.perf_counter()
            progress_calls = []
            segments, lang = transcribe(audio_path, progress_callback=progress_calls.append)
            elapsed = time.perf_counter() - t0

            with capsys.disabled():
                print(f"\n[실제 STT - 30초 영상]")
                print(f"  감지 언어: {lang}")
                print(f"  세그먼트 수: {len(segments)}")
                print(f"  진행률 콜백 호출 횟수: {len(progress_calls)}")
                print(f"  처리 시간: {elapsed:.2f}s")

            assert len(progress_calls) > 0, "STT progress_callback이 호출되지 않았습니다"
            assert progress_calls[-1] == 100
            assert isinstance(segments, list)
            assert lang is not None
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
