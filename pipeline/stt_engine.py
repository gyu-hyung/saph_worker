"""STT 엔진 모듈 — faster-whisper를 사용해 오디오를 텍스트 세그먼트로 변환합니다."""

import os
from typing import Generator
from faster_whisper import WhisperModel

# 환경 변수로 모델 크기 선택 (GPU: large-v3, CPU: base/medium)
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "int8")

MAX_WORDS = 10    # 세그먼트 당 최대 단어 수
MAX_CHARS = 60    # 세그먼트 당 최대 글자 수

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """싱글톤 패턴으로 모델을 로드합니다 (Worker 프로세스당 1회)."""
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def _split_by_words(segments) -> list[dict]:
    """word_timestamps 기반으로 MAX_WORDS / MAX_CHARS 단위로 세그먼트를 재분할합니다."""
    results = []
    for seg in segments:
        words = list(seg.words) if seg.words else []
        if not words:
            results.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            continue

        chunk_words = []
        chunk_start = words[0].start

        for i, w in enumerate(words):
            chunk_words.append(w)
            current_text = " ".join(x.word.strip() for x in chunk_words)

            if len(chunk_words) >= MAX_WORDS or len(current_text) >= MAX_CHARS:
                results.append({
                    "start": chunk_start,
                    "end": w.end,
                    "text": current_text,
                })
                chunk_words = []
                if i + 1 < len(words):
                    chunk_start = words[i + 1].start

        if chunk_words:
            results.append({
                "start": chunk_start,
                "end": chunk_words[-1].end,
                "text": " ".join(x.word.strip() for x in chunk_words),
            })

    return results


def transcribe(
    audio_path: str,
    language: str = "auto",
    progress_callback=None,
) -> tuple[list[dict], str]:
    """
    오디오 파일을 STT로 변환합니다.

    Args:
        audio_path: WAV 파일 경로
        language: 원본 언어 코드 (기본값 "auto" → 자동 감지)
        progress_callback: (percent: int) → None, 진행률 콜백

    Returns:
        (segments, detected_language)
        segments: [{"start": float, "end": float, "text": str}, ...]
        detected_language: 감지된 언어 코드 (e.g. "en", "ko")
    """
    model = _get_model()

    detect_lang = None if language == "auto" else language

    raw_segments, info = model.transcribe(
        audio_path,
        language=detect_lang,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )

    total_duration = info.duration or 0.0
    raw_list = []
    for seg in raw_segments:
        raw_list.append(seg)
        if progress_callback and total_duration > 0:
            pct = int(min(seg.end / total_duration * 100, 99))
            progress_callback(pct)

    if progress_callback:
        progress_callback(100)

    segments = _split_by_words(raw_list)

    return segments, info.language
