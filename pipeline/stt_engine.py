"""STT 엔진 모듈 — faster-whisper를 사용해 오디오를 텍스트 세그먼트로 변환합니다."""

import os
import re
from faster_whisper import WhisperModel

# 환경 변수로 모델 크기 선택 (GPU: large-v3, CPU: base/medium)
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "int8")

# 자막 세그먼트 제약
MAX_LINE_CHARS = 42   # 자막 한 줄 최대 글자 수
MAX_LINES = 2         # 세그먼트 당 최대 줄 수
MAX_SEGMENT_CHARS = MAX_LINE_CHARS * MAX_LINES  # 세그먼트 당 최대 글자 수 (84)
MAX_SEGMENT_DURATION = 7.0  # 세그먼트 최대 지속시간 (초)
MIN_SEGMENT_DURATION = 1.0  # 세그먼트 최소 지속시간 (초)

# 문장 종결 부호
_SENTENCE_END = re.compile(r'[.!?]$')
# 절 경계 — 쉼표 뒤 또는 접속사 앞에서 끊기
_CLAUSE_BREAK_AFTER = {',', ';', ':'}
_CLAUSE_BREAK_BEFORE = {'and', 'but', 'or', 'nor', 'yet', 'so', 'when', 'where',
                        'while', 'because', 'although', 'though', 'if', 'that',
                        'which', 'who', 'whom', 'whose', 'for', 'in', 'with'}

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """싱글톤 패턴으로 모델을 로드합니다 (Worker 프로세스당 1회)."""
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def _collect_words(raw_segments) -> list[dict]:
    """Whisper raw segment에서 모든 word를 플랫 리스트로 수집합니다."""
    words = []
    for seg in raw_segments:
        seg_words = list(seg.words) if seg.words else []
        if not seg_words:
            # word timestamp가 없으면 세그먼트 전체를 하나의 가상 word로 처리
            words.append({
                "start": seg.start,
                "end": seg.end,
                "word": seg.text.strip(),
            })
            continue
        for w in seg_words:
            words.append({
                "start": w.start,
                "end": w.end,
                "word": w.word.strip(),
            })
    return words


def _find_best_split(words: list[dict], start_idx: int) -> int:
    """
    start_idx부터 시작해서 자막 세그먼트로 적합한 끝 인덱스(exclusive)를 찾습니다.
    문장 종결 > 절 경계 > 글자수 제한 순으로 우선 분할합니다.
    """
    n = len(words)
    if start_idx >= n:
        return n

    # 누적 텍스트를 추적하며 최적 분할점 탐색
    best_sentence_end = -1      # 문장 종결 부호 위치
    best_clause_break = -1      # 절 경계 위치
    running_text = ""
    segment_start_time = words[start_idx]["start"]

    for i in range(start_idx, n):
        w = words[i]["word"]
        running_text = (running_text + " " + w).strip() if running_text else w
        duration = words[i]["end"] - segment_start_time

        # 최대 글자수 초과 → 즉시 분할
        if len(running_text) > MAX_SEGMENT_CHARS:
            # 가장 가까운 좋은 분할점 반환
            if best_sentence_end > start_idx:
                return best_sentence_end + 1
            if best_clause_break > start_idx:
                return best_clause_break + 1
            return i  # 분할점 없으면 현재 위치에서 끊기

        # 최대 지속시간 초과
        if duration > MAX_SEGMENT_DURATION:
            if best_sentence_end > start_idx:
                return best_sentence_end + 1
            if best_clause_break > start_idx:
                return best_clause_break + 1
            return i

        # 문장 종결 부호 감지
        if _SENTENCE_END.search(w):
            best_sentence_end = i

        # 절 경계 감지: 쉼표 등 뒤에서 끊기
        if w and w[-1] in _CLAUSE_BREAK_AFTER:
            best_clause_break = i

        # 접속사 앞에서 끊기 (최소 2단어 이후)
        if i > start_idx and w.lower().rstrip('.,!?;:') in _CLAUSE_BREAK_BEFORE:
            if len(running_text) >= MAX_LINE_CHARS:
                best_clause_break = i - 1

    # 전체 남은 텍스트가 제한 이내면 끝까지 반환
    return n


def _split_by_sentences(raw_segments) -> list[dict]:
    """
    문장/구 경계 기반으로 자막 세그먼트를 분할합니다.
    Maestra 스타일: 자연스러운 문장 경계에서 끊고, 세그먼트당 최대 2줄 구성.
    """
    words = _collect_words(raw_segments)
    if not words:
        return []

    results = []
    idx = 0
    while idx < len(words):
        end_idx = _find_best_split(words, idx)
        if end_idx <= idx:
            end_idx = idx + 1  # 최소 1단어씩 진행

        segment_words = words[idx:end_idx]
        text = " ".join(w["word"] for w in segment_words)

        results.append({
            "start": segment_words[0]["start"],
            "end": segment_words[-1]["end"],
            "text": text,
        })
        idx = end_idx

    return results


def _merge_short_segments(segments: list[dict]) -> list[dict]:
    """너무 짧은 세그먼트를 인접 세그먼트와 병합합니다."""
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        prev_duration = prev["end"] - prev["start"]
        combined_text = prev["text"] + " " + seg["text"]
        combined_duration = seg["end"] - prev["start"]

        # 이전 세그먼트가 짧고 합쳐도 제한 이내면 병합
        if (prev_duration < MIN_SEGMENT_DURATION
                and len(combined_text) <= MAX_SEGMENT_CHARS
                and combined_duration <= MAX_SEGMENT_DURATION):
            prev["end"] = seg["end"]
            prev["text"] = combined_text
        else:
            merged.append(seg)

    return merged


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
        vad_parameters={
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 100,           # 기본 400ms → 100ms: 타임스탬프 패딩 축소
        },
        hallucination_silence_threshold=0.5,  # 0.5초 이상 무음 뒤 환각 구간 제거
        condition_on_previous_text=True,
        initial_prompt=(
            "This is a transcript with proper punctuation, "
            "capitalization, and natural sentence structure."
        ),
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

    segments = _split_by_sentences(raw_list)
    segments = _merge_short_segments(segments)

    return segments, info.language
