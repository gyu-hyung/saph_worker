"""SRT 빌더 모듈 — STT 세그먼트와 번역 텍스트로 SRT 파일을 생성합니다."""

import os

MAX_LINE_CHARS = 42  # 자막 한 줄 최대 글자 수


def _fmt_timestamp(sec: float) -> str:
    """초(float)를 SRT 타임스탬프 포맷(HH:MM:SS,mmm)으로 변환합니다."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _wrap_subtitle_text(text: str) -> str:
    """
    긴 자막 텍스트를 최대 2줄로 분할합니다.
    가능한 한 균등 분배하되, 단어 경계에서만 끊습니다.
    """
    text = text.strip()
    if len(text) <= MAX_LINE_CHARS:
        return text

    words = text.split()
    if len(words) <= 1:
        return text

    # 중간 지점에서 가장 가까운 단어 경계를 찾아 분할
    mid = len(text) // 2
    best_pos = -1
    best_dist = len(text)

    pos = 0
    for i, word in enumerate(words[:-1]):
        pos += len(word)
        if i > 0:
            pos += 1  # 공백
        dist = abs(pos - mid)
        if dist < best_dist:
            best_dist = dist
            best_pos = pos

    if best_pos > 0:
        line1 = text[:best_pos].rstrip()
        line2 = text[best_pos:].lstrip()
        return f"{line1}\n{line2}"

    return text


def build_original_srt(segments: list[dict], output_path: str) -> None:
    """원본 언어 자막 SRT 파일을 생성합니다."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_timestamp(seg['start'])} --> {_fmt_timestamp(seg['end'])}\n")
            f.write(f"{_wrap_subtitle_text(seg['text'].strip())}\n\n")


def build_translated_srt(
    segments: list[dict],
    translations: list[str],
    output_path: str,
) -> None:
    """번역 자막 SRT 파일을 생성합니다."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (seg, trans) in enumerate(zip(segments, translations), start=1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_timestamp(seg['start'])} --> {_fmt_timestamp(seg['end'])}\n")
            f.write(f"{_wrap_subtitle_text(trans.strip())}\n\n")


def build_dual_srt(
    segments: list[dict],
    translations: list[str],
    output_path: str,
) -> None:
    """원본 + 번역 듀얼 자막 SRT 파일을 생성합니다 (원본 위, 번역 아래)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (seg, trans) in enumerate(zip(segments, translations), start=1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_timestamp(seg['start'])} --> {_fmt_timestamp(seg['end'])}\n")
            f.write(f"{_wrap_subtitle_text(seg['text'].strip())}\n")
            f.write(f"{_wrap_subtitle_text(trans.strip())}\n\n")


def build_all(
    segments: list[dict],
    translations: list[str],
    results_dir: str,
    job_id: str,
) -> dict[str, str]:
    """
    원본 / 번역 / 듀얼 SRT 파일을 모두 생성합니다.

    Returns:
        {"original": path, "translated": path, "dual": path}
    """
    original_path = os.path.join(results_dir, f"{job_id}_original.srt")
    translated_path = os.path.join(results_dir, f"{job_id}_translated.srt")
    dual_path = os.path.join(results_dir, f"{job_id}_dual.srt")

    build_original_srt(segments, original_path)
    build_translated_srt(segments, translations, translated_path)
    build_dual_srt(segments, translations, dual_path)

    return {
        "original": original_path,
        "translated": translated_path,
        "dual": dual_path,
    }
