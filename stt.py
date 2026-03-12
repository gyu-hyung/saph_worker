import sys
import os
import subprocess
from faster_whisper import WhisperModel

# ── 설정 ──────────────────────────────────────────────
MODEL_SIZE      = "small"    # tiny | base | small | medium | large-v3
DEVICE          = "cpu"      # mac 은 cpu (Apple Silicon 은 추후 coreml 지원)
COMPUTE         = "int8"     # cpu 최적화

# 세그먼트 분할 옵션
MAX_WORDS       = 10         # 세그먼트 당 최대 단어 수
MAX_CHARS       = 60         # 세그먼트 당 최대 글자 수
VAD_FILTER      = True       # 무음 구간 기준 자동 분리
VAD_MIN_SILENCE = 300        # 무음으로 간주할 최소 ms
# ──────────────────────────────────────────────────────


def extract_audio(video_path: str) -> str:
    """FFmpeg 로 영상에서 오디오(wav 16kHz mono) 추출"""
    audio_path = video_path.rsplit(".", 1)[0] + "_audio.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        audio_path,
    ]
    print(f"[1/3] 오디오 추출 중: {video_path} → {audio_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FFmpeg 에러:", result.stderr)
        sys.exit(1)
    print(f"      완료: {audio_path}")
    return audio_path


def split_by_words(segments) -> list[dict]:
    """word_timestamps 기반으로 MAX_WORDS / MAX_CHARS 단위로 세그먼트 재분할"""
    results = []
    for seg in segments:
        words = list(seg.words) if seg.words else []
        if not words:
            results.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            continue

        chunk_words  = []
        chunk_start  = words[0].start

        for w in words:
            chunk_words.append(w)
            current_text  = " ".join(x.word.strip() for x in chunk_words)
            word_count    = len(chunk_words)
            char_count    = len(current_text)

            if word_count >= MAX_WORDS or char_count >= MAX_CHARS:
                results.append({
                    "start": chunk_start,
                    "end":   w.end,
                    "text":  current_text,
                })
                chunk_words = []
                # 다음 단어가 있으면 그 단어부터 시작
                next_idx = words.index(w) + 1
                if next_idx < len(words):
                    chunk_start = words[next_idx].start

        # 남은 단어 처리
        if chunk_words:
            results.append({
                "start": chunk_start,
                "end":   chunk_words[-1].end,
                "text":  " ".join(x.word.strip() for x in chunk_words),
            })

    return results


def transcribe(audio_path: str) -> list[dict]:
    """faster-whisper 로 STT 수행, 세그먼트 리스트 반환"""
    print(f"\n[2/3] STT 시작 (model={MODEL_SIZE}, device={DEVICE})")
    print(f"      옵션: vad_filter={VAD_FILTER}, max_words={MAX_WORDS}, max_chars={MAX_CHARS}")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE)
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=VAD_FILTER,
        vad_parameters={"min_silence_duration_ms": VAD_MIN_SILENCE},
    )
    print(f"      감지 언어: {info.language} (확률 {info.language_probability:.0%})")

    # 단어 단위로 세그먼트 재분할
    raw = list(segments)
    results = split_by_words(raw)

    for seg in results:
        print(f"  [{seg['start']:6.2f}s → {seg['end']:6.2f}s]  {seg['text']}")

    return results


def build_srt(segments: list[dict], output_path: str) -> None:
    """세그먼트 리스트를 SRT 파일로 저장"""

    def fmt(sec: float) -> str:
        h  = int(sec // 3600)
        m  = int((sec % 3600) // 60)
        s  = int(sec % 60)
        ms = int(round((sec - int(sec)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    print(f"\n[3/3] SRT 생성 중: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{fmt(seg['start'])} --> {fmt(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")
    print(f"      완료: {output_path}")


def unique_path(base: str, ext: str) -> str:
    """base.ext 가 이미 존재하면 base_1.ext, base_2.ext … 형태로 새 경로 반환"""
    candidate = f"{base}{ext}"
    if not os.path.exists(candidate):
        return candidate
    counter = 1
    while True:
        candidate = f"{base}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def main():
    if len(sys.argv) < 2:
        print("사용법: python stt.py <영상파일경로>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"파일을 찾을 수 없습니다: {video_path}")
        sys.exit(1)

    audio_path  = extract_audio(video_path)
    segments    = transcribe(audio_path)
    srt_base    = video_path.rsplit(".", 1)[0]
    srt_path    = unique_path(srt_base, ".srt")
    build_srt(segments, srt_path)

    # 임시 오디오 파일 삭제
    os.remove(audio_path)
    print(f"\n✅ 완료! SRT 파일: {srt_path}")


if __name__ == "__main__":
    main()

