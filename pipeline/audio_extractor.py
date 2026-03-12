"""오디오 추출 모듈 — FFmpeg를 사용해 영상에서 16kHz mono WAV를 추출합니다."""

import os
import subprocess
import tempfile


def extract_audio(video_path: str) -> str:
    """
    영상 파일에서 오디오를 WAV (16kHz, mono)로 추출합니다.

    Args:
        video_path: 원본 영상 파일 경로 (mp4, mov 등)

    Returns:
        추출된 WAV 파일 경로 (임시 파일, 처리 후 직접 삭제 필요)

    Raises:
        ValueError: 지원하지 않는 확장자
        RuntimeError: FFmpeg 실행 실패 또는 오디오 트랙 없음
    """
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        raise ValueError(f"지원하지 않는 영상 포맷: {ext}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    tmp_fd, audio_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",               # 비디오 스트림 제거
        "-ar", "16000",      # 샘플레이트 16kHz (Whisper 권장)
        "-ac", "1",          # Mono
        "-f", "wav",
        audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        os.unlink(audio_path)
        stderr = result.stderr
        if "no audio" in stderr.lower() or "audio stream" in stderr.lower():
            raise RuntimeError("영상에 오디오 트랙이 없습니다.")
        raise RuntimeError(f"FFmpeg 오류:\n{stderr}")

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError("오디오 추출 결과 파일이 비어 있습니다.")

    return audio_path
