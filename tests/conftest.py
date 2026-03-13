"""공통 pytest 픽스처 및 헬퍼"""

import subprocess
import pytest


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


FFMPEG_AVAILABLE = _ffmpeg_available()


def _make_video(path: str, duration: int) -> None:
    """FFmpeg으로 사인파 오디오 + 흑색 영상 테스트 파일을 생성합니다."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-f", "lavfi", "-i", f"color=black:size=160x120:duration={duration}:rate=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-c:a", "aac", "-shortest",
            path,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="session")
def test_video_30s(tmp_path_factory):
    """30초 테스트 영상 (FFmpeg 필요)"""
    if not FFMPEG_AVAILABLE:
        pytest.skip("FFmpeg이 설치되지 않았습니다.")
    path = str(tmp_path_factory.mktemp("videos") / "test_30s.mp4")
    _make_video(path, 30)
    return path


@pytest.fixture(scope="session")
def test_video_60s(tmp_path_factory):
    """60초 테스트 영상 (FFmpeg 필요)"""
    if not FFMPEG_AVAILABLE:
        pytest.skip("FFmpeg이 설치되지 않았습니다.")
    path = str(tmp_path_factory.mktemp("videos") / "test_60s.mp4")
    _make_video(path, 60)
    return path


@pytest.fixture
def sample_segments():
    return [
        {"start": 0.0,  "end": 2.5, "text": "Hello world"},
        {"start": 2.5,  "end": 5.0, "text": "This is a test"},
        {"start": 5.0,  "end": 8.0, "text": "Pipeline integration test"},
    ]


@pytest.fixture
def sample_translations():
    return ["안녕 세상", "이것은 테스트입니다", "파이프라인 통합 테스트"]


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return str(d)
