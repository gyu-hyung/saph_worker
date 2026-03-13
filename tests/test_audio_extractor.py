"""오디오 추출 테스트 — FFmpeg 필요"""

import os
import tempfile

import pytest

from pipeline.audio_extractor import extract_audio


class TestExtractAudio:
    def test_returns_wav_path(self, test_video_30s):
        audio = extract_audio(test_video_30s)
        try:
            assert audio.endswith(".wav")
        finally:
            if os.path.exists(audio):
                os.unlink(audio)

    def test_output_file_exists_and_nonempty(self, test_video_30s):
        audio = extract_audio(test_video_30s)
        try:
            assert os.path.exists(audio)
            assert os.path.getsize(audio) > 0
        finally:
            if os.path.exists(audio):
                os.unlink(audio)

    def test_caller_responsibility_to_delete(self, test_video_30s):
        """extract_audio는 임시 파일을 반환하며 삭제 책임은 호출자에게 있습니다."""
        audio = extract_audio(test_video_30s)
        assert os.path.exists(audio)
        os.unlink(audio)
        assert not os.path.exists(audio)

    def test_unsupported_format_raises_value_error(self, tmp_path):
        txt = tmp_path / "file.txt"
        txt.write_text("not a video")
        with pytest.raises(ValueError, match="지원하지 않는"):
            extract_audio(str(txt))

    def test_nonexistent_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_audio("/nonexistent/path/video.mp4")

    def test_corrupt_file_raises_runtime_error(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"\x00" * 1024)
        with pytest.raises(RuntimeError):
            extract_audio(str(bad))

    def test_temp_file_cleaned_on_failure(self, tmp_path):
        """FFmpeg 실패 시 임시 WAV 파일이 삭제됩니다."""
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"\x00" * 1024)
        tmpdir = tempfile.gettempdir()
        before = {f for f in os.listdir(tmpdir) if f.endswith(".wav")}
        try:
            extract_audio(str(bad))
        except RuntimeError:
            pass
        after = {f for f in os.listdir(tmpdir) if f.endswith(".wav")}
        assert after == before, f"임시 WAV 파일이 남아있습니다: {after - before}"
