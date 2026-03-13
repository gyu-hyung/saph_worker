"""SRT 빌더 유닛 테스트 — 외부 의존성 없음"""

import os
import re

import pytest

from pipeline.srt_builder import (
    _fmt_timestamp,
    build_all,
    build_dual_srt,
    build_original_srt,
    build_translated_srt,
)

SRT_TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}")


class TestFmtTimestamp:
    def test_zero(self):
        assert _fmt_timestamp(0.0) == "00:00:00,000"

    def test_one_second(self):
        assert _fmt_timestamp(1.0) == "00:00:01,000"

    def test_one_minute(self):
        assert _fmt_timestamp(60.0) == "00:01:00,000"

    def test_one_hour(self):
        assert _fmt_timestamp(3600.0) == "01:00:00,000"

    def test_milliseconds(self):
        assert _fmt_timestamp(1.5) == "00:00:01,500"

    def test_complex(self):
        assert _fmt_timestamp(3723.456) == "01:02:03,456"

    def test_999ms(self):
        assert _fmt_timestamp(0.999) == "00:00:00,999"


class TestBuildOriginalSrt:
    def test_creates_file(self, sample_segments, tmp_path):
        out = str(tmp_path / "original.srt")
        build_original_srt(sample_segments, out)
        assert os.path.exists(out)

    def test_srt_block_format(self, sample_segments, tmp_path):
        out = str(tmp_path / "original.srt")
        build_original_srt(sample_segments, out)
        content = open(out, encoding="utf-8").read()
        blocks = [b.strip() for b in content.strip().split("\n\n") if b.strip()]
        assert len(blocks) == len(sample_segments)
        for i, block in enumerate(blocks, start=1):
            lines = block.split("\n")
            assert lines[0] == str(i)
            assert SRT_TIMESTAMP_RE.match(lines[1]), f"블록 {i}: 타임스탬프 형식 오류"
            assert lines[2] == sample_segments[i - 1]["text"]

    def test_all_texts_present(self, sample_segments, tmp_path):
        out = str(tmp_path / "original.srt")
        build_original_srt(sample_segments, out)
        content = open(out, encoding="utf-8").read()
        for seg in sample_segments:
            assert seg["text"] in content

    def test_empty_segments(self, tmp_path):
        out = str(tmp_path / "empty.srt")
        build_original_srt([], out)
        assert open(out, encoding="utf-8").read().strip() == ""

    def test_creates_parent_dir(self, tmp_path):
        out = str(tmp_path / "nested" / "deep" / "original.srt")
        build_original_srt([{"start": 0.0, "end": 1.0, "text": "hi"}], out)
        assert os.path.exists(out)


class TestBuildTranslatedSrt:
    def test_creates_file(self, sample_segments, sample_translations, tmp_path):
        out = str(tmp_path / "translated.srt")
        build_translated_srt(sample_segments, sample_translations, out)
        assert os.path.exists(out)

    def test_translated_texts_present(self, sample_segments, sample_translations, tmp_path):
        out = str(tmp_path / "translated.srt")
        build_translated_srt(sample_segments, sample_translations, out)
        content = open(out, encoding="utf-8").read()
        for trans in sample_translations:
            assert trans in content

    def test_original_texts_absent(self, sample_segments, sample_translations, tmp_path):
        out = str(tmp_path / "translated.srt")
        build_translated_srt(sample_segments, sample_translations, out)
        content = open(out, encoding="utf-8").read()
        for seg in sample_segments:
            assert seg["text"] not in content

    def test_timestamps_match_original(self, sample_segments, sample_translations, tmp_path):
        orig = str(tmp_path / "original.srt")
        trans = str(tmp_path / "translated.srt")
        build_original_srt(sample_segments, orig)
        build_translated_srt(sample_segments, sample_translations, trans)
        assert SRT_TIMESTAMP_RE.findall(open(orig).read()) == SRT_TIMESTAMP_RE.findall(open(trans).read())


class TestBuildDualSrt:
    def test_creates_file(self, sample_segments, sample_translations, tmp_path):
        out = str(tmp_path / "dual.srt")
        build_dual_srt(sample_segments, sample_translations, out)
        assert os.path.exists(out)

    def test_contains_both_languages(self, sample_segments, sample_translations, tmp_path):
        out = str(tmp_path / "dual.srt")
        build_dual_srt(sample_segments, sample_translations, out)
        content = open(out, encoding="utf-8").read()
        for seg in sample_segments:
            assert seg["text"] in content
        for trans in sample_translations:
            assert trans in content

    def test_original_before_translation(self, sample_segments, sample_translations, tmp_path):
        out = str(tmp_path / "dual.srt")
        build_dual_srt(sample_segments, sample_translations, out)
        content = open(out, encoding="utf-8").read()
        blocks = [b.strip() for b in content.strip().split("\n\n") if b.strip()]
        for i, block in enumerate(blocks):
            lines = block.split("\n")
            # lines[0]=index, lines[1]=timestamp, lines[2]=원본, lines[3]=번역
            assert lines[2] == sample_segments[i]["text"]
            assert lines[3] == sample_translations[i]

    def test_unicode_text(self, tmp_path):
        segments = [{"start": 0.0, "end": 1.0, "text": "日本語テスト 🎬"}]
        translations = ["일본어 테스트 🎬"]
        out = str(tmp_path / "dual_unicode.srt")
        build_dual_srt(segments, translations, out)
        content = open(out, encoding="utf-8").read()
        assert "日本語テスト 🎬" in content
        assert "일본어 테스트 🎬" in content


class TestBuildAll:
    def test_returns_three_keys(self, sample_segments, sample_translations, results_dir):
        paths = build_all(sample_segments, sample_translations, results_dir, "job-001")
        assert set(paths.keys()) == {"original", "translated", "dual"}

    def test_all_files_exist(self, sample_segments, sample_translations, results_dir):
        paths = build_all(sample_segments, sample_translations, results_dir, "job-002")
        for key, path in paths.items():
            assert os.path.exists(path), f"{key} 파일이 존재하지 않음"

    def test_files_not_empty(self, sample_segments, sample_translations, results_dir):
        paths = build_all(sample_segments, sample_translations, results_dir, "job-003")
        for key, path in paths.items():
            assert os.path.getsize(path) > 0, f"{key} 파일이 비어있음"

    def test_filename_contains_job_id(self, sample_segments, sample_translations, results_dir):
        job_id = "my-unique-job-xyz"
        paths = build_all(sample_segments, sample_translations, results_dir, job_id)
        for path in paths.values():
            assert job_id in os.path.basename(path)
