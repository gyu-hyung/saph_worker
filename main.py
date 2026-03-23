"""Worker 진입점 — Redis Stream Consumer Group으로 번역 작업을 처리합니다."""

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
import redis

# .env.local 로드 (파일 위치 기준)
env_path = Path(__file__).parent / '.env.local'
load_dotenv(env_path)

from pipeline.audio_extractor import extract_audio
from pipeline.srt_builder import build_all
from pipeline.stt_engine import transcribe
from pipeline.translator import translate_with_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
STREAM_KEY = "stream:jobs"
GROUP_NAME = "workers"
CONSUMER_NAME = os.getenv("WORKER_NAME", f"worker-{os.getpid()}")
RESULTS_DIR = os.getenv("RESULTS_DIR", "/storage/results")
MAX_RETRIES = 2


def get_redis() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def ensure_consumer_group(r: redis.Redis) -> None:
    try:
        r.xgroup_create(STREAM_KEY, GROUP_NAME, id="0", mkstream=True)
        log.info("Consumer group '%s' 생성됨", GROUP_NAME)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            log.info("Consumer group '%s' 이미 존재함", GROUP_NAME)
        else:
            raise


def publish_progress(r: redis.Redis, job_id: str, step: str, percent: int, message: str = "") -> None:
    payload = json.dumps({
        "jobId": job_id,
        "step": step,
        "percent": percent,
        "message": message,
    })
    r.publish(f"job:progress:{job_id}", payload)


def process_job(r: redis.Redis, job_id: str, video_path: str, target_lang: str) -> dict:
    """단일 Job 처리 파이프라인을 실행합니다."""
    audio_path = None

    def progress(step: str, percent: int, msg: str = ""):
        publish_progress(r, job_id, step, percent, msg)
        log.info("[%s] %s %d%%", job_id, step, percent)

    try:
        # [1] 오디오 추출 (0 → 10%)
        progress("audio_extraction", 0, "오디오 추출 중...")
        audio_path = extract_audio(video_path)
        progress("audio_extraction", 10, "오디오 추출 완료")

        # [2] STT (10 → 60%)
        progress("stt", 10, "음성 인식 중...")

        def stt_progress(pct: int):
            progress("stt", 10 + int(pct * 0.5), "음성 인식 중...")

        segments, detected_lang = transcribe(audio_path, progress_callback=stt_progress)
        progress("stt", 60, f"음성 인식 완료 ({detected_lang})")

        # [3] 번역 (60 → 90%)
        progress("translation", 60, "AI 번역 중...")

        def trans_progress(pct: int):
            progress("translation", 60 + int(pct * 0.3), "AI 번역 중...")

        translations = translate_with_context(
            segments, target_lang=target_lang, progress_callback=trans_progress
        )
        progress("translation", 90, "번역 완료")

        # [4] SRT 생성 (90 → 100%)
        progress("subtitle_build", 90, "자막 파일 생성 중...")
        paths = build_all(segments, translations, RESULTS_DIR, job_id)
        progress("subtitle_build", 100, "완료")

        return {
            "status": "COMPLETED",
            "original_srt": paths["original"],
            "translated_srt": paths["translated"],
            "dual_srt": paths["dual"],
        }

    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


def handle_message(r: redis.Redis, stream_id: str, data: dict) -> None:
    job_id = data.get("jobId")
    video_path = data.get("videoPath")
    target_lang = data.get("targetLang", "ko")

    if not job_id or not video_path:
        log.error("잘못된 메시지 형식: %s", data)
        r.xack(STREAM_KEY, GROUP_NAME, stream_id)
        return

    log.info("작업 시작: jobId=%s", job_id)
    publish_progress(r, job_id, "processing", 0, "작업 시작")

    retry_count = 0
    while retry_count <= MAX_RETRIES:
        try:
            result = process_job(r, job_id, video_path, target_lang)

            # 완료 이벤트 발행
            r.publish(f"job:done:{job_id}", json.dumps(result))
            r.xack(STREAM_KEY, GROUP_NAME, stream_id)
            log.info("작업 완료: jobId=%s", job_id)
            return

        except Exception as e:
            retry_count += 1
            log.error("작업 실패 (시도 %d/%d): jobId=%s, 오류=%s", retry_count, MAX_RETRIES, job_id, e)

            if retry_count > MAX_RETRIES:
                # 최종 실패 처리
                r.publish(f"job:done:{job_id}", json.dumps({
                    "status": "FAILED",
                    "error": str(e),
                }))
                r.xack(STREAM_KEY, GROUP_NAME, stream_id)
                log.error("최종 실패 처리: jobId=%s", job_id)
                return

            time.sleep(2 ** retry_count)  # 지수 백오프


def main():
    log.info("Worker 시작: %s", CONSUMER_NAME)
    r = get_redis()
    ensure_consumer_group(r)

    log.info("작업 대기 중... (stream=%s, group=%s)", STREAM_KEY, GROUP_NAME)

    while True:
        try:
            messages = r.xreadgroup(
                GROUP_NAME,
                CONSUMER_NAME,
                {STREAM_KEY: ">"},
                count=1,
                block=5000,  # 5초 블로킹
            )

            if not messages:
                continue

            for stream, entries in messages:
                for stream_id, data in entries:
                    handle_message(r, stream_id, data)

        except redis.exceptions.ConnectionError as e:
            log.error("Redis 연결 오류: %s. 5초 후 재시도...", e)
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("Worker 종료")
            break


if __name__ == "__main__":
    main()
