# Worker Service — CLAUDE.md

Python AI 파이프라인. Redis Stream에서 Job을 수신하고 FFmpeg → STT → 번역 → SRT 생성을 수행한다.

---

## 실행

```bash
# 전제조건: make infra-up + Ollama 실행 (make pull-model)
pip install -r requirements.txt
python main.py
```

환경 변수 (로컬):
```bash
REDIS_URL=redis://localhost:6379
WHISPER_MODEL=base        # 로컬 CPU: base, GPU: large-v3
WHISPER_DEVICE=cpu
TRANSLATOR_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b    # 로컬: 4b (속도), 프로덕션: 12b
STORAGE_PATH=../../storage
```

---

## 파이프라인 구조

```
pipeline/
├── audio_extractor.py  # FFmpeg: 영상 → WAV 16kHz mono
├── stt_engine.py       # faster-whisper: WAV → List[Segment]
├── translator.py       # Provider 패턴: Segment 번역 (Ollama / OpenAI / Claude)
└── srt_builder.py      # TranslatedSegment → .srt 파일 생성
```

---

## 핵심 설계 결정 (WHY)

### Worker 1인스턴스 = 1Job 동시 처리
`XREADGROUP count=1`으로 한 번에 Job 1개만 수신한다.
faster-whisper(`large-v3` ~6GB VRAM) + Ollama(`gemma3:12b` ~8GB RAM)를 동시에 로드하면 OOM 위험이 있기 때문이다.
처리량 확장은 Worker 인스턴스를 수평으로 늘리는 방식으로 대응한다.

### Ollama는 Worker와 별도 컨테이너
번역 LLM(Ollama)은 Worker에 내장하지 않고 HTTP로 호출한다 (`OLLAMA_BASE_URL`).
다수의 Worker가 Ollama 1개를 공유하는 구조이며, STT가 전체 시간의 ~50%를 차지하므로 각 Worker의 Ollama 요청 시점이 자연스럽게 분산된다.

### 번역: 2-Pass 전략
- Pass 1 (문맥 분석): STT 전문 → LLM → `summary`, `domain`, `keywords` 추출 (temperature=0.1)
- Pass 2 (본 번역): 문맥 정보를 시스템 프롬프트에 포함하여 고품질 번역 (temperature=0.3)

MVP 기준 5분 이하 영상은 청크 분할 없이 전체를 한 번에 번역한다 (문맥 손실 방지).

### LLM 응답 파싱 형식
번역 요청/응답 형식: `index|text` (한 줄 한 항목)
파싱 실패한 세그먼트는 원문(`seg.text`)으로 대체한다. 번역 실패 시 전체를 에러 처리하지 않는다.

### Provider 패턴 (번역 엔진 추상화)
`TRANSLATOR_PROVIDER` 환경 변수로 번역 엔진을 교체한다:
- `ollama` — 개발/로컬 (무료)
- `openai` — 프로덕션 (GPT-4o)
- `claude` — 프로덕션 대안 (Claude 3.5 Sonnet)

Ollama는 OpenAI-compatible API이므로 `base_url`만 변경하면 동일 코드로 동작한다.

---

## Redis 메시지 규약

| 항목 | 값 |
|---|---|
| 작업 큐 | `stream:jobs` (Consumer Group: `worker-group`) |
| 진행률 콜백 | Pub/Sub `channel:job-progress` |
| Consumer 이름 | `worker-{hostname}` |

**진행률 콜백 percent 범위:**

| step | percent |
|---|---|
| `audio_extraction` | 0 → 10 |
| `stt` | 10 → 60 |
| `translation` | 60 → 90 |
| `subtitle_build` | 90 → 100 |

---

## 재시도 및 에러 처리

- 재시도 가능 에러: `EXTRACTION_TIMEOUT`, `STT_FAILED`, `OLLAMA_TIMEOUT`, `OLLAMA_CONNECTION`, `TRANSLATION_PARSE_ERROR`, `SRT_BUILD_ERROR`
- 재시도 불가 에러: `FILE_NOT_FOUND`, `NO_AUDIO_TRACK`, `CORRUPTED_FILE`, `STT_EMPTY_RESULT`
- 최대 2회 재시도 (딜레이: 5초, 15초)
- 모든 재시도 실패 시: `failed` 콜백 발행 → API가 크레딧 복구 처리

**ACK 정책:** 처리 성공 시에만 `XACK`. 실패/crash 시 ACK하지 않아 Redis Stream pending 상태로 남고, 모니터링 루프에서 5분 이상 pending인 메시지를 다른 Worker에 재할당(`XCLAIM`)한다.

---

## 파일 경로 규약

| 파일 | 경로 |
|---|---|
| 입력 영상 | `storage/videos/{uuid}.mp4` |
| 임시 오디오 | `/tmp/{job_id}_audio.wav` (처리 후 삭제) |
| 원본 자막 | `storage/results/{job_id}_original.srt` |
| 번역 자막 | `storage/results/{job_id}_translated.srt` |
