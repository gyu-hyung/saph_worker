"""번역 엔진 모듈 — Provider 패턴으로 Ollama(개발) / OpenAI(프로덕션)를 추상화합니다."""

import os
from abc import ABC, abstractmethod
from openai import OpenAI


class TranslatorProvider(ABC):
    @abstractmethod
    def translate(self, segments: list[dict], context: str, target_lang: str) -> list[str]:
        """
        세그먼트 목록을 번역합니다.

        Args:
            segments: [{"start": float, "end": float, "text": str}, ...]
            context: Pass 1에서 추출한 문맥 요약 문자열
            target_lang: 번역 대상 언어 코드 (e.g. "ko")

        Returns:
            번역된 텍스트 목록 (segments와 동일한 길이)
        """


class OllamaTranslator(TranslatorProvider):
    """Ollama 로컬 LLM을 사용한 번역 — 개발 환경 (무료)."""

    def __init__(self):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = os.getenv("OLLAMA_MODEL", "gemma3:12b")
        self.client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")

    def _chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def analyze_context(self, full_text: str) -> str:
        """Pass 1: 전체 STT 텍스트에서 문맥 요약 + 도메인 키워드를 추출합니다."""
        system = (
            "You are a translation assistant. Analyze the provided transcript and respond "
            "with a one-sentence summary and up to 5 domain-specific keywords. "
            "Format: SUMMARY: <summary>\nKEYWORDS: <kw1>, <kw2>, ..."
        )
        return self._chat(system, full_text[:4000])

    def translate(self, segments: list[dict], context: str, target_lang: str) -> list[str]:
        """Pass 2: 문맥 정보를 포함하여 세그먼트를 번역합니다."""
        lang_map = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
        lang_name = lang_map.get(target_lang, target_lang)

        system = (
            f"You are a professional subtitle translator. Translate each line into natural {lang_name}. "
            f"Context about this video:\n{context}\n\n"
            "Rules:\n"
            "- Translate each line independently, preserving meaning and tone.\n"
            "- Output ONLY the translated lines, one per line, in the same order.\n"
            "- Do NOT add explanations, numbering, or extra text."
        )
        numbered = "\n".join(f"{i + 1}. {seg['text']}" for i, seg in enumerate(segments))
        raw = self._chat(system, numbered)

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        # 번호 제거 (e.g. "1. text" → "text")
        cleaned = []
        for line in lines:
            if line and line[0].isdigit() and ". " in line:
                cleaned.append(line.split(". ", 1)[1])
            else:
                cleaned.append(line)

        # 길이 불일치 시 빈 문자열로 패딩
        if len(cleaned) < len(segments):
            cleaned += [""] * (len(segments) - len(cleaned))

        return cleaned[: len(segments)]


class OpenAITranslator(TranslatorProvider):
    """OpenAI GPT API를 사용한 번역 — 프로덕션 환경."""

    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def analyze_context(self, full_text: str) -> str:
        system = (
            "Analyze the transcript and respond with: "
            "SUMMARY: <one sentence>\nKEYWORDS: <kw1>, <kw2>, ..."
        )
        return self._chat(system, full_text[:4000])

    def translate(self, segments: list[dict], context: str, target_lang: str) -> list[str]:
        lang_map = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
        lang_name = lang_map.get(target_lang, target_lang)

        system = (
            f"Translate each subtitle line into natural {lang_name}. "
            f"Context:\n{context}\n\n"
            "Output ONLY translated lines, one per line, same order, no extra text."
        )
        numbered = "\n".join(f"{i + 1}. {seg['text']}" for i, seg in enumerate(segments))
        raw = self._chat(system, numbered)

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        cleaned = []
        for line in lines:
            if line and line[0].isdigit() and ". " in line:
                cleaned.append(line.split(". ", 1)[1])
            else:
                cleaned.append(line)

        if len(cleaned) < len(segments):
            cleaned += [""] * (len(segments) - len(cleaned))

        return cleaned[: len(segments)]


def get_translator() -> TranslatorProvider:
    """환경 변수 TRANSLATOR_PROVIDER에 따라 번역 엔진을 반환합니다."""
    provider = os.getenv("TRANSLATOR_PROVIDER", "ollama").lower()
    if provider == "openai":
        return OpenAITranslator()
    return OllamaTranslator()


def translate_with_context(
    segments: list[dict],
    target_lang: str = "ko",
    progress_callback=None,
) -> list[str]:
    """
    2-Pass 번역을 수행합니다.

    Pass 1: 전체 텍스트 → 문맥 요약 추출
    Pass 2: 문맥 + 세그먼트 → 번역

    5분(300초) 이하 영상은 전체를 한 번에 번역합니다.
    """
    translator = get_translator()

    full_text = " ".join(seg["text"] for seg in segments)
    total_duration = segments[-1]["end"] if segments else 0

    # Pass 1: 문맥 분석
    if progress_callback:
        progress_callback(62)
    context = ""
    if hasattr(translator, "analyze_context"):
        context = translator.analyze_context(full_text)

    # Pass 2: 본 번역
    # 5분 이하는 전체 처리, 초과 시 300초 단위 청크로 분할
    CHUNK_SEC = 300

    if total_duration <= CHUNK_SEC:
        if progress_callback:
            progress_callback(75)
        return translator.translate(segments, context, target_lang)

    # 청크 분할 번역
    results: list[str] = []
    chunk: list[dict] = []
    chunk_start_sec = 0.0

    for seg in segments:
        if seg["start"] - chunk_start_sec >= CHUNK_SEC and chunk:
            translated = translator.translate(chunk, context, target_lang)
            results.extend(translated)
            chunk = []
            chunk_start_sec = seg["start"]
        chunk.append(seg)

    if chunk:
        results.extend(translator.translate(chunk, context, target_lang))

    return results
