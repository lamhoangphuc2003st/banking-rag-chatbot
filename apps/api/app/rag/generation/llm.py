from __future__ import annotations

import re
import unicodedata
from collections.abc import AsyncIterator
from typing import Any

from apps.api.app.core.config import Settings
from apps.api.app.rag.prompts import ANSWER_TEMPLATE, SYSTEM_PROMPT
from packages.shared.schemas import RetrievedChunk

MARKDOWN_LINK_PATTERN = re.compile(
    r"\[([^\]]+)\]\((?:https?://|www\.)[^\s)]+(?:\s+\"[^\"]*\")?\)",
    flags=re.IGNORECASE,
)
MARKDOWN_LINK_ONLY_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?\[([^\]]+)\]\((?:https?://|www\.)[^\s)]+(?:\s+\"[^\"]*\")?\)\s*[.!?]?\s*$",
    flags=re.IGNORECASE,
)
REFERENCE_URL_LINE_PATTERN = re.compile(
    r"^\s*(?:URL|Source|Link|Nguồn|Đường dẫn)\s*:\s*(?:https?://|www\.)\S+\s*$",
    flags=re.IGNORECASE,
)
BARE_URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s)\]]+", flags=re.IGNORECASE)
DISPOSABLE_LINK_LABELS = {
    "chi tiết",
    "xem chi tiết",
    "xem thêm",
    "tham khảo",
    "tham khảo thêm",
    "nguồn",
    "link",
}


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate_answer(
        self,
        *,
        question: str,
        history: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        context = self._format_context(chunks)
        if not context:
            return (
                "Tôi chưa tìm thấy thông tin phù hợp trong nguồn dữ liệu hiện có. "
                "Bạn vui lòng kiểm tra lại trên website hoặc kênh chính thức của Vietcombank."
            )

        if self._uses_local_generation():
            return self._deterministic_local_answer(chunks)

        from litellm import acompletion

        response = await acompletion(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": self._build_prompt(question, history, context)}],
            temperature=self.settings.llm_temperature,
            api_key=self.settings.openai_api_key or self.settings.litellm_api_key,
        )
        return sanitize_answer_text(str(response["choices"][0]["message"]["content"]).strip())

    async def stream_answer(
        self,
        *,
        question: str,
        history: str,
        chunks: list[RetrievedChunk],
    ) -> AsyncIterator[str]:
        context = self._format_context(chunks)
        if not context:
            yield (
                "Tôi chưa tìm thấy thông tin phù hợp trong nguồn dữ liệu hiện có. "
                "Bạn vui lòng kiểm tra lại trên website hoặc kênh chính thức của Vietcombank."
            )
            return

        if self._uses_local_generation():
            for part in _chunk_stream_text(self._deterministic_local_answer(chunks)):
                yield part
            return

        from litellm import acompletion

        stream = await acompletion(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": self._build_prompt(question, history, context)}],
            temperature=self.settings.llm_temperature,
            api_key=self.settings.openai_api_key or self.settings.litellm_api_key,
            stream=True,
        )
        async for chunk in stream:
            content = _extract_stream_content(chunk)
            if content:
                yield content

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            content = _sanitize_context_text(chunk.text)
            lines = [
                f"[{index}] {chunk.title}",
                f"SECTION: {chunk.section or 'unknown'}",
            ]
            subquery = chunk.metadata.get("subquery")
            if isinstance(subquery, str) and subquery.strip():
                lines.append(f"SUBQUERY: {subquery.strip()}")
            lines.append(f"CONTENT: {content}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def _deterministic_local_answer(self, chunks: list[RetrievedChunk]) -> str:
        first = chunks[0]
        excerpt = first.text[:900].strip()
        return sanitize_answer_text(
            "Dựa trên nguồn dữ liệu đã truy xuất, thông tin liên quan nhất là:\n\n"
            f"{excerpt}\n\n"
            "Thông tin công khai của Vietcombank có thể thay đổi theo thời điểm."
        )

    def _build_prompt(self, question: str, history: str, context: str) -> str:
        return ANSWER_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            history=history,
            question=question,
        )

    def _uses_local_generation(self) -> bool:
        provider = self.settings.llm_provider.strip().lower()
        return provider in {"", "local", "none"} or (
            provider == "openai" and not self.settings.openai_api_key
        )


def _extract_stream_content(chunk: Any) -> str:
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        delta = choice.get("delta") or {}
        if isinstance(delta, dict):
            return str(delta.get("content") or "")
        return str(choice.get("text") or "")

    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    choice = choices[0]
    delta = getattr(choice, "delta", None)
    if isinstance(delta, dict):
        return str(delta.get("content") or "")
    content = getattr(delta, "content", None)
    if content:
        return str(content)
    text = getattr(choice, "text", None)
    return str(text or "")


def _chunk_stream_text(text: str, *, chunk_size: int = 80) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    current = ""
    for token in re.split(r"(\s+)", text):
        if len(current) + len(token) > chunk_size and current:
            chunks.append(current)
            current = token
        else:
            current += token
    if current:
        chunks.append(current)
    return chunks


def sanitize_answer_text(answer: str) -> str:
    cleaned_lines: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        link_only_match = MARKDOWN_LINK_ONLY_PATTERN.match(stripped)
        if link_only_match and _is_disposable_reference_label(link_only_match.group(1)):
            continue
        if REFERENCE_URL_LINE_PATTERN.match(stripped):
            continue
        if _is_missing_information_disclaimer(stripped):
            continue

        cleaned = MARKDOWN_LINK_PATTERN.sub(lambda match: match.group(1).strip(), line)
        cleaned = BARE_URL_PATTERN.sub("", cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).rstrip()
        if _is_missing_information_disclaimer(cleaned):
            continue
        if _is_disposable_reference_label(cleaned):
            continue
        if cleaned.strip():
            cleaned_lines.append(cleaned)

    return _collapse_blank_lines(cleaned_lines).strip()


def _sanitize_context_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or REFERENCE_URL_LINE_PATTERN.match(stripped)
            or "no product detail chunks were found" in stripped.casefold()
        ):
            continue
        cleaned = BARE_URL_PATTERN.sub("", line).rstrip()
        if cleaned.strip():
            cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)


def _collapse_blank_lines(lines: list[str]) -> str:
    collapsed: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        collapsed.append(line)
        previous_blank = is_blank
    return "\n".join(collapsed)


def _is_disposable_reference_label(text: str) -> bool:
    label = text.strip().strip("-*:：. ").casefold()
    return label in DISPOSABLE_LINK_LABELS


def _is_missing_information_disclaimer(text: str) -> bool:
    normalized = _normalize_filter_text(text)
    if not normalized:
        return False

    missing_markers = (
        "chua cung cap",
        "chua co thong tin",
        "chua tim thay",
        "khong co thong tin",
        "khong du thong tin",
        "khong tim thay",
        "nguon du lieu hien co chua",
        "nguon hien co chua",
    )
    advice_markers = (
        "ban can cung cap them",
        "ban vui long cung cap them",
        "kiem tra kenh chinh thuc",
        "kiem tra lai tren website",
        "de biet chi tiet hon",
    )
    missing_fields = (
        "bieu phi",
        "chi tiet",
        "dieu kien",
        "doi tuong",
        "han muc",
        "ho so",
        "lai suat",
        "ngay hieu luc",
        "thu tuc",
        "yeu cau",
    )

    has_missing_marker = any(marker in normalized for marker in missing_markers)
    has_advice_marker = any(marker in normalized for marker in advice_markers)
    has_missing_field = any(marker in normalized for marker in missing_fields)
    return (has_missing_marker and (has_missing_field or has_advice_marker)) or (
        has_advice_marker and has_missing_field
    )


def _normalize_filter_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(character for character in normalized if not unicodedata.combining(character))
    ascii_text = ascii_text.replace("đ", "d")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))
