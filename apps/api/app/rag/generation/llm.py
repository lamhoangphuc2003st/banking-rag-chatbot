from __future__ import annotations

from apps.api.app.core.config import Settings
from apps.api.app.rag.prompts import ANSWER_TEMPLATE, SYSTEM_PROMPT
from packages.shared.schemas import RetrievedChunk


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

        if not self.settings.openai_api_key and self.settings.llm_provider == "openai":
            return self._deterministic_local_answer(chunks)

        from litellm import acompletion

        prompt = ANSWER_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            history=history,
            question=question,
        )
        response = await acompletion(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.settings.llm_temperature,
            api_key=self.settings.openai_api_key or self.settings.litellm_api_key,
        )
        return str(response["choices"][0]["message"]["content"]).strip()

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            parts.append(
                "\n".join(
                    [
                        f"[{index}] {chunk.title}",
                        f"URL: {chunk.source_url}",
                        f"SECTION: {chunk.section or 'unknown'}",
                        f"CONTENT: {chunk.text}",
                    ]
                )
            )
        return "\n\n".join(parts)

    def _deterministic_local_answer(self, chunks: list[RetrievedChunk]) -> str:
        first = chunks[0]
        excerpt = first.text[:900].strip()
        return (
            "Dựa trên nguồn dữ liệu đã truy xuất, thông tin liên quan nhất là:\n\n"
            f"{excerpt}\n\n"
            "Bạn nên kiểm tra lại trên kênh chính thức của Vietcombank vì biểu phí, "
            "điều kiện và lãi suất có thể thay đổi theo thời điểm."
        )
