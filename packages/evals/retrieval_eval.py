from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from apps.api.app.core.config import get_settings
from apps.api.app.rag.retrieval.hybrid import HybridRetriever


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    expected_chunk_ids: list[str]
    product_type: str | None = None
    section: str | None = None


async def evaluate(golden_path: Path, k: int) -> dict[str, object]:
    cases = load_cases(golden_path)
    retriever = HybridRetriever(get_settings())

    total = len(cases)
    if total == 0:
        return {"total": 0, "recall_at_k": 0.0, "mrr": 0.0}

    hits = 0
    reciprocal_rank_sum = 0.0
    details: list[dict[str, object]] = []

    for case in cases:
        retrieved = await retriever.retrieve(case.query, top_k=k)
        retrieved_ids = [item.chunk_id for item in retrieved]
        expected = set(case.expected_chunk_ids)
        found = [chunk_id for chunk_id in retrieved_ids if chunk_id in expected]

        if found:
            hits += 1
            rank = retrieved_ids.index(found[0]) + 1
            reciprocal_rank_sum += 1 / rank
        else:
            rank = None

        details.append(
            {
                "query": case.query,
                "expected_chunk_ids": case.expected_chunk_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "hit": bool(found),
                "rank": rank,
                "product_type": case.product_type,
                "section": case.section,
            }
        )

    return {
        "total": total,
        "k": k,
        "hits": hits,
        "recall_at_k": hits / total,
        "mrr": reciprocal_rank_sum / total,
        "details": details,
    }


def load_cases(path: Path) -> list[RetrievalCase]:
    if not path.exists():
        return []

    cases: list[RetrievalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            cases.append(
                RetrievalCase(
                    query=payload["query"],
                    expected_chunk_ids=list(payload["expected_chunk_ids"]),
                    product_type=payload.get("product_type"),
                    section=payload.get("section"),
                )
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/reports/retrieval_eval.json"))
    args = parser.parse_args()

    report = asyncio.run(evaluate(args.golden, args.k))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
