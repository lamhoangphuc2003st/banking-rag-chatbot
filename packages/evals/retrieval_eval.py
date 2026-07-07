"""Retrieval quality evaluation.

Runs a golden query set through a retriever and reports standard information-
retrieval metrics at the *document* level: a retrieved chunk counts as relevant
when it comes from one of the source documents labelled relevant for that query.
Document-level relevance is robust to re-chunking (chunk ids change when content
is re-crawled; source URLs do not), which is what previously made the golden set
go stale.

Two backends are supported:

* ``lexical`` (default) — fully offline, reproducible, CI-friendly. Ranks the
  committed corpus with the deployed lexical scorer. No Qdrant/OpenAI needed.
* ``hybrid`` — the live production retriever (dense + lexical). Requires a
  running Qdrant collection and an embedding provider.

Usage::

    python -m packages.evals.retrieval_eval --golden data/golden/retrieval_golden.jsonl
    python -m packages.evals.retrieval_eval --backend hybrid  # needs Qdrant
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from packages.evals.corpus import (
    DEFAULT_CORPUS_PATH,
    LexicalCorpusRetriever,
    load_corpus,
)
from packages.shared.schemas import RetrievedChunk

DEFAULT_KS = (1, 3, 5, 10)


class Retriever(Protocol):
    async def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_source_urls: list[str]
    expected_chunk_ids: list[str]
    product_type: str | None = None
    section: str | None = None
    difficulty: str | None = None
    category: str | None = None
    note: str | None = None

    @property
    def is_negative(self) -> bool:
        """Out-of-scope query: nothing in the corpus should be retrieved."""

        return not self.relevant_source_urls


def distinct_docs(retrieved: list[RetrievedChunk]) -> list[str]:
    """Collapse ranked chunks into a ranked list of distinct source documents."""

    ordered: list[str] = []
    seen: set[str] = set()
    for chunk in retrieved:
        url = chunk.source_url
        if not url or url in seen:
            continue
        seen.add(url)
        ordered.append(url)
    return ordered


def recall_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    found = len(set(retrieved_docs[:k]) & relevant)
    return found / len(relevant)


def hit_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved_docs[:k]) & relevant else 0.0


def reciprocal_rank(retrieved_docs: list[str], relevant: set[str]) -> float:
    for index, url in enumerate(retrieved_docs, start=1):
        if url in relevant:
            return 1.0 / index
    return 0.0


def _dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def ndcg_at_k(retrieved_docs: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k over the ranked list of distinct documents."""

    relevances = [1.0 if url in relevant else 0.0 for url in retrieved_docs[:k]]
    idcg = _dcg([1.0] * min(len(relevant), k))
    return _dcg(relevances) / idcg if idcg else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def evaluate(
    cases: list[RetrievalCase],
    retriever: Retriever,
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, object]:
    max_k = max(ks)
    positives = [case for case in cases if not case.is_negative]
    negatives = [case for case in cases if case.is_negative]

    per_case: list[dict[str, object]] = []
    # Accumulators for overall + sliced (by section, by difficulty) aggregation.
    buckets: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    def record(bucket: str, metric: str, value: float) -> None:
        buckets[bucket][metric].append(value)

    for case in positives:
        retrieved = await retriever.retrieve(case.query, top_k=max_k)
        retrieved_docs = distinct_docs(retrieved)
        relevant = set(case.relevant_source_urls)

        metrics: dict[str, float] = {}
        for k in ks:
            metrics[f"recall@{k}"] = recall_at_k(retrieved_docs, relevant, k)
            metrics[f"hit@{k}"] = hit_at_k(retrieved_docs, relevant, k)
        metrics["mrr"] = reciprocal_rank(retrieved_docs, relevant)
        metrics[f"ndcg@{max_k}"] = ndcg_at_k(retrieved_docs, relevant, max_k)

        for metric, value in metrics.items():
            record("overall", metric, value)
            if case.section:
                record(f"section:{case.section}", metric, value)
            if case.difficulty:
                record(f"difficulty:{case.difficulty}", metric, value)

        first_hit_rank = next(
            (i for i, url in enumerate(retrieved_docs, start=1) if url in relevant),
            None,
        )
        per_case.append(
            {
                "query": case.query,
                "section": case.section,
                "product_type": case.product_type,
                "difficulty": case.difficulty,
                "relevant_source_urls": case.relevant_source_urls,
                "retrieved_source_urls": retrieved_docs[:max_k],
                "first_relevant_rank": first_hit_rank,
                **{name: round(value, 4) for name, value in metrics.items()},
            }
        )

    negative_details: list[dict[str, object]] = []
    suppressed = 0
    for case in negatives:
        retrieved = await retriever.retrieve(case.query, top_k=max_k)
        is_suppressed = len(retrieved) == 0
        suppressed += int(is_suppressed)
        negative_details.append(
            {
                "query": case.query,
                "category": case.category,
                "suppressed": is_suppressed,
                "leaked_source_urls": distinct_docs(retrieved)[:3],
            }
        )

    report: dict[str, object] = {
        "total_cases": len(cases),
        "positives": len(positives),
        "negatives": len(negatives),
        "ks": list(ks),
        "overall": _summarize(buckets["overall"]),
        "by_section": {
            name.split(":", 1)[1]: _summarize(metrics)
            for name, metrics in sorted(buckets.items())
            if name.startswith("section:")
        },
        "by_difficulty": {
            name.split(":", 1)[1]: _summarize(metrics)
            for name, metrics in sorted(buckets.items())
            if name.startswith("difficulty:")
        },
        "negatives_detail": {
            "count": len(negatives),
            "suppressed": suppressed,
            "context_suppression_rate": round(suppressed / len(negatives), 4)
            if negatives
            else 1.0,
            "cases": negative_details,
        },
        "details": per_case,
    }
    return report


def _summarize(metrics: dict[str, list[float]]) -> dict[str, object]:
    summary: dict[str, object] = {"count": len(next(iter(metrics.values()))) if metrics else 0}
    for name, values in metrics.items():
        summary[name] = round(_mean(values), 4)
    return summary


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
                    query=str(payload["query"]),
                    relevant_source_urls=list(payload.get("relevant_source_urls") or []),
                    expected_chunk_ids=list(payload.get("expected_chunk_ids") or []),
                    product_type=payload.get("product_type"),
                    section=payload.get("section"),
                    difficulty=payload.get("difficulty"),
                    category=payload.get("category"),
                    note=payload.get("note"),
                )
            )
    return cases


async def _build_retriever(backend: str, corpus_path: Path) -> Retriever:
    if backend == "lexical":
        return LexicalCorpusRetriever(load_corpus(corpus_path))
    if backend == "hybrid":
        from apps.api.app.core.config import get_settings
        from apps.api.app.rag.retrieval.hybrid import HybridRetriever

        return HybridRetriever(get_settings())
    raise SystemExit(f"unknown backend: {backend!r} (choose 'lexical' or 'hybrid')")


async def _run(args: argparse.Namespace) -> dict[str, object]:
    retriever = await _build_retriever(args.backend, args.corpus)
    cases = load_cases(args.golden)
    report = await evaluate(cases, retriever, ks=tuple(args.k))
    report["backend"] = args.backend
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--golden", type=Path, default=Path("data/golden/retrieval_golden.jsonl"))
    parser.add_argument("--backend", choices=("lexical", "hybrid"), default="lexical")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--output", type=Path, default=Path("data/reports/retrieval_eval.json"))
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    printable = {key: value for key, value in report.items() if key != "details"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
