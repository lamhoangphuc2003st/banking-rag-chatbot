from __future__ import annotations

from pathlib import Path

import pytest

from packages.evals import build_golden
from packages.evals.corpus import (
    LexicalCorpusRetriever,
    chunk_ids_for_source_url,
    corpus_chunk_ids,
    load_corpus,
)
from packages.evals.retrieval_eval import (
    distinct_docs,
    evaluate,
    hit_at_k,
    load_cases,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from packages.shared.schemas import RetrievedChunk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = PROJECT_ROOT / "data" / "chunks" / "vietcombank_chunks.jsonl"
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden" / "retrieval_golden.jsonl"


def _chunk(chunk_id: str, source_url: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc",
        title="t",
        source_url=source_url,
        text="x",
        score=1.0,
    )


def test_distinct_docs_dedupes_by_source_url_keeping_order() -> None:
    retrieved = [_chunk("a", "u1"), _chunk("b", "u1"), _chunk("c", "u2")]
    assert distinct_docs(retrieved) == ["u1", "u2"]


def test_recall_and_hit_at_k() -> None:
    retrieved_docs = ["u3", "u1", "u2"]
    relevant = {"u1", "u2"}
    assert recall_at_k(retrieved_docs, relevant, 1) == 0.0
    assert recall_at_k(retrieved_docs, relevant, 2) == 0.5
    assert recall_at_k(retrieved_docs, relevant, 3) == 1.0
    assert hit_at_k(retrieved_docs, relevant, 1) == 0.0
    assert hit_at_k(retrieved_docs, relevant, 2) == 1.0


def test_reciprocal_rank_uses_first_relevant_position() -> None:
    assert reciprocal_rank(["u3", "u1", "u2"], {"u1"}) == pytest.approx(0.5)
    assert reciprocal_rank(["u1"], {"u1"}) == 1.0
    assert reciprocal_rank(["u9"], {"u1"}) == 0.0


def test_ndcg_rewards_relevant_docs_ranked_higher() -> None:
    relevant = {"u1", "u2"}
    top = ndcg_at_k(["u1", "u2", "u3"], relevant, 3)
    bottom = ndcg_at_k(["u3", "u1", "u2"], relevant, 3)
    assert top == 1.0
    assert bottom < top


def test_golden_labels_all_exist_in_committed_corpus() -> None:
    """Regression guard: every label must resolve against the indexed corpus.

    This is exactly the check that was missing when the previous golden set
    drifted 70% stale after a re-crawl.
    """

    corpus = load_corpus(CORPUS_PATH)
    known_chunk_ids = corpus_chunk_ids(corpus)
    known_source_urls = set(chunk_ids_for_source_url(corpus))

    cases = load_cases(GOLDEN_PATH)
    assert cases, "golden set is empty"

    missing_chunks: list[str] = []
    missing_urls: list[str] = []
    for case in cases:
        for chunk_id in case.expected_chunk_ids:
            if chunk_id not in known_chunk_ids:
                missing_chunks.append(chunk_id)
        for url in case.relevant_source_urls:
            if url not in known_source_urls:
                missing_urls.append(url)

    assert missing_chunks == [], f"{len(missing_chunks)} golden chunk ids not in corpus"
    assert missing_urls == [], f"{len(missing_urls)} golden source urls not in corpus"


def test_curated_paraphrase_anchors_resolve() -> None:
    _, title_index = build_golden.load_docs(CORPUS_PATH)
    unresolved = [anchor for _, anchor in build_golden.CURATED_PARAPHRASES if anchor not in title_index]
    assert unresolved == [], f"paraphrase anchors no longer in corpus: {unresolved}"


async def test_offline_eval_meets_reproducible_thresholds() -> None:
    corpus = load_corpus(CORPUS_PATH)
    retriever = LexicalCorpusRetriever(corpus)
    report = await evaluate(load_cases(GOLDEN_PATH), retriever)

    overall = report["overall"]
    assert isinstance(overall, dict)
    # Conservative floors: catch a real retrieval regression without being flaky.
    assert overall["recall@10"] >= 0.85
    assert overall["mrr"] >= 0.75

    negatives = report["negatives_detail"]
    assert isinstance(negatives, dict)
    # The lexical retriever must never fabricate context for out-of-scope queries.
    assert negatives["context_suppression_rate"] == 1.0

    # The hard band should stay materially below the easy band — proof the eval
    # is actually discriminating, not trivially saturated at 1.0.
    by_difficulty = report["by_difficulty"]
    assert isinstance(by_difficulty, dict)
    assert by_difficulty["paraphrase"]["recall@1"] < by_difficulty["verbatim"]["recall@1"]
