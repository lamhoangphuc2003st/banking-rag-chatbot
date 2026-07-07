from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerChecks:
    has_citation: bool
    refused_when_required: bool
    contains_unsupported_claim: bool


@dataclass(frozen=True)
class AnswerCase:
    answer: str
    source_count: int
    should_refuse: bool
    refusal: bool


def run_static_answer_checks(
    *,
    answer: str,
    source_count: int,
    should_refuse: bool,
    refusal: bool,
) -> AnswerChecks:
    has_citation = source_count > 0
    refused_when_required = not should_refuse or refusal
    contains_unsupported_claim = _contains_high_risk_claim(answer) and source_count == 0
    return AnswerChecks(
        has_citation=has_citation,
        refused_when_required=refused_when_required,
        contains_unsupported_claim=contains_unsupported_claim,
    )


def evaluate_answer_set(cases: Sequence[AnswerCase]) -> dict[str, float | int]:
    """Aggregate static answer-quality metrics over a set of graded answers.

    Grounded (non-refusal) answers are expected to cite a source; required
    refusals must actually refuse; high-risk claims (rates, fees, conditions)
    must never appear without a supporting source.
    """

    total = len(cases)
    if total == 0:
        return {
            "total": 0,
            "citation_rate": 0.0,
            "refusal_accuracy": 0.0,
            "unsupported_claim_rate": 0.0,
        }

    grounded = [case for case in cases if not case.should_refuse]
    citations = sum(
        1 for case in grounded if run_static_answer_checks(
            answer=case.answer,
            source_count=case.source_count,
            should_refuse=case.should_refuse,
            refusal=case.refusal,
        ).has_citation
    )
    refusal_correct = 0
    unsupported = 0
    for case in cases:
        checks = run_static_answer_checks(
            answer=case.answer,
            source_count=case.source_count,
            should_refuse=case.should_refuse,
            refusal=case.refusal,
        )
        refusal_correct += int(checks.refused_when_required)
        unsupported += int(checks.contains_unsupported_claim)

    return {
        "total": total,
        "grounded": len(grounded),
        "citation_rate": round(citations / len(grounded), 4) if grounded else 1.0,
        "refusal_accuracy": round(refusal_correct / total, 4),
        "unsupported_claim_rate": round(unsupported / total, 4),
    }


def _contains_high_risk_claim(answer: str) -> bool:
    lowered = answer.lower()
    high_risk_terms = ["lãi suất", "%", "biểu phí", "điều kiện", "hồ sơ"]
    return any(term in lowered for term in high_risk_terms)
