from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerChecks:
    has_citation: bool
    refused_when_required: bool
    contains_unsupported_claim: bool


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


def _contains_high_risk_claim(answer: str) -> bool:
    lowered = answer.lower()
    high_risk_terms = ["lãi suất", "%", "biểu phí", "điều kiện", "hồ sơ"]
    return any(term in lowered for term in high_risk_terms)
