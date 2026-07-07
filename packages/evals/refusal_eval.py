"""Offline guardrail / refusal evaluation.

Unlike retrieval and answer evaluation, this suite needs no vector store or
LLM: ``inspect_query`` and ``is_likely_supported_domain`` are deterministic
pure functions, so this eval runs in CI and gates regressions in safety
behaviour (credential/PII blocking and out-of-scope detection).

Usage::

    python -m packages.evals.refusal_eval --golden data/golden/guardrail_golden.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from apps.api.app.rag.guardrails import inspect_query, is_likely_supported_domain


@dataclass(frozen=True)
class GuardrailCase:
    query: str
    category: str
    expect_blocked: bool
    expect_in_scope: bool


@dataclass(frozen=True)
class ConfusionMatrix:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.true_positive + self.true_negative) / self.total

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    def with_prediction(self, *, expected: bool, predicted: bool) -> ConfusionMatrix:
        return ConfusionMatrix(
            true_positive=self.true_positive + int(expected and predicted),
            false_positive=self.false_positive + int(not expected and predicted),
            true_negative=self.true_negative + int(not expected and not predicted),
            false_negative=self.false_negative + int(expected and not predicted),
        )


def load_cases(path: Path) -> list[GuardrailCase]:
    cases: list[GuardrailCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            cases.append(
                GuardrailCase(
                    query=str(payload["query"]),
                    category=str(payload.get("category", "unknown")),
                    expect_blocked=bool(payload["expect_blocked"]),
                    expect_in_scope=bool(payload.get("expect_in_scope", True)),
                )
            )
    return cases


def evaluate(cases: list[GuardrailCase]) -> dict[str, object]:
    blocking = ConfusionMatrix()
    scope = ConfusionMatrix()
    failures: list[dict[str, object]] = []

    for case in cases:
        guardrail = inspect_query(case.query)
        blocked = not guardrail.allowed
        blocking = blocking.with_prediction(
            expected=case.expect_blocked,
            predicted=blocked,
        )
        if blocked != case.expect_blocked:
            failures.append(
                {
                    "query": case.query,
                    "category": case.category,
                    "check": "blocking",
                    "expected_blocked": case.expect_blocked,
                    "actual_blocked": blocked,
                    "reason": guardrail.reason,
                }
            )

        # Scope detection only matters for prompts the guardrail lets through.
        if not case.expect_blocked:
            in_scope = is_likely_supported_domain(case.query)
            scope = scope.with_prediction(
                expected=case.expect_in_scope,
                predicted=in_scope,
            )
            if in_scope != case.expect_in_scope:
                failures.append(
                    {
                        "query": case.query,
                        "category": case.category,
                        "check": "scope",
                        "expected_in_scope": case.expect_in_scope,
                        "actual_in_scope": in_scope,
                    }
                )

    return {
        "total": len(cases),
        "blocking": _matrix_summary(blocking),
        "scope": _matrix_summary(scope),
        "failures": failures,
    }


def _matrix_summary(matrix: ConfusionMatrix) -> dict[str, float | int]:
    return {
        "accuracy": round(matrix.accuracy, 4),
        "precision": round(matrix.precision, 4),
        "recall": round(matrix.recall, 4),
        "true_positive": matrix.true_positive,
        "false_positive": matrix.false_positive,
        "true_negative": matrix.true_negative,
        "false_negative": matrix.false_negative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate guardrail / refusal behaviour.")
    parser.add_argument("--golden", type=Path, default=Path("data/golden/guardrail_golden.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/guardrail_eval.json"))
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=1.0,
        help="Fail (exit 1) if blocking accuracy drops below this threshold.",
    )
    args = parser.parse_args()

    cases = load_cases(args.golden)
    report = evaluate(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))

    blocking_accuracy = float(report["blocking"]["accuracy"])  # type: ignore[index]
    if blocking_accuracy < args.min_accuracy:
        raise SystemExit(
            f"Guardrail blocking accuracy {blocking_accuracy} below threshold {args.min_accuracy}"
        )


if __name__ == "__main__":
    main()
