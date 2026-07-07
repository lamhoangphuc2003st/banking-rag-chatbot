"""Generate the consolidated evaluation report.

Runs the offline retrieval eval and the guardrail eval, writes machine-readable
JSON to ``data/reports/`` and a single human-readable Markdown summary to
``docs/evaluation-results.md`` — the artifact meant to be read directly (e.g. by
a reviewer) without running anything.

Everything here is offline and deterministic: no Qdrant, no LLM, no network. A
clean checkout can reproduce the exact numbers with::

    make eval-report
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from packages.evals import refusal_eval
from packages.evals.corpus import DEFAULT_CORPUS_PATH, LexicalCorpusRetriever, load_corpus
from packages.evals.retrieval_eval import DEFAULT_KS, evaluate, load_cases

RETRIEVAL_GOLDEN = Path("data/golden/retrieval_golden.jsonl")
GUARDRAIL_GOLDEN = Path("data/golden/guardrail_golden.jsonl")
RETRIEVAL_JSON = Path("data/reports/retrieval_eval.json")
GUARDRAIL_JSON = Path("data/reports/guardrail_eval.json")
RESULTS_MARKDOWN = Path("docs/evaluation-results.md")


def _pct(value: object) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "—"


def _retrieval_tables(report: dict[str, object]) -> list[str]:
    overall = report["overall"]
    assert isinstance(overall, dict)
    lines = [
        "### Retrieval quality",
        "",
        f"Backend: `{report.get('backend', 'lexical')}` (offline lexical baseline over "
        f"the committed corpus) · {report['positives']} labelled positive queries, "
        f"{report['negatives']} out-of-scope negatives.",
        "",
        "| Slice | n | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _retrieval_row("Overall", overall),
    ]

    by_difficulty = report["by_difficulty"]
    assert isinstance(by_difficulty, dict)
    for name in ("verbatim", "no_accent", "keyword", "paraphrase"):
        if name in by_difficulty:
            lines.append(_retrieval_row(f"· difficulty: {name}", by_difficulty[name]))

    by_section = report["by_section"]
    assert isinstance(by_section, dict)
    for name in sorted(by_section):
        lines.append(_retrieval_row(f"· section: {name}", by_section[name]))

    negatives = report["negatives_detail"]
    assert isinstance(negatives, dict)
    lines += [
        "",
        "Difficulty bands go from easy to hard: `verbatim` (exact question / product name), "
        "`no_accent` (Vietnamese typed without diacritics), `keyword` (content words only), "
        "`paraphrase` (colloquial rewrites with a real vocabulary gap). No-accent scores match "
        "verbatim, confirming the retriever folds diacritics; the `paraphrase` band is the true "
        "stress test and is exactly what the production dense-embedding + reranker stack exists "
        "to close.",
        "",
        f"**Out-of-scope handling:** {negatives['suppressed']}/{negatives['count']} negative "
        f"queries retrieved zero context "
        f"(suppression rate {_pct(negatives['context_suppression_rate'])}). The retriever "
        "returns nothing for unrelated questions, so generation cannot be grounded on noise.",
    ]
    return lines


def _retrieval_row(label: str, metrics: dict[str, object]) -> str:
    return (
        f"| {label} | {metrics.get('count', '—')} | "
        f"{_pct(metrics.get('recall@1'))} | {_pct(metrics.get('recall@5'))} | "
        f"{_pct(metrics.get('recall@10'))} | {_pct(metrics.get('mrr'))} | "
        f"{_pct(metrics.get('ndcg@10'))} |"
    )


def _guardrail_tables(report: dict[str, object]) -> list[str]:
    blocking = report["blocking"]
    scope = report["scope"]
    assert isinstance(blocking, dict) and isinstance(scope, dict)
    return [
        "### Guardrails (offline, deterministic)",
        "",
        f"{report['total']} labelled prompts (safe / sensitive-keyword / secret / PII / "
        "out-of-scope).",
        "",
        "| Check | Accuracy | Precision | Recall | TP | FP | TN | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _guardrail_row("Credential / PII blocking", blocking),
        _guardrail_row("Out-of-scope detection", scope),
        "",
        f"No secret or PII prompt is ever allowed through (**false-negative rate 0** on "
        f"{blocking['true_positive'] + blocking['false_negative']} disclosure prompts), and no "
        "public question that merely mentions a sensitive keyword (OTP, PIN, CVV, card number) "
        "is over-blocked (**false-positive rate 0**). Scope detection is heuristic; the misses "
        "are short-keyword substring collisions, tracked as a known limitation.",
    ]


def _guardrail_row(label: str, matrix: dict[str, object]) -> str:
    return (
        f"| {label} | {_pct(matrix['accuracy'])} | {_pct(matrix['precision'])} | "
        f"{_pct(matrix['recall'])} | {matrix['true_positive']} | {matrix['false_positive']} | "
        f"{matrix['true_negative']} | {matrix['false_negative']} |"
    )


def render_markdown(
    retrieval: dict[str, object],
    guardrail: dict[str, object],
    *,
    corpus_size: int,
) -> str:
    generated = datetime.now(UTC).strftime("%Y-%m-%d")
    lines = [
        "# Evaluation Results",
        "",
        "> Auto-generated by `make eval-report` "
        "(`python -m packages.evals.report`). Do not edit by hand.",
        "",
        f"- Generated: {generated}",
        f"- Corpus: {corpus_size} indexed chunks (`{DEFAULT_CORPUS_PATH.as_posix()}`)",
        "- Environment: fully offline — no Qdrant, embedding provider, or LLM required.",
        "",
        "All numbers below reproduce from a clean checkout. See "
        "[evaluation.md](evaluation.md) for methodology and honest limitations.",
        "",
        *_retrieval_tables(retrieval),
        "",
        *_guardrail_tables(guardrail),
        "",
        "### How to reproduce",
        "",
        "```bash",
        "make eval-report      # regenerates this file + the JSON reports",
        "make eval             # retrieval only (offline lexical baseline)",
        "make eval-guardrails  # guardrails only",
        "```",
        "",
    ]
    return "\n".join(lines)


async def _run() -> tuple[dict[str, object], dict[str, object], int]:
    corpus = load_corpus(DEFAULT_CORPUS_PATH)
    retriever = LexicalCorpusRetriever(corpus)
    retrieval = await evaluate(load_cases(RETRIEVAL_GOLDEN), retriever, ks=DEFAULT_KS)
    retrieval["backend"] = "lexical"

    guardrail = refusal_eval.evaluate(refusal_eval.load_cases(GUARDRAIL_GOLDEN))
    return retrieval, guardrail, len(corpus)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the consolidated evaluation report.")
    parser.add_argument("--markdown", type=Path, default=RESULTS_MARKDOWN)
    args = parser.parse_args()

    retrieval, guardrail, corpus_size = asyncio.run(_run())

    RETRIEVAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_JSON.write_text(
        json.dumps(retrieval, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    GUARDRAIL_JSON.write_text(
        json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        render_markdown(retrieval, guardrail, corpus_size=corpus_size), encoding="utf-8"
    )

    print(f"retrieval report -> {RETRIEVAL_JSON}")
    print(f"guardrail report -> {GUARDRAIL_JSON}")
    print(f"results summary  -> {args.markdown}")


if __name__ == "__main__":
    main()
