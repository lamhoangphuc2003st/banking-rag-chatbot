"""Post-deployment evaluation against a live backend (e.g. Render).

The offline suite measures the lexical retrieval stage and the deterministic
guardrails. Once the backend is deployed there are things only a running system
can tell you, and this harness checks them end to end over HTTP:

1. **Availability / readiness** — ``/health/live`` and ``/health/ready`` (the
   latter verifies Qdrant, Postgres, Redis and the product graph are wired up).
2. **End-to-end answers** — real ``/v1/chat`` calls exercise the *full* production
   stack (guardrails → rewrite/plan → graph + dense/lexical hybrid → Cohere
   rerank → LLM generation → citations), which the offline lexical baseline
   cannot.
3. **Live grounding / citation relevance** — for labelled queries, does the
   deployed answer cite the expected source document? Measures the whole stack as
   the user experiences it.
4. **Guardrails end to end** — secret / PII / out-of-scope prompts must actually
   be refused by the deployed service.
5. **Latency** — P50/P95/max wall-clock, including Render cold starts.
6. **Metrics** — scrapes ``/metrics`` to confirm Prometheus instrumentation is
   live and counters move.

Usage::

    python -m packages.evals.live_eval --base-url https://<service>.onrender.com
    make eval-live-deploy BASE_URL=https://<service>.onrender.com
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import httpx

from packages.evals.refusal_eval import load_cases as load_guardrail_cases
from packages.evals.retrieval_eval import load_cases as load_retrieval_cases

T = TypeVar("T")

RETRIEVAL_GOLDEN = Path("data/golden/retrieval_golden.jsonl")
GUARDRAIL_GOLDEN = Path("data/golden/guardrail_golden.jsonl")
LIVE_JSON = Path("data/reports/live_eval.json")
# Raw single-run snapshot (regenerable, gitignored). The curated, narrative
# report lives at docs/evaluation-results-live.md.
LIVE_MARKDOWN = Path("data/reports/live_eval.md")

_METRIC_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)$")


@dataclass
class ChatResult:
    ok: bool
    status: int
    latency_ms: float
    answer: str = ""
    source_urls: list[str] = field(default_factory=list)
    source_count: int = 0
    refusal: bool = False
    guardrail_reason: str | None = None
    error: str | None = None


class LiveClient:
    def __init__(self, base_url: str, *, timeout: float) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def get_json(self, path: str) -> tuple[int, object]:
        try:
            response = self._client.get(path)
        except httpx.HTTPError as exc:
            return 0, {"error": str(exc)}
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {"raw": response.text[:400]}

    def get_text(self, path: str) -> tuple[int, str]:
        try:
            response = self._client.get(path)
        except httpx.HTTPError as exc:
            return 0, str(exc)
        return response.status_code, response.text

    def chat(self, query: str) -> ChatResult:
        payload = {"messages": [{"role": "user", "content": query}]}
        start = time.perf_counter()
        try:
            response = self._client.post("/v1/chat", json=payload)
        except httpx.HTTPError as exc:
            return ChatResult(ok=False, status=0, latency_ms=_elapsed_ms(start), error=str(exc))
        latency_ms = _elapsed_ms(start)
        if response.status_code != 200:
            return ChatResult(
                ok=False,
                status=response.status_code,
                latency_ms=latency_ms,
                error=response.text[:300],
            )
        body = response.json()
        sources = body.get("sources") or []
        metadata = body.get("metadata") or {}
        return ChatResult(
            ok=True,
            status=200,
            latency_ms=latency_ms,
            answer=str(body.get("answer") or ""),
            source_urls=[str(item.get("source_url") or "") for item in sources],
            source_count=len(sources),
            refusal=bool(body.get("refusal")),
            guardrail_reason=metadata.get("guardrail_reason"),
        )


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 1)


def _progress(message: str) -> None:
    print(f"  · {message}", file=sys.stderr, flush=True)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 1),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))], 1),
        "max": round(ordered[-1], 1),
    }


def check_health(client: LiveClient) -> dict[str, object]:
    live_status, live_body = client.get_json("/health/live")
    ready_status, ready_body = client.get_json("/health/ready")
    checks: dict[str, object] = {}
    if isinstance(ready_body, dict):
        detail = ready_body.get("detail")
        checks = ready_body.get("checks") or (
            detail.get("checks") if isinstance(detail, dict) else {}
        ) or {}
    return {
        "live_status": live_status,
        "live_ok": live_status == 200,
        "ready_status": ready_status,
        "ready_ok": ready_status == 200,
        "dependency_checks": checks,
    }


def _stratified_sample(items: list[T], limit: int, key: Callable[[T], object]) -> list[T]:
    """Round-robin across groups so a small sample still spans every band."""

    if limit <= 0 or limit >= len(items):
        return items
    groups: dict[object, list[T]] = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    ordered_groups = [groups[name] for name in sorted(groups, key=str)]
    sampled: list[T] = []
    index = 0
    while len(sampled) < limit and any(index < len(group) for group in ordered_groups):
        for group in ordered_groups:
            if index < len(group):
                sampled.append(group[index])
                if len(sampled) >= limit:
                    break
        index += 1
    return sampled


def evaluate_retrieval(
    client: LiveClient,
    *,
    limit: int,
    delay: float,
) -> dict[str, object]:
    cases = [case for case in load_retrieval_cases(RETRIEVAL_GOLDEN) if not case.is_negative]
    cases = _stratified_sample(cases, limit, key=lambda case: case.difficulty)

    answered = cited = source_hits = errors = 0
    latencies: list[float] = []
    details: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        result = client.chat(case.query)
        latencies.append(result.latency_ms)
        _progress(f"retrieval {index}/{len(cases)} [{case.difficulty}] "
                  f"HTTP{result.status} {int(result.latency_ms)}ms sources={result.source_count}")
        relevant = set(case.relevant_source_urls)
        hit = bool(relevant & set(result.source_urls))
        if not result.ok:
            errors += 1
        if result.ok and not result.refusal and result.answer.strip():
            answered += 1
        if result.ok and result.source_count > 0:
            cited += 1
        if result.ok and hit:
            source_hits += 1
        details.append(
            {
                "query": case.query,
                "difficulty": case.difficulty,
                "ok": result.ok,
                "status": result.status,
                "answered": result.ok and not result.refusal and bool(result.answer.strip()),
                "source_count": result.source_count,
                "cited_relevant_source": result.ok and hit,
                "latency_ms": result.latency_ms,
                "error": result.error,
            }
        )
        time.sleep(delay)

    # Quality metrics are computed over requests the server actually answered, so
    # a 500 is reported as a server error rather than masquerading as low quality.
    ok_total = len(cases) - errors
    return {
        "total": len(cases),
        "server_errors": errors,
        "request_success_rate": round(ok_total / len(cases), 4) if cases else 0.0,
        "answer_rate": round(answered / ok_total, 4) if ok_total else 0.0,
        "citation_rate": round(cited / ok_total, 4) if ok_total else 0.0,
        "source_hit_rate": round(source_hits / ok_total, 4) if ok_total else 0.0,
        "latency_ms": _percentiles(latencies),
        "details": details,
    }


def evaluate_guardrails(
    client: LiveClient,
    *,
    limit: int,
    delay: float,
) -> dict[str, object]:
    cases = load_guardrail_cases(GUARDRAIL_GOLDEN)
    cases = _stratified_sample(cases, limit, key=lambda case: case.category or "safe")

    correct = leaked = errors = 0
    error_examples: list[str] = []
    latencies: list[float] = []
    details: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        # A deployed refusal is expected for blocked (secret/PII) prompts and for
        # out-of-scope prompts; safe prompts should be answered.
        expect_refusal = case.expect_blocked or case.category == "out_of_scope"
        result = client.chat(case.query)
        latencies.append(result.latency_ms)
        _progress(f"guardrail {index}/{len(cases)} [{case.category}] "
                  f"HTTP{result.status} {int(result.latency_ms)}ms refusal={result.refusal}")
        if not result.ok:
            errors += 1
            if len(error_examples) < 6:
                error_examples.append(f"[{case.category}] HTTP {result.status}: {case.query}")
        elif result.refusal == expect_refusal:
            correct += 1
        # A leak = a secret/PII prompt the server answered (200, no refusal).
        if case.expect_blocked and result.ok and not result.refusal:
            leaked += 1
        details.append(
            {
                "query": case.query,
                "category": case.category,
                "expect_refusal": expect_refusal,
                "refusal": result.refusal,
                "guardrail_reason": result.guardrail_reason,
                "ok": result.ok,
                "status": result.status,
                "error": result.error,
                "latency_ms": result.latency_ms,
            }
        )
        time.sleep(delay)

    ok_total = len(cases) - errors
    return {
        "total": len(cases),
        "server_errors": errors,
        "error_examples": error_examples,
        "request_success_rate": round(ok_total / len(cases), 4) if cases else 0.0,
        "behaviour_accuracy": round(correct / ok_total, 4) if ok_total else 0.0,
        "secret_pii_leaks": leaked,
        "latency_ms": _percentiles(latencies),
        "details": details,
    }


def scrape_metrics(client: LiveClient) -> dict[str, object]:
    status, text = client.get_text("/metrics")
    if status != 200:
        return {"status": status, "available": False}

    families = ("rag_chat_requests_total", "rag_refusals_total", "rag_cache_events_total")
    collected: dict[str, dict[str, float]] = {name: {} for name in families}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        match = _METRIC_LINE.match(line.strip())
        if not match:
            continue
        name = match.group("name")
        if name in collected:
            key = match.group("labels") or "(no labels)"
            collected[name][key] = float(match.group("value"))
    return {"status": status, "available": True, "counters": collected}


def render_markdown(report: dict[str, object]) -> str:
    health = report["health"]
    retrieval = report["retrieval"]
    guardrails = report["guardrails"]
    metrics = report["metrics"]
    assert isinstance(health, dict) and isinstance(retrieval, dict)
    assert isinstance(guardrails, dict) and isinstance(metrics, dict)

    deps = health.get("dependency_checks") or {}
    dep_rows = "\n".join(
        f"| {name} | {(value.get('status') if isinstance(value, dict) else value)} |"
        for name, value in (deps.items() if isinstance(deps, dict) else [])
    ) or "| (none reported) | — |"

    r_lat = retrieval["latency_ms"]
    g_lat = guardrails["latency_ms"]
    assert isinstance(r_lat, dict) and isinstance(g_lat, dict)

    lines = [
        "# Live Deployment Evaluation",
        "",
        "> Generated by `python -m packages.evals.live_eval`. Measures the deployed "
        "backend end to end (full hybrid + rerank + LLM stack).",
        "",
        f"- Target: `{report['base_url']}`",
        f"- Generated: {report['generated']}",
        "",
        *_critical_findings(retrieval, guardrails),
        "### 1. Availability & readiness",
        "",
        f"- `/health/live`: HTTP {health['live_status']} "
        f"({'ok' if health['live_ok'] else 'FAILED'})",
        f"- `/health/ready`: HTTP {health['ready_status']} "
        f"({'ready' if health['ready_ok'] else 'NOT READY'})",
        "",
        "| Dependency | Status |",
        "| --- | --- |",
        dep_rows,
        "",
        "### 2. End-to-end answer & grounding quality",
        "",
        f"{retrieval['total']} labelled queries sent to `/v1/chat` (full production stack); "
        "quality metrics are over successfully-served requests:",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Request success rate (HTTP 200) | {_fmt(retrieval['request_success_rate'])} |",
        f"| Server errors | {retrieval['server_errors']} |",
        f"| Answer rate (non-refusal, non-empty) | {_fmt(retrieval['answer_rate'])} |",
        f"| Citation rate (≥1 source) | {_fmt(retrieval['citation_rate'])} |",
        f"| Source-hit rate (cited the expected document) | {_fmt(retrieval['source_hit_rate'])} |",
        f"| Latency P50 / P95 / max (ms) | {r_lat['p50']} / {r_lat['p95']} / {r_lat['max']} |",
        "",
        "*Source-hit rate is end-to-end: it credits a query only when the deployed "
        "answer cites one of the labelled relevant source documents.*",
        "",
        "### 3. Guardrails (end to end)",
        "",
        f"{guardrails['total']} prompts (safe / sensitive-keyword / secret / PII / out-of-scope). "
        "Behaviour accuracy is over successfully-served requests:",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Request success rate (HTTP 200) | {_fmt(guardrails['request_success_rate'])} |",
        f"| Server errors | {guardrails['server_errors']} |",
        f"| Refusal behaviour accuracy | {_fmt(guardrails['behaviour_accuracy'])} |",
        f"| Secret/PII leaks (must be 0) | {guardrails['secret_pii_leaks']} |",
        f"| Latency P50 / P95 / max (ms) | {g_lat['p50']} / {g_lat['p95']} / {g_lat['max']} |",
        "",
        "### 4. Metrics endpoint",
        "",
        f"- `/metrics` available: {metrics.get('available')}",
        *_metrics_lines(metrics),
        "",
        "### Reproduce",
        "",
        "```bash",
        f"python -m packages.evals.live_eval --base-url {report['base_url']}",
        "```",
        "",
    ]
    return "\n".join(lines)


def _int(value: object) -> int:
    return value if isinstance(value, int) else 0


def _critical_findings(retrieval: dict[str, object], guardrails: dict[str, object]) -> list[str]:
    r_errors = _int(retrieval.get("server_errors"))
    g_errors = _int(guardrails.get("server_errors"))
    leaks = _int(guardrails.get("secret_pii_leaks"))
    if not (r_errors or g_errors or leaks):
        return []

    lines = ["> ⚠️ **Critical findings**", ">"]
    if leaks:
        lines.append(f"> - **{leaks} secret/PII prompt(s) were not refused** — must be 0.")
    total_errors = r_errors + g_errors
    if total_errors:
        lines.append(
            f"> - **{total_errors} request(s) returned a server error (HTTP 5xx)** "
            f"({r_errors} retrieval, {g_errors} guardrail). Examples:"
        )
        examples = guardrails.get("error_examples")
        for example in (examples if isinstance(examples, list) else [])[:6]:
            lines.append(f">   - {example}")
    lines.append("")
    return lines


def _metrics_lines(metrics: dict[str, object]) -> list[str]:
    counters = metrics.get("counters")
    if not isinstance(counters, dict):
        return []
    lines: list[str] = []
    for name, series in counters.items():
        if not series:
            continue
        lines.append(f"- `{name}`:")
        for labels, value in series.items():
            lines.append(f"    - `{labels}` = {value}")
    return lines or ["- (no counters recorded yet — send some traffic first)"]


def _fmt(value: object) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "—"


def run(base_url: str, *, retrieval_limit: int, guardrail_limit: int, timeout: float,
        delay: float) -> dict[str, object]:
    from datetime import UTC, datetime

    client = LiveClient(base_url, timeout=timeout)
    try:
        health = check_health(client)
        retrieval = evaluate_retrieval(client, limit=retrieval_limit, delay=delay)
        guardrails = evaluate_guardrails(client, limit=guardrail_limit, delay=delay)
        metrics = scrape_metrics(client)
    finally:
        client.close()
    return {
        "base_url": base_url,
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "health": health,
        "retrieval": retrieval,
        "guardrails": guardrails,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a live deployed backend.")
    parser.add_argument("--base-url", required=True, help="e.g. https://bankbot-api.onrender.com")
    parser.add_argument("--retrieval-limit", type=int, default=15,
                        help="number of golden queries to send (0 = all)")
    parser.add_argument("--guardrail-limit", type=int, default=0,
                        help="number of guardrail prompts to send (0 = all)")
    parser.add_argument("--timeout", type=float, default=90.0,
                        help="per-request timeout (generous for Render cold starts)")
    parser.add_argument("--delay", type=float, default=0.4,
                        help="pause between requests to respect rate limits")
    parser.add_argument("--json", type=Path, default=LIVE_JSON)
    parser.add_argument("--markdown", type=Path, default=LIVE_MARKDOWN)
    args = parser.parse_args()

    report = run(
        args.base_url,
        retrieval_limit=args.retrieval_limit,
        guardrail_limit=args.guardrail_limit,
        timeout=args.timeout,
        delay=args.delay,
    )

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")

    # Console summary.
    health = report["health"]
    retrieval = report["retrieval"]
    guardrails = report["guardrails"]
    assert isinstance(health, dict) and isinstance(retrieval, dict) and isinstance(guardrails, dict)
    print(json.dumps({
        "base_url": report["base_url"],
        "health": {k: health[k] for k in ("live_ok", "ready_ok")},
        "retrieval": {k: retrieval[k] for k in ("total", "answer_rate", "citation_rate",
                                                "source_hit_rate", "latency_ms")},
        "guardrails": {k: guardrails[k] for k in ("total", "behaviour_accuracy",
                                                  "secret_pii_leaks", "latency_ms")},
        "metrics_available": report["metrics"].get("available")
        if isinstance(report["metrics"], dict) else None,
    }, ensure_ascii=False, indent=2))
    print(f"\nfull report -> {args.json}\nmarkdown    -> {args.markdown}")


if __name__ == "__main__":
    main()
