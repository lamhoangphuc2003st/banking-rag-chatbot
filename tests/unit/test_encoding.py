from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXT_ROOTS = [
    PROJECT_ROOT / "apps",
    PROJECT_ROOT / "packages",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / "data" / "golden",
]
ROOT_TEXT_FILES = [
    PROJECT_ROOT / ".dockerignore",
    PROJECT_ROOT / ".editorconfig",
    PROJECT_ROOT / ".env.example",
    PROJECT_ROOT / ".gitattributes",
    PROJECT_ROOT / ".gitignore",
    PROJECT_ROOT / "docker-compose.yml",
    PROJECT_ROOT / "Makefile",
    PROJECT_ROOT / "pyproject.toml",
    PROJECT_ROOT / "README.md",
]
TEXT_SUFFIXES = {
    ".css",
    ".ini",
    ".json",
    ".jsonl",
    ".mako",
    ".md",
    ".mjs",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
MOJIBAKE_MARKERS = (
    "\u00c3",
    "\u00c4",
    "\u00c6",
    "\u00d0",
    "\u00e1\u00ba",
    "\u00e1\u00bb",
)


def test_project_text_files_are_utf8_without_mojibake() -> None:
    offenders: list[str] = []

    for path in _iter_project_text_files():
        content = path.read_text(encoding="utf-8")
        if any(marker in content for marker in MOJIBAKE_MARKERS):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def _iter_project_text_files():
    for path in ROOT_TEXT_FILES:
        if path.exists():
            yield path

    for root in TEXT_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                yield path
