from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
NORMALIZED_DIR = DATA_DIR / "normalized"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "indexes"
REPORTS_DIR = DATA_DIR / "reports"


def ensure_data_dirs() -> None:
    for path in [RAW_DIR, NORMALIZED_DIR, CHUNKS_DIR, INDEX_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
