"""Deterministically build the retrieval golden set from the committed corpus.

Every query is labelled with the *source documents* that answer it, resolved
directly from ``data/chunks/vietcombank_chunks.jsonl`` at build time. Because the
labels are derived from the corpus rather than hand-copied chunk ids, they can
never silently drift when the data is re-crawled or re-chunked — the failure
mode that made the previous golden set 70% stale.

The set mixes four difficulty bands so the report can show an honest curve
rather than a single inflated number:

* ``verbatim``   — the exact FAQ question / a templated product query (easy)
* ``no_accent``  — Vietnamese typed without diacritics (very common in practice)
* ``keyword``    — content words only, mimicking terse search queries
* ``paraphrase`` — curated colloquial rewrites with a real vocabulary gap (hard)

Plus a handful of out-of-scope negatives that should retrieve nothing.

Run with::

    python -m packages.evals.build_golden
"""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from packages.evals.corpus import DEFAULT_CORPUS_PATH

GOLDEN_PATH = Path("data/golden/retrieval_golden.jsonl")

# How many documents to sample per product type for the auto-generated bands.
PRODUCT_DOCS_PER_TYPE = 2
FAQ_DOCS_PER_TYPE = 3
MAX_FAQ_TITLE_CHARS = 140

VIETNAMESE_STOPWORDS = {
    "co", "cua", "de", "gi", "khi", "la", "nao", "nhu", "nhung", "o", "ra",
    "sao", "the", "thi", "toi", "va", "vay", "ve", "voi", "duoc", "cac", "cho",
}

PRODUCT_QUERY_TEMPLATES = {
    "account": "{title} Vietcombank có tiện ích gì?",
    "card": "Thẻ {title} có ưu đãi gì?",
    "digital_banking": "Dịch vụ {title} của Vietcombank là gì?",
    "insurance": "Bảo hiểm {title} có quyền lợi gì?",
    "investment": "{title} của Vietcombank là gì?",
    "loan": "{title} tại Vietcombank có điều kiện và hồ sơ gì?",
    "saving": "{title} của Vietcombank có đặc điểm gì?",
    "transfer": "{title} qua Vietcombank như thế nào?",
}
DEFAULT_PRODUCT_TEMPLATE = "{title} của Vietcombank có đặc điểm gì?"

# Curated hard paraphrases: colloquial user phrasings with a genuine vocabulary
# gap vs. the source title. Each is anchored to an EXACT document title so the
# builder resolves the correct label(s) from the corpus (and fails loudly if the
# anchor ever stops matching).
CURATED_PARAPHRASES: list[tuple[str, str]] = [
    ("Mở thẻ hạng bạch kim Visa của VCB thì được hưởng ưu đãi gì?",
     "Vietcombank Visa Platinum"),
    ("Thẻ hoàn tiền Cashplus của Vietcombank hoàn tiền bao nhiêu?",
     "Vietcombank Cashplus Platinum American Express®"),
    ("Thẻ số DigiCard của VCB phí thường niên ra sao?",
     "VCB DigiCard"),
    ("Thẻ tích dặm bay Vietnam Airlines của VCB cộng dặm thế nào?",
     "Vietcombank Vietnam Airlines American Express®"),
    ("Tôi muốn mua xe hơi trả góp thì VCB cho vay trong bao lâu?",
     "Vay mua ô tô"),
    ("Vay không cần tài sản đảm bảo dựa trên lương ở VCB cần những gì?",
     "Vay tín chấp theo lương"),
    ("Gói vay Nhà Mới Thành Đạt cho vay tối đa mấy năm?",
     "Nhà Mới Thành Đạt"),
    ("Tôi có sổ tiết kiệm muốn cầm cố để vay tiền ở VCB thì sao?",
     "Vay cầm cố giấy tờ có giá"),
    ("Sản phẩm tiết kiệm An Vui của VCB có gì đặc biệt?",
     "Tiền gửi An Vui"),
    ("VCB có kiểu tự động trích tiền để bỏ ống tiết kiệm không?",
     "Tiết kiệm tự động"),
    ("Bảo hiểm sức khỏe cho cả gia đình của VCB quyền lợi tới đâu?",
     "FWD Cả nhà vui khỏe"),
    ("Bảo hiểm lo cho con ăn học FWD Con vươn xa bảo vệ những gì?",
     "FWD Con vươn xa 2.0"),
    ("Đầu tư quỹ mở VCBF tối thiểu cần bao nhiêu tiền?",
     "Quỹ mở - VCBF"),
    ("Mở tài khoản chứng khoán qua VCB có được miễn phí không?",
     "Giao dịch Chứng khoán"),
    ("Tôi nhận tiền từ nước ngoài gửi về thì nên dùng dịch vụ nào của VCB?",
     "Nhận kiều hối tại Việt Nam"),
    ("Gửi tiền cho con đang du học ở nước ngoài qua VCB thế nào?",
     "Chuyển tiền ra nước ngoài"),
    ("Dịch vụ nhắn tin báo biến động số dư của VCB là gì?",
     "SMS Banking"),
    ("Thẻ quẹt không cần chạm của VCB nghĩa là gì?",
     "Thẻ không tiếp xúc là gì?"),
    ("Mã dùng một lần OTP để làm gì khi giao dịch?",
     "One-time-password (OTP) là gì?"),
    ("Tôi xem điểm thưởng tích luỹ Loyalty ở chỗ nào?",
     "Tôi có thể kiểm tra số điểm tích lũy VCB Loyalty tại đâu?"),
]

# Out-of-scope negatives — public but unrelated to Vietcombank products.
NEGATIVES: list[tuple[str, str]] = [
    ("Hôm nay thời tiết Hà Nội thế nào?", "out_of_scope"),
    ("Cho tôi công thức nấu phở bò truyền thống", "out_of_scope"),
    ("Kết quả trận bóng đá tối qua ra sao?", "out_of_scope"),
    ("Dịch giúp tôi câu này sang tiếng Anh", "out_of_scope"),
    ("Giá Bitcoin hôm nay bao nhiêu?", "out_of_scope"),
    ("Thủ đô của nước Pháp là gì?", "out_of_scope"),
]


@dataclass
class Doc:
    source_url: str
    title: str
    section: str | None
    product_type: str | None
    chunk_ids: list[str] = field(default_factory=list)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return stripped.replace("Đ", "D").replace("đ", "d")


def _keyword_query(title: str) -> str:
    ascii_title = _strip_accents(title).lower()
    tokens = [tok for tok in _split_words(ascii_title) if tok not in VIETNAMESE_STOPWORDS]
    return " ".join(tokens)


def _split_words(text: str) -> list[str]:
    return [tok for tok in "".join(ch if ch.isalnum() else " " for ch in text).split() if tok]


def load_docs(corpus_path: Path = DEFAULT_CORPUS_PATH) -> tuple[dict[str, Doc], dict[str, list[str]]]:
    """Return docs keyed by source_url, and an exact-title -> source_urls index."""

    docs: dict[str, Doc] = {}
    title_index: dict[str, list[str]] = defaultdict(list)
    with corpus_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            url = str(payload.get("source_url") or "")
            if not url:
                continue
            doc = docs.get(url)
            if doc is None:
                doc = Doc(
                    source_url=url,
                    title=str(payload.get("title") or ""),
                    section=payload.get("section"),
                    product_type=payload.get("product_type"),
                )
                docs[url] = doc
                title_index[doc.title].append(url)
            doc.chunk_ids.append(str(payload["chunk_id"]))
    return docs, title_index


def _case(
    query: str,
    urls: list[str],
    docs: dict[str, Doc],
    *,
    difficulty: str,
) -> dict[str, object]:
    chunk_ids = [cid for url in urls for cid in docs[url].chunk_ids]
    primary = docs[urls[0]]
    return {
        "query": query,
        "relevant_source_urls": urls,
        "expected_chunk_ids": chunk_ids,
        "section": primary.section,
        "product_type": primary.product_type,
        "difficulty": difficulty,
    }


def _sample_docs(docs: dict[str, Doc], section: str, per_type: int) -> list[Doc]:
    by_type: dict[str | None, list[Doc]] = defaultdict(list)
    for url in sorted(docs):
        doc = docs[url]
        if doc.section != section:
            continue
        by_type[doc.product_type].append(doc)

    sampled: list[Doc] = []
    seen_titles: set[str] = set()
    for product_type in sorted(by_type, key=lambda value: value or ""):
        picked = 0
        for doc in by_type[product_type]:
            if picked >= per_type:
                break
            if not doc.title or doc.title in seen_titles:
                continue
            if section == "faq" and len(doc.title) > MAX_FAQ_TITLE_CHARS:
                continue
            seen_titles.add(doc.title)
            sampled.append(doc)
            picked += 1
    return sampled


def build_cases() -> list[dict[str, object]]:
    docs, title_index = load_docs()
    cases: list[dict[str, object]] = []

    def urls_for_title(title: str) -> list[str]:
        return title_index.get(title, [])

    # Product-detail bands: verbatim (templated) + no-accent.
    for doc in _sample_docs(docs, "product_detail", PRODUCT_DOCS_PER_TYPE):
        template = PRODUCT_QUERY_TEMPLATES.get(doc.product_type or "", DEFAULT_PRODUCT_TEMPLATE)
        query = template.format(title=doc.title)
        urls = urls_for_title(doc.title)
        cases.append(_case(query, urls, docs, difficulty="verbatim"))
        cases.append(_case(_strip_accents(query), urls, docs, difficulty="no_accent"))

    # FAQ bands: verbatim question + no-accent + keyword.
    for doc in _sample_docs(docs, "faq", FAQ_DOCS_PER_TYPE):
        urls = urls_for_title(doc.title)
        cases.append(_case(doc.title, urls, docs, difficulty="verbatim"))
        cases.append(_case(_strip_accents(doc.title), urls, docs, difficulty="no_accent"))
        keyword = _keyword_query(doc.title)
        if len(keyword.split()) >= 3:
            cases.append(_case(keyword, urls, docs, difficulty="keyword"))

    # Curated hard paraphrases, anchored to exact document titles.
    for query, anchor_title in CURATED_PARAPHRASES:
        urls = urls_for_title(anchor_title)
        if not urls:
            raise SystemExit(f"paraphrase anchor not found in corpus: {anchor_title!r}")
        cases.append(_case(query, urls, docs, difficulty="paraphrase"))

    # Out-of-scope negatives.
    for query, category in NEGATIVES:
        cases.append(
            {
                "query": query,
                "relevant_source_urls": [],
                "expected_chunk_ids": [],
                "section": None,
                "product_type": None,
                "difficulty": None,
                "category": category,
            }
        )

    return cases


def main() -> None:
    cases = build_cases()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_PATH.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")

    by_difficulty: dict[str, int] = defaultdict(int)
    for case in cases:
        by_difficulty[str(case.get("difficulty"))] += 1
    print(f"wrote {len(cases)} cases to {GOLDEN_PATH}")
    print("by difficulty:", dict(sorted(by_difficulty.items())))


if __name__ == "__main__":
    main()
