from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from apps.api.app.rag.security_intent import classify_security_intent

PII_PATTERNS = [
    re.compile(r"\b\d{9,12}\b"),
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
]

SECRET_KEYWORDS = {
    "otp",
    "mã otp",
    "mat khau",
    "mật khẩu",
    "password",
    "pin",
    "cvv",
    "so the",
    "số thẻ",
}

SECRET_DISCLOSURE_MARKERS = {
    "cua toi la",
    "cua minh la",
    "day la",
    "mat khau cua toi",
    "otp cua toi",
    "pin cua toi",
    "cvv cua toi",
    "so the cua toi",
    "password cua toi",
    "otp la",
    "mat khau la",
    "password la",
    "pin la",
    "cvv la",
    "so the la",
}

PASSWORD_DISCLOSURE_MARKERS = {
    "cua minh la",
    "cua toi la",
    "day la",
    "mat khau cua toi",
    "mat khau la",
    "password cua toi",
    "password la",
}

SECRET_DISCLOSURE_FILLERS = {"la"}

PUBLIC_SECRET_VALUE_PREFIXES = {
    "bao",
    "cach",
    "can",
    "co",
    "dich",
    "duoc",
    "gi",
    "ho",
    "khac",
    "khi",
    "khong",
    "lam",
    "ma",
    "nao",
    "nen",
    "neu",
    "ngan",
    "nhu",
    "phai",
    "phuong",
    "quy",
    "sao",
    "su",
    "tai",
    "the",
    "thong",
    "toi",
    "trong",
    "vi",
}

SECRET_VALUE_TOKEN_PATTERN = re.compile(r"[a-z0-9_@#$%^&*+=.!-]+")

SUPPORTED_BANK_KEYWORDS = {
    "vietcombank",
    "vcb",
    "ngân hàng",
    "ngan hang",
    "sản phẩm",
    "san pham",
    "tài khoản",
    "tai khoan",
    "đăng nhập",
    "dang nhap",
    "mật khẩu",
    "mat khau",
    "password",
    "otp",
    "pin",
    "mã pin",
    "ma pin",
    "rút tiền",
    "rut tien",
    "khóa truy cập",
    "khoa truy cap",
    "quên mật khẩu",
    "quen mat khau",
    "tiền gửi",
    "tien gui",
    "tiết kiệm",
    "tiet kiem",
    "vay",
    "thẻ",
    "the",
    "bảo hiểm",
    "bao hiem",
    "fwd",
    "con vườn xa",
    "con vuon xa",
    "chuyển tiền",
    "chuyen tien",
    "kiều hối",
    "kieu hoi",
    "digibank",
    "loyalty",
    "quỹ",
    "quy",
    "đầu tư",
    "dau tu",
    "chứng khoán",
    "chung khoan",
    "lãi suất",
    "lai suat",
    "biểu phí",
    "bieu phi",
    "hồ sơ",
    "ho so",
    "điều kiện",
    "dieu kien",
}


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    safe_response: str | None = None


def inspect_query(query: str) -> GuardrailResult:
    normalized = _normalize_text(query)
    if not normalized:
        return GuardrailResult(
            allowed=False,
            reason="empty_query",
            safe_response="Bạn vui lòng nhập câu hỏi cần tra cứu.",
        )

    if _contains_exposed_secret(normalized):
        return GuardrailResult(
            allowed=False,
            reason="credential_or_secret",
            safe_response=(
                "Tôi không thể tiếp nhận hoặc xử lý mật khẩu, OTP, PIN, CVV, số thẻ "
                "hoặc thông tin đăng nhập. Bạn vui lòng không gửi dữ liệu nhạy cảm."
            ),
        )

    if any(pattern.search(normalized) for pattern in PII_PATTERNS):
        return GuardrailResult(
            allowed=False,
            reason="possible_pii",
            safe_response=(
                "Tôi không thể xử lý thông tin định danh hoặc số tài khoản cá nhân. "
                "Nếu cần hỗ trợ tài khoản, bạn nên liên hệ kênh chính thức của Vietcombank."
            ),
        )

    return GuardrailResult(allowed=True)


def _contains_exposed_secret(normalized: str) -> bool:
    if any(pattern.search(normalized) for pattern in PII_PATTERNS):
        return True
    if not any(keyword in normalized for keyword in SECRET_KEYWORDS):
        return False
    return any(_marker_has_secret_value(normalized, marker) for marker in SECRET_DISCLOSURE_MARKERS)


def _marker_has_secret_value(normalized: str, marker: str) -> bool:
    allow_plain_password = marker in PASSWORD_DISCLOSURE_MARKERS
    start = normalized.find(marker)
    while start != -1:
        tail = normalized[start + len(marker) :]
        if _tail_starts_with_secret_value(tail, allow_plain_password=allow_plain_password):
            return True
        start = normalized.find(marker, start + 1)
    return False


def _tail_starts_with_secret_value(tail: str, *, allow_plain_password: bool) -> bool:
    tokens = SECRET_VALUE_TOKEN_PATTERN.findall(tail)
    while tokens and tokens[0] in SECRET_DISCLOSURE_FILLERS:
        tokens.pop(0)
    if not tokens:
        return False

    first_token = tokens[0]
    if first_token in PUBLIC_SECRET_VALUE_PREFIXES:
        return any(_is_concrete_secret_value(token) for token in tokens[1:4])

    if _is_concrete_secret_value(first_token):
        return True
    return allow_plain_password and len(first_token) >= 4 and len(tokens) <= 3


def _is_concrete_secret_value(token: str) -> bool:
    if token.isdigit():
        return 3 <= len(token) <= 8
    if len(token) < 4:
        return False
    return any(character.isdigit() for character in token) or any(
        character in token for character in "@#$%^&*+=.!-"
    )


def is_likely_supported_domain(query: str) -> bool:
    normalized = _normalize_text(query)
    return classify_security_intent(query).is_security_related or any(
        keyword in normalized for keyword in SUPPORTED_BANK_KEYWORDS
    )


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("-", " "))
    no_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    no_accents = no_accents.replace("Đ", "D").replace("đ", "d")
    return " ".join(no_accents.casefold().split())
