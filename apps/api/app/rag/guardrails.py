from __future__ import annotations

import re
from dataclasses import dataclass


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

SUPPORTED_BANK_KEYWORDS = {
    "vietcombank",
    "vcb",
    "ngân hàng",
    "ngan hang",
    "vay",
    "thẻ",
    "the",
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
    normalized = query.strip().lower()
    if not normalized:
        return GuardrailResult(
            allowed=False,
            reason="empty_query",
            safe_response="Bạn vui lòng nhập câu hỏi cần tra cứu.",
        )

    if any(keyword in normalized for keyword in SECRET_KEYWORDS):
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


def is_likely_supported_domain(query: str) -> bool:
    normalized = query.lower()
    return any(keyword in normalized for keyword in SUPPORTED_BANK_KEYWORDS)
