from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

SecurityIntentKind = Literal["none", "public_info", "account_recovery"]

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

SECURITY_TOPIC_MARKERS = (
    "an toan",
    "bao mat",
    "dang nhap",
    "khoa truy cap",
    "ma pin",
    "ma xac nhan",
    "mat khau",
    "one time password",
    "otp",
    "password",
    "pin",
    "rut tien",
    "smart otp",
    "vcb digibank",
    "xac thuc",
)

RECOVERY_ACTION_MARKERS = (
    "cap lai",
    "dat lai",
    "doi lai",
    "khong nho",
    "khoi phuc",
    "lay lai",
    "mat ma",
    "mo khoa",
    "quen",
    "reset",
)

PUBLIC_INFO_MARKERS = (
    "bao lau",
    "bi khoa khong",
    "can dat",
    "cac phuong thuc",
    "co bi khoa",
    "co khoa",
    "duoc dat",
    "duoc khong",
    "la gi",
    "nhap sai",
    "nen dat",
    "nen doi",
    "nhu the nao",
    "phan biet",
    "phai doi",
    "phuong thuc",
    "so sanh",
    "su dung",
)


@dataclass(frozen=True)
class SecurityIntent:
    kind: SecurityIntentKind
    normalized_query: str

    @property
    def is_security_related(self) -> bool:
        return self.kind != "none"

    @property
    def is_public_info(self) -> bool:
        return self.kind == "public_info"

    @property
    def is_account_recovery(self) -> bool:
        return self.kind == "account_recovery"


def classify_security_intent(query: str) -> SecurityIntent:
    normalized = normalize_security_text(query)
    if not _contains_any(normalized, SECURITY_TOPIC_MARKERS):
        return SecurityIntent(kind="none", normalized_query=normalized)

    if _contains_any(normalized, PUBLIC_INFO_MARKERS):
        return SecurityIntent(kind="public_info", normalized_query=normalized)

    if _contains_any(normalized, RECOVERY_ACTION_MARKERS):
        return SecurityIntent(kind="account_recovery", normalized_query=normalized)

    return SecurityIntent(kind="public_info", normalized_query=normalized)


def normalize_security_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("-", " "))
    no_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    no_accents = no_accents.replace("Đ", "D").replace("đ", "d")
    return " ".join(TOKEN_PATTERN.findall(no_accents.casefold()))


def _contains_any(normalized_query: str, markers: tuple[str, ...]) -> bool:
    return any(_has_phrase(normalized_query, marker) for marker in markers)


def _has_phrase(normalized_query: str, phrase: str) -> bool:
    return f" {phrase} " in f" {normalized_query} "
