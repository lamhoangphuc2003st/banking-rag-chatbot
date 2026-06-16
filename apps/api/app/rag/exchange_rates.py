from __future__ import annotations

import asyncio
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import httpx

from apps.api.app.models.chat import SourceCitation

EXCHANGE_RATE_PAGE_URL = "https://www.vietcombank.com.vn/vi-VN/KHCN/Cong-cu-Tien-ich/Ty-gia"
EXCHANGE_RATE_XML_URL = "https://portal.vietcombank.com.vn/Usercontrols/TVPortal.TyGia/pXML.aspx"
EXCHANGE_RATE_CACHE_SECONDS = 300.0

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
NUMBER_PATTERN = re.compile(r"\b\d[\d.,]*\b")

EXCHANGE_RATE_MARKERS = (
    "bang bao nhieu",
    "chuyen doi ngoai te",
    "doi ngoai te",
    "doi ra tien viet",
    "doi sang vnd",
    "doi tien",
    "gia ban",
    "gia mua",
    "ngoai te",
    "quy doi",
    "ty gia",
)

CURRENCY_QUOTE_MARKERS = (
    "bao nhieu",
    "gia",
    "gia hom nay",
    "hom nay",
)

NON_EXCHANGE_MARKERS = (
    "bieu phi",
    "lai suat",
    "phi",
    "tiet kiem",
)

CONVERSION_MARKERS = (
    "bang bao nhieu",
    "doi",
    "doi ra",
    "doi sang",
    "mua",
    "quy doi",
    "sang vnd",
)

CURRENCY_ALIASES = {
    "AUD": ("aud", "australian dollar", "do uc"),
    "CAD": ("cad", "canadian dollar", "do canada"),
    "CHF": ("chf", "franc thuy si", "swiss franc"),
    "CNY": ("cny", "nhan dan te", "yuan"),
    "EUR": ("eur", "euro"),
    "GBP": ("bang anh", "gbp", "pound sterling"),
    "HKD": ("hkd", "hongkong dollar"),
    "JPY": ("jpy", "yen", "yen nhat"),
    "KRW": ("krw", "won", "won han"),
    "SGD": ("do singapore", "sgd", "singapore dollar"),
    "THB": ("baht", "thb"),
    "USD": ("do", "do la", "dollar", "usd", "us dollar"),
}

DISPLAY_CODES = ("USD", "EUR", "JPY", "GBP", "AUD", "SGD", "CNY")


@dataclass(frozen=True)
class ExchangeRateQuote:
    code: str
    name: str
    buy_cash: Decimal | None
    buy_cash_display: str
    transfer: Decimal | None
    transfer_display: str
    sell: Decimal | None
    sell_display: str


@dataclass(frozen=True)
class ExchangeRateTable:
    updated_at: str
    source: str
    quotes: tuple[ExchangeRateQuote, ...]

    def quote_for(self, code: str) -> ExchangeRateQuote | None:
        normalized_code = code.upper()
        for quote in self.quotes:
            if quote.code == normalized_code:
                return quote
        return None


@dataclass(frozen=True)
class CurrencyAmount:
    code: str
    amount: Decimal


@dataclass(frozen=True)
class ExchangeRateIntent:
    currency_codes: tuple[str, ...] = ()
    amount: Decimal | None = None
    amount_currency: str | None = None
    foreign_amounts: tuple[CurrencyAmount, ...] = ()
    requested_rate: str | None = None

    @property
    def currency_code(self) -> str | None:
        return self.currency_codes[0] if self.currency_codes else None


@dataclass(frozen=True)
class ExchangeRateAnswer:
    answer: str
    sources: list[SourceCitation]
    metadata: dict[str, object]


class ExchangeRateService:
    def __init__(
        self,
        *,
        xml_url: str = EXCHANGE_RATE_XML_URL,
        page_url: str = EXCHANGE_RATE_PAGE_URL,
        user_agent: str = "BankChatbotResearch/0.1",
        cache_seconds: float = EXCHANGE_RATE_CACHE_SECONDS,
        xml_fetcher: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        self.xml_url = xml_url
        self.page_url = page_url
        self.user_agent = user_agent
        self.cache_seconds = cache_seconds
        self._xml_fetcher = xml_fetcher
        self._cached_table: ExchangeRateTable | None = None
        self._cache_expires_at = 0.0
        self._cache_lock = asyncio.Lock()

    async def answer_query(self, query: str) -> ExchangeRateAnswer | None:
        intent = parse_exchange_rate_intent(query)
        if intent is None:
            return None

        try:
            table = await self.get_table()
        except Exception:
            return ExchangeRateAnswer(
                answer=(
                    "Tôi chưa lấy được bảng tỷ giá trực tuyến của Vietcombank lúc này. "
                    f"Bạn có thể kiểm tra trực tiếp tại {self.page_url}."
                ),
                sources=[_exchange_rate_source(self.page_url)],
                metadata={
                    "retrieval_route": "live_exchange_rates_error",
                    "tool": "vietcombank_exchange_rates",
                    "source_url": self.page_url,
                },
            )

        return build_exchange_rate_answer(intent, table, page_url=self.page_url)

    async def get_table(self) -> ExchangeRateTable:
        now = time.monotonic()
        if self._cached_table is not None and now < self._cache_expires_at:
            return self._cached_table

        async with self._cache_lock:
            now = time.monotonic()
            if self._cached_table is not None and now < self._cache_expires_at:
                return self._cached_table

            xml_text = await self._fetch_xml()
            table = parse_exchange_rate_xml(xml_text)
            self._cached_table = table
            self._cache_expires_at = now + self.cache_seconds
            return table

    async def _fetch_xml(self) -> str:
        if self._xml_fetcher is not None:
            return await self._xml_fetcher()

        async with httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            timeout=httpx.Timeout(10.0),
            follow_redirects=True,
        ) as client:
            response = await client.get(self.xml_url)
            response.raise_for_status()
            return response.text


def parse_exchange_rate_intent(query: str) -> ExchangeRateIntent | None:
    normalized = _normalize_text(query)
    if not normalized:
        return None

    currency_codes = _currency_codes_from_query(normalized)
    has_exchange_marker = _contains_any(normalized, EXCHANGE_RATE_MARKERS)
    has_conversion_marker = _contains_any(normalized, CONVERSION_MARKERS)
    has_quote_marker = _contains_any(normalized, CURRENCY_QUOTE_MARKERS)
    mentions_vnd = _has_phrase(normalized, "vnd") or _has_phrase(normalized, "tien viet")

    if not has_exchange_marker and _contains_any(normalized, NON_EXCHANGE_MARKERS):
        return None

    if not has_exchange_marker and not (
        currency_codes and (mentions_vnd or has_conversion_marker or has_quote_marker)
    ):
        return None

    amount_text = _normalize_amount_text(query)
    foreign_amounts = _foreign_amounts_from_query(amount_text, currency_codes)
    foreign_amount = foreign_amounts[0].amount if foreign_amounts else None
    vnd_amount = None if foreign_amount is not None else _vnd_amount_from_query(query)

    return ExchangeRateIntent(
        currency_codes=currency_codes,
        amount=foreign_amount or vnd_amount,
        amount_currency="foreign" if foreign_amount is not None else ("VND" if vnd_amount is not None else None),
        foreign_amounts=foreign_amounts,
        requested_rate=_requested_rate(normalized),
    )


def parse_exchange_rate_xml(xml_text: str) -> ExchangeRateTable:
    root = ET.fromstring(xml_text.strip())
    updated_at = (root.findtext("DateTime") or "").strip()
    source = (root.findtext("Source") or "").strip()
    quotes: list[ExchangeRateQuote] = []

    for element in root.findall("Exrate"):
        code = str(element.attrib.get("CurrencyCode") or "").strip().upper()
        name = str(element.attrib.get("CurrencyName") or "").strip()
        if not code:
            continue

        buy = str(element.attrib.get("Buy") or "-").strip()
        transfer = str(element.attrib.get("Transfer") or "-").strip()
        sell = str(element.attrib.get("Sell") or "-").strip()
        quotes.append(
            ExchangeRateQuote(
                code=code,
                name=name,
                buy_cash=_parse_rate_value(buy),
                buy_cash_display=buy,
                transfer=_parse_rate_value(transfer),
                transfer_display=transfer,
                sell=_parse_rate_value(sell),
                sell_display=sell,
            )
        )

    return ExchangeRateTable(updated_at=updated_at, source=source, quotes=tuple(quotes))


def build_exchange_rate_answer(
    intent: ExchangeRateIntent,
    table: ExchangeRateTable,
    *,
    page_url: str,
) -> ExchangeRateAnswer:
    source = _exchange_rate_source(page_url)
    if intent.currency_codes:
        quotes = [
            quote
            for code in intent.currency_codes
            if (quote := table.quote_for(code)) is not None
        ]
        missing_codes = [code for code in intent.currency_codes if table.quote_for(code) is None]
        if not quotes:
            return ExchangeRateAnswer(
                answer=(
                    f"Tôi chưa thấy các mã ngoại tệ {', '.join(intent.currency_codes)} trong bảng tỷ giá Vietcombank "
                    f"cập nhật lúc {table.updated_at or 'không rõ thời điểm'}."
                ),
                sources=[source],
                metadata=_exchange_rate_metadata(table, intent, route="live_exchange_rates"),
            )
        if len(quotes) == 1:
            answer = _single_quote_answer(quotes[0], intent, table)
        else:
            answer = _multi_quote_answer(quotes, intent=intent, missing_codes=missing_codes, table=table)
    else:
        answer = _overview_answer(table)

    return ExchangeRateAnswer(
        answer=answer,
        sources=[source],
        metadata=_exchange_rate_metadata(table, intent, route="live_exchange_rates"),
    )


def _single_quote_answer(
    quote: ExchangeRateQuote,
    intent: ExchangeRateIntent,
    table: ExchangeRateTable,
) -> str:
    lines = [
        f"Tỷ giá {quote.code} ({quote.name}) tại Vietcombank cập nhật lúc {table.updated_at}:",
        f"- Mua tiền mặt: {quote.buy_cash_display} VND",
        f"- Mua chuyển khoản: {quote.transfer_display} VND",
        f"- Bán: {quote.sell_display} VND",
    ]

    if intent.amount is not None:
        conversion_lines = _conversion_lines(quote, intent)
        if conversion_lines:
            lines.extend(["", *conversion_lines])

    lines.append("")
    lines.append(
        "Lưu ý: tỷ giá chỉ mang tính tham khảo; giao dịch thực tế theo thời điểm và kênh giao dịch của Vietcombank."
    )
    return "\n".join(lines)


def _multi_quote_answer(
    quotes: list[ExchangeRateQuote],
    *,
    intent: ExchangeRateIntent,
    missing_codes: list[str],
    table: ExchangeRateTable,
) -> str:
    lines = [f"Tỷ giá Vietcombank cập nhật lúc {table.updated_at}:"]
    for quote in quotes:
        lines.extend(
            [
                f"- {quote.code} ({quote.name}):",
                f"  + Mua tiền mặt: {quote.buy_cash_display} VND",
                f"  + Mua chuyển khoản: {quote.transfer_display} VND",
                f"  + Bán: {quote.sell_display} VND",
            ]
        )
    if intent.amount is not None and intent.amount_currency == "VND":
        lines.append("")
        lines.append(f"Quy đổi tham khảo với {_format_vnd(intent.amount)} VND:")
        for quote in quotes:
            if quote.sell is None:
                lines.append(f"- {quote.code}: chưa có tỷ giá bán để quy đổi.")
                continue
            foreign_amount = intent.amount / quote.sell
            lines.append(
                f"- Mua {quote.code} theo tỷ giá bán {quote.sell_display} VND/{quote.code}: "
                f"khoảng {_format_foreign_amount(foreign_amount)} {quote.code}"
            )
    elif intent.foreign_amounts:
        conversion_lines = _multi_foreign_to_vnd_conversion_lines(quotes, intent)
        if conversion_lines:
            lines.extend(["", *conversion_lines])
    if missing_codes:
        lines.append(f"- Chưa thấy mã {', '.join(missing_codes)} trong bảng tỷ giá hiện tại.")
    lines.append("")
    lines.append(
        "Lưu ý: tỷ giá chỉ mang tính tham khảo; giao dịch thực tế theo thời điểm và kênh giao dịch của Vietcombank."
    )
    return "\n".join(lines)


def _overview_answer(table: ExchangeRateTable) -> str:
    quotes = [quote for code in DISPLAY_CODES if (quote := table.quote_for(code)) is not None]
    if not quotes:
        quotes = list(table.quotes[:8])

    lines = [f"Một số tỷ giá Vietcombank cập nhật lúc {table.updated_at}:"]
    lines.extend(
        f"- {quote.code}: mua tiền mặt {quote.buy_cash_display}, "
        f"mua chuyển khoản {quote.transfer_display}, bán {quote.sell_display} VND"
        for quote in quotes
    )
    lines.append("")
    lines.append("Bạn có thể hỏi cụ thể như: 'tỷ giá USD', '100 USD đổi sang VND', hoặc 'giá bán EUR'.")
    return "\n".join(lines)


def _conversion_lines(quote: ExchangeRateQuote, intent: ExchangeRateIntent) -> list[str]:
    if intent.amount is None:
        return []

    if intent.amount_currency == "VND":
        return _vnd_to_foreign_conversion_lines(quote, intent.amount)

    amount = _foreign_amount_for_code(intent, quote.code) or intent.amount
    amount_display = _format_decimal(amount)
    if intent.requested_rate == "sell":
        return _single_conversion_line(
            amount=amount,
            amount_display=amount_display,
            code=quote.code,
            label="Nếu khách hàng mua ngoại tệ từ Vietcombank theo tỷ giá bán",
            rate=quote.sell,
        )
    if intent.requested_rate == "buy_cash":
        return _single_conversion_line(
            amount=amount,
            amount_display=amount_display,
            code=quote.code,
            label="Nếu khách hàng bán tiền mặt cho Vietcombank",
            rate=quote.buy_cash,
        )
    if intent.requested_rate == "transfer":
        return _single_conversion_line(
            amount=amount,
            amount_display=amount_display,
            code=quote.code,
            label="Nếu khách hàng bán chuyển khoản cho Vietcombank",
            rate=quote.transfer,
        )

    lines: list[str] = [f"Quy đổi tham khảo cho {amount_display} {quote.code}:"]
    if quote.buy_cash is not None:
        lines.append(
            f"- Khách hàng bán tiền mặt: khoảng {_format_vnd(amount * quote.buy_cash)} VND"
        )
    if quote.transfer is not None:
        lines.append(
            f"- Khách hàng bán chuyển khoản: khoảng {_format_vnd(amount * quote.transfer)} VND"
        )
    return lines


def _vnd_to_foreign_conversion_lines(quote: ExchangeRateQuote, vnd_amount: Decimal) -> list[str]:
    if quote.sell is None:
        return [f"Chưa có tỷ giá bán để quy đổi {_format_vnd(vnd_amount)} VND sang {quote.code}."]

    foreign_amount = vnd_amount / quote.sell
    return [
        (
            f"Với {_format_vnd(vnd_amount)} VND, nếu khách hàng mua {quote.code} từ Vietcombank "
            f"theo tỷ giá bán {quote.sell_display} VND/{quote.code}, số tiền nhận tham khảo khoảng "
            f"{_format_foreign_amount(foreign_amount)} {quote.code}."
        )
    ]


def _multi_foreign_to_vnd_conversion_lines(
    quotes: list[ExchangeRateQuote],
    intent: ExchangeRateIntent,
) -> list[str]:
    amount_by_code = {currency_amount.code: currency_amount.amount for currency_amount in intent.foreign_amounts}
    quotes_with_amounts = [quote for quote in quotes if quote.code in amount_by_code]
    if not quotes_with_amounts:
        return []

    if intent.requested_rate == "transfer":
        return _foreign_to_vnd_total_lines(
            quotes_with_amounts,
            amount_by_code=amount_by_code,
            rate_name="mua chuyển khoản",
            rate_attr="transfer",
        )
    if intent.requested_rate == "buy_cash":
        return _foreign_to_vnd_total_lines(
            quotes_with_amounts,
            amount_by_code=amount_by_code,
            rate_name="mua tiền mặt",
            rate_attr="buy_cash",
        )

    lines = ["Quy đổi tham khảo sang VND:"]
    cash_lines = _foreign_to_vnd_total_lines(
        quotes_with_amounts,
        amount_by_code=amount_by_code,
        rate_name="mua tiền mặt",
        rate_attr="buy_cash",
    )
    transfer_lines = _foreign_to_vnd_total_lines(
        quotes_with_amounts,
        amount_by_code=amount_by_code,
        rate_name="mua chuyển khoản",
        rate_attr="transfer",
    )
    lines.extend(f"- {line}" for line in cash_lines)
    lines.extend(f"- {line}" for line in transfer_lines)
    return lines


def _foreign_to_vnd_total_lines(
    quotes: list[ExchangeRateQuote],
    *,
    amount_by_code: dict[str, Decimal],
    rate_name: str,
    rate_attr: str,
) -> list[str]:
    parts: list[str] = []
    total = Decimal("0")
    for quote in quotes:
        amount = amount_by_code[quote.code]
        rate = getattr(quote, rate_attr)
        if rate is None:
            parts.append(f"{_format_decimal(amount)} {quote.code}: chưa có tỷ giá {rate_name}")
            continue
        converted = amount * rate
        total += converted
        parts.append(
            f"{_format_decimal(amount)} {quote.code} x {_rate_display_for_attr(quote, rate_attr)} = {_format_vnd(converted)} VND"
        )

    if total <= 0:
        return [f"Không có đủ tỷ giá {rate_name} để quy đổi các ngoại tệ đã nêu."]
    return [f"Theo tỷ giá {rate_name}: {'; '.join(parts)}. Tổng khoảng {_format_vnd(total)} VND."]


def _foreign_amount_for_code(intent: ExchangeRateIntent, code: str) -> Decimal | None:
    for currency_amount in intent.foreign_amounts:
        if currency_amount.code == code:
            return currency_amount.amount
    return None


def _rate_display_for_attr(quote: ExchangeRateQuote, attr: str) -> str:
    return {
        "buy_cash": quote.buy_cash_display,
        "transfer": quote.transfer_display,
        "sell": quote.sell_display,
    }[attr]


def _single_conversion_line(
    *,
    amount: Decimal,
    amount_display: str,
    code: str,
    label: str,
    rate: Decimal | None,
) -> list[str]:
    if rate is None:
        return [f"Chưa có tỷ giá phù hợp để quy đổi {amount_display} {code} theo yêu cầu này."]
    return [f"{label}: {amount_display} {code} tương đương khoảng {_format_vnd(amount * rate)} VND."]


def _exchange_rate_metadata(
    table: ExchangeRateTable,
    intent: ExchangeRateIntent,
    *,
    route: str,
) -> dict[str, object]:
    return {
        "retrieval_route": route,
        "retrieved_count": 1,
        "reranked_count": 1,
        "tool": "vietcombank_exchange_rates",
        "exchange_rate_updated_at": table.updated_at,
        "exchange_rate_currency": intent.currency_code,
        "exchange_rate_currencies": list(intent.currency_codes),
        "exchange_rate_amount": str(intent.amount) if intent.amount is not None else None,
        "exchange_rate_amount_currency": intent.amount_currency,
        "exchange_rate_foreign_amounts": {
            currency_amount.code: str(currency_amount.amount)
            for currency_amount in intent.foreign_amounts
        },
        "exchange_rate_requested_rate": intent.requested_rate,
    }


def _exchange_rate_source(page_url: str) -> SourceCitation:
    return SourceCitation(
        chunk_id="live:vietcombank:exchange-rates",
        title="Tỷ giá ngoại tệ Vietcombank",
        source_url=page_url,
        section="exchange_rate",
        score=1.0,
    )


def _currency_codes_from_query(normalized: str) -> tuple[str, ...]:
    matches: list[tuple[int, int, int, str]] = []
    for code, aliases in CURRENCY_ALIASES.items():
        for alias in aliases:
            pattern = re.compile(rf"(?:^|(?<=\s)){re.escape(alias)}(?=\s|$)")
            for match in pattern.finditer(normalized):
                matches.append((match.start(), match.end(), -len(alias), code))

    seen: set[str] = set()
    occupied_spans: list[tuple[int, int]] = []
    codes: list[str] = []
    for start, end, _, code in sorted(matches, key=lambda item: (item[0], item[2], item[1], item[3])):
        if any(start < occupied_end and occupied_start < end for occupied_start, occupied_end in occupied_spans):
            continue
        if code in seen:
            continue
        seen.add(code)
        occupied_spans.append((start, end))
        codes.append(code)
    return tuple(codes)


def _foreign_amounts_from_query(
    normalized: str,
    currency_codes: tuple[str, ...],
) -> tuple[CurrencyAmount, ...]:
    amounts: list[CurrencyAmount] = []
    for currency_code in currency_codes:
        code_aliases = CURRENCY_ALIASES.get(currency_code, (currency_code.lower(),))
        for alias in code_aliases:
            match = re.search(rf"(?P<amount>\d[\d.,]*)\s+{re.escape(alias)}\b", normalized)
            if not match:
                continue
            amount = _parse_user_amount(match.group("amount"))
            if amount is None:
                continue
            amounts.append(CurrencyAmount(code=currency_code, amount=amount))
            break
    return tuple(amounts)


def _vnd_amount_from_query(query: str) -> Decimal | None:
    normalized = _normalize_amount_text(query)
    patterns = (
        r"(?P<amount>\d+(?:[.,]\d+)?)\s*(?:tr|trieu)\s*(?:vnd|vnđ|dong|d)?\b",
        r"(?P<amount>\d[\d.,]*)\s*(?:vnd|vnđ|dong)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        amount = _parse_user_amount(match.group("amount"))
        if amount is None:
            continue
        if "tr" in match.group(0) or "trieu" in match.group(0):
            return amount * Decimal("1000000")
        return amount
    return None


def _requested_rate(normalized: str) -> str | None:
    if _has_phrase(normalized, "mua tien mat") or _has_phrase(normalized, "tien mat"):
        return "buy_cash"
    if _has_phrase(normalized, "mua chuyen khoan") or _has_phrase(normalized, "chuyen khoan"):
        return "transfer"
    if _has_phrase(normalized, "gia ban") or _has_phrase(normalized, "ban ra") or _has_phrase(
        normalized, "ngan hang ban"
    ):
        return "sell"
    return None


def _parse_rate_value(value: str) -> Decimal | None:
    if not value or value == "-":
        return None
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def _parse_user_amount(value: str) -> Decimal | None:
    normalized = value.strip()
    if not normalized:
        return None

    if "," in normalized and "." in normalized:
        if normalized.rfind(",") > normalized.rfind("."):
            normalized = normalized.replace(".", "").replace(",", ".")
        else:
            normalized = normalized.replace(",", "")
    elif normalized.count(",") > 1:
        normalized = normalized.replace(",", "")
    elif normalized.count(".") > 1:
        normalized = normalized.replace(".", "")
    elif "," in normalized:
        left, _, right = normalized.rpartition(",")
        normalized = f"{left}{right}" if len(right) == 3 else f"{left}.{right}"
    elif "." in normalized:
        left, _, right = normalized.rpartition(".")
        normalized = f"{left}{right}" if len(right) == 3 else normalized

    try:
        amount = Decimal(normalized)
    except InvalidOperation:
        return None
    return amount if amount > 0 else None


def _format_vnd(value: Decimal) -> str:
    return f"{value.quantize(Decimal('1')):,.0f}"


def _format_foreign_amount(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01')):,.2f}"


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral_value():
        return f"{value:,.0f}"
    return f"{value.normalize():f}"


def _contains_any(normalized: str, markers: tuple[str, ...]) -> bool:
    return any(marker in normalized for marker in markers)


def _has_phrase(normalized: str, phrase: str) -> bool:
    return f" {phrase} " in f" {normalized} "


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("-", " "))
    no_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    no_accents = no_accents.replace("Đ", "D").replace("đ", "d")
    return " ".join(TOKEN_PATTERN.findall(no_accents.casefold()))


def _normalize_amount_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.replace("-", " "))
    no_accents = "".join(character for character in normalized if not unicodedata.combining(character))
    no_accents = no_accents.replace("Đ", "D").replace("đ", "d")
    return " ".join(no_accents.casefold().split())
