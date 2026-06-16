from __future__ import annotations

import asyncio

from apps.api.app.core.config import Settings
from apps.api.app.models.chat import ChatMessage, ChatRequest
from apps.api.app.rag.exchange_rates import (
    ExchangeRateService,
    parse_exchange_rate_intent,
    parse_exchange_rate_xml,
)
from apps.api.app.rag.pipeline import RagPipeline

SAMPLE_XML = """<!--For reference only. Only one request every 5 minutes!-->
<ExrateList>
  <DateTime>6/13/2026 2:45:53 PM</DateTime>
  <Exrate CurrencyCode="AUD" CurrencyName="AUSTRALIAN DOLLAR" Buy="18,061.05" Transfer="18,243.49" Sell="18,827.73" />
  <Exrate CurrencyCode="CNY" CurrencyName="YUAN RENMINBI" Buy="3,792.81" Transfer="3,831.13" Sell="3,953.82" />
  <Exrate CurrencyCode="EUR" CurrencyName="EURO" Buy="29,687.29" Transfer="29,987.16" Sell="31,252.39" />
  <Exrate CurrencyCode="JPY" CurrencyName="YEN" Buy="158.69" Transfer="160.29" Sell="168.77" />
  <Exrate CurrencyCode="USD" CurrencyName="US DOLLAR" Buy="26,092.00" Transfer="26,122.00" Sell="26,412.00" />
  <Source>Joint Stock Commercial Bank for Foreign Trade of Vietnam - Vietcombank</Source>
</ExrateList>
"""


def _test_settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="local",
        rag_cache_backend="memory",
        openai_api_key=None,
        litellm_api_key=None,
    )


def test_exchange_rate_xml_parser_reads_quotes() -> None:
    table = parse_exchange_rate_xml(SAMPLE_XML)
    usd = table.quote_for("USD")

    assert table.updated_at == "6/13/2026 2:45:53 PM"
    assert usd is not None
    assert usd.name == "US DOLLAR"
    assert usd.buy_cash_display == "26,092.00"
    assert usd.transfer_display == "26,122.00"
    assert usd.sell_display == "26,412.00"


def test_exchange_rate_intent_ignores_non_exchange_usd_query() -> None:
    assert parse_exchange_rate_intent("Lãi suất tiết kiệm USD là bao nhiêu?") is None


def test_exchange_rate_intent_accepts_short_currency_quote_query() -> None:
    usd_today = parse_exchange_rate_intent("USD hôm nay bao nhiêu?")
    dollar_today = parse_exchange_rate_intent("Giá đô hôm nay")

    assert usd_today is not None
    assert usd_today.currency_code == "USD"
    assert dollar_today is not None
    assert dollar_today.currency_code == "USD"


def test_exchange_rate_intent_accepts_multiple_currencies_in_order() -> None:
    intent = parse_exchange_rate_intent("Cho tôi tỉ giá của USD và CNY")

    assert intent is not None
    assert intent.currency_codes == ("USD", "CNY")
    assert intent.currency_code == "USD"


def test_exchange_rate_intent_extracts_multiple_foreign_amounts() -> None:
    intent = parse_exchange_rate_intent("Tôi có 10 USD và 20 CNY thì đổi ra được bao nhiêu VND")

    assert intent is not None
    assert intent.currency_codes == ("USD", "CNY")
    assert [(item.code, str(item.amount)) for item in intent.foreign_amounts] == [
        ("USD", "10"),
        ("CNY", "20"),
    ]
    assert intent.amount_currency == "foreign"


def test_exchange_rate_intent_extracts_grouped_dollar_amount() -> None:
    intent = parse_exchange_rate_intent(
        "Hiện tại tôi có khoảng 100.000 đô thì đổi ra được bao nhiêu tiền"
    )

    assert intent is not None
    assert intent.currency_codes == ("USD",)
    assert intent.amount == 100000
    assert intent.amount_currency == "foreign"
    assert [(item.code, str(item.amount)) for item in intent.foreign_amounts] == [("USD", "100000")]


def test_exchange_rate_intent_prefers_long_currency_aliases() -> None:
    intent = parse_exchange_rate_intent("Giá đô Úc hôm nay")

    assert intent is not None
    assert intent.currency_codes == ("AUD",)


async def test_exchange_rate_service_answers_usd_quote_and_caches_xml() -> None:
    fetch_count = 0

    async def fetch_xml() -> str:
        nonlocal fetch_count
        fetch_count += 1
        return SAMPLE_XML

    service = ExchangeRateService(xml_fetcher=fetch_xml)

    first = await service.answer_query("Tỷ giá USD hôm nay?")
    second = await service.answer_query("Giá bán USD là bao nhiêu?")

    assert first is not None
    assert second is not None
    assert fetch_count == 1
    assert "Tỷ giá USD" in first.answer
    assert "Mua tiền mặt: 26,092.00 VND" in first.answer
    assert "Bán: 26,412.00 VND" in second.answer
    assert first.metadata["retrieval_route"] == "live_exchange_rates"


async def test_exchange_rate_service_answers_multiple_currency_quotes() -> None:
    async def fetch_xml() -> str:
        return SAMPLE_XML

    service = ExchangeRateService(xml_fetcher=fetch_xml)

    result = await service.answer_query("Cho tôi tỉ giá của USD và CNY")

    assert result is not None
    assert "USD (US DOLLAR)" in result.answer
    assert "CNY (YUAN RENMINBI)" in result.answer
    assert "Mua tiền mặt: 26,092.00 VND" in result.answer
    assert "Mua tiền mặt: 3,792.81 VND" in result.answer
    assert result.metadata["exchange_rate_currency"] == "USD"
    assert result.metadata["exchange_rate_currencies"] == ["USD", "CNY"]


async def test_exchange_rate_service_converts_vnd_amount_to_multiple_currencies() -> None:
    async def fetch_xml() -> str:
        return SAMPLE_XML

    service = ExchangeRateService(xml_fetcher=fetch_xml)

    result = await service.answer_query("Tôi có khoảng 10tr vnđ thì đổi ra được bao nhiêu USD và CNY")

    assert result is not None
    assert "Quy đổi tham khảo với 10,000,000 VND" in result.answer
    assert "khoảng 378.62 USD" in result.answer
    assert "khoảng 2,529.20 CNY" in result.answer
    assert result.metadata["exchange_rate_amount"] == "10000000"
    assert result.metadata["exchange_rate_amount_currency"] == "VND"
    assert result.metadata["exchange_rate_currencies"] == ["USD", "CNY"]


async def test_exchange_rate_service_coalesces_concurrent_fetches() -> None:
    fetch_count = 0

    async def fetch_xml() -> str:
        nonlocal fetch_count
        fetch_count += 1
        await asyncio.sleep(0)
        return SAMPLE_XML

    service = ExchangeRateService(xml_fetcher=fetch_xml)

    first, second = await asyncio.gather(
        service.answer_query("Tỷ giá USD hôm nay?"),
        service.answer_query("Tỷ giá EUR hôm nay?"),
    )

    assert first is not None
    assert second is not None
    assert fetch_count == 1


async def test_exchange_rate_service_converts_foreign_amount_to_vnd() -> None:
    async def fetch_xml() -> str:
        return SAMPLE_XML

    service = ExchangeRateService(xml_fetcher=fetch_xml)

    result = await service.answer_query("100 USD đổi sang VND được bao nhiêu?")

    assert result is not None
    assert "Quy đổi tham khảo cho 100 USD" in result.answer
    assert "2,609,200 VND" in result.answer
    assert "2,612,200 VND" in result.answer
    assert "2,641,200 VND" not in result.answer


async def test_exchange_rate_service_converts_grouped_dollar_amount_to_vnd() -> None:
    async def fetch_xml() -> str:
        return SAMPLE_XML

    service = ExchangeRateService(xml_fetcher=fetch_xml)

    result = await service.answer_query(
        "Hiện tại tôi có khoảng 100.000 đô thì đổi ra được bao nhiêu tiền"
    )

    assert result is not None
    assert "Quy đổi tham khảo cho 100,000 USD" in result.answer
    assert "2,609,200,000 VND" in result.answer
    assert "2,612,200,000 VND" in result.answer
    assert "2,641,200,000 VND" not in result.answer
    assert result.metadata["exchange_rate_amount"] == "100000"
    assert result.metadata["exchange_rate_amount_currency"] == "foreign"


async def test_exchange_rate_service_converts_multiple_foreign_amounts_to_vnd() -> None:
    async def fetch_xml() -> str:
        return SAMPLE_XML

    service = ExchangeRateService(xml_fetcher=fetch_xml)

    result = await service.answer_query("Tôi có 10 USD và 20 CNY thì đổi ra được bao nhiêu VND")

    assert result is not None
    assert "Quy đổi tham khảo sang VND" in result.answer
    assert "10 USD x 26,092.00 = 260,920 VND" in result.answer
    assert "20 CNY x 3,792.81 = 75,856 VND" in result.answer
    assert "Tổng khoảng 336,776 VND" in result.answer
    assert "Tổng khoảng 337,843 VND" in result.answer
    assert result.metadata["exchange_rate_foreign_amounts"] == {"USD": "10", "CNY": "20"}


async def test_exchange_rate_service_converts_vnd_amount_to_foreign_currency() -> None:
    async def fetch_xml() -> str:
        return SAMPLE_XML

    service = ExchangeRateService(xml_fetcher=fetch_xml)

    result = await service.answer_query("Tôi có 2tr vnđ thì đổi được bao nhiêu USD")

    assert result is not None
    assert "2,000,000 VND" in result.answer
    assert "75.72 USD" in result.answer
    assert "tỷ giá bán 26,412.00 VND/USD" in result.answer
    assert result.metadata["exchange_rate_amount"] == "2000000"
    assert result.metadata["exchange_rate_amount_currency"] == "VND"


async def test_pipeline_routes_exchange_rate_query_before_rag() -> None:
    async def fetch_xml() -> str:
        return SAMPLE_XML

    pipeline = RagPipeline(_test_settings())
    pipeline.exchange_rates = ExchangeRateService(xml_fetcher=fetch_xml)  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(messages=[ChatMessage(role="user", content="Tỷ giá USD hôm nay?")])
    )

    assert response.metadata["retrieval_route"] == "live_exchange_rates"
    assert response.metadata["tool"] == "vietcombank_exchange_rates"
    assert response.sources[0].section == "exchange_rate"
    assert "Tỷ giá USD" in response.answer


async def test_pipeline_routes_multi_currency_exchange_rate_query() -> None:
    async def fetch_xml() -> str:
        return SAMPLE_XML

    pipeline = RagPipeline(_test_settings())
    pipeline.exchange_rates = ExchangeRateService(xml_fetcher=fetch_xml)  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(messages=[ChatMessage(role="user", content="Cho tôi tỉ giá của USD và CNY")])
    )

    assert response.metadata["retrieval_route"] == "live_exchange_rates"
    assert response.metadata["exchange_rate_currencies"] == ["USD", "CNY"]
    assert "USD (US DOLLAR)" in response.answer
    assert "CNY (YUAN RENMINBI)" in response.answer


async def test_pipeline_converts_vnd_to_requested_currency() -> None:
    async def fetch_xml() -> str:
        return SAMPLE_XML

    pipeline = RagPipeline(_test_settings())
    pipeline.exchange_rates = ExchangeRateService(xml_fetcher=fetch_xml)  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(messages=[ChatMessage(role="user", content="Tôi có 2tr vnđ thì đổi được bao nhiêu USD")])
    )

    assert response.metadata["retrieval_route"] == "live_exchange_rates"
    assert response.metadata["exchange_rate_amount_currency"] == "VND"
    assert "75.72 USD" in response.answer


async def test_pipeline_converts_multiple_foreign_amounts_to_vnd() -> None:
    async def fetch_xml() -> str:
        return SAMPLE_XML

    pipeline = RagPipeline(_test_settings())
    pipeline.exchange_rates = ExchangeRateService(xml_fetcher=fetch_xml)  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Tôi có 10 USD và 20 CNY thì đổi ra được bao nhiêu VND",
                )
            ]
        )
    )

    assert response.metadata["retrieval_route"] == "live_exchange_rates"
    assert response.metadata["exchange_rate_amount_currency"] == "foreign"
    assert response.metadata["exchange_rate_foreign_amounts"] == {"USD": "10", "CNY": "20"}
    assert "Tổng khoảng 336,776 VND" in response.answer


async def test_pipeline_converts_vnd_to_multiple_requested_currencies() -> None:
    async def fetch_xml() -> str:
        return SAMPLE_XML

    pipeline = RagPipeline(_test_settings())
    pipeline.exchange_rates = ExchangeRateService(xml_fetcher=fetch_xml)  # type: ignore[assignment]

    response = await pipeline.answer(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content="Tôi có khoảng 10tr vnđ thì đổi ra được bao nhiêu USD và CNY",
                )
            ]
        )
    )

    assert response.metadata["retrieval_route"] == "live_exchange_rates"
    assert response.metadata["exchange_rate_amount_currency"] == "VND"
    assert response.metadata["exchange_rate_currencies"] == ["USD", "CNY"]
    assert "378.62 USD" in response.answer
    assert "2,529.20 CNY" in response.answer
