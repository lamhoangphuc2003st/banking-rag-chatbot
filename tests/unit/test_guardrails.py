from apps.api.app.rag.guardrails import inspect_query, is_likely_supported_domain


def test_guardrails_reject_sensitive_credentials() -> None:
    result = inspect_query("OTP của tôi là 123456")
    assert result.allowed is False
    assert result.reason == "credential_or_secret"


def test_domain_detection_accepts_banking_question() -> None:
    assert is_likely_supported_domain("Điều kiện vay mua nhà Vietcombank là gì?")
