from apps.api.app.rag.guardrails import inspect_query, is_likely_supported_domain


def test_guardrails_reject_sensitive_credentials() -> None:
    result = inspect_query("OTP của tôi là 123456")
    assert result.allowed is False
    assert result.reason == "credential_or_secret"


def test_guardrails_reject_otp_value_disclosure() -> None:
    result = inspect_query("OTP là 123456")
    assert result.allowed is False
    assert result.reason == "credential_or_secret"


def test_guardrails_reject_otp_value_after_public_prefix() -> None:
    result = inspect_query("OTP là mã 123456")
    assert result.allowed is False
    assert result.reason == "credential_or_secret"


def test_guardrails_allow_public_otp_usage_question() -> None:
    result = inspect_query("OTP là gì? Khi nào tôi phải sử dụng OTP?")
    assert result.allowed is True


def test_guardrails_allow_general_password_policy_question() -> None:
    result = inspect_query("Nếu tôi nhập sai Mật khẩu, Ngân hàng có khóa truy cập dịch vụ VCB Digibank của tôi không?")
    assert result.allowed is True


def test_guardrails_allow_general_forgot_password_question_without_secret() -> None:
    result = inspect_query("Nếu tôi quên mật khẩu rút tiền thì phải làm sao?")
    assert result.allowed is True


def test_domain_detection_accepts_banking_question() -> None:
    assert is_likely_supported_domain("Điều kiện vay mua nhà Vietcombank là gì?")


def test_domain_detection_accepts_indexed_insurance_product() -> None:
    assert is_likely_supported_domain("Cho tôi biết toàn bộ thông tin về sản phẩm FWD Con vươn xa 2.0")


def test_domain_detection_accepts_general_credential_support_question() -> None:
    assert is_likely_supported_domain("Nếu tôi quên mật khẩu rút tiền thì phải làm sao?")


def test_domain_detection_accepts_otp_faq_question() -> None:
    assert is_likely_supported_domain("Phân biệt các phương thức nhận OTP?")
