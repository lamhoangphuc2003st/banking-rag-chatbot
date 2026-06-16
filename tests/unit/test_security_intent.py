from apps.api.app.rag.security_intent import classify_security_intent


def test_security_intent_classifies_password_policy_as_public_info() -> None:
    intent = classify_security_intent(
        "Sau bao lâu thì phải đổi mật khẩu một lần và mật khẩu nên đặt như thế nào để an toàn"
    )

    assert intent.kind == "public_info"


def test_security_intent_classifies_wrong_password_lock_as_public_info() -> None:
    intent = classify_security_intent("Nếu tôi nhập sai mật khẩu, ngân hàng có khóa truy cập không?")

    assert intent.kind == "public_info"


def test_security_intent_classifies_otp_definition_as_public_info() -> None:
    intent = classify_security_intent("Mã OTP giao dịch là gì?")

    assert intent.kind == "public_info"


def test_security_intent_classifies_otp_comparison_with_punctuation_as_public_info() -> None:
    intent = classify_security_intent("Phân biệt các phương thức nhận OTP?")

    assert intent.kind == "public_info"


def test_security_intent_classifies_forgot_password_as_account_recovery() -> None:
    intent = classify_security_intent("Quên mật khẩu rút tiền, làm sao để lấy lại?")

    assert intent.kind == "account_recovery"


def test_security_intent_ignores_non_security_banking_query() -> None:
    intent = classify_security_intent("VCB có cho vay mua ô tô không?")

    assert intent.kind == "none"
