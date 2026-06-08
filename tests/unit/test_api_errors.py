from api.errors import screening_error_from_exception


def test_maps_gemini_rate_limit() -> None:
    from google.genai.errors import ClientError

    err = ClientError(429, {"error": {"message": "quota exceeded"}}, None)
    status, code, message = screening_error_from_exception(err)
    assert status == 503
    assert code == "LLM_RATE_LIMIT"
    assert "quota" in message.lower()


def test_maps_gemini_server_unavailable() -> None:
    from google.genai.errors import ServerError

    err = ServerError(503, {"error": {"message": "service unavailable"}}, None)
    status, code, message = screening_error_from_exception(err)
    assert status == 503
    assert code == "LLM_RATE_LIMIT"
    assert "unavailable" in message.lower()
