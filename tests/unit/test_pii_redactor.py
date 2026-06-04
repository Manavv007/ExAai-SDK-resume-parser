import re

import pytest

from agent.security.pii_redactor import DEFAULT_ENTITIES, redact_text

SAMPLE_RESUME = """
Jane Doe
Email: jane.doe@example.com
Phone: +1 (415) 555-0132
Location: San Francisco, California
Date of birth: March 15, 1990
Portfolio: https://github.com/janedoe
Nationality: American
""".strip()


@pytest.fixture(scope="module", autouse=True)
def _require_spacy_model() -> None:
    try:
        import spacy

        spacy.load("en_core_web_sm")
    except OSError as exc:
        pytest.skip(f"spaCy model en_core_web_sm required for PII tests: {exc}")


def test_default_entities_cover_spec() -> None:
    expected = {
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "LOCATION",
        "DATE_TIME",
        "URL",
        "NRP",
        "AGE",
    }
    assert expected <= set(DEFAULT_ENTITIES)


def test_redacts_email_phone_and_person() -> None:
    redacted, summary = redact_text(SAMPLE_RESUME)

    assert "jane.doe@example.com" not in redacted
    assert "415" not in redacted or "[PHONE_NUMBER_" in redacted
    assert "Jane Doe" not in redacted
    assert summary.redaction_count >= 3
    assert "EMAIL_ADDRESS" in summary.fields_redacted
    assert summary.counts_by_type["EMAIL_ADDRESS"] >= 1


def test_placeholder_format() -> None:
    redacted, _ = redact_text("Contact jane.doe@example.com for info.")
    assert re.search(r"\[EMAIL_ADDRESS_\d+\]", redacted)


def test_skip_url_redaction_preserves_links() -> None:
    text = "See https://github.com/janedoe for code."
    redacted, summary = redact_text(text, redact_urls=False)
    assert "https://github.com/janedoe" in redacted
    assert "URL" not in summary.fields_redacted


def test_empty_text_returns_empty_summary() -> None:
    redacted, summary = redact_text("   ")
    assert redacted.strip() == ""
    assert summary.redaction_count == 0


def test_no_pii_unchanged() -> None:
    text = "Experienced backend engineer with Python and distributed systems."
    redacted, summary = redact_text(text)
    assert redacted == text
    assert summary.redaction_count == 0
