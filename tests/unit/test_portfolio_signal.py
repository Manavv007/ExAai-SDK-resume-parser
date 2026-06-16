from agent.tools.portfolio_signal import (
    apply_portfolio_penalties,
    assess_crawl_quality,
    build_portfolio_red_flags,
    evaluate_portfolio_signal,
    infer_role_category,
    is_personal_website,
    normalize_role_category,
)


def test_infer_role_category_for_software_engineer() -> None:
    category = infer_role_category(
        job_title="Senior Backend Software Engineer",
        domain="technical",
        must_have=["Python", "PostgreSQL"],
    )
    assert category == "software_engineering"


def test_infer_role_category_for_non_portfolio_role() -> None:
    category = infer_role_category(
        job_title="Sales Development Representative",
        domain="business",
    )
    assert category == "non_portfolio"


def test_assess_crawl_quality_rejects_cloudflare_page() -> None:
    ok, reason = assess_crawl_quality(
        "===BEGIN EXTERNAL CONTENT: https://github.com/user===\n"
        + ("Cloudflare security challenge enable javascript blocked page. " * 10)
        + "\n===END EXTERNAL CONTENT==="
    )
    assert ok is False
    assert "cloudflare" in reason


def test_evaluate_portfolio_signal_strong_for_crawled_github() -> None:
    signal = evaluate_portfolio_signal(
        role_category="software_engineering",
        profile_urls=["https://github.com/janedoe"],
        enriched_contents={
            "https://github.com/janedoe": (
                "===BEGIN EXTERNAL CONTENT: https://github.com/janedoe===\n"
                + ("portfolio readme content " * 20)
                + "\n===END EXTERNAL CONTENT==="
            )
        },
        experience_years=5,
    )
    assert signal["signal_strength"] == "strong"
    assert signal["penalty_points"] == 0


def test_evaluate_portfolio_signal_none_applies_penalty_and_red_flag() -> None:
    signal = evaluate_portfolio_signal(
        role_category="software_engineering",
        profile_urls=["https://linkedin.com/in/janedoe"],
        enriched_contents={},
        experience_years=10,
    )
    assert signal["signal_strength"] == "none"
    assert signal["penalty_points"] > 0
    flags = build_portfolio_red_flags(signal)
    assert flags[0]["flag"] == "missing_portfolio_verification"
    assert flags[0]["severity"] == "high"


def test_apply_portfolio_penalties_hard_caps_none_signal() -> None:
    signal = evaluate_portfolio_signal(
        role_category="software_engineering",
        profile_urls=[],
        enriched_contents={},
        experience_years=10,
    )
    adjusted, applied, hard_cap = apply_portfolio_penalties(95, signal)
    assert applied > 0
    assert adjusted <= 75
    assert hard_cap is True


def test_infer_role_category_prefers_software_over_design_domain() -> None:
    """Regression: some JDs get domain=design but are clearly SWE/SDE."""
    category = infer_role_category(
        job_title="Software Development Intern",
        domain="design",
        jd_text="Looking for a software engineer intern to build web features and APIs.",
        must_have=["React", "Git"],
        nice_to_have=["CI/CD"],
    )
    assert category == "software_engineering"


def test_personal_website_detection() -> None:
    assert is_personal_website("https://manavpatel.dev") is True
    assert is_personal_website("https://linkedin.com/in/manav") is False


def test_normalize_role_category_aliases() -> None:
    assert normalize_role_category("machine_learning") == "aiml"
    assert normalize_role_category("software") == "software_engineering"
