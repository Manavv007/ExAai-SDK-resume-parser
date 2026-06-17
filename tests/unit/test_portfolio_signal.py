from agent.tools.portfolio_signal import (
    apply_portfolio_penalties,
    assess_crawl_quality,
    build_portfolio_red_flags,
    evaluate_portfolio_signal,
    infer_role_category,
    is_personal_website,
    is_portfolio_like_url,
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
    assert signal["required_platforms_found"] == ["github.com"]
    assert signal["penalty_points"] == 0


def test_evaluate_portfolio_signal_none_applies_penalty_and_red_flag() -> None:
    signal = evaluate_portfolio_signal(
        role_category="software_engineering",
        profile_urls=["https://linkedin.com/in/janedoe"],
        enriched_contents={},
        experience_years=10,
    )
    assert signal["required_platforms_found"] == []
    assert signal["penalty_points"] > 0
    flags = build_portfolio_red_flags(signal)
    assert flags[0]["flag"] == "missing_portfolio_verification"
    assert flags[0]["severity"] == "high"


def test_evaluate_portfolio_signal_required_platform_link_avoids_penalty() -> None:
    signal = evaluate_portfolio_signal(
        role_category="software_engineering",
        profile_urls=["https://github.com/janedoe"],
        enriched_contents={"https://github.com/janedoe": "too short"},
        experience_years=1,
    )
    assert signal["required_platforms_found"] == ["github.com"]
    assert signal["penalty_points"] == 0


def test_design_role_verified_personal_portfolio_avoids_penalty_without_github() -> None:
    url = "https://janedoe.design"
    content = (
        "===BEGIN EXTERNAL CONTENT: https://janedoe.design===\n"
        "Jane Doe | UX Designer — my portfolio\n"
        "I designed case studies for mobile apps. View project live demo.\n"
        + ("case study showcase " * 20)
        + "\n===END EXTERNAL CONTENT==="
    )
    signal = evaluate_portfolio_signal(
        role_category="design",
        profile_urls=[url],
        enriched_contents={url: content},
        resume_structured={"candidate_name": "Jane Doe"},
    )
    assert signal["penalty_points"] == 0
    assert signal["verified_personal_portfolio"] is True
    assert "github.com" not in (signal["required_platforms_found"] or [])


def test_design_role_without_portfolio_still_penalized() -> None:
    signal = evaluate_portfolio_signal(
        role_category="design",
        profile_urls=["https://linkedin.com/in/janedoe"],
        enriched_contents={},
        experience_years=3,
    )
    assert signal["penalty_points"] > 0


def test_research_academic_includes_github_required_platform() -> None:
    signal = evaluate_portfolio_signal(
        role_category="research_academic",
        profile_urls=["https://scholar.google.com/citations?user=abc"],
        enriched_contents={},
        experience_years=6,
    )
    assert "github.com" in signal["required_platforms_missing"]


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


def test_google_docs_with_github_links_is_portfolio_like() -> None:
    assert (
        is_portfolio_like_url(
            "https://docs.google.com/document/d/abc/edit",
            "Portfolio links: https://github.com/janedoe/app",
        )
        is True
    )


def test_normalize_role_category_aliases() -> None:
    assert normalize_role_category("machine_learning") == "aiml"
    assert normalize_role_category("software") == "software_engineering"


# ---------------------------------------------------------------------------
# Scoring-based is_portfolio_like_url tests
# ---------------------------------------------------------------------------


def test_known_portfolio_platform_scores_high_without_content() -> None:
    """A behance.net URL qualifies on domain alone via allowlist (+50)."""
    assert is_portfolio_like_url("https://www.behance.net/janedoe") is True


def test_codepen_qualifies_on_allowlist() -> None:
    assert is_portfolio_like_url("https://codepen.io/janedoe") is True


def test_corporate_site_with_projects_word_does_not_qualify() -> None:
    """'our products', 'sign up', 'get started' must suppress a page
    that would otherwise score from a single weak content keyword."""
    corporate_content = (
        "Welcome to AcmeCorp. Our products help teams grow. "
        "Our services include cloud and AI. Sign up today for free trial. "
        "Get started with our platform. Our team is here to help. "
        "We offer enterprise solutions."
    )
    # stripe.com is NOT in _GENERIC_DOMAINS so is_personal_website returns True (+40),
    # but corporate anti-signals must pull the total below threshold.
    result = is_portfolio_like_url("https://stripe.com/about", corporate_content)
    assert result is False, "Corporate site should not qualify even if domain passes personal check"


def test_name_in_domain_boosts_score() -> None:
    """Candidate name tokens matching the domain add +30 (identity signal)."""
    # manavpatel.dev: is_personal_website=True (+40) + name match (+30) = 70 → True
    assert (
        is_portfolio_like_url(
            "https://manavpatel.dev",
            candidate_name="Manav Patel",
        )
        is True
    )


def test_name_in_domain_no_content_still_qualifies() -> None:
    """Personal domain + name match alone is enough without any page content."""
    assert (
        is_portfolio_like_url(
            "https://janedoe.io",
            "",
            candidate_name="Jane Doe",
        )
        is True
    )


def test_handle_in_content_boosts_score() -> None:
    """GitHub handle found in page content adds +25 (ownership signal)."""
    # Notion page (+35) + handle match (+25) = 60 → True
    content = "Check out my work at github.com/janedoe/myapp and notion.so/notes."
    assert (
        is_portfolio_like_url(
            "https://janedoe.notion.site/portfolio",
            content,
            known_handles=["janedoe"],
        )
        is True
    )


def test_no_identity_context_unverified_custom_domain_still_qualifies() -> None:
    """Backward compat: calling without candidate_name/known_handles still works.
    A custom domain alone (+40) is above threshold (40) so returns True."""
    assert is_portfolio_like_url("https://manavpatel.dev") is True


def test_error_page_does_not_qualify() -> None:
    """Hard error signatures produce large negative penalties."""
    error_content = "403 Forbidden. Access denied. Page not found. Enable javascript."
    assert is_portfolio_like_url("https://manavpatel.dev", error_content) is False


def test_multiple_github_repos_in_content_qualifies_without_personal_domain() -> None:
    """A Notion doc with 3+ distinct repo links is a hub even without a personal domain."""
    content = (
        "My projects: github.com/user/project-alpha, "
        "github.com/user/project-beta, github.com/user/project-gamma"
    )
    assert (
        is_portfolio_like_url(
            "https://user.notion.site/portfolio",
            content,
        )
        is True
    )


def test_single_github_mention_on_blog_does_not_qualify() -> None:
    """One 'github.com/' mention on a non-personal blog should NOT qualify."""
    blog_content = (
        "Today we explore open source. Check out github.com/someorg/sometool "
        "for reference. Our team uses many open source libraries. "
        "Sign up to our newsletter. Get started free."
    )
    # blog.medium.com → not personal, not allowlisted for portfolio
    assert is_portfolio_like_url("https://blog.medium.com/some-post-123", blog_content) is False


def test_first_person_work_language_contributes_to_score() -> None:
    """First-person phrases on a Notion page push it over threshold."""
    content = "I built this app. I designed the UI. I created the backend API."
    # notion.site (+50) + first-person phrases: min(3*10, 25)=25 = 75 → True
    assert is_portfolio_like_url("https://abc.notion.site/page", content) is True


def test_hire_me_signal_on_custom_domain_qualifies() -> None:
    """Personal domain + hire intent is strong evidence of a portfolio."""
    content = "Available for freelance work. Contact me for projects. Hire me!"
    assert (
        is_portfolio_like_url(
            "https://johndoe.dev",
            content,
            candidate_name="John Doe",
        )
        is True
    )


def test_personal_website_rejects_script_js_domain() -> None:
    assert is_personal_website("https://script.js") is False


def test_crawl_status_github_repo_is_not_applicable() -> None:
    signal = evaluate_portfolio_signal(
        role_category="software_engineering",
        profile_urls=[
            "https://github.com/user/repo-one",
            "https://manavbhavsar-portfolio.vercel.app/",
        ],
        enriched_contents={
            "https://manavbhavsar-portfolio.vercel.app/": (
                "===BEGIN EXTERNAL CONTENT===\n"
                + ("portfolio content " * 20)
                + "\n===END EXTERNAL CONTENT==="
            )
        },
        github_repo_analyses={
            "username": "user",
            "repo_analyses": [{"url": "https://github.com/user/repo-one"}],
            "sandbox_reports": [
                {"url": "https://github.com/user/repo-one", "clone_ok": True}
            ],
        },
    )
    assert (
        signal["crawl_status_log"]["https://github.com/user/repo-one"]
        == "not_applicable_github_repo"
    )
    assert signal["github_status_log"]["https://github.com/user/repo-one"] == "sandbox_ok"
    assert signal["sandbox_status_log"]["https://github.com/user/repo-one"] == "clone_ok"
    assert (
        signal["crawl_status_log"]["https://manavbhavsar-portfolio.vercel.app/"]
        == "valid_content"
    )


def test_crawl_status_uncrawled_linkedin_is_not_crawled() -> None:
    signal = evaluate_portfolio_signal(
        role_category="software_engineering",
        profile_urls=["https://linkedin.com/in/janedoe"],
        enriched_contents=[],
    )
    assert signal["crawl_status_log"]["https://linkedin.com/in/janedoe"] == "not_crawled"
