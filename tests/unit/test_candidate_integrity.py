"""Unit tests for candidate integrity scoring."""



from __future__ import annotations



from agent.security.candidate_integrity import (
    compare_url_sets,
    compute_candidate_integrity,
    compute_overall_integrity,
    parse_resume_timeline_anchor,
)





def test_compare_url_sets_overlap() -> None:

    resume = ["https://github.com/alice-dev", "https://www.linkedin.com/in/alice-dev"]

    linkedin = ["https://github.com/alice-dev", "https://behance.net/alice"]

    result = compare_url_sets(resume, linkedin)

    assert "https://github.com/alice-dev" in result["overlap"]

    assert not result["conflicting"]





def test_compare_url_sets_conflicting_github_slug() -> None:

    resume = ["https://github.com/alice-dev"]

    linkedin = ["https://github.com/bob-other"]

    result = compare_url_sets(resume, linkedin)

    assert result["conflicting"]

    assert not result["overlap"]





def test_github_timeline_bad_when_account_newer_than_activity() -> None:

    github = {

        "user_profile": {

            "login": "candidate",

            "html_url": "https://github.com/candidate",

            "created_at": "2024-06-01T00:00:00Z",

        },

        "activity_timeline": {

            "earliest_activity_at": "2022-01-01T00:00:00Z",

            "latest_activity_at": "2023-01-01T00:00:00Z",

        },

    }

    out = compute_candidate_integrity(

        profile_urls=["https://github.com/candidate"],

        enriched_contents=[],

        github_repo_analyses=github,

        resume_structured={},

    )

    timeline = next(s for s in out["signals"] if s["signal_id"] == "github_account_timeline")

    assert timeline["indication"] == "bad"

    assert out["indicators"]["github_account_timeline"] == "bad"
    assert out["indicators"]["overall"] == "bad"





def test_github_timeline_good() -> None:

    github = {

        "user_profile": {

            "login": "candidate",

            "html_url": "https://github.com/candidate",

            "created_at": "2019-01-01T00:00:00Z",

        },

        "activity_timeline": {

            "earliest_activity_at": "2022-01-01T00:00:00Z",

        },

    }

    out = compute_candidate_integrity(

        profile_urls=["https://github.com/candidate"],

        enriched_contents=[],

        github_repo_analyses=github,

        resume_structured={},

    )

    timeline = next(s for s in out["signals"] if s["signal_id"] == "github_account_timeline")

    assert timeline["indication"] == "good"

    assert out["indicators"]["github_account_timeline"] == "good"
    assert out["indicators"]["overall"] == "good"





def test_linkedin_contact_match_good() -> None:

    out = compute_candidate_integrity(

        profile_urls=[

            "https://github.com/alice-dev",

            "https://www.linkedin.com/in/alice-dev",

        ],

        enriched_contents=[

            {

                "url": "https://www.linkedin.com/in/alice-dev",

                "content": "Contact: https://github.com/alice-dev",

                "ok": True,

            }

        ],

        github_repo_analyses={},

        resume_structured={},

    )

    linkedin = next(s for s in out["signals"] if s["signal_id"] == "linkedin_contact_links")

    assert linkedin["indication"] == "good"

    assert out["indicators"]["linkedin_contact_links"] == "good"
    assert out["indicators"]["overall"] == "good"





def test_integrity_signals_are_limited_to_three_checks() -> None:

    out = compute_candidate_integrity(

        profile_urls=["https://github.com/candidate"],

        enriched_contents=[],

        github_repo_analyses={},

        resume_structured={},

    )

    signal_ids = {s["signal_id"] for s in out["signals"]}

    assert signal_ids == {
        "github_account_timeline",
        "linkedin_contact_links",
        "github_profile_readme_links",
    }
    assert set(out["indicators"].keys()) == signal_ids | {"overall"}





def test_github_profile_readme_conflict_bad() -> None:

    out = compute_candidate_integrity(

        profile_urls=["https://www.behance.net/alice-dev"],

        enriched_contents=[],

        github_repo_analyses={

            "user_profile": {"html_url": "https://github.com/alice-dev"},

            "profile_readme": "Portfolio: https://www.behance.net/other-person",

        },

        resume_structured={},

    )

    readme = next(s for s in out["signals"] if s["signal_id"] == "github_profile_readme_links")

    assert readme["indication"] == "bad"

    assert out["indicators"]["github_profile_readme_links"] == "bad"





def test_linkedin_github_mismatch_bad() -> None:

    out = compute_candidate_integrity(

        profile_urls=["https://github.com/alice-dev"],

        enriched_contents=[

            {

                "url": "https://www.linkedin.com/in/alice-dev",

                "content": "GitHub: https://github.com/bob-other",

                "ok": True,

            }

        ],

        github_repo_analyses={},

        resume_structured={},

    )

    linkedin = next(s for s in out["signals"] if s["signal_id"] == "linkedin_contact_links")

    assert linkedin["indication"] == "bad"





def test_parse_resume_timeline_anchor_from_education() -> None:

    anchor = parse_resume_timeline_anchor(

        {"education": ["Pandit University Expected May 2027 Bachelor of Technology"]}

    )

    assert anchor is not None

    assert anchor.year == 2027





def test_insufficient_data_yields_not_enough_evidence() -> None:

    out = compute_candidate_integrity(

        profile_urls=["https://example.com/portfolio"],

        enriched_contents=[],

        github_repo_analyses=None,

        resume_structured={},

    )

    assert out["indicators"]["github_account_timeline"] == "not_enough_evidence"

    assert out["indicators"]["linkedin_contact_links"] == "not_enough_evidence"

    assert out["indicators"]["github_profile_readme_links"] == "not_enough_evidence"
    assert out["indicators"]["overall"] == "not_enough_evidence"

    assert "score" not in out


def test_compute_overall_integrity_rules() -> None:
    assert compute_overall_integrity(
        {
            "github_account_timeline": "good",
            "linkedin_contact_links": "good",
            "github_profile_readme_links": "good",
        }
    ) == "good"
    assert compute_overall_integrity(
        {
            "github_account_timeline": "good",
            "linkedin_contact_links": "not_enough_evidence",
            "github_profile_readme_links": "not_enough_evidence",
        }
    ) == "good"
    assert compute_overall_integrity(
        {
            "github_account_timeline": "bad",
            "linkedin_contact_links": "good",
            "github_profile_readme_links": "good",
        }
    ) == "bad"
    assert compute_overall_integrity(
        {
            "github_account_timeline": "not_enough_evidence",
            "linkedin_contact_links": "not_enough_evidence",
            "github_profile_readme_links": "not_enough_evidence",
        }
    ) == "not_enough_evidence"

