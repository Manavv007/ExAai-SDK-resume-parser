# Resume Link Identity Verification Parameters
Goal: objectively bound the probability that a submitted link profile belongs to the same person as the resume, without relying on name/email equality.

## Finalized Parameters (User-Confirmed)
1. Account creation date vs resume timeline → **implemented** (`github_account_timeline` in `agent/security/candidate_integrity.py`)
2. LinkedIn contact section → GitHub link → **implemented** (`linkedin_contact_links`)
3. Mutual cross-links between profiles → **implemented** (`mutual_cross_links`)
4. GitHub public email = resume email → planned (not yet in integrity v1)
5. GitHub twitter_username field match → planned
6. Profile README links → **implemented** (`github_profile_readme_links`)

---

## Additional Parameters by Platform (Non-GitHub Focus)

### LinkedIn
7. **"Verified on LinkedIn" badge** — platform-issued identity verification (Persona/clearcheck). Visible on profile; API-accessible via LinkedIn Verified API for partners.
8. **Work email verification badge** — company email domain verified via LinkedIn's email challenge flow. Shows as "verified" next to current employer.
9. **LinkedIn Skill Assessments passed** — proctored timed assessments; results tied to profile. Hard to fake at scale.
10. **LinkedIn "Open to Work" / job preferences timestamp** — API-visible timestamp aligns with resume timeline.
11. **Recommendation count + recommender profile quality** — recommendations from verified/current colleagues at claimed employers. Corroborates employment dates.
12. **Profile view / search appearance analytics (if shared)** — candidate can share "who viewed your profile" showing recruiter views from claimed companies.

### Kaggle
13. **Identity-verified badge for prize competitions** — Kaggle now requires government ID + selfie verification for prize-eligible competitions. Badge shown on profile.
14. **Competition medals (Gold/Silver/Bronze) with timestamps** — immutable, platform-awarded, dated. Medal progression timeline must align with resume claims.
15. **Notebook / dataset / discussion awards** — peer-awarded, timestamped, visible on profile.
16. **GitHub-linked notebook sync** — "File → Link to GitHub" creates visible link on notebook + kernel metadata. Detectable by scraping public profile/notebooks.
17. **Kaggle "Contributor" / "Expert" / "Master" / "Grandmaster" tier progression** — tier thresholds are public; progression dates are immutable.
18. **Kaggle API token creation date (if disclosed)** — token creation timestamp is account-bound; can be cross-referenced with resume claims about when they started using Kaggle.

### Hugging Face
19. **Email-verified badge** — HF enforces email verification for OAuth; profile shows "verified email" indicator.
20. **GPG-signed commits on HF repos** — HF shows "Verified" badge on commits signed with GPG key registered to that HF account. Mirrors GitHub's verified commit model.
21. **HF OAuth-linked external accounts** — HF profile shows connected GitHub, GitLab, Bitbucket, etc. via OAuth. Visible on public profile settings.
22. **Model / dataset / Space authorship with creation dates** — immutable timestamps; owner field matches HF username. Corroborates project timeline.
23. **HF "Pro" / "Enterprise" subscription badge** — paid tier requires billing identity; harder to fake.

### Stack Overflow
24. **GitHub OAuth authentication linked** — SO profile shows "Signed in with GitHub" badge. Requires active GitHub session at link time.
25. **Reputation + badge timeline** — reputation history is immutable; top tags must align with resume tech stack. Sudden reputation spikes are detectable.
26. **Developer Story (legacy) / Collector badge** — shows timeline of technologies, roles, projects with dates. Cross-referenceable with resume.
27. **Top answers in claimed tech stack** — answer dates, scores, and tags provide objective skill evidence.

### Personal Domain / Portfolio
28. **Custom domain DNS TXT challenge (GitHub Pages / Vercel / Netlify / Cloudflare Pages / Render)** — platform-issued challenge record in DNS. Proves domain owner = platform account holder.
29. **WHOIS registrant match (when public)** — registrant name/email aligns with resume. Most domains use privacy protection, so use only when available.
30. **Canonical author URL on blog platforms** — Dev.to, Hashnode, Medium, Substack author profile has "canonical URL" or "profile URL" field pointing to same domain/GitHub handle.
31. **GitHub Actions / Vercel / Netlify deploy history tied to domain** — deploy logs show committing GitHub account. Cross-reference with candidate's GitHub.
32. **SSL certificate transparency logs** — Certificate Transparency logs show domain issuance dates; matches resume timeline for personal site launch.

### GitLab / Bitbucket / Gitea
33. **Email-verified badge** — GitLab requires email verification; shows "verified" on profile.
34. **GPG/SSH-signed commit badges** — GitLab shows "Verified" badge on signed commits with registered keys.
35. **Identity verification tiers** — GitLab.com enforces progressive identity verification (email → phone → ID) based on risk score. Profile shows verification level.
36. **SAML/SSO linked badge** — if candidate's employer enforces SSO, profile shows "Linked to SAML" — corroborates employment claim.
37. **Cross-forge SSH public key match** — same SSH public key uploaded to GitHub + GitLab + Bitbucket. Proves single private key holder controls all accounts.

### Competitive Programming (LeetCode / Codeforces / AtCoder)
38. **Username consistency + rating history** — rating graph with timestamps is immutable. Rating progression must align with resume timeline.
39. **Contest participation history** — contest dates, ranks, problems solved are public and timestamped.
40. **Platform-specific badges** — LeetCode "Contest Badge", Codeforces rating badges, AtCoder color ranks. Hard to fake without real participation.
41. **Reclaim Protocol / zkProof integration** — emerging cryptographic proofs of LeetCode/Codeforces username ownership (e.g., Reclaim Protocol providers).

### Academic / Research (ORCID / Google Scholar / ResearchGate)
42. **ORCID iD with authenticated works** — ORCID is a persistent researcher identifier; works list is curated by the researcher and linked to DOIs. Two-way sync with Scopus, Web of Science, Crossref.
43. **Google Scholar profile "Verified email" badge** — GS requires institutional email verification; shows "Verified email at domain.edu".
44. **ResearchGate / Academia.edu institutional affiliation verified** — platform verifies .edu/.ac email for affiliation badge.

### Coding Activity / Time Tracking
45. **WakaTime public profile + coding stats** — WakaTime shows languages, editors, OS, daily/weekly totals. Public profile URL can be linked from GitHub README. Stats are derived from IDE plugins — hard to fabricate long-term history.
46. **WakaTime "Display coding activity publicly" toggle** — must be explicitly enabled by account holder. Shows project-level breakdown.

### Container / Package Registries
47. **npm / PyPI / Cargo / Go / pub.dev / conda maintainer record** — package publication dates, owner email/username, and repo links are immutable. Cross-reference with resume project dates.
48. **GitHub Container Registry / Docker Hub / Quay.io image authorship** — image push logs tied to account; repository settings show owning account.
49. **Homebrew / Chocolatey / Scoop formula maintainer** — formula repo commit history shows author.

### Open-Source Foundations / Programs
50. **Apache Foundation / Linux Foundation / CNCF / Eclipse / OpenSSF contributor ID** — foundation-issued, requires identity verification (ICLA, email challenge). Membership list is public.
51. **Google Summer of Code / Outreachy / MLH Fellowship alumni badge** — program-verified participation with dates and project links.
52. **all-contributors bot entries** — automated entries in repo README/contributors list tied to merged PRs. Third-party attestation.

### Social / Secondary Platforms
53. **Twitter/X / LinkedIn / Mastodon bio contains resume-linked URLs** — consistent handles + URLs across platforms. Use as corroboration only.
54. **GitHub Sponsors / Buy Me a Coffee / Patreon / Ko-fi links in profile** — requires account ownership to set up.
55. **Conference speaker profile (dev.events, PyCon, JSConf, etc.)** — speaker bio URLs, talk titles, dates. Third-party event organizer verification.
56. **Meetup.com event organizer / speaker history** — organizer role requires account ownership; event dates align with resume community involvement.

---

## Tier Classification for Scoring

| Tier | Weight | Signals | Rationale |
|------|--------|---------|-----------|
| **S — Cryptographic / Registry** | 5 | 7, 13, 19, 20, 28, 32, 34, 35, 36, 42 | Platform-issued after identity challenge; requires secret control or private registry |
| **A — Platform Metadata** | 3 | 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 33, 37, 38, 39, 40, 43, 44, 45, 46 | Emitted by platform; hard to fabricate at scale |
| **B — Behavioral Consistency** | 2 | 12, 41, 47, 48, 49, 50, 51, 52 | Requires sustained coherent activity over time |
| **C — Domain / Website Anchors** | 3 | 28, 29, 30, 31 | Domain ownership is strong identity anchor when verifiable |
| **D — Social / Second-Order** | 1 | 53, 54, 55, 56 | Corroborative; platforms vary in verification strictness |
| **E — Lightweight** | 0.5 | Username/handle consistency across platforms | Cheap to check; supports but never proves |

---

## Practical Notes
- **Absence ≠ fraud** — many users hide emails, disable public contributions, or use privacy protection. Score what you can observe.
- **Prefer independent corroboration** — two Tier A signals from different platforms > one Tier S signal.
- **Respect rate limits & ToS** — use official APIs where available; scrape only public profile pages with polite delays.
- **Store evidence, not conclusions** — log timestamped API responses / page hashes; let scoring engine apply weights.
- **No single signal is decisive** — the composite score with minimum per-link threshold prevents false positives.
