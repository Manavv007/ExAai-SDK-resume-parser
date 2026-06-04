    Now I have a thorough understanding of the entire codebase. Let me produce the final analysis and schema.



    Codebase Investigation Report

    1. Existing Data Models

    Jobs -- src/lib/types/jobs.ts
    - job_id: UUID (string), job_title: string, interview_duration: number, is_active: boolean, initial_job_desc_json: Record<string, unknown>, final_job_desc_json: JobDetailsV14, organization_id: UUID, author_id: UUID, tags: string[], job_status: "draft"|"published"|"archived", job_visibility: "Public"|"Private", objectives: InterviewObjective[], resume_threshold: number (default -100), application_status: "open"|"closed", created_at: Timestamp, updated_at: Timestamp
    - JobDetailsV14 is the rich JSON schema (Schema_version1.4.json) with coreDetails, responsibilities, qualifications (skills with technical/soft arrays of {skillName}, experience with requiredYears: integer, education), evaluationStrategy, companyContext, compensation (salary min/max/currency), meta (schemaVersion, lastUpdated in ISO date-time)

    Job Applications -- src/lib/types/job-applications.ts
    - application_id: UUID, job_id: UUID, candidate_user_id: UUID, organization_id: UUID, status: JobApplicationStatus, applied_at: Timestamp, resume_url: string, source_text: "self-apply"|"invited"|"prospect", final_score: number, final_feedback: string, decided_by: UUID, resume_similarity_score: ResumeSimilarityScore, resume_screening_status: ResumeScreeningStatus, label_ids: string[], application_notes/candidate_notes/internal_notes: string, email_activity: EmailActivityLog[], various stage timestamps (task_assigned_at, task_submitted_at, interview_scheduled_at, interview_completed_at, reviewed_at, finalized_at), created_at, updated_at

    Candidates -- src/lib/types/user-profiles.ts (referenced as candidate_user_id)
    - user_id: UUID, user_email: string, first_name/last_name: string, phone: string, profile_picture_url: string, resumes: jsonb (array of {url, filename, uploaded_at}), experience: jsonb, bio: string

    Resumes -- src/lib/types/resume.ts
    - Stored as resumes: jsonb[] on user_profiles: {name, size, created_at, updated_at, metadata}
    - Application-level: resume_url: string on job_applications

    Resume Similarity Score -- src/lib/types/job-applications.ts (line 131-134)

    ResumeSimilarityScore = { score: number; reasoning: string; }

    Stored as jsonb column resume_similarity_score on job_applications.

    Evaluations / Interview Reports -- src/lib/utils/interview-utils/types.ts
    - InterviewReport { overallScore: 0-10, candidateSummary, objectivesReport, recommendations: string[], strengths: string[], areasForImprovement, generatedAt }
    - Stored as jsonb column interview_report on interview_sessions
    - Quality tier uses 0-10 scale (final_score), with per-objective scores 0-10
    - Integrity tier uses enum: "low"|"medium"|"high"

    Interview Reviews -- src/lib/types/interview-reviews.ts
    - review_id: bigint, session_id: UUID, candidate_id: UUID, overall_experience_rating: smallint (1-5), additional_comments: text

    Zoom Meetings -- DB schema line 894-922
    - rating_score: integer (1-10), rating_notes: text

    Naming convention: TypeScript types use camelCase. DB columns use snake_case (but the TS layer maps them to camelCase). IDs are UUID (string). Timestamps are ISO 8601 strings (Timestamp = string).

    2. Identifiers & Relationships

    - Jobs: job_id: UUID (PK, gen_random_uuid())
    - Applications: application_id: UUID (PK, gen_random_uuid()), links to job_id, candidate_user_id, organization_id
    - Candidates: user_id: UUID (PK, references auth.users.id)
    - Interviews: session_id: UUID (PK), links to application_id
    - Resume screening results are stored directly on the job_applications table as resume_similarity_score jsonb and resume_screening_status text -- NOT in a separate table.
    - The existing evaluation entity is the interview_report jsonb on interview_sessions, plus interview_reviews table with the 1-5 rating.

    3. Scoring & Status Conventions

    Resume screening score: 0-100 integer (Zod validation: z.number().min(0).max(100) in prospect-resume-screening.ts line 46)

    Interview quality tier score: 0-10 scale (final_score in QualityTierView), computed as average of objective_fulfillment, behavioral_traits, conversational_skills (each 0-10)

    Interview experience rating: 1-5 (smallint, interview_reviews.overall_experience_rating)

    Zoom meeting rating: 1-10 (zoom_meetings.rating_score)

    Application status (DB enum): "prospect"|"applied"|"invited"|"task_assigned"|"task_submitted"|"interview_scheduled"|"interview_inprogress"|"interview_completed"|"under_review"|"accepted"|"rejected"|"withdrawn"

    Resume screening status (DB enum): "queued"|"processing"|"completed"|"failed"

    Integrity tier (TypeScript type): "low"|"medium"|"high"

    Job descriptions use a weight split: evaluationStrategy.jobDescription + evaluationStrategy.resume = 100% (slider UI)

    Recommendation field: evaluationSummary.recommendation -- appears to be a free-text string displayed in the PDF report. No enum found.

    4. Integration Point

    Resume screening results are consumed via REST API -> direct Supabase DB write:

    - Score submission endpoint: POST /api/applications/update-score -- accepts { application_id: UUID, resume_similarity_score: ResumeSimilarityScore } -- src/app/api/applications/update-score/route.ts
    - Alternative: POST /api/applications/prospect/screen-resume -- triggers the full screening pipeline (download resume, extract text, call LLM, write score, update status) -- src/app/api/applications/prospect/screen-resume/route.ts
    - The DB write function: updateApplicationResumeScore(applicationId, resumeSimilarityScore) in src/lib/supabase/job-applications-operations.ts (line 812-832) -- does .update({ resume_similarity_score: resumeSimilarityScore }).eq("application_id", applicationId)
    - Response envelope: { success: boolean, message: string, resumeSimilarityScore?: ResumeSimilarityScore } -- NOT a generic {data, error, meta} pattern. Each route defines its own response type.
    - Validation: Zod schemas in the screening code (BatchResumeMatchSchema), manual validation in API route handlers
    - No JSON Schema / Pydantic / protobuf -- it's a Next.js + Supabase + TypeScript stack with Zod for runtime validation

    5. Error & Metadata Conventions

    Error format in API routes: { success: false, message: string } for success-shaped responses, or { error: string } for simple error responses. HTTP status codes: 400, 401, 403, 404, 409, 500.

    Operation result pattern (DB layer): { data: T | null, error: Error | null } -- see SupabaseOperationResult<T> in common.ts and every function in job-applications-operations.ts.

    Timestamps: ISO 8601 strings. Timestamp = string type alias. DB uses timestamp with time zone. Auto-set via default now() and update_updated_at_column() triggers.

    Audit metadata: credit_operation_logs table for credit operations. email_activity jsonb arrays on applications/sessions. No dedicated audit log for score changes.

    Source URLs: Stored as plain text columns (resume_url, recording_url).



    Proposed JSON Output Schema for Your Screening Agent

    Based on the existing codebase, here is the contract your agent should output. It is designed to plug directly into the existing resume_similarity_score jsonb column and resume_screening_status column on job_applications.

    json
    {
      "$schema": "resume-screening-result-v1",
      "application_id": "string (UUID)",
      "job_id": "string (UUID)",

      "resume_screening_status": "queued | processing | completed | failed",

      "resume_similarity_score": {
        "score": "number (integer, 0-100)",
        "reasoning": "string (single sentence, max 500 chars)"
      },

      "requirement_matches": [
        {
          "requirement": "string",
          "requirement_type": "technical_skill | soft_skill | experience | education | responsibility",
          "match_score": "number (integer, 0-100)",
          "evidence": "string (brief justification)",
          "source_quote": "string (verbatim from resume, optional)"
        }
      ],

      "recommendation": "advance | hold | reject",
      "recommendation_reasoning": "string",

      "red_flags": [
        {
          "flag": "string",
          "severity": "low | medium | high",
          "evidence": "string"
        }
      ],

      "sources_crawled": [
        {
          "url": "string",
          "title": "string (optional)",
          "relevance": "high | medium | low"
        }
      ],

      "metadata": {
        "schema_version": "1.0",
        "model_version": "string (e.g. agent-name/model-tag)",
        "processed_at": "string (ISO 8601 timestamp)",
        "processing_time_ms": "number (optional, integer)",
        "resume_text_chars": "number (integer)",
        "job_desc_version": "string (optional, e.g. 1.4)",
        "agent_version": "string (semver, e.g. 0.1.0)"
      },

      "errors": [
        {
          "code": "string (e.g. EXTRACTION_FAILED, CRAWL_TIMEOUT, LLM_ERROR)",
          "message": "string",
          "source_url": "string (optional)"
        }
      ]
    }


    Field-to-Source Mapping Table

    Field: application_id
    Type: UUID string
    Conforms To / Extends: job_applications.application_id
    Source: src/lib/types/job-applications.ts:42
    Column 5:
    Column 6:
    Column 7:
    ────────────────────────────────────────
    Field: job_id
    Type: UUID string
    Conforms To / Extends: job_applications.job_id
    Source: src/lib/types/job-applications.ts:12
    Column 5:
    Column 6:
    Column 7:
    ────────────────────────────────────────
    Field: resume_screening_status
    Type: enum "queued"\
    Conforms To / Extends: "processing"\
    Source: "completed"\
    Column 5: "failed"
    Column 6: ResumeScreeningStatus type and DB constraint
    Column 7: src/lib/types/job-applications.ts:4-8, migration 20260527124500:180
    ────────────────────────────────────────
    Field: resume_similarity_score.score
    Type: number 0-100
    Conforms To / Extends: ResumeSimilarityScore.score, Zod min(0).max(100)
    Source: src/lib/types/job-applications.ts:132, src/lib/supabase/prospect-resume-screening.ts:46
    Column 5:
    Column 6:
    Column 7:
    ────────────────────────────────────────
    Field: resume_similarity_score.reasoning
    Type: string (max 500 chars)
    Conforms To / Extends: ResumeSimilarityScore.reasoning
    Source: src/lib/types/job-applications.ts:133, screening.ts:20-39
    Column 5:
    Column 6:
    Column 7:
    ────────────────────────────────────────
    Field: requirement_matches
    Type: array of match objects
    Conforms To / Extends: NEW -- not in existing schema. Extends ResumeSimilarityScore with per-requirement breakdown mirroring
      the JD structure
    Source: No existing equivalent. JD requirements are in Schema_version1.4.json and JobDetailsV14
    Column 5:
    Column 6:
    Column 7:
    ────────────────────────────────────────
    Field: requirement_matches[].requirement_type
    Type: enum
    Conforms To / Extends: NEW -- maps to JD section types
    Source: Mirror of JobDetailsV14 structure: qualifications.skills.technical, qualifications.skills.soft,
      qualifications.experience, qualifications.education, responsibilities
    Column 5:
    Column 6:
    Column 7:
    ────────────────────────────────────────
    Field: recommendation
    Type: enum "advance"\
    Conforms To / Extends: "hold"\
    Source: "reject"
    Column 5: NEW -- existing system uses free-text evaluationSummary.recommendation (PDF) and application status enum
    Column 6: No existing enum for screening-level recommendation. Closest: job_applications.status (src/lib/types/common.ts:61-73)
    Column 7:
    ────────────────────────────────────────
    Field: red_flags
    Type: array
    Conforms To / Extends: NEW -- no existing red flag structure
    Source: Partial parallel to integrity.flags in IntegrityFlag (src/lib/types/evaluation-tiers.ts:44-48)
    Column 5:
    Column 6:
    Column 7:
    ────────────────────────────────────────
    Field: sources_crawled
    Type: array
    Conforms To / Extends: NEW -- your Exa AI crawl results
    Source: No existing source tracking for resume screening
    Column 5:
    Column 6:
    Column 7:
    ────────────────────────────────────────
    Field: metadata.schema_version
    Type: string
    Conforms To / Extends: Follows jobProfile.meta.schemaVersion convention
    Source: json schemas/Schema_version1.4.json:184-186
    Column 5:
    Column 6:
    Column 7:
    ────────────────────────────────────────
    Field: metadata.model_version
    Type: string
    Conforms To / Extends: NEW -- existing system does not track model version on scores
    Source: No existing equivalent
    Column 5:
    Column 6:
    Column 7:
    ────────────────────────────────────────
    Field: metadata.processed_at
    Type: ISO 8601 string
    Conforms To / Extends: Follows Timestamp = string and meta.lastUpdated conventions
    Source: src/lib/types/common.ts:2, Schema_version1.4.json:188-190
    Column 5:
    Column 6:
    Column 7:
    ────────────────────────────────────────
    Field: metadata.job_desc_version
    Type: string
    Conforms To / Extends: Mirrors JobDetailsV14.meta.schemaVersion
    Source: src/lib/types/jobs.ts:70
    Column 5:
    Column 6:
    Column 7:
    ────────────────────────────────────────
    Field: errors
    Type: array
    Conforms To / Extends: NEW -- partial parallel to ResumeParseResponse.error
    Source: src/lib/types/resume.ts:131-134
    Column 5:
    Column 6:
    Column 7:

    Gaps / Things to Add

    1. requirement_matches -- The existing ResumeSimilarityScore only has score + reasoning. You will need to extend this type. I recommend either:
       - Extending ResumeSimilarityScore with an optional matches array, or
       - Creating a new type DetailedResumeSimilarityScore extends ResumeSimilarityScore with the extra fields, and writing a DB migration to update the column type comment.

    2. recommendation enum -- No existing enum at the recommendation level. The application status enum handles workflow state (accepted/rejected), not screening recommendation. You should add this as a new column or embed it in a new jsonb field on job_applications.

    3. red_flags -- No existing structure for this. Consider whether these should be stored in resume_similarity_score jsonb or in a separate column.

    4. errors array -- The existing pattern is to represent failures through resume_screening_status: "failed" with a message string. An errors array in the jsonb is an extension.

    5. sources_crawled -- Entirely new. This is your Exa AI specific data. Store it in the resume_similarity_score jsonb or add a new column.

    Recommended Integration Approach

    The cleanest path: Your agent writes a ResumeSimilarityScore-conformant { score, reasoning } to the existing resume_similarity_score column and sets resume_screening_status to "completed" (or "failed" on error). Store your extended data (requirement_matches, red_flags, sources_crawled) by extending the ResumeSimilarityScore type:

    typescript
    // Proposed extension in src/lib/types/job-applications.ts
    export type ResumeSimilarityScore = {
      score: number;
      reasoning: string;
      // Extended fields for detailed screening:
      matches?: RequirementMatch[];
      recommendation?: ScreeningRecommendation;
      redFlags?: RedFlag[];
      sources?: CrawledSource[];
      errors?: ScreeningError[];
      meta?: ScreeningMetadata;
    };
