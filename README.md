# Submission Grader API

A REST API that helps coaches review learner submissions by providing AI-powered rubric-based assessments.

## Approach

### Architecture Overview

The solution follows a clean, modular architecture:

```
src/app/
├── main.py      # FastAPI application and endpoints
├── models.py    # Pydantic models for request/response validation
├── parser.py    # HTML parsing and text extraction
├── grader.py    # LLM-based grading logic
└── config.py    # Configuration management
```

### Key Design Decisions

1. **HTML Structure Analysis**: The parser extracts sections based on heading tags (h1-h6), enabling the grader to understand document organization and evaluate structure quality.

2. **LLM-Powered Grading**: Uses OpenAI's GPT models for sophisticated, context-aware assessment with:
   - Low temperature (0.3) for consistent grading
   - Structured JSON output format
   - Evidence-based feedback to prevent hallucination

3. **Rich Response Format**: The API returns detailed assessments including:
   - Per-criterion grades with evidence and feedback
   - Structure analysis (headings, word count, formatting)
   - Overall grade calculation
   - Strengths and areas for improvement

4. **Fail-Fast Design**: The API requires a valid OpenAI API key and returns specific error messages for different failure scenarios (rate limits, timeouts, auth errors).

5. **Containerized Deployment**: Docker support for consistent, reproducible deployments.

### Reliability and Safety Hardening

- **Validated LLM output**: the model's JSON is parsed into a Pydantic schema (`LLMGrading`) and checked for completeness (every rubric criterion must be graded). On failure, the validation error is fed back to the model for a corrected answer (up to `LLM_MAX_ATTEMPTS`, default 3), then the API returns 502. No more silent default grades.
- **Prompt-injection resistance**: the submission is fenced in `<submission>` tags, the system prompt declares it untrusted data, and a submission cannot close the fence.
- **Runtime evidence check**: quoted passages in each criterion's evidence are verified against the submission text; `evidence_grounded` is `false` when a quote can't be found (possible hallucination), `null` when nothing was quoted.
- **Target-consistent overall grade**: the overall grade is derived from each criterion's `meets_target` (all targets met -> `meets_expectations`, or `exceeds_expectations` if all exceed; any missed target -> `needs_improvement`; any unaddressed criterion -> `not_met` as a hard floor), so it can't contradict the per-criterion flags.
- **Typed error mapping**: OpenAI exceptions map to 429/502/503/504 instead of matching on error-message strings. The client has an explicit timeout and built-in transport retries.

### How Structure Informs Analysis

The parser identifies:
- **Section headings** to verify required components (e.g., "Poem" and "Short story" sections)
- **Document organization** to assess presentation quality
- **Word count** as a basic metric of effort/completeness

This structural information is passed to the grader, which uses it to:
- Locate specific content sections for evaluation
- Provide feedback on organization
- Assess whether the submission follows expected format

## How to Run

### Prerequisites

- Python 3.12+ (or Docker)
- OpenAI API key (required)

### Option 1: Docker (Recommended)

```bash
# Clone and navigate to the project
cd submission-grader

# Copy environment file and add your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# Build and run with Docker Compose
docker-compose up --build

# The API will be available at http://localhost:8000
```

### Option 2: Local Python

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your-key-here

# Run the server
uvicorn src.app.main:app --reload

# The API will be available at http://localhost:8000
```

### Running Tests

```bash
# With Docker
docker-compose --profile test run tests

# Or locally
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Test Coverage

The test suite includes **80 tests** across five categories:

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_api.py` | 16 | API endpoint tests (happy paths, errors, validation) |
| `test_parser.py` | 11 | HTML parsing unit tests |
| `test_grader.py` | 13 | Grading logic unit tests |
| `test_ai_evaluation.py` | 17 | AI-specific evaluation metrics |
| `test_hardening.py` | 23 | Validation/retry, prompt-injection fencing, evidence grounding, error mapping, overall-grade rollup |

### AI Evaluation Tests

These tests evaluate the AI grading system using ML/AI evaluation metrics:

- **Consistency**: Same input produces same/similar grades
- **Evidence Grounding**: BERT embeddings (sentence-transformers) for semantic similarity - catches synonyms, paraphrases, and word forms
- **Discriminative Ability**: Good submissions score higher than poor ones
- **Criterion Coverage**: All rubric criteria are assessed without duplicates
- **Robustness**: Handles edge cases (unicode, minimal content, malformed HTML)

### Evidence Grounding with BERT Embeddings

```
Evidence: "The bird inhabits the wilderness"
Submission: "Parakeets live in the wild Australian outback"

BERT Embedding Similarity → 0.72 (semantic match)
  - "bird" ≈ "parakeets" (same concept)
  - "inhabits" ≈ "live" (synonyms)
  - "wilderness" ≈ "wild outback" (paraphrase)
```

Uses `all-MiniLM-L6-v2` model - local, free, no API calls needed.

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check (includes `llm_configured` status) |
| POST | `/grade` | Grade a submission |

### Error Responses

The API returns specific error messages based on the failure type:

| Status | Scenario | Message |
|--------|----------|---------|
| 400 | Empty submission | "Submission content cannot be empty" |
| 400 | Submission too long | "Submission exceeds maximum length..." |
| 400 | Empty rubric | "Rubric must contain at least one standard" |
| 429 | Rate limited | "Service is temporarily busy..." |
| 503 | Missing API key | "Service configuration error..." |
| 504 | Timeout | "Grading request timed out..." |

## Example Request/Response

### Request

```bash
curl -X POST http://localhost:8000/grade \
  -H "Content-Type: application/json" \
  -d '{
    "submission_html": "<h1>Poem</h1><p>A parakeet so green...</p><h1>Short story</h1><p>In the Australian outback...</p>",
    "rubric": {
      "project_id": "test-project",
      "project_description": "Demonstrate understanding of parakeets",
      "standards": [{
        "id": "std-1",
        "description": "Creative writing about parakeets",
        "tag": "parakeet",
        "grading_primer": "Parakeets are small parrots with long tail feathers.",
        "grading_criteria": [{
          "criterion_id": "crit-1",
          "criterion_text": "Write a short story about parakeets.",
          "criterion_target_grade": "meets_expectations"
        }]
      }]
    }
  }'
```

### Response

```json
{
  "project_id": "test-project",
  "submission_summary": "A 45-word submission about parakeets with 2 section(s).",
  "structure_analysis": {
    "has_clear_sections": true,
    "section_headings": ["Poem", "Short story"],
    "word_count": 45,
    "is_well_formatted": true,
    "structure_notes": "Well-structured with clear headings"
  },
  "standards_assessments": [{
    "standard_id": "std-1",
    "standard_description": "Creative writing about parakeets",
    "tag": "parakeet",
    "criteria_assessments": [{
      "criterion_id": "crit-1",
      "criterion_text": "Write a short story about parakeets.",
      "target_grade": "meets_expectations",
      "assigned_grade": "meets_expectations",
      "meets_target": true,
      "evidence": "Story content present in 'Short story' section",
      "feedback": "A short story about parakeets is present.",
      "suggestions": null
    }],
    "overall_standard_met": true
  }],
  "overall_grade": "meets_expectations",
  "overall_feedback": "Good effort with clear structure.",
  "strengths": ["Well-organized submission", "Includes both poem and story"],
  "areas_for_improvement": []
}
```

## Assumptions and Limitations

### Assumptions

1. **Submission Format**: HTML or plain text input is expected. The parser handles both gracefully.
2. **Rubric Structure**: Follows the provided JSON schema with standards and criteria.
3. **Grade Levels**: Four-tier system (not_met, needs_improvement, meets_expectations, exceeds_expectations).
4. **Language**: English submissions.

### Limitations

1. **LLM Dependency**: Requires a valid OpenAI API key - no offline mode.
2. **Rate Limiting**: No built-in rate limiting (recommend adding for production).
3. **Authentication**: No authentication layer (recommend adding for production).
4. **Persistence**: Stateless design - no storage of submissions or grades.

### Future Improvements

Given more time, I would add:
- Authentication and rate limiting
- Caching for repeated submissions
- Support for additional LLM providers (Anthropic, local models via Ollama)
- Batch grading endpoint
- Async processing for large submissions

## AI Tools Used

This solution was built with assistance from **Claude Code (Claude Opus 4.5)**, Anthropic's AI coding assistant. It was used for:

- **Code Generation**: Writing the initial structure, models, and business logic
- **Test Writing**: Creating comprehensive test cases
- **Documentation**: Generating this README and inline documentation
- **Problem Solving**: Discussing architecture decisions and trade-offs

All AI-generated code was reviewed and understood before inclusion.
