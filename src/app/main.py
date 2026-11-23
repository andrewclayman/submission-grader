"""FastAPI application for the Submission Grader API."""

import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .models import GradeRequest, GradeResponse
from .parser import parse_submission
from .grader import grade_submission, GradingError

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API for grading learner submissions against rubrics using AI-powered assessment.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "llm_configured": bool(settings.openai_api_key),
    }


@app.post("/grade", response_model=GradeResponse)
async def grade(request: GradeRequest) -> GradeResponse:
    """
    Grade a submission against a rubric.

    Accepts a learner submission (HTML or plain text) and a structured rubric,
    returns a detailed assessment with grades, evidence, and feedback for each criterion.
    """
    # Validate submission length
    if len(request.submission_html) > settings.max_submission_length:
        raise HTTPException(
            status_code=400,
            detail=f"Submission exceeds maximum length of {settings.max_submission_length} characters",
        )

    if not request.submission_html.strip():
        raise HTTPException(
            status_code=400,
            detail="Submission content cannot be empty",
        )

    if not request.rubric.standards:
        raise HTTPException(
            status_code=400,
            detail="Rubric must contain at least one standard",
        )

    try:
        # Parse the submission
        parsed = parse_submission(request.submission_html)

        # Grade against rubric
        result = grade_submission(parsed, request.rubric)

        return result

    except GradingError as e:
        # The model kept returning invalid/incomplete output after retries
        raise HTTPException(
            status_code=502,
            detail="Received invalid response from grading service. Please try again.",
        ) from e
    except ValueError as e:
        # Configuration errors (e.g., missing API key)
        raise HTTPException(
            status_code=503,
            detail=f"Service configuration error: {str(e)}",
        )
    except openai.AuthenticationError:
        raise HTTPException(
            status_code=503,
            detail="Authentication error with grading service. Please contact support.",
        )
    except openai.RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Service is temporarily busy. Please try again in a few moments.",
        )
    except openai.APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Grading request timed out. Please try again.",
        )
    except openai.APIConnectionError:
        raise HTTPException(
            status_code=502,
            detail="Could not reach the grading service. Please try again.",
        )
    except Exception as e:
        # For other errors, provide debug details if enabled
        if settings.debug:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while grading. Please try again or contact support.",
        )


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /grade": "Grade a submission against a rubric",
        },
    }
