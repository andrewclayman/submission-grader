"""Tests for the grading module."""

import pytest

from src.app.grader import (
    build_response,
    create_grading_prompt,
    grade_submission,
)
from src.app.parser import parse_submission
from src.app.models import Rubric, GradeLevel


@pytest.fixture
def rubric_model(sample_rubric) -> Rubric:
    """Convert sample rubric dict to Rubric model."""
    return Rubric(**sample_rubric)


@pytest.fixture
def mock_llm_result() -> dict:
    """Sample LLM result for testing build_response."""
    return {
        "submission_summary": "A submission about parakeets with poem and story.",
        "criteria_grades": [
            {
                "criterion_id": "e678ce81-98e4-44a8-98a5-5ac4803d2643",
                "assigned_grade": "meets_expectations",
                "evidence": "Story about parakeets present",
                "feedback": "Good story.",
                "suggestions": None,
            },
            {
                "criterion_id": "163ecfd8-4df6-4af2-b5e6-5463b258a785",
                "assigned_grade": "meets_expectations",
                "evidence": "Limerick present",
                "feedback": "Nice poem.",
                "suggestions": None,
            },
            {
                "criterion_id": "d18e3f3a-3f2b-47db-b408-ded7e76b6c3a",
                "assigned_grade": "exceeds_expectations",
                "evidence": "Australian outback mentioned",
                "feedback": "Excellent habitat detail.",
                "suggestions": None,
            },
        ],
        "overall_feedback": "Excellent work.",
        "strengths": ["Creative writing", "Good structure"],
        "areas_for_improvement": [],
    }


class TestGradeSubmission:
    """Tests for the grade_submission function."""

    def test_grade_submission_calls_llm(self, good_submission_html, rubric_model, mock_openai):
        """grade_submission should call OpenAI and return response."""
        parsed = parse_submission(good_submission_html)
        result = grade_submission(parsed, rubric_model)

        # Verify OpenAI was called
        mock_openai.chat.completions.create.assert_called_once()

        # Verify response structure
        assert result.project_id == rubric_model.project_id
        assert len(result.standards_assessments) == 1

    def test_grade_submission_returns_valid_grades(self, good_submission_html, rubric_model, mock_openai):
        """grade_submission should return valid grade levels."""
        parsed = parse_submission(good_submission_html)
        result = grade_submission(parsed, rubric_model)

        assert result.overall_grade in [
            GradeLevel.NOT_MET,
            GradeLevel.NEEDS_IMPROVEMENT,
            GradeLevel.MEETS_EXPECTATIONS,
            GradeLevel.EXCEEDS_EXPECTATIONS,
        ]


class TestBuildResponse:
    """Tests for building the final response."""

    def test_build_response_structure(self, good_submission_html, rubric_model, mock_llm_result):
        """Build response should create proper structure."""
        parsed = parse_submission(good_submission_html)
        response = build_response(rubric_model, parsed, mock_llm_result)

        assert response.project_id == rubric_model.project_id
        assert len(response.standards_assessments) == 1
        assert len(response.standards_assessments[0].criteria_assessments) == 3

    def test_build_response_calculates_meets_target(self, good_submission_html, rubric_model, mock_llm_result):
        """Build response should calculate if criteria meet targets."""
        parsed = parse_submission(good_submission_html)
        response = build_response(rubric_model, parsed, mock_llm_result)

        for standard in response.standards_assessments:
            for criterion in standard.criteria_assessments:
                # meets_target should be a boolean
                assert isinstance(criterion.meets_target, bool)

    def test_build_response_overall_grade(self, good_submission_html, rubric_model, mock_llm_result):
        """Build response should calculate overall grade."""
        parsed = parse_submission(good_submission_html)
        response = build_response(rubric_model, parsed, mock_llm_result)

        assert response.overall_grade in [
            GradeLevel.NOT_MET,
            GradeLevel.NEEDS_IMPROVEMENT,
            GradeLevel.MEETS_EXPECTATIONS,
            GradeLevel.EXCEEDS_EXPECTATIONS,
        ]

    def test_build_response_includes_structure_analysis(self, good_submission_html, rubric_model, mock_llm_result):
        """Build response should include structure analysis."""
        parsed = parse_submission(good_submission_html)
        response = build_response(rubric_model, parsed, mock_llm_result)

        assert response.structure_analysis.has_clear_sections is True
        assert response.structure_analysis.word_count > 0
        assert len(response.structure_analysis.section_headings) > 0

    def test_build_response_all_exceeds_gives_exceeds(self, good_submission_html, rubric_model):
        """If all criteria exceed expectations, overall should be exceeds."""
        parsed = parse_submission(good_submission_html)
        llm_result = {
            "submission_summary": "Excellent submission.",
            "criteria_grades": [
                {"criterion_id": "e678ce81-98e4-44a8-98a5-5ac4803d2643", "assigned_grade": "exceeds_expectations", "evidence": "x", "feedback": "x", "suggestions": None},
                {"criterion_id": "163ecfd8-4df6-4af2-b5e6-5463b258a785", "assigned_grade": "exceeds_expectations", "evidence": "x", "feedback": "x", "suggestions": None},
                {"criterion_id": "d18e3f3a-3f2b-47db-b408-ded7e76b6c3a", "assigned_grade": "exceeds_expectations", "evidence": "x", "feedback": "x", "suggestions": None},
            ],
            "overall_feedback": "Excellent.",
            "strengths": ["All criteria exceeded"],
            "areas_for_improvement": [],
        }
        response = build_response(rubric_model, parsed, llm_result)
        assert response.overall_grade == GradeLevel.EXCEEDS_EXPECTATIONS

    def test_build_response_any_not_met_gives_not_met(self, good_submission_html, rubric_model):
        """If any criterion is not_met, overall should be not_met."""
        parsed = parse_submission(good_submission_html)
        llm_result = {
            "submission_summary": "Incomplete submission.",
            "criteria_grades": [
                {"criterion_id": "e678ce81-98e4-44a8-98a5-5ac4803d2643", "assigned_grade": "meets_expectations", "evidence": "x", "feedback": "x", "suggestions": None},
                {"criterion_id": "163ecfd8-4df6-4af2-b5e6-5463b258a785", "assigned_grade": "not_met", "evidence": "x", "feedback": "x", "suggestions": "Add poem"},
                {"criterion_id": "d18e3f3a-3f2b-47db-b408-ded7e76b6c3a", "assigned_grade": "meets_expectations", "evidence": "x", "feedback": "x", "suggestions": None},
            ],
            "overall_feedback": "Missing requirements.",
            "strengths": [],
            "areas_for_improvement": ["Poem missing"],
        }
        response = build_response(rubric_model, parsed, llm_result)
        assert response.overall_grade == GradeLevel.NOT_MET


class TestGradingPrompt:
    """Tests for the LLM prompt creation."""

    def test_prompt_includes_submission(self, rubric_model):
        """Prompt should include the submission text."""
        submission = "My test submission about parakeets."
        prompt = create_grading_prompt(submission, rubric_model)

        assert submission in prompt

    def test_prompt_includes_criteria(self, rubric_model):
        """Prompt should include all criteria."""
        submission = "Test submission"
        prompt = create_grading_prompt(submission, rubric_model)

        for standard in rubric_model.standards:
            for criterion in standard.grading_criteria:
                assert criterion.criterion_text in prompt

    def test_prompt_includes_grade_levels(self, rubric_model):
        """Prompt should describe grade levels."""
        submission = "Test submission"
        prompt = create_grading_prompt(submission, rubric_model)

        assert "not_met" in prompt
        assert "needs_improvement" in prompt
        assert "meets_expectations" in prompt
        assert "exceeds_expectations" in prompt

    def test_prompt_requests_json_format(self, rubric_model):
        """Prompt should request JSON output format."""
        submission = "Test submission"
        prompt = create_grading_prompt(submission, rubric_model)

        assert "JSON" in prompt
        assert "evidence" in prompt.lower()
        assert "feedback" in prompt.lower()

    def test_prompt_includes_grading_primer(self, rubric_model):
        """Prompt should include the grading primer context."""
        submission = "Test submission"
        prompt = create_grading_prompt(submission, rubric_model)

        assert "parakeet" in prompt.lower()
        assert "parrot" in prompt.lower()  # From grading primer
