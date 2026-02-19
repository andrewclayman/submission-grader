"""Tests for the FastAPI endpoints."""

import pytest
from fastapi import status


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_returns_200(self, client):
        """Health endpoint should return 200 status."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

    def test_health_check_returns_status(self, client):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "llm_configured" in data


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_api_info(self, client):
        """Root endpoint should return API information."""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestGradeEndpoint:
    """Tests for the /grade endpoint."""

    def test_grade_good_submission(self, client, good_submission_html, sample_rubric):
        """Grade endpoint should successfully grade a well-structured submission."""
        response = client.post(
            "/grade",
            json={
                "submission_html": good_submission_html,
                "rubric": sample_rubric,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check response structure
        assert "project_id" in data
        assert "submission_summary" in data
        assert "structure_analysis" in data
        assert "standards_assessments" in data
        assert "overall_grade" in data
        assert "overall_feedback" in data
        assert "strengths" in data
        assert "areas_for_improvement" in data

    def test_grade_poor_submission(self, client, poor_submission_html, sample_rubric):
        """Grade endpoint should handle poorly structured submissions."""
        response = client.post(
            "/grade",
            json={
                "submission_html": poor_submission_html,
                "rubric": sample_rubric,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Poor submission should have structure issues noted
        assert data["structure_analysis"]["has_clear_sections"] is False

    def test_grade_plain_text(self, client, plain_text_submission, sample_rubric):
        """Grade endpoint should handle plain text submissions."""
        response = client.post(
            "/grade",
            json={
                "submission_html": plain_text_submission,
                "rubric": sample_rubric,
            },
        )
        assert response.status_code == status.HTTP_200_OK

    def test_grade_returns_criteria_assessments(self, client, good_submission_html, sample_rubric):
        """Grade endpoint should return assessments for all criteria."""
        response = client.post(
            "/grade",
            json={
                "submission_html": good_submission_html,
                "rubric": sample_rubric,
            },
        )
        data = response.json()

        # Should have one standard assessment
        assert len(data["standards_assessments"]) == 1

        # Standard should have 3 criteria assessments
        standard = data["standards_assessments"][0]
        assert len(standard["criteria_assessments"]) == 3

        # Each criterion should have required fields
        for criterion in standard["criteria_assessments"]:
            assert "criterion_id" in criterion
            assert "assigned_grade" in criterion
            assert "evidence" in criterion
            assert "feedback" in criterion
            assert "meets_target" in criterion

    def test_grade_empty_submission_fails(self, client, sample_rubric):
        """Grade endpoint should reject empty submissions."""
        response = client.post(
            "/grade",
            json={
                "submission_html": "",
                "rubric": sample_rubric,
            },
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_grade_whitespace_submission_fails(self, client, sample_rubric):
        """Grade endpoint should reject whitespace-only submissions."""
        response = client.post(
            "/grade",
            json={
                "submission_html": "   \n\t  ",
                "rubric": sample_rubric,
            },
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_grade_empty_rubric_fails(self, client, good_submission_html):
        """Grade endpoint should reject rubrics with no standards."""
        response = client.post(
            "/grade",
            json={
                "submission_html": good_submission_html,
                "rubric": {
                    "project_id": "test",
                    "project_description": "test",
                    "standards": [],
                },
            },
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_grade_invalid_json_fails(self, client):
        """Grade endpoint should reject invalid JSON."""
        response = client.post(
            "/grade",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_grade_missing_fields_fails(self, client):
        """Grade endpoint should reject requests missing required fields."""
        response = client.post(
            "/grade",
            json={"submission_html": "some content"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_grade_valid_grades_returned(self, client, good_submission_html, sample_rubric):
        """Grade endpoint should return valid grade values."""
        response = client.post(
            "/grade",
            json={
                "submission_html": good_submission_html,
                "rubric": sample_rubric,
            },
        )
        data = response.json()

        valid_grades = ["not_met", "needs_improvement", "meets_expectations", "exceeds_expectations"]

        assert data["overall_grade"] in valid_grades

        for standard in data["standards_assessments"]:
            for criterion in standard["criteria_assessments"]:
                assert criterion["assigned_grade"] in valid_grades


class TestStructureAnalysis:
    """Tests for structure analysis in grade responses."""

    def test_structured_submission_detected(self, client, good_submission_html, sample_rubric):
        """Good submission should be detected as well-structured."""
        response = client.post(
            "/grade",
            json={
                "submission_html": good_submission_html,
                "rubric": sample_rubric,
            },
        )
        data = response.json()
        structure = data["structure_analysis"]

        assert structure["has_clear_sections"] is True
        assert len(structure["section_headings"]) > 0
        assert structure["word_count"] > 0

    def test_unstructured_submission_detected(self, client, poor_submission_html, sample_rubric):
        """Poor submission should be detected as lacking structure."""
        response = client.post(
            "/grade",
            json={
                "submission_html": poor_submission_html,
                "rubric": sample_rubric,
            },
        )
        data = response.json()
        structure = data["structure_analysis"]

        assert structure["has_clear_sections"] is False
        assert len(structure["section_headings"]) == 0

    def test_word_count_calculated(self, client, good_submission_html, sample_rubric):
        """Word count should be calculated correctly."""
        response = client.post(
            "/grade",
            json={
                "submission_html": good_submission_html,
                "rubric": sample_rubric,
            },
        )
        data = response.json()

        # Should have reasonable word count for this submission
        assert data["structure_analysis"]["word_count"] > 100
