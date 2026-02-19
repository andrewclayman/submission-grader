"""Pydantic models for the submission grader API."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class GradeLevel(str, Enum):
    """Possible grade levels for a criterion."""
    NOT_MET = "not_met"
    NEEDS_IMPROVEMENT = "needs_improvement"
    MEETS_EXPECTATIONS = "meets_expectations"
    EXCEEDS_EXPECTATIONS = "exceeds_expectations"


class GradingCriterion(BaseModel):
    """A single criterion within a rubric standard."""
    criterion_id: str
    criterion_text: str
    criterion_target_grade: str


class RubricStandard(BaseModel):
    """A standard containing multiple grading criteria."""
    id: str
    description: str
    tag: str
    grading_primer: str
    grading_criteria: list[GradingCriterion]


class Rubric(BaseModel):
    """The complete rubric structure."""
    project_id: str
    project_description: str
    standards: list[RubricStandard]


class GradeRequest(BaseModel):
    """Request model for the grading endpoint."""
    submission_html: str = Field(
        ...,
        description="The submission content as HTML or plain text"
    )
    rubric: Rubric = Field(
        ...,
        description="The rubric to grade the submission against"
    )


class CriterionAssessment(BaseModel):
    """Assessment result for a single criterion."""
    criterion_id: str
    criterion_text: str
    target_grade: str
    assigned_grade: GradeLevel
    meets_target: bool
    evidence: str = Field(
        ...,
        description="Specific quotes or references from the submission supporting this grade"
    )
    feedback: str = Field(
        ...,
        description="Constructive feedback for the learner"
    )
    suggestions: Optional[str] = Field(
        None,
        description="Suggestions for improvement if applicable"
    )


class StandardAssessment(BaseModel):
    """Assessment result for a rubric standard."""
    standard_id: str
    standard_description: str
    tag: str
    criteria_assessments: list[CriterionAssessment]
    overall_standard_met: bool


class StructureAnalysis(BaseModel):
    """Analysis of the submission's structure."""
    has_clear_sections: bool
    section_headings: list[str]
    word_count: int
    is_well_formatted: bool
    structure_notes: str


class GradeResponse(BaseModel):
    """Response model for the grading endpoint."""
    project_id: str
    submission_summary: str = Field(
        ...,
        description="Brief summary of what the submission contains"
    )
    structure_analysis: StructureAnalysis
    standards_assessments: list[StandardAssessment]
    overall_grade: GradeLevel
    overall_feedback: str = Field(
        ...,
        description="Overall feedback for the coach to share with the learner"
    )
    strengths: list[str]
    areas_for_improvement: list[str]
