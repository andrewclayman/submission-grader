"""Grading service using LLM for assessment."""

import json
from openai import OpenAI

from .models import (
    Rubric,
    GradeResponse,
    GradeLevel,
    CriterionAssessment,
    StandardAssessment,
    StructureAnalysis,
)
from .parser import ParsedSubmission, extract_text_for_grading
from .config import get_settings


def create_grading_prompt(submission_text: str, rubric: Rubric) -> str:
    """Create the prompt for the LLM to grade the submission."""

    criteria_text = ""
    for standard in rubric.standards:
        criteria_text += f"\n### Standard: {standard.description}\n"
        criteria_text += f"Context: {standard.grading_primer}\n"
        criteria_text += "Criteria:\n"
        for criterion in standard.grading_criteria:
            criteria_text += f"- [{criterion.criterion_id}] {criterion.criterion_text} "
            criteria_text += f"(Target: {criterion.criterion_target_grade})\n"

    prompt = f"""You are an expert educational assessor. Evaluate the following learner submission against the provided rubric.

## Submission Content:
{submission_text}

## Rubric:
Project: {rubric.project_description}
{criteria_text}

## Your Task:
Assess each criterion and provide:
1. An assigned grade: "not_met", "needs_improvement", "meets_expectations", or "exceeds_expectations"
2. Specific evidence from the submission (direct quotes or references)
3. Constructive feedback
4. Suggestions for improvement (if not meeting/exceeding expectations)

## Grading Guidelines:
- "not_met": The criterion is not addressed at all
- "needs_improvement": Partial attempt but significant gaps
- "meets_expectations": Adequately addresses the criterion
- "exceeds_expectations": Goes above and beyond the requirements

Respond with a JSON object matching this exact structure:
{{
    "submission_summary": "Brief summary of the submission",
    "criteria_grades": [
        {{
            "criterion_id": "the criterion ID",
            "assigned_grade": "one of the grade levels",
            "evidence": "specific quotes or references from submission",
            "feedback": "constructive feedback",
            "suggestions": "improvement suggestions or null"
        }}
    ],
    "overall_feedback": "Overall feedback for the learner",
    "strengths": ["list", "of", "strengths"],
    "areas_for_improvement": ["list", "of", "areas"]
}}

Important: Base your assessment ONLY on the content provided. Quote specific text as evidence."""

    return prompt


def grade_with_llm(submission_text: str, rubric: Rubric) -> dict:
    """Call the LLM to grade the submission."""
    settings = get_settings()

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = create_grading_prompt(submission_text, rubric)

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert educational assessor. Always respond with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,  # Lower temperature for more consistent grading
        response_format={"type": "json_object"},
    )

    result = response.choices[0].message.content
    return json.loads(result)


def build_response(
    rubric: Rubric,
    parsed: ParsedSubmission,
    llm_result: dict,
) -> GradeResponse:
    """Build the final GradeResponse from LLM results."""

    # Build structure analysis
    structure_analysis = StructureAnalysis(
        has_clear_sections=parsed.has_structure,
        section_headings=parsed.headings,
        word_count=parsed.word_count,
        is_well_formatted=parsed.has_structure and len(parsed.headings) > 0,
        structure_notes=(
            "Well-structured with clear headings"
            if parsed.has_structure
            else "Lacks clear structure; consider adding section headings"
        ),
    )

    # Build criteria assessments grouped by standard
    standards_assessments = []
    criteria_by_id = {cg["criterion_id"]: cg for cg in llm_result["criteria_grades"]}

    for standard in rubric.standards:
        criteria_assessments = []
        all_met = True

        for criterion in standard.grading_criteria:
            cg = criteria_by_id.get(criterion.criterion_id, {})
            assigned_grade = GradeLevel(cg.get("assigned_grade", "needs_improvement"))

            # Determine if criterion meets target
            grade_order = [
                GradeLevel.NOT_MET,
                GradeLevel.NEEDS_IMPROVEMENT,
                GradeLevel.MEETS_EXPECTATIONS,
                GradeLevel.EXCEEDS_EXPECTATIONS,
            ]
            target_grade = GradeLevel(criterion.criterion_target_grade)
            meets_target = grade_order.index(assigned_grade) >= grade_order.index(target_grade)

            if not meets_target:
                all_met = False

            criteria_assessments.append(
                CriterionAssessment(
                    criterion_id=criterion.criterion_id,
                    criterion_text=criterion.criterion_text,
                    target_grade=criterion.criterion_target_grade,
                    assigned_grade=assigned_grade,
                    meets_target=meets_target,
                    evidence=cg.get("evidence", "No evidence provided"),
                    feedback=cg.get("feedback", "No feedback provided"),
                    suggestions=cg.get("suggestions"),
                )
            )

        standards_assessments.append(
            StandardAssessment(
                standard_id=standard.id,
                standard_description=standard.description,
                tag=standard.tag,
                criteria_assessments=criteria_assessments,
                overall_standard_met=all_met,
            )
        )

    # Calculate overall grade
    all_grades = [
        ca.assigned_grade
        for sa in standards_assessments
        for ca in sa.criteria_assessments
    ]

    if all(g == GradeLevel.EXCEEDS_EXPECTATIONS for g in all_grades):
        overall_grade = GradeLevel.EXCEEDS_EXPECTATIONS
    elif all(g in [GradeLevel.MEETS_EXPECTATIONS, GradeLevel.EXCEEDS_EXPECTATIONS] for g in all_grades):
        overall_grade = GradeLevel.MEETS_EXPECTATIONS
    elif any(g == GradeLevel.NOT_MET for g in all_grades):
        overall_grade = GradeLevel.NOT_MET
    else:
        overall_grade = GradeLevel.NEEDS_IMPROVEMENT

    return GradeResponse(
        project_id=rubric.project_id,
        submission_summary=llm_result["submission_summary"],
        structure_analysis=structure_analysis,
        standards_assessments=standards_assessments,
        overall_grade=overall_grade,
        overall_feedback=llm_result["overall_feedback"],
        strengths=llm_result["strengths"],
        areas_for_improvement=llm_result["areas_for_improvement"],
    )


def grade_submission(parsed: ParsedSubmission, rubric: Rubric) -> GradeResponse:
    """
    Grade a submission against a rubric using LLM.

    Raises:
        ValueError: If OpenAI API key is not configured.
    """
    settings = get_settings()

    if not settings.openai_api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

    submission_text = extract_text_for_grading(parsed)
    llm_result = grade_with_llm(submission_text, rubric)

    return build_response(rubric, parsed, llm_result)
