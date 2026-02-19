"""Pytest fixtures for the submission grader tests."""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

import os
# Clean up any old mock mode env var and set API key
os.environ.pop("USE_MOCK_LLM", None)
os.environ["OPENAI_API_KEY"] = "test-api-key"

# Clear settings cache before importing app
from src.app.config import get_settings
get_settings.cache_clear()

from src.app.main import app
from src.app.models import Rubric, GradingCriterion, RubricStandard


def create_mock_llm_response(criteria_ids: list[str], good_submission: bool = True) -> dict:
    """Create a mock LLM response for testing."""
    criteria_grades = []
    for cid in criteria_ids:
        if good_submission:
            criteria_grades.append({
                "criterion_id": cid,
                "assigned_grade": "meets_expectations",
                "evidence": "Mock evidence from submission",
                "feedback": "Good work on this criterion.",
                "suggestions": None,
            })
        else:
            criteria_grades.append({
                "criterion_id": cid,
                "assigned_grade": "needs_improvement",
                "evidence": "Limited evidence found",
                "feedback": "This area needs more work.",
                "suggestions": "Add more detail and structure.",
            })

    return {
        "submission_summary": "A submission about parakeets.",
        "criteria_grades": criteria_grades,
        "overall_feedback": "Good effort overall." if good_submission else "Needs improvement.",
        "strengths": ["Creative writing"] if good_submission else ["Attempted the assignment"],
        "areas_for_improvement": [] if good_submission else ["Structure", "Detail"],
    }


@pytest.fixture(autouse=True)
def mock_openai():
    """Mock OpenAI client for all tests."""
    with patch("src.app.grader.OpenAI") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Default response for good submissions
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''{
            "submission_summary": "A submission about parakeets with poem and story.",
            "criteria_grades": [
                {"criterion_id": "e678ce81-98e4-44a8-98a5-5ac4803d2643", "assigned_grade": "meets_expectations", "evidence": "Story about Pip the parakeet", "feedback": "Good story.", "suggestions": null},
                {"criterion_id": "163ecfd8-4df6-4af2-b5e6-5463b258a785", "assigned_grade": "meets_expectations", "evidence": "Limerick present", "feedback": "Nice poem.", "suggestions": null},
                {"criterion_id": "d18e3f3a-3f2b-47db-b408-ded7e76b6c3a", "assigned_grade": "exceeds_expectations", "evidence": "Australian outback, grasslands, eucalyptus", "feedback": "Excellent habitat detail.", "suggestions": null}
            ],
            "overall_feedback": "Excellent work meeting all criteria.",
            "strengths": ["Creative poem", "Detailed story", "Natural habitat included"],
            "areas_for_improvement": []
        }'''

        mock_client.chat.completions.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_rubric() -> dict:
    """Sample rubric matching the provided example."""
    return {
        "project_id": "f1629c3c-b988-42ae-bd4b-0d3af588f001",
        "project_description": "Demonstrate your understanding of the parakeet",
        "standards": [
            {
                "id": "a8cf2f27-ccb5-4d8b-8131-c1920781bfb8",
                "description": "A poem and a short story",
                "tag": "parakeet",
                "grading_primer": "A parakeet is any one of many small- to medium-sized species of parrot.",
                "grading_criteria": [
                    {
                        "criterion_id": "e678ce81-98e4-44a8-98a5-5ac4803d2643",
                        "criterion_text": "Write a short story about parakeets.",
                        "criterion_target_grade": "meets_expectations",
                    },
                    {
                        "criterion_id": "163ecfd8-4df6-4af2-b5e6-5463b258a785",
                        "criterion_text": "Write a limerick or a short poem about parakeets.",
                        "criterion_target_grade": "meets_expectations",
                    },
                    {
                        "criterion_id": "d18e3f3a-3f2b-47db-b408-ded7e76b6c3a",
                        "criterion_text": "Describe the natural habitat of a parakeet within the poem or story.",
                        "criterion_target_grade": "exceeds_expectations",
                    },
                ],
            }
        ],
    }


@pytest.fixture
def good_submission_html() -> str:
    """Well-structured submission that should pass all criteria."""
    return """<html>
    <head><title>Parakeets are awesome</title></head>
    <body>
        <h1>Poem</h1>
        <p>
            A parakeet, vibrant and green,
            Made the loudest squawk ever heard, keen.
            It loved to eat seed,
            A feathery creed,
            And bossed all the birds in the scene.
        </p>

        <h1>Short story</h1>
        <p>
            The air over the sun-scorched Australian outback shimmered with heat,
            but high above, a flash of emerald and gold cut through the glare.
            This was the home of Pip, a budgerigar—or common parakeet—who lived
            among hundreds of his kin. Their natural habitat, the arid and semi-arid
            grasslands of Australia, stretched out beneath them, a vast landscape
            dotted with hardy eucalyptus trees and low shrubs.
        </p>
        <p>
            Life revolved around two critical resources: water and seed. Every morning,
            the flock would explode from their nesting hollows with a roar of wings,
            painting the sky in a breathtaking, undulating green wave.
        </p>
    </body>
</html>"""


@pytest.fixture
def poor_submission_html() -> str:
    """Poorly structured submission that needs improvement."""
    return """<html>
    <head><title>Parakeets...</title></head>
    <body>
        A parakeet, quite fluffy and small, Lived inside a bright, colourful hall.
        He tried hard to sing, But just made a ping, He flew away to his food dish.

        Pipsqueek, a brilliant blue and yellow chap, were Basil's favourite companion.
        They spend their entire day chatterring about the new swing Pipsqueek had
        organised for their shared colourful play area. Both of them, they is very
        fond of crunchy millet sprays.

        TODO: Write a bit more about the birdz l8ter.
    </body>
</html>"""


@pytest.fixture
def empty_submission_html() -> str:
    """Empty submission."""
    return "<html><body></body></html>"


@pytest.fixture
def plain_text_submission() -> str:
    """Plain text submission without HTML structure."""
    return """My Parakeet Poem

A little green bird sat on a tree,
In the Australian outback, wild and free.
It chirped all day long,
Singing its song,
The happiest parakeet you'd ever see.

My Parakeet Story

In the grasslands of Australia, a young parakeet named Sunny
lived with her flock. They nested in eucalyptus trees and
flew across the vast natural landscape searching for seeds."""
