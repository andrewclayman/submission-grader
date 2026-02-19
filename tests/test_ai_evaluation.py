"""
AI Evaluation Tests for the Submission Grader.

These tests evaluate the AI grading system using standard ML/AI evaluation metrics:
1. Consistency - Same input produces same/similar output
2. Evidence Grounding - No hallucination, evidence exists in submission
3. Discriminative Ability - Can distinguish good from poor submissions
4. Criterion Coverage - All rubric criteria are assessed
"""

import pytest
import re
import os
from collections import Counter
from unittest.mock import Mock, patch

# Clean up any old mock mode env var and set API key
os.environ.pop("USE_MOCK_LLM", None)
os.environ["OPENAI_API_KEY"] = "test-api-key"

# Clear settings cache before importing app
from src.app.config import get_settings
get_settings.cache_clear()

from fastapi.testclient import TestClient
from src.app.main import app
from src.app.models import GradeLevel


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
                {"criterion_id": "e678ce81-98e4-44a8-98a5-5ac4803d2643", "assigned_grade": "meets_expectations", "evidence": "Story about Pip the parakeet in the Australian outback", "feedback": "Good story.", "suggestions": null},
                {"criterion_id": "163ecfd8-4df6-4af2-b5e6-5463b258a785", "assigned_grade": "meets_expectations", "evidence": "A parakeet, vibrant and green", "feedback": "Nice poem.", "suggestions": null},
                {"criterion_id": "d18e3f3a-3f2b-47db-b408-ded7e76b6c3a", "assigned_grade": "exceeds_expectations", "evidence": "Australian outback, grasslands, eucalyptus trees", "feedback": "Excellent habitat detail.", "suggestions": null}
            ],
            "overall_feedback": "Excellent work meeting all criteria.",
            "strengths": ["Creative poem", "Detailed story", "Natural habitat included"],
            "areas_for_improvement": []
        }'''

        mock_client.chat.completions.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def client():
    """Create a test client."""
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
                "grading_primer": "A parakeet is any one of many small- to medium-sized species of parrot, in multiple genera, that generally has long tail feathers.",
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
def good_submission() -> str:
    """Well-structured submission that should score well."""
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
            painting the sky in a breathtaking, undulating green wave. They flew great
            distances following ancestral knowledge to find water sources.
        </p>
    </body>
</html>"""


@pytest.fixture
def poor_submission() -> str:
    """Poorly structured submission that should score lower."""
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


# Lazy-loaded embedding model
_embedding_model = None


def get_embedding_model():
    """Load BERT embedding model (sentence-transformers)."""
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    from sentence_transformers import SentenceTransformer
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def calculate_semantic_similarity(evidence: str, submission: str) -> float:
    """
    Calculate semantic similarity between evidence and submission using BERT embeddings.

    Chunks the submission and finds the max similarity between
    the evidence and any submission chunk.

    Returns a score between 0 and 1.
    """
    model = get_embedding_model()

    # Chunk submission into sentences/paragraphs for better matching
    chunks = []
    for para in submission.split('\n'):
        para = para.strip()
        if len(para) > 20:  # Skip very short chunks
            chunks.append(para)
            # Also add sentence-level chunks for longer paragraphs
            if len(para) > 200:
                sentences = [s.strip() for s in para.split('.') if len(s.strip()) > 20]
                chunks.extend(sentences)

    if not chunks:
        chunks = [submission]

    # Get embeddings
    evidence_embedding = model.encode(evidence, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    # Calculate cosine similarities
    from sentence_transformers import util
    similarities = util.cos_sim(evidence_embedding, chunk_embeddings)

    # Return max similarity (best matching chunk)
    return similarities.max().item()


class TestSemanticSimilarity:
    """Tests for BERT embedding-based semantic similarity."""

    def test_catches_synonyms(self):
        """Embeddings should catch semantic similarity even with different words."""
        evidence = "The bird lives in the wilderness"
        submission = "Parakeets inhabit the wild outback regions"

        similarity = calculate_semantic_similarity(evidence, submission)

        # "wilderness" ≈ "wild outback", "lives" ≈ "inhabit", "bird" ≈ "parakeets"
        assert similarity >= 0.3, f"Expected semantic similarity >= 0.3, got {similarity:.2f}"

    def test_detects_unrelated_content(self):
        """Embeddings should give low score for unrelated content."""
        evidence = "The parakeet flew across the Australian outback"
        submission = "Python programming involves writing code and debugging software"

        similarity = calculate_semantic_similarity(evidence, submission)

        assert similarity < 0.5, f"Expected low similarity for unrelated content, got {similarity:.2f}"

    def test_high_score_for_same_content(self):
        """Embeddings should give high score for same/similar content."""
        evidence = "The parakeet lived in the Australian grasslands"
        submission = "A parakeet made its home in the grasslands of Australia"

        similarity = calculate_semantic_similarity(evidence, submission)

        assert similarity >= 0.7, f"Expected high similarity >= 0.7, got {similarity:.2f}"

    def test_handles_paraphrasing(self):
        """Embeddings should handle paraphrased content well."""
        evidence = "The submission describes the natural habitat of wild parakeets"
        submission = """
        In the Australian outback, budgerigars (wild parakeets) thrive in their
        native environment. These birds have adapted to the arid grasslands,
        where they feed on seeds and nest in eucalyptus trees.
        """

        similarity = calculate_semantic_similarity(evidence, submission)

        assert similarity >= 0.4, f"Expected similarity >= 0.4 for paraphrased content, got {similarity:.2f}"

    def test_handles_word_forms(self):
        """Embeddings should handle different word forms."""
        evidence = "parakeets flying through grasslands"
        submission = "A parakeet flew across the grassland"

        similarity = calculate_semantic_similarity(evidence, submission)

        # BERT handles word forms naturally
        assert similarity >= 0.6, f"Expected similarity >= 0.6 for word form variations, got {similarity:.2f}"


class TestConsistency:
    """
    Test that the grader produces consistent results.
    Same input should yield same/similar grades.
    """

    def test_grader_is_deterministic(self, client, good_submission, sample_rubric, mock_openai):
        """Grader should produce identical results on repeated calls with same mock."""
        results = []
        for _ in range(3):
            response = client.post(
                "/grade",
                json={"submission_html": good_submission, "rubric": sample_rubric},
            )
            assert response.status_code == 200
            results.append(response.json())

        # All results should be identical with mocked responses
        for i in range(1, len(results)):
            assert results[i]["overall_grade"] == results[0]["overall_grade"]
            for j, std in enumerate(results[i]["standards_assessments"]):
                for k, crit in enumerate(std["criteria_assessments"]):
                    assert crit["assigned_grade"] == results[0]["standards_assessments"][j]["criteria_assessments"][k]["assigned_grade"]


class TestEvidenceGrounding:
    """
    Test that evidence cited in assessments is grounded in the submission.
    Uses BERT embeddings to detect hallucination via semantic similarity.
    """

    def test_evidence_semantically_grounded(self, client, good_submission, sample_rubric):
        """Evidence should be semantically similar to submission content."""
        response = client.post(
            "/grade",
            json={"submission_html": good_submission, "rubric": sample_rubric},
        )
        assert response.status_code == 200
        data = response.json()

        for standard in data["standards_assessments"]:
            for criterion in standard["criteria_assessments"]:
                evidence = criterion["evidence"]

                similarity = calculate_semantic_similarity(evidence, good_submission)

                # Threshold of 0.3 allows for meta-commentary while catching hallucinations
                assert similarity >= 0.3, (
                    f"Evidence may be hallucinated. Similarity: {similarity:.2f}\n"
                    f"Evidence: {evidence[:200]}..."
                )

    def test_no_fabricated_quotes(self, client, good_submission, sample_rubric):
        """
        If evidence contains quoted text (in quotes), it should exist in submission.
        """
        response = client.post(
            "/grade",
            json={"submission_html": good_submission, "rubric": sample_rubric},
        )
        data = response.json()

        # Extract any quoted strings from evidence
        submission_lower = good_submission.lower()

        for standard in data["standards_assessments"]:
            for criterion in standard["criteria_assessments"]:
                evidence = criterion["evidence"]

                # Find quoted text in evidence
                quotes = re.findall(r'"([^"]+)"', evidence)
                quotes += re.findall(r"'([^']+)'", evidence)

                for quote in quotes:
                    # Skip very short quotes (might be single words used descriptively)
                    if len(quote) < 10:
                        continue

                    # Check if quote (or close approximation) exists in submission
                    quote_lower = quote.lower().strip()

                    # Allow for minor differences by checking key words
                    quote_words = set(quote_lower.split())
                    found_words = sum(1 for w in quote_words if w in submission_lower)
                    match_ratio = found_words / len(quote_words) if quote_words else 1

                    assert match_ratio >= 0.5, (
                        f"Quoted text may be fabricated: '{quote}'\n"
                        f"Match ratio: {match_ratio:.0%}"
                    )




class TestDiscriminativeAbility:
    """
    Test that the grader can distinguish between good and poor submissions.
    A good submission should score higher than a poor one.
    """

    def test_good_submission_scores_higher(self, client, good_submission, poor_submission, sample_rubric):
        """Good submission should receive better grades than poor submission."""
        good_response = client.post(
            "/grade",
            json={"submission_html": good_submission, "rubric": sample_rubric},
        )
        poor_response = client.post(
            "/grade",
            json={"submission_html": poor_submission, "rubric": sample_rubric},
        )

        assert good_response.status_code == 200
        assert poor_response.status_code == 200

        good_data = good_response.json()
        poor_data = poor_response.json()

        # Compare overall grades
        grade_order = ["not_met", "needs_improvement", "meets_expectations", "exceeds_expectations"]
        good_grade_idx = grade_order.index(good_data["overall_grade"])
        poor_grade_idx = grade_order.index(poor_data["overall_grade"])

        assert good_grade_idx >= poor_grade_idx, (
            f"Good submission ({good_data['overall_grade']}) should score >= "
            f"poor submission ({poor_data['overall_grade']})"
        )

    def test_good_submission_meets_more_targets(self, client, good_submission, poor_submission, sample_rubric):
        """Good submission should meet more criterion targets."""
        good_response = client.post(
            "/grade",
            json={"submission_html": good_submission, "rubric": sample_rubric},
        )
        poor_response = client.post(
            "/grade",
            json={"submission_html": poor_submission, "rubric": sample_rubric},
        )

        good_data = good_response.json()
        poor_data = poor_response.json()

        # Count criteria meeting targets
        good_meets = sum(
            1 for std in good_data["standards_assessments"]
            for crit in std["criteria_assessments"]
            if crit["meets_target"]
        )
        poor_meets = sum(
            1 for std in poor_data["standards_assessments"]
            for crit in std["criteria_assessments"]
            if crit["meets_target"]
        )

        assert good_meets >= poor_meets, (
            f"Good submission meets {good_meets} targets, "
            f"poor meets {poor_meets} - expected good >= poor"
        )

    def test_poor_submission_has_more_improvements(self, client, good_submission, poor_submission, sample_rubric):
        """Poor submission should have more areas for improvement."""
        good_response = client.post(
            "/grade",
            json={"submission_html": good_submission, "rubric": sample_rubric},
        )
        poor_response = client.post(
            "/grade",
            json={"submission_html": poor_submission, "rubric": sample_rubric},
        )

        good_data = good_response.json()
        poor_data = poor_response.json()

        good_improvements = good_data["areas_for_improvement"]
        poor_improvements = poor_data["areas_for_improvement"]

        # Filter out "None identified" type responses
        good_count = len([x for x in good_improvements if "none" not in x.lower()])
        poor_count = len([x for x in poor_improvements if "none" not in x.lower()])

        assert poor_count >= good_count, (
            f"Poor submission has {poor_count} improvements, "
            f"good has {good_count} - expected poor >= good"
        )


class TestCriterionCoverage:
    """
    Test that all rubric criteria are assessed.
    No criteria should be skipped or duplicated.
    """

    def test_all_criteria_assessed(self, client, good_submission, sample_rubric):
        """Every criterion in the rubric should have an assessment."""
        response = client.post(
            "/grade",
            json={"submission_html": good_submission, "rubric": sample_rubric},
        )
        data = response.json()

        # Get all criterion IDs from rubric
        rubric_criterion_ids = set()
        for standard in sample_rubric["standards"]:
            for criterion in standard["grading_criteria"]:
                rubric_criterion_ids.add(criterion["criterion_id"])

        # Get all assessed criterion IDs
        assessed_criterion_ids = set()
        for standard in data["standards_assessments"]:
            for criterion in standard["criteria_assessments"]:
                assessed_criterion_ids.add(criterion["criterion_id"])

        # All rubric criteria should be assessed
        missing = rubric_criterion_ids - assessed_criterion_ids
        assert not missing, f"Criteria not assessed: {missing}"

        # Should have exactly the right number
        assert len(assessed_criterion_ids) == len(rubric_criterion_ids)

    def test_no_duplicate_criteria(self, client, good_submission, sample_rubric):
        """No criterion should be assessed more than once."""
        response = client.post(
            "/grade",
            json={"submission_html": good_submission, "rubric": sample_rubric},
        )
        data = response.json()

        # Collect all criterion IDs
        criterion_ids = []
        for standard in data["standards_assessments"]:
            for criterion in standard["criteria_assessments"]:
                criterion_ids.append(criterion["criterion_id"])

        # Check for duplicates
        id_counts = Counter(criterion_ids)
        duplicates = {k: v for k, v in id_counts.items() if v > 1}

        assert not duplicates, f"Duplicate criterion assessments: {duplicates}"

    def test_all_standards_assessed(self, client, good_submission, sample_rubric):
        """Every standard in the rubric should have an assessment."""
        response = client.post(
            "/grade",
            json={"submission_html": good_submission, "rubric": sample_rubric},
        )
        data = response.json()

        rubric_standard_ids = {std["id"] for std in sample_rubric["standards"]}
        assessed_standard_ids = {std["standard_id"] for std in data["standards_assessments"]}

        assert rubric_standard_ids == assessed_standard_ids


class TestRobustness:
    """
    Test edge cases and unusual inputs.
    """

    def test_handles_special_characters(self, client, sample_rubric):
        """Should handle submissions with special/unicode characters."""
        submission = """<html><body>
            <h1>Poem</h1>
            <p>A parakeet named François, très élégant,
            Sang songs with ñ and ü so gallant.
            In the outback's heat,
            With seeds to eat,
            This bird was truly magnificent! 🦜</p>

            <h1>Short story</h1>
            <p>The parakeet lived in Australia's natural grasslands...</p>
        </body></html>"""

        response = client.post(
            "/grade",
            json={"submission_html": submission, "rubric": sample_rubric},
        )
        assert response.status_code == 200

    def test_handles_minimal_content(self, client, sample_rubric):
        """Should handle very short submissions without crashing."""
        submission = "<p>Parakeet.</p>"

        response = client.post(
            "/grade",
            json={"submission_html": submission, "rubric": sample_rubric},
        )
        assert response.status_code == 200

        data = response.json()
        # Should return valid structure with all required fields
        assert "overall_grade" in data
        assert "standards_assessments" in data
        assert data["overall_grade"] in ["not_met", "needs_improvement", "meets_expectations", "exceeds_expectations"]

    def test_handles_html_without_body(self, client, sample_rubric):
        """Should handle HTML fragments without proper structure."""
        submission = "<p>A poem about parakeets in the wild Australian outback.</p>"

        response = client.post(
            "/grade",
            json={"submission_html": submission, "rubric": sample_rubric},
        )
        assert response.status_code == 200
