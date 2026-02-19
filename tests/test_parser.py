"""Tests for the HTML parser module."""

import pytest
from src.app.parser import parse_submission, extract_text_for_grading


class TestParseSubmission:
    """Tests for the parse_submission function."""

    def test_parse_html_with_headings(self):
        """Parser should extract headings from HTML."""
        html = """
        <html>
            <body>
                <h1>First Section</h1>
                <p>Some content here.</p>
                <h2>Second Section</h2>
                <p>More content.</p>
            </body>
        </html>
        """
        result = parse_submission(html)

        assert "First Section" in result.headings
        assert "Second Section" in result.headings
        assert result.has_structure is True

    def test_parse_html_without_headings(self):
        """Parser should handle HTML without headings."""
        html = """
        <html>
            <body>
                <p>Just a paragraph of text without any headings.</p>
            </body>
        </html>
        """
        result = parse_submission(html)

        assert len(result.headings) == 0
        assert result.has_structure is False

    def test_parse_plain_text(self):
        """Parser should handle plain text input."""
        text = "This is just plain text without any HTML tags."
        result = parse_submission(text)

        assert result.raw_text == text
        assert result.word_count == 9

    def test_parse_extracts_text_content(self):
        """Parser should extract text content from HTML."""
        html = "<p>Hello <b>world</b>!</p>"
        result = parse_submission(html)

        assert "Hello" in result.raw_text
        assert "world" in result.raw_text

    def test_parse_calculates_word_count(self):
        """Parser should calculate word count correctly."""
        html = "<p>One two three four five</p>"
        result = parse_submission(html)

        assert result.word_count == 5

    def test_parse_extracts_sections(self):
        """Parser should extract content sections based on headings."""
        html = """
        <body>
            <h1>Poem</h1>
            <p>This is my poem.</p>
            <h1>Story</h1>
            <p>This is my story.</p>
        </body>
        """
        result = parse_submission(html)

        assert "Poem" in result.sections
        assert "Story" in result.sections
        assert "poem" in result.sections["Poem"].lower()
        assert "story" in result.sections["Story"].lower()

    def test_parse_handles_nested_tags(self):
        """Parser should handle nested HTML tags."""
        html = """
        <div>
            <p>
                <span>Nested <b>content</b> here.</span>
            </p>
        </div>
        """
        result = parse_submission(html)

        assert "Nested" in result.raw_text
        assert "content" in result.raw_text

    def test_parse_handles_empty_html(self):
        """Parser should handle empty HTML gracefully."""
        html = "<html><body></body></html>"
        result = parse_submission(html)

        assert result.raw_text == ""
        assert result.word_count == 0


class TestExtractTextForGrading:
    """Tests for the extract_text_for_grading function."""

    def test_extract_structured_content(self):
        """Should format structured content with section headers."""
        html = """
        <body>
            <h1>Title</h1>
            <p>Content under title.</p>
        </body>
        """
        parsed = parse_submission(html)
        text = extract_text_for_grading(parsed)

        assert "## Title" in text
        assert "Content under title" in text

    def test_extract_unstructured_content(self):
        """Should return raw text for unstructured content."""
        html = "<p>Just plain content without sections.</p>"
        parsed = parse_submission(html)
        text = extract_text_for_grading(parsed)

        assert "Just plain content" in text

    def test_extract_multiple_sections(self):
        """Should format multiple sections correctly."""
        html = """
        <body>
            <h1>First</h1>
            <p>First content.</p>
            <h1>Second</h1>
            <p>Second content.</p>
        </body>
        """
        parsed = parse_submission(html)
        text = extract_text_for_grading(parsed)

        assert "## First" in text
        assert "## Second" in text
        assert "First content" in text
        assert "Second content" in text
