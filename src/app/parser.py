"""HTML parsing and text extraction utilities."""

from bs4 import BeautifulSoup
from dataclasses import dataclass


@dataclass
class ParsedSubmission:
    """Result of parsing a submission."""
    raw_text: str
    sections: dict[str, str]
    headings: list[str]
    word_count: int
    has_structure: bool


def parse_submission(html_content: str) -> ParsedSubmission:
    """
    Parse HTML submission and extract structured content.

    Args:
        html_content: Raw HTML or plain text submission

    Returns:
        ParsedSubmission with extracted text and structure analysis
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract all text
    raw_text = soup.get_text(separator="\n", strip=True)

    # Find all headings
    headings = []
    heading_tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    for tag in heading_tags:
        text = tag.get_text(strip=True)
        if text:
            headings.append(text)

    # Extract sections based on headings
    sections = extract_sections(soup)

    # Calculate word count
    words = raw_text.split()
    word_count = len(words)

    # Determine if submission has good structure
    has_structure = len(headings) > 0 or len(sections) > 1

    return ParsedSubmission(
        raw_text=raw_text,
        sections=sections,
        headings=headings,
        word_count=word_count,
        has_structure=has_structure,
    )


def extract_sections(soup: BeautifulSoup) -> dict[str, str]:
    """
    Extract content sections based on heading tags.

    Returns a dictionary mapping heading text to section content.
    Content appearing before the first heading is labeled as 'Preamble'.
    """
    sections = {}
    current_heading = "Preamble"  # Content before first heading
    current_content = []

    # Get the body or the whole soup if no body
    body = soup.find("body") or soup

    for element in body.children:
        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # Save previous section
            if current_content:
                sections[current_heading] = "\n".join(current_content).strip()
            # Start new section
            current_heading = element.get_text(strip=True)
            current_content = []
        elif hasattr(element, "get_text"):
            text = element.get_text(strip=True)
            if text:
                current_content.append(text)
        elif isinstance(element, str) and element.strip():
            current_content.append(element.strip())

    # Save final section
    if current_content:
        sections[current_heading] = "\n".join(current_content).strip()

    return sections


def extract_text_for_grading(parsed: ParsedSubmission) -> str:
    """
    Format the parsed submission for LLM grading.

    Returns a clean text representation suitable for analysis.
    """
    if parsed.has_structure and parsed.sections:
        parts = []
        for heading, content in parsed.sections.items():
            parts.append(f"## {heading}\n{content}")
        return "\n\n".join(parts)
    return parsed.raw_text
