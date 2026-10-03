import re

from crewai.tools import tool


@tool("Source Analyzer Tool")
def source_analyzer_tool(research_text: str) -> str:
    """
    Analyze research findings and identify source references,
    evidence statements, claims, and potential verification needs.
    """

    if not research_text or not research_text.strip():
        return "No research text was provided."

    urls = re.findall(
        r"https?://[^\s)\]>]+",
        research_text
    )

    lines = [
        line.strip()
        for line in research_text.splitlines()
        if line.strip()
    ]

    evidence_lines = []

    evidence_keywords = [
        "study",
        "research",
        "report",
        "according",
        "data",
        "survey",
        "statistics",
        "evidence",
        "published",
        "university",
        "government",
        "journal",
    ]

    for line in lines:
        lower_line = line.lower()

        if any(keyword in lower_line for keyword in evidence_keywords):
            evidence_lines.append(line)

    result = []

    result.append("SOURCE ANALYSIS")
    result.append("=" * 50)

    result.append(f"URLs detected: {len(urls)}")
    result.append(f"Evidence-related statements detected: {len(evidence_lines)}")

    if urls:
        result.append("\nDetected sources:")
        for index, url in enumerate(urls, start=1):
            result.append(f"{index}. {url}")

    if evidence_lines:
        result.append("\nEvidence-related statements:")
        for index, line in enumerate(evidence_lines[:20], start=1):
            result.append(f"{index}. {line}")

    result.append(
        "\nVerification note: "
        "Claims should be checked against their original sources "
        "before being treated as established facts."
    )

    return "\n".join(result)
