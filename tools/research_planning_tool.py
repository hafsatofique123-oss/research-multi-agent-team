from crewai.tools import tool


@tool("Research Planning Tool")
def research_planning_tool(research_question: str) -> str:
    """
    Create a basic research planning checklist from a research question.
    """

    if not research_question or not research_question.strip():
        return "ERROR: Research question is empty."

    question = research_question.strip()

    return f"""
RESEARCH PLANNING CHECKLIST

Research Question:
{question}

Required research areas:

1. Definition and background
2. Key concepts and terminology
3. Current evidence
4. Important studies or reports
5. Quantitative data and statistics
6. Different perspectives
7. Benefits or positive findings
8. Risks, limitations, or challenges
9. Recent developments
10. Primary and authoritative sources
11. Areas where evidence is uncertain or conflicting
12. Final conclusions supported by evidence

Source priority:
- Primary research
- Academic journals
- Universities
- Government organizations
- International organizations
- Official institutional reports
- Reputable professional organizations

Avoid:
- Unsupported claims
- Invented statistics
- Unverified sources
- Treating opinions as facts
"""
