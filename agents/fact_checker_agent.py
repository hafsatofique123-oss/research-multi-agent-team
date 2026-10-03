from crewai import Agent

from tools.source_analyzer_tool import source_analyzer_tool
from tools.web_search_tool import web_research_tool


def create_fact_checker_agent(llm) -> Agent:
    return Agent(
        role="Research Fact Checker",
        goal=(
            "Evaluate research findings, identify unsupported claims, "
            "check source quality, and distinguish strong evidence "
            "from uncertain or conflicting information."
        ),
        backstory=(
            "You are a rigorous fact-checking specialist. "
            "You do not automatically accept claims. "
            "You examine evidence, source quality, dates, and "
            "potential contradictions before information is used "
            "in the final report."
        ),
        llm=llm,
        tools=[
            source_analyzer_tool,
            web_research_tool,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=8,
    )
