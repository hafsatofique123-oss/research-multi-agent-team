from crewai import Agent

from tools.web_search_tool import web_research_tool


def create_researcher_agent(llm) -> Agent:
    return Agent(
        role="Web Research Specialist",
        goal=(
            "Find accurate, current and relevant evidence from "
            "the live web and provide source-backed research findings."
        ),
        backstory=(
            "You are a professional research analyst skilled in "
            "finding reliable academic, institutional, government "
            "and other authoritative information online."
        ),
        llm=llm,
        tools=[web_research_tool],
        allow_delegation=False,
        verbose=True,
        max_iter=8,
    )
