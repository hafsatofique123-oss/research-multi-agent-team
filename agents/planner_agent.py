from crewai import Agent

from tools.research_planning_tool import research_planning_tool


def create_planner_agent(llm) -> Agent:
    return Agent(
        role="Research Planning Specialist",
        goal=(
            "Turn a research question into a clear, logical and "
            "evidence-focused research plan."
        ),
        backstory=(
            "You are an experienced academic research planner. "
            "You break complex research questions into manageable "
            "research areas and identify what evidence is needed."
        ),
        llm=llm,
        tools=[research_planning_tool],
        allow_delegation=False,
        verbose=True,
        max_iter=4,
    )
