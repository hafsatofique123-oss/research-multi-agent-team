from crewai import Agent


def create_writer_agent(llm) -> Agent:
    return Agent(
        role="Research Report Writer",
        goal=(
            "Produce a clear, structured and professional research "
            "report based only on the research and fact-checking "
            "evidence provided by the previous agents."
        ),
        backstory=(
            "You are an academic research writer. "
            "You organize complex evidence into a readable report. "
            "You clearly separate established findings, uncertainty, "
            "limitations and conclusions."
        ),
        llm=llm,
        tools=[],
        allow_delegation=False,
        verbose=True,
        max_iter=5,
    )
