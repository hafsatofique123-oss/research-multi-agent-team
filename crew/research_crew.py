import os
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from groq import Groq

from agents.planner_agent import create_planner_agent
from agents.researcher_agent import create_researcher_agent
from agents.fact_checker_agent import create_fact_checker_agent
from agents.writer_agent import create_writer_agent


GROQ_MODEL = "openai/gpt-oss-120b"


class GroqCrewLLM(BaseLLM):
    """
    Custom CrewAI LLM adapter for Groq.

    It uses the official Groq Python SDK and supports
    CrewAI native tool calling.
    """

    def __init__(
        self,
        model: str = GROQ_MODEL,
        temperature: float = 0.2,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
        )

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.client = Groq(api_key=api_key)

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        **kwargs: Any,
    ):
        if isinstance(messages, str):
            messages = [
                {
                    "role": "user",
                    "content": messages,
                }
            ]

        request = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": 8192,
            "reasoning_effort": "low",
        }

        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        response = self.client.chat.completions.create(
            **request
        )

        message = response.choices[0].message

        if message.tool_calls:
            return list(message.tool_calls)

        return message.content or ""

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 131072


def create_research_crew() -> Crew:
    llm = GroqCrewLLM()

    planner = create_planner_agent(llm)
    researcher = create_researcher_agent(llm)
    fact_checker = create_fact_checker_agent(llm)
    writer = create_writer_agent(llm)

    planning_task = Task(
        description=(
            "Create a detailed research plan for the following "
            "research question:\n\n"
            "{research_question}\n\n"
            "Use the Research Planning Tool to identify the "
            "important research areas, evidence requirements, "
            "and source priorities."
        ),
        expected_output=(
            "A structured research plan containing the main "
            "research areas, subtopics, evidence requirements, "
            "and source priorities."
        ),
        agent=planner,
    )

    research_task = Task(
        description=(
            "Conduct web research based on the research plan "
            "created by the Research Planning Specialist.\n\n"
            "Original research question:\n"
            "{research_question}\n\n"
            "Use the Web Research Tool to collect current and "
            "relevant evidence. Search for authoritative sources "
            "and provide source names and URLs where available. "
            "Do not invent citations or statistics."
        ),
        expected_output=(
            "Detailed research findings organized by topic, "
            "with evidence, important facts, dates where relevant, "
            "and source references."
        ),
        agent=researcher,
        context=[planning_task],
    )

    fact_check_task = Task(
        description=(
            "Fact-check the research findings produced by the "
            "Web Research Specialist.\n\n"
            "Original research question:\n"
            "{research_question}\n\n"
            "Use the Source Analyzer Tool and Web Research Tool. "
            "Identify unsupported claims, weak evidence, outdated "
            "information, conflicting findings, and claims that "
            "need additional verification."
        ),
        expected_output=(
            "A fact-checking report containing verified findings, "
            "claims requiring caution, source-quality observations, "
            "contradictions or uncertainties, and recommended "
            "evidence for the final report."
        ),
        agent=fact_checker,
        context=[research_task],
    )

    writing_task = Task(
        description=(
            "Write the final research report using ONLY the "
            "research findings and fact-checking results provided "
            "by the previous agents.\n\n"
            "Original research question:\n"
            "{research_question}\n\n"
            "The report must be balanced, evidence-based, clear, "
            "professional and easy to read.\n\n"
            "Use this structure:\n"
            "1. Title\n"
            "2. Executive Summary\n"
            "3. Introduction\n"
            "4. Key Findings\n"
            "5. Evidence and Research\n"
            "6. Different Perspectives\n"
            "7. Limitations and Uncertainty\n"
            "8. Conclusion\n"
            "9. Sources\n\n"
            "Do not invent information or citations."
        ),
        expected_output=(
            "A complete professional research report with clear "
            "sections, evidence-based findings, limitations, "
            "conclusion and source references."
        ),
        agent=writer,
        context=[research_task, fact_check_task],
    )

    crew = Crew(
        agents=[
            planner,
            researcher,
            fact_checker,
            writer,
        ],
        tasks=[
            planning_task,
            research_task,
            fact_check_task,
            writing_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def run_research(research_question: str):
    if not research_question or not research_question.strip():
        raise ValueError(
            "Please enter a research question."
        )

    crew = create_research_crew()

    result = crew.kickoff(
        inputs={
            "research_question": research_question.strip()
        }
    )

    return result
