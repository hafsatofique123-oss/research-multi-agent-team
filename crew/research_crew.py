import os
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from groq import Groq

from agents.planner_agent import create_planner_agent
from agents.researcher_agent import create_researcher_agent
from agents.fact_checker_agent import create_fact_checker_agent
from agents.writer_agent import create_writer_agent


# =========================================================
# GROQ MODEL
# =========================================================

GROQ_MODEL = "openai/gpt-oss-20b"


# =========================================================
# CUSTOM GROQ LLM FOR CREWAI
# =========================================================

class GroqCrewLLM(BaseLLM):
    """
    Custom CrewAI LLM adapter for Groq.

    This adapter removes CrewAI's internal
    cache_breakpoint field before messages are
    sent to Groq.
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
                "GROQ_API_KEY is not configured. "
                "Please add it to Streamlit Secrets."
            )

        self.client = Groq(
            api_key=api_key
        )

    # =====================================================
    # CLEAN CREWAI MESSAGES
    # =====================================================

    @staticmethod
    def _clean_messages(messages):

        cleaned_messages = []

        for message in messages:

            if isinstance(message, dict):

                cleaned_message = {
                    key: value
                    for key, value in message.items()
                    if key != "cache_breakpoint"
                }

                cleaned_messages.append(
                    cleaned_message
                )

            else:

                cleaned_messages.append(
                    message
                )

        return cleaned_messages

    # =====================================================
    # CALL GROQ
    # =====================================================

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        **kwargs: Any,
    ):

        # Convert string input into a message
        if isinstance(messages, str):

            messages = [
                {
                    "role": "user",
                    "content": messages,
                }
            ]

        # Remove unsupported CrewAI field
        messages = self._clean_messages(
            messages
        )

        # Groq request
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": 4096,
            "reasoning_effort": "low",
        }

        # Add tools when CrewAI provides them
        if tools:

            request["tools"] = tools
            request["tool_choice"] = "auto"

        # Send request to Groq
        response = self.client.chat.completions.create(
            **request
        )

        message = response.choices[0].message

        # Return tool calls when available
        if message.tool_calls:

            return list(
                message.tool_calls
            )

        # Otherwise return normal text
        return message.content or ""

    # =====================================================
    # CREWAI CAPABILITIES
    # =====================================================

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 131072


# =========================================================
# CREATE RESEARCH CREW
# =========================================================

def create_research_crew() -> Crew:

    # Create Groq LLM
    llm = GroqCrewLLM()

    # =====================================================
    # CREATE AGENTS
    # =====================================================

    planner = create_planner_agent(
        llm
    )

    researcher = create_researcher_agent(
        llm
    )

    fact_checker = create_fact_checker_agent(
        llm
    )

    writer = create_writer_agent(
        llm
    )

    # =====================================================
    # TASK 1 — RESEARCH PLANNING
    # =====================================================

    planning_task = Task(

        description=(
            "Create a clear and practical research plan "
            "for the following research question:\n\n"

            "{research_question}\n\n"

            "Use the Research Planning Tool to identify:\n"
            "- main research areas\n"
            "- important subtopics\n"
            "- evidence requirements\n"
            "- useful source types\n\n"

            "Keep the plan focused and concise."
        ),

        expected_output=(
            "A structured research plan containing the "
            "main research areas, important subtopics, "
            "evidence requirements, and source priorities."
        ),

        agent=planner,
    )

    # =====================================================
    # TASK 2 — WEB RESEARCH
    # =====================================================

    research_task = Task(

        description=(
            "Conduct research based on the research plan "
            "created by the Research Planning Specialist.\n\n"

            "Original research question:\n"
            "{research_question}\n\n"

            "Use the Web Research Tool to collect relevant "
            "and current information.\n\n"

            "Prefer:\n"
            "- academic sources\n"
            "- government sources\n"
            "- universities\n"
            "- official organizations\n"
            "- reputable research sources\n\n"

            "Record useful source names and URLs when "
            "available.\n\n"

            "Do not invent citations, statistics, facts, "
            "or sources.\n\n"

            "Keep the research focused on the question."
        ),

        expected_output=(
            "Research findings organized by topic, "
            "including important facts, evidence, "
            "dates where relevant, and source references."
        ),

        agent=researcher,

        context=[
            planning_task
        ],
    )

    # =====================================================
    # TASK 3 — FACT CHECKING
    # =====================================================

    fact_check_task = Task(

        description=(
            "Fact-check the research findings produced "
            "by the Web Research Specialist.\n\n"

            "Original research question:\n"
            "{research_question}\n\n"

            "Use the Source Analyzer Tool and Web Research "
            "Tool when appropriate.\n\n"

            "Check for:\n"
            "- unsupported claims\n"
            "- weak evidence\n"
            "- outdated information\n"
            "- conflicting findings\n"
            "- questionable statistics\n"
            "- claims requiring verification\n\n"

            "Clearly identify information that is uncertain.\n\n"

            "Do not invent evidence."
        ),

        expected_output=(
            "A concise fact-checking report containing "
            "verified findings, questionable claims, "
            "source-quality observations, contradictions, "
            "uncertainties, and evidence that requires caution."
        ),

        agent=fact_checker,

        context=[
            research_task
        ],
    )

    # =====================================================
    # TASK 4 — FINAL REPORT
    # =====================================================

    writing_task = Task(

        description=(
            "Write the final research report using the "
            "research findings and fact-checking results "
            "from the previous agents.\n\n"

            "Original research question:\n"
            "{research_question}\n\n"

            "The report must be:\n"
            "- clear\n"
            "- professional\n"
            "- evidence-based\n"
            "- balanced\n"
            "- easy to read\n\n"

            "Use this structure:\n\n"

            "1. Title\n"
            "2. Executive Summary\n"
            "3. Introduction\n"
            "4. Key Findings\n"
            "5. Evidence and Research\n"
            "6. Different Perspectives\n"
            "7. Limitations and Uncertainty\n"
            "8. Conclusion\n"
            "9. Sources\n\n"

            "Use only information supported by the previous "
            "agent outputs.\n\n"

            "Do not invent facts, statistics, or citations."
        ),

        expected_output=(
            "A complete professional research report with "
            "a title, executive summary, introduction, "
            "key findings, evidence, perspectives, "
            "limitations, conclusion, and sources."
        ),

        agent=writer,

        context=[
            research_task,
            fact_check_task
        ],
    )

    # =====================================================
    # CREATE CREW
    # =====================================================

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


# =========================================================
# RUN RESEARCH
# =========================================================

def run_research(
    research_question: str
):

    if not research_question:
        raise ValueError(
            "Please enter a research question."
        )

    research_question = research_question.strip()

    if not research_question:
        raise ValueError(
            "Please enter a research question."
        )

    crew = create_research_crew()

    result = crew.kickoff(
        inputs={
            "research_question": research_question
        }
    )

    return result
